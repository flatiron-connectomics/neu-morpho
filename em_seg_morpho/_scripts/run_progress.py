"""Point-in-time progress for a detached em-morpho run.

Reads the work dir only — the JSONL manifests, ``run_plan.json`` and the fragment
stores. **It never contacts the dask cluster, S3, or SLURM**, so it is cheap and
safe to run as often as you like from any terminal:

    em-morpho progress /mnt/ceph/users/<you>/morpho-run

    # measure a rate and extrapolate, by sampling the manifests twice
    em-morpho progress <work-dir> --sample 60

Denominators come from ``run_plan.json``, which the CLI writes at startup
(ROI ∩ occupancy block counts, which are expensive to re-derive because they need
an occupancy-array read). Without that file, absolute counts are still reported.

For cluster-side questions this cannot answer — are the workers alive, is anything
queued — use the dashboard URL the driver logged, or one shot of
``squeue -u "$USER"``. Do not poll SLURM in a loop; it is a shared service.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

# The two-stage ops key their manifest groups by pipeline phase. Stage 1 is per
# BLOCK, stage 2 per BODY, so their totals are not comparable and are reported
# separately — a combined percentage would be meaningless.
#
# The per-BODY groups have no count in run_plan.json — the body set is not known
# until stage 1 has run. It is knowable afterwards, though: stage 1 writes one
# directory per body under the fragment store, so that count is exactly how many
# bodies stage 2 has to fuse (verified: 19,989 dirs == 19,989 num_bodies_fused).
# The 4th field names the store to count, and is only trusted once stage 1 is
# COMPLETE — while stage 1 is still running the store keeps growing, so using it
# early would divide by a moving target and flatter the progress bar.
STAGE_GROUPS = {
    "seg":  [("seg-*", "block", "seg", None)],
    "mesh": [("chunk", "block", "mesh", None), ("assemble", "body", None, "chunked")],
    "skel": [("skel-chunk", "block", "skel", None),
             ("skel-fuse", "body", None, "skel_chunked")],
}
# Statuses that mean the task is finished and will not be retried. "failed" is
# deliberately excluded: em-blockrun's is_done tests key presence, so a resumed run
# retries failures (see ops/_progress.is_complete).
DONE = ("written", "empty", "dust", "skipped")


def _read_manifest(path: str) -> Counter:
    """(group, status) -> count. Tolerates a torn final line from a hard crash."""
    counts: Counter = Counter()
    if not os.path.exists(path):
        return counts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts[(rec.get("group", rec.get("level")), rec.get("status"))] += 1
    return counts


def _match(counts: Counter, group_pattern: str) -> Counter:
    """Status counts for one group, or for every group sharing a ``*`` prefix."""
    out: Counter = Counter()
    prefix = group_pattern[:-1] if group_pattern.endswith("*") else None
    for (group, status), n in counts.items():
        g = str(group)
        if (prefix is not None and g.startswith(prefix)) or g == group_pattern:
            out[status] += n
    return out


def _bar(done: int, total: int | None, width: int = 24) -> str:
    if not total:
        return ""
    frac = min(1.0, done / total)
    filled = int(frac * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {100 * frac:5.1f}%"


def _store_bodies(path: str) -> int | None:
    """How many bodies stage 1 produced fragments for — stage 2's denominator.

    One directory per body (``fragments.body_dir``), so a single top-level listing
    answers it. Do NOT walk into them: that is a millions-of-inodes traversal.
    """
    try:
        return len(os.listdir(path))
    except OSError:
        return None


def _fragment_stores(work: str) -> list[tuple[str, int, int]]:
    """(name, body-dir count, recursive bytes) for the stage-1 fragment stores.

    Only the top level is listed — one entry per body, tens of thousands at most.
    Walking every body's fragments would be a millions-of-inodes traversal, which
    is exactly the thing you are checking the size of. On ceph a directory's
    ``st_size`` is already its recursive size, so that comes for free.
    """
    rows = []
    for name in ("chunked", "skel_chunked"):
        path = os.path.join(work, name)
        if not os.path.isdir(path):
            continue
        try:
            bodies = len(os.listdir(path))
        except OSError:
            bodies = -1
        rows.append((name, bodies, os.stat(path).st_size))
    return rows


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}EB"


def _snapshot(work: str) -> dict[str, Counter]:
    return {stage: _read_manifest(os.path.join(work, f"progress.{stage}.jsonl"))
            for stage in STAGE_GROUPS}


def _report(work: str, plan: dict, snap: dict[str, Counter]) -> dict[str, int]:
    planned = plan.get("planned_blocks") or {}
    seg_total = sum(s["blocks"] for s in plan.get("seg_scales") or [])
    done_now: dict[str, int] = {}

    for stage, groups in STAGE_GROUPS.items():
        counts = snap[stage]
        if not counts:
            continue
        print(f"\n{stage}")
        # Stage 1 of this stage, for gating the per-body denominator below.
        stage1 = groups[0]
        s1 = _match(counts, stage1[0])
        s1_done = sum(n for s, n in s1.items() if s in DONE)
        s1_total = seg_total if stage == "seg" else planned.get(stage1[2])
        s1_complete = bool(s1_total) and s1_done >= s1_total

        for group_pattern, unit, plan_key, store in groups:
            g = _match(counts, group_pattern)
            if not g:
                continue
            done = sum(n for s, n in g.items() if s in DONE)
            failed = g.get("failed", 0)
            total = (seg_total if stage == "seg" else planned.get(plan_key)) if plan_key else None
            if total is None and store and s1_complete:
                total = _store_bodies(os.path.join(work, store))
            label = f"{group_pattern:11s} {unit:5s}"
            detail = "  ".join(f"{s}={n}" for s, n in sorted(g.items()))
            print(f"  {label} {done:>7}/{total if total else '?':>7}  "
                  f"{_bar(done, total)}")
            print(f"  {'':17} {detail}" + (f"   <-- {failed} FAILED" if failed else ""))
            done_now[f"{stage}:{group_pattern}"] = done

    # Per-scale detail for the seg copy: it is the only stage with many groups, and
    # knowing which scale is in flight tells you how much cost is left (scale 0
    # dominates by ~8x per level).
    if plan.get("seg_scales") and snap["seg"]:
        print("\n  seg by scale")
        for s in plan["seg_scales"]:
            g = _match(snap["seg"], f"seg-{s['scale']}")
            done = sum(n for st, n in g.items() if st in DONE)
            state = "done" if done >= s["blocks"] else ("active" if done else "pending")
            print(f"    scale {s['scale']}  {done:>7}/{s['blocks']:>7}  "
                  f"{_bar(done, s['blocks'])}  {state}")
    return done_now


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dir", help="the run's --work-dir")
    ap.add_argument("--sample", type=int, default=0, metavar="SECONDS",
                    help="re-read after N seconds to report a rate and ETA")
    ap.add_argument("--failures", type=int, default=3, metavar="N",
                    help="show N recent failure reasons per stage (0 to skip)")
    args = ap.parse_args(argv)

    work = args.work_dir.rstrip("/")
    if not os.path.isdir(work):
        raise SystemExit(f"no such work dir: {work}")

    plan = {}
    plan_path = os.path.join(work, "run_plan.json")
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            plan = json.load(f)
        scales = plan.get("scales", {})
        print(f"{work}\n  started {plan.get('started', '?')}  "
              f"stages={','.join(plan.get('stages', []))}  "
              f"scales index/mesh/skel = "
              f"{scales.get('index')}/{scales.get('mesh')}/{scales.get('skel')}")
        print(f"  dst {plan.get('dst')}")
    else:
        print(f"{work}\n  (no run_plan.json — absolute counts only; it is written "
              f"by the driver at startup, so a run from before this existed has none)")

    snap = _snapshot(work)
    if not any(snap.values()):
        print("\nno manifests yet — the first stage has not recorded anything")
    done_before = _report(work, plan, snap)

    frags = _fragment_stores(work)
    if frags:
        print("\nfragment stores (stage-1 output, on the work filesystem)")
        for name, bodies, size in frags:
            print(f"  {name:13s} {bodies:>8} bodies   {_human(size):>9}")
        print("  NOTE one file per (body, block): watch inode quota, not just bytes")

    if args.failures:
        for stage in STAGE_GROUPS:
            path = os.path.join(work, f"failures.{stage}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                recs = [json.loads(l) for l in f if l.strip()]
            print(f"\nfailures.{stage}.jsonl — {len(recs)} recorded (retried on re-run)")
            for r in recs[-args.failures:]:
                print(f"  body {r.get('body_id')}: {str(r.get('error'))[:150]}")

    run_summary = os.path.join(work, "run_summary.json")
    if os.path.exists(run_summary):
        with open(run_summary) as f:
            s = json.load(f)
        print(f"\nrun_summary.json present — the run finished. "
              f"timing(min)={s.get('timing_min')}  failed_bodies={s.get('n_failed_bodies')}")

    if args.sample:
        print(f"\nsampling for {args.sample}s to estimate a rate ...")
        time.sleep(args.sample)
        after = _snapshot(work)
        done_after = _report(work, plan, after)
        print("\nrate")
        for key, before in done_before.items():
            delta = done_after.get(key, before) - before
            per_min = 60.0 * delta / args.sample
            line = f"  {key:22s} {delta:>6} in {args.sample}s = {per_min:8.1f}/min"
            print(line)
        print("  (a stage that is not currently running shows 0; ETA needs a "
              "denominator from run_plan.json and a nonzero rate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
