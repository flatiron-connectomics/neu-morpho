#!/usr/bin/env python
"""Human-readable HTML summary of a run, from its work directory.

    em-morpho run-report <work-dir> [-o report.html]

Reads whatever is present — ``run_plan.json``, ``run_summary.json``, the progress
manifests, ``fusion_stats.jsonl`` and ``metrics.db`` — so it also works on a run
still in flight, where it reports progress against the plan instead of a result.

Self-contained: no JS, no external assets, charts are inline SVG. It therefore needs
no optional dependency and survives being copied off the cluster.

**It reports what it finds and says what is missing.** A stage that did not run leaves
empty columns in the metrics DB (``--stages skel`` never populates ``voxel_count``),
and printing those as zeros would read as "these bodies have no voxels".
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime

# Validated categorical slots 1-3, plus status red. Colourblind-safe pairs.
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def read_run(work: str) -> dict:
    plan = _json(f"{work}/run_plan.json") or {}
    summary = _json(f"{work}/run_summary.json")
    progress = {}
    for stage in ("seg", "index", "mesh", "skel"):
        recs = _jsonl(f"{work}/progress.{stage}.jsonl")
        if recs:
            progress[stage] = Counter((r.get("group"), r.get("status")) for r in recs)
    failures = {}
    for stage in ("mesh", "skel"):
        recs = _jsonl(f"{work}/failures.{stage}.jsonl")
        if recs:
            failures[stage] = recs
    return {"work": work, "plan": plan, "summary": summary,
            "progress": progress, "failures": failures,
            "metrics": read_metrics(f"{work}/metrics.db")}


def store_bodies(path: str) -> int | None:
    """Bodies stage 1 produced fragments for — stage 2's denominator.

    One directory per body (``fragments.body_dir``), so one top-level listing
    answers it. Do NOT walk into them: that is a millions-of-inodes traversal.
    """
    try:
        return len(os.listdir(path))
    except OSError:
        return None


def read_metrics(db_path: str) -> dict:
    """Per-body columns that are actually populated. Empty ones are reported as such."""
    if not os.path.exists(db_path):
        return {}
    cols = ["cable_length_nm", "n_branches", "n_tips", "max_radius_nm",
            "voxel_count", "n_mesh_components", "mesh_verts"]
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        have = {r[1] for r in con.execute("PRAGMA table_info(bodies)")}
        cols = [c for c in cols if c in have]
        rows = con.execute(f"select {', '.join(cols)} from bodies").fetchall()
        con.close()
    except sqlite3.Error:
        return {}
    out = {}
    for i, c in enumerate(cols):
        vals = [r[i] for r in rows if r[i] is not None]
        # A column the run never wrote is all-zero, not a measurement of zero.
        out[c] = vals if any(v for v in vals) else []
    out["_n_bodies"] = len(rows)
    return out


# --------------------------------------------------------------------------- #
# small stats, without numpy
# --------------------------------------------------------------------------- #
def quantiles(vals, qs=(0.5, 0.9)):
    if not vals:
        return {}
    s = sorted(vals)
    out = {}
    for q in qs:
        i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        out[q] = s[i]
    return out


def histogram(vals, bins=32):
    if not vals:
        return [], 0, 0
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [len(vals)], lo, hi
    counts = [0] * bins
    span = hi - lo
    for v in vals:
        k = int((v - lo) / span * bins)
        counts[min(k, bins - 1)] += 1
    return counts, lo, hi


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def svg_hist(vals, colour, unit="", bins=32, w=640, h=170):
    """Inline SVG histogram. Returns '' when there is nothing to draw."""
    counts, lo, hi = histogram(vals, bins)
    if not counts:
        return ""
    top = max(counts) or 1
    pad_b, pad_l = 22, 4
    bw = (w - 2 * pad_l) / len(counts)
    bars = []
    for i, c in enumerate(counts):
        bh = (h - pad_b) * (c / top)
        x = pad_l + i * bw
        y = h - pad_b - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bw - 1.5):.1f}" '
                    f'height="{bh:.1f}" rx="1.5" fill="{colour}"><title>'
                    f'{c:,} bodies</title></rect>')
    fmt = (lambda v: f"{v:,.0f}") if hi >= 10 else (lambda v: f"{v:,.2f}")
    return (
        f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
        f'aria-label="distribution histogram">{"".join(bars)}'
        f'<line x1="{pad_l}" y1="{h-pad_b}" x2="{w-pad_l}" y2="{h-pad_b}" class="axis"/>'
        f'<text x="{pad_l}" y="{h-6}" class="tick">{fmt(lo)}{unit}</text>'
        f'<text x="{w-pad_l}" y="{h-6}" class="tick end">{fmt(hi)}{unit}</text></svg>')


def stat(label, value, note=""):
    note = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (f'<div class="stat"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{value}</div>{note}</div>')


def table(headers, rows, aligns=None):
    aligns = aligns or ["l"] * len(headers)
    th = "".join(f'<th class="{a}">{html.escape(str(x))}</th>'
                 for x, a in zip(headers, aligns))
    body = []
    for r in rows:
        tds = "".join(f'<td class="{a}">{c}</td>' for c, a in zip(r, aligns))
        body.append(f"<tr>{tds}</tr>")
    return (f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def fmt_nm(nm):
    """nm -> the unit a human wants: um above a micron, mm above a millimetre."""
    if nm is None:
        return "—"
    if nm >= 1e6:
        return f"{nm / 1e6:,.1f} mm"
    if nm >= 1e3:
        return f"{nm / 1e3:,.1f} µm"
    return f"{nm:,.0f} nm"


def render(run: dict) -> str:
    plan, summary = run["plan"], run["summary"]
    m = run["metrics"]
    skel = (summary or {}).get("stages", {}).get("skel", {})
    fs = skel.get("fusion_stats", {})
    done = summary is not None

    parts = []
    title = os.path.basename(run["work"].rstrip("/")) or run["work"]
    status = ("Completed" if done else "In progress — no run_summary.json yet")
    parts.append(f'<h1>{html.escape(title)}</h1>')
    parts.append(f'<p class="sub">{html.escape(status)}')
    if summary and summary.get("finished"):
        parts.append(f' · finished {html.escape(str(summary["finished"]))}')
    elif plan.get("started"):
        parts.append(f' · started {html.escape(str(plan["started"]))}')
    parts.append("</p>")

    # ---- headline numbers -------------------------------------------------
    cable = m.get("cable_length_nm") or []
    # Fall back to the manifests when there is no summary yet: on an in-flight run
    # those are the only live source, and reporting 0 would read as "nothing done".
    counted = Counter()
    for counter in run["progress"].values():
        for (group, _), n in counter.items():
            counted[group] += n
    n_bodies = (skel.get("num_bodies_fused") or counted.get("skel-fuse")
                or m.get("_n_bodies") or 0)
    n_blocks = skel.get("num_blocks") or counted.get("skel-chunk") or 0
    planned = (plan.get("planned_blocks") or {}).get("skel")
    blocks_txt = (f"{n_blocks:,} / {planned:,}"
                  if planned and not done else f"{n_blocks:,}" if n_blocks else "—")
    # Stage 2's denominator is not in run_plan.json — the body set is unknown until
    # stage 1 runs. Afterwards it is exact: one fragment directory per body. Only
    # trustworthy once stage 1 is complete, since the store grows while it runs.
    bodies_txt = f"{n_bodies:,}"
    if not done and planned and n_blocks >= planned:
        expected = store_bodies(os.path.join(run["work"], "skel_chunked"))
        if expected:
            bodies_txt = f"{n_bodies:,} / {expected:,}"
    n_failed = (summary or {}).get("n_failed_bodies", 0)
    timing = (summary or {}).get("timing_min", {})
    elapsed = ""
    if not done and plan.get("started"):
        try:
            began = datetime.fromisoformat(plan["started"])
            elapsed = f"{(datetime.now().astimezone() - began).total_seconds() / 60:.0f} min"
        except ValueError:
            pass
    stats = [
        stat("Bodies", bodies_txt, "" if done else "fused / to fuse"),
        stat("Total cable", fmt_nm(sum(cable)) if cable else "—",
             "" if done else "so far"),
        stat("Blocks", blocks_txt, "" if done else "done / planned"),
        stat("Wall time", f"{timing.get('total', 0):.1f} min" if timing
             else elapsed or "—", "" if done else "elapsed"),
        stat("Failed bodies", f"{n_failed:,}",
             "retried on the next run" if n_failed else ""),
    ]
    parts.append(f'<div class="stats">{"".join(stats)}</div>')

    # ---- what was run -----------------------------------------------------
    rows = []
    for k, label in (("tracer", "Tracer"), ("neutu_cost", "Cost"),
                     ("stages", "Stages"), ("roi", "ROI"),
                     ("roi_scale", "ROI scale")):
        v = plan.get(k)
        if v is not None:
            rows.append((label, f"<code>{html.escape(str(v))}</code>"))
    if plan.get("scales"):
        rows.append(("Scales", f"<code>{html.escape(str(plan['scales']))}</code>"))
    if summary and summary.get("neuroglancer_source"):
        rows.append(("Neuroglancer source",
                     f"<code>{html.escape(summary['neuroglancer_source'])}</code>"))
    if rows:
        parts.append("<h2>Configuration</h2>")
        parts.append(table(["", ""], rows, ["l", "l"]))
    if plan.get("command"):
        parts.append(f'<pre class="cmd">{html.escape(plan["command"])}</pre>')

    # ---- per-stage outcome ------------------------------------------------
    if run["progress"]:
        parts.append("<h2>Task outcomes</h2>")
        parts.append(
            '<p class="sub">Share is within the group, not progress — a stage can be '
            '100% done and still be mostly <code>empty</code>, which is the normal '
            'result for blocks that hold no allowlisted labels. Progress against the '
            'plan is the <b>done</b> column.</p>')
        rows = []
        for stage, counter in run["progress"].items():
            groups = sorted({g for g, _ in counter})
            for group in groups:
                items = [(st, n) for (g, st), n in counter.items() if g == group]
                total = sum(n for _, n in items)
                # planned_blocks is per stage and counts BLOCKS, so it is the
                # denominator for the per-block group only; a per-body group
                # (skel-fuse) has no predeclared count.
                planned = (plan.get("planned_blocks") or {}).get(stage)
                is_block_group = group and group.endswith("chunk")
                done = (f"{total:,} / {planned:,}"
                        if is_block_group and planned else f"{total:,}")
                first = True
                for st, n in sorted(items, key=lambda kv: -kv[1]):
                    cls = ' class="bad"' if st in ("failed", "error") else ""
                    rows.append((
                        html.escape(stage) if first else "",
                        html.escape(str(group)) if first else "",
                        done if first else "",
                        f"<span{cls}>{html.escape(str(st))}</span>",
                        f"{n:,}",
                        f"{100.0 * n / total:.0f}%" if total else ""))
                    first = False
        parts.append(table(["Stage", "Group", "Done", "Status", "Count", "Share"],
                           rows, ["l", "l", "r", "l", "r", "r"]))

    # ---- fusion --------------------------------------------------------------
    if fs:
        parts.append("<h2>Fusion</h2>")
        parts.append(
            '<p class="sub">Stage 2 welds each body\'s per-block fragments, then '
            'postprocesses. <b>Dropped</b> cable is arbor removed by dust/tick '
            'filtering; <b>inferred</b> cable is what the joins invented — the '
            'number to watch, because a wide join manufactures neurite that never '
            'existed.</p>')
        rows = [
            ("Cable in → out",
             f"{fmt_nm(fs.get('cable_in_nm'))} → {fmt_nm(fs.get('cable_out_nm'))}"),
            ("Dropped", f"{fmt_nm(fs.get('dropped_cable_nm'))} "
                        f"({fs.get('dropped_cable_fraction', 0) * 100:.1f}%)"),
            ("Inferred by joins", f"{fmt_nm(fs.get('inferred_cable_nm'))} "
                                  f"({fs.get('inferred_cable_fraction', 0) * 100:.2f}%)"),
            ("Seam joins", f"{fs.get('seam_join_comps_merged', 0):,} components merged, "
                           f"+{fmt_nm(fs.get('seam_join_cable_added_nm'))}"),
            ("Ticks removed", f"{fs.get('tick_branches_removed', 0):,} branches, "
                              f"−{fmt_nm(fs.get('tick_cable_removed_nm'))}"),
            ("Components in → out",
             f"{fs.get('comps_in', 0):,} → {fs.get('comps_out', 0):,}"),
            ("Multi-component bodies", f"{fs.get('bodies_multi_component', 0):,} "
                                       f"of {fs.get('n_bodies', 0):,}"),
        ]
        parts.append(table(["", ""], [(a, b) for a, b in rows], ["l", "l"]))

    # ---- per-body distributions ------------------------------------------
    charts = [("cable_length_nm", "Cable length per body", BLUE, ""),
              ("n_tips", "Tips per body", ORANGE, ""),
              ("max_radius_nm", "Max radius per body", GREEN, " nm")]
    drawn = [(k, lab, col, unit) for k, lab, col, unit in charts if m.get(k)]
    if drawn:
        parts.append("<h2>Per-body distributions</h2>")
        parts.append('<div class="grid">')
        for key, label, colour, unit in drawn:
            vals = m[key]
            q = quantiles(vals)
            scale = 1e3 if key == "cable_length_nm" else 1.0
            sub = (f"median {q[0.5] / scale:,.0f} · p90 {q[0.9] / scale:,.0f}"
                   f"{' µm' if scale != 1 else unit}")
            parts.append(
                f'<figure><figcaption>{html.escape(label)}'
                f'<span class="sub">{html.escape(sub)}</span></figcaption>'
                f'{svg_hist([v / scale for v in vals], colour, " µm" if scale != 1 else unit)}'
                f'</figure>')
        parts.append("</div>")

    # ---- what is NOT here --------------------------------------------------
    missing = [c for c in ("voxel_count", "mesh_verts", "n_mesh_components")
               if c in m and not m[c]]
    if missing:
        parts.append(
            f'<p class="sub missing"><b>Not measured in this run:</b> '
            f'{html.escape(", ".join(missing))} — the stage that populates these '
            f'(index / mesh) did not run, so the columns are empty rather than zero.'
            f'</p>')

    # ---- failures ----------------------------------------------------------
    for stage, recs in run["failures"].items():
        parts.append(f"<h2>Failures — {html.escape(stage)}</h2>")
        rows = [(html.escape(str(r.get("key", r.get("body_id", "?")))),
                 f'<code>{html.escape(str(r.get("error", ""))[:160])}</code>')
                for r in recs[:25]]
        parts.append(table(["Body / block", "Error"], rows, ["l", "l"]))
        if len(recs) > 25:
            parts.append(f'<p class="sub">…and {len(recs) - 25:,} more in '
                         f'failures.{html.escape(stage)}.jsonl</p>')

    parts.append(f'<footer>{html.escape(run["work"])} · generated '
                 f'{datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}</footer>')
    return "\n".join(parts)


CSS = """
:root { color-scheme: light dark;
  --bg:#fff; --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --card:#f7f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#15171c; --fg:#e8eaee; --mut:#9aa2b1; --line:#2a2e37; --card:#1c1f26; } }
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  max-width:56rem; margin-inline:auto; }
h1 { font-size:1.6rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1.05rem; margin:2.25rem 0 .6rem; padding-bottom:.3rem;
  border-bottom:1px solid var(--line); }
.sub { color:var(--mut); font-size:.9rem; margin:.2rem 0 .8rem; }
.missing { border-left:3px solid var(--line); padding-left:.75rem; }
.stats { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
  margin:1.2rem 0 .5rem; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:.7rem .8rem; }
.stat .label { color:var(--mut); font-size:.75rem; text-transform:uppercase;
  letter-spacing:.04em; }
.stat .value { font-size:1.3rem; font-weight:600; margin-top:.15rem;
  font-variant-numeric:tabular-nums; }
.stat .note { color:var(--mut); font-size:.75rem; }
.tw { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th,td { text-align:left; padding:.4rem .7rem; border-bottom:1px solid var(--line);
  vertical-align:top; }
th { color:var(--mut); font-weight:600; font-size:.75rem; text-transform:uppercase;
  letter-spacing:.04em; }
td.r,th.r { text-align:right; font-variant-numeric:tabular-nums; }
tbody tr:last-child td { border-bottom:none; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em;
  background:var(--card); padding:.08em .35em; border-radius:4px; }
pre.cmd { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:.7rem .8rem; overflow-x:auto; font-size:.78rem; color:var(--mut); }
.grid { display:grid; gap:1.1rem; grid-template-columns:repeat(auto-fit,minmax(19rem,1fr)); }
figure { margin:0; }
figcaption { font-size:.85rem; font-weight:600; margin-bottom:.2rem; }
figcaption .sub { display:block; font-weight:400; margin:0; }
.chart { width:100%; height:auto; display:block; }
.axis { stroke:var(--line); stroke-width:1; }
.tick { fill:var(--mut); font-size:10px; }
.tick.end { text-anchor:end; }
.bad { color:#e34948; font-weight:600; }
footer { margin-top:3rem; padding-top:.8rem; border-top:1px solid var(--line);
  color:var(--mut); font-size:.78rem; }
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dir")
    ap.add_argument("-o", "--out", default=None,
                    help="output HTML (default: <work-dir>/report.html)")
    a = ap.parse_args(argv)

    run = read_run(a.work_dir)
    if not run["plan"] and not run["summary"] and not run["progress"]:
        raise SystemExit(f"{a.work_dir}: no run_plan.json, run_summary.json or "
                         f"progress manifests — is this a work directory?")
    out = a.out or os.path.join(a.work_dir, "report.html")
    title = os.path.basename(a.work_dir.rstrip("/")) or "run"
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{html.escape(title)} — em-seg-morpho</title>'
           f'<style>{CSS}</style></head><body>{render(run)}</body></html>')
    with open(out, "w") as f:
        f.write(doc)
    print(f"{out}  ({os.path.getsize(out) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
