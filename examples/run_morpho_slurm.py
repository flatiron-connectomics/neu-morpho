"""Driver: index -> allowlist -> meshes -> skeletons, locally or on Rusty/SLURM.

Run this ON A WORKSTATION, in a session that outlives your terminal. It starts a
dask cluster whose workers are SLURM jobs (dask submits the sbatch itself via
``scale()``), then runs the requested stages against it. Every stage is
idempotent, so re-running the same command resumes.

    # 0. look at the pyramid first, and pick your scales from real metadata
    python examples/run_morpho_slurm.py --src /mnt/ceph/.../seg --describe

    # 1. small ROI, locally, to see it work end to end
    python examples/run_morpho_slurm.py --src ... \\
        --dst /mnt/ceph/.../morpho/segmentation --work-dir /mnt/ceph/.../morpho \\
        --config configs/dask-local.yaml --workers 4 \\
        --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh,skel

    # 2. the same ROI on SLURM, surviving logout, publishing to s3
    nohup python -u examples/run_morpho_slurm.py --src ... \\
        --dst s3://bucket/sample3/segmentation --work-dir /mnt/ceph/.../morpho \\
        --config configs/dask-slurm-any.yaml --workers 48 \\
        --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh,skel > run.log 2>&1 &
    squeue -u "$USER"        # watch your jobs (read-only; don't poll in a tight loop)

**--dst is the published volume; --work-dir is everything else.** --dst holds the
segmentation scales with meshes and skeletons inside it, and may be an ``s3://``
URL. --work-dir must be an ordinary filesystem path: it holds the stage-1
fragments, the sqlite metrics DB and the appended JSONL manifests, none of which
work over an object store.

Because the two are separate, a manifest can outlive the data it describes —
clear --dst and a resumed run would skip everything and report success having
written nothing. Each stage refuses to resume when its manifest records completed
work but its output ``info`` is gone; --no-resume is the explicit override.

The ROI is in the voxels of **--roi-scale, which is required with --roi** — the same
six numbers name a box 4x smaller per pyramid level, and the wrong scale silently
processes the wrong region rather than failing. It filters blocks on the *global*
grid, so widening it later re-uses everything already done rather than redoing it
(see em_seg_morpho/roi.py). Drop --roi for the whole volume.

Scales are integers (--mesh-scale / --skel-scale / --index-scale); the voxel size
of each is read from the source metadata, never assumed to be 2**scale — that
assumption is what misaligns meshes against skeletons on non-standard pyramids.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

from em_blockrun import start_dask

from em_seg_morpho.config import MeshConfig, OutputConfig, SkeletonConfig
from em_seg_morpho.metrics_db import MetricsDB
from em_seg_morpho.ops.export_roi_seg import DEFAULT_COPY_BLOCK as SEG_COPY_BLOCK
from em_seg_morpho.ops.export_roi_seg import scale_cost
from em_seg_morpho.ops import (export_roi_seg, index_segments, meshify,
                               skeletonize_segments)
from em_seg_morpho.precomputed import link_subresources
from em_seg_morpho.roi import parse_roi, scale_roi
from em_seg_morpho.scales import describe, read_scales, scale_spec

log = logging.getLogger("em-seg-morpho")

STAGES = ("seg", "index", "mesh", "skel")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="segmentation volume (path or s3://...)")
    p.add_argument("--dst", help="the published precomputed VOLUME: labels with "
                                 "meshes/skeletons inside it. Local path or s3://... "
                                 "NOTE this used to mean the run root; bookkeeping "
                                 "now goes to --work-dir")
    p.add_argument("--work-dir", help="run directory: stage-1 fragments, manifests, "
                                      "metrics DB, failures, run summary. Must be a "
                                      "filesystem path. Defaults to the PARENT of a "
                                      "local --dst (reproducing the old layout); "
                                      "required when --dst is remote")
    p.add_argument("--describe", action="store_true",
                   help="print the source pyramid and exit (no cluster, no writes)")

    p.add_argument("--stages", default="index,mesh,skel",
                   help=f"comma-separated subset of {','.join(STAGES)}; 'seg' copies the "
                        "ROI's labels out as a precomputed volume for viewing")
    p.add_argument("--seg-encoding", default="compressed_segmentation",
                   help="encoding for the exported segmentation (or 'raw')")
    p.add_argument("--seg-scales", default="all",
                   help="source scales to copy in the seg stage: 'all', a range "
                        "like '1-6', or a comma list. Scale 0 usually dominates "
                        "the cost by an order of magnitude — check the log line")
    p.add_argument("--roi", help="z0,y0,x0,z1,y1,x1 voxels, at --roi-scale")
    p.add_argument("--roi-scale", type=int, default=None,
                   help="scale the --roi values are given in. REQUIRED with --roi: "
                        "the same six numbers name a different box at every scale, "
                        "and guessing wrong silently processes the wrong region")

    p.add_argument("--index-scale", type=int, default=2, help="scale to scan for bboxes")
    p.add_argument("--mesh-scale", type=int, default=2)
    p.add_argument("--skel-scale", type=int, default=2)
    p.add_argument("--tracer", default="neutu", choices=("kimimaro", "neutu"),
                   help="stage-1 skeletonizer (default neutu). 'neutu' is "
                        "em_seg_morpho.neutu_trace, which reproduces NeuTu's skeletons "
                        "with fewer vertices and requires isotropic voxels; 'kimimaro' "
                        "is the older tracer, and the one to use on an anisotropic "
                        "pyramid (see docs/skeletonization-plan.md)")
    p.add_argument("--neutu-cost", default="edge", choices=("voxel", "edge"),
                   help="--tracer neutu only (default edge). 'edge' is NeuTu's own "
                        "symmetric cost and matches it more closely. 'voxel' is "
                        "cheaper in memory, which matters only if a component exceeds "
                        "SkeletonConfig.neutu_edge_max_gb (see "
                        "docs/skeletonization-plan.md)")
    p.add_argument("--block", default="256,256,256", help="block shape (z,y,x) voxels")

    p.add_argument("--min-voxels", type=int, default=0,
                   help="only mesh/skeletonize bodies with at least this many voxels")
    p.add_argument("--limit-bodies", type=int, default=None,
                   help="cap the allowlist to the N largest bodies")
    p.add_argument("--allowlist", help="explicit body-id file (overrides --min-voxels)")
    p.add_argument("--no-metrics-db", action="store_true",
                   help="skip the per-body metrics DB (its enrichment is observational; "
                        "with an explicit --allowlist the index stage is unnecessary too)")

    p.add_argument("--occupancy-scale", type=int, default=5,
                   help="coarse scale used to skip empty blocks; -1 disables")
    p.add_argument("--occupancy-dilate", type=int, default=1,
                   help="grow the occupied set by N blocks. Keep >=1: a coarse scale "
                        "misses sparse blocks and the miss does not converge, so an "
                        "un-dilated filter silently skips real data")

    p.add_argument("--config", default="configs/dask-local.yaml")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--serial", action="store_true",
                   help="no dask at all — run in this process (smallest smoke test)")
    p.add_argument("--no-resume", action="store_true", help="start over, ignoring manifests")
    p.add_argument("--max-consecutive-failures", type=int, default=10,
                   help="stop a stage after this many per-body failures in a row "
                        "(0 disables; systemic errors abort immediately regardless)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan (scales, ROI, block counts) and exit")
    p.add_argument("--store-logs", action="store_true",
                   help="keep TensorStore's benign S3 credential-chain logging "
                        "(suppressed by default; real errors are never suppressed)")
    args = p.parse_args(argv)

    # --roi-scale used to default to --mesh-scale. That is a silent default on a
    # number that changes what gets processed: the same six values name a box 4x
    # smaller per level, so an ROI meant for scale 0 read as scale 2 quietly
    # processes 1/64th of the intended volume — and the run still succeeds, still
    # reports blocks, still writes output. Nothing downstream can catch it, because
    # a smaller ROI is indistinguishable from a deliberately smaller ROI. Be explicit.
    if args.roi and args.roi_scale is None:
        p.error("--roi-scale is required with --roi: the same six values name a "
                "different box at every pyramid level, and the wrong one silently "
                "processes the wrong region. Pass the scale the numbers are in "
                "(e.g. --roi-scale 2).")
    if args.roi_scale is not None and not args.roi:
        p.error("--roi-scale has no effect without --roi")
    return args


def _ng_source(volume_dir: str) -> str:
    """The neuroglancer source URL for a volume — local paths need a file:// prefix."""
    from em_volume_tools import is_local

    return (f"precomputed://file://{volume_dir}" if is_local(volume_dir)
            else f"precomputed://{volume_dir}")


def _blocks_in(shape, block, roi, voxel_size=None, occ=None):
    """How many blocks a stage will actually process — the honest size estimate.

    Applies BOTH filters. Reporting the ROI count alone overstates the work by
    the occupancy factor (2.2x on this dataset), which makes --dry-run useless
    for deciding whether a run is affordable.
    """
    from em_blockrun import iter_blocks

    from em_seg_morpho.occupancy import occupied_blocks
    from em_seg_morpho.roi import clip_to_shape, filter_blocks

    blocks = filter_blocks(iter_blocks(shape, block), clip_to_shape(roi, shape))
    if occ is None:
        return len(blocks)
    arr, occ_voxel, dilate = occ
    grid = tuple(-(-shape[a] // block[a]) for a in range(3))
    keep = occupied_blocks(arr, occ_voxel_size=occ_voxel, mesh_voxel_size=voxel_size,
                           block_shape=block, grid_shape=grid, allowlist=None,
                           dilate=dilate)
    return sum(1 for b in blocks if b.index in keep)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    from em_volume_tools.logs import quiet_store_logs

    # TensorStore's S3 stack logs its credential-provider chain at ERROR severity
    # on success; on a real run that buried the actual output. Real failures
    # (PERMISSION_DENIED and friends) are never suppressed — see logs.NEVER_DROP.
    with quiet_store_logs(not args.store_logs):
        return _main(args)


def _main(args) -> int:

    scales = read_scales(args.src)
    if args.describe:
        print(describe(args.src))
        return 0
    if not args.dst:
        raise SystemExit("--dst is required unless --describe")

    # --dst is the volume (possibly s3://); --work-dir is the POSIX run dir. For a
    # local --dst the work dir defaults to its parent, which reproduces the layout
    # from when --dst meant the run root: <run>/segmentation plus <run>/chunked,
    # metrics.db and the manifests beside it.
    from em_volume_tools import is_local

    dst = args.dst.rstrip("/")
    if args.work_dir:
        work = args.work_dir.rstrip("/")
    elif is_local(dst):
        work = os.path.dirname(os.path.abspath(dst))
    else:
        raise SystemExit(
            f"--work-dir is required when --dst is remote ({dst}). It holds the "
            "stage-1 fragments, the sqlite metrics DB and the JSONL manifests, "
            "none of which work over an object store.")
    if not is_local(work):
        raise SystemExit(f"--work-dir must be a filesystem path, got {work}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s) {sorted(unknown)}; pick from {list(STAGES)}")

    block = tuple(int(v) for v in args.block.split(","))
    finest = scales[0]
    for name, idx in (("index", args.index_scale), ("mesh", args.mesh_scale),
                      ("skel", args.skel_scale)):
        if not 0 <= idx < len(scales):
            raise SystemExit(f"--{name}-scale {idx} out of range (source has {len(scales)})")

    idx_s, mesh_s, skel_s = (scales[args.index_scale], scales[args.mesh_scale],
                             scales[args.skel_scale])

    # The ROI is quoted in one scale; convert it into each stage's own voxels
    # using real voxel sizes, so the same physical cube is used everywhere.
    roi_base = parse_roi(args.roi)
    # _parse_args guarantees roi_scale is set whenever there is an ROI; with no ROI
    # the value is unused (roi_for returns None either way).
    roi_scale = scales[args.roi_scale if args.roi_scale is not None else args.mesh_scale]

    def roi_for(target):
        factor = tuple(roi_scale.voxel_size[a] / target.voxel_size[a] for a in range(3))
        return scale_roi(roi_base, factor)

    roi_index, roi_mesh, roi_skel = roi_for(idx_s), roi_for(mesh_s), roi_for(skel_s)

    seg_scales = None
    if args.seg_scales and args.seg_scales != "all":
        if "-" in args.seg_scales:
            a, _, b = args.seg_scales.partition("-")
            seg_scales = list(range(int(a), int(b) + 1))
        else:
            seg_scales = [int(x) for x in args.seg_scales.split(",")]

    occ_spec = occ_voxel = None
    if args.occupancy_scale is not None and args.occupancy_scale >= 0:
        if not 0 <= args.occupancy_scale < len(scales):
            raise SystemExit(f"--occupancy-scale {args.occupancy_scale} out of range")
        occ = scales[args.occupancy_scale]
        occ_spec = scale_spec(args.src, occ.index)
        occ_voxel = occ.voxel_size

    db_path = None if args.no_metrics_db else f"{work}/metrics.db"
    allowlist_path = args.allowlist or f"{work}/allowlist.csv"
    out = OutputConfig(dst=dst, work_dir=work)

    occ = None
    if occ_spec is not None:
        from em_volume_tools.backends.base import open_backend
        _be = open_backend(occ_spec)
        occ = (_be.read_region(tuple(slice(0, x) for x in _be.shape)),
               occ_voxel, args.occupancy_dilate)

    log.info("source pyramid:\n%s", describe(args.src))
    # Keep the counts: they are the denominators scripts/run_progress.py needs, and
    # recomputing them means re-reading the occupancy array.
    planned = {
        "index": _blocks_in(idx_s.shape, block, roi_index, idx_s.voxel_size, occ),
        "mesh": _blocks_in(mesh_s.shape, block, roi_mesh, mesh_s.voxel_size, occ),
        "skel": _blocks_in(skel_s.shape, block, roi_skel, skel_s.voxel_size, occ),
    }
    for name, s in (("index", idx_s), ("mesh", mesh_s), ("skel", skel_s)):
        log.info("%-5s  scale %d  voxel %s nm  shape %s  -> %d blocks",
                 name, s.index, s.voxel_size, s.shape, planned[name])
    log.info("stages=%s roi=%s occupancy=%s", stages, roi_base,
             f"scale {args.occupancy_scale} dilate {args.occupancy_dilate}"
             if occ else "off")
    # The tracer decides what the skeletons ARE, so it belongs in the log and not
    # only in run_plan.json -- a log is what you have when diagnosing an old run.
    if "skel" in stages:
        log.info("tracer        = %s%s", args.tracer,
                 f" (cost={args.neutu_cost})" if args.tracer == "neutu" else "")
    log.info("dst (volume)  = %s", dst)
    log.info("work dir      = %s", work)
    if args.dry_run:
        log.info("--dry-run: nothing executed")
        return 0

    os.makedirs(work, exist_ok=True)
    # The plan, so progress can be reported as a fraction without re-deriving the
    # ROI ∩ occupancy block counts (which costs an occupancy-array read).
    with open(f"{work}/run_plan.json", "w") as f:
        json.dump({
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "dst": dst, "work_dir": work, "stages": stages,
            "block_shape": list(block), "roi": list(roi_base) if roi_base else None,
            # without the scale the six ROI numbers are uninterpretable later
            "roi_scale": args.roi_scale,
            "scales": {"index": idx_s.index, "mesh": mesh_s.index, "skel": skel_s.index},
            # which skeletonizer produced this run's output — the whole point of
            # run_plan is being able to tell later what made the data
            "tracer": args.tracer, "neutu_cost": args.neutu_cost,
            "planned_blocks": planned,
            "seg_scales": [
                {"scale": c["scale"], "shape": list(c["shape"]),
                 "blocks": math.prod(-(-c["shape"][a] // SEG_COPY_BLOCK[a])
                                     for a in range(3))}
                for c in scale_cost(args.src, roi_mesh, mesh_s.voxel_size,
                                    block_shape=block, scale_indices=seg_scales)
            ] if "seg" in stages else [],
        }, f, indent=2)
    t_start = time.time()
    mesh_cfg = MeshConfig(mesh_scale=mesh_s.index, block_shape=block)
    skel_cfg = SkeletonConfig(skeleton_scale=skel_s.index, block_shape=block,
                              anisotropy=skel_s.voxel_size,
                              tracer=args.tracer, neutu_cost=args.neutu_cost)
    resume = not args.no_resume
    summaries: dict[str, dict] = {}
    timing: dict[str, float] = {}          # per-stage minutes, for run_summary

    def run_all(client):
        if "seg" in stages:
            t = time.time()
            for c in scale_cost(args.src, roi_mesh, mesh_s.voxel_size,
                                block_shape=block, scale_indices=seg_scales):
                log.info("  seg scale %d (%3.0f nm): %-22s %6.2f Gvox  ~%6.2f GB raw",
                         c["scale"], c["voxel_size"][0], str(c["shape"]),
                         c["n_voxels"] / 1e9, c["raw_gb"])
            summaries["seg"] = export_roi_seg(
                args.src, out.volume_dir(), roi=roi_mesh,
                roi_voxel_size=mesh_s.voxel_size, scale_indices=seg_scales,
                block_shape=block, encoding=args.seg_encoding,
                progress_path=f"{work}/progress.seg.jsonl",
                client=client, resume=resume)
            timing["seg"] = (time.time() - t) / 60
            log.info("seg: %d scales, %.2f Gvox -> %s  (%.1f min)",
                     len(summaries["seg"]["scales"]),
                     summaries["seg"]["n_voxels_total"] / 1e9,
                     summaries["seg"]["out_dir"], timing["seg"])

        if "index" in stages:
            if db_path is None:
                raise SystemExit("--no-metrics-db is incompatible with the index "
                                 "stage, whose only output is that DB")
            t = time.time()
            summaries["index"] = index_segments(
                scale_spec(args.src, idx_s.index), db_path,
                scan_voxel_size=idx_s.voxel_size, scan_scale=idx_s.index,
                fullres_factor=idx_s.factor_from(finest),
                block_shape=block, roi=roi_index,
                client=client, resume=resume)
            timing["index"] = (time.time() - t) / 60
            log.info("index: %s  (%.1f min)", summaries["index"], timing["index"])

        # size-filter the indexed bodies into an allowlist the later stages honour
        if db_path and args.allowlist is None and (args.min_voxels or args.limit_bodies):
            db = MetricsDB(db_path)
            n = db.write_allowlist(allowlist_path, min_voxels=args.min_voxels,
                                   limit=args.limit_bodies)
            db.close()
            log.info("allowlist: %d bodies -> %s", n, allowlist_path)
        allow = allowlist_path if os.path.exists(allowlist_path) else None

        if "mesh" in stages:
            t = time.time()
            summaries["mesh"] = meshify(
                scale_spec(args.src, mesh_s.index), out, mesh_cfg,
                mesh_voxel_size=mesh_s.voxel_size, allowlist=allow, roi=roi_mesh,
                occupancy_spec=occ_spec, occupancy_voxel_size=occ_voxel,
                occupancy_dilate=args.occupancy_dilate, db_path=db_path,
                max_consecutive_failures=args.max_consecutive_failures,
                client=client, resume=resume)
            timing["mesh"] = (time.time() - t) / 60
            log.info("mesh: %s  (%.1f min)", summaries["mesh"], timing["mesh"])

        if "skel" in stages:
            t = time.time()
            summaries["skel"] = skeletonize_segments(
                scale_spec(args.src, skel_s.index), out, skel_cfg,
                allowlist=allow, roi=roi_skel,
                occupancy_spec=occ_spec, occupancy_voxel_size=occ_voxel,
                occupancy_dilate=args.occupancy_dilate, db_path=db_path,
                fusion_stats_path=f"{work}/fusion_stats.jsonl",
                max_consecutive_failures=args.max_consecutive_failures,
                client=client, resume=resume)
            timing["skel"] = (time.time() - t) / 60
            log.info("skel: %s  (%.1f min)", summaries["skel"], timing["skel"])

    if args.serial:
        log.info("serial mode: no dask client")
        run_all(None)
    else:
        # Keep workers x cores within your QOS CPU cap.
        with start_dask(args.workers, config_path=args.config, label="em-seg-morpho") as client:
            run_all(client)

    # Point the volume info at whichever subresources now exist, so one
    # neuroglancer layer carries labels + meshes + skeletons. After the stages
    # because the seg stage rewrites info, which would drop the keys.
    # Existence is probed via the kvstore, not os.path: vol may be an s3:// URL,
    # and a subresource is "there" when its own info is, not when a directory is.
    from em_volume_tools import exists as _exists

    vol = out.volume_dir()
    linked: dict[str, str] = {}
    if _exists(vol, "info"):
        subs = {k: d for k, d in (("mesh", out.mesh_dir), ("skeletons", out.skeleton_dir))
                if _exists(vol, d, "info")}
        if subs:
            linked = link_subresources(vol, **subs)
            log.info("volume info -> %s", linked)
        log.info("neuroglancer source: %s", _ng_source(vol))
    else:
        log.warning("no volume at %s — run --stages seg to export the labels, "
                    "otherwise meshes/skeletons have no volume to attach to", vol)

    # Isolated per-body failures do not stop the run, so say so loudly and exit
    # non-zero — otherwise a scripted pipeline treats a partial result as clean.
    failed = {stage: s.get("failed_bodies") or [] for stage, s in summaries.items()}
    n_failed = sum(len(v) for v in failed.values())
    for stage, bodies in failed.items():
        if bodies:
            log.warning("%s: %d bodies FAILED and were skipped: %s%s", stage, len(bodies),
                        bodies[:10], " ..." if len(bodies) > 10 else "")
            log.warning("%s: tracebacks in %s; re-run to retry them",
                        stage, summaries[stage].get("failures_path"))

    # Written last, so it describes the finished state including the linking.
    wall = (time.time() - t_start) / 60
    summary = {
        "timing_min": {**{k: round(v, 2) for k, v in timing.items()},
                       "total": round(wall, 2)},
        "started": datetime.fromtimestamp(t_start).astimezone().isoformat(timespec="seconds"),
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "volume": vol,
        "work_dir": work,
        "neuroglancer_source": _ng_source(vol),
        "linked_subresources": linked,
        "n_failed_bodies": n_failed,
        "stages": {k: dict(v) for k, v in summaries.items()},
    }
    with open(f"{work}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("timing (min): %s", summary["timing_min"])
    log.info("done -> %s", f"{work}/run_summary.json")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
