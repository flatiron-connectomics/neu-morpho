"""Driver: index -> allowlist -> meshes -> skeletons, locally or on Rusty/SLURM.

Run this ON A WORKSTATION, in a session that outlives your terminal. It starts a
dask cluster whose workers are SLURM jobs (dask submits the sbatch itself via
``scale()``), then runs the requested stages against it. Every stage is
idempotent, so re-running the same command resumes.

    # 0. look at the pyramid first, and pick your scales from real metadata
    python examples/run_morpho_slurm.py --src /mnt/ceph/.../seg --describe

    # 1. small ROI, locally, to see it work end to end
    python examples/run_morpho_slurm.py --src ... --dst /mnt/ceph/.../morpho \\
        --config configs/dask-local.yaml --workers 4 \\
        --roi 0,0,0,512,2048,2048 --stages index,mesh,skel

    # 2. the same ROI on SLURM, surviving logout
    nohup python -u examples/run_morpho_slurm.py --src ... --dst ... \\
        --config configs/dask-slurm-any.yaml --workers 48 \\
        --roi 0,0,0,512,2048,2048 --stages index,mesh,skel > run.log 2>&1 &
    squeue -u "$USER"        # watch your jobs (read-only; don't poll in a tight loop)

The ROI is in **mesh/skeleton-scale voxels** and filters blocks on the *global*
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
import os
import time

from em_blockrun import start_dask

from em_seg_morpho.config import MeshConfig, OutputConfig, SkeletonConfig
from em_seg_morpho.metrics_db import MetricsDB
from em_seg_morpho.ops import index_segments, meshify, skeletonize_segments
from em_seg_morpho.roi import parse_roi, scale_roi
from em_seg_morpho.scales import describe, read_scales, scale_spec

log = logging.getLogger("em-seg-morpho")

STAGES = ("index", "mesh", "skel")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="segmentation volume (path or s3://...)")
    p.add_argument("--dst", help="output root (meshes, skeletons, fragments, DB)")
    p.add_argument("--describe", action="store_true",
                   help="print the source pyramid and exit (no cluster, no writes)")

    p.add_argument("--stages", default="index,mesh,skel",
                   help=f"comma-separated subset of {','.join(STAGES)}")
    p.add_argument("--roi", help="z0,y0,x0,z1,y1,x1 in mesh/skeleton-scale voxels")
    p.add_argument("--roi-scale", type=int, default=None,
                   help="scale the --roi values are given in (default: --mesh-scale)")

    p.add_argument("--index-scale", type=int, default=2, help="scale to scan for bboxes")
    p.add_argument("--mesh-scale", type=int, default=2)
    p.add_argument("--skel-scale", type=int, default=2)
    p.add_argument("--block", default="256,256,256", help="block shape (z,y,x) voxels")

    p.add_argument("--min-voxels", type=int, default=0,
                   help="only mesh/skeletonize bodies with at least this many voxels")
    p.add_argument("--limit-bodies", type=int, default=None,
                   help="cap the allowlist to the N largest bodies")
    p.add_argument("--allowlist", help="explicit body-id file (overrides --min-voxels)")

    p.add_argument("--config", default="configs/dask-local.yaml")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--serial", action="store_true",
                   help="no dask at all — run in this process (smallest smoke test)")
    p.add_argument("--no-resume", action="store_true", help="start over, ignoring manifests")
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan (scales, ROI, block counts) and exit")
    return p.parse_args(argv)


def _blocks_in(shape, block, roi):
    """How many blocks a stage will actually process — the honest size estimate."""
    from em_blockrun import iter_blocks

    from em_seg_morpho.roi import clip_to_shape, filter_blocks
    return len(filter_blocks(iter_blocks(shape, block), clip_to_shape(roi, shape)))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    scales = read_scales(args.src)
    if args.describe:
        print(describe(args.src))
        return 0
    if not args.dst:
        raise SystemExit("--dst is required unless --describe")

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
    roi_scale = scales[args.roi_scale if args.roi_scale is not None else args.mesh_scale]

    def roi_for(target):
        factor = tuple(roi_scale.voxel_size[a] / target.voxel_size[a] for a in range(3))
        return scale_roi(roi_base, factor)

    roi_index, roi_mesh, roi_skel = roi_for(idx_s), roi_for(mesh_s), roi_for(skel_s)

    dst = args.dst.rstrip("/")
    db_path = f"{dst}/metrics.db"
    allowlist_path = args.allowlist or f"{dst}/allowlist.csv"
    out = OutputConfig(dst=dst)

    log.info("source pyramid:\n%s", describe(args.src))
    log.info("index  scale %d  voxel %s nm  shape %s  -> %d blocks",
             idx_s.index, idx_s.voxel_size, idx_s.shape,
             _blocks_in(idx_s.shape, block, roi_index))
    log.info("mesh   scale %d  voxel %s nm  shape %s  -> %d blocks",
             mesh_s.index, mesh_s.voxel_size, mesh_s.shape,
             _blocks_in(mesh_s.shape, block, roi_mesh))
    log.info("skel   scale %d  voxel %s nm  shape %s  -> %d blocks",
             skel_s.index, skel_s.voxel_size, skel_s.shape,
             _blocks_in(skel_s.shape, block, roi_skel))
    log.info("stages=%s roi=%s dst=%s", stages, roi_base, dst)
    if args.dry_run:
        log.info("--dry-run: nothing executed")
        return 0

    os.makedirs(dst, exist_ok=True)
    mesh_cfg = MeshConfig(mesh_scale=mesh_s.index, block_shape=block)
    skel_cfg = SkeletonConfig(skeleton_scale=skel_s.index, block_shape=block,
                              anisotropy=skel_s.voxel_size)
    resume = not args.no_resume
    summaries: dict[str, dict] = {}

    def run_all(client):
        if "index" in stages:
            t = time.time()
            summaries["index"] = index_segments(
                scale_spec(args.src, idx_s.index), db_path,
                scan_voxel_size=idx_s.voxel_size, scan_scale=idx_s.index,
                fullres_factor=idx_s.factor_from(finest),
                block_shape=block, roi=roi_index,
                client=client, resume=resume)
            log.info("index: %s  (%.1f min)", summaries["index"], (time.time() - t) / 60)

        # size-filter the indexed bodies into an allowlist the later stages honour
        if args.allowlist is None and (args.min_voxels or args.limit_bodies):
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
                db_path=db_path, client=client, resume=resume)
            log.info("mesh: %s  (%.1f min)", summaries["mesh"], (time.time() - t) / 60)

        if "skel" in stages:
            t = time.time()
            summaries["skel"] = skeletonize_segments(
                scale_spec(args.src, skel_s.index), out, skel_cfg,
                allowlist=allow, roi=roi_skel, db_path=db_path,
                fusion_stats_path=f"{dst}/fusion_stats.jsonl",
                client=client, resume=resume)
            log.info("skel: %s  (%.1f min)", summaries["skel"], (time.time() - t) / 60)

    if args.serial:
        log.info("serial mode: no dask client")
        run_all(None)
    else:
        # Keep workers x cores within your QOS CPU cap.
        with start_dask(args.workers, config_path=args.config, label="em-seg-morpho") as client:
            run_all(client)

    with open(f"{dst}/run_summary.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in summaries.items()},
                  f, indent=2, default=str)

    # Isolated per-body failures do not stop the run, so say so loudly and exit
    # non-zero — otherwise a scripted pipeline treats a partial result as clean.
    failed = {stage: s.get("failed_bodies") or [] for stage, s in summaries.items()}
    n_failed = sum(len(v) for v in failed.values())
    if n_failed:
        for stage, bodies in failed.items():
            if bodies:
                log.warning("%s: %d bodies FAILED and were skipped: %s%s", stage, len(bodies),
                            bodies[:10], " ..." if len(bodies) > 10 else "")
                log.warning("%s: tracebacks in %s; re-run to retry them",
                            stage, summaries[stage].get("failures_path"))
    log.info("done -> %s", f"{dst}/run_summary.json")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
