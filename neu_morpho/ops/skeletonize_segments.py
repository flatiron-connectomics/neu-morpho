"""Block-first, two-stage skeletonization orchestrated with blockrun.

    stage 1 "skel-chunk" : block_map over non-empty blocks -> per-(body,block) fragments
    stage 2 "skel-fuse"  : block_map over bodies (from fragments) -> precomputed skeletons

Deliberately the same shape as ``ops/meshify``: a manifest with two groups
("skel-chunk" keyed by block index, "skel-fuse" keyed by body id), the same
occupancy prefilter and fragment-store pattern, resume by skipping done keys, and
running only stage 2 reuses fragments already on disk. It defaults to its **own**
manifest file, separate from meshing's — the groups would keep resume correct in
a shared one, but ``resume=False`` truncates the file, which would silently throw
away the other pipeline's progress.

The scale is the caller's choice, expressed as ``SkeletonConfig.anisotropy`` —
the skeleton scale's voxel size in nm. Vertices are physical nm throughout, so
they share one model space with the meshes and the ``info`` transform is identity
(see coords.py).
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Sequence

from blockrun import Manifest, block_map, iter_blocks

from ..allowlist import load_allowlist
from ..config import OutputConfig, SkeletonConfig
from ..occupancy import occupied_blocks
from ..precomputed import write_body_skeleton, write_skeleton_info
from ..skeleton import (fuse_body, fusion_stats_summary, skeleton_metrics,
                        skeletonize_block)
from .. import fragments as _frag
from .. import roi as _roi
from ._progress import (FAILED, FailureBreaker, check_manifest_matches_output,
                        group_counts, guarded, is_complete, write_failures)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Picklable workers
# --------------------------------------------------------------------------- #
def _chunk_block(block, *, seg_spec: dict, chunked_dir: str, cfg: SkeletonConfig,
                 allow: set[int] | None) -> tuple:
    from neu_vol.backends.base import open_backend

    seg = open_backend(seg_spec).read_region(block.region)   # one block at the skeleton scale
    origin_vox = tuple(s.start for s in block.region)
    skels = skeletonize_block(seg, origin_vox, cfg, allow)
    for body_id, skel in skels.items():
        _frag.write_skel_fragment(chunked_dir, body_id, block.index, skel, cfg.fragment_format)
    return (block.index, "written" if skels else "empty")


def _fuse_one_body(body_id: int, *, chunked_dir: str, out_dir: str,
                   cfg: SkeletonConfig) -> tuple:
    """Returns ``(body_id, status, metrics, info)`` — the driver splits these out.

    ``empty`` means the body had no fragments; ``dust`` means postprocess consumed
    it (its whole skeleton was shorter than ``postprocess_dust_nm``). They are
    distinct statuses because the second is a thresholding decision that silently
    deletes small bodies, and you should be able to see how often it fires.
    Exceptions become ``failed`` via ``_progress.guarded``.
    """
    frags = _frag.read_body_skel_fragments(chunked_dir, body_id, cfg.fragment_format)
    if not frags:
        return (body_id, "empty", None, {})
    stats: dict = {"body_id": int(body_id)}
    skel = fuse_body(frags, cfg, body_id=body_id, stats=stats)
    if skel is None:
        return (body_id, "dust", None, stats)
    write_body_skeleton(out_dir, body_id, skel)
    return (body_id, "written", skeleton_metrics(skel), stats)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def skeletonize_segments(
    seg_spec: dict,
    out: OutputConfig,
    cfg: SkeletonConfig | None = None,
    *,
    allowlist: Any = None,
    occupancy_spec: dict | None = None,
    occupancy_voxel_size: Sequence[float] | None = None,
    occupancy_dilate: int = 1,
    roi: Sequence[int] | str | None = None,
    db_path: str | None = None,
    fusion_stats_path: str | None = None,
    max_consecutive_failures: int = 10,
    stages: Sequence[str] = ("skel-chunk", "skel-fuse"),
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Skeletonize a segmentation body-by-body via block-first chunking + fusion.

    ``seg_spec`` must be opened at the skeleton scale, whose voxel size (nm) is
    ``cfg.anisotropy``. With ``db_path``, each fused body's metrics (cable length,
    branch/tip counts, max radius) are written to the metrics DB by the driver,
    which is its sole writer.

    ``roi`` (``"z0,y0,x0,z1,y1,x1"`` or a 6-sequence, in skeleton-scale voxels)
    restricts stage 1 to the blocks intersecting it, on the same global grid — so
    a trial run is a prefix of the full run, not a separate one. Note that a body
    straddling the ROI edge is skeletonized only from the blocks inside it, so its
    skeleton is truncated until the neighbouring blocks are run.

    The returned ``fusion_stats`` totals what stage 2 threw away (dust components,
    ticks) and what it inferred (join edges) — the numbers to look at when
    choosing ``postprocess_dust_nm`` / ``postprocess_tick_nm``. Pass
    ``fusion_stats_path`` to also dump the per-body rows as JSONL.

    Stage-2 failures are isolated per body (stage 1 is not — see
    ``ops/_progress.py``). ``max_consecutive_failures`` stops the stage when the
    failures stop looking incidental; 0 disables it. A systemic error
    (``MemoryError``, a full disk, a broken import) aborts immediately regardless.
    """
    from neu_vol.backends.base import open_backend

    cfg = cfg or SkeletonConfig()
    allow = load_allowlist(allowlist)

    shape = open_backend(seg_spec).shape                     # (z, y, x) at skeleton scale
    grid_shape = tuple(-(-shape[a] // cfg.block_shape[a]) for a in range(3))
    # Tile the FULL grid, then filter — block indices and regions stay identical
    # to a full run, so an ROI run's fragments and manifest carry over (roi.py).
    roi = _roi.clip_to_shape(_roi.parse_roi(roi), shape)
    blocks = _roi.filter_blocks(iter_blocks(shape, cfg.block_shape), roi)

    if occupancy_spec is not None:
        if occupancy_voxel_size is None:
            raise ValueError("occupancy_voxel_size is required with occupancy_spec")
        occ_be = open_backend(occupancy_spec)
        occ_arr = occ_be.read_region(tuple(slice(0, s) for s in occ_be.shape))
        # `mesh_voxel_size` is occupancy.py's name for "the target block grid's
        # voxel size" — here that grid is the skeleton scale.
        occupied = occupied_blocks(occ_arr, occ_voxel_size=occupancy_voxel_size,
                                   mesh_voxel_size=cfg.anisotropy,
                                   block_shape=cfg.block_shape, grid_shape=grid_shape,
                                   allowlist=None, dilate=occupancy_dilate)
        blocks = [b for b in blocks if b.index in occupied]

    out.check_work_dir_is_local()
    out_dir = out.skeleton_out()             # inside the volume; may be s3://
    # Fragments and manifest in the POSIX work dir (see meshify) — they no longer
    # share dst's fate, so the driver's resume guard covers the stale case.
    chunked_dir = out.skel_chunked_dir or out.work("skel_chunked")
    progress = out.progress_path or out.work("progress.skel.jsonl")

    manifest = Manifest(progress)
    manifest.load() if resume else manifest.reset()
    # Must precede write_skeleton_info, which would recreate the very 'info' the
    # check probes for and so mask a destination that was cleared.
    check_manifest_matches_output(manifest, out_dir, stage="skel",
                                  progress_path=progress, resume=resume)

    write_skeleton_info(out_dir)             # identity transform (vertices are nm)

    db = None
    if db_path is not None:
        from ..metrics_db import MetricsDB
        db = MetricsDB(db_path)
    fused = 0
    fusion_stats: list[dict] = []       # per-body accounting of what fusion changed
    failures: list[dict] = []           # isolated stage-2 failures (see _progress.py)
    failures_path = None
    breaker = FailureBreaker(max_consecutive_failures)
    try:
        if "skel-chunk" in stages:
            # Stage 1 is deliberately NOT fault-isolated: see ops/_progress.py.
            todo = [b for b in blocks if not (resume and is_complete(manifest, "skel-chunk", b.index))]
            worker = functools.partial(_chunk_block, seg_spec=seg_spec,
                                       chunked_dir=chunked_dir, cfg=cfg, allow=allow)
            block_map(todo, worker, client=client, npartitions=npartitions,
                      on_result=lambda r: manifest.record("skel-chunk", r))

        if "skel-fuse" in stages:
            bodies = _frag.list_bodies(chunked_dir)
            if allow is not None:
                bodies = [b for b in bodies if b in allow]
            todo = [b for b in bodies if not (resume and is_complete(manifest, "skel-fuse", b))]
            fused = len(todo)
            worker = functools.partial(
                guarded, functools.partial(_fuse_one_body, chunked_dir=chunked_dir,
                                           out_dir=out_dir, cfg=cfg))

            def on_result(batch):             # runs in the driver (sole DB/manifest writer)
                rows = []
                for body_id, status, metrics, info in batch:
                    if status == FAILED:
                        logger.warning("skel-fuse failed for body %s: %s",
                                       body_id, (info or {}).get("error"))
                        failures.append({"body_id": int(body_id), **(info or {})})
                        breaker.failure(body_id, info)
                        continue
                    breaker.success()
                    if metrics:
                        rows.append((body_id, metrics))
                    if info:
                        fusion_stats.append(info)
                # One transaction for the whole batch: per-body commits made the
                # driver a serial fsync bottleneck while workers idled.
                if db is not None and rows:
                    db.update_bodies(rows)
                # Mark done only AFTER the metrics are durable — the manifest is
                # the "this body is finished" record, so it goes last. And record
                # before the breaker can abort, or the batch's work is forgotten.
                manifest.record("skel-fuse", [(b, s) for b, s, _, _ in batch])
                breaker.check()               # raises StageAborted if a trigger fired

            block_map(todo, worker, client=client, npartitions=npartitions, on_result=on_result)
    finally:
        # In `finally` so an abort (breaker, or a stage-1 crash) still leaves the
        # diagnostics behind — they are most useful precisely when it aborts.
        manifest.close()
        if db is not None:
            db.close()
        if fusion_stats_path and fusion_stats:
            with open(fusion_stats_path, "w") as f:
                for s in fusion_stats:
                    f.write(json.dumps(s) + "\n")
        failures_path = write_failures(out.work("failures.skel.jsonl"), failures)
        if failures:
            logger.warning("skel-fuse: %d bodies failed and were skipped -> %s "
                           "(re-run to retry them)", len(failures), failures_path)

    return {"out_dir": out_dir, "chunked_dir": chunked_dir, "num_blocks": len(blocks),
            "num_bodies_fused": fused,
            "status_counts": group_counts(manifest, "skel-fuse"),
            "chunk_counts": group_counts(manifest, "skel-chunk"),
            "failed_bodies": [f["body_id"] for f in failures],
            "failures_path": failures_path,
            "fusion_stats": fusion_stats_summary(fusion_stats),
            "fusion_stats_path": fusion_stats_path if fusion_stats else None,
            "progress_path": progress}
