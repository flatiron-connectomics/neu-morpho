"""Block-first, two-stage meshing orchestrated with em-blockrun.

    stage 1 "chunk"    : block_map over non-empty blocks  -> per-(body,block) fragments
    stage 2 "assemble" : block_map over bodies (from fragments) -> multires meshes

One manifest, two groups ("chunk" keyed by block index, "assemble" keyed by body
id — the generalized em-blockrun keys). Resume skips done blocks/bodies; running
only stage 2 reuses fragments on disk (mesh-n-bone's ``reuse_existing_chunked``).

Scale handling is the caller's: pass ``seg_spec`` opened at the meshing scale and
``occupancy_spec`` at a coarse scale, with their voxel sizes (nm) — so the op
stays format-agnostic (precomputed scale_index / zarr level).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Sequence

from em_blockrun import Manifest, block_map, iter_blocks

from ..allowlist import load_allowlist
from ..config import MeshConfig, OutputConfig
from ..coords import block_chunk_shape_xyz, physical_box
from ..mesh import assemble_body, mesh_block, mesh_metrics
from ..occupancy import occupied_blocks
from ..precomputed import check_quantization, write_body_multires, write_mesh_info
from .. import fragments as _frag
from .. import roi as _roi
from ._progress import (FAILED, FailureBreaker, check_manifest_matches_output,
                        group_counts, guarded, is_complete, write_failures)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Picklable workers
# --------------------------------------------------------------------------- #
def _chunk_block(block, *, seg_spec: dict, chunked_dir: str, mesh_cfg: MeshConfig,
                 allow: set[int] | None, mesh_voxel_size: Sequence[float]) -> tuple:
    from em_volume_tools.backends.base import open_backend

    seg = open_backend(seg_spec).read_region(block.region)      # one block at the mesh scale
    meshes = mesh_block(seg, physical_box(block.region, mesh_voxel_size), mesh_cfg, allow)
    for body_id, m in meshes.items():
        _frag.write_fragment(chunked_dir, body_id, block.index, m, mesh_cfg.fragment_format)
    return (block.index, "written" if meshes else "empty")


def _assemble_body(body_id: int, *, chunked_dir: str, out_dir: str, mesh_cfg: MeshConfig,
                   chunk_shape_xyz: Sequence[int], grid_origin_xyz: Sequence[int]) -> tuple:
    """Returns ``(body_id, status, metrics, info)`` — the driver splits these out.

    Metrics are taken from the assembled LOD-0 mesh, before
    ``write_body_multires`` decimates it in place for the coarser LODs.
    Exceptions become ``failed`` via ``_progress.guarded``.
    """
    frags = _frag.read_body_fragments(chunked_dir, body_id, mesh_cfg.fragment_format)
    if not frags:
        return (body_id, "empty", None, {})
    mesh = assemble_body(frags, mesh_cfg)
    metrics = mesh_metrics(mesh)
    n = write_body_multires(out_dir, body_id, mesh, mesh_cfg,
                            chunk_shape_xyz=chunk_shape_xyz, grid_origin_xyz=grid_origin_xyz)
    return (body_id, "written" if n else "empty", metrics if n else None, {})


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def meshify(
    seg_spec: dict,
    out: OutputConfig,
    mesh_cfg: MeshConfig | None = None,
    *,
    mesh_voxel_size: Sequence[float],
    allowlist: Any = None,
    occupancy_spec: dict | None = None,
    occupancy_voxel_size: Sequence[float] | None = None,
    occupancy_dilate: int = 1,
    roi: Sequence[int] | str | None = None,
    db_path: str | None = None,
    max_consecutive_failures: int = 10,
    stages: Sequence[str] = ("chunk", "assemble"),
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Mesh a segmentation body-by-body via block-first chunking + assembly.

    ``roi`` (``"z0,y0,x0,z1,y1,x1"`` or a 6-sequence, in mesh-scale voxels)
    restricts stage 1 to the blocks intersecting it, on the same global grid, so a
    trial run is a prefix of the full run (see roi.py). A body straddling the ROI
    edge is meshed only from the blocks inside it.

    With ``db_path``, each assembled body's mesh metrics (surface area, vertex
    count, component count) are written to the metrics DB by the driver, which is
    its sole writer — the same arrangement skeletonization uses. Note the metrics
    then describe whatever was meshed: under an ROI, that is the truncated body.

    Stage-2 failures are isolated per body (stage 1 is not — see
    ``ops/_progress.py``). ``max_consecutive_failures`` stops the stage when the
    failures stop looking incidental; 0 disables it. A systemic error
    (``MemoryError``, a full disk, a broken import) aborts immediately regardless.
    """
    from em_volume_tools.backends.base import open_backend

    mesh_cfg = mesh_cfg or MeshConfig()
    allow = load_allowlist(allowlist)

    shape = open_backend(seg_spec).shape                        # (z, y, x) at mesh scale
    grid_shape = tuple(-(-shape[a] // mesh_cfg.block_shape[a]) for a in range(3))
    roi = _roi.clip_to_shape(_roi.parse_roi(roi), shape)
    blocks = _roi.filter_blocks(iter_blocks(shape, mesh_cfg.block_shape), roi)

    if occupancy_spec is not None:
        if occupancy_voxel_size is None:
            raise ValueError("occupancy_voxel_size is required with occupancy_spec")
        occ_be = open_backend(occupancy_spec)
        occ_arr = occ_be.read_region(tuple(slice(0, s) for s in occ_be.shape))
        occupied = occupied_blocks(occ_arr, occ_voxel_size=occupancy_voxel_size,
                                   mesh_voxel_size=mesh_voxel_size,
                                   block_shape=mesh_cfg.block_shape, grid_shape=grid_shape,
                                   allowlist=None, dilate=occupancy_dilate)
        blocks = [b for b in blocks if b.index in occupied]

    out.check_work_dir_is_local()
    out_dir = out.mesh_out()                 # inside the volume; may be s3://
    # Fragments and the manifest live in the POSIX work dir, never in dst: the
    # fragment store is read back with ordinary file I/O and the manifest is
    # appended to. Because they no longer share dst's fate, a manifest can
    # outlive its data — the driver's resume guard is what catches that.
    chunked_dir = out.chunked_dir or out.work("chunked")
    progress = out.progress_path or out.work("progress.mesh.jsonl")
    # multires octree base, in nm (model space = physical nm, grid origin at 0)
    chunk_shape_xyz = block_chunk_shape_xyz(mesh_cfg.block_shape, mesh_voxel_size)
    grid_origin_xyz = [0.0, 0.0, 0.0]

    # Before anything expensive: would Draco quantization collapse coarse LODs?
    check_quantization(mesh_cfg, chunk_shape_xyz, mesh_voxel_size)

    manifest = Manifest(progress)
    manifest.load() if resume else manifest.reset()
    # Must precede write_mesh_info, which would recreate the very 'info' the
    # check probes for and so mask a destination that was cleared.
    check_manifest_matches_output(manifest, out_dir, stage="mesh",
                                  progress_path=progress, resume=resume)

    write_mesh_info(out_dir, mesh_cfg)      # identity transform (vertices are nm)

    db = None
    if db_path is not None:
        from ..metrics_db import MetricsDB
        db = MetricsDB(db_path)
    failures: list[dict] = []           # isolated stage-2 failures (see _progress.py)
    failures_path = None
    breaker = FailureBreaker(max_consecutive_failures)
    try:
        if "chunk" in stages:
            # Stage 1 is deliberately NOT fault-isolated: see ops/_progress.py.
            todo = [b for b in blocks if not (resume and is_complete(manifest, "chunk", b.index))]
            worker = functools.partial(_chunk_block, seg_spec=seg_spec, chunked_dir=chunked_dir,
                                       mesh_cfg=mesh_cfg, allow=allow,
                                       mesh_voxel_size=tuple(mesh_voxel_size))
            block_map(todo, worker, client=client, npartitions=npartitions,
                      on_result=lambda r: manifest.record("chunk", r))

        assembled = 0
        if "assemble" in stages:
            bodies = _frag.list_bodies(chunked_dir)
            if allow is not None:
                bodies = [b for b in bodies if b in allow]
            todo = [b for b in bodies if not (resume and is_complete(manifest, "assemble", b))]
            assembled = len(todo)
            worker = functools.partial(
                guarded, functools.partial(_assemble_body, chunked_dir=chunked_dir,
                                           out_dir=out_dir, mesh_cfg=mesh_cfg,
                                           chunk_shape_xyz=chunk_shape_xyz,
                                           grid_origin_xyz=grid_origin_xyz))

            def on_result(batch):         # runs in the driver (sole DB/manifest writer)
                rows = []
                for body_id, status, metrics, info in batch:
                    if status == FAILED:
                        logger.warning("assemble failed for body %s: %s",
                                       body_id, (info or {}).get("error"))
                        failures.append({"body_id": int(body_id), **(info or {})})
                        breaker.failure(body_id, info)
                        continue
                    breaker.success()
                    if metrics:
                        rows.append((body_id, metrics))
                # One transaction per batch, not per body (see metrics_db).
                if db is not None and rows:
                    db.update_bodies(rows)
                # Manifest last: it marks the body finished, so it must follow the
                # metrics being durable. Still before the breaker can abort.
                manifest.record("assemble", [(b, s) for b, s, _, _ in batch])
                breaker.check()           # raises StageAborted if a trigger fired

            block_map(todo, worker, client=client, npartitions=npartitions,
                      on_result=on_result)
    finally:
        # In `finally` so an abort still leaves the diagnostics behind.
        manifest.close()
        if db is not None:
            db.close()
        failures_path = write_failures(out.work("failures.mesh.jsonl"), failures)
        if failures:
            logger.warning("assemble: %d bodies failed and were skipped -> %s "
                           "(re-run to retry them)", len(failures), failures_path)

    return {"out_dir": out_dir, "chunked_dir": chunked_dir, "num_blocks": len(blocks),
            "num_bodies_assembled": assembled,
            "status_counts": group_counts(manifest, "assemble"),
            "chunk_counts": group_counts(manifest, "chunk"),
            "failed_bodies": [f["body_id"] for f in failures],
            "failures_path": failures_path,
            "progress_path": progress}
