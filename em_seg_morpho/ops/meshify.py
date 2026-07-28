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
import os
from typing import Any, Sequence

from em_blockrun import Manifest, block_map, iter_blocks

from ..allowlist import load_allowlist
from ..config import MeshConfig, OutputConfig
from ..coords import block_chunk_shape_xyz, physical_box
from ..mesh import assemble_body, mesh_block, mesh_metrics
from ..occupancy import occupied_blocks
from ..precomputed import write_body_multires, write_mesh_info
from .. import fragments as _frag
from .. import roi as _roi
from ._progress import group_counts


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
    """Returns ``(body_id, status, metrics)`` — the driver splits off the metrics.

    Metrics are taken from the assembled LOD-0 mesh, before
    ``write_body_multires`` decimates it in place for the coarser LODs.
    """
    frags = _frag.read_body_fragments(chunked_dir, body_id, mesh_cfg.fragment_format)
    if not frags:
        return (body_id, "empty", None)
    mesh = assemble_body(frags, mesh_cfg)
    metrics = mesh_metrics(mesh)
    n = write_body_multires(out_dir, body_id, mesh, mesh_cfg,
                            chunk_shape_xyz=chunk_shape_xyz, grid_origin_xyz=grid_origin_xyz)
    return (body_id, "written" if n else "empty", metrics if n else None)


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
    roi: Sequence[int] | str | None = None,
    db_path: str | None = None,
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
                                   allowlist=allow)
        blocks = [b for b in blocks if b.index in occupied]

    out_dir = out.dst.rstrip("/") + "/" + out.mesh_dir
    chunked_dir = out.chunked_dir or (out.dst.rstrip("/") + "/chunked")
    # INSIDE dst, like the fragments and the metrics DB. A manifest sitting beside
    # dst outlives `rm -rf dst`, and the next run then skips every task as "done"
    # and reports success while writing nothing.
    progress = out.progress_path or (out.dst.rstrip("/") + "/progress.mesh.jsonl")
    write_mesh_info(out_dir, mesh_cfg)      # identity transform (vertices are nm)

    # multires octree base, in nm (model space = physical nm, grid origin at 0)
    chunk_shape_xyz = block_chunk_shape_xyz(mesh_cfg.block_shape, mesh_voxel_size)
    grid_origin_xyz = [0.0, 0.0, 0.0]

    db = None
    if db_path is not None:
        from ..metrics_db import MetricsDB
        db = MetricsDB(db_path)

    manifest = Manifest(progress)
    manifest.load() if resume else manifest.reset()
    try:
        if "chunk" in stages:
            todo = [b for b in blocks if not (resume and manifest.is_done("chunk", b.index))]
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
            todo = [b for b in bodies if not (resume and manifest.is_done("assemble", b))]
            assembled = len(todo)
            worker = functools.partial(_assemble_body, chunked_dir=chunked_dir, out_dir=out_dir,
                                       mesh_cfg=mesh_cfg, chunk_shape_xyz=chunk_shape_xyz,
                                       grid_origin_xyz=grid_origin_xyz)

            def on_result(batch):         # runs in the driver (sole DB/manifest writer)
                if db is not None:
                    for body_id, _status, metrics in batch:
                        if metrics:
                            db.update_body(body_id, **metrics)
                manifest.record("assemble", [(b, s) for b, s, _ in batch])

            block_map(todo, worker, client=client, npartitions=npartitions,
                      on_result=on_result)
    finally:
        manifest.close()
        if db is not None:
            db.close()

    return {"out_dir": out_dir, "chunked_dir": chunked_dir, "num_blocks": len(blocks),
            "num_bodies_assembled": assembled,
            "status_counts": group_counts(manifest, "assemble"),
            "chunk_counts": group_counts(manifest, "chunk"),
            "progress_path": progress}
