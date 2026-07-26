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
from ..mesh import assemble_body, mesh_block
from ..occupancy import occupied_blocks
from ..precomputed import write_body_multires, write_mesh_info
from .. import fragments as _frag


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
    frags = _frag.read_body_fragments(chunked_dir, body_id, mesh_cfg.fragment_format)
    if not frags:
        return (body_id, "empty")
    mesh = assemble_body(frags, mesh_cfg)
    n = write_body_multires(out_dir, body_id, mesh, mesh_cfg,
                            chunk_shape_xyz=chunk_shape_xyz, grid_origin_xyz=grid_origin_xyz)
    return (body_id, "written" if n else "empty")


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
    stages: Sequence[str] = ("chunk", "assemble"),
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Mesh a segmentation body-by-body via block-first chunking + assembly."""
    from em_volume_tools.backends.base import open_backend

    mesh_cfg = mesh_cfg or MeshConfig()
    allow = load_allowlist(allowlist)

    shape = open_backend(seg_spec).shape                        # (z, y, x) at mesh scale
    grid_shape = tuple(-(-shape[a] // mesh_cfg.block_shape[a]) for a in range(3))
    blocks = list(iter_blocks(shape, mesh_cfg.block_shape))

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
    progress = out.progress_path or (out.dst.rstrip("/") + ".progress.jsonl")
    write_mesh_info(out_dir, mesh_cfg)      # identity transform (vertices are nm)

    # multires octree base, in nm (model space = physical nm, grid origin at 0)
    chunk_shape_xyz = block_chunk_shape_xyz(mesh_cfg.block_shape, mesh_voxel_size)
    grid_origin_xyz = [0.0, 0.0, 0.0]

    manifest = Manifest(progress)
    manifest.load() if resume else manifest.reset()
    counts_before = manifest.counts()
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
            block_map(todo, worker, client=client, npartitions=npartitions,
                      on_result=lambda r: manifest.record("assemble", r))
    finally:
        manifest.close()

    return {"out_dir": out_dir, "chunked_dir": chunked_dir, "num_blocks": len(blocks),
            "status_counts": manifest.counts(), "progress_path": progress}
