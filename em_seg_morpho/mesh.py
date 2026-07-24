"""Per-segment mesh generation via vol2mesh.

Small segments: mesh the whole binary mask (``Mesh.from_binary_vol``). Large
segments (bbox mask above the configured budget): mesh **chunked and stitched**
(``Mesh.from_binary_blocks(..., stitch=True)``) so peak memory is bounded by one
block at a time rather than the whole mask.

Masks are read at the configured meshing LOD/scale (``MeshConfig.start_lod``,
default 2) — reading a coarser scale already cuts mask memory ~8×/level, which is
most of what avoids the large-object OOM.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import MeshConfig


def mesh_from_mask(mask_zyx: np.ndarray, fullres_box_zyx, cfg: MeshConfig):
    """Mesh a single binary mask; returns a ``vol2mesh.Mesh``.

    ``fullres_box_zyx`` places the mesh in full-resolution physical coordinates
    even when ``mask_zyx`` is at a downsampled LOD.
    """
    from vol2mesh import Mesh

    mesh = Mesh.from_binary_vol(mask_zyx, np.asarray(fullres_box_zyx))
    if cfg.decimation_fraction and cfg.decimation_fraction < 1.0:
        mesh.simplify(cfg.decimation_fraction)
    return mesh


def mesh_from_blocks(blocks_zyx: Sequence[np.ndarray], fullres_boxes_zyx: Sequence, cfg: MeshConfig):
    """Chunked mesh: mesh each block mask and stitch into one watertight mesh.

    ``blocks_zyx`` are per-block binary masks (each bounded memory), with matching
    ``fullres_boxes_zyx``. vol2mesh meshes each and welds shared boundaries.
    """
    from vol2mesh import Mesh

    mesh = Mesh.from_binary_blocks(list(blocks_zyx), list(fullres_boxes_zyx), stitch=True)
    if cfg.decimation_fraction and cfg.decimation_fraction < 1.0:
        mesh.simplify(cfg.decimation_fraction)
    return mesh


def should_chunk(bbox_shape_zyx: Sequence[int], cfg: MeshConfig) -> bool:
    """True if the segment's (LOD-scaled) bbox mask would exceed the memory budget."""
    return int(np.prod(bbox_shape_zyx)) > cfg.max_mask_voxels
