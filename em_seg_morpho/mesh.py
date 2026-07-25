"""Block-first meshing: stage 1 (mesh a block's labels) + stage 2 (assemble a body).

Stage 1 (`mesh_block`): read a segmentation block once and mesh all present (or
allowlisted) labels with ``Mesh.from_label_volume`` — one marching-cubes pass per
label, halo'd for boundary stitching. No whole-object mask is ever built; peak
memory is one block. Per-block simplification keeps stage-2 assembly light.

Stage 2 (`assemble_body`): concatenate a body's block fragments and
``stitch_adjacent_faces`` into one watertight mesh — keeping **all** components
(this is why we can capture full bodies that the old CGAL path split).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import MeshConfig

BBox = tuple  # np.ndarray [[z0,y0,x0],[z1,y1,x1]]


def fullres_box(region: Sequence[slice], fullres_factor: Sequence[int]) -> np.ndarray:
    """Full-resolution bounding box (2x3, zyx) for a block region at the mesh scale."""
    fz, fy, fx = fullres_factor
    (z0, z1), (y0, y1), (x0, x1) = [(s.start, s.stop) for s in region]
    return np.array([[z0 * fz, y0 * fy, x0 * fx], [z1 * fz, y1 * fy, x1 * fx]])


def mesh_block(seg_block_zyx: np.ndarray, fullres_box_zyx: np.ndarray, cfg: MeshConfig,
               allowlist: set[int] | None = None) -> dict[int, "object"]:
    """Mesh every (allowlisted) label in one block. Returns ``{body_id: Mesh}``.

    Background (0) is excluded. Meshes are per-block simplified so downstream
    assembly works on already-reduced geometry.
    """
    from vol2mesh import Mesh

    labels = sorted(allowlist) if allowlist is not None else None
    meshes = Mesh.from_label_volume(seg_block_zyx, fullres_box_zyx, labels=labels,
                                    ensure_halo=True, progress=False)
    meshes.pop(0, None)                                   # drop background
    for m in meshes.values():
        if cfg.decimation_fraction and cfg.decimation_fraction < 1.0:
            # TODO: fixed-edge simplification to preserve block boundaries for stitching.
            m.simplify(cfg.decimation_fraction)
        if cfg.smoothing_iterations:
            m.laplacian_smooth(cfg.smoothing_iterations)
    return meshes


def assemble_body(fragment_meshes: Sequence["object"], cfg: MeshConfig):
    """Concatenate + stitch a body's block fragments into one mesh (all components)."""
    from vol2mesh import concatenate_meshes

    mesh = concatenate_meshes(list(fragment_meshes))
    mesh.stitch_adjacent_faces()                          # weld shared block boundaries
    return mesh
