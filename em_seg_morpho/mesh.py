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


def mesh_block(seg_block_zyx: np.ndarray, box_nm_zyx: np.ndarray, cfg: MeshConfig,
               allowlist: set[int] | None = None) -> dict[int, "object"]:
    """Mesh every (allowlisted) label in one block. Returns ``{body_id: Mesh}``.

    ``box_nm_zyx`` is the block's physical (nm) box (see coords.physical_box), so
    vertices come out in nm world coords. Background (0) is excluded; meshes are
    per-block simplified so downstream assembly works on already-reduced geometry.
    """
    # Only mesh labels actually present (intersect with the allowlist). This
    # skips all-background blocks and avoids vol2mesh choking on empty label lists.
    present = np.unique(seg_block_zyx)
    present = present[present != 0]
    if allowlist is not None:
        present = present[np.isin(present, np.array(sorted(allowlist), dtype=present.dtype))]
    if present.size == 0:
        return {}

    from vol2mesh import Mesh

    meshes = Mesh.from_label_volume(seg_block_zyx, box_nm_zyx, labels=present.tolist(),
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


def surface_area_nm2(mesh) -> float:
    """Total triangle area. Vertices are nm (coords.py), so this is nm²."""
    tri = mesh.vertices_zyx[mesh.faces]
    if len(tri) == 0:
        return 0.0
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def count_components(mesh) -> int:
    """Connected components of the mesh, over shared **vertex indices**.

    Worth watching as QC rather than just a descriptor: fragments from adjacent
    blocks only merge once ``stitch_adjacent_faces`` welds their coincident
    boundary vertices, so a spanning body that still reports >1 component is
    telling you the stitch did not take. (A body genuinely split by the
    segmentation also reports >1 — the count alone cannot separate the two.)
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(mesh.vertices_zyx)
    faces = mesh.faces
    if n == 0 or len(faces) == 0:
        return 0
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = sp.coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])),
                          shape=(n, n))
    return int(connected_components(graph, directed=False, return_labels=False))


def mesh_metrics(mesh) -> dict:
    """Per-body mesh metrics for the metrics DB (columns match metrics_db._EXTRA)."""
    return {"mesh_area_nm2": surface_area_nm2(mesh),
            "mesh_verts": int(len(mesh.vertices_zyx)),
            "n_mesh_components": count_components(mesh)}
