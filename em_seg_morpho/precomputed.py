"""Write neuroglancer precomputed meshes and skeletons (stage-2 output).

Meshes — thin wrappers over ``vol2mesh.multires``:
  - :func:`write_mesh_info` -> ``multires.write_info`` (the mesh ``info``, once).
  - :func:`write_body_multires` -> per-LOD octree fragments
    (``multires.split_mesh_for_lod``) + ``multires.write_object_mesh``.

Skeletons — the ``neuroglancer_skeletons`` format written directly (via osteoid's
``Skeleton.to_precomputed``, which kimimaro already depends on, so no CloudVolume):
  - :func:`write_skeleton_info` -> the skeleton ``info``, once.
  - :func:`write_body_skeleton` -> one binary blob per body.

**Axis order is the alignment trap.** Model space is physical nm, held *zyx*
in memory (``Mesh.vertices_zyx``, kimimaro vertices). Both precomputed formats
store *xyz*. vol2mesh flips for meshes internally; :func:`encode_skeleton` does
the same flip for skeletons. Get it wrong and skeletons come out mirrored through
the z=x diagonal relative to their meshes — the same class of bug as the dropped
crop origin (see coords.py).
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

from .config import MeshConfig

# Written for every body, and declared in the skeleton ``info``. The binary must
# carry exactly these attributes, in this order, so we normalize every skeleton
# to them rather than trusting whatever the producer happened to attach.
SKELETON_VERTEX_ATTRIBUTES = [
    {"id": "radius", "data_type": "float32", "num_components": 1},
    {"id": "vertex_types", "data_type": "uint8", "num_components": 1},
]

IDENTITY_TRANSFORM = [1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0]


def write_mesh_info(output_dir: str, cfg: MeshConfig, *, transform=None,
                    lod_scale_multiplier: float = 1.0) -> None:
    """Write the multi-resolution mesh ``info`` (once per output volume)."""
    from vol2mesh import multires

    multires.write_info(output_dir, vertex_quantization_bits=cfg.draco_quantization_bits,
                        transform=transform, lod_scale_multiplier=lod_scale_multiplier)


def write_body_multires(output_dir: str, body_id: int, mesh, cfg: MeshConfig,
                        *, chunk_shape_xyz: Sequence[float], grid_origin_xyz: Sequence[float]) -> int:
    """Write one body's multi-resolution mesh; returns fragments written (0 if empty).

    LOD 0 is the assembled mesh; each coarser LOD decimates by
    ``cfg.lod_decimation_factor`` and is octree-partitioned by
    ``multires.split_mesh_for_lod``. ``encode_multilod_object`` produces the
    ``<body>`` data file and ``<body>.index`` manifest (unsharded multires format).
    Vertices are in nm (model space); ``info`` transform is identity.
    """
    from vol2mesh import multires

    fragments_by_lod: dict[int, dict] = {}
    lod_mesh = mesh
    for lod in range(cfg.num_lods):
        if lod > 0:
            try:
                lod_mesh.simplify(1.0 / cfg.lod_decimation_factor)   # coarsen (mutates)
            except Exception:
                pass
        fragments_by_lod[lod] = multires.split_mesh_for_lod(lod_mesh, list(chunk_shape_xyz), lod)

    data_bytes, index_bytes, counts = multires.encode_multilod_object(
        fragments_by_lod, list(chunk_shape_xyz), list(grid_origin_xyz),
        vertex_quantization_bits=cfg.draco_quantization_bits)
    if not data_bytes:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{int(body_id)}", "wb") as f:
        f.write(data_bytes)
    with open(f"{output_dir}/{int(body_id)}.index", "wb") as f:
        f.write(index_bytes)
    return sum(counts)


# --------------------------------------------------------------------------- #
# Skeletons
# --------------------------------------------------------------------------- #
def write_skeleton_info(output_dir: str, *, transform: Sequence[float] | None = None,
                        vertex_attributes: Sequence[dict] | None = None) -> None:
    """Write the ``neuroglancer_skeletons`` ``info`` (once per output volume).

    The transform is **identity**: vertices are already physical nm in the same
    model space as the meshes, so neuroglancer must not re-scale them.
    """
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": list(transform if transform is not None else IDENTITY_TRANSFORM),
        "vertex_attributes": list(vertex_attributes or SKELETON_VERTEX_ATTRIBUTES),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/info", "w") as f:
        json.dump(info, f, indent=2)


def encode_skeleton(skeleton) -> bytes:
    """Encode a global-nm **zyx** skeleton as precomputed **xyz** bytes.

    Normalizes the vertex attributes to :data:`SKELETON_VERTEX_ATTRIBUTES` so the
    blob always matches the ``info``. Radii kimimaro did not supply are left as
    its ``-1`` sentinel.
    """
    from osteoid import Skeleton

    verts_zyx = np.asarray(skeleton.vertices, dtype=np.float32).reshape(-1, 3)
    verts_xyz = np.ascontiguousarray(verts_zyx[:, ::-1])          # <-- the flip
    out = Skeleton(vertices=verts_xyz,
                   edges=np.asarray(skeleton.edges, dtype=np.uint32).reshape(-1, 2),
                   segid=getattr(skeleton, "id", None))
    n = len(verts_xyz)
    radius = getattr(skeleton, "radius", None)
    if radius is not None and len(radius) == n:
        out.radius = np.asarray(radius, dtype=np.float32)
    vtypes = getattr(skeleton, "vertex_types", None)
    if vtypes is not None and len(vtypes) == n:
        out.vertex_types = np.asarray(vtypes, dtype=np.uint8)
    out.extra_attributes = [dict(a) for a in SKELETON_VERTEX_ATTRIBUTES]
    return out.to_precomputed()


def write_body_skeleton(output_dir: str, body_id: int, skeleton) -> int:
    """Write one body's skeleton blob; returns the vertex count (0 if empty)."""
    if skeleton is None or len(skeleton.vertices) == 0:
        return 0
    data = encode_skeleton(skeleton)
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{int(body_id)}"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)                     # never leave a torn blob behind
    return len(skeleton.vertices)
