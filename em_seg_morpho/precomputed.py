"""Write neuroglancer multi-resolution precomputed meshes (stage-2 output).

Thin wrappers over ``vol2mesh.multires``:
  - :func:`write_mesh_info` -> ``multires.write_info`` (the mesh ``info``, once).
  - :func:`write_body_multires` -> per-LOD octree fragments
    (``multires.split_mesh_for_lod``) + ``multires.write_object_mesh``.
"""

from __future__ import annotations

import os
from typing import Sequence

from .config import MeshConfig


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
