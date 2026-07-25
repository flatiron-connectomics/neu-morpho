"""Write neuroglancer multi-resolution precomputed meshes (stage-2 output).

Thin wrappers over ``vol2mesh.multires``:
  - :func:`write_mesh_info` -> ``multires.write_info`` (the mesh ``info``, once).
  - :func:`write_body_multires` -> per-LOD octree fragments
    (``multires.split_mesh_for_lod``) + ``multires.write_object_mesh``.
"""

from __future__ import annotations

from typing import Sequence

from .config import MeshConfig


def write_mesh_info(output_dir: str, cfg: MeshConfig, *, transform=None,
                    lod_scale_multiplier: float = 1.0) -> None:
    """Write the multi-resolution mesh ``info`` (once per output volume)."""
    from vol2mesh import multires

    multires.write_info(output_dir, vertex_quantization_bits=cfg.draco_quantization_bits,
                        transform=transform, lod_scale_multiplier=lod_scale_multiplier)


def write_body_multires(output_dir: str, body_id: int, mesh, cfg: MeshConfig,
                        *, chunk_shape_xyz: Sequence[int], grid_origin_xyz: Sequence[int]) -> None:
    """Write one body's multi-resolution mesh from its assembled mesh.

    TODO: generate ``cfg.num_lods`` levels (progressively decimated) and split
    each into octree fragments via ``multires.split_mesh_for_lod`` before
    ``multires.write_object_mesh`` (wire ``lod_scales`` / per-LOD decimation once
    tuned against real data).
    """
    raise NotImplementedError(
        "LOD-fragment generation + multires.write_object_mesh — implement against real data"
    )
