"""Write neuroglancer multi-resolution precomputed meshes (and skeletons).

Thin wrappers over ``vol2mesh.multires``:
  - :func:`write_mesh_info` -> ``multires.write_info`` (the mesh ``info`` file).
  - :func:`write_segment_mesh` -> split the mesh into per-LOD octree fragments
    (``multires.split_mesh_for_lod``) and write the object
    (``multires.write_object_mesh`` / ``encode_multilod_object``).

Output location may be local or ``s3://…`` (see OutputConfig.dst). Skeleton
precomputed writing is stubbed pending the skeleton format decision.
"""

from __future__ import annotations

from typing import Sequence

from .config import MeshConfig


def write_mesh_info(output_dir: str, cfg: MeshConfig, *, transform=None, lod_scale_multiplier: float = 1.0) -> None:
    """Write the multi-resolution mesh ``info`` (once per output volume)."""
    from vol2mesh import multires

    multires.write_info(output_dir, vertex_quantization_bits=cfg.draco_quantization_bits,
                        transform=transform, lod_scale_multiplier=lod_scale_multiplier)


def write_segment_mesh(output_dir: str, segment_id: int, mesh, cfg: MeshConfig,
                       *, chunk_shape_xyz: Sequence[int], grid_origin_xyz: Sequence[int]) -> None:
    """Write one segment's multi-resolution mesh.

    TODO: generate ``num_lods`` levels (progressively simplified) and split each
    into octree fragments via ``multires.split_mesh_for_lod`` before
    ``multires.write_object_mesh``. Wire ``lod_scales`` and per-LOD decimation
    from ``cfg`` once tuned against real data.
    """
    raise NotImplementedError(
        "LOD-fragment generation + multires.write_object_mesh — implement against real data"
    )
