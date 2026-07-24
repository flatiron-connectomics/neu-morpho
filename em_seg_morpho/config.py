"""Configuration for meshing and skeletonization.

All tunables live here as dataclasses so runs are reproducible and scriptable.
Notably, the meshing **starting LOD/scale** is configurable (default 2, not the
full-detail scale 0) — whether scale-0 detail is worth the size/time is
data-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeshConfig:
    """Parameters for multi-resolution mesh generation."""

    # Meshing detail: which segmentation scale to mesh at. 0 = highest detail
    # (most vertices, largest, slowest); 2 = a reasonable coarser default.
    start_lod: int = 2
    num_lods: int = 3                       # multi-resolution levels to emit
    decimation_fraction: float = 0.1        # target fraction of faces after simplify
    draco_quantization_bits: int = 10       # Draco position quantization

    # Large-object handling: if a segment's bbox mask exceeds this many voxels,
    # mesh it chunked-and-stitched instead of materializing the whole mask.
    max_mask_voxels: int = 512 ** 3
    chunk_shape: tuple[int, int, int] = (256, 256, 256)

    sharded: bool = False                   # neuroglancer sharded vs unsharded mesh
    min_segment_voxels: int = 0             # skip segments smaller than this


@dataclass
class SkeletonConfig:
    """Parameters for kimimaro TEASAR skeletonization."""

    # Voxel size (z, y, x) in physical units; kimimaro works in physical space.
    anisotropy: tuple[float, float, float] = (8.0, 8.0, 8.0)
    # TEASAR knobs (kimimaro naming).
    scale: float = 4.0
    const: float = 500.0                    # physical units (e.g. nm)
    pdrf_scale: float = 100000.0
    pdrf_exponent: int = 4
    dust_threshold: int = 1000              # skip components below this many voxels
    max_paths: int | None = None

    chunk_shape: tuple[int, int, int] = (256, 256, 256)
    max_mask_voxels: int = 512 ** 3


@dataclass
class OutputConfig:
    """Where results go and how they're addressed."""

    dst: str = ""                           # local path or s3://bucket/prefix (precomputed)
    mesh_dir: str = "mesh"                  # subpath for mesh fragments/index
    skeleton_dir: str = "skeleton"
    progress_path: str | None = None        # em-blockrun manifest (default: derived from dst)
    extra: dict = field(default_factory=dict)
