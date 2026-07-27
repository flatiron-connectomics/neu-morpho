"""Configuration for the block-first meshing pipeline and skeletonization.

Meshing is **block-first, two-stage** (see docs/DESIGN.md): stage 1 meshes each
non-empty block (all/allowlisted labels at once) into per-(body, block) fragments;
stage 2 concatenates + stitches each body's fragments into a multi-resolution
mesh. No whole-object binary mask is ever built.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeshConfig:
    """Tunables for block-first meshing."""

    # Segmentation scale to read/mesh at. 0 = full detail (biggest/slowest); a
    # coarser scale is smaller and cuts per-block mask memory ~8×/level. The
    # caller passes the meshing scale's voxel size (nm) so coordinates stay in
    # physical world space — no assumed 2**scale pyramid factor (see coords.py).
    mesh_scale: int = 2
    block_shape: tuple[int, int, int] = (256, 256, 256)   # block size at mesh_scale

    # Per-block simplification during stage 1 (fixed-edge preserves block
    # boundaries so stage-2 assembly works on already-reduced meshes).
    decimation_fraction: float = 0.1
    smoothing_iterations: int = 0

    # Multi-resolution output.
    num_lods: int = 3
    lod_decimation_factor: float = 2.0       # face reduction per coarser LOD
    draco_quantization_bits: int = 10        # must be 10 or 16
    sharded: bool = False

    min_segment_voxels: int = 0             # skip bodies smaller than this (if known)
    fragment_format: str = "drc"            # per-fragment on-disk format (Draco)


@dataclass
class SkeletonConfig:
    """kimimaro TEASAR params. Skeletonization stays per-body (bbox-seed crop)."""

    anisotropy: tuple[float, float, float] = (8.0, 8.0, 8.0)   # (z, y, x) nm
    skeleton_scale: int = 2
    scale: float = 1.5
    const: float = 150.0
    pdrf_scale: float = 100000.0
    pdrf_exponent: int = 4
    dust_threshold: int = 50
    bbox_margin_vox: int = 2                  # pad the DB bbox when cropping a body

    # Optional mask cleanup before kimimaro, to tame convolution from imperfect
    # segmentation. KEEP SMALL (1 iter = 1 voxel): opening removes tiny boundary
    # protrusions (spurious-branch sources) but at a COARSE skeleton_scale a
    # voxel is large and can sever thin processes / merge dense arbors — so both
    # default OFF; enable cautiously per data.
    mask_opening_iters: int = 0
    mask_closing_iters: int = 0


@dataclass
class OutputConfig:
    """Where results go."""

    dst: str = ""                            # final multires meshes (local or s3://)
    chunked_dir: str = ""                    # stage-1 fragments (local/ceph); default dst+"/chunked"
    mesh_dir: str = "mesh"                   # multires mesh subpath under dst
    skeleton_dir: str = "skeleton"
    progress_path: str | None = None         # em-blockrun manifest (default derived from dst)
    extra: dict = field(default_factory=dict)
