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
    """kimimaro params for **block-first** skeletonization (mirrors meshing).

    Stage 1 skeletonizes each block's labels with ``fix_borders=True`` (kimimaro
    then routes a skeleton through the centre of each block-face contact area, so
    adjacent blocks' fragments meet at the seam); stage 2 welds a body's fragments
    with ``join_close_components`` + ``postprocess``.

    Block-first is what makes this memory-safe: the OOM risk in per-body cropping
    is the bounding-box **extent**, not the voxel count — a sparse arbor has a
    huge bbox but few voxels, so its dense crop array blows up. One block is a
    fixed, bounded array.
    """

    anisotropy: tuple[float, float, float] = (8.0, 8.0, 8.0)   # (z, y, x) nm at skeleton_scale
    skeleton_scale: int = 2
    block_shape: tuple[int, int, int] = (256, 256, 256)        # block size at skeleton_scale
    scale: float = 1.5
    const: float = 150.0
    pdrf_scale: float = 100000.0
    pdrf_exponent: int = 4
    dust_threshold: int = 50                  # kimimaro per-block dust, in VOXELS
    fix_borders: bool = True                  # required for fragments to meet at block seams
    bbox_margin_vox: int = 2                  # pad the DB bbox when cropping a body

    # -- stage 2 fusion. Thresholds are in **nm** (vertices are physical nm). --
    # join_radius_nm bounds the explicit seam-welding join. Block seams are ~1
    # voxel wide by construction (fix_borders puts both fragments' endpoints at
    # the centre of the contact area), so the default only needs to reach that
    # far. Do NOT raise it to bridge segmentation splits: the join is a straight
    # edge between the nearest vertex pair, so a wide radius invents cable that
    # never existed (measured: a 400 nm split doubled cable_length, 416 -> 858).
    #   None            -> auto, 2 x the largest voxel dimension (seam scale)
    #   0               -> skip the explicit join; rely on postprocess's own
    #                      radius-restricted join (joins only where the two
    #                      pieces' cross-sections nearly touch)
    #   float("inf")    -> join every component into one tree, at any distance
    join_radius_nm: float | None = None
    # postprocess: dust removal, loop breaking, a radius-restricted join, then
    # tick removal. dust_threshold is applied to each COMPONENT of this one body,
    # so a body whose entire skeleton is shorter than it is deleted outright
    # (kimimaro's 1500 nm default does this to genuinely small bodies) — the op
    # reports those as "dust" rather than silently dropping them.
    postprocess_dust_nm: float = 1500.0       # drop disconnected components shorter than this
    postprocess_tick_nm: float = 3000.0       # drop short spurious side branches

    fragment_format: str = "skel"             # per-fragment on-disk extension

    # Optional mask cleanup before kimimaro, to tame convolution from imperfect
    # segmentation. KEEP SMALL (1 iter = 1 voxel): opening removes tiny boundary
    # protrusions (spurious-branch sources) but at a COARSE skeleton_scale a
    # voxel is large and can sever thin processes / merge dense arbors — so both
    # default OFF; enable cautiously per data. In block mode this costs one
    # morphology pass **per label per block**, so it is far from free.
    mask_opening_iters: int = 0
    mask_closing_iters: int = 0


@dataclass
class OutputConfig:
    """Where results go."""

    dst: str = ""                            # final multires meshes (local or s3://)
    chunked_dir: str = ""                    # stage-1 fragments (local/ceph); default dst+"/chunked"
    skel_chunked_dir: str = ""               # stage-1 skeleton fragments; default dst+"/skel_chunked"
    mesh_dir: str = "mesh"                   # multires mesh subpath under dst
    skeleton_dir: str = "skeleton"
    progress_path: str | None = None         # em-blockrun manifest (default derived from dst)
    extra: dict = field(default_factory=dict)
