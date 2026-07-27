"""Per-body skeletonization via kimimaro.

Skeletonization stays per-body (a bbox-seed crop), unlike block-first meshing.
The alignment discipline (see coords.py) is the important part: run kimimaro with
``anisotropy = voxel_size`` so its vertices are physical nm, then shift by the
crop origin so the skeleton lands in the **same physical-nm world space as the
meshes** — this is what prevents the mesh↔skeleton offset (mesh-n-bone 231668).
"""

from __future__ import annotations

import numpy as np

from .config import SkeletonConfig
from .coords import crop_origin_nm, skeleton_to_physical


def _teasar_params(cfg: SkeletonConfig) -> dict:
    return {"scale": cfg.scale, "const": cfg.const,
            "pdrf_scale": cfg.pdrf_scale, "pdrf_exponent": cfg.pdrf_exponent}


def skeletonize_body(mask_zyx: np.ndarray, body_id: int, crop_origin_vox_zyx, cfg: SkeletonConfig):
    """Skeletonize one body's crop; returns a kimimaro Skeleton in **global nm** (or None).

    ``mask_zyx`` is the (binary) body crop at the skeleton scale; ``crop_origin_vox_zyx``
    is the crop's origin in that scale's voxels. Vertices are placed in the same
    physical-nm world as the meshes.
    """
    import kimimaro

    if cfg.mask_opening_iters or cfg.mask_closing_iters:
        import scipy.ndimage as ndi          # small morphological cleanup (see SkeletonConfig)
        b = mask_zyx.astype(bool)
        if cfg.mask_opening_iters:
            b = ndi.binary_opening(b, iterations=cfg.mask_opening_iters)
        if cfg.mask_closing_iters:
            b = ndi.binary_closing(b, iterations=cfg.mask_closing_iters)
        mask_zyx = b
    labels = mask_zyx.astype(np.uint64) * np.uint64(body_id)
    skels = kimimaro.skeletonize(
        labels, teasar_params=_teasar_params(cfg), anisotropy=cfg.anisotropy,
        object_ids=[body_id], dust_threshold=cfg.dust_threshold, progress=False,
    )
    skel = skels.get(body_id)
    if skel is None:
        return None
    origin_nm = crop_origin_nm(crop_origin_vox_zyx, cfg.anisotropy)   # anisotropy = skel voxel size
    skel.vertices = skeleton_to_physical(skel.vertices, origin_nm)
    return skel


def write_skeleton(output_dir: str, body_id: int, skeleton, cfg: SkeletonConfig) -> None:
    """Write a skeleton in neuroglancer precomputed skeleton format (identity transform).

    TODO: choose the writer (cloud-volume Skeleton has a precomputed encoding, or
    write the format directly) + the skeleton ``info`` (identity transform, since
    vertices are already nm and aligned with the meshes).
    """
    raise NotImplementedError("precomputed skeleton writing — implement against the format")
