"""Per-segment skeletonization via kimimaro.

``kimimaro.skeletonize(labels, teasar_params, anisotropy, object_ids, ...)``
returns a skeleton per label. For large segments, skeletonize in blocks and merge
with ``kimimaro.join_close_components`` (analogous to the chunked-mesh path).
"""

from __future__ import annotations

import numpy as np

from .config import SkeletonConfig


def _teasar_params(cfg: SkeletonConfig) -> dict:
    return {
        "scale": cfg.scale,
        "const": cfg.const,
        "pdrf_scale": cfg.pdrf_scale,
        "pdrf_exponent": cfg.pdrf_exponent,
    }


def skeletonize_mask(mask_zyx: np.ndarray, segment_id: int, cfg: SkeletonConfig):
    """Skeletonize one segment's binary mask; returns a kimimaro Skeleton (or None)."""
    import kimimaro

    labels = mask_zyx.astype(np.uint64) * np.uint64(segment_id)
    skels = kimimaro.skeletonize(
        labels, teasar_params=_teasar_params(cfg), anisotropy=cfg.anisotropy,
        object_ids=[segment_id], dust_threshold=cfg.dust_threshold, progress=False,
    )
    return skels.get(segment_id)


def write_skeleton(output_dir: str, segment_id: int, skeleton, cfg: SkeletonConfig) -> None:
    """Write a skeleton in neuroglancer precomputed skeleton format.

    TODO: choose the writer (kimimaro/cloud-volume Skeleton has a precomputed
    encoding, or write the format directly) + the skeleton ``info``.
    """
    raise NotImplementedError("precomputed skeleton writing — implement against the format")
