"""Skeletonization via kimimaro — block-first, mirroring the meshing pipeline.

    stage 1 (:func:`skeletonize_block`) : one block's labels -> per-body fragments
    stage 2 (:func:`fuse_body`)         : a body's fragments -> one skeleton

Block-first rather than per-body-crop because the memory hazard is the bounding
box **extent**, not the voxel count: a sparse arbor occupies few voxels but a
huge bbox, and its dense crop array is what OOMs. A block is bounded by
construction. A body wholly inside one block yields a single fragment, so fusion
is a no-op for it and only large/spanning bodies see real welding.

:func:`skeletonize_body` (crop-based) is kept for one-off work and the comparison
harness.

The alignment discipline (see coords.py) is the important part: run kimimaro with
``anisotropy = voxel_size`` so its vertices are physical nm, then shift by the
block/crop origin so skeletons land in the **same physical-nm world space as the
meshes** — this is what prevents the mesh<->skeleton offset (mesh-n-bone 231668).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import SkeletonConfig
from .coords import crop_origin_nm, skeleton_to_physical


def _teasar_params(cfg: SkeletonConfig) -> dict:
    return {"scale": cfg.scale, "const": cfg.const,
            "pdrf_scale": cfg.pdrf_scale, "pdrf_exponent": cfg.pdrf_exponent}


def _clean_mask(mask_zyx: np.ndarray, cfg: SkeletonConfig) -> np.ndarray:
    """Optional small morphological cleanup before kimimaro (see SkeletonConfig)."""
    import scipy.ndimage as ndi

    b = mask_zyx.astype(bool)
    if cfg.mask_opening_iters:
        b = ndi.binary_opening(b, iterations=cfg.mask_opening_iters)
    if cfg.mask_closing_iters:
        b = ndi.binary_closing(b, iterations=cfg.mask_closing_iters)
    return b


# --------------------------------------------------------------------------- #
# Stage 1: one block
# --------------------------------------------------------------------------- #
def skeletonize_block(seg_block_zyx: np.ndarray, block_origin_vox_zyx: Sequence[int],
                      cfg: SkeletonConfig, allowlist: set[int] | None = None) -> dict[int, object]:
    """Skeletonize every (allowlisted) label in one block. Returns ``{body_id: Skeleton}``.

    ``block_origin_vox_zyx`` is the block's origin in skeleton-scale voxels;
    vertices come back in **global physical nm (zyx)**. ``fix_borders`` makes
    kimimaro route each skeleton through the centre of every block-face contact
    area, so fragments from adjacent blocks meet at the seam and stage 2 can weld
    them.
    """
    import kimimaro

    present = np.unique(seg_block_zyx)
    present = present[present != 0]
    if allowlist is not None:
        present = present[np.isin(present, np.array(sorted(allowlist), dtype=present.dtype))]
    if present.size == 0:
        return {}

    labels = seg_block_zyx
    if cfg.mask_opening_iters or cfg.mask_closing_iters:
        # Per-label cleanup: morphology is a binary op, so there is no way around
        # one pass per body. Closing can make labels overlap; last one wins.
        cleaned = np.zeros_like(labels)
        for lab in present:
            cleaned[_clean_mask(labels == lab, cfg)] = lab
        labels = cleaned

    skels = kimimaro.skeletonize(
        labels, teasar_params=_teasar_params(cfg), anisotropy=cfg.anisotropy,
        object_ids=[int(v) for v in present], dust_threshold=cfg.dust_threshold,
        fix_borders=cfg.fix_borders, progress=False,
    )
    origin_nm = crop_origin_nm(block_origin_vox_zyx, cfg.anisotropy)   # anisotropy = skel voxel size
    out: dict[int, object] = {}
    for body_id, skel in skels.items():
        if len(skel.vertices) == 0:
            continue
        skel.vertices = skeleton_to_physical(skel.vertices, origin_nm).astype(np.float32)
        skel.id = int(body_id)
        out[int(body_id)] = skel
    return out


# --------------------------------------------------------------------------- #
# Stage 2: one body
# --------------------------------------------------------------------------- #
def join_radius_nm(cfg: SkeletonConfig) -> float:
    """Resolve ``cfg.join_radius_nm``; ``None`` means "seam scale" (2 voxels)."""
    if cfg.join_radius_nm is None:
        return 2.0 * max(cfg.anisotropy)
    return float(cfg.join_radius_nm)


def fuse_body(fragments: Sequence[object], cfg: SkeletonConfig, body_id: int | None = None):
    """Weld a body's block fragments into one skeleton (or None if nothing survives).

    Two joins happen, and it matters that they are different:

    1. An explicit ``join_close_components`` bounded by :func:`join_radius_nm`,
       whose only job is the **block seams**. Fragments already share physical nm
       coordinates and ``fix_borders`` put their endpoints at the centre of the
       contact area, so a seam is ~a voxel wide. This is bounded because the join
       adds a *straight edge between the nearest vertex pair* — with an unbounded
       radius it will happily bridge a genuine segmentation split with hundreds of
       nm of cable that no biology produced.
    2. ``postprocess``, which drops dust components, breaks loops, runs its own
       ``restrict_by_radius=True`` join (connects two pieces only where the gap is
       smaller than the sum of their local radii — i.e. their cross-sections
       nearly touch), then removes short ticks.

    Components too far apart to join are **kept, disconnected**, not discarded —
    the only thing that deletes a component is ``postprocess``'s dust threshold.
    """
    import kimimaro
    from osteoid import Skeleton

    frags = [f for f in fragments if f is not None and len(f.vertices)]
    if not frags:
        return None

    radius = join_radius_nm(cfg)
    if radius > 0:
        skel = kimimaro.join_close_components(frags, radius=radius)
    else:
        skel = Skeleton.simple_merge(frags)     # postprocess's own join does the welding
    skel = kimimaro.postprocess(skel, dust_threshold=cfg.postprocess_dust_nm,
                                tick_threshold=cfg.postprocess_tick_nm)
    if skel is None or len(skel.vertices) == 0:
        return None
    # join_close_components rebuilds from components, so the segid does not survive.
    skel.id = int(body_id) if body_id is not None else getattr(frags[0], "id", None)
    return skel


def skeleton_metrics(skeleton) -> dict:
    """Per-body morphology metrics for the metrics DB (all lengths in nm)."""
    radius = np.asarray(getattr(skeleton, "radius", []), dtype=float)
    radius = radius[radius > 0]                 # kimimaro leaves -1 where unknown
    return {
        "cable_length_nm": float(skeleton.cable_length()),
        "n_branches": int(len(skeleton.branches())),
        "n_tips": int(len(skeleton.terminals())),
        "max_radius_nm": float(radius.max()) if radius.size else None,
    }


# --------------------------------------------------------------------------- #
# Per-body crop path (one-offs / comparison harness)
# --------------------------------------------------------------------------- #
def skeletonize_body(mask_zyx: np.ndarray, body_id: int, crop_origin_vox_zyx, cfg: SkeletonConfig):
    """Skeletonize one body's crop; returns a Skeleton in **global nm** (or None).

    ``mask_zyx`` is the (binary) body crop at the skeleton scale; ``crop_origin_vox_zyx``
    is the crop's origin in that scale's voxels. Vertices are placed in the same
    physical-nm world as the meshes. Beware the bbox-extent memory hazard that
    motivated the block-first pipeline — prefer :func:`skeletonize_block` at scale.
    """
    import kimimaro

    if cfg.mask_opening_iters or cfg.mask_closing_iters:
        mask_zyx = _clean_mask(mask_zyx, cfg)
    labels = mask_zyx.astype(np.uint64) * np.uint64(body_id)
    skels = kimimaro.skeletonize(
        labels, teasar_params=_teasar_params(cfg), anisotropy=cfg.anisotropy,
        object_ids=[body_id], dust_threshold=cfg.dust_threshold, progress=False,
    )
    skel = skels.get(body_id)
    if skel is None:
        return None
    origin_nm = crop_origin_nm(crop_origin_vox_zyx, cfg.anisotropy)   # anisotropy = skel voxel size
    skel.vertices = skeleton_to_physical(skel.vertices, origin_nm).astype(np.float32)
    skel.id = int(body_id)
    return skel
