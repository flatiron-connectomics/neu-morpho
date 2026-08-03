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

    origin_nm = crop_origin_nm(block_origin_vox_zyx, cfg.anisotropy)   # anisotropy = skel voxel size

    if cfg.tracer == "neutu":
        skels = _skeletonize_block_neutu(labels, present, cfg)
    elif cfg.tracer == "kimimaro":
        skels = kimimaro.skeletonize(
            labels, teasar_params=_teasar_params(cfg), anisotropy=cfg.anisotropy,
            object_ids=[int(v) for v in present], dust_threshold=cfg.dust_threshold,
            fix_borders=cfg.fix_borders, progress=False,
        )
    else:
        raise ValueError(f"unknown tracer {cfg.tracer!r}; expected 'kimimaro' or 'neutu'")

    out: dict[int, object] = {}
    for body_id, skel in skels.items():
        if len(skel.vertices) == 0:
            continue
        skel.vertices = skeleton_to_physical(skel.vertices, origin_nm).astype(np.float32)
        skel.id = int(body_id)
        out[int(body_id)] = skel
    return out


def _label_boxes(labels):
    """Per-label bounding box and voxel count, in **one pass** over the block.

    Doing this per label costs a full-array scan each time, and a 256^3 block holds
    ~1000 allowlisted labels — measured at 561 s for the block versus 72 s when each
    label is cropped to its own extent first.
    """
    idx = np.argwhere(labels != 0)
    if not len(idx):
        return {}
    vals = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    order = np.argsort(vals, kind="stable")
    vals = vals[order]
    idx = idx[order]
    bounds = np.flatnonzero(np.diff(vals)) + 1
    out = {}
    for part, lab in zip(np.split(idx, bounds), vals[np.concatenate(([0], bounds))]):
        out[int(lab)] = (part.min(0), part.max(0) + 1, len(part))
    return out


def _skeletonize_block_neutu(labels, present, cfg: SkeletonConfig) -> dict[int, object]:
    """Trace each label in a block with ``neutu_trace``. Vertices/radii in **nm**.

    Unlike ``kimimaro.skeletonize``, which batches every ``object_ids`` entry in one
    call, ``neutu_trace.skeletonize`` takes a single binary mask — so this loops. Two
    things are hoisted out of that loop because they are full-block passes and the
    block holds ~1000 labels (measured: 561 s per block naively, 72 s cropped):

    - **bounding boxes**, so each trace sees only the label's own extent;
    - **face-contact targets**, computed once over the raw label array. They must be
      found on the BLOCK's faces, not the crop's — cropping moves the faces, and
      adjacent blocks would stop agreeing on the seam point.

    **The unit conversion is load-bearing.** kimimaro is handed
    ``anisotropy=voxel_size`` and returns nm, so the caller only adds the crop origin.
    ``neutu_trace`` works in *voxels* by design (its cost is not scale-invariant), so
    both vertices **and radii** must be scaled here. Missing the radii would publish
    values 32x too small at scale 2 and pass every geometric test, since positions
    would still be right.
    """
    from . import neutu_trace, swc_simplify

    vs = np.asarray(cfg.anisotropy, dtype=float)
    if not np.allclose(vs, vs[0]):
        raise ValueError(
            f"tracer='neutu' requires isotropic voxels, got anisotropy={cfg.anisotropy}. "
            "Its 1/(1+r^2) cost is not scale-invariant and NeuTu's own EDT is not "
            "anisotropy-aware, so there is no faithful anisotropic form to use.")

    boxes = _label_boxes(labels)
    face_pts = {}
    if cfg.fix_borders:
        # compute_border_targets remaps through get_mapping, so passing the raw body
        # ids (not cc labels) returns targets keyed by body id in one pass
        face_pts = neutu_trace.border_targets(np.ascontiguousarray(labels))

    wanted = {int(v) for v in present}
    out: dict[int, object] = {}
    for body_id in wanted:
        box = boxes.get(body_id)
        if box is None:
            continue
        lo, hi, count = box
        if count < cfg.dust_threshold:
            continue
        lo = np.maximum(0, lo - 1)
        hi = np.minimum(np.asarray(labels.shape), hi + 1)
        sub = np.ascontiguousarray(
            labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == body_id)
        ft = face_pts.get(body_id)
        ft = (np.asarray(ft, np.int64) - lo) if ft is not None and len(ft) else None
        skel = neutu_trace.skeletonize(
            sub, scale=cfg.neutu_scale, const=cfg.neutu_const_vox,
            min_length=cfg.neutu_min_length_vox,
            face_targets=ft if ft is not None else np.zeros((0, 3), np.int64),
            dust_threshold=cfg.dust_threshold)
        v = np.asarray(skel.vertices, dtype=float)
        if not len(v):
            continue
        r = np.asarray(skel.radii, dtype=float)
        e = np.asarray(skel.edges, dtype=int)
        if cfg.neutu_simplify:
            v, r, e = swc_simplify.simplify(v, r, e)
            if not len(v):
                continue
        # Build a FRESH skeleton rather than mutating the one neutu_trace returned.
        # osteoid carries attribute arrays beyond vertices/radii/edges — vertex_types
        # among them — sized to the original vertex count. Reassigning vertices after
        # swc_simplify shrinks them leaves those stale, and to_precomputed then
        # refuses the fragment ("Number of uint8 vertex_types (34) must match the
        # number of vertices (16)"). Unit tests on vertices/radii/edges cannot see it;
        # the end-to-end op can.
        from osteoid import Skeleton

        v = v + lo                                            # crop -> block voxels
        fresh = Skeleton(vertices=(v * vs).astype(np.float32),
                         edges=np.asarray(e, dtype=np.uint32).reshape(-1, 2),
                         segid=body_id)
        fresh.radii = (r * float(vs[0])).astype(np.float32)    # voxels -> nm
        out[body_id] = fresh
    return out


# --------------------------------------------------------------------------- #
# Stage 2: one body
# --------------------------------------------------------------------------- #
def join_radius_nm(cfg: SkeletonConfig) -> float:
    """Resolve ``cfg.join_radius_nm``; ``None`` means "seam scale" (2 voxels)."""
    if cfg.join_radius_nm is None:
        return 2.0 * max(cfg.anisotropy)
    return float(cfg.join_radius_nm)


def normalize_dtypes(skeleton):
    """Force vertices to float32 and edges to uint32, in place.

    kimimaro's compiled ``skeletontricks.create_distance_graph`` (reached from
    ``remove_ticks``) declares ``float`` / ``uint32_t`` buffers and raises
    ``ValueError: Buffer dtype mismatch`` on anything else. Both dtypes drift in
    normal use: ``kimimaro.skeletonize`` hands back int64 edges under numpy 2,
    arithmetic on vertices promotes them to float64, and — the one that actually
    bites — ``kimimaro.post.remove_loops`` rebuilds edges as int64. Since
    ``postprocess`` runs ``remove_loops`` before ``remove_ticks``, stock
    postprocess raises for some arbors at any non-zero ``tick_threshold``, no
    matter how clean its input was. Call this between the two.
    """
    if skeleton is None:
        return skeleton
    if skeleton.vertices.dtype != np.float32:
        skeleton.vertices = skeleton.vertices.astype(np.float32)
    if skeleton.edges.dtype != np.uint32:
        skeleton.edges = skeleton.edges.astype(np.uint32)
    radius = getattr(skeleton, "radius", None)
    if radius is not None and radius.dtype != np.float32:
        skeleton.radius = radius.astype(np.float32)
    return skeleton


def _profile(skeleton) -> dict:
    """Cheap shape summary used to attribute what each fusion step changed."""
    if skeleton is None or len(skeleton.vertices) == 0:
        return {"comps": 0, "cable": 0.0, "verts": 0, "tips": 0, "branches": 0}
    return {"comps": len(skeleton.components()), "cable": float(skeleton.cable_length()),
            "verts": int(len(skeleton.vertices)), "tips": int(len(skeleton.terminals())),
            "branches": int(len(skeleton.branches()))}


def fuse_body(fragments: Sequence[object], cfg: SkeletonConfig, body_id: int | None = None,
              stats: dict | None = None):
    """Weld a body's block fragments into one skeleton (or None if nothing survives).

    This inlines ``kimimaro.postprocess``'s own sequence (dust -> loops ->
    radius-restricted join -> ticks) rather than calling it, so every step's
    effect can be measured and reported; ``tests/test_skeletonize_e2e.py`` pins
    the result against ``kimimaro.postprocess`` so the two cannot drift.

    Two joins happen, and it matters that they are different:

    1. An explicit ``join_close_components`` bounded by :func:`join_radius_nm`,
       whose only job is the **block seams**. Fragments already share physical nm
       coordinates and ``fix_borders`` put their endpoints at the centre of the
       contact area, so a seam is ~a voxel wide. This is bounded because the join
       adds a *straight edge between the nearest vertex pair* — with an unbounded
       radius it will happily bridge a genuine segmentation split with hundreds of
       nm of cable that no biology produced.
    2. postprocess's own join, with ``restrict_by_radius=True``: connects two
       pieces only where the gap is smaller than the sum of their local radii —
       i.e. their cross-sections nearly touch.

    Components too far apart to join are **kept, disconnected**, not discarded —
    the only thing that deletes a component is the dust threshold.

    Pass a dict as ``stats`` to receive per-step accounting (see
    :func:`fusion_stats_summary` for what the fields mean).
    """
    import kimimaro
    from kimimaro.post import remove_dust, remove_loops, remove_ticks
    from osteoid import Skeleton

    rec = stats if stats is not None else {}
    frags = [f for f in fragments if f is not None and len(f.vertices)]
    rec["n_fragments"] = len(frags)
    if not frags:
        rec["deleted"] = False
        return None

    merged = normalize_dtypes(Skeleton.simple_merge(frags).consolidate())
    before = _profile(merged)
    rec["comps_in"] = before["comps"]
    rec["cable_in_nm"] = before["cable"]

    # 1. seam join (bounded)
    radius = join_radius_nm(cfg)
    skel = kimimaro.join_close_components(frags, radius=radius) if radius > 0 else merged
    skel = normalize_dtypes(skel.consolidate())      # the join concatenates int64 edges
    after_seam = _profile(skel)
    rec["cable_joined_nm"] = after_seam["cable"]
    rec["seam_join_comps_merged"] = before["comps"] - after_seam["comps"]
    rec["seam_join_cable_added_nm"] = after_seam["cable"] - before["cable"]

    # 2. dust — the ONLY step that deletes whole components
    skel = remove_dust(skel, cfg.postprocess_dust_nm)
    after_dust = _profile(skel)
    rec["dust_comps_dropped"] = after_seam["comps"] - after_dust["comps"]
    rec["dust_cable_dropped_nm"] = after_seam["cable"] - after_dust["cable"]

    # 3. loops (collapsed / broken arbitrarily)
    skel = remove_loops(skel)
    after_loops = _profile(skel)
    rec["loop_cable_delta_nm"] = after_loops["cable"] - after_dust["cable"]

    # 4. radius-restricted join (always on, inside postprocess)
    skel = normalize_dtypes(kimimaro.join_close_components(skel, restrict_by_radius=True))
    after_rjoin = _profile(skel)
    rec["radius_join_comps_merged"] = after_loops["comps"] - after_rjoin["comps"]
    rec["radius_join_cable_added_nm"] = after_rjoin["cable"] - after_loops["cable"]

    # 5. ticks — short side branches off the main arbor. normalize_dtypes above is
    #    load-bearing: remove_ticks hits compiled code that rejects int64/float64.
    skel = remove_ticks(skel, cfg.postprocess_tick_nm)
    skel = normalize_dtypes(skel.consolidate())
    out = _profile(skel)
    rec["tick_branches_removed"] = max(0, after_rjoin["tips"] - out["tips"])
    rec["tick_cable_removed_nm"] = after_rjoin["cable"] - out["cable"]

    rec.update({"comps_out": out["comps"], "cable_out_nm": out["cable"],
                "tips_out": out["tips"], "branches_out": out["branches"]})
    rec["deleted"] = out["verts"] == 0

    if out["verts"] == 0:
        return None
    # join_close_components rebuilds from components, so the segid does not survive.
    skel.id = int(body_id) if body_id is not None else getattr(frags[0], "id", None)
    return skel


# Fields worth totalling across a run; the rest are per-body shape descriptors.
_SUMMABLE = ["n_fragments", "comps_in", "cable_in_nm", "cable_joined_nm",
             "seam_join_comps_merged", "seam_join_cable_added_nm",
             "dust_comps_dropped", "dust_cable_dropped_nm",
             "loop_cable_delta_nm", "radius_join_comps_merged", "radius_join_cable_added_nm",
             "tick_branches_removed", "tick_cable_removed_nm", "comps_out", "cable_out_nm"]


def fusion_stats_summary(per_body: Sequence[dict]) -> dict:
    """Aggregate per-body fusion stats into run totals.

    Answers "how much did postprocess actually throw away?":

    - ``dust_comps_dropped`` / ``dust_cable_dropped_nm`` — components deleted for
      being shorter than ``postprocess_dust_nm``.
    - ``bodies_deleted`` — bodies that dust consumed *entirely*.
    - ``tick_branches_removed`` / ``tick_cable_removed_nm`` — side branches pruned
      by ``postprocess_tick_nm``.
    - ``*_join_*`` — how much cable the two joins *added*, i.e. edges that are
      inferred rather than measured.
    - ``dropped_cable_fraction`` — dust + ticks as a share of the cable present
      *after* joining (the cable those steps actually saw). The number to watch
      when choosing thresholds.
    """
    out: dict = {"n_bodies": len(per_body)}
    for field in _SUMMABLE:
        out[field] = sum(s.get(field) or 0 for s in per_body)
    out["bodies_deleted"] = sum(1 for s in per_body if s.get("deleted"))
    out["bodies_multi_component"] = sum(1 for s in per_body if (s.get("comps_out") or 0) > 1)
    dropped = out["dust_cable_dropped_nm"] + out["tick_cable_removed_nm"]
    out["dropped_cable_nm"] = dropped
    out["dropped_cable_fraction"] = (dropped / out["cable_joined_nm"]) if out["cable_joined_nm"] else 0.0
    inferred = out["seam_join_cable_added_nm"] + out["radius_join_cable_added_nm"]
    out["inferred_cable_nm"] = inferred
    out["inferred_cable_fraction"] = (inferred / out["cable_joined_nm"]) if out["cable_joined_nm"] else 0.0
    return out


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
