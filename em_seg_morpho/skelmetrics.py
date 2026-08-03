"""Score a skeleton by what it looks like when you render it.

**Why this file exists.** Comparing skeletonizers by node count or cable length
ranks them on the wrong axis, and comparing reported radius against the distance
transform is circular for kimimaro (it sets ``radii = DBF[vertex]``, so it scores
zero error by construction). The question that actually matters here is: *render
the SWC as a tapered tube — how much of the segment does it fill, and how much
does it bulge outside?*

**Fill, not inscribe.** These segmentations are imperfect, so the mask is not
ground truth and agreement with its distance transform is not the target. The
working definition of a "good" radius in this package is the one that makes the
rendered tube fill the segment. That is a pragmatic definition, not a rigorous
one — it has no calibrated notion of correctness behind it, and it would be the
wrong definition if these radii were ever reused for volume or surface-area
measurement. Say which one you mean before optimising against either.

**The bug this module was written to avoid.** The first version of this metric
stamped isolated spheres at each vertex. Real SWC viewers draw connected frusta
between parent and child, so sphere-stamping silently penalised any method that
places fewer nodes — and it reversed the ranking of the two tools under test.
:func:`sweep` exists specifically so that never happens again: always pass
``edges``.

Neither tool reaches 100% coverage and none can: a chain of circular
cross-sections cannot fill an irregular one. Read these numbers against each
other, never against perfection.
"""

from __future__ import annotations

import numpy as np

_BALL_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _ball(r: int):
    """Offsets of a discrete ball of integer radius ``r``, cached per radius."""
    if r not in _BALL_CACHE:
        if r <= 0:
            z = np.zeros(1, dtype=np.int64)
            _BALL_CACHE[r] = (z, z.copy(), z.copy())
        else:
            g = np.arange(-r, r + 1)
            dz, dy, dx = np.meshgrid(g, g, g, indexing="ij")
            sel = (dz ** 2 + dy ** 2 + dx ** 2) <= r * r
            _BALL_CACHE[r] = (dz[sel].astype(np.int64),
                              dy[sel].astype(np.int64),
                              dx[sel].astype(np.int64))
    return _BALL_CACHE[r]


def sweep(zyx: np.ndarray, radii: np.ndarray, edges: np.ndarray | None,
          step: float = 0.5):
    """Sample points along every edge with linearly interpolated radius.

    This is the tapered-frustum model a renderer uses. With ``edges=None`` you get
    vertices only, which is the sphere-stamping mistake described in the module
    docstring — don't, unless you specifically want that comparison.
    """
    zyx = np.asarray(zyx, float)
    radii = np.asarray(radii, float)
    if edges is None or len(edges) == 0:
        return zyx, radii
    a, b = edges[:, 0], edges[:, 1]
    pa, pb, ra, rb = zyx[a], zyx[b], radii[a], radii[b]
    n = np.maximum(1, np.ceil(np.linalg.norm(pb - pa, axis=1) / step)).astype(int)
    reps = n + 1
    t = np.concatenate([np.linspace(0, 1, k) for k in reps])
    idx = np.repeat(np.arange(len(a)), reps)
    pts = pa[idx] + (pb[idx] - pa[idx]) * t[:, None]
    rr = ra[idx] + (rb[idx] - ra[idx]) * t
    return np.vstack([pts, zyx]), np.concatenate([rr, radii])


def rasterize(shape, zyx, radii, edges=None, step: float = 0.5) -> np.ndarray:
    """Boolean volume of the rendered tube.

    Groups samples by rounded radius so each ball offset-table is applied once as a
    single vectorised scatter, rather than looping per node.
    """
    pts, rr = sweep(zyx, radii, edges, step)
    pts = np.round(pts).astype(np.int64)
    key = np.maximum(0, np.round(rr).astype(int))
    canvas = np.zeros(tuple(shape), dtype=bool)
    for k in np.unique(key):
        sel = key == k
        dz, dy, dx = _ball(int(k))
        zz = (pts[sel, 0][:, None] + dz[None, :]).ravel()
        yy = (pts[sel, 1][:, None] + dy[None, :]).ravel()
        xx = (pts[sel, 2][:, None] + dx[None, :]).ravel()
        ok = ((zz >= 0) & (zz < shape[0]) & (yy >= 0) & (yy < shape[1]) &
              (xx >= 0) & (xx < shape[2]))
        canvas[zz[ok], yy[ok], xx[ok]] = True
    return canvas


def score(mask_zyx: np.ndarray, zyx, radii, edges=None, step: float = 0.5) -> dict:
    """Fill quality of a skeleton against the segment it came from.

    ``coverage``  fraction of segment voxels inside the rendered tube (higher better)
    ``spill``     fraction of tube volume outside the segment (lower better)
    ``nodes_per_1k_covered``  node economy — how many nodes buy 1000 covered voxels
    """
    mask = np.asarray(mask_zyx).astype(bool)
    tube = rasterize(mask.shape, zyx, radii, edges, step)
    inter = int((tube & mask).sum())
    n_tube, n_mask = int(tube.sum()), int(mask.sum())
    return {
        "coverage": inter / max(1, n_mask),
        "spill": 1.0 - inter / max(1, n_tube),
        "nodes": int(len(radii)),
        "tube_voxels": n_tube,
        "mask_voxels": n_mask,
        "nodes_per_1k_covered": 1000.0 * len(radii) / max(1, inter),
    }


def agreement(zyx_a, radii_a, edges_a, zyx_b, radii_b, edges_b,
              step: float = 0.5) -> dict:
    """How closely skeleton A reproduces reference skeleton B.

    **This is the metric to optimise against, not** :func:`score`. Coverage is
    confounded by branch count — more branches fill more, so a skeletonizer that
    invents spurious neurites scores *better*. Measured on the benchmark, the
    Python port beat NeuTu on coverage while carrying 5–10× its tip count and
    being 1.4–3.5× *less* coverage-efficient per unit cable. Optimising coverage
    optimises for the defect.

    There is no ground truth to appeal to instead: these segmentations are dense
    (every voxel belongs to some segment) and many segments are pieces of one
    neuron incorrectly split, so "what should this body's skeleton cover" is not
    answerable per segment. NeuTu is the reference because it is known to behave
    well enough, not because it is right.

    ``a_to_b`` / ``b_to_a`` are distances between densely resampled centrelines,
    so they measure the *paths*, not the node positions — a skeleton with the
    same shape but different node placement scores near zero. ``b_to_a`` is the
    one that catches missing branches; ``a_to_b`` catches invented ones.
    """
    from scipy.spatial import cKDTree

    pa, ra = sweep(zyx_a, radii_a, edges_a, step)
    pb, rb = sweep(zyx_b, radii_b, edges_b, step)
    if not len(pa) or not len(pb):
        return {}
    ta, tb = cKDTree(pa), cKDTree(pb)
    d_ab, _ = tb.query(pa)               # each point of A -> nearest on B
    d_ba, ia = ta.query(pb)              # each point of B -> nearest on A

    def tips(v, e):
        e = np.asarray(e, int).reshape(-1, 2)
        if not len(e):
            return 0
        return int((np.bincount(e.ravel(), minlength=len(v)) == 1).sum())

    def cable(v, e):
        v, e = np.asarray(v, float), np.asarray(e, int).reshape(-1, 2)
        return float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).sum()) if len(e) else 0.0

    ca, cb = cable(zyx_a, edges_a), cable(zyx_b, edges_b)
    return {
        "a_to_b_median": float(np.median(d_ab)),
        "a_to_b_p90": float(np.percentile(d_ab, 90)),
        "b_to_a_median": float(np.median(d_ba)),
        "b_to_a_p90": float(np.percentile(d_ba, 90)),
        "node_ratio": len(np.asarray(zyx_a)) / max(1, len(np.asarray(zyx_b))),
        "tip_ratio": tips(zyx_a, edges_a) / max(1, tips(zyx_b, edges_b)),
        "cable_ratio": ca / max(1e-9, cb),
        # radius of A at the point of A nearest each B sample, vs B's own radius
        "radius_ratio_median": float(np.median(ra[ia] / np.maximum(1e-9, rb))),
    }


def spill_by_neighbour_size(mask_zyx, nbr_size_class, zyx, radii, edges=None,
                            small_bin_max: int = 3, step: float = 0.5) -> dict:
    """Where the tube spills to, graded by how large the neighbour it enters is.

    **Diagnostic, not an objective.** There is no ground truth here, so nothing
    in this function establishes that a given spill is wrong; optimise against
    :func:`agreement` instead. It exists because plain "spill" is uninformative
    on a dense segmentation: every voxel belongs to some segment, so a tube that
    leaves the body necessarily enters a neighbour, and 0.0% of spill lands on
    background.

    The graded version carries some signal, on one assumption worth stating
    because it is not verified: **most false splits are small fragments**. Under
    it, spill into a small fragment is likely reclaiming the same neuron, while
    spill into a large, morphologically complete neighbour is more likely a real
    error. Neither is certain — a small fragment can be a genuinely separate
    structure, and a large neighbour can be the same neuron badly split.

    ``nbr_size_class`` is the uint8 map from ``export_benchmark_masks.py``: 0 for
    this body or background, higher bins for larger neighbours.
    ``small_bin_max`` is the highest bin still counted as a fragment; sweep it,
    since the boundary is a judgement call rather than a measurement.
    """
    mask = np.asarray(mask_zyx).astype(bool)
    cls = np.asarray(nbr_size_class)
    tube = rasterize(mask.shape, zyx, radii, edges, step)
    inter = int((tube & mask).sum())
    n_tube = max(1, int(tube.sum()))
    out = tube & ~mask
    small = int((out & (cls > 0) & (cls <= small_bin_max)).sum())
    large = int((out & (cls > small_bin_max)).sum())
    return {
        "coverage": inter / max(1, int(mask.sum())),
        "spill_into_fragments": small / n_tube,
        "spill_into_large_neighbours": large / n_tube,
        "spill_into_background": (n_tube - inter - small - large) / n_tube,
        "nodes": int(len(radii)),
    }


def radius_vs_edt(mask_zyx: np.ndarray, zyx, radii) -> dict:
    """Diagnostic: reported radius against the mask's own distance transform.

    Deliberately **not** the headline metric — kimimaro satisfies it by
    construction and the mask is not ground truth. Useful only to detect a tool
    reporting a radius larger than anything that fits in the segment at all, which
    is a real defect regardless of your definition of correctness.
    """
    import edt

    mask = np.asarray(mask_zyx).astype(bool)
    e = edt.edt(mask.astype(np.uint8), anisotropy=(1, 1, 1), black_border=False)
    z, y, x = [np.clip(np.round(zyx[:, k]).astype(int), 0, mask.shape[k] - 1)
               for k in range(3)]
    true_r = e[z, y, x]
    d = np.asarray(radii, float) - true_r
    return {
        "median_error": float(np.median(d)),
        "mean_error": float(d.mean()),
        "frac_over_2vox": float(np.mean(d > 2.0)),
        "reported_max": float(np.max(radii)),
        "mask_max_inscribed": float(e.max()),
    }
