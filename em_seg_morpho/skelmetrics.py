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
