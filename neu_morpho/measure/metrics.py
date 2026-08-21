"""Per-body measurements from a skeleton. Pure: arrays in, numbers out.

Everything here takes a :class:`neu_lib.Skeleton` — the type ``readback`` +
``Skeleton.from_precomputed`` produce — so a measurement can be made of a *published*
volume without rerunning anything, and of an excluded skeleton
(``skel.exclude(inside)``) without this module knowing what a compartment is.

Three things about the numbers, all of which have bitten this project before:

**Diameter statistics are weighted by edge length, never per vertex.** An unweighted
vertex mean answers "what is the average radius of a node", and node density is a property
of the *skeletonizer* — a region that happened to be sampled densely dominates the answer.
Weighting by the cable each measurement represents answers the question actually being
asked.

**The radii are INSCRIBED radii**, the largest sphere that fits. They are exact at the
node (measured median error 0.00 voxels against the mask's own EDT) and **systematically
small wherever a cross-section is not round**, since the inscribed sphere fits the minor
axis. Voxel quantization is random and averages away over thousands of vertices; that
eccentricity bias does not. Volume goes as r^2, which doubles it. So
:func:`frustum_volume_nm3` is a lower bound whose tightness depends on how round the
neurites are, and it must never be presented as *the* volume without that said.

**Missing radii give NaN, not zero.** A skeleton with no ``radius`` attribute has an
*unknown* calibre, and zero would read as "infinitely thin" — quietly, in an average. NaN
fails every comparison instead, so it cannot be mistaken for a small number.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from neu_lib import Skeleton


def _edges(skeleton: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """``(edge_lengths_nm, edge_index)`` with zero-length edges kept.

    Zero-length edges contribute nothing to a weighted statistic and are not dropped,
    because dropping them would change the topology counts, which are a different
    question asked of the same array.
    """
    v = np.asarray(skeleton.vertices_zyx_nm, dtype=float)
    e = np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2)
    if not len(e):
        return np.zeros(0), e
    return np.linalg.norm(v[e[:, 1]] - v[e[:, 0]], axis=1), e


def cable_length_nm(skeleton: Skeleton) -> float:
    """Total edge length. The most robust metric here — a length, not a radius.

    Its real sensitivity is *postprocessing*, not measurement: a 400 nm join doubled this
    number (416 -> 858 nm) on one body, and a 500 nm tick threshold removed 17.4% of cable
    on one body set. So it is only comparable across datasets skeletonized with the **same
    settings**, which is a fact about the pipeline rather than about this function.
    """
    lengths, _ = _edges(skeleton)
    return float(lengths.sum())


def weighted_quantile(values: Any, weights: Any, q: Any) -> np.ndarray:
    """Quantiles of ``values`` weighted by ``weights``. **Step, not interpolated.**

    numpy has no weighted percentile, and the unweighted one is the wrong question here
    (see the module docstring). The step definition is the right one for *this* question:
    each edge is a length of cable that has a diameter, so the weighted median means "the
    diameter at which half the cable is thinner and half thicker" — an empirical
    distribution over cable, not point samples of a continuum.

    Interpolating instead lets a few sparse samples drag the answer a long way. Measured
    on the fixture in the tests: a thin arm finely noded (90 nm of cable at diameter 2)
    against a thick arm coarsely noded (900 nm at diameter 200) gives a median of **200**
    by this definition and **180** by linear interpolation — 91% of the cable is at 200,
    so 180 is simply wrong for what the number claims to say.

    Weights of zero are allowed and do not count.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if v.shape != w.shape:
        raise ValueError(f"values {v.shape} and weights {w.shape} differ in shape")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    keep = np.isfinite(v) & (w > 0)
    if not keep.any():
        return np.full(len(q), np.nan)
    v, w = v[keep], w[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    idx = np.clip(np.searchsorted(cum, q * cum[-1], side="left"), 0, len(v) - 1)
    return v[idx]


def diameter_stats(skeleton: Skeleton) -> dict:
    """Edge-length-weighted diameter statistics, in nm.

    Each edge contributes the mean of its two endpoint radii, weighted by its length.
    Returns NaN throughout when the skeleton carries no radii — see the module docstring
    on why that is not zero.

    **p10-p90 describes the twigs, not the trunk**, and that is a property of neurons
    rather than of this function: the thick part of a cell is always a small *length*
    fraction, so it sits above p90. Measured on a synthetic neuron whose trunk is 600 nm
    across but only 5.6% of the cable: p10/p50/p90 came out 118/145/177 — entirely inside
    the twig population — and the trunk did not appear until **p99** (680). So read p10-p90
    as the spread of the neurite, `p99`/`max` as the thick end, and do not expect p90 to
    move when a soma is excluded.

    Percentiles are **steps** (see :func:`weighted_quantile`), so on a body where one
    edge dominates the length they legitimately collapse onto that edge's diameter.
    """
    lengths, e = _edges(skeleton)
    keys = ("diameter_nm_mean", "diameter_nm_p10", "diameter_nm_median",
            "diameter_nm_p90", "diameter_nm_p99", "diameter_nm_max")
    nan = {k: np.nan for k in keys}
    if skeleton.radii_nm is None or not len(e):
        return nan

    r = np.asarray(skeleton.radii_nm, dtype=float)
    edge_diameter = r[e[:, 0]] + r[e[:, 1]]          # (r0 + r1)/2 * 2 == r0 + r1
    keep = lengths > 0
    if not keep.any():
        return nan
    d, w = edge_diameter[keep], lengths[keep]
    p10, median, p90, p99 = weighted_quantile(d, w, [0.1, 0.5, 0.9, 0.99])
    return {
        "diameter_nm_mean": float(np.average(d, weights=w)),
        "diameter_nm_p10": float(p10),
        "diameter_nm_median": float(median),
        "diameter_nm_p90": float(p90),
        "diameter_nm_p99": float(p99),
        "diameter_nm_max": float(d.max()),
    }


def frustum_volume_nm3(skeleton: Skeleton) -> float:
    """Volume of the tube the radii imply: one truncated cone per edge.

    ``V = (pi/3) * L * (r0^2 + r0*r1 + r1^2)`` per edge, summed. Overlapping cones at a
    junction are **double-counted**, which inflates the number where branches are dense
    and is the opposite sign to the inscribed-radius bias — so do not read the two as
    cancelling. NaN when there are no radii.
    """
    lengths, e = _edges(skeleton)
    if skeleton.radii_nm is None or not len(e):
        return float("nan")
    r = np.asarray(skeleton.radii_nm, dtype=float)
    r0, r1 = r[e[:, 0]], r[e[:, 1]]
    return float((np.pi / 3.0 * lengths * (r0 * r0 + r0 * r1 + r1 * r1)).sum())


def frustum_area_nm2(skeleton: Skeleton) -> float:
    """Lateral surface area of that same tube: ``pi * (r0 + r1) * slant`` per edge.

    End caps are excluded — a neurite's ends are cuts through a continuing process, not
    real surface. NaN when there are no radii.
    """
    lengths, e = _edges(skeleton)
    if skeleton.radii_nm is None or not len(e):
        return float("nan")
    r = np.asarray(skeleton.radii_nm, dtype=float)
    r0, r1 = r[e[:, 0]], r[e[:, 1]]
    slant = np.sqrt(lengths * lengths + (r1 - r0) ** 2)
    return float((np.pi * (r0 + r1) * slant).sum())


def topology(skeleton: Skeleton) -> dict:
    """Tip, branch-point and connected-component counts from the edge list.

    Components are counted because **bodies in these datasets are genuinely fragmented**
    (verified against the raw array, not a pipeline artefact), so a count above one is
    correct reporting rather than a defect — and a *finer* scale finds more, not fewer.
    """
    _, e = _edges(skeleton)
    n = len(np.asarray(skeleton.vertices_zyx_nm, dtype=float))
    if not len(e):
        return {"n_vertices": n, "n_edges": 0, "n_tips": 0, "n_branches": 0,
                "n_components": n}

    degree = np.bincount(e.ravel(), minlength=n)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    w = np.ones(len(e))
    graph = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))
    n_components, _ = connected_components(graph, directed=False)
    return {
        "n_vertices": n,
        "n_edges": int(len(e)),
        "n_tips": int((degree == 1).sum()),
        "n_branches": int((degree > 2).sum()),
        "n_components": int(n_components),
    }


def measure_skeleton(skeleton: Skeleton, *, variant: str = "all",
                     body_id: Optional[int] = None,
                     dataset: Optional[str] = None) -> dict:
    """Every skeleton-derived measurement for one body, as one flat row.

    ``variant`` labels *what was measured* — ``"all"``, ``"minus_nucleus"``,
    ``"minus_soma_nucleus"``, ``"neurite_positive"`` — and is carried through to the
    output table rather than implied by the file it lands in. The two cohorts in the
    driving comparison are **not measured the same way** (one excludes compartments by
    mask, the other selects anucleate cells), so a row that does not say which is a row
    that cannot be compared safely.
    """
    row: dict[str, Any] = {
        "dataset": dataset,
        "body_id": None if body_id is None else int(body_id),
        "variant": str(variant),
        "cable_length_nm": cable_length_nm(skeleton),
        "volume_nm3_frustum": frustum_volume_nm3(skeleton),
        "area_nm2_frustum": frustum_area_nm2(skeleton),
    }
    row.update(diameter_stats(skeleton))
    row.update(topology(skeleton))
    return row
