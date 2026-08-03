"""Tests for the NeuTu-style tracer.

The load-bearing one is :func:`test_per_voxel_weights_match_neutu_edge_cost` —
the substitution the plan flags as assumed-but-unverified.
``dijkstra3d.parental_field`` takes **per-voxel** weights, whereas NeuTu's cost is
the symmetric **edge** form ``d·[f(v₁) + f(v₂)]``. For a path with uniform step
length these telescope to the same argmin up to constant endpoint terms, but 3D
diagonal moves have mixed step lengths (1, √2, √3), so the equivalence is not
free and is checked against an independent reference implementation.
"""

from __future__ import annotations

import heapq

import numpy as np
import pytest

from em_seg_morpho import neutu_trace


def _tube(length=40, r=3, pad=4, bend=False):
    """A cylinder along z, optionally bent, as a Fortran-order uint8 mask."""
    s = (length + 2 * pad, 2 * (r + pad) + (length // 2 if bend else 0), 2 * (r + pad))
    zz, yy, xx = np.indices(s)
    cy, cx = r + pad, r + pad
    off = np.clip(zz - pad, 0, None) // 2 if bend else 0
    m = ((zz >= pad) & (zz < pad + length) &
         ((yy - cy - off) ** 2 + (xx - cx) ** 2 <= r * r))
    return np.asfortranarray(m.astype(np.uint8))


def _weights(mask):
    import edt
    import kimimaro.skeletontricks

    dbf = edt.edt(mask, anisotropy=(1, 1, 1), black_border=False, order="F")
    return neutu_trace.neutu_pdrf(kimimaro.skeletontricks.zero2inf(dbf))


def test_neutu_pdrf_values():
    dbf = np.asfortranarray(np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32))
    p = neutu_trace.neutu_pdrf(dbf)
    assert p[0, 0, 0] == pytest.approx(0.5)          # 1/(1+1)
    assert p[0, 0, 1] == pytest.approx(0.2)          # 1/(1+4)
    assert p[0, 0, 2] == pytest.approx(0.1)          # 1/(1+9)
    assert p.dtype == np.float32


def test_neutu_pdrf_forbids_background():
    """Background must be inf. 1/(1+inf²) is 0 — the *cheapest* weight there is,
    which would make empty space free to cross."""
    dbf = np.asfortranarray(np.array([[[1.0, np.inf]]], dtype=np.float32))
    p = neutu_trace.neutu_pdrf(dbf)
    assert np.isinf(p[0, 0, 1])
    assert p[0, 0, 0] < p[0, 0, 1]


def _neutu_edge_cost_path(weights, src, dst):
    """Reference Dijkstra using NeuTu's symmetric edge cost ``d·[f(u)+f(v)]``.

    Deliberately plain, slow and obviously correct — it exists to check the fast
    per-voxel path, so sharing any code with it would defeat the purpose.
    26-connected, matching dijkstra3d.
    """
    shape = weights.shape
    offs = [(dz, dy, dx)
            for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)]
    step = {o: float(np.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)) for o in offs}

    best = {src: 0.0}
    prev: dict = {src: None}
    pq = [(0.0, src)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        fu = weights[u]
        for o in offs:
            v = (u[0] + o[0], u[1] + o[1], u[2] + o[2])
            if not all(0 <= v[k] < shape[k] for k in range(3)):
                continue
            fv = weights[v]
            if not np.isfinite(fv):             # background is forbidden
                continue
            nd = d + step[o] * (fu + fv)
            if nd < best.get(v, float("inf")) - 1e-12:
                best[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in prev:
        return None
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return np.array(path[::-1], dtype=int)


@pytest.mark.parametrize("bend", [False, True])
def test_per_voxel_weights_match_neutu_edge_cost(bend):
    """The substitution the plan says to confirm rather than assume.

    Not asserting identical paths a priori: with mixed 3D step lengths the two
    costs are not exactly proportional, so ties could break differently. What
    must hold is that the per-voxel route is no worse *under NeuTu's own cost*
    than NeuTu's optimum. (In practice both tubes come out bit-identical.)
    """
    import dijkstra3d

    m = _tube(length=30, r=3, bend=bend)
    w = _weights(m)
    inside = np.argwhere(m > 0)
    src = tuple(int(v) for v in inside[0])
    dst = tuple(int(v) for v in inside[-1])

    ours = np.asarray(dijkstra3d.path_from_parents(dijkstra3d.parental_field(w, src), dst))
    ref = _neutu_edge_cost_path(w, src, dst)
    assert ref is not None and len(ours) > 0

    def neutu_cost(path):
        d = np.linalg.norm(np.diff(path, axis=0), axis=1)
        f = w[path[:, 0], path[:, 1], path[:, 2]]
        return float(np.sum(d * (f[:-1] + f[1:])))

    c_ours, c_ref = neutu_cost(ours), neutu_cost(ref)
    assert c_ref <= c_ours + 1e-9                      # ref is the true optimum
    assert c_ours <= c_ref * 1.02, (
        f"per-voxel weighting chose a materially worse path under NeuTu's own "
        f"cost: {c_ours:.6g} vs optimum {c_ref:.6g}")


def test_skeleton_stays_inside_the_tube():
    m = _tube(length=40, r=3)
    skel = neutu_trace.skeletonize(m)
    v = np.asarray(skel.vertices).astype(int)
    assert len(v) > 0
    assert v[:, 0].max() - v[:, 0].min() >= 35              # spans the tube
    assert m[v[:, 0], v[:, 1], v[:, 2]].all()               # never leaves it


def test_radii_are_inscribed_not_inflated():
    m = _tube(length=40, r=3)
    skel = neutu_trace.skeletonize(m)
    assert np.asarray(skel.radii).max() <= 3.5 + 1e-6


def test_empty_mask_returns_empty_skeleton():
    skel = neutu_trace.skeletonize(np.zeros((8, 8, 8), dtype=np.uint8))
    assert len(np.asarray(skel.vertices)) == 0


def test_tighter_invalidation_yields_more_vertices():
    """A smaller invalidation ball leaves more to cover, so more paths."""
    m = _tube(length=40, r=3, bend=True)
    tight = neutu_trace.skeletonize(m, scale=1.0, const=2.0)
    looser = neutu_trace.skeletonize(m, scale=2.0, const=8.0)
    assert len(np.asarray(tight.vertices)) >= len(np.asarray(looser.vertices))


def test_all_connected_components_are_traced():
    """A single-root trace covers exactly one component and silently drops the rest.

    TEASAR grows from one root, and both the parent field and the rolling-ball
    invalidation are confined to that root's component. The root comes from
    ``first_label`` — whichever voxel is first in memory order — so on a
    fragmented body the trace can land on a speck. This reproduces that in
    miniature: a small tube placed *first* in memory order, a large one after.
    Measured on real body 6308993 (7 components, largest 96.9% of voxels) a
    single-root trace covered 3%.
    """
    m = np.zeros((60, 40, 20), dtype=np.uint8)
    m[2:8, 3:9, 3:9] = 1                                    # small, first in z
    zz, yy, xx = np.indices(m.shape)
    m[((zz >= 20) & (zz < 55) & ((yy - 25) ** 2 + (xx - 10) ** 2 <= 16))] = 1
    m = np.asfortranarray(m)

    skel = neutu_trace.skeletonize(m)
    v = np.asarray(skel.vertices).astype(int)
    assert len(v) > 0
    in_small = ((v[:, 0] < 15)).sum()
    in_large = ((v[:, 0] >= 15)).sum()
    assert in_small > 0, "the small leading component was not traced"
    assert in_large > 0, "the large component was not traced (single-root bug)"


def test_component_crop_does_not_inflate_radii():
    """Cropping to a component's bbox must pad with background.

    With the crop flush against the component, ``black_border=False`` reads the
    out-of-bounds side as non-background and the EDT at that face is too large.
    """
    m = np.zeros((30, 20, 20), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[((zz >= 0) & (zz < 30) & ((yy - 10) ** 2 + (xx - 10) ** 2 <= 9))] = 1
    skel = neutu_trace.skeletonize(np.asfortranarray(m))
    assert np.asarray(skel.radii).max() <= 3.5 + 1e-6


def test_min_length_rejects_short_spurs_but_keeps_the_trunk():
    """NeuTu's `minimalLength`: reject a branch by the NEW territory it covers.

    A long tube with a 2-voxel bump. The bump is a branch whose un-invalidated
    geodesic tail is ~nothing, so it must not become cable; the trunk must
    survive untouched.
    """
    m = _tube(length=50, r=3, pad=5)
    m = np.asfortranarray(m.copy())
    m[28:30, 12:14, 4] = 1                                  # a tiny nub

    strict = neutu_trace.skeletonize(m, min_length=10.0)
    loose = neutu_trace.skeletonize(m, min_length=0.0)
    assert len(np.asarray(strict.vertices)) <= len(np.asarray(loose.vertices))
    v = np.asarray(strict.vertices)
    assert v[:, 0].max() - v[:, 0].min() >= 40, "the trunk was truncated too"


def test_min_length_is_measured_on_uninvalidated_length_not_geometry():
    """The distinction that makes the criterion work.

    ``_uninvalidated_length`` must measure only the trailing run of still-valid
    voxels, so a long path that mostly retreads covered ground scores low.
    """
    path = np.array([[i, 0, 0] for i in range(10)])
    labels = np.ones((12, 2, 2), dtype=np.uint8)
    assert neutu_trace._uninvalidated_length(path, labels) == pytest.approx(9.0)

    labels_covered = labels.copy()
    labels_covered[:8] = 0                      # only the last 2 are new
    assert neutu_trace._uninvalidated_length(path, labels_covered) == pytest.approx(1.0)

    labels_all = np.zeros((12, 2, 2), dtype=np.uint8)
    assert neutu_trace._uninvalidated_length(path, labels_all) == 0.0


def test_branches_are_traced():
    """A Y should produce a branch point (a degree-3 vertex), not one path."""
    s = (60, 60, 24)
    zz, yy, xx = np.indices(s)
    stem = (zz >= 6) & (zz < 30) & ((yy - 30) ** 2 + (xx - 12) ** 2 <= 9)
    t = np.clip(zz - 30, 0, None)
    arm1 = (zz >= 30) & (zz < 54) & ((yy - 30 - t) ** 2 + (xx - 12) ** 2 <= 9)
    arm2 = (zz >= 30) & (zz < 54) & ((yy - 30 + t) ** 2 + (xx - 12) ** 2 <= 9)
    m = np.asfortranarray((stem | arm1 | arm2).astype(np.uint8))

    skel = neutu_trace.skeletonize(m)
    e = np.asarray(skel.edges)
    degree = np.bincount(e.ravel(), minlength=len(np.asarray(skel.vertices)))
    assert (degree >= 3).any(), "no branch point found on a Y-shaped body"
