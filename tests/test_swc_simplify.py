"""Tests for the radius-aware node reduction passes.

The properties that matter are structural: these passes must not disconnect a
skeleton, invent cycles, lose branch points, or move nodes outside the object.
Node count going down is the easy part.
"""

from __future__ import annotations

import numpy as np
import pytest

from em_seg_morpho import swc_simplify


def _chain(n=40, r=2.0, spacing=1.0):
    v = np.zeros((n, 3), dtype=float)
    v[:, 0] = np.arange(n) * spacing
    e = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return v, np.full(n, r), e


def _y(arm=15, r=2.0):
    """Stem along z, splitting into two arms — a branch point to preserve."""
    stem = np.stack([np.arange(arm), np.zeros(arm), np.zeros(arm)], axis=1).astype(float)
    a1 = np.stack([np.arange(arm) + arm, np.arange(arm) + 1.0, np.zeros(arm)], axis=1)
    a2 = np.stack([np.arange(arm) + arm, -np.arange(arm) - 1.0, np.zeros(arm)], axis=1)
    v = np.vstack([stem, a1, a2])
    e = [(i, i + 1) for i in range(arm - 1)]
    e += [(arm - 1, arm)] + [(arm + i, arm + i + 1) for i in range(arm - 1)]
    e += [(arm - 1, 2 * arm)] + [(2 * arm + i, 2 * arm + i + 1) for i in range(arm - 1)]
    return v, np.full(len(v), r), np.array(e, dtype=int)


def _n_components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})


# --------------------------------------------------------------------------- #
# region_sample
# --------------------------------------------------------------------------- #
def test_region_sample_thins_a_uniform_chain():
    v, r, e = _chain(n=40, r=2.0, spacing=1.0)
    v2, r2, e2 = swc_simplify.region_sample(v, r, e)
    assert len(v2) < len(v)
    # nodes 1 apart inside balls of radius 2 -> survivors roughly 2 apart
    d = np.linalg.norm(np.diff(v2[np.argsort(v2[:, 0])], axis=0), axis=1)
    assert d.min() >= 1.5


def test_region_sample_spacing_follows_radius():
    """The point of the pass: sparse where thick, dense where thin."""
    n = 60
    v = np.zeros((n, 3))
    v[:, 0] = np.arange(n)
    r = np.where(np.arange(n) < 30, 1.0, 6.0)      # thin half, thick half
    e = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    v2, r2, _ = swc_simplify.region_sample(v, r, e)
    thin = np.sort(v2[r2 < 3][:, 0])
    thick = np.sort(v2[r2 >= 3][:, 0])
    assert len(thin) > len(thick), "thick region should be sampled more sparsely"


def test_region_sample_keeps_it_connected():
    v, r, e = _y()
    v2, r2, e2 = swc_simplify.region_sample(v, r, e)
    assert _n_components(len(v2), e2) == 1
    assert len(e2) == len(v2) - 1, "a contracted tree must stay a tree (no cycles)"


def test_region_sample_preserves_the_branch_point():
    v, r, e = _y(arm=20, r=1.5)
    v2, r2, e2 = swc_simplify.region_sample(v, r, e)
    deg = np.bincount(e2.ravel(), minlength=len(v2))
    assert (deg >= 3).any(), "branch point lost"


def test_region_sample_keeps_nodes_from_the_original():
    """It selects, never interpolates — every survivor is an input node."""
    v, r, e = _y()
    v2, _, _ = swc_simplify.region_sample(v, r, e)
    orig = {tuple(p) for p in v}
    assert all(tuple(p) in orig for p in v2)


def test_region_sample_empty_input():
    v, r, e = swc_simplify.region_sample(np.zeros((0, 3)), np.zeros(0),
                                         np.zeros((0, 2), dtype=int))
    assert len(v) == 0 and len(e) == 0


# --------------------------------------------------------------------------- #
# optimal_downsample
# --------------------------------------------------------------------------- #
def test_downsample_collapses_a_straight_uniform_chain():
    """Interpolating parent<->child reproduces every interior node exactly."""
    v, r, e = _chain(n=30, r=2.0, spacing=1.0)
    v2, r2, e2 = swc_simplify.optimal_downsample(v, r, e)
    assert len(v2) < len(v)
    assert _n_components(len(v2), e2) == 1


def test_downsample_keeps_it_connected_and_acyclic():
    v, r, e = _y()
    v2, r2, e2 = swc_simplify.optimal_downsample(v, r, e)
    assert _n_components(len(v2), e2) == 1
    assert len(e2) == len(v2) - 1


def test_downsample_is_a_fixpoint():
    """Running it twice must change nothing the second time."""
    v, r, e = _y()
    v1 = swc_simplify.optimal_downsample(v, r, e)
    v2 = swc_simplify.optimal_downsample(*v1)
    assert len(v2[0]) == len(v1[0])


def test_downsample_will_not_collapse_a_sharp_turn():
    """A right-angle bend is not reproduced by interpolation, so it must survive."""
    v = np.array([[0., 0, 0], [5, 0, 0], [10, 0, 0], [10, 5, 0], [10, 10, 0]])
    r = np.full(5, 1.0)
    e = np.array([(0, 1), (1, 2), (2, 3), (3, 4)])
    v2, r2, e2 = swc_simplify.optimal_downsample(v, r, e)
    kept = {tuple(p) for p in v2}
    assert (10., 0., 0.) in kept, "the corner was interpolated away"


def test_downsample_preserves_multiple_components():
    v1, r1, e1 = _chain(n=20)
    v2, r2, e2 = _chain(n=20)
    v2 = v2 + np.array([0.0, 500.0, 0.0])
    v = np.vstack([v1, v2])
    r = np.concatenate([r1, r2])
    e = np.vstack([e1, e2 + len(v1)])
    ov, orr, oe = swc_simplify.optimal_downsample(v, r, e)
    assert _n_components(len(ov), oe) == 2


# --------------------------------------------------------------------------- #
def test_simplify_runs_both_passes():
    v, r, e = _y(arm=25, r=2.0)
    v2, r2, e2 = swc_simplify.simplify(v, r, e)
    only_first = swc_simplify.region_sample(v, r, e)[0]
    assert len(v2) <= len(only_first) < len(v)
    assert _n_components(len(v2), e2) == 1
