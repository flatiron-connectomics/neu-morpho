"""Tests for the agreement metric — the thing we now optimise against.

It replaced fill/spill because those are confounded (fill rewards inventing
branches; spill cannot distinguish reclaiming a false split from trespassing on a
dense segmentation). So it matters that this one is actually directional: it must
tell a missing branch apart from an invented one.
"""

from __future__ import annotations

import numpy as np
import pytest

from em_seg_morpho import skelmetrics


def _line(n=40, axis=0, offset=(0, 0, 0), r=2.0):
    v = np.zeros((n, 3), dtype=float)
    v[:, axis] = np.arange(n)
    v += np.asarray(offset, dtype=float)
    e = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return v, np.full(n, r), e


def _with_extra_branch(n=40):
    """A trunk plus a side branch that the reference does not have."""
    v, r, e = _line(n)
    br = np.stack([np.full(12, 20.0), np.arange(1, 13, dtype=float),
                   np.zeros(12)], axis=1)
    off = len(v)
    v = np.vstack([v, br])
    r = np.concatenate([r, np.full(12, 1.0)])
    e = np.vstack([e, [[20, off]], np.stack([np.arange(off, off + 11),
                                             np.arange(off + 1, off + 12)], axis=1)])
    return v, r, e


def test_identical_skeletons_agree_exactly():
    v, r, e = _line()
    a = skelmetrics.agreement(v, r, e, v, r, e)
    assert a["a_to_b_median"] == pytest.approx(0.0, abs=1e-9)
    assert a["b_to_a_median"] == pytest.approx(0.0, abs=1e-9)
    assert a["node_ratio"] == pytest.approx(1.0)
    assert a["tip_ratio"] == pytest.approx(1.0)
    assert a["cable_ratio"] == pytest.approx(1.0)


def test_offset_skeleton_reports_the_offset():
    v, r, e = _line()
    v2 = v + np.array([0.0, 3.0, 0.0])
    a = skelmetrics.agreement(v2, r, e, v, r, e)
    assert a["a_to_b_median"] == pytest.approx(3.0, abs=0.2)
    assert a["b_to_a_median"] == pytest.approx(3.0, abs=0.2)


def test_node_placement_alone_does_not_count_as_disagreement():
    """Same path, different sampling -> near-zero distance.

    The metric compares densely resampled centrelines, so a skeleton with the
    same shape but sparser nodes must not be penalised as a different shape.
    """
    v, r, e = _line(n=40)
    v2, r2, e2 = _line(n=9)
    v2 = v2.copy()
    v2[:, 0] = np.linspace(0, 39, 9)                    # same path, 9 nodes
    a = skelmetrics.agreement(v2, r2, e2, v, r, e)
    assert a["a_to_b_median"] < 0.6
    assert a["b_to_a_median"] < 0.6
    assert a["node_ratio"] < 0.3                        # but the ratio still shows it


def test_invented_branch_shows_in_a_to_b_not_b_to_a():
    """An extra branch is structure B lacks: A->B rises, B->A stays small."""
    va, ra, ea = _with_extra_branch()
    vb, rb, eb = _line()
    a = skelmetrics.agreement(va, ra, ea, vb, rb, eb)
    assert a["b_to_a_p90"] < 1.0, "reference is still covered"
    assert a["a_to_b_p90"] > 3.0, "the invented branch should show up"
    assert a["tip_ratio"] > 1.0


def test_missing_branch_shows_in_b_to_a_not_a_to_b():
    """The mirror image: A missing a branch B has."""
    va, ra, ea = _line()
    vb, rb, eb = _with_extra_branch()
    a = skelmetrics.agreement(va, ra, ea, vb, rb, eb)
    assert a["a_to_b_p90"] < 1.0
    assert a["b_to_a_p90"] > 3.0, "the missing branch should show up"


def test_radius_ratio_tracks_radius():
    v, r, e = _line(r=2.0)
    a = skelmetrics.agreement(v, r * 1.5, e, v, r, e)
    assert a["radius_ratio_median"] == pytest.approx(1.5, abs=0.05)


def test_empty_input_returns_empty():
    v, r, e = _line()
    assert skelmetrics.agreement(np.zeros((0, 3)), np.zeros(0),
                                 np.zeros((0, 2), int), v, r, e) == {}
