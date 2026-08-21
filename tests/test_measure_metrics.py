"""Per-body skeleton measurements.

The properties worth asserting are the ones that would otherwise be wrong *quietly*:
diameter weighted by cable rather than by node, missing radii coming back as NaN rather
than zero, and every number surviving an `exclude()` that cuts the skeleton in half.
"""

import numpy as np
import pytest
from neu_lib import BBox, Skeleton, box_predicate

from neu_morpho.measure import (cable_length_nm, diameter_stats, frustum_area_nm2,
                                frustum_volume_nm3, measure_skeleton, topology,
                                weighted_quantile)


def _rod(n=5, spacing=100.0, radius=10.0):
    """A straight rod along z, uniformly noded."""
    v = np.zeros((n, 3))
    v[:, 0] = np.arange(n) * spacing
    e = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return Skeleton(v, e, radii_nm=np.full(n, radius), name="rod")


# --------------------------------------------------------------------------- #
# cable length
# --------------------------------------------------------------------------- #
def test_cable_length_is_the_sum_of_edge_lengths():
    assert cable_length_nm(_rod(n=5, spacing=100.0)) == pytest.approx(400.0)


def test_cable_length_of_an_edgeless_skeleton_is_zero_not_nan():
    """A body whose skeleton is all dust has zero cable, which is a real answer."""
    bare = Skeleton(np.zeros((3, 3)), np.zeros((0, 2), dtype=np.int64))
    assert cable_length_nm(bare) == 0.0


# --------------------------------------------------------------------------- #
# diameter — the weighting is the point
# --------------------------------------------------------------------------- #
def test_diameter_is_weighted_by_cable_not_by_node_count():
    """Node density is a property of the skeletonizer, so an unweighted mean answers
    the wrong question. Here a thin arm is finely noded and a thick arm coarsely: the
    unweighted vertex mean leans thin, the length-weighted mean does not."""
    # thin arm: 10 nodes over 90 nm at r=1 ; thick arm: 2 nodes over 900 nm at r=100
    thin = np.stack([np.arange(10) * 10.0, np.zeros(10), np.zeros(10)], axis=1)
    thick = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 900.0]])
    v = np.vstack([thin, thick])
    e = np.vstack([np.stack([np.arange(9), np.arange(1, 10)], axis=1),
                   np.array([[10, 11]])])
    r = np.concatenate([np.ones(10), np.full(2, 100.0)])
    skel = Skeleton(v, e, radii_nm=r)

    stats = diameter_stats(skel)
    unweighted_diameter_mean = 2 * r.mean()               # ~34 nm, dragged down by nodes
    # 900 of the 990 nm of cable is the thick arm, so the weighted mean sits near 200
    assert stats["diameter_nm_mean"] > 150.0
    assert stats["diameter_nm_mean"] > unweighted_diameter_mean
    assert stats["diameter_nm_median"] == pytest.approx(200.0)


def test_diameter_is_twice_the_radius():
    stats = diameter_stats(_rod(radius=7.0))
    assert stats["diameter_nm_mean"] == pytest.approx(14.0)
    assert stats["diameter_nm_max"] == pytest.approx(14.0)


def test_p90_describes_the_twigs_and_p99_the_trunk():
    """A neuron's thick part is always a small LENGTH fraction, so it sits above p90.
    Reported because p90 not moving when a soma is excluded is otherwise surprising."""
    rng = np.random.default_rng(0)
    twig_d = rng.normal(144.0, 20.0, 2000)
    trunk_d = rng.normal(600.0, 60.0, 40)
    twig_w = rng.normal(300.0, 50.0, 2000)
    trunk_w = rng.normal(900.0, 100.0, 40)
    d = np.concatenate([twig_d, trunk_d])
    w = np.concatenate([twig_w, trunk_w])
    p10, p50, p90, p99 = weighted_quantile(d, w, [0.1, 0.5, 0.9, 0.99])

    assert trunk_w.sum() / w.sum() < 0.1          # the trunk is a small share of cable
    assert p90 < 250.0                            # ...so p90 is still inside the twigs
    assert p99 > 500.0                            # ...and only p99 reaches the trunk
    assert p10 < p50 < p90 < p99


def test_missing_radii_give_nan_not_zero():
    """Zero would read as 'infinitely thin' inside an average, silently. NaN fails
    every comparison instead, so it cannot pass for a small number."""
    bare = Skeleton(np.array([[0.0, 0, 0], [0.0, 0, 100.0]]), np.array([[0, 1]]))
    stats = diameter_stats(bare)
    assert all(np.isnan(v) for v in stats.values())
    assert np.isnan(frustum_volume_nm3(bare))
    assert np.isnan(frustum_area_nm2(bare))
    # and it is NOT zero
    assert not any(v == 0 for v in stats.values())


def test_a_skeleton_of_only_zero_length_edges_has_no_diameter():
    dup = Skeleton(np.zeros((2, 3)), np.array([[0, 1]]), radii_nm=np.array([5.0, 5.0]))
    assert np.isnan(diameter_stats(dup)["diameter_nm_mean"])


# --------------------------------------------------------------------------- #
# weighted_quantile
# --------------------------------------------------------------------------- #
def test_weighted_quantile_matches_the_unweighted_one_at_equal_weights():
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    got = weighted_quantile(v, np.ones_like(v), [0.5])
    assert got[0] == pytest.approx(np.median(v))


def test_the_weighted_median_is_a_step_not_an_interpolation():
    """The question is "half the cable is thinner than what", so the answer must be a
    diameter that some cable actually has. Interpolating between a sparse thin sample and
    a dominant thick one returns a value nothing measured — 180 here, when 91% of the
    cable is at 200."""
    got = weighted_quantile([2.0, 200.0], [90.0, 900.0], [0.5])
    assert got[0] == 200.0


def test_a_single_sample_is_its_own_every_quantile():
    got = weighted_quantile([42.0], [1.0], [0.0, 0.5, 1.0])
    assert got.tolist() == [42.0, 42.0, 42.0]


def test_zero_weights_do_not_count():
    got = weighted_quantile([1.0, 1000.0], [1.0, 0.0], [0.5])
    assert got[0] == pytest.approx(1.0)


def test_weighted_quantile_refuses_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_quantile([1.0, 2.0], [1.0, -1.0], [0.5])


def test_all_non_finite_values_give_nan():
    assert np.isnan(weighted_quantile([np.nan, np.nan], [1.0, 1.0], [0.5])[0])


# --------------------------------------------------------------------------- #
# frustum volume and area
# --------------------------------------------------------------------------- #
def test_a_uniform_rod_matches_the_cylinder_formula():
    skel = _rod(n=2, spacing=1000.0, radius=10.0)
    assert frustum_volume_nm3(skel) == pytest.approx(np.pi * 100.0 * 1000.0)
    assert frustum_area_nm2(skel) == pytest.approx(2 * np.pi * 10.0 * 1000.0)


def test_a_cone_matches_the_cone_formula():
    """r0=0 to r1=R over L is a cone: V = pi R^2 L / 3."""
    skel = Skeleton(np.array([[0.0, 0, 0], [1000.0, 0, 0]]), np.array([[0, 1]]),
                    radii_nm=np.array([0.0, 30.0]))
    assert frustum_volume_nm3(skel) == pytest.approx(np.pi * 900.0 * 1000.0 / 3.0)


def test_volume_scales_as_the_square_of_the_radius():
    """Which is why the inscribed-radius bias is doubled in a volume, and why this
    number is a lower bound rather than the volume."""
    a = frustum_volume_nm3(_rod(n=2, spacing=100.0, radius=10.0))
    b = frustum_volume_nm3(_rod(n=2, spacing=100.0, radius=20.0))
    assert b / a == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #
def test_a_rod_has_two_tips_and_no_branches():
    t = topology(_rod(n=5))
    assert (t["n_tips"], t["n_branches"], t["n_components"]) == (2, 0, 1)


def test_a_y_has_three_tips_and_one_branch():
    v = np.array([[0.0, 0, 0], [100.0, 0, 0], [200.0, 100.0, 0], [200.0, -100.0, 0]])
    e = np.array([[0, 1], [1, 2], [1, 3]])
    t = topology(Skeleton(v, e))
    assert (t["n_tips"], t["n_branches"]) == (3, 1)


def test_disconnected_pieces_are_counted_as_components():
    """Bodies here are genuinely fragmented, verified against the raw array — so more
    than one component is correct reporting, not a defect."""
    v = np.array([[0.0, 0, 0], [100.0, 0, 0], [5000.0, 0, 0], [5100.0, 0, 0]])
    e = np.array([[0, 1], [2, 3]])
    assert topology(Skeleton(v, e))["n_components"] == 2


def test_isolated_vertices_each_count_as_a_component():
    bare = Skeleton(np.zeros((3, 3)), np.zeros((0, 2), dtype=np.int64))
    assert topology(bare)["n_components"] == 3


# --------------------------------------------------------------------------- #
# the row, and exclusion
# --------------------------------------------------------------------------- #
def test_measure_skeleton_labels_what_was_measured():
    """The two cohorts are not measured the same way, so a row that does not say which
    variant it is cannot be compared safely."""
    row = measure_skeleton(_rod(), variant="minus_nucleus", body_id=7, dataset="male-cns")
    assert row["dataset"] == "male-cns" and row["body_id"] == 7
    assert row["variant"] == "minus_nucleus"
    assert row["cable_length_nm"] == pytest.approx(400.0)
    assert row["n_tips"] == 2


def test_excluding_a_region_reduces_cable_and_the_row_still_measures():
    """The whole point of the exclusion path: measure the same body twice, with and
    without a compartment, and get two comparable rows."""
    skel = _rod(n=11, spacing=100.0, radius=10.0)          # 0..1000 nm, 1000 nm cable
    whole = measure_skeleton(skel, variant="all")
    # drop the first 300 nm, as a compartment mask would
    kept = skel.exclude(box_predicate(BBox((-50, -50, -50), (350, 50, 50))),
                        tolerance_nm=1.0)
    part = measure_skeleton(kept, variant="minus_nucleus")

    assert part["cable_length_nm"] < whole["cable_length_nm"]
    assert part["cable_length_nm"] == pytest.approx(700.0, abs=60.0)
    # diameter is unchanged by removing a uniform-calibre stretch
    assert part["diameter_nm_mean"] == pytest.approx(whole["diameter_nm_mean"], abs=1e-6)


def test_every_row_has_the_same_keys_whatever_the_skeleton():
    """A table is only joinable if the columns do not depend on the body."""
    with_radii = set(measure_skeleton(_rod()))
    without = set(measure_skeleton(Skeleton(np.zeros((2, 3)), np.array([[0, 1]]))))
    assert with_radii == without
