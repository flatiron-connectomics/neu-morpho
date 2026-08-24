"""The voxel-counting sweep: block selection, counting, binning, histograms.

Everything here is arrays in and numbers out, so it runs with no store. The properties
worth pinning are the ones that were *measured* against real data and would be silent if
they regressed: zeros dropped before counting, the ROI block set dilated, the histogram
summing to its input, and V/L recovering a tube's true cross-section.
"""

import numpy as np
import pytest

from neu_morpho.measure.sweep import (DEFAULT_BLOCK, DEFAULT_VOXEL_NM, SweepTotals,
                                      bin_to_nodes, blob_signal, blocks_from_mask,
                                      cable_shares, count_labels, log_bin_edges,
                                      mean_cross_section, node_radii, roi_block_mask,
                                      weighted_histogram)


# --------------------------------------------------------------------------- #
# block selection from an ROI
# --------------------------------------------------------------------------- #
def test_a_block_holding_one_roi_voxel_is_kept():
    """Reduced by `any`, not by majority: one voxel of tissue is tissue."""
    roi = np.zeros((8, 8, 8), dtype=np.uint64)
    roi[3, 3, 3] = 1
    grid = roi_block_mask(roi, [1], factor=4, dilate=0)
    assert grid.shape == (2, 2, 2)
    assert grid.sum() == 1 and grid[0, 0, 0]


def test_dilation_adds_the_six_neighbours():
    """A false-positive block costs one read that finds nothing; a false negative
    silently truncates a body. So the default errs outward."""
    roi = np.zeros((12, 12, 12), dtype=np.uint64)
    roi[5, 5, 5] = 1
    bare = roi_block_mask(roi, [1], factor=4, dilate=0)
    grown = roi_block_mask(roi, [1], factor=4, dilate=1)
    assert bare.sum() == 1
    assert grown.sum() == 7                      # itself plus 6 face neighbours
    assert np.all(grown[bare])                   # and it never drops one


def test_dilation_does_not_wrap_around_the_volume():
    """A corner block has three face neighbours, not six.

    The interior fixture above passes under a periodic dilation too, which is how a
    wrapping `np.roll` implementation survived: it marked the block on the OPPOSITE
    face as occupied. Harmless per-block — a false positive is one empty read — but it
    inflates the block count, and on a small grid it inflates it a lot. The observable
    symptom is a coarser occupancy scale reporting more blocks than a finer one.
    """
    roi = np.zeros((12, 12, 12), dtype=np.uint64)
    roi[0, 0, 0] = 1
    grown = roi_block_mask(roi, [1], factor=4, dilate=1)
    assert grown.shape == (3, 3, 3)
    assert grown.sum() == 4                      # itself + 3 in-bounds face neighbours
    for far in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
        assert not grown[far], f"dilation wrapped to the opposite face at {far}"


def test_only_the_named_labels_count():
    roi = np.zeros((4, 4, 4), dtype=np.uint64)
    roi[0, 0, 0] = 1        # brain
    roi[3, 3, 3] = 5        # VNC
    grid = roi_block_mask(roi, [1, 2, 3], factor=4, dilate=0)
    assert grid.sum() == 1


def test_a_ragged_roi_is_padded_not_truncated():
    """The last partial block still holds tissue."""
    roi = np.zeros((5, 5, 5), dtype=np.uint64)
    roi[4, 4, 4] = 1
    grid = roi_block_mask(roi, [1], factor=4, dilate=0)
    assert grid.shape == (2, 2, 2) and grid[1, 1, 1]


def test_blocks_outside_the_volume_are_dropped():
    grid = np.ones((3, 1, 1), dtype=bool)
    got = blocks_from_mask(grid, block=512, shape=(1000, 512, 512))
    assert got == [(0, 0, 0), (512, 0, 0)]        # 1024 is past the 1000-voxel extent


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
def test_count_labels_excludes_background():
    blk = np.array([[[0, 0], [7, 7]], [[7, 0], [9, 0]]], dtype=np.uint64)
    assert count_labels(blk) == {7: 3, 9: 1}


def test_an_empty_block_counts_nothing():
    assert count_labels(np.zeros((4, 4, 4), dtype=np.uint64)) == {}


def test_a_non_zero_background_can_be_named():
    """Annotation tools that number from 1 make background 1, not 0."""
    blk = np.ones((3, 3, 3), dtype=np.uint64)
    blk[0, 0, 0] = 42
    assert count_labels(blk, background=1) == {42: 1}


def test_counts_are_python_ints_not_numpy_scalars():
    """They get summed across thousands of blocks and written to a table; a uint64
    numpy scalar overflows silently where a Python int does not."""
    got = count_labels(np.full((2, 2, 2), 5, dtype=np.uint64))
    (label, n), = got.items()
    assert type(label) is int and type(n) is int


# --------------------------------------------------------------------------- #
# V/L, on a tube whose answer is known
# --------------------------------------------------------------------------- #
def test_v_over_l_recovers_a_tubes_cross_section():
    """The whole method, on the one geometry where the truth is arithmetic."""
    area, radius = mean_cross_section(volume_nm3=np.pi * 100.0**2 * 1000.0,
                                      cable_nm=1000.0)
    assert area == pytest.approx(np.pi * 100.0**2)
    assert radius == pytest.approx(100.0)


def test_v_over_l_under_reads_a_sphere_by_the_expected_factor():
    """V/L is a TUBE measure. A sphere of radius R with a centreline crossing it gives
    (4/3)piR^3 / 2R = (2/3)piR^2, so the radius comes back at sqrt(2/3) = 0.816 R. This
    is why the thick tail is compressed, and why somata are excluded rather than trusted."""
    R = 1000.0
    _, r = mean_cross_section(volume_nm3=(4/3) * np.pi * R**3, cable_nm=2 * R)
    assert r / R == pytest.approx(np.sqrt(2/3), rel=1e-6)


def test_no_cable_gives_nan_not_inf():
    """A body with volume and no centreline is a fact about the skeleton; an inf would
    propagate into every aggregate downstream."""
    area, radius = mean_cross_section(1e9, 0.0)
    assert np.isnan(area) and np.isnan(radius)


# --------------------------------------------------------------------------- #
# nearest-node binning
# --------------------------------------------------------------------------- #
def _rod_block(radius_vox=3, length=16, voxel_nm=32.0, body=5):
    """A cylinder along z inside a block, plus a decoy body."""
    n = 2 * radius_vox + 8
    blk = np.zeros((length, n, n), dtype=np.uint64)
    cz, cy = n // 2, n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    disc = (yy - cz) ** 2 + (xx - cy) ** 2 <= radius_vox ** 2
    blk[:, disc] = body
    blk[0, 0, 0] = 999                       # a different body, must not be counted
    return blk


def test_binning_assigns_every_voxel_of_the_body_and_nothing_else():
    blk = _rod_block()
    nodes = np.array([[4 * 32.0, 224.0, 224.0], [12 * 32.0, 224.0, 224.0]])
    got = bin_to_nodes(blk, (0, 0, 0), 32.0, nodes, body_id=5)
    assert sum(got["counts"].values()) == int((blk == 5).sum())
    assert set(got["counts"]) <= {0, 1}


def test_binning_splits_a_rod_between_two_nodes_evenly():
    blk = _rod_block(length=16)
    nodes = np.array([[4 * 32.0, 224.0, 224.0], [12 * 32.0, 224.0, 224.0]])
    got = bin_to_nodes(blk, (0, 0, 0), 32.0, nodes, body_id=5)
    a, b = got["counts"][0], got["counts"][1]
    assert abs(a - b) <= max(a, b) * 0.05


def test_binning_a_body_that_is_absent_returns_empty():
    got = bin_to_nodes(np.zeros((4, 4, 4), dtype=np.uint64), (0, 0, 0), 32.0,
                       np.zeros((1, 3)), body_id=7)
    assert got == {"counts": {}, "dist_sum": {}}


def test_node_ids_are_carried_through():
    """Blocks are accumulated per body across a sweep, so a local index is useless —
    the caller's own node ids have to come back."""
    blk = _rod_block()
    nodes = np.array([[4 * 32.0, 224.0, 224.0], [12 * 32.0, 224.0, 224.0]])
    got = bin_to_nodes(blk, (0, 0, 0), 32.0, nodes, body_id=5, node_ids=[101, 102])
    assert set(got["counts"]) <= {101, 102}


def test_the_rod_recovers_its_own_radius_through_the_full_path():
    """Block -> bin -> node_radii, on a cylinder of known radius."""
    r_vox, length, vox = 4, 32, 32.0
    blk = _rod_block(radius_vox=r_vox, length=length, voxel_nm=vox)
    zc = np.arange(2, length, 4) * vox
    nodes = np.stack([zc, np.full_like(zc, 224.0), np.full_like(zc, 224.0)], axis=1)
    got = bin_to_nodes(blk, (0, 0, 0), vox, nodes, body_id=5)
    share = {i: 4 * vox for i in range(len(nodes))}       # 4 voxels of cable per node
    radii, weights = node_radii(got["counts"], share, voxel_nm=vox)
    # a digitised disc of radius 4 voxels has area ~pi*4^2 voxels, so r ~ 4 voxels
    assert np.median(radii) == pytest.approx(r_vox * vox, rel=0.15)
    assert weights.sum() == pytest.approx(len(nodes) * 4 * vox)


# --------------------------------------------------------------------------- #
# accumulation
# --------------------------------------------------------------------------- #
def test_totals_sum_across_blocks():
    tot = SweepTotals(voxel_nm=32.0)
    tot.add_block({7: 10, 9: 1})
    tot.add_block({7: 5})
    assert tot.body_voxels == {7: 15, 9: 1} and tot.blocks == 2
    assert tot.volume_nm3(7) == 15 * 32.0**3


def test_node_totals_sum_across_blocks_for_one_body():
    """A body straddles blocks, so a node's voxels arrive in pieces."""
    tot = SweepTotals()
    tot.add_nodes(7, {"counts": {1: 4}, "dist_sum": {1: 40.0}})
    tot.add_nodes(7, {"counts": {1: 6, 2: 2}, "dist_sum": {1: 60.0, 2: 10.0}})
    assert tot.node_voxels[7] == {1: 10, 2: 2}
    assert tot.node_dist_sum[7] == {1: 100.0, 2: 10.0}


def test_an_unseen_body_has_zero_volume_not_a_keyerror():
    assert SweepTotals().volume_nm3(12345) == 0.0


# --------------------------------------------------------------------------- #
# the blob diagnostic
# --------------------------------------------------------------------------- #
def _cylinder_expected(r, s):
    """What a cylinder of radius r whose node owns s of cable actually produces."""
    return np.sqrt(r * r / 2.0 + s * s / 12.0)


def test_a_tube_reads_as_tube_like():
    """The comparison must be against a CYLINDER's expected mean distance, not against
    r: a voxel's distance to its node has an axial component from the node spacing as
    well as a radial one. Comparing against r alone reported 86% of a real thin arbor as
    blob-like, at 181 nm spacing and a 115 nm radius."""
    counts, share = {0: 100}, {0: 1000.0}
    r = node_radii(counts, share)[0][0]
    exp = _cylinder_expected(r, 1000.0)
    assert blob_signal(counts, {0: 100 * exp}, share) == 0.0


def test_a_node_spacing_comparable_to_the_radius_is_still_a_tube():
    """The regression that motivated the fix: s ~ r is the ordinary case, not a corner."""
    counts, share = {0: 400}, {0: 181.0}
    r = node_radii(counts, share, voxel_nm=32.0)[0][0]
    exp = _cylinder_expected(r, 181.0)
    assert exp > r * 0.5                      # the axial term is not negligible here
    assert blob_signal(counts, {0: 400 * exp}, share, voxel_nm=32.0) == 0.0


def test_a_blob_reads_as_blob_like():
    counts, share = {0: 100}, {0: 1000.0}
    r = node_radii(counts, share)[0][0]
    exp = _cylinder_expected(r, 1000.0)
    assert blob_signal(counts, {0: 100 * 3.0 * exp}, share) == 1.0


def test_the_blob_signal_is_cable_weighted():
    """One long blob-like stretch matters more than many short tube-like ones."""
    counts = {0: 100, 1: 100}
    share = {0: 900.0, 1: 100.0}
    r0 = node_radii({0: 100}, {0: 900.0})[0][0]
    r1 = node_radii({1: 100}, {1: 100.0})[0][0]
    got = blob_signal(counts, {0: 100 * 3 * _cylinder_expected(r0, 900.0),
                               1: 100 * _cylinder_expected(r1, 100.0) * 0.5}, share)
    assert got == pytest.approx(0.9)


def test_no_cable_gives_nan_blob_signal():
    assert np.isnan(blob_signal({}, {}, {}))


# --------------------------------------------------------------------------- #
# cable shares
# --------------------------------------------------------------------------- #
def test_cable_shares_sum_to_the_total_cable():
    v = np.array([[0.0, 0, 0], [0.0, 0, 100.0], [0.0, 0, 300.0]])
    e = np.array([[0, 1], [1, 2]])
    share = cable_shares(v, e)
    assert share.sum() == pytest.approx(300.0)
    assert share[1] == pytest.approx(150.0)      # half of each incident edge


def test_an_edgeless_skeleton_has_no_shares():
    assert cable_shares(np.zeros((3, 3)), np.zeros((0, 2), dtype=np.int64)).sum() == 0.0


# --------------------------------------------------------------------------- #
# the histogram
# --------------------------------------------------------------------------- #
def test_edges_are_log_spaced():
    e = log_bin_edges(16.0, 16000.0, 3)
    assert len(e) == 4
    assert np.allclose(np.diff(np.log(e)), np.log(10.0))


def test_edges_refuse_a_non_positive_range():
    with pytest.raises(ValueError, match="need 0 < lo < hi"):
        log_bin_edges(0.0, 100.0, 8)
    with pytest.raises(ValueError, match="need 0 < lo < hi"):
        log_bin_edges(100.0, 10.0, 8)


def test_the_histogram_has_catch_all_bins_and_conserves_its_input():
    """hist.sum() == weights.sum() is the consistency check that catches a wrong bin
    range. Clipping instead would lose cable silently."""
    edges = log_bin_edges(100.0, 1000.0, 4)
    values = np.array([1.0, 150.0, 900.0, 5000.0])       # under, in, in, over
    weights = np.array([7.0, 1.0, 2.0, 3.0])
    hist = weighted_histogram(values, weights, edges)
    assert len(hist) == len(edges) + 1
    assert hist.sum() == pytest.approx(weights.sum())
    assert hist[0] == 7.0 and hist[-1] == 3.0


def test_a_value_on_an_edge_lands_in_the_upper_bin():
    edges = log_bin_edges(100.0, 1000.0, 2)              # 100, 316.2, 1000
    hist = weighted_histogram(np.array([edges[1]]), np.array([1.0]), edges)
    assert hist[2] == 1.0 and hist[1] == 0.0


def test_the_top_edge_is_overflow_not_the_last_bin():
    edges = log_bin_edges(100.0, 1000.0, 2)
    hist = weighted_histogram(np.array([1000.0]), np.array([1.0]), edges)
    assert hist[-1] == 1.0


def test_non_finite_values_are_dropped_not_binned():
    """A body with no cable gives a NaN radius; it must not silently land in a bin."""
    edges = log_bin_edges(100.0, 1000.0, 2)
    hist = weighted_histogram(np.array([np.nan, 200.0]), np.array([5.0, 1.0]), edges)
    assert hist.sum() == 1.0


def test_an_empty_input_gives_an_all_zero_histogram_of_the_right_width():
    edges = log_bin_edges(100.0, 1000.0, 4)
    hist = weighted_histogram(np.array([]), np.array([]), edges)
    assert hist.shape == (len(edges) + 1,) and not hist.any()


def test_mismatched_values_and_weights_raise():
    with pytest.raises(ValueError, match="differ"):
        weighted_histogram(np.zeros(3), np.zeros(2), log_bin_edges(1.0, 10.0, 2))


def test_the_defaults_are_the_measured_ones():
    """512^3 read unit and 32 nm: both chosen from throughput and scale-bias
    measurements, so a silent change to either would invalidate the numbers."""
    assert DEFAULT_BLOCK == 512
    assert DEFAULT_VOXEL_NM == 32.0
