"""Path length from a reference region, for synapses attached to a skeleton.

Synthetic skeletons only: every expected distance below can be worked out by hand, which is
the point — a real neuron could only check the code against itself.
"""

import numpy as np
import pytest

from neu_lib import Frame, Skeleton
from neu_morpho.measure import paths

TOL = 0.01          # bisection tolerance, so boundary positions are exact to ~0.01 nm


def _z_below(limit):
    return lambda p: np.asarray(p)[:, 0] < limit


def _nowhere(p):
    return np.zeros(len(p), dtype=bool)


def _chain(z_values, y=0.0):
    z = np.asarray(z_values, dtype=float)
    verts = np.column_stack([z, np.full(len(z), y), np.zeros(len(z))])
    edges = np.stack([np.arange(len(z) - 1), np.arange(1, len(z))], axis=1)
    return verts, edges


def _join(*parts):
    verts, edges, offset = [], [], 0
    for v, e in parts:
        verts.append(v)
        edges.append(e + offset)
        offset += len(v)
    return Skeleton(np.concatenate(verts), np.concatenate(edges))


def _y_shape():
    """Trunk along z from 0 to 1000 (100 nm spacing); a branch leaves z=600 along +y."""
    trunk_v, trunk_e = _chain(np.arange(0, 1001, 100))
    branch_v = np.array([[600.0, 200.0, 0.0], [600.0, 400.0, 0.0]])
    verts = np.concatenate([trunk_v, branch_v])
    edges = np.concatenate([trunk_e, [[6, 11], [11, 12]]])
    return Skeleton(verts, edges)


def _broken(gap_lo=500.0, gap_hi=700.0):
    """Two collinear pieces along z, separated by a gap."""
    return _join(_chain(np.arange(0, gap_lo + 1, 100)),
                 _chain(np.arange(gap_hi, gap_hi + 501, 100)))


# --------------------------------------------------------------------------- #
# distances on a connected skeleton
# --------------------------------------------------------------------------- #
def test_distance_is_measured_along_the_cable_from_the_boundary():
    pts = np.array([[800.0, 50.0, 0.0],      # trunk, 500 past the boundary at z=300
                    [600.0, 300.0, 0.0],     # branch: 300 up the trunk + 300 along it
                    [100.0, 0.0, 0.0]])      # inside, 200 from the boundary
    per, summary = paths.synapse_distances(_y_shape(), pts, _z_below(300.0),
                                           max_gap_nm=0, tolerance_nm=TOL)
    assert per["distance_nm"] == pytest.approx([500.0, 600.0, -200.0], abs=0.1)
    assert per["inside"].tolist() == [False, False, True]
    assert per["snap_nm"] == pytest.approx([50.0, 0.0, 0.0], abs=1e-6)
    assert list(per["status"]) == ["ok"] * 3
    assert summary["n_boundary_crossings"] == 1
    assert summary["cable_nm"] == pytest.approx(1400.0)
    assert summary["cable_inside_nm"] == pytest.approx(300.0, abs=0.1)


def test_a_point_attaches_to_the_nearest_edge_not_the_nearest_vertex():
    skel = Skeleton(np.array([[0.0, 0, 0], [1000.0, 0, 0]]), np.array([[0, 1]]))
    att = paths.attach_points(skel, [[500.0, 30.0, 0.0]])
    assert att["edge"].tolist() == [0]
    assert att["t"] == pytest.approx([0.5])
    assert att["snap_nm"] == pytest.approx([30.0])
    assert att["position"][0] == pytest.approx([500.0, 0.0, 0.0])


def test_two_entries_into_the_region_each_act_as_a_boundary():
    """A U: both arms dip into the region. A point between them is measured from the
    nearer crossing, not from a single designated one."""
    left_v, left_e = _chain([0.0, 400.0, 800.0])
    right_v = np.array([[800.0, 1000.0, 0.0], [400.0, 1000.0, 0.0], [0.0, 1000.0, 0.0]])
    verts = np.concatenate([left_v, right_v])
    edges = np.concatenate([left_e, [[2, 3], [3, 4], [4, 5]]])
    per, summary = paths.synapse_distances(
        Skeleton(verts, edges), [[800.0, 200.0, 0.0], [600.0, 1000.0, 0.0]],
        _z_below(300.0), max_gap_nm=0, tolerance_nm=TOL, region_pieces="all")
    assert summary["n_boundary_crossings"] == 2
    # (800,200): up the left arm 500, then 200 along the top
    # (600,1000): 300 from the right arm's crossing
    assert per["distance_nm"] == pytest.approx([700.0, 300.0], abs=0.1)


def test_zero_length_and_duplicate_edges_do_not_change_the_graph():
    """scipy reads an explicit zero weight as no edge, and sums duplicate entries."""
    verts = np.array([[0.0, 0, 0], [100.0, 0, 0], [100.0, 0, 0], [300.0, 0, 0]])
    skel = Skeleton(verts, np.array([[0, 1], [1, 2], [2, 3], [0, 1]]))
    n, _ = paths.components(skel)
    assert n == 1
    per, _ = paths.synapse_distances(skel, [[300.0, 0, 0]], _z_below(50.0),
                                     max_gap_nm=0, tolerance_nm=TOL)
    assert per["distance_nm"] == pytest.approx([250.0], abs=0.1)


# --------------------------------------------------------------------------- #
# breaks and bridges
# --------------------------------------------------------------------------- #
def test_a_break_leaves_the_far_piece_unreachable_without_bridging():
    per, summary = paths.synapse_distances(_broken(), [[1000.0, 0, 0]], _z_below(300.0),
                                           max_gap_nm=0, tolerance_nm=TOL)
    assert list(per["status"]) == ["unreachable"]
    assert np.isnan(per["distance_nm"][0])
    assert summary["n_components"] == summary["n_components_bridged"] == 2


def test_a_bridge_counts_its_length_and_is_reported():
    per, summary = paths.synapse_distances(_broken(), [[1000.0, 0, 0]], _z_below(300.0),
                                           max_gap_nm=250, tolerance_nm=TOL)
    # 200 to the gap, 200 across it, 300 beyond
    assert per["distance_nm"] == pytest.approx([700.0], abs=0.1)
    assert per["n_bridges"].tolist() == [1]
    assert per["bridge_nm"] == pytest.approx([200.0])
    assert summary["n_bridges"] == 1 and summary["bridge_nm_total"] == pytest.approx(200.0)
    assert summary["n_components_bridged"] == 1
    assert summary["cable_nm"] == pytest.approx(1000.0)     # bridges are not cable


def test_a_gap_wider_than_the_margin_is_not_bridged():
    per, _ = paths.synapse_distances(_broken(), [[1000.0, 0, 0]], _z_below(300.0),
                                     max_gap_nm=150, tolerance_nm=TOL)
    assert list(per["status"]) == ["unreachable"]


def test_a_point_on_the_bridge_itself_is_flagged():
    per, _ = paths.synapse_distances(_broken(), [[600.0, 40.0, 0]], _z_below(300.0),
                                     max_gap_nm=250, tolerance_nm=TOL)
    assert per["on_bridge"].tolist() == [True]
    assert per["distance_nm"] == pytest.approx([300.0], abs=0.1)
    assert per["bridge_nm"] == pytest.approx([100.0])


def test_a_bridge_that_crosses_the_boundary_counts_only_its_outside_part():
    """The boundary falls inside the gap, so split_at cuts the BRIDGE — and the bridge
    flag has to survive that split."""
    per, summary = paths.synapse_distances(_broken(), [[1000.0, 0, 0], [300.0, 0, 0]],
                                           _z_below(600.0), max_gap_nm=250,
                                           tolerance_nm=TOL)
    assert summary["n_boundary_crossings"] == 1
    assert per["distance_nm"] == pytest.approx([400.0, -300.0], abs=0.1)
    assert per["bridge_nm"] == pytest.approx([100.0, 100.0], abs=0.1)
    assert per["n_bridges"].tolist() == [1, 1]


def test_bridging_joins_every_piece_without_a_cycle_and_keeps_vertex_order():
    skel = _join(_chain([0.0, 100.0]), _chain([250.0, 350.0]), _chain([500.0, 600.0]))
    out, is_bridge = paths.bridge_gaps(skel, 200.0)
    assert np.array_equal(out.vertices_zyx_nm, skel.vertices_zyx_nm)
    assert np.array_equal(out.edges[:len(skel.edges)], skel.edges)
    assert is_bridge.tolist() == [False, False, False, True, True]
    assert sorted(map(sorted, out.edges[is_bridge].tolist())) == [[1, 2], [3, 4]]
    assert paths.components(out)[0] == 1


def test_bridging_takes_the_shortest_link_between_two_pieces():
    """Kimimaro's greedy merge joins the closest pair; so must this."""
    a = _chain([0.0, 100.0, 200.0])
    # vertex 4 (140,80) is 89.4 from vertex 1; vertex 3 (300,0) is 100 from vertex 2
    b_v = np.array([[300.0, 0.0, 0.0], [140.0, 80.0, 0.0]])
    skel = _join(a, (b_v, np.array([[0, 1]])))
    out, is_bridge = paths.bridge_gaps(skel, 500.0)
    assert sorted(out.edges[is_bridge][0].tolist()) == [1, 4]


def test_a_negative_gap_is_refused():
    with pytest.raises(ValueError, match="max_gap_nm"):
        paths.bridge_gaps(_broken(), -1)


# --------------------------------------------------------------------------- #
# nothing is dropped
# --------------------------------------------------------------------------- #
def test_a_body_that_never_enters_the_region_is_no_compartment():
    per, summary = paths.synapse_distances(_y_shape(), [[800.0, 0, 0]], _nowhere,
                                           max_gap_nm=0)
    assert list(per["status"]) == ["no_compartment"]
    assert np.isnan(per["distance_nm"][0]) and summary["n_boundary_crossings"] == 0


def test_the_table_carries_inputs_through_and_reports_missing_skeletons():
    pd = pytest.importorskip("pandas")
    frame = Frame((10.0, 10.0, 10.0))                       # coordinates in 10 nm voxels
    syn = pd.DataFrame({"post_z": [80, 10, 50], "post_y": [0, 0, 0], "post_x": [0, 0, 0],
                        "post_body": [7, 7, 9], "pre_type": ["Mi1", "Tm1", "Mi1"]})
    rows, bodies = paths.synapse_distance_table(
        {7: _y_shape()}, syn, _z_below(300.0), frame=frame, max_gap_nm=0,
        tolerance_nm=TOL, dataset="toy")
    assert rows["pre_type"].tolist() == ["Mi1", "Tm1", "Mi1"]
    assert rows["distance_nm"].iloc[:2].tolist() == pytest.approx([500.0, -200.0], abs=0.1)
    assert rows["status"].tolist() == ["ok", "ok", "no_skeleton"]
    assert (rows["max_gap_nm"] == 0).all() and (rows["dataset"] == "toy").all()
    assert rows[["attached_z_nm", "attached_y_nm", "attached_x_nm"]].iloc[0].tolist() == \
        pytest.approx([800.0, 0.0, 0.0])
    assert bodies.set_index("body_id").loc[9, "status"] == "no_skeleton"
    assert bodies.set_index("body_id").loc[7, "n_ok"] == 2


def test_a_table_missing_coordinates_says_which():
    pd = pytest.importorskip("pandas")
    with pytest.raises(KeyError, match="post_x"):
        paths.synapse_distance_table({}, pd.DataFrame({"post_z": [1], "post_y": [1],
                                                       "post_body": [1]}),
                                     _nowhere, frame=Frame((1, 1, 1)), max_gap_nm=0)


# --------------------------------------------------------------------------- #
# what each distance was measured from, and against
# --------------------------------------------------------------------------- #
def test_each_point_names_the_crossing_it_was_measured_from():
    left_v, left_e = _chain([0.0, 400.0, 800.0])
    right_v = np.array([[800.0, 1000.0, 0.0], [400.0, 1000.0, 0.0], [0.0, 1000.0, 0.0]])
    skel = Skeleton(np.concatenate([left_v, right_v]),
                    np.concatenate([left_e, [[2, 3], [3, 4], [4, 5]]]))
    per, summary = paths.synapse_distances(
        skel, [[700.0, 0.0, 0.0], [600.0, 1000.0, 0.0]], _z_below(300.0),
        max_gap_nm=0, tolerance_nm=TOL, region_pieces="all")
    assert per["entry"][0] == pytest.approx([300.0, 0.0, 0.0], abs=0.1)
    assert per["entry"][1] == pytest.approx([300.0, 1000.0, 0.0], abs=0.1)
    assert summary["n_entries_used"] == 2


def test_an_unreachable_point_has_no_entry():
    per, summary = paths.synapse_distances(_broken(), [[1000.0, 0, 0], [100.0, 0, 0]],
                                           _z_below(300.0), max_gap_nm=0, tolerance_nm=TOL)
    assert np.isnan(per["entry"][0]).all() and not np.isnan(per["entry"][1]).any()
    assert summary["n_entries_used"] == 1


def test_the_radius_at_the_attachment_is_interpolated():
    skel = Skeleton(np.array([[0.0, 0, 0], [1000.0, 0, 0]]), np.array([[0, 1]]),
                    radii_nm=np.array([100.0, 300.0]))
    per, _ = paths.synapse_distances(skel, [[750.0, 400.0, 0.0]], _z_below(10.0),
                                     max_gap_nm=0, tolerance_nm=TOL)
    assert per["radius_nm"] == pytest.approx([250.0], abs=0.5)
    assert per["snap_nm"] == pytest.approx([400.0])


def test_no_radii_gives_nan_radius_not_zero():
    per, _ = paths.synapse_distances(_y_shape(), [[800.0, 0, 0]], _z_below(300.0),
                                     max_gap_nm=0)
    assert np.isnan(per["radius_nm"]).all()


# --------------------------------------------------------------------------- #
# which in-region cable is the compartment
# --------------------------------------------------------------------------- #
def _terminal_and_dip(points):
    """Inside z < 1000 (the terminal, 1000 nm of cable) or 2400 < z < 2500 (a dip)."""
    p = np.asarray(points)[:, 0]
    return (p < 1000.0) | ((p > 2400.0) & (p < 2500.0))


def test_largest_ignores_a_dip_into_the_region():
    """The measured failure: a dendrite brushing a coarse ROI. Measured from the dip, the
    synapse at z=2800 would be 300 nm out; from the terminal it is 1800."""
    skel = Skeleton(*_chain(np.arange(0, 3001, 50)))
    pts = [[2800.0, 0, 0], [2450.0, 0, 0], [500.0, 0, 0]]
    largest, summary = paths.synapse_distances(skel, pts, _terminal_and_dip,
                                               max_gap_nm=0, tolerance_nm=TOL)
    every, _ = paths.synapse_distances(skel, pts, _terminal_and_dip, max_gap_nm=0,
                                       tolerance_nm=TOL, region_pieces="all")
    assert largest["distance_nm"] == pytest.approx([1800.0, 1450.0, -500.0], abs=0.2)
    assert every["distance_nm"] == pytest.approx([300.0, -50.0, -500.0], abs=0.2)
    assert largest["inside"].tolist() == [False, False, True]
    assert largest["in_roi"].tolist() == [False, True, True]
    assert summary["n_region_pieces"] == 2 and summary["n_compartment_crossings"] == 1
    assert summary["compartment_cable_nm"] == pytest.approx(1000.0, abs=0.2)
    assert summary["cable_inside_nm"] == pytest.approx(1100.0, abs=0.5)


def test_an_unknown_region_pieces_is_refused():
    with pytest.raises(ValueError, match="region_pieces"):
        paths.synapse_distances(_y_shape(), [[0, 0, 0]], _z_below(300.0), max_gap_nm=0,
                                region_pieces="biggest")


def _weave_and_dip(points):
    """Terminal z < 1000, a weave piece 1100..1300 just past it, a dip at 2400..2500."""
    p = np.asarray(points)[:, 0]
    return (p < 1000.0) | ((p > 1100.0) & (p < 1300.0)) | ((p > 2400.0) & (p < 2500.0))


def test_merge_absorbs_a_nearby_weave_but_not_a_distant_dip():
    skel = Skeleton(*_chain(np.arange(0, 3001, 50)))
    pts = [[2000.0, 0, 0], [1200.0, 0, 0]]
    alone, s0 = paths.synapse_distances(skel, pts, _weave_and_dip, max_gap_nm=0,
                                        tolerance_nm=TOL)
    merged, s1 = paths.synapse_distances(skel, pts, _weave_and_dip, max_gap_nm=0,
                                         tolerance_nm=TOL, merge_nm=200.0)
    # alone: from the terminal at 1000. merged: the weave joins, so from 1300.
    assert alone["distance_nm"] == pytest.approx([1000.0, 200.0], abs=0.2)
    assert merged["distance_nm"] == pytest.approx([700.0, -100.0], abs=0.2)
    assert (s0["n_compartment_pieces"], s1["n_compartment_pieces"]) == (1, 2)
    assert s1["n_region_pieces"] == 3


def test_merge_is_transitive():
    """Each piece only needs to be near the GROWING compartment, not the first piece."""
    skel = Skeleton(*_chain(np.arange(0, 3001, 50)))
    chain = lambda p: ((np.asarray(p)[:, 0] < 1000)
                       | ((np.asarray(p)[:, 0] > 1150) & (np.asarray(p)[:, 0] < 1300))
                       | ((np.asarray(p)[:, 0] > 1450) & (np.asarray(p)[:, 0] < 1600)))
    _, s = paths.synapse_distances(skel, [[2000.0, 0, 0]], chain, max_gap_nm=0,
                                   tolerance_nm=TOL, merge_nm=200.0)
    assert s["n_compartment_pieces"] == 3


# --------------------------------------------------------------------------- #
# a compartment marked by the cell's own points
# --------------------------------------------------------------------------- #
def test_near_points_region_is_a_union_of_balls():
    inside = paths.near_points_region([[0.0, 0, 0], [1000.0, 0, 0]], 100.0)
    q = np.array([[50.0, 0, 0], [0.0, 99.0, 0], [500.0, 0, 0], [1000.0, 0, 101.0]])
    assert inside(q).tolist() == [True, True, False, False]


def test_near_points_region_with_no_points_is_inside_nothing():
    assert not paths.near_points_region(np.zeros((0, 3)), 100.0)(np.zeros((3, 3))).any()


def test_outputs_mark_the_compartment_end_to_end():
    """T-bars along the far end of a chain, offset from the centreline like membrane
    sites: distances are measured from where the cable reaches the first of them."""
    skel = Skeleton(*_chain(np.arange(0, 3001, 50)))
    tbars = [[2200.0, 150.0, 0], [2500.0, -150.0, 0], [2900.0, 150.0, 0]]
    inside = paths.near_points_region(tbars, 300.0)
    per, s = paths.synapse_distances(skel, [[500.0, 100.0, 0]], inside, max_gap_nm=0,
                                     tolerance_nm=TOL, merge_nm=500.0)
    # the ball around (2200,150) reaches the centreline from z = 2200 - sqrt(300²-150²)
    assert per["distance_nm"] == pytest.approx([2200 - np.sqrt(300**2 - 150**2) - 500],
                                               abs=0.5)
    assert s["n_compartment_pieces"] == s["n_region_pieces"]


def test_the_table_takes_one_region_per_body():
    pd = pytest.importorskip("pandas")
    syn = pd.DataFrame({"post_z": [800, 800, 800], "post_y": [0, 0, 0], "post_x": [0, 0, 0],
                        "post_body": [1, 2, 3]})
    rows, bodies = paths.synapse_distance_table(
        {1: _y_shape(), 2: _y_shape(), 3: _y_shape()}, syn,
        {1: _z_below(300.0), 2: _z_below(500.0)}, frame=Frame((1, 1, 1)), max_gap_nm=0,
        tolerance_nm=TOL)
    assert rows["distance_nm"].iloc[:2].tolist() == pytest.approx([500.0, 300.0], abs=0.1)
    assert rows["status"].tolist() == ["ok", "ok", "no_compartment"]
    assert bodies.set_index("body_id").loc[3, "status"] == "no_compartment"
