"""The per-body skeleton pass: isolated failures, terminal absence, matched columns."""

import numpy as np
import pytest

from neu_morpho.measure.skeletons import _measure_one, sweep_skeletons
from neu_morpho.metrics_db import MetricsDB


class _Skel:
    """What `precomputed.encode_skeleton` reads: zyx `vertices`, `edges`, `radius`."""

    def __init__(self, verts_zyx, edges, radii=None, segid=None):
        self.vertices = np.asarray(verts_zyx, dtype=np.float32)
        self.edges = np.asarray(edges, dtype=np.uint32)
        self.radius = None if radii is None else np.asarray(radii, dtype=np.float32)
        self.id = segid


def _write_skeletons(tmp_path, bodies: dict, *, radii=True):
    """Publish a precomputed skeleton per body; returns the volume root."""
    from neu_morpho import precomputed
    from neu_vol import location

    vol = str(tmp_path / "volume")
    info = {"@type": "neuroglancer_skeletons"}
    if radii:
        info["vertex_attributes"] = [{"id": "radius", "data_type": "float32",
                                      "num_components": 1}]
    location.write_json(vol, "skeleton", "info", info)
    for body, (verts_zyx, edges, r) in bodies.items():
        blob = precomputed.encode_skeleton(
            _Skel(verts_zyx, edges, r if radii else None, body))
        location.write_bytes(vol, blob, "skeleton", str(body))
    return vol


def _line(n=4, step=100.0, radius=50.0):
    """A straight chain of n nodes `step` nm apart along z: cable = (n-1)*step."""
    verts = np.array([[i * step, 0.0, 0.0] for i in range(n)])
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.uint32)
    return verts, edges, np.full(n, radius)


def test_cable_length_and_columns(tmp_path):
    vol = _write_skeletons(tmp_path, {100: _line(4, 100.0, 50.0),
                                      200: _line(3, 250.0, 80.0)})
    db_path = str(tmp_path / "m.sqlite")
    out = sweep_skeletons(vol, db_path, bodies=[100, 200], threads=2, client=None)
    assert out["written"] == 2 and out["absent"] == 0 and out["failed"] == 0

    db = MetricsDB(db_path)
    cable, tips, rmax = db.con.execute(
        "SELECT cable_length_nm, n_tips, max_radius_nm FROM bodies WHERE body_id=100"
    ).fetchone()
    assert cable == pytest.approx(300.0)          # 3 edges x 100 nm
    assert tips == 2
    assert rmax == pytest.approx(50.0)            # a RADIUS, not a diameter
    assert db.con.execute(
        "SELECT cable_length_nm FROM bodies WHERE body_id=200").fetchone()[0] \
        == pytest.approx(500.0)


def test_a_missing_body_is_absent_and_terminal(tmp_path):
    """Absence is not failure: an unpublished body will not grow a skeleton, so
    retrying it forever is waste. `failed` is the state that gets another attempt."""
    vol = _write_skeletons(tmp_path, {100: _line()})
    db_path = str(tmp_path / "m.sqlite")
    out = sweep_skeletons(vol, db_path, bodies=[100, 999], client=None)
    assert out["written"] == 1 and out["absent"] == 1

    db = MetricsDB(db_path)
    assert db.skel_counts() == {"written": 1, "absent": 1}
    assert db.done_skel_bodies() == {100, 999}     # both terminal, neither retried
    second = sweep_skeletons(vol, db_path, bodies=[100, 999], client=None)
    assert second["processed"] == 0


def test_one_bad_body_does_not_kill_the_run(tmp_path):
    """Per-BODY faults isolate (invariant 5). A per-block task fails fast because a
    skipped block truncates every body in it; that reasoning does not apply here."""
    vol = _write_skeletons(tmp_path, {100: _line(), 200: _line()})
    from neu_vol import location
    location.write_bytes(vol, b"not a skeleton", "skeleton", "300")

    db_path = str(tmp_path / "m.sqlite")
    out = sweep_skeletons(vol, db_path, bodies=[100, 200, 300], client=None)
    assert out["written"] == 2 and out["failed"] == 1

    db = MetricsDB(db_path)
    status, detail = db.con.execute(
        "SELECT status, detail FROM skel_status WHERE body_id=300").fetchone()
    assert status == "failed" and detail            # the reason is kept, not discarded
    # a failed body IS retried by default, and is not reported done
    assert 300 not in db.done_skel_bodies()
    assert 300 in db.done_skel_bodies(include_failed=True)


def test_a_centreline_only_source_still_yields_cable(tmp_path):
    """The published source in the driving comparison has no radii at all."""
    vol = _write_skeletons(tmp_path, {100: _line(4, 100.0)}, radii=False)
    body, status, row = _measure_one(100, volume=vol, skeleton_dir="skeleton",
                                     require_radii=False)
    assert status == "written"
    assert row["cable_length_nm"] == pytest.approx(300.0)
    assert np.isnan(row["max_radius_nm"])          # NaN, not zero


def test_batching_covers_every_body(tmp_path):
    bodies = {i: _line(3, 100.0) for i in range(100, 110)}
    vol = _write_skeletons(tmp_path, bodies)
    db_path = str(tmp_path / "m.sqlite")
    out = sweep_skeletons(vol, db_path, bodies=list(bodies), batch=3, threads=2,
                          client=None)
    assert out["written"] == 10
    assert MetricsDB(db_path).con.execute(
        "SELECT COUNT(*) FROM bodies WHERE cable_length_nm>0").fetchone()[0] == 10


def test_the_stage_records_its_total(tmp_path):
    vol = _write_skeletons(tmp_path, {100: _line()})
    db_path = str(tmp_path / "m.sqlite")
    sweep_skeletons(vol, db_path, bodies=[100, 999], client=None)
    assert MetricsDB(db_path).read_stage_meta()["skel"]["total"] == 2
