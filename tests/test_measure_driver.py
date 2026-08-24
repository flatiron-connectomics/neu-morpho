"""The block-mapped volume sweep: exact counts, resumable, and refuses to double-count."""

import numpy as np
import pytest

from neu_morpho.measure import sweep_volumes
from neu_morpho.metrics_db import MetricsDB


def _write_seg(path, vol):
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _seg(tmp_path):
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100            # 512 voxels
    vol[20:28, 6:10, 6:10] = 200           # 128 voxels
    vol[12:20, 24:28, 24:28] = 300         # 128, crossing the z-block boundary at 16
    return _write_seg(str(tmp_path / "seg.zarr"), vol), vol


def test_counts_are_exact_across_block_boundaries(tmp_path):
    spec, vol = _seg(tmp_path)
    db_path = str(tmp_path / "m.sqlite")
    out = sweep_volumes(spec, db_path, voxel_size=(8, 8, 8), block=16, client=None)
    assert out["n_bodies"] == 3 and out["block_shape"] == (16, 16, 16)

    db = MetricsDB(db_path)
    for body, n in ((100, 512), (200, 128), (300, 128)):
        vc, v = db.con.execute(
            "SELECT voxel_count, volume_nm3 FROM bodies WHERE body_id=?", (body,)).fetchone()
        assert vc == n and v == n * 8 ** 3          # 300 spans two blocks and still totals
    assert db.get_bbox(100) == (None,) * 6          # counts-only: no bbox measured


def test_resume_does_not_double_count(tmp_path):
    spec, _ = _seg(tmp_path)
    db_path = str(tmp_path / "m.sqlite")
    kw = dict(voxel_size=(8, 8, 8), block=16, client=None)
    sweep_volumes(spec, db_path, **kw)
    second = sweep_volumes(spec, db_path, resume=True, **kw)
    assert second["processed"] == 0
    assert MetricsDB(db_path).con.execute(
        "SELECT voxel_count FROM bodies WHERE body_id=100").fetchone()[0] == 512


def test_keep_records_only_the_listed_bodies(tmp_path):
    spec, _ = _seg(tmp_path)
    db_path = str(tmp_path / "m.sqlite")
    out = sweep_volumes(spec, db_path, voxel_size=(8, 8, 8), block=16,
                        keep=[100, 300], client=None)
    assert out["n_kept"] == 2 and out["n_bodies"] == 2
    db = MetricsDB(db_path)
    assert db.con.execute("SELECT 1 FROM bodies WHERE body_id=200").fetchone() is None
    # the kept body that crosses a block boundary is still complete
    assert db.con.execute(
        "SELECT voxel_count FROM bodies WHERE body_id=300").fetchone()[0] == 128


def test_refuses_a_db_that_already_holds_an_index_scan(tmp_path):
    """Both fill voxel_count on the same grid, so running both would double every count.

    Separate progress tables make the collision detectable instead of silent.
    """
    spec, _ = _seg(tmp_path)
    db_path = str(tmp_path / "m.sqlite")
    from neu_morpho.ops.index_segments import index_segments
    index_segments(spec, db_path, scan_scale=0, scan_voxel_size=(8, 8, 8),
                   block_shape=(16, 16, 16), client=None)
    with pytest.raises(ValueError, match="already holds an index scan"):
        sweep_volumes(spec, db_path, voxel_size=(8, 8, 8), block=16, client=None)


def test_no_resume_clears_the_previous_totals(tmp_path):
    spec, _ = _seg(tmp_path)
    db_path = str(tmp_path / "m.sqlite")
    kw = dict(voxel_size=(8, 8, 8), block=16, client=None)
    sweep_volumes(spec, db_path, **kw)
    sweep_volumes(spec, db_path, resume=False, **kw)   # re-runs every block
    assert MetricsDB(db_path).con.execute(
        "SELECT voxel_count FROM bodies WHERE body_id=100").fetchone()[0] == 512


# --------------------------------------------------------------------------- #
# keep resolution
# --------------------------------------------------------------------------- #
def _write_props(path, ids):
    import json
    import os
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "info"), "w") as fh:
        json.dump({"@type": "neuroglancer_segment_properties",
                   "inline": {"ids": [str(i) for i in ids],
                              "properties": [{"id": "type", "type": "label",
                                              "values": ["x"] * len(ids)}]}}, fh)
    return path


def test_resolve_keep_unions_files_and_properties_sources(tmp_path):
    from neu_morpho.measure.driver import resolve_keep

    ids_file = tmp_path / "ids.csv"
    ids_file.write_text("body_id\n100\n200\n")
    a = _write_props(str(tmp_path / "propsA"), [200, 300])
    b = _write_props(str(tmp_path / "propsB"), [400])

    assert resolve_keep(None) is None
    assert resolve_keep([str(ids_file)]) == {100, 200}
    assert resolve_keep([a]) == {200, 300}
    # union, de-duplicated across kinds — a real dataset splits properties over several
    assert resolve_keep([str(ids_file), a, b]) == {100, 200, 300, 400}


def test_resolve_keep_refuses_a_volume_info(tmp_path):
    """A segmentation's own info has no id list, and pointing --keep at one is an easy
    slip: both are `info` files under a precomputed source."""
    import json
    import os
    from neu_morpho.measure.driver import resolve_keep

    vol = tmp_path / "seg"
    os.makedirs(vol)
    (vol / "info").write_text(json.dumps({"@type": "neuroglancer_multiscale_volume"}))
    with pytest.raises(ValueError, match="not a neuroglancer_segment_properties"):
        resolve_keep([str(vol)])


# --------------------------------------------------------------------------- #
# the progress denominator
# --------------------------------------------------------------------------- #
def test_the_sweep_records_its_total_before_dispatch(tmp_path):
    """A progress table counts tasks done; only the driver knows the denominator."""
    spec, _ = _seg(tmp_path)
    db_path = str(tmp_path / "m.sqlite")
    sweep_volumes(spec, db_path, voxel_size=(8, 8, 8), block=16, keep=[100], client=None)
    meta = MetricsDB(db_path).read_stage_meta()
    assert meta["sweep"]["total"] == 8            # 32^3 volume, 16^3 blocks
    assert meta["sweep"]["block_shape"] == "16,16,16"
    assert meta["sweep"]["n_keep"] == 1
    assert meta["sweep"]["started"]


def test_the_index_records_its_total_too(tmp_path):
    from neu_morpho.ops.index_segments import index_segments
    spec, _ = _seg(tmp_path)
    db_path = str(tmp_path / "m.sqlite")
    index_segments(spec, db_path, scan_scale=0, scan_voxel_size=(8, 8, 8),
                   block_shape=(16, 16, 16), client=None)
    meta = MetricsDB(db_path).read_stage_meta()
    assert meta["index"]["total"] == 8 and meta["index"]["level"] == 0


# --------------------------------------------------------------------------- #
# concurrency: a reader must not be able to kill a writer
# --------------------------------------------------------------------------- #
def test_a_read_only_open_writes_nothing(tmp_path):
    """Opening to LOOK at progress must not take a write lock.

    The default constructor issues `PRAGMA journal_mode` and four
    `CREATE TABLE IF NOT EXISTS` statements, which are writes — so a progress command
    built on a default open locked a live DB and killed a 25-minute sweep with
    'database is locked'.
    """
    import sqlite3

    db_path = str(tmp_path / "m.sqlite")
    MetricsDB(db_path).close()                       # create it normally

    ro = MetricsDB(db_path, read_only=True)
    try:
        assert ro.read_stage_meta() == {}
        assert ro.stage_counts() == dict.fromkeys(MetricsDB.BLOCK_STAGES, 0)
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            ro.con.execute("CREATE TABLE nope (x INTEGER)")
    finally:
        ro.close()


def test_read_only_survives_a_db_without_the_newer_tables(tmp_path):
    """An older DB has no stage_meta, and a read-only open cannot create it."""
    import sqlite3

    db_path = str(tmp_path / "old.sqlite")
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE bodies (body_id INTEGER PRIMARY KEY, voxel_count INTEGER)")
    con.commit()
    con.close()

    ro = MetricsDB(db_path, read_only=True)
    try:
        assert ro.read_stage_meta() == {}            # absent table, not a crash
        assert ro.stage_counts() == dict.fromkeys(MetricsDB.BLOCK_STAGES, 0)
    finally:
        ro.close()


def test_a_writer_waits_out_a_lock_instead_of_failing(tmp_path):
    """busy_timeout is the fix that protects against readers predating the convention."""
    db_path = str(tmp_path / "m.sqlite")
    db = MetricsDB(db_path)
    try:
        got = db.con.execute("PRAGMA busy_timeout").fetchone()[0]
        assert got >= 30_000
    finally:
        db.close()
