from neu_morpho.metrics_db import MetricsDB


def test_apply_blocks_merges_bbox_and_sums_count(tmp_path):
    db = MetricsDB(str(tmp_path / "m.sqlite"))
    # body 7 spans two blocks; body 9 in one. (z0,y0,x0,z1,y1,x1,count)
    db.apply_index_block("0_0_0", {7: (0, 0, 0, 10, 10, 10, 100), 9: (0, 0, 0, 4, 4, 4, 8)}, 512.0)
    db.apply_index_block("1_0_0", {7: (10, 2, 2, 20, 8, 8, 50)}, 512.0)
    assert db.get_bbox(7) == (0, 0, 0, 20, 10, 10)      # min of mins, max of maxes
    assert db.get_bbox(9) == (0, 0, 0, 4, 4, 4)
    r = db.con.execute("SELECT voxel_count, volume_nm3 FROM bodies WHERE body_id=7").fetchone()
    assert r == (150, 150 * 512.0)


def test_apply_block_is_idempotent(tmp_path):
    db = MetricsDB(str(tmp_path / "m.sqlite"))
    db.apply_index_block("0_0_0", {7: (0, 0, 0, 10, 10, 10, 100)}, 1.0)
    db.apply_index_block("0_0_0", {7: (0, 0, 0, 10, 10, 10, 100)}, 1.0)   # replay -> no double count
    assert db.con.execute("SELECT voxel_count FROM bodies WHERE body_id=7").fetchone()[0] == 100
    assert db.done_blocks() == {"0_0_0"}


def test_bodies_by_size_and_allowlist(tmp_path):
    db = MetricsDB(str(tmp_path / "m.sqlite"))
    db.apply_index_block("0", {1: (0, 0, 0, 1, 1, 1, 5), 2: (0, 0, 0, 9, 9, 9, 900),
                               3: (0, 0, 0, 3, 3, 3, 50)}, 1.0)
    assert db.bodies_by_size(min_voxels=10) == [2, 3]        # sorted by count desc
    assert db.bodies_by_size(min_voxels=10, limit=1) == [2]
    p = str(tmp_path / "allow.csv")
    n = db.write_allowlist(p, min_voxels=10)
    assert n == 2 and open(p).read().splitlines() == ["body_id", "2", "3"]


def test_crop_at_scale(tmp_path):
    db = MetricsDB(str(tmp_path / "m.sqlite"))
    db.apply_index_block("0", {7: (16, 16, 16, 48, 48, 48, 100)}, 1.0)   # full-res bbox
    # read at factor 8 (scale 3): //8 -> 2..6; +margin1 -> 1..7; clip 10
    assert db.crop_at_scale(7, (8, 8, 8), margin_vox=1, clip_shape=(10, 10, 10)) == (1, 1, 1, 7, 7, 7)
    assert db.crop_at_scale(999, (8, 8, 8)) is None


def test_enrichment_update(tmp_path):
    db = MetricsDB(str(tmp_path / "m.sqlite"))
    db.apply_index_block("0", {7: (0, 0, 0, 10, 10, 10, 100)}, 1.0)
    db.update_body(7, cable_length_nm=1234.5, n_branches=3)
    r = db.con.execute("SELECT cable_length_nm, n_branches FROM bodies WHERE body_id=7").fetchone()
    assert r == (1234.5, 3)


def test_update_bodies_batches_into_one_commit(tmp_path):
    """Per-body commits made the driver a serial fsync bottleneck; batch them."""
    from neu_morpho.metrics_db import MetricsDB

    db = MetricsDB(str(tmp_path / "m.db"))
    n = db.update_bodies([(7, {"cable_length_nm": 1.0, "n_tips": 2}),
                          (8, {"cable_length_nm": 3.0}),
                          (9, {})])                      # empty cols skipped
    assert n == 2
    rows = dict(db.con.execute("SELECT body_id, cable_length_nm FROM bodies"))
    assert rows == {7: 1.0, 8: 3.0}
    db.close()

    # durable after close, i.e. it really committed
    db2 = MetricsDB(str(tmp_path / "m.db"))
    assert db2.con.execute(
        "SELECT n_tips FROM bodies WHERE body_id=7").fetchone()[0] == 2
    db2.close()


def test_update_bodies_upserts_over_existing_rows(tmp_path):
    from neu_morpho.metrics_db import MetricsDB

    db = MetricsDB(str(tmp_path / "m.db"))
    db.apply_index_block("0_0_0", {7: (0, 0, 0, 10, 10, 10, 500)}, 1.0)
    db.update_bodies([(7, {"cable_length_nm": 9.0})])
    row = db.con.execute(
        "SELECT voxel_count, cable_length_nm FROM bodies WHERE body_id=7").fetchone()
    assert row == (500, 9.0)               # index data preserved, metrics added
    db.close()
