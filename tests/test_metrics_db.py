from em_seg_morpho.metrics_db import MetricsDB


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
