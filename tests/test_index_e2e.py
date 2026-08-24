"""End-to-end index scan: synthetic segmentation -> exact per-body bbox + count."""

import numpy as np

from neu_morpho.metrics_db import MetricsDB
from neu_morpho.ops.index_segments import index_segments


def _write_seg(path, vol):
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region((slice(0, vol.shape[0]), slice(0, vol.shape[1]), slice(0, vol.shape[2])), vol)


def test_index_matches_ground_truth(tmp_path):
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100            # 8^3 = 512 voxels, bbox [4:12]
    vol[20:28, 6:10, 6:10] = 200           # 8*4*4 = 128 voxels, spans 2 blocks in z? no (20:28 in block 1)
    vol[12:20, 24:28, 24:28] = 300         # crosses z-block boundary (16)
    src = str(tmp_path / "seg.zarr")
    _write_seg(src, vol)
    db_path = str(tmp_path / "metrics.sqlite")

    summary = index_segments({"backend": "zarr3", "path": src}, db_path,
                             scan_scale=0, scan_voxel_size=(8, 8, 8),
                             block_shape=(16, 16, 16), client=None)
    assert summary["n_bodies"] == 3

    db = MetricsDB(db_path)
    # bbox is full-res voxels (scan_scale 0 -> factor 1); half-open max
    assert db.get_bbox(100) == (4, 4, 4, 12, 12, 12)
    assert db.get_bbox(300) == (12, 24, 24, 20, 28, 28)     # correct across the z-block boundary
    for body, n in [(100, 512), (200, 128), (300, 8 * 4 * 4)]:
        vc, vol_nm = db.con.execute(
            "SELECT voxel_count, volume_nm3 FROM bodies WHERE body_id=?", (body,)).fetchone()
        assert vc == n and vol_nm == n * 8 * 8 * 8


def test_keep_records_only_the_listed_bodies_but_records_them_in_full(tmp_path):
    """`keep` filters the recorded rows, not the blocks — so a kept body crossing a
    block boundary is still complete. Filtering blocks instead would truncate it."""
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100
    vol[20:28, 6:10, 6:10] = 200
    vol[12:20, 24:28, 24:28] = 300         # crosses the z-block boundary at 16
    src = str(tmp_path / "seg.zarr")
    _write_seg(src, vol)
    db_path = str(tmp_path / "m.sqlite")

    summary = index_segments({"backend": "zarr3", "path": src}, db_path,
                             scan_scale=0, scan_voxel_size=(8, 8, 8),
                             block_shape=(16, 16, 16), keep=[100, 300], client=None)
    assert summary["n_bodies"] == 2 and summary["n_kept"] == 2

    db = MetricsDB(db_path)
    assert db.con.execute("SELECT body_id FROM bodies WHERE body_id=200").fetchone() is None
    # 300 spans two blocks: both halves landed, so the filter did not become a crop
    assert db.get_bbox(300) == (12, 24, 24, 20, 28, 28)
    assert db.con.execute(
        "SELECT voxel_count FROM bodies WHERE body_id=300").fetchone()[0] == 8 * 4 * 4


def test_keep_none_records_everything(tmp_path):
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100
    vol[20:28, 6:10, 6:10] = 200
    src = str(tmp_path / "seg.zarr")
    _write_seg(src, vol)
    db_path = str(tmp_path / "m.sqlite")
    summary = index_segments({"backend": "zarr3", "path": src}, db_path,
                             scan_scale=0, scan_voxel_size=(8, 8, 8),
                             block_shape=(16, 16, 16), keep=None, client=None)
    assert summary["n_bodies"] == 2 and summary["n_kept"] is None


def test_index_fills_the_bbox_of_a_row_the_skel_stage_created(tmp_path):
    """Indexing AFTER skeletonizing must still record bboxes.

    The skel stage creates rows through update_body with no bbox, and SQLite's scalar
    `min(NULL, 5)` is NULL — so a bare min/max merge left those bodies with a NULL bbox
    forever while voxel_count accumulated correctly. Half-populated and silent.
    """
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100
    src = str(tmp_path / "seg.zarr")
    _write_seg(src, vol)
    db_path = str(tmp_path / "m.sqlite")

    db = MetricsDB(db_path)                      # stand in for the skel stage
    db.update_body(100, cable_length_nm=1234.0)
    assert db.get_bbox(100) == (None,) * 6       # the row exists with no bbox
    db.close()

    index_segments({"backend": "zarr3", "path": src}, db_path,
                   scan_scale=0, scan_voxel_size=(8, 8, 8),
                   block_shape=(16, 16, 16), client=None)

    db = MetricsDB(db_path)
    assert db.get_bbox(100) == (4, 4, 4, 12, 12, 12)
    vc, cable = db.con.execute(
        "SELECT voxel_count, cable_length_nm FROM bodies WHERE body_id=100").fetchone()
    assert vc == 512 and cable == 1234.0         # and the skel column survived


def test_index_resume(tmp_path):
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100
    src = str(tmp_path / "seg.zarr")
    _write_seg(src, vol)
    db_path = str(tmp_path / "m.sqlite")
    kw = dict(scan_scale=0, scan_voxel_size=(8, 8, 8), block_shape=(16, 16, 16), client=None)
    index_segments({"backend": "zarr3", "path": src}, db_path, **kw)
    # rerun (resume): all blocks already done -> counts unchanged (no double-count)
    index_segments({"backend": "zarr3", "path": src}, db_path, resume=True, **kw)
    assert MetricsDB(db_path).con.execute(
        "SELECT voxel_count FROM bodies WHERE body_id=100").fetchone()[0] == 512
