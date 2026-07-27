"""End-to-end index scan: synthetic segmentation -> exact per-body bbox + count."""

import numpy as np

from em_seg_morpho.metrics_db import MetricsDB
from em_seg_morpho.ops.index_segments import index_segments


def _write_seg(path, vol):
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec
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
