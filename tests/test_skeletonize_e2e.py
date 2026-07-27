"""End-to-end: synthetic volume -> skel-chunk -> skel-fuse -> precomputed skeletons.

The load-bearing case is a body spanning several blocks: its fragments must weld
into ONE component covering the full extent (that is what ``fix_borders`` +
``join_close_components`` buy us), not a pile of disconnected block-length stubs.
"""

import os

import numpy as np

from em_seg_morpho.config import OutputConfig, SkeletonConfig
from em_seg_morpho.ops.skeletonize_segments import skeletonize_segments


def _write_seg_zarr(path, vol):
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _load(out_dir, body):
    from osteoid import Skeleton
    with open(os.path.join(out_dir, str(body)), "rb") as f:
        return Skeleton.from_precomputed(f.read(), segid=body)


VS = (8.0, 8.0, 8.0)


def _volume():
    # 96 x 32 x 32 at 8 nm, blocks of 32 along z -> 3 blocks
    vol = np.zeros((96, 32, 32), np.uint64)
    vol[:, 14:18, 14:18] = 7          # rod spanning all 3 blocks
    vol[40:48, 4:9, 4:9] = 9          # blob wholly inside block 1
    return vol


def _cfg():
    return SkeletonConfig(anisotropy=VS, skeleton_scale=0, block_shape=(32, 32, 32),
                          const=30.0, dust_threshold=0,
                          postprocess_dust_nm=0.0, postprocess_tick_nm=0.0)


def test_skeletonize_end_to_end(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))
    db_path = str(tmp_path / "metrics.db")

    summary = skeletonize_segments(src, out, _cfg(), db_path=db_path, client=None)
    out_dir = summary["out_dir"]

    assert os.path.exists(os.path.join(out_dir, "info"))
    assert summary["num_blocks"] == 3
    assert summary["status_counts"].get("written") == 2

    # the spanning body produced one fragment per block...
    assert len(os.listdir(os.path.join(summary["chunked_dir"], "7"))) == 3
    assert len(os.listdir(os.path.join(summary["chunked_dir"], "9"))) == 1

    # ...and fused into a single connected skeleton over the full z extent.
    rod = _load(out_dir, 7)
    assert len(rod.components()) == 1
    z = rod.vertices[:, 2]                      # stored xyz -> z is column 2
    assert z.min() <= 8 and z.max() >= 744      # 95 voxels * 8 nm = 760
    assert rod.cable_length() > 700

    blob = _load(out_dir, 9)
    bz = blob.vertices[:, 2]
    assert 300 <= bz.min() and bz.max() <= 400  # block 1: z voxels 40..48 -> 320..376 nm


def test_metrics_land_in_db(tmp_path):
    from em_seg_morpho.metrics_db import MetricsDB

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))
    db_path = str(tmp_path / "metrics.db")
    skeletonize_segments(src, out, _cfg(), db_path=db_path, client=None)

    db = MetricsDB(db_path)
    row = db.con.execute(
        "SELECT cable_length_nm, n_branches, n_tips FROM bodies WHERE body_id=7").fetchone()
    db.close()
    assert row is not None
    cable, branches, tips = row
    assert cable > 700                          # the FUSED length, not one block's
    assert branches == 0 and tips == 2          # an unbranched rod


def test_allowlist_restricts_bodies(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))
    summary = skeletonize_segments(src, out, _cfg(), allowlist=[7], client=None)

    assert os.path.exists(os.path.join(summary["out_dir"], "7"))
    assert not os.path.exists(os.path.join(summary["out_dir"], "9"))
    assert not os.path.exists(os.path.join(summary["chunked_dir"], "9"))


def test_resume_skips_done_and_stage2_reuses_fragments(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))

    first = skeletonize_segments(src, out, _cfg(), stages=("skel-chunk",), client=None)
    assert first["chunk_counts"].get("written") == 3
    assert not os.path.exists(os.path.join(first["out_dir"], "7"))

    # stage 2 alone reuses the fragments already on disk
    second = skeletonize_segments(src, out, _cfg(), stages=("skel-fuse",), client=None)
    assert second["num_bodies_fused"] == 2
    assert os.path.exists(os.path.join(second["out_dir"], "7"))

    # a third full run has nothing left to do
    third = skeletonize_segments(src, out, _cfg(), client=None)
    assert third["num_bodies_fused"] == 0


def test_fuse_body_welds_seams_not_just_concatenates():
    """Guard the property the whole design rests on: fragments become one tree."""
    import kimimaro

    from em_seg_morpho.coords import crop_origin_nm
    from em_seg_morpho.skeleton import fuse_body, skeletonize_block

    cfg = _cfg()
    vol = _volume()
    frags = [skeletonize_block(vol[z0:z0 + 32], (z0, 0, 0), cfg)[7] for z0 in (0, 32, 64)]
    assert len(frags) == 3

    fused = fuse_body(frags, cfg, body_id=7)
    assert fused.id == 7
    assert len(fused.components()) == 1
    # welding adds the seam edges, so the fused cable exceeds the sum of the parts
    assert fused.cable_length() > sum(f.cable_length() for f in frags)
