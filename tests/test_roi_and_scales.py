"""ROI block filtering and per-scale metadata.

The property that matters for ROI: a restricted run must be a *prefix* of the
full run. Same global grid, same block indices, same regions — so fragments and
manifest entries from a trial ROI are reused, not redone.
"""

import numpy as np
import pytest

from em_blockrun import iter_blocks

from em_seg_morpho import roi as R


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
def test_parse_roi_forms_and_validation():
    assert R.parse_roi("0,1,2,10,11,12") == (0, 1, 2, 10, 11, 12)
    assert R.parse_roi([0, 1, 2, 10, 11, 12]) == (0, 1, 2, 10, 11, 12)
    assert R.parse_roi(None) is None
    with pytest.raises(ValueError, match="6 values"):
        R.parse_roi("0,1,2,3")
    with pytest.raises(ValueError, match="half-open"):
        R.parse_roi("0,0,0,0,10,10")          # empty on z


def test_filter_keeps_global_block_identity():
    shape, block = (128, 128, 128), (32, 32, 32)
    full = list(iter_blocks(shape, block))
    roi = (32, 0, 0, 96, 64, 64)              # z blocks 1-2, y/x blocks 0-1
    kept = R.filter_blocks(full, roi)

    assert len(kept) == 2 * 2 * 2
    by_index = {b.index: b for b in full}
    for b in kept:
        assert b.index in by_index
        assert b.region == by_index[b.index].region   # regions identical, never clipped
    assert {b.index[0] for b in kept} == {1, 2}
    assert {b.index[1] for b in kept} == {0, 1}


def test_filter_none_returns_everything():
    full = list(iter_blocks((64, 64, 64), (32, 32, 32)))
    assert len(R.filter_blocks(full, None)) == len(full) == 8


def test_partially_overlapping_blocks_are_kept_whole():
    """A block straddling the ROI edge is included entire, not trimmed.

    Trimming would make a block's content depend on the ROI, so the same block
    index would mean different data in two runs and resume would reuse the wrong
    fragment.
    """
    full = list(iter_blocks((64, 64, 64), (32, 32, 32)))
    kept = R.filter_blocks(full, (30, 0, 0, 34, 64, 64))     # 2 voxels either side
    assert {b.index[0] for b in kept} == {0, 1}
    for b in kept:
        assert b.region[0].stop - b.region[0].start == 32


def test_clip_to_shape():
    assert R.clip_to_shape((0, 0, 0, 999, 999, 999), (64, 64, 64)) == (0, 0, 0, 64, 64, 64)
    assert R.clip_to_shape(None, (64, 64, 64)) is None
    with pytest.raises(ValueError, match="does not intersect"):
        R.clip_to_shape((100, 0, 0, 200, 10, 10), (64, 64, 64))


def test_scale_roi_covers_the_same_physical_region():
    # an ROI in scale-2 voxels, expressed in scale-0 voxels: 4x more of them
    assert R.scale_roi((10, 10, 10, 20, 20, 20), (4.0, 4.0, 4.0)) == (40, 40, 40, 80, 80, 80)
    # going coarser, starts floor and stops ceil so the region is still covered
    assert R.scale_roi((5, 5, 5, 11, 11, 11), (0.5, 0.5, 0.5)) == (2, 2, 2, 6, 6, 6)
    # anisotropic factors are handled per axis
    assert R.scale_roi((0, 0, 0, 8, 8, 8), (1.0, 2.0, 4.0)) == (0, 0, 0, 8, 16, 32)
    assert R.scale_roi(None, (2, 2, 2)) is None


# --------------------------------------------------------------------------- #
# ROI through the real ops
# --------------------------------------------------------------------------- #
def _write_seg_zarr(path, vol):
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _two_body_volume():
    vol = np.zeros((64, 32, 32), np.uint64)
    vol[4:28, 12:20, 12:20] = 7      # lives in z-blocks 0-1
    vol[36:60, 12:20, 12:20] = 9     # lives in z-blocks 2-3
    return vol


def test_roi_restricts_skeletonization_then_extends_on_resume(tmp_path):
    """The payoff: widening the ROI later reuses the first run's work."""
    from em_seg_morpho.config import OutputConfig, SkeletonConfig
    from em_seg_morpho.ops.skeletonize_segments import skeletonize_segments

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _two_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    cfg = SkeletonConfig(anisotropy=(8.0, 8.0, 8.0), block_shape=(16, 16, 16),
                         const=30.0, dust_threshold=0,
                         postprocess_dust_nm=0.0, postprocess_tick_nm=0.0)

    # grid is 4x2x2 = 16 blocks; the ROI covers z-blocks 0-1 -> 2*2*2 = 8
    first = skeletonize_segments(src, out, cfg, roi="0,0,0,32,32,32", client=None)
    assert first["num_blocks"] == 8
    import os
    assert os.path.exists(os.path.join(first["out_dir"], "7"))
    assert not os.path.exists(os.path.join(first["out_dir"], "9"))

    # widen to the whole volume: the first 8 blocks are already done, 8 remain
    done_after_first = set(first["chunk_counts"])
    assert sum(first["chunk_counts"].values()) == 8
    second = skeletonize_segments(src, out, cfg, client=None)
    assert second["num_blocks"] == 16
    assert sum(second["chunk_counts"].values()) == 16     # 8 reused + 8 new
    assert os.path.exists(os.path.join(second["out_dir"], "9"))


def test_a_manifest_outliving_its_output_is_refused_not_silently_skipped(tmp_path):
    """A manifest outliving its outputs would make the next run a silent no-op.

    dst may be an object store, so bookkeeping cannot live inside it and the two
    no longer share a fate: clearing dst leaves the manifest claiming everything
    is done. Resuming would then skip every task and report success having
    written nothing. The ops refuse instead — see
    ``ops/_progress.check_manifest_matches_output``.
    """
    import os
    import shutil

    import pytest

    from em_seg_morpho.config import MeshConfig, OutputConfig, SkeletonConfig
    from em_seg_morpho.ops._progress import StaleManifest
    from em_seg_morpho.ops.meshify import meshify
    from em_seg_morpho.ops.skeletonize_segments import skeletonize_segments

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _two_body_volume())
    work = str(tmp_path / "out")
    dst = str(tmp_path / "out" / "segmentation")
    out = OutputConfig(dst=dst, work_dir=work)
    mcfg = MeshConfig(mesh_scale=0, block_shape=(16, 16, 16), num_lods=2,
                      decimation_fraction=1.0)
    scfg = SkeletonConfig(anisotropy=(8.0, 8.0, 8.0), block_shape=(16, 16, 16),
                          const=30.0, dust_threshold=0,
                          postprocess_dust_nm=0.0, postprocess_tick_nm=0.0)

    m1 = meshify(src, out, mcfg, mesh_voxel_size=(8, 8, 8), client=None)
    s1 = skeletonize_segments(src, out, scfg, client=None)
    # Manifests live in the POSIX work dir, NOT in dst — dst may be an object
    # store, and it is the served volume, which bookkeeping must stay out of.
    for path in (m1["progress_path"], s1["progress_path"]):
        assert os.path.commonpath([os.path.abspath(path), os.path.abspath(work)]) == \
            os.path.abspath(work), f"{path} is not inside {work}"
        assert not os.path.abspath(path).startswith(os.path.abspath(dst) + os.sep), \
            f"{path} is inside the served volume {dst}"
    assert os.path.exists(os.path.join(m1["out_dir"], "7"))
    assert os.path.exists(os.path.join(s1["out_dir"], "7"))

    # The data is cleared but the manifest in work_dir survives.
    shutil.rmtree(dst)
    for op in (lambda: meshify(src, out, mcfg, mesh_voxel_size=(8, 8, 8), client=None),
               lambda: skeletonize_segments(src, out, scfg, client=None)):
        with pytest.raises(StaleManifest, match="no 'info' at"):
            op()

    # resume=False is an explicit fresh start, so it proceeds and rewrites.
    m2 = meshify(src, out, mcfg, mesh_voxel_size=(8, 8, 8), client=None, resume=False)
    s2 = skeletonize_segments(src, out, scfg, client=None, resume=False)
    assert m2["num_bodies_assembled"] > 0 and s2["num_bodies_fused"] > 0
    assert os.path.exists(os.path.join(m2["out_dir"], "7"))
    assert os.path.exists(os.path.join(s2["out_dir"], "7"))


def test_guard_allows_a_first_run_and_an_ordinary_resume(tmp_path):
    """The guard must be silent in both normal cases, or it is useless.

    An empty manifest has nothing to be stale about, and a resume whose output is
    still present is the common walltime-recovery path.
    """
    import os

    from em_seg_morpho.config import MeshConfig, OutputConfig
    from em_seg_morpho.ops.meshify import meshify

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _two_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"),
                       work_dir=str(tmp_path / "out"))
    mcfg = MeshConfig(mesh_scale=0, block_shape=(16, 16, 16), num_lods=2,
                      decimation_fraction=1.0)

    m1 = meshify(src, out, mcfg, mesh_voxel_size=(8, 8, 8), client=None)  # first run
    assert m1["num_bodies_assembled"] > 0
    m2 = meshify(src, out, mcfg, mesh_voxel_size=(8, 8, 8), client=None)  # resume
    assert os.path.exists(os.path.join(m2["out_dir"], "7"))


def test_roi_restricts_index_scan(tmp_path):
    from em_seg_morpho.metrics_db import MetricsDB
    from em_seg_morpho.ops.index_segments import index_segments

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _two_body_volume())
    db_path = str(tmp_path / "m.db")
    summary = index_segments(src, db_path, scan_voxel_size=(8, 8, 8), scan_scale=0,
                             block_shape=(16, 16, 16), roi="0,0,0,32,32,32", client=None)
    assert summary["n_blocks"] == 8            # 4x2x2 grid, z-blocks 0-1 kept
    db = MetricsDB(db_path)
    bodies = {r[0] for r in db.con.execute("SELECT body_id FROM bodies")}
    db.close()
    assert bodies == {7}


# --------------------------------------------------------------------------- #
# Scale metadata
# --------------------------------------------------------------------------- #
def test_read_scales_from_precomputed_info(tmp_path):
    """Voxel size must come from each scale's own metadata, not 2**index.

    The pyramid here downsamples 2x in x/y but never in z — the exact shape that
    a 2**scale assumption gets wrong.
    """
    import json

    from em_seg_morpho.scales import read_scales, scale_spec

    root = tmp_path / "seg.precomputed"
    root.mkdir()
    info = {
        "@type": "neuroglancer_multiscale_volume", "type": "segmentation",
        "data_type": "uint64", "num_channels": 1,
        "scales": [
            {"key": "8_8_40", "resolution": [8, 8, 40], "size": [1024, 1024, 100],
             "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
            {"key": "16_16_40", "resolution": [16, 16, 40], "size": [512, 512, 100],
             "chunk_sizes": [[64, 64, 64]], "encoding": "raw"},
        ],
    }
    (root / "info").write_text(json.dumps(info))

    scales = read_scales(str(root))
    assert len(scales) == 2
    # resolution/size are xyz in the file, zyx in ScaleInfo
    assert scales[0].voxel_size == (40.0, 8.0, 8.0) and scales[0].shape == (100, 1024, 1024)
    assert scales[1].voxel_size == (40.0, 16.0, 16.0) and scales[1].shape == (100, 512, 512)

    # the real factor is (1, 2, 2) — NOT (2, 2, 2) as 2**index would give
    assert scales[1].factor_from(scales[0]) == (1.0, 2.0, 2.0)

    spec = scale_spec(str(root), 1)
    assert spec["backend"] == "neuroglancer_precomputed" and spec["scale_index"] == 1


def test_read_scales_rejects_sources_without_metadata(tmp_path):
    from em_seg_morpho.scales import read_scales

    bare = tmp_path / "nothing"
    bare.mkdir()
    with pytest.raises(ValueError):
        read_scales(str(bare))
