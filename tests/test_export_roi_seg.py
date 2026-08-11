"""Exporting the ROI's labels as a precomputed volume for inspection.

The load-bearing property is the **offset**: the copy must declare where it sits
in the source volume, or neuroglancer draws it at the origin while the meshes and
skeletons sit tens of microns away — the alignment bug this package exists to
avoid, reintroduced at the viewing layer.
"""

import json
import os

import numpy as np
import pytest

from em_seg_morpho.ops.export_roi_seg import block_align, export_roi_seg


def test_block_align_expands_to_whole_blocks_and_clips():
    # matches what stage 1 meshed: intersecting blocks are kept whole
    assert block_align((406, 675, 1133, 819, 1742, 1407), (256, 256, 256),
                       (2815, 2250, 3438)) == (256, 512, 1024, 1024, 1792, 1536)
    # already aligned -> unchanged
    assert block_align((256, 0, 0, 512, 256, 256), (256, 256, 256),
                       (1024, 1024, 1024)) == (256, 0, 0, 512, 256, 256)
    # clipped to the volume rather than running off the end
    assert block_align((0, 0, 0, 100, 100, 100), (256, 256, 256),
                       (200, 200, 200)) == (0, 0, 0, 200, 200, 200)


def _write_seg(path, vol, voxel=(32.0, 32.0, 32.0)):
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(32, 32, 32)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def test_export_carries_the_voxel_offset(tmp_path):
    vol = np.zeros((128, 128, 128), np.uint64)
    vol[70:90, 70:90, 70:90] = 5
    src = _write_seg(str(tmp_path / "seg.zarr"), vol)
    out = str(tmp_path / "segmentation")

    summary = export_roi_seg(src, out, roi=(64, 64, 64, 128, 128, 128),
                             roi_voxel_size=(32.0, 32.0, 32.0), block_shape=(64, 64, 64),
                             scale_indices=[0], encoding="raw", client=None)
    assert summary["scales"][0]["shape"] == (64, 64, 64)

    info = json.load(open(os.path.join(out, "info")))
    assert info["@type"] == "neuroglancer_multiscale_volume"
    assert info["type"] == "segmentation" and info["data_type"] == "uint64"
    s0 = info["scales"][0]
    assert s0["size"] == [64, 64, 64]                       # xyz
    assert s0["voxel_offset"] == [64, 64, 64]               # <-- the whole point
    assert s0["resolution"] == [32.0, 32.0, 32.0]

    # a real chunk tree, named x_y_z, not just an info file
    files = os.listdir(os.path.join(out, s0["key"]))
    assert files and all(len(f.split("_")) == 3 for f in files)


def test_exported_labels_land_at_the_same_nm_as_the_source(tmp_path):
    """Read the copy back through its own offset and confirm the labels coincide."""
    vol = np.zeros((128, 128, 128), np.uint64)
    vol[70:90, 70:90, 70:90] = 5
    src = _write_seg(str(tmp_path / "seg.zarr"), vol)
    out = str(tmp_path / "segmentation")
    export_roi_seg(src, out, roi=(64, 64, 64, 128, 128, 128), roi_voxel_size=(32.0, 32.0, 32.0),
                   block_shape=(64, 64, 64), scale_indices=[0], encoding="raw", client=None)

    from em_volume_tools.backends.base import open_backend
    be = open_backend({"backend": "neuroglancer_precomputed", "path": out, "scale_index": 0})
    got = be.read_region((slice(0, 64), slice(0, 64), slice(0, 64)))
    # the copy's local [0:64] is the source's [64:128]
    np.testing.assert_array_equal(got, vol[64:128, 64:128, 64:128])
    assert got[6:26, 6:26, 6:26].min() == 5      # the cube, at its shifted position


def _volume_with_subresources(tmp_path):
    """A precomputed segmentation volume with mesh/ and skeleton/ inside it."""
    import numpy as np

    from em_seg_morpho.config import MeshConfig
    from em_seg_morpho.precomputed import (write_body_skeleton, write_mesh_info,
                                           write_skeleton_info)

    vol = np.zeros((128, 128, 128), np.uint64)
    vol[70:90, 70:90, 70:90] = 5
    src = _write_seg(str(tmp_path / "seg.zarr"), vol)
    out = str(tmp_path / "segmentation")
    export_roi_seg(src, out, roi=(64, 64, 64, 128, 128, 128), roi_voxel_size=(32.0,) * 3,
                   block_shape=(64, 64, 64), scale_indices=[0], encoding="raw", client=None)
    write_mesh_info(os.path.join(out, "mesh"), MeshConfig())

    from osteoid import Skeleton
    write_skeleton_info(os.path.join(out, "skeleton"))
    s = Skeleton(vertices=np.array([[0.0, 0, 0], [8, 8, 8]], np.float32),
                 edges=np.array([[0, 1]], np.uint32), segid=5)
    write_body_skeleton(os.path.join(out, "skeleton"), 5, s)
    return out


def test_link_subresources_sets_the_spec_keys(tmp_path):
    from em_seg_morpho.precomputed import link_subresources

    out = _volume_with_subresources(tmp_path)
    assert link_subresources(out, mesh="mesh", skeletons="skeleton") == {
        "mesh": "mesh", "skeletons": "skeleton"}

    info = json.load(open(os.path.join(out, "info")))
    assert info["mesh"] == "mesh" and info["skeletons"] == "skeleton"
    assert info["type"] == "segmentation"          # untouched
    assert len(info["scales"]) == 1
    # the value names a subdirectory of the volume root, per the precomputed spec
    assert os.path.isdir(os.path.join(out, info["mesh"]))
    assert os.path.isdir(os.path.join(out, info["skeletons"]))


def test_link_subresources_rejects_a_wrong_or_missing_target(tmp_path):
    """Pointing a volume at the wrong directory fails silently in the viewer."""
    from em_seg_morpho.precomputed import link_subresources

    out = _volume_with_subresources(tmp_path)
    # existence is tested by reading <sub>/info, so object stores (no directories)
    # take the same path as local files
    with pytest.raises(FileNotFoundError, match="does not exist"):
        link_subresources(out, mesh="nope")
    # mesh and skeleton dirs swapped -> caught by @type
    with pytest.raises(ValueError, match="expected 'neuroglancer_multilod_draco'"):
        link_subresources(out, mesh="skeleton")
    with pytest.raises(ValueError, match="expected 'neuroglancer_skeletons'"):
        link_subresources(out, skeletons="mesh")


def test_link_subresources_refuses_an_image_volume(tmp_path):
    import numpy as np

    from em_seg_morpho.precomputed import link_subresources

    out = _volume_with_subresources(tmp_path)
    path = os.path.join(out, "info")
    info = json.load(open(path))
    info["type"] = "image"
    json.dump(info, open(path, "w"))
    with pytest.raises(ValueError, match="not a segmentation volume"):
        link_subresources(out, mesh="mesh")


def test_ops_write_inside_the_volume():
    """mesh_dir/skeleton_dir are subdirectories of the VOLUME; bookkeeping is not."""
    from em_seg_morpho.config import OutputConfig

    out = OutputConfig(dst="/data/run/segmentation", work_dir="/scratch/run")
    assert out.volume_dir() == "/data/run/segmentation"
    assert out.mesh_out() == "/data/run/segmentation/mesh"
    assert out.skeleton_out() == "/data/run/segmentation/skeleton"
    # bookkeeping goes to the work dir, never into the served volume
    assert out.work("progress.mesh.jsonl") == "/scratch/run/progress.mesh.jsonl"
    assert not out.work("chunked").startswith(out.volume_dir())


def test_dst_may_be_remote_but_work_dir_may_not():
    """Only the data destination can be an object store."""
    from em_seg_morpho.config import OutputConfig

    remote = OutputConfig(dst="s3://bucket/sample3/segmentation", work_dir="/scratch/run")
    assert remote.volume_dir() == "s3://bucket/sample3/segmentation"
    assert remote.mesh_out() == "s3://bucket/sample3/segmentation/mesh"
    remote.check_work_dir_is_local()          # local work dir: fine

    # sqlite, appended JSONL and the fragment store all need a filesystem
    bad = OutputConfig(dst="s3://bucket/vol", work_dir="s3://bucket/work")
    with pytest.raises(ValueError, match="work_dir must be a filesystem path"):
        bad.check_work_dir_is_local()
    with pytest.raises(ValueError, match="work_dir is required"):
        OutputConfig(dst="/data/vol").check_work_dir_is_local()


def test_align_can_be_turned_off(tmp_path):
    vol = np.zeros((128, 128, 128), np.uint64)
    vol[70:90, 70:90, 70:90] = 5
    src = _write_seg(str(tmp_path / "seg.zarr"), vol)
    summary = export_roi_seg(src, str(tmp_path / "seg2"), roi=(70, 70, 70, 90, 90, 90),
                             roi_voxel_size=(32.0,) * 3, block_shape=(64, 64, 64),
                             align_to_blocks=False, scale_indices=[0], encoding="raw",
                             client=None)
    assert summary["scales"][0]["shape"] == (20, 20, 20)
    assert summary["n_voxels_total"] == 20 ** 3


def _write_pyramid(tmp_path, levels=3):
    """A precomputed source with `levels` scales, 2x isotropic, distinct labels."""
    import numpy as np
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import precomputed_create_spec

    path = str(tmp_path / "src.precomputed")
    full = np.zeros((64, 64, 64), np.uint64)
    full[16:48, 16:48, 16:48] = 5
    full[8:12, 8:12, 8:12] = 9
    for i in range(levels):
        vol = full[:: 2 ** i, :: 2 ** i, :: 2 ** i]
        be = TensorStoreBackend.create(
            precomputed_create_spec("s3-neuroglancer", path, vol.shape, "uint64",
                                    resolution_zyx=(8.0 * 2 ** i,) * 3, scale_index=i,
                                    type_="segmentation", chunk=(16, 16, 16),
                                    encoding="raw"),
            delete_existing=(i == 0))
        be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return path


def test_export_copies_every_scale_with_scaled_offsets(tmp_path):
    """All source levels are copied, and each carries its own correct offset.

    The offsets must scale with resolution: get that wrong and coarse levels sit
    at the wrong place, which looks like a rendering glitch when you zoom out.
    """
    from em_volume_tools.backends.base import open_backend

    src = _write_pyramid(tmp_path, levels=3)
    out = str(tmp_path / "segmentation")
    summary = export_roi_seg(src, out, roi=(16, 16, 16, 48, 48, 48),
                             roi_voxel_size=(8.0, 8.0, 8.0), block_shape=(16, 16, 16),
                             encoding="raw", client=None)
    assert [w["scale"] for w in summary["scales"]] == [0, 1, 2]

    info = json.load(open(os.path.join(out, "info")))
    assert len(info["scales"]) == 3
    for i, sc in enumerate(info["scales"]):
        assert sc["resolution"] == [8.0 * 2 ** i] * 3
        assert sc["voxel_offset"] == [16 // 2 ** i] * 3          # offset scales with res
        assert sc["size"] == [32 // 2 ** i] * 3
        # and the labels match the source level, read through the offset
        dst = open_backend({"backend": "neuroglancer_precomputed", "path": out,
                            "scale_index": i})
        got = dst.read_region(tuple(slice(0, x) for x in dst.shape))
        s_be = open_backend({"backend": "neuroglancer_precomputed", "path": src,
                             "scale_index": i})
        o = 16 // 2 ** i
        exp = s_be.read_region(tuple(slice(o, o + n) for n in got.shape))
        np.testing.assert_array_equal(got, exp)


def test_scale_indices_restricts_the_export(tmp_path):
    """A whole-volume run must be able to skip scale 0, which dominates the cost."""
    src = _write_pyramid(tmp_path, levels=3)
    out = str(tmp_path / "segmentation")
    summary = export_roi_seg(src, out, roi=(16, 16, 16, 48, 48, 48),
                             roi_voxel_size=(8.0,) * 3, block_shape=(16, 16, 16),
                             scale_indices=[1, 2], encoding="raw", client=None)
    assert [w["scale"] for w in summary["scales"]] == [1, 2]
    info = json.load(open(os.path.join(out, "info")))
    assert [s["resolution"][0] for s in info["scales"]] == [16.0, 32.0]


def test_scale_cost_reports_the_expensive_level(tmp_path):
    from em_seg_morpho.ops.export_roi_seg import scale_cost

    src = _write_pyramid(tmp_path, levels=3)
    cost = scale_cost(src, (16, 16, 16, 48, 48, 48), (8.0,) * 3, block_shape=(16, 16, 16))
    assert [c["scale"] for c in cost] == [0, 1, 2]
    assert cost[0]["n_voxels"] == 32 ** 3
    # each coarser level is 8x cheaper, so level 0 dominates
    assert cost[0]["n_voxels"] == 8 * cost[1]["n_voxels"] == 64 * cost[2]["n_voxels"]


def _one_block_and_specs(tmp_path):
    """A real source, a real precomputed destination, and one block to copy."""
    from em_blockrun import iter_blocks

    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import precomputed_create_spec

    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 3
    src = _write_seg(str(tmp_path / "seg.zarr"), vol)
    dst_dir = str(tmp_path / "out")
    dst_spec = precomputed_create_spec("s3-neuroglancer", dst_dir, (32, 32, 32), "uint64",
                                       resolution_zyx=(32.0,) * 3, scale_index=0,
                                       type_="segmentation", encoding="raw",
                                       voxel_offset_zyx=(0, 0, 0))
    dst = TensorStoreBackend.open_or_create(dst_spec, resume=False,
                                            delete_existing=True).to_spec()
    return src, dst, list(iter_blocks((32, 32, 32), (32, 32, 32)))[0]


def _flaky_open(monkeypatch, error: str, fail_on: int):
    """Wrap open_backend so the ``fail_on``-th call raises ``error``."""
    from em_volume_tools.backends import base as be_base

    calls = {"n": 0}
    original = be_base.open_backend

    def wrapper(spec):
        calls["n"] += 1
        if calls["n"] == fail_on:
            raise ValueError(error)
        return original(spec)

    monkeypatch.setattr(be_base, "open_backend", wrapper)
    return calls


def test_copy_block_survives_a_transient_store_failure(tmp_path, monkeypatch):
    """A TLS reset mid-copy must not end a run that has already copied terabytes.

    Stage-1 blocks are fail-fast by design, but a copy block feeds no aggregation
    and rewriting the same voxels is idempotent, so it retries instead.
    """
    from em_seg_morpho.ops.export_roi_seg import _copy_block

    # What is under test is that the copy retries at all, not how long it waits —
    # the backoff schedule itself is em-volume-tools' test_retry. Left real, the
    # default 1 s base delay is spent sleeping in the one test that reaches it.
    monkeypatch.setattr("em_volume_tools.retry.time.sleep", lambda *_: None)
    src, dst, block = _one_block_and_specs(tmp_path)
    calls = _flaky_open(monkeypatch,
                        "UNAVAILABLE: CURL error SSL connect error: Recv failure: "
                        "Connection reset by peer [curl_code='35']", fail_on=1)

    idx, status = _copy_block(block, src_spec=src, dst_spec=dst, src_origin=(0, 0, 0),
                              attempts=3)
    assert status == "written"
    assert calls["n"] > 1, "the transient failure was never triggered"


def test_a_403_during_copy_is_not_retried(tmp_path, monkeypatch):
    """Permissions will not fix themselves; retrying just burns the budget."""
    from em_seg_morpho.ops.export_roi_seg import _copy_block

    src, dst, block = _one_block_and_specs(tmp_path)
    calls = _flaky_open(monkeypatch,
                        "PERMISSION_DENIED: AccessDenied [http_response_code='403']",
                        fail_on=1)

    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        _copy_block(block, src_spec=src, dst_spec=dst, src_origin=(0, 0, 0), attempts=5)
    assert calls["n"] == 1, "a 403 must fail on the first attempt, with no retries"


def test_empty_blocks_write_no_objects(tmp_path):
    """'empty' describes the DATA, not the action — TensorStore elides fill chunks.

    So an all-zero block creates no object at all, and neuroglancer reads the
    absent chunk as zero. This is why there is no point skipping the write; the
    cost worth avoiding is the READ.
    """
    import os

    vol = np.zeros((64, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 3            # only the first 32-block has labels
    src = _write_seg(str(tmp_path / "seg.zarr"), vol)
    out = str(tmp_path / "out")

    summary = export_roi_seg(src, out, roi=None, roi_voxel_size=(32.0, 32.0, 32.0),
                            block_shape=(32, 32, 32), copy_block=(32, 32, 32),
                            scale_indices=[0], encoding="raw", client=None,
                            progress_path=str(tmp_path / "p.jsonl"))
    counts = summary["scales"][0]["counts"]
    assert counts.get("empty", 0) >= 1 and counts.get("written", 0) >= 1

    objects = [f for _, _, files in os.walk(out) for f in files if f != "info"]
    assert len(objects) == counts["written"], \
        f"{len(objects)} objects for {counts['written']} written + {counts.get('empty')} empty"
