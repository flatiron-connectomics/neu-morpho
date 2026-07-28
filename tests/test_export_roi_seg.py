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
                             voxel_size=(32.0, 32.0, 32.0), block_shape=(64, 64, 64),
                             multiscale=False, encoding="raw", client=None)
    assert summary["region"] == (64, 64, 64, 128, 128, 128)

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
    export_roi_seg(src, out, roi=(64, 64, 64, 128, 128, 128), voxel_size=(32.0, 32.0, 32.0),
                   block_shape=(64, 64, 64), multiscale=False, encoding="raw", client=None)

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
    export_roi_seg(src, out, roi=(64, 64, 64, 128, 128, 128), voxel_size=(32.0,) * 3,
                   block_shape=(64, 64, 64), multiscale=False, encoding="raw", client=None)
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
    with pytest.raises(FileNotFoundError, match="not a directory"):
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


def test_ops_write_inside_the_volume(tmp_path):
    """mesh_dir/skeleton_dir are subdirectories of the VOLUME, not of dst."""
    from em_seg_morpho.config import OutputConfig

    out = OutputConfig(dst="/data/run")
    assert out.volume_dir() == "/data/run/segmentation"
    # bookkeeping stays outside the volume so the volume stays servable
    assert not out.volume_dir().startswith("/data/run/segmentation/progress")


def test_align_can_be_turned_off(tmp_path):
    vol = np.zeros((128, 128, 128), np.uint64)
    vol[70:90, 70:90, 70:90] = 5
    src = _write_seg(str(tmp_path / "seg.zarr"), vol)
    summary = export_roi_seg(src, str(tmp_path / "seg2"), roi=(70, 70, 70, 90, 90, 90),
                             voxel_size=(32.0,) * 3, block_shape=(64, 64, 64),
                             align_to_blocks=False, multiscale=False, encoding="raw",
                             client=None)
    assert summary["region"] == (70, 70, 70, 90, 90, 90)
    assert summary["n_voxels"] == 20 ** 3
