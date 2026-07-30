"""Draco quantization vs. voxel size — a real-data failure, now caught up front.

On the first production ROI (scale 2, 32 nm voxels, 256-blocks) the shipped
default of 10 bits made the LOD-2 quantization step exactly 32 nm — the voxel
size — and 29% of bodies failed to encode with "All triangles are degenerate".
The step is a *relationship* between chunk size, LOD count and bit depth, so the
guard computes it rather than trusting a constant.
"""

import numpy as np
import pytest

from em_seg_morpho.config import MeshConfig, OutputConfig
from em_seg_morpho.precomputed import (VALID_QUANTIZATION_BITS, check_quantization,
                                       quantization_step_nm)


def test_step_doubles_per_lod():
    """A LOD-l cell is chunk * 2**l wide, so the step coarsens with depth."""
    cfg = MeshConfig(num_lods=1, draco_quantization_bits=10)
    assert quantization_step_nm(cfg, [8192, 8192, 8192]) == pytest.approx(8.0)
    assert quantization_step_nm(MeshConfig(num_lods=2, draco_quantization_bits=10),
                                [8192] * 3) == pytest.approx(16.0)
    assert quantization_step_nm(MeshConfig(num_lods=3, draco_quantization_bits=10),
                                [8192] * 3) == pytest.approx(32.0)
    # 16 bits buys 64x headroom at the same depth
    assert quantization_step_nm(MeshConfig(num_lods=3, draco_quantization_bits=16),
                                [8192] * 3) == pytest.approx(0.5)


def test_rejects_the_configuration_that_failed_on_real_data():
    """scale 2 (32 nm), 256-blocks, 3 LODs, 10 bits — the 29%-failure setup."""
    cfg = MeshConfig(block_shape=(256, 256, 256), num_lods=3, draco_quantization_bits=10)
    with pytest.raises(ValueError, match="too close to"):
        check_quantization(cfg, [8192, 8192, 8192], (32.0, 32.0, 32.0))


def test_accepts_the_shipped_default():
    cfg = MeshConfig(block_shape=(256, 256, 256), num_lods=3)
    assert cfg.draco_quantization_bits == 16
    check_quantization(cfg, [8192, 8192, 8192], (32.0, 32.0, 32.0))     # no raise


def test_rejects_bits_the_neuroglancer_spec_forbids():
    """12 and 14 encode fine in Draco but neuroglancer cannot read them."""
    assert VALID_QUANTIZATION_BITS == (10, 16)
    for bits in (8, 12, 14, 32):
        with pytest.raises(ValueError, match="must be one of"):
            check_quantization(MeshConfig(draco_quantization_bits=bits),
                               [8192] * 3, (32.0,) * 3)


def test_deep_pyramids_are_caught_even_at_16_bits():
    """The guard is about the relationship, not about 10 bits specifically."""
    cfg = MeshConfig(block_shape=(512, 512, 512), num_lods=9, draco_quantization_bits=16)
    with pytest.raises(ValueError, match="Raise draco_quantization_bits"):
        check_quantization(cfg, [512 * 32.0] * 3, (32.0, 32.0, 32.0))


def test_meshify_checks_before_doing_any_work(tmp_path):
    """The guard must fire up front, not after an hour of block meshing."""
    from em_seg_morpho.ops.meshify import meshify
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:28, 4:28, 4:28] = 1
    src = str(tmp_path / "seg.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)

    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    # chunk = 16 * 32 = 512 nm; 7 LODs -> coarsest cell 32768 nm; 10 bits -> 32 nm
    # step, equal to the voxel size — the real-data failure, in miniature.
    cfg = MeshConfig(mesh_scale=0, block_shape=(16, 16, 16), num_lods=7,
                     draco_quantization_bits=10)
    assert quantization_step_nm(cfg, [512.0] * 3) == pytest.approx(32.0)
    with pytest.raises(ValueError, match="too close to"):
        meshify({"backend": "zarr3", "path": src}, out, cfg,
                mesh_voxel_size=(32, 32, 32), client=None)

    # nothing was written — not even the info file
    assert not (tmp_path / "out" / "chunked").exists()
