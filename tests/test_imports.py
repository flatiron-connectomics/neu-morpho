"""Scaffold smoke tests: modules import and the toolchain (vol2mesh multires,
kimimaro, em-blockrun, em-volume-tools) is present."""


def test_package_modules_import():
    import em_seg_morpho
    from em_seg_morpho import config, mesh, precomputed, skeleton, segments
    from em_seg_morpho.ops import mesh_segments
    assert em_seg_morpho.__version__
    assert config.MeshConfig().start_lod == 2   # default meshing scale (not 0)


def test_toolchain_available():
    import vol2mesh
    from vol2mesh import Mesh, multires          # multires requires DracoPy
    import kimimaro
    import DracoPy
    # the chunked-mesh + multires-writer primitives we depend on:
    assert hasattr(Mesh, "from_binary_vol")
    assert hasattr(Mesh, "from_binary_blocks")
    for fn in ("write_info", "write_object_mesh", "split_mesh_for_lod"):
        assert hasattr(multires, fn), fn
    assert hasattr(kimimaro, "skeletonize")


def test_shared_packages_available():
    import em_blockrun
    import em_volume_tools
    from em_blockrun import block_map, Manifest, start_dask  # noqa: F401
    assert em_blockrun.__version__ and em_volume_tools.__version__


def test_mesh_config_defaults():
    from em_seg_morpho.config import MeshConfig, SkeletonConfig
    m = MeshConfig()
    assert m.max_mask_voxels == 512 ** 3
    assert SkeletonConfig().anisotropy == (8.0, 8.0, 8.0)
