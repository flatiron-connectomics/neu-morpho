"""Scaffold smoke tests: modules import and the toolchain is present."""


def test_package_modules_import():
    import em_seg_morpho
    from em_seg_morpho import config, mesh, precomputed, skeleton, occupancy, fragments, allowlist
    from em_seg_morpho.ops import meshify
    assert em_seg_morpho.__version__
    assert config.MeshConfig().mesh_scale == 2      # default meshing scale (not 0)
    assert callable(meshify)


def test_toolchain_block_first_primitives():
    from vol2mesh import Mesh, multires, concatenate_meshes
    import kimimaro, DracoPy  # noqa: F401
    # stage-1 (all labels per block) + stage-2 (assemble) primitives:
    assert hasattr(Mesh, "from_label_volume")
    assert hasattr(Mesh, "stitch_adjacent_faces")
    assert callable(concatenate_meshes)
    for fn in ("write_info", "write_object_mesh", "split_mesh_for_lod"):
        assert hasattr(multires, fn), fn


def test_shared_packages_available():
    import em_blockrun, em_volume_tools  # noqa: F401
    from em_blockrun import block_map, Manifest, iter_blocks  # noqa: F401
    assert em_blockrun.__version__ and em_volume_tools.__version__
