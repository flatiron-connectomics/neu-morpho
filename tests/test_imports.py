"""Scaffold smoke tests: modules import and the toolchain is present."""


def test_package_modules_import():
    import neu_morpho
    from neu_morpho import config, mesh, precomputed, skeleton, occupancy, fragments, allowlist
    from neu_morpho.ops import meshify
    assert neu_morpho.__version__
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
    import blockrun, neu_vol  # noqa: F401
    from blockrun import block_map, Manifest, iter_blocks  # noqa: F401
    assert blockrun.__version__ and neu_vol.__version__
