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


def test_measure_does_not_cost_pandas_at_import():
    """`measure`'s deps are an optional extra, so importing the package — which every
    CLI invocation does — must not pay for them. pandas is imported inside the functions
    that need it; `metrics.py` needs numpy only.

    Asserted in a fresh interpreter, because the regression is one convenience import at
    module scope away and nothing else would notice.
    """
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", """
import sys
before = set(sys.modules)
import neu_morpho, neu_morpho.measure
added = {m.split('.')[0] for m in set(sys.modules) - before}
assert 'pandas' not in added, 'importing neu_morpho.measure pulled in pandas'
assert 'pyarrow' not in added, 'importing neu_morpho.measure pulled in pyarrow'
print('clean')
"""], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "clean"
