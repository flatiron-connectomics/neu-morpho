"""Reading a published volume back.

The round-trip tests are the reason this module exists in the package rather than in
a script: ``precomputed.py`` writes both formats and nothing else verified that what
it writes can be read back as what went in. A writer-only suite cannot see that,
because a wrong-but-consistent encoding passes every check it makes.
"""

import numpy as np
import pytest

from neu_morpho.config import MeshConfig
from neu_morpho.readback import read_body_mesh, read_body_skeleton


# --------------------------------------------------------------------------- #
# round trips: precomputed.write_* -> readback.read_*
# --------------------------------------------------------------------------- #
def _skeleton(n=12):
    from osteoid import Skeleton

    zyx = np.stack([np.arange(n, dtype=np.float32) * 40.0,
                    np.full(n, 100.0, np.float32),
                    np.full(n, 200.0, np.float32)], axis=1)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.uint32)
    skel = Skeleton(vertices=zyx, edges=edges, segid=7)
    skel.radii = (np.arange(n, dtype=np.float32) + 5.0)
    return skel


def test_skeleton_round_trip_preserves_geometry_and_radii(tmp_path):
    from neu_morpho import precomputed

    vol = str(tmp_path / "segmentation")
    precomputed.write_skeleton_info(vol + "/skeleton")
    skel = _skeleton()
    precomputed.write_body_skeleton(vol + "/skeleton", 7, skel)

    v, e, r = read_body_skeleton(vol, 7)
    assert len(v) == len(skel.vertices) and len(e) == len(skel.edges)
    np.testing.assert_allclose(r, np.asarray(skel.radii, float), rtol=0, atol=1e-4)
    # The writer flips zyx -> xyz on the way out, so the vertex that was written
    # z-varying must come back x-varying. Getting this wrong mirrors the skeleton
    # through the z=x diagonal and is invisible in any per-vertex count.
    written_zyx = np.asarray(skel.vertices, float)
    np.testing.assert_allclose(v[:, 0], written_zyx[:, 2], atol=1e-4)   # x <- x
    np.testing.assert_allclose(v[:, 2], written_zyx[:, 0], atol=1e-4)   # z <- z


def test_read_body_skeleton_is_none_when_absent(tmp_path):
    vol = str(tmp_path / "segmentation")
    from neu_morpho import precomputed
    precomputed.write_skeleton_info(vol + "/skeleton")
    assert read_body_skeleton(vol, 999) is None


def test_read_body_skeleton_rejects_the_no_radius_sentinel(tmp_path):
    """A skeleton whose info declares no radius attribute must raise.

    osteoid does not report this as missing — it returns one value per vertex filled
    with ``-1``. Right length, finite, and physically impossible, so only a sign
    check catches it. This test failed against a length-only guard, which is exactly
    the hole it exists to hold shut.
    """
    from osteoid import Skeleton

    from neu_morpho import precomputed

    vol = str(tmp_path / "segmentation")
    # info WITHOUT the radius attribute, and a body encoded to match
    precomputed.write_skeleton_info(vol + "/skeleton", vertex_attributes=[])
    n = 4
    skel = Skeleton(
        vertices=np.stack([np.arange(n, dtype=np.float32) * 10.0,
                           np.zeros(n, np.float32), np.zeros(n, np.float32)], axis=1),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.uint32),
        segid=3)
    skel.radii = None
    precomputed.write_body_skeleton(vol + "/skeleton", 3, skel)
    with pytest.raises(ValueError, match="negative radii"):
        read_body_skeleton(vol, 3)


def _cube_mesh(cfg):
    from neu_morpho.coords import physical_box
    from neu_morpho.mesh import assemble_body, mesh_block

    block = np.zeros((24, 24, 24), np.uint64)
    block[6:18, 6:18, 6:18] = 1
    meshes = mesh_block(block, physical_box((slice(0, 24),) * 3, (8, 8, 8)), cfg)
    return assemble_body([meshes[1]], cfg)


def test_mesh_round_trip_returns_the_written_geometry(tmp_path):
    from neu_morpho import precomputed

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    mesh = _cube_mesh(cfg)
    written = np.asarray(mesh.vertices_zyx, float)[:, ::-1]        # zyx -> xyz

    vol = str(tmp_path / "segmentation")
    chunk_xyz, origin_xyz = [192.0] * 3, [0.0, 0.0, 0.0]
    precomputed.write_mesh_info(vol + "/mesh", cfg)
    n = precomputed.write_body_multires(vol + "/mesh", 5, mesh, cfg,
                                        chunk_shape_xyz=chunk_xyz,
                                        grid_origin_xyz=origin_xyz)
    assert n > 0

    got = read_body_mesh(vol, 5, lod=0)
    assert got is not None
    v, f, lod = got
    assert lod == 0 and len(v) and len(f)
    assert f.max() < len(v), "face index out of range after fragment concatenation"
    # Draco is lossy by quantization, not by displacement: the decoded corner must
    # land on the written one to within the quantization step, not merely nearby.
    assert np.allclose(v.min(axis=0), written.min(axis=0), atol=2.0)
    assert np.allclose(v.max(axis=0), written.max(axis=0), atol=2.0)


def test_read_body_mesh_defaults_to_the_coarsest_lod(tmp_path):
    from neu_morpho import precomputed

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0,
                     num_lods=3)
    vol = str(tmp_path / "segmentation")
    precomputed.write_mesh_info(vol + "/mesh", cfg)
    precomputed.write_body_multires(vol + "/mesh", 5, _cube_mesh(cfg), cfg,
                                    chunk_shape_xyz=[192.0] * 3,
                                    grid_origin_xyz=[0.0, 0.0, 0.0])
    default_lod = read_body_mesh(vol, 5)[2]
    assert default_lod == max(
        read_body_mesh(vol, 5, lod=l)[2] for l in range(cfg.num_lods)
        if read_body_mesh(vol, 5, lod=l) is not None)
    with pytest.raises(ValueError, match="not present"):
        read_body_mesh(vol, 5, lod=99)


def test_read_body_mesh_is_none_when_absent(tmp_path):
    from neu_morpho import precomputed

    cfg = MeshConfig(mesh_scale=0)
    vol = str(tmp_path / "segmentation")
    precomputed.write_mesh_info(vol + "/mesh", cfg)
    assert read_body_mesh(vol, 999) is None
