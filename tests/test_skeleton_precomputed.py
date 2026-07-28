"""The precomputed skeleton writer — format, and the zyx->xyz flip.

The flip is the skeleton counterpart of the dropped-crop-origin bug: it produces
skeletons that look plausible but sit mirrored through the z=x diagonal relative
to their meshes.
"""

import json
import os

import numpy as np
import pytest

from em_seg_morpho.precomputed import (SKELETON_VERTEX_ATTRIBUTES, encode_skeleton,
                                       write_body_skeleton, write_skeleton_info)


def _skel(vertices_zyx, edges, radius=None):
    from osteoid import Skeleton
    s = Skeleton(vertices=np.asarray(vertices_zyx, np.float32),
                 edges=np.asarray(edges, np.uint32), segid=42)
    if radius is not None:
        s.radius = np.asarray(radius, np.float32)
    return s


def test_info_is_identity_transform(tmp_path):
    out = str(tmp_path / "skeleton")
    write_skeleton_info(out)
    info = json.load(open(os.path.join(out, "info")))
    assert info["@type"] == "neuroglancer_skeletons"
    assert info["transform"] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    assert info["vertex_attributes"] == SKELETON_VERTEX_ATTRIBUTES


def test_encode_flips_zyx_to_xyz():
    s = _skel([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], [[0, 1]])
    from osteoid import Skeleton
    back = Skeleton.from_precomputed(encode_skeleton(s))
    # stored coordinates are xyz: the zyx (1,2,3) must come back as (3,2,1)
    np.testing.assert_array_equal(back.vertices, [[3, 2, 1], [30, 20, 10]])
    np.testing.assert_array_equal(back.edges, [[0, 1]])


def test_encode_preserves_radius():
    s = _skel([[0.0, 0, 0], [1, 1, 1]], [[0, 1]], radius=[5.0, 7.0])
    from osteoid import Skeleton
    back = Skeleton.from_precomputed(encode_skeleton(s))
    np.testing.assert_array_equal(back.radius, [5.0, 7.0])


def test_declared_attributes_are_all_float32():
    """Neuroglancer rejects the whole layer on a non-float32 vertex attribute.

    The precomputed spec permits uint8 and friends, so this is a viewer
    constraint the spec will not warn you about: it surfaces only in the browser,
    as "Data type not supported by WebGL: UINT8".
    """
    assert [a["data_type"] for a in SKELETON_VERTEX_ATTRIBUTES] == ["float32"]


def test_write_skeleton_info_rejects_non_float32_attributes(tmp_path):
    from em_seg_morpho.precomputed import write_skeleton_info as w

    with pytest.raises(ValueError, match="must be float32"):
        w(str(tmp_path / "s"),
          vertex_attributes=[{"id": "radius", "data_type": "float32", "num_components": 1},
                             {"id": "vertex_types", "data_type": "uint8", "num_components": 1}])


def test_encode_drops_osteoid_default_vertex_types():
    """osteoid attaches a uint8 vertex_types by default; it must not reach the file.

    Byte-exact: 2 uint32 header + 2 verts * 3 float32 + 1 edge * 2 uint32 +
    2 radii * float32 = 8 + 24 + 8 + 8 = 48. A stray uint8 attribute adds 2.
    """
    s = _skel([[0.0, 0, 0], [1, 1, 1]], [[0, 1]], radius=[5.0, 7.0])
    s.vertex_types = np.array([2, 3], np.uint8)          # producer attached one
    assert len(encode_skeleton(s)) == 48


def test_write_body_skeleton_roundtrip(tmp_path):
    out = str(tmp_path / "skeleton")
    s = _skel([[0.0, 0, 0], [8, 16, 24]], [[0, 1]])
    assert write_body_skeleton(out, 42, s) == 2
    assert write_body_skeleton(out, 43, None) == 0

    path = os.path.join(out, "42")
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")
    from osteoid import Skeleton
    back = Skeleton.from_precomputed(open(path, "rb").read(), segid=42)
    np.testing.assert_array_equal(back.vertices, [[0, 0, 0], [24, 16, 8]])   # xyz
    assert not os.path.exists(os.path.join(out, "43"))


def test_mesh_and_skeleton_agree_after_encoding():
    """End of the alignment chain: an encoded skeleton lands inside its mesh bbox.

    test_alignment.py checks this in the in-memory zyx model space; this checks it
    survives the flip into stored xyz, which is what neuroglancer actually reads.
    """
    import kimimaro
    from vol2mesh import Mesh
    from osteoid import Skeleton

    from em_seg_morpho.coords import physical_box
    from em_seg_morpho.config import SkeletonConfig
    from em_seg_morpho.skeleton import skeletonize_block

    vs = (8.0, 8.0, 8.0)
    # asymmetric placement, so a z<->x swap cannot pass by accident
    block = np.zeros((24, 16, 40), np.uint64)
    block[4:20, 6:10, 30:34] = 7

    box = physical_box((slice(0, 24), slice(0, 16), slice(0, 40)), vs)
    mesh = Mesh.from_label_volume(block, box, labels=[7], ensure_halo=True, progress=False)[7]
    m_lo_xyz = mesh.vertices_zyx.min(0)[::-1]
    m_hi_xyz = mesh.vertices_zyx.max(0)[::-1]

    cfg = SkeletonConfig(anisotropy=vs, const=30.0, dust_threshold=0)
    skel = skeletonize_block(block, (0, 0, 0), cfg)[7]
    sv = Skeleton.from_precomputed(encode_skeleton(skel)).vertices

    assert (sv.min(0) >= m_lo_xyz - 8).all() and (sv.max(0) <= m_hi_xyz + 8).all(), \
        (m_lo_xyz, m_hi_xyz, sv.min(0), sv.max(0))
    # x is the narrow axis (~240..272 nm) and z the long one (~32..160): a flip bug swaps them
    assert sv[:, 0].min() > 200 and sv[:, 2].max() < 200
