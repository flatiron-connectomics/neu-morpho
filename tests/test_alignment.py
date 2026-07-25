"""Coordinate/alignment tests — the failure mode mesh-n-bone hit (offset meshes
and skeletons). Everything lives in one physical-nm world space."""

import numpy as np

from em_seg_morpho.config import SkeletonConfig
from em_seg_morpho.coords import crop_origin_nm, physical_box, skeleton_to_physical


def test_physical_box_scales_region_by_voxel_size():
    region = (slice(10, 20), slice(0, 6), slice(3, 9))
    box = physical_box(region, (8, 8, 8))
    np.testing.assert_array_equal(box, [[80, 0, 24], [160, 48, 72]])


def test_skeleton_to_physical_adds_crop_origin():
    verts = np.array([[0.0, 0.0, 0.0], [16.0, 24.0, 32.0]])   # local nm
    origin = crop_origin_nm((10, 10, 10), (8, 8, 8))          # (80, 80, 80) nm
    out = skeleton_to_physical(verts, origin)
    np.testing.assert_array_equal(out, [[80, 80, 80], [96, 104, 112]])


def test_mesh_vertices_land_in_nm():
    # cube at mesh-voxels [4:12] of a 16^3 block, voxel size 8 nm -> nm [32:96]
    block = np.zeros((16, 16, 16), np.uint64)
    block[4:12, 4:12, 4:12] = 1
    from vol2mesh import Mesh
    box = physical_box((slice(0, 16),) * 3, (8, 8, 8))        # [[0,0,0],[128,128,128]]
    m = Mesh.from_label_volume(block, box, labels=[1], ensure_halo=True, progress=False)[1]
    lo, hi = m.vertices_zyx.min(0), m.vertices_zyx.max(0)
    assert np.allclose(lo, 32, atol=1.0) and np.allclose(hi, 96, atol=1.0)


def test_mesh_and_skeleton_coincide():
    """A body at a global offset: its mesh bbox and its skeleton must overlap.

    Reproduces the 231668 scenario — if the skeleton's crop origin were dropped,
    the skeleton would sit at local coords, far outside the mesh bbox.
    """
    import kimimaro
    from vol2mesh import Mesh

    vs = (8, 8, 8)
    origin_vox = (10, 10, 10)                                  # crop starts at global voxel 10
    crop = np.zeros((16, 6, 6), np.uint64)
    crop[:, 2:5, 2:5] = 7                                      # 3x3 rod along z

    # mesh: region is the crop's global extent -> nm box
    region = (slice(10, 26), slice(10, 16), slice(10, 16))
    box = physical_box(region, vs)                            # z:[80,208] y,x:[80,128]
    mesh = Mesh.from_label_volume(crop, box, labels=[7], ensure_halo=True, progress=False)[7]
    m_lo, m_hi = mesh.vertices_zyx.min(0), mesh.vertices_zyx.max(0)

    # skeleton: crop-local, then shifted to global nm by the crop origin
    skels = kimimaro.skeletonize(crop, anisotropy=vs, object_ids=[7],
                                 teasar_params={"scale": 1.5, "const": 30, "pdrf_scale": 100000,
                                                "pdrf_exponent": 4}, dust_threshold=0, progress=False)
    sv = skeleton_to_physical(skels[7].vertices, crop_origin_nm(origin_vox, vs))
    s_lo, s_hi = sv.min(0), sv.max(0)

    # skeleton lies within the mesh bbox (a small margin for surface vs centerline)
    assert (s_lo >= m_lo - 8).all() and (s_hi <= m_hi + 8).all(), (m_lo, m_hi, s_lo, s_hi)
    # z extent matches the GLOBAL rod (~80..208), not the local one (~0..128) —
    # this is what fails if the crop origin is dropped (the 231668 bug).
    assert s_lo[0] >= 72 and s_hi[0] > 180
