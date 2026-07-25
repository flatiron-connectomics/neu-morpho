import numpy as np

from em_seg_morpho.occupancy import occupied_blocks


def _occ():
    # coarse array (4^3) for a 16^3 mesh volume with block 8 -> 2^3 block grid,
    # occ voxel 32nm vs mesh 8nm -> 2 occ voxels per block per axis.
    occ = np.zeros((4, 4, 4), dtype=np.uint64)
    occ[0:2, 0:2, 0:2] = 7      # occupies block (0,0,0)
    occ[2:4, 2:4, 2:4] = 9      # occupies block (1,1,1)
    return occ


def test_occupied_blocks_nonempty():
    occ = occupied_blocks(_occ(), occ_voxel_size=(32, 32, 32), mesh_voxel_size=(8, 8, 8),
                          block_shape=(8, 8, 8), grid_shape=(2, 2, 2))
    assert occ == {(0, 0, 0), (1, 1, 1)}


def test_occupied_blocks_allowlist_restricts():
    occ = occupied_blocks(_occ(), occ_voxel_size=(32, 32, 32), mesh_voxel_size=(8, 8, 8),
                          block_shape=(8, 8, 8), grid_shape=(2, 2, 2), allowlist={7})
    assert occ == {(0, 0, 0)}       # block with body 9 excluded


def test_all_empty_returns_nothing():
    occ = occupied_blocks(np.zeros((4, 4, 4), np.uint64), occ_voxel_size=(32, 32, 32),
                          mesh_voxel_size=(8, 8, 8), block_shape=(8, 8, 8), grid_shape=(2, 2, 2))
    assert occ == set()
