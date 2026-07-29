"""Dilation of the occupancy prefilter, and why it is not optional.

A coarse scale does not see everything the meshing scale does, and the miss does
not converge with depth — measured on real data, blocks found were 354 (scale 6),
360 (scale 5), 368 (scale 4), strictly nested. So an un-dilated coarse filter
silently skips blocks that hold data. Dilation trades a wasted read (cheap: the
block is read, has no labels, returns) for not losing a body (not cheap).
"""

import numpy as np

from em_seg_morpho.occupancy import occupied_blocks


def _occ(arr, dilate=0, allowlist=None, grid=(4, 4, 4)):
    return occupied_blocks(arr, occ_voxel_size=(4.0, 4.0, 4.0),
                           mesh_voxel_size=(1.0, 1.0, 1.0), block_shape=(16, 16, 16),
                           grid_shape=grid, allowlist=allowlist, dilate=dilate)


def test_finds_only_blocks_with_labels():
    # coarse array 16^3 maps 4 coarse voxels per block over a 4x4x4 block grid
    arr = np.zeros((16, 16, 16), np.uint64)
    arr[5, 5, 5] = 7                      # -> block (1, 1, 1)
    assert _occ(arr) == {(1, 1, 1)}


def test_dilation_grows_by_whole_blocks():
    arr = np.zeros((16, 16, 16), np.uint64)
    arr[5, 5, 5] = 7
    d1 = _occ(arr, dilate=1)
    assert (1, 1, 1) in d1
    # 6-connected neighbours of (1,1,1) come in; corners do not (binary_dilation
    # default structuring element)
    for n in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        assert n in d1, n
    assert len(d1) == 7


def test_dilation_is_a_superset_and_clipped_to_the_grid():
    arr = np.zeros((16, 16, 16), np.uint64)
    arr[1, 1, 1] = 7                      # block (0,0,0) — at the grid corner
    base, grown = _occ(arr), _occ(arr, dilate=1)
    assert base <= grown
    assert all(0 <= i < 4 for idx in grown for i in idx)


def test_dilation_recovers_a_block_a_coarse_scale_missed():
    """The real failure mode, in miniature.

    Two adjacent blocks of tissue; downsampling drops the sparser one. Without
    dilation that block is skipped and its bodies never meshed.
    """
    fine = np.zeros((16, 16, 16), np.uint64)
    fine[4:8, 4:8, 4:8] = 7               # block (1,1,1): dense
    fine[8, 5, 5] = 7                     # block (2,1,1): a single sparse voxel

    coarse = fine.copy()
    coarse[8, 5, 5] = 0                   # ...which downsampling erased
    assert _occ(coarse) == {(1, 1, 1)}                       # the miss
    assert _occ(fine) == {(1, 1, 1), (2, 1, 1)}              # ground truth
    assert _occ(fine) <= _occ(coarse, dilate=1)              # dilation covers it


def test_allowlist_at_a_coarse_scale_can_hide_blocks():
    """Why the ops pass allowlist=None to the occupancy filter.

    A small allowlisted body that downsampling erased makes its block look empty,
    so allowlist-based occupancy would skip exactly the body you asked for.
    """
    arr = np.zeros((16, 16, 16), np.uint64)
    arr[5, 5, 5] = 7                      # body 7 survived downsampling
    arr[9, 9, 9] = 99                     # body 99 did too, but 42 did not

    assert _occ(arr, allowlist={42}) == set()          # body 42 vanished -> no blocks
    assert _occ(arr) == {(1, 1, 1), (2, 2, 2)}         # plain !=0 keeps both
