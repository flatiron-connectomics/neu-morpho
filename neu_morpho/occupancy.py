"""Coarse-scale occupancy prefilter — skip empty blocks up front.

mesh-n-bone read a coarse scale (e.g. 256 nm) once and skipped the ~85% of blocks
containing no data, so empty space is never read or meshed. We do the same:
reduce a small coarse-scale occupancy array to a boolean over the *meshing* block
grid. When a body allowlist is given, "occupied" means ``isin(coarse, allowlist)``
so only blocks containing a target body are processed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def occupied_blocks(
    occ_array_zyx: np.ndarray,
    *,
    occ_voxel_size: Sequence[float],
    mesh_voxel_size: Sequence[float],
    block_shape: Sequence[int],
    grid_shape: Sequence[int],
    allowlist: set[int] | None = None,
    dilate: int = 0,
) -> set[tuple[int, int, int]]:
    """Return the set of block indices (in the meshing grid) that are non-empty.

    ``occ_array_zyx`` is the whole coarse-scale segmentation (small). Physical
    voxel sizes map the coarse grid onto the meshing block grid; ``grid_shape`` is
    the number of blocks per axis at the meshing scale.

    ``dilate`` grows the occupied set by N blocks in every direction. **Use it.**
    A coarse scale does not see everything the meshing scale does — downsampling
    drops sparse tissue, and the miss does not converge: on real data, blocks
    found were 354 (scale 6) / 360 (scale 5) / 368 (scale 4), strictly nested. So
    any un-dilated coarse filter silently skips blocks that hold data, and you
    would never learn which. One block of dilation made scale 5 a superset of
    scale 4 (618 blocks instead of 360) — and the trade is cheap, because a
    false-positive block costs one read that finds no labels, while a false
    negative costs a body.

    ``allowlist`` is supported but discouraged here: at a coarse scale many small
    allowlisted bodies have been downsampled away, so their blocks look empty.
    Prefer the default ``!= 0`` test and let the allowlist filter inside the block.
    """
    if allowlist is None:
        occ_mask = occ_array_zyx != 0
    else:
        vals = np.array(sorted(allowlist), dtype=occ_array_zyx.dtype)
        occ_mask = np.isin(occ_array_zyx, vals)          # one pass over the small array

    # coarse-scale voxels spanned by one meshing block, per axis
    occ_per_block = [block_shape[a] * mesh_voxel_size[a] / occ_voxel_size[a] for a in range(3)]

    grid = np.zeros(tuple(int(g) for g in grid_shape), dtype=bool)
    for index in np.ndindex(grid.shape):
        sl = tuple(
            slice(int(round(i * opb)), int(round((i + 1) * opb)))
            for i, opb in zip(index, occ_per_block)
        )
        sub = occ_mask[sl]
        if sub.size and sub.any():
            grid[index] = True

    if dilate > 0:
        from scipy.ndimage import binary_dilation
        grid = binary_dilation(grid, iterations=int(dilate))

    return {tuple(int(v) for v in idx) for idx in np.argwhere(grid)}
