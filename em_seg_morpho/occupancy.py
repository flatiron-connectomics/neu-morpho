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
) -> set[tuple[int, int, int]]:
    """Return the set of block indices (in the meshing grid) that are non-empty.

    ``occ_array_zyx`` is the whole coarse-scale segmentation (small). Physical
    voxel sizes map the coarse grid onto the meshing block grid; ``grid_shape`` is
    the number of blocks per axis at the meshing scale.
    """
    if allowlist is None:
        occ_mask = occ_array_zyx != 0
    else:
        vals = np.array(sorted(allowlist), dtype=occ_array_zyx.dtype)
        occ_mask = np.isin(occ_array_zyx, vals)          # one pass over the small array

    # coarse-scale voxels spanned by one meshing block, per axis
    occ_per_block = [block_shape[a] * mesh_voxel_size[a] / occ_voxel_size[a] for a in range(3)]

    occupied: set[tuple[int, int, int]] = set()
    for iz in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            for ix in range(grid_shape[2]):
                sl = tuple(
                    slice(int(round(i * opb)), int(round((i + 1) * opb)))
                    for i, opb in zip((iz, iy, ix), occ_per_block)
                )
                sub = occ_mask[sl]
                if sub.size and sub.any():
                    occupied.add((iz, iy, ix))
    return occupied
