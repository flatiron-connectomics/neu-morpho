"""Coordinate contract — the single thing that keeps meshes and skeletons aligned.

mesh-n-bone hit *mesh↔skeleton offset* bugs (e.g. segment 231668). The root cause
is meshes and skeletons ending up in different coordinate spaces. We avoid it by
putting **both** in one model space and using **identity** neuroglancer transforms:

    model space = physical nanometers (full-resolution world coords), zyx.

Everything is expressed via each scale's **voxel size (nm)** — never an assumed
``2**scale`` pyramid factor — so non-standard/anisotropic pyramids stay aligned.

- Mesh: ``Mesh.from_label_volume(block, physical_box(region, voxel_size_mesh))``
  yields vertices directly in nm (verified: a block region maps into the box).
- Skeleton: kimimaro (run with ``anisotropy = voxel_size_skel``) returns vertices
  in nm *local to the crop*; ``skeleton_to_physical(v, crop_origin_nm)`` shifts to
  global nm. The crop origin is the piece that, if dropped, causes the offset.
- neuroglancer mesh/skeleton ``info`` transforms stay identity (vertices are nm).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def physical_box(region: Sequence[slice], voxel_size_zyx: Sequence[float]) -> np.ndarray:
    """Full-res physical (nm) bounding box (2x3, zyx) for a block region at a scale.

    ``region`` is at some scale; ``voxel_size_zyx`` is that scale's voxel size (nm),
    so ``voxel * voxel_size`` lands in the same nm world regardless of scale.
    """
    vz, vy, vx = (float(v) for v in voxel_size_zyx)
    (z0, z1), (y0, y1), (x0, x1) = [(s.start, s.stop) for s in region]
    return np.array([[z0 * vz, y0 * vy, x0 * vx], [z1 * vz, y1 * vy, x1 * vx]])


def crop_origin_nm(origin_vox_zyx: Sequence[int], voxel_size_zyx: Sequence[float]) -> np.ndarray:
    """Physical (nm) origin of a crop given its origin in that scale's voxels."""
    return np.asarray(origin_vox_zyx, float) * np.asarray(voxel_size_zyx, float)


def skeleton_to_physical(vertices_nm_zyx: np.ndarray, origin_nm_zyx: Sequence[float]) -> np.ndarray:
    """Shift crop-local kimimaro vertices (nm) to global physical nm (zyx).

    kimimaro must be run with ``anisotropy = voxel_size`` so its vertices are
    already physical; this only adds the crop origin — the step whose omission is
    the classic mesh↔skeleton offset.
    """
    return np.asarray(vertices_nm_zyx, float) + np.asarray(origin_nm_zyx, float)


def block_chunk_shape_xyz(block_shape_zyx: Sequence[int], voxel_size_zyx: Sequence[float]) -> list[float]:
    """Octree LOD-0 cell size in nm (xyz) for the multires writer = block extent in nm."""
    vz, vy, vx = voxel_size_zyx
    bz, by, bx = block_shape_zyx
    return [bx * vx, by * vy, bz * vz]     # xyz
