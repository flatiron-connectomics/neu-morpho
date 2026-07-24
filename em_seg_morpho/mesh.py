"""Per-segment mesh generation via vol2mesh.

Small segments: mesh the whole binary mask (``Mesh.from_binary_vol``).

Large segments: **stream** block masks into ``Mesh.from_binary_blocks`` — vol2mesh
meshes each block then discards it (docstring: "you may pass any iterable of
blocks, including a generator object"), so peak *mask* memory is a single block,
never the whole object. This is the key point: the OOM came from materializing
the whole-object binary mask, so we must pass a **generator** of block masks —
NOT a list — to keep memory bounded. Only the per-block *meshes* (much smaller
than masks) accumulate before stitching.

Masks are read at the configured meshing LOD/scale (``MeshConfig.start_lod``,
default 2), which also cuts per-block mask size ~8×/level.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np

from .config import MeshConfig

# Bounding box in canonical (z, y, x): (z0, y0, x0, z1, y1, x1), half-open.
BBox = tuple[int, int, int, int, int, int]


def mesh_from_mask(mask_zyx: np.ndarray, fullres_box_zyx, cfg: MeshConfig):
    """Mesh a single binary mask (fits in memory); returns a ``vol2mesh.Mesh``."""
    from vol2mesh import Mesh

    mesh = Mesh.from_binary_vol(mask_zyx, np.asarray(fullres_box_zyx))
    if cfg.decimation_fraction and cfg.decimation_fraction < 1.0:
        mesh.simplify(cfg.decimation_fraction)
    return mesh


def mesh_from_block_stream(block_masks: Iterable[np.ndarray],
                           fullres_boxes_zyx: Sequence[BBox], cfg: MeshConfig):
    """Mesh a large segment by streaming block masks (bounded memory) + stitching.

    ``block_masks`` MUST be a lazy iterable/generator aligned with
    ``fullres_boxes_zyx`` (a small list of coordinates) — do not materialize the
    blocks into a list, or you reintroduce the whole-mask OOM.
    """
    from vol2mesh import Mesh

    mesh = Mesh.from_binary_blocks(block_masks, list(fullres_boxes_zyx), stitch=True)
    if cfg.decimation_fraction and cfg.decimation_fraction < 1.0:
        mesh.simplify(cfg.decimation_fraction)
    return mesh


def block_boxes(bbox_zyx: BBox, chunk_shape_zyx: Sequence[int], halo: int = 1) -> list[BBox]:
    """Tile a segment bbox into block boxes, overlapping by ``halo`` voxels.

    A 1-voxel halo lets adjacent block meshes share boundary geometry so
    ``stitch=True`` can weld them into a watertight surface. Boxes are clipped to
    the segment bbox. Cheap (just coordinates) — safe to hold as a list.
    """
    z0, y0, x0, z1, y1, x1 = bbox_zyx
    cz, cy, cx = chunk_shape_zyx
    boxes: list[BBox] = []
    for zs in range(z0, z1, cz):
        for ys in range(y0, y1, cy):
            for xs in range(x0, x1, cx):
                boxes.append((
                    max(z0, zs - halo), max(y0, ys - halo), max(x0, xs - halo),
                    min(z1, zs + cz + halo), min(y1, ys + cy + halo), min(x1, xs + cx + halo),
                ))
    return boxes


def stream_block_masks(read_box, boxes: Sequence[BBox], segment_id: int) -> Iterator[np.ndarray]:
    """Lazily yield one binary block mask per box (only one block in memory at a time).

    ``read_box(box) -> ndarray`` reads that region of the segmentation (caller
    supplies it, closing over an em-volume-tools backend at the meshing LOD).
    """
    for box in boxes:
        yield read_box(box) == segment_id


def should_chunk(bbox_shape_zyx: Sequence[int], cfg: MeshConfig) -> bool:
    """True if the segment's (LOD-scaled) bbox mask would exceed the memory budget."""
    return int(np.prod(bbox_shape_zyx)) > cfg.max_mask_voxels
