"""Enumerate the segments to process and their bounding boxes.

This is the biggest open question (see docs/DESIGN.md): where do per-segment
bounding boxes come from? Options, in rough order of preference:
  - a **label / spatial index** shipped with the source (DVID label indices,
    precomputed spatial index, a sidecar table) — cheap and exact;
  - otherwise **scan blocks** of the segmentation once, accumulating per-label
    bounding boxes (and voxel counts for the min-size filter). This is itself a
    block-map over the volume (em-blockrun) with a reduction.

Kept as an interface here; the concrete source is wired when we know the platform
(DVID vs precomputed vs zarr).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# A bounding box in canonical (z, y, x): (z0, y0, x0, z1, y1, x1), half-open.
BBox = tuple[int, int, int, int, int, int]


def iter_segment_ids(seg_spec: Mapping, *, min_voxels: int = 0,
                     exclude: Sequence[int] = (0,)) -> list[int]:
    """Segment ids to process (excluding background, applying a min-size filter)."""
    raise NotImplementedError("segment enumeration — source TBD (label index vs scan)")


def segment_bounding_boxes(seg_spec: Mapping, segment_ids: Iterable[int]) -> dict[int, BBox]:
    """Per-segment bounding boxes in canonical (z, y, x)."""
    raise NotImplementedError("bbox source — label/spatial index vs. block scan")
