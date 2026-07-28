"""Restrict a run to a portion of the volume, without moving the block grid.

The grid is always tiled over the **whole** volume and then filtered: a block's
index and its region are the same numbers in an ROI run as in a later full run.
That is the point — fragments, manifest keys and nm coordinates all carry over,
so a test run over one cube is a *prefix* of the full run rather than a separate
universe you have to redo.

ROIs are half-open ``(z0, y0, x0, z1, y1, x1)`` in the voxels of the scale being
read. Blocks are kept whole when they intersect, never clipped: clipping would
make a block's content depend on the ROI, so the same block index would hold
different data in two runs and resume would silently reuse the wrong fragment.
"""

from __future__ import annotations

import math
from typing import Sequence


def parse_roi(text: str | Sequence[int] | None) -> tuple[int, ...] | None:
    """Parse ``"z0,y0,x0,z1,y1,x1"`` (or a 6-sequence) into a tuple, or None."""
    if text is None:
        return None
    parts = [int(v) for v in text.split(",")] if isinstance(text, str) else [int(v) for v in text]
    if len(parts) != 6:
        raise ValueError(f"ROI needs 6 values (z0,y0,x0,z1,y1,x1), got {len(parts)}: {text!r}")
    if any(hi <= lo for lo, hi in zip(parts[:3], parts[3:])):
        raise ValueError(f"ROI must be half-open with stop > start per axis: {parts}")
    return tuple(parts)


def scale_roi(roi: Sequence[int] | None, factor_zyx: Sequence[float]) -> tuple[int, ...] | None:
    """Convert an ROI between scales; ``factor_zyx`` = target voxels per source voxel.

    Starts floor, stops ceil, so the scaled ROI always covers the original region
    rather than shaving a partial voxel off its edge.
    """
    if roi is None:
        return None
    lo = [math.floor(roi[a] * factor_zyx[a]) for a in range(3)]
    hi = [math.ceil(roi[3 + a] * factor_zyx[a]) for a in range(3)]
    return tuple(lo + [max(h, l + 1) for l, h in zip(lo, hi)])


def intersects(region: Sequence[slice], roi: Sequence[int]) -> bool:
    """Does a block region overlap the ROI (half-open, per axis)?"""
    for axis, sl in enumerate(region[:3]):
        if sl.stop <= roi[axis] or sl.start >= roi[3 + axis]:
            return False
    return True


def filter_blocks(blocks, roi: Sequence[int] | None) -> list:
    """Keep the blocks of the global grid that intersect ``roi`` (all, if None)."""
    if roi is None:
        return list(blocks)
    return [b for b in blocks if intersects(b.region, roi)]


def clip_to_shape(roi: Sequence[int] | None, shape: Sequence[int]) -> tuple[int, ...] | None:
    """Clamp an ROI to a volume's bounds, so an over-wide stop is not an error."""
    if roi is None:
        return None
    lo = [max(0, int(roi[a])) for a in range(3)]
    hi = [min(int(shape[a]), int(roi[3 + a])) for a in range(3)]
    if any(h <= l for l, h in zip(lo, hi)):
        raise ValueError(f"ROI {tuple(roi)} does not intersect volume shape {tuple(shape)}")
    return tuple(lo + hi)
