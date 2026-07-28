"""Build the per-body metrics DB by a parallel scan of the segmentation.

A block-map over the volume: each block reports per-label bounding box + voxel
count (in full-res voxel coords); the single-writer driver merges them into the
SQLite DB (min/max bbox, summed counts) atomically per block, so it's exact,
covers all bodies, and resumes correctly. Solves the "need a body's bbox before
we can crop/skeletonize it" problem — skeletonization reads bboxes from the DB.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import numpy as np

from em_blockrun import block_map, iter_blocks

from ..metrics_db import MetricsDB
from .. import roi as _roi


def _block_key(index) -> str:
    return "_".join(str(int(i)) for i in index)


def _index_block(block, *, seg_spec: dict, fullres_factor: Sequence[int]) -> tuple:
    """Per-label bbox (full-res voxels) + count for one block. Returns (key, partials)."""
    from em_volume_tools.backends.base import open_backend

    seg = open_backend(seg_spec).read_region(block.region)
    key = _block_key(block.index)
    nz = seg != 0
    if not nz.any():
        return (key, {})
    coords = np.argwhere(nz)                                   # local zyx, C-order
    labs = seg[nz]                                             # aligned with coords
    off = np.array([s.start for s in block.region])
    g = coords + off                                          # scan-scale global zyx
    factor = np.asarray(fullres_factor)
    order = np.argsort(labs, kind="stable")
    labs_s, g_s = labs[order], g[order]
    uniq, starts = np.unique(labs_s, return_index=True)
    starts = np.append(starts, len(labs_s))
    partials = {}
    for i, lab in enumerate(uniq):
        pts = g_s[starts[i]:starts[i + 1]]
        lo = (pts.min(0) * factor).astype(np.int64)
        hi = ((pts.max(0) + 1) * factor).astype(np.int64)     # half-open, full-res voxels
        partials[int(lab)] = (int(lo[0]), int(lo[1]), int(lo[2]),
                              int(hi[0]), int(hi[1]), int(hi[2]), int(len(pts)))
    return (key, partials)


def index_segments(
    seg_spec: dict,
    db_path: str,
    *,
    scan_voxel_size: Sequence[float],
    scan_scale: int = 0,
    fullres_factor: Sequence[int] | None = None,
    block_shape: Sequence[int] = (256, 256, 256),
    roi: Sequence[int] | str | None = None,
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Scan ``seg_spec`` (opened at ``scan_scale``) and fill the per-body metrics DB.

    ``scan_voxel_size`` (nm, zyx) is that scale's voxel size (for volume). Bbox is
    stored in full-res voxels via ``fullres_factor`` (default ``2**scan_scale``).

    ``roi`` (scan-scale voxels) restricts the scan to the blocks intersecting it,
    on the same global grid. Bear in mind the resulting bboxes and voxel counts
    then describe only the scanned portion of each body — fine for driving a trial
    run, not a substitute for a full index.
    """
    fullres_factor = tuple(fullres_factor or (2 ** scan_scale,) * 3)
    voxel_volume = float(np.prod(scan_voxel_size))

    from em_volume_tools.backends.base import open_backend
    shape = open_backend(seg_spec).shape
    roi = _roi.clip_to_shape(_roi.parse_roi(roi), shape)
    blocks = _roi.filter_blocks(iter_blocks(shape, block_shape), roi)

    db = MetricsDB(db_path)
    done = db.done_blocks() if resume else (db.reset_index() or set())
    todo = [b for b in blocks if _block_key(b.index) not in done]

    worker = functools.partial(_index_block, seg_spec=seg_spec, fullres_factor=fullres_factor)

    def on_result(batch):                                     # runs in the driver (single writer)
        for key, partials in batch:
            db.apply_index_block(key, partials, voxel_volume)

    try:
        block_map(todo, worker, client=client, npartitions=npartitions, on_result=on_result)
        n_bodies = db.con.execute("SELECT COUNT(*) FROM bodies").fetchone()[0]
    finally:
        db.close()
    return {"db": db_path, "n_blocks": len(blocks), "processed": len(todo), "n_bodies": n_bodies}
