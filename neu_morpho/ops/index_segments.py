"""Build the per-body metrics DB by a parallel scan of the segmentation.

A block-map over the volume: each block reports per-label bounding box + voxel
count (in full-res voxel coords); the single-writer driver merges them into the
SQLite DB (min/max bbox, summed counts) atomically per block, so it's exact,
covers all bodies, and resumes correctly.

**Who actually consumes this, as of now** — the docstring used to say "skeletonization
reads bboxes from the DB", and that consumer is gone: `meshify` and
`skeletonize_segments` are both block-based and read no bbox. What remains is

- ``voxel_count``, for ``MetricsDB.write_allowlist`` (the ``--min-voxels`` /
  ``--limit-bodies`` size filter). Counts only; no bbox needed.
- the bbox, for ``scripts/sweep_postprocess.py`` via ``crop_at_scale``.

So the bbox half is now paid for by one script rather than by the pipeline, and it is
the expensive half: an ``argwhere`` plus an ``argsort`` over every labelled voxel, ~30 s
against ~3 s for the counts on a dense block. For per-body volume alone, prefer
``neu_morpho.measure.sweep_volumes`` (``neu-morpho measure volumes``).
"""

from __future__ import annotations

import functools
from typing import Any, Iterable, Sequence

import numpy as np

from blockrun import block_map, iter_blocks

from ..metrics_db import MetricsDB
from .. import roi as _roi


def _block_key(index) -> str:
    return "_".join(str(int(i)) for i in index)


def _index_block(block, *, seg_spec: dict, fullres_factor: Sequence[int],
                 keep: frozenset | None = None) -> tuple:
    """Per-label bbox (full-res voxels) + count for one block. Returns (key, partials).

    ``keep`` filters the per-label RESULT, never the array. A dense block of a
    fragmented segmentation can hold hundreds of thousands of distinct labels, nearly
    all of them single-voxel specks, and each one costs a driver-side upsert — so
    filtering here is what keeps the single writer from becoming the bottleneck.
    Testing membership against the array instead (``np.isin(seg, keep)``) costs twenty
    times the block read, which is why the filter is applied to ``uniq`` and not to
    ``seg``.
    """
    from neu_vol.backends.base import open_backend

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
        if keep is not None and int(lab) not in keep:
            continue
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
    keep: Iterable[int] | None = None,
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

    ``keep`` restricts which BODIES are recorded, leaving every block read in full, so
    unlike ``roi`` the rows it does write are complete. It is the right knob for a
    fragmented segmentation, where the specks outnumber the verified neurons by orders
    of magnitude and every one of them costs a driver-side upsert. The trade is that a
    later change of cohort needs another scan — cheap for a small volume, and the
    alternative is a table whose real content is buried.
    """
    fullres_factor = tuple(fullres_factor or (2 ** scan_scale,) * 3)
    voxel_volume = float(np.prod(scan_voxel_size))
    keep = None if keep is None else frozenset(int(b) for b in keep)

    from neu_vol.backends.base import open_backend
    shape = open_backend(seg_spec).shape
    roi = _roi.clip_to_shape(_roi.parse_roi(roi), shape)
    blocks = _roi.filter_blocks(iter_blocks(shape, block_shape), roi)

    db = MetricsDB(db_path)
    done = db.done_blocks() if resume else (db.reset_index() or set())
    todo = [b for b in blocks if _block_key(b.index) not in done]
    # Before dispatch, so `measure progress` has a denominator mid-run (invariant 11).
    db.record_stage_meta("index", total=len(blocks), block_shape=block_shape,
                         level=scan_scale, voxel_size=scan_voxel_size,
                         n_keep=None if keep is None else len(keep))

    worker = functools.partial(_index_block, seg_spec=seg_spec,
                               fullres_factor=fullres_factor, keep=keep)

    def on_result(batch):                                     # runs in the driver (single writer)
        for key, partials in batch:
            db.apply_index_block(key, partials, voxel_volume)

    try:
        block_map(todo, worker, client=client, npartitions=npartitions, on_result=on_result)
        n_bodies = db.con.execute("SELECT COUNT(*) FROM bodies").fetchone()[0]
    finally:
        db.close()
    return {"db": db_path, "n_blocks": len(blocks), "processed": len(todo),
            "n_bodies": n_bodies,
            "n_kept": None if keep is None else len(keep)}
