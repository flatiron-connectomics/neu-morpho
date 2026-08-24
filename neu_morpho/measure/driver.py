"""Block-mapped per-body volume over a published segmentation.

The one module in `measure` that opens a store and touches dask; everything in
`sweep.py` is pure so it can be tested without either. Fills `voxel_count` /
`volume_nm3` in a `MetricsDB`, block by block, resumably.

Three choices here follow from measurements recorded in `docs/measure-calibration.md`:

- **No occupancy or ROI prefilter by default.** An ROI built from a compartment mask
  omits somata that sit outside it, and `V/L` then divides a truncated volume by a
  complete cable length — wrong, and invisible. An empty block costs ~0.22 s against
  ~3 s for a dense one, so reading everything is affordable rather than heroic.
- **Counts, not bboxes.** `MetricsDB.apply_counts_block` says why.
- **Per-block retry, but fail-fast on a permanent error.** Tens of thousands of tasks
  against an object store make a transient failure near-certain, and two earlier runs
  in this suite were lost to exactly one. Retry does not soften the fail-fast policy:
  an error that persists still kills the run.
"""

from __future__ import annotations

import functools
from typing import Any, Iterable, Sequence

import numpy as np

from blockrun import block_map, iter_blocks

from ..metrics_db import MetricsDB
from .. import roi as _roi
from .sweep import DEFAULT_BLOCK, count_labels


def _block_key(index) -> str:
    return "_".join(str(int(i)) for i in index)


def resolve_keep(sources: Iterable[str] | None) -> set[int] | None:
    """Union the body ids named by each source; ``None`` for "record every label".

    A source is either a path to a file of ids (one per line, or a first CSV column) or
    a ``neuroglancer_segment_properties`` source — local directory or URL — whose ``ids``
    list is taken. Several may be given: a published dataset often splits its properties
    across more than one source, and their id sets are not identical.

    The two are told apart by ``os.path.isfile``: an id list is a file, a properties
    source is a directory or a URL containing an ``info``. Sniffing for ``"://"``
    instead would misread a local properties directory as an id file.

    A properties source is usually the better cohort gate than a size threshold, because
    a body only has properties if somebody named it — so the union already excludes
    unlabeled fragments, which in a fragmented segmentation outnumber the real bodies by
    orders of magnitude.
    """
    if not sources:
        return None
    import os

    from ..allowlist import load_allowlist

    out: set[int] = set()
    for src in sources:
        src = str(src)
        if os.path.isfile(src):
            got = load_allowlist(src)
            if got:
                out |= got
            continue
        from neu_vol import location
        info = location.read_json(src, "info")
        if info is None:
            raise ValueError(f"--keep {src}: not a file, and no 'info' found there")
        kind = info.get("@type")
        if kind != "neuroglancer_segment_properties":
            raise ValueError(
                f"--keep {src} is a '{kind}', not a neuroglancer_segment_properties "
                f"source. A segmentation volume's own info carries no id list.")
        out |= {int(i) for i in info["inline"]["ids"]}
    return out


def _count_block(block, *, seg_spec: dict, keep: frozenset | None,
                 background: int = 0, attempts: int = 5) -> tuple:
    """``(key, {body: voxel_count})`` for one block. The whole block is the retry unit."""
    from neu_vol.backends.base import open_backend
    from neu_vol.retry import with_retry

    key = _block_key(block.index)

    def read_and_count():
        seg = open_backend(seg_spec).read_region(block.region)
        return count_labels(seg, background=background)

    counts = with_retry(read_and_count, attempts=attempts, label=f"block {key}")
    if keep is not None:
        counts = {k: v for k, v in counts.items() if k in keep}
    return (key, counts)


def sweep_volumes(
    seg_spec: dict,
    db_path: str,
    *,
    voxel_size: Sequence[float],
    block: int | Sequence[int] = DEFAULT_BLOCK,
    keep: Iterable[int] | None = None,
    roi: Sequence[int] | str | None = None,
    background: int = 0,
    attempts: int = 5,
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Count each body's voxels across ``seg_spec`` and accumulate into ``db_path``.

    ``voxel_size`` (nm, zyx) is the read level's own voxel size — read it from
    ``neu_vol.read_scales``, never as ``2**level`` (see invariant 1). ``keep`` restricts
    which bodies are recorded, not which blocks are read, so the rows written are
    complete; without it a fragmented segmentation records every single-voxel speck.

    Refuses to run on a DB that already holds an index scan, because both fill
    ``voxel_count`` and running both would double it.
    """
    shape = tuple(int(s) for s in np.atleast_1d(block)) if not np.isscalar(block) else None
    block_shape = shape or (int(block),) * 3
    keep = None if keep is None else frozenset(int(b) for b in keep)
    voxel_volume = float(np.prod(voxel_size))

    from neu_vol.backends.base import open_backend
    vol_shape = open_backend(seg_spec).shape
    roi = _roi.clip_to_shape(_roi.parse_roi(roi), vol_shape)
    blocks = _roi.filter_blocks(iter_blocks(vol_shape, block_shape), roi)

    db = MetricsDB(db_path)
    try:
        n_indexed = db.con.execute("SELECT COUNT(*) FROM index_progress").fetchone()[0]
        if n_indexed:
            raise ValueError(
                f"{db_path} already holds an index scan ({n_indexed} blocks). Both it and "
                f"this sweep accumulate into voxel_count, so running both would double "
                f"every count. Use a separate DB, or read the counts that are already there.")
        done = db.done_sweep_blocks() if resume else db.reset_sweep()
        todo = [b for b in blocks if _block_key(b.index) not in done]
        # Before dispatch, so a reader has a denominator even mid-run (invariant 11).
        db.record_stage_meta("sweep", total=len(blocks), block_shape=block_shape,
                             voxel_size=voxel_size,
                             n_keep=None if keep is None else len(keep))

        worker = functools.partial(_count_block, seg_spec=seg_spec, keep=keep,
                                   background=background, attempts=attempts)

        def on_result(batch):                     # driver-side, single writer
            for key, counts in batch:
                db.apply_counts_block(key, counts, voxel_volume)

        block_map(todo, worker, client=client, npartitions=npartitions,
                  on_result=on_result)
        n_bodies = db.con.execute(
            "SELECT COUNT(*) FROM bodies WHERE voxel_count>0").fetchone()[0]
    finally:
        db.close()
    return {"db": db_path, "n_blocks": len(blocks), "processed": len(todo),
            "n_bodies": n_bodies, "block_shape": block_shape,
            "n_kept": None if keep is None else len(keep)}
