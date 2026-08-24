"""Cable length and topology for many bodies, from published skeletons.

The per-BODY counterpart of `driver.sweep_volumes`, and it differs from it in every way
that matters:

- **The task unit is a batch of bodies, not a block.** Each read is one small object, so
  per-task dispatch overhead (~1.5 s, measured on the volume sweep) would dominate
  completely at one body per task. Bodies are batched, and within a batch a thread pool
  runs the reads concurrently — they are I/O-bound, and `location._kv` caches the opened
  store per prefix, so every body under `skeleton/` shares one store and threads add no
  store opens.
- **Failures are ISOLATED, not fatal** (invariant 5). A per-block task fails fast because
  a skipped block truncates every body passing through it; a per-body task records the
  failure and continues, because one unreadable skeleton says nothing about the others.
- **Absence is not failure.** A body with no published skeleton is recorded `absent` and
  never retried; a body that raised is recorded `failed` and retried next run. Testing
  for mere presence of a status would make a failure look done, which is invariant 3's
  trap restated for this table.

Radii are optional: a source publishing centrelines only yields NaN diameters rather
than raising, so cable length and topology still land.
"""

from __future__ import annotations

import functools
from typing import Any, Iterable, Sequence

import numpy as np

from blockrun import block_map

from ..metrics_db import MetricsDB

#: Bodies per task. Large enough to amortise dispatch, small enough that a batch is a
#: useful resume granule and that one slow body cannot stall a worker for long.
DEFAULT_BATCH = 200

#: Concurrent reads within a batch. I/O-bound, so well above the core count.
DEFAULT_THREADS = 8

#: Columns written per body — deliberately the SAME four the pipeline's skel stage
#: writes, so a dataset measured here and one measured by a production run are
#: schema-identical and directly comparable. Richer per-body rows (diameter percentiles,
#: frustum volume, component counts) belong in the export table, not in this DB.
_COLUMNS = ("cable_length_nm", "n_branches", "n_tips", "max_radius_nm")


def _measure_one(body_id: int, *, volume: str, skeleton_dir: str,
                 require_radii: bool) -> tuple:
    """``(body_id, status, payload)`` for one body. Never raises."""
    from neu_lib import Skeleton

    from ..readback import read_body_skeleton
    from .metrics import cable_length_nm, diameter_stats, topology

    try:
        got = read_body_skeleton(volume, body_id, skeleton_dir,
                                 require_radii=require_radii)
        if got is None:
            return (body_id, "absent", None)
        verts_xyz, edges, radii = got
        # The reader returns the order the FORMAT stores, which is xyz; everything here
        # is zyx. Invariant 2, and getting it wrong mirrors the body through the z=x
        # diagonal — which for a cable length is invisible, since a mirrored skeleton has
        # the same edge lengths. So this flip is load-bearing only for the coordinates,
        # and silent if omitted. It cost a debugging round earlier today.
        skel = Skeleton(vertices_zyx_nm=np.asarray(verts_xyz, dtype=float)[:, ::-1],
                        edges=edges, radii_nm=radii, name=str(body_id))
        row = {"cable_length_nm": cable_length_nm(skel)}
        row.update(topology(skel))
        d = diameter_stats(skel)
        # max_radius_nm, not a diameter: the column is a radius in the pipeline's schema
        # and mixing the two is a silent factor of 2.
        row["max_radius_nm"] = (np.nan if not np.isfinite(d["diameter_nm_max"])
                                else d["diameter_nm_max"] / 2.0)
        return (body_id, "written", {k: row[k] for k in _COLUMNS if k in row})
    except Exception as exc:                                   # noqa: BLE001 - isolated
        return (body_id, "failed", f"{type(exc).__name__}: {exc}"[:400])


def _measure_batch(bodies: Sequence[int], *, volume: str, skeleton_dir: str,
                   require_radii: bool, threads: int) -> list[tuple]:
    from concurrent.futures import ThreadPoolExecutor

    fn = functools.partial(_measure_one, volume=volume, skeleton_dir=skeleton_dir,
                           require_radii=require_radii)
    if threads <= 1:
        return [fn(b) for b in bodies]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(fn, bodies))


def sweep_skeletons(
    volume: str,
    db_path: str,
    *,
    bodies: Iterable[int],
    skeleton_dir: str = "skeleton",
    require_radii: bool = False,
    batch: int = DEFAULT_BATCH,
    threads: int = DEFAULT_THREADS,
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
    retry_failed: bool = True,
) -> dict:
    """Measure each body's published skeleton into ``db_path``.

    ``resume`` skips bodies already recorded ``written`` or ``absent``; ``retry_failed``
    controls whether ``failed`` ones are attempted again (they are, by default — that is
    the point of recording them separately).
    """
    bodies = [int(b) for b in bodies]
    db = MetricsDB(db_path)
    try:
        done = db.done_skel_bodies(include_failed=not retry_failed) if resume else \
            db.reset_skel()
        todo = [b for b in bodies if b not in done]
        db.record_stage_meta("skel", total=len(bodies), n_keep=len(bodies))

        batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
        worker = functools.partial(_measure_batch, volume=volume,
                                   skeleton_dir=skeleton_dir,
                                   require_radii=require_radii, threads=threads)

        counts = {"written": 0, "absent": 0, "failed": 0}

        def on_result(results):                    # driver-side, single writer
            for batch_out in results:
                rows = []
                for body_id, status, payload in batch_out:
                    counts[status] = counts.get(status, 0) + 1
                    rows.append((body_id, status,
                                 None if status != "failed" else payload,
                                 payload if status == "written" else None))
                db.apply_skel_batch(rows)

        block_map(batches, worker, client=client, npartitions=npartitions,
                  on_result=on_result)
    finally:
        db.close()
    return {"db": db_path, "n_bodies": len(bodies), "processed": len(todo),
            "skeleton_dir": skeleton_dir, **counts}
