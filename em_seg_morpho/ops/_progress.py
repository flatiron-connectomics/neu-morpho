"""Manifest status policy shared by the ops.

Two things live here because both two-stage ops need them and getting either
wrong is silent:

1. :func:`is_complete` — resume must skip *successful* tasks, not merely recorded
   ones. em-blockrun's ``Manifest.is_done`` tests key **presence**, so a recorded
   ``failed`` reads as done and the task that most needs retrying would never run
   again.
2. :func:`guarded` — per-task fault isolation, applied to **stage-2 workers only**.

Why only stage 2. Stage-2 tasks are one *body* each and bodies are independent,
so skipping a bad one costs exactly that body and is recorded. Stage-1 tasks are
one *block* each, and stage 2 **aggregates across blocks** — a silently skipped
block does not leave a hole in one block, it truncates every body passing through
it and erases outright any body lying wholly inside it, with the output still
looking complete. That failure reaches the science; a crash does not. So stage 1
stays fail-fast, where resume already makes relaunch cheap.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

FAILED = "failed"


def is_complete(manifest: Any, group: str, key: Any) -> bool:
    """Has this task finished *successfully*? (``failed`` records must be retried.)"""
    return manifest.status(group, key) not in (None, FAILED)


def group_counts(manifest: Any, group: str) -> dict[str, int]:
    """Per-group status tallies.

    ``Manifest.counts()`` aggregates every group in the file, so a two-stage op
    reporting it conflates "blocks chunked" with "bodies assembled" and the
    totals exceed the block count.
    """
    out: dict[str, int] = {}
    for key in manifest.done_keys(group):
        status = manifest.status(group, key)
        out[status] = out.get(status, 0) + 1
    return out


def guarded(worker: Callable[[Any], tuple], key: Any) -> tuple:
    """Run a stage-2 worker, converting an exception into a ``failed`` result.

    Workers return ``(key, status, metrics, info)``; on failure ``info`` carries
    the traceback so the driver can report it instead of losing it on a worker.
    """
    try:
        return worker(key)
    except Exception as exc:                       # noqa: BLE001 - deliberate isolation
        return (key, FAILED, None, {"error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()})


def write_failures(path: str, failures: list[dict]) -> str | None:
    """Dump per-task failures as JSONL (tracebacks would bloat the run summary)."""
    if not failures:
        return None
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")
    return path
