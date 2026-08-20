"""Manifest status policy shared by the ops.

Two things live here because both two-stage ops need them and getting either
wrong is silent:

1. :func:`is_complete` — resume must skip *successful* tasks, not merely recorded
   ones. blockrun's ``Manifest.is_done`` tests key **presence**, so a recorded
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

import errno
import traceback
from typing import Any, Callable

FAILED = "failed"

# Exceptions that say "the environment is broken", not "this body is odd" — every
# remaining task would fail the same way, so isolating them just burns the run.
_SYSTEMIC_TYPES = (MemoryError, ImportError)
# OSError is mostly transient (a flaky read is worth isolating); these errnos are
# not — they will not fix themselves before the next task.
_SYSTEMIC_ERRNOS = frozenset({errno.ENOSPC,     # no space left on device
                              errno.EDQUOT,     # over disk quota
                              errno.EROFS})     # read-only filesystem


class StageAborted(RuntimeError):
    """Stage stopped early by the failure breaker (not a per-task failure)."""


class StaleManifest(RuntimeError):
    """A manifest records completed work whose output is no longer there."""


def check_manifest_matches_output(manifest: Any, out_dir: str, *, stage: str,
                                  progress_path: str, resume: bool) -> None:
    """Refuse to resume a manifest that has outlived the data it describes.

    Bookkeeping lives in the POSIX work dir while the data may sit in an object
    store, so the two no longer share a fate. Clear the destination and the
    manifest still says every task is done: the run would skip all of them, write
    nothing, and exit reporting success. Nothing raises, and the loss only shows
    up as an empty layer in the viewer.

    The probe is the stage's *own* ``info`` — the file this op writes on every
    run — not the segmentation ``info``, so meshing into a volume whose labels
    were never exported (a legitimate ``--stages mesh`` run) is not mistaken for
    a cleared destination.

    Only checked when resuming; ``resume=False`` is an explicit fresh start.
    """
    if not resume:
        return
    recorded = sum(manifest.counts().values())
    if not recorded:
        return                                   # nothing to be stale about
    from ..precomputed import volume_exists

    if volume_exists(out_dir):
        return
    raise StaleManifest(
        f"{progress_path} records {recorded} completed {stage} task(s), but there is "
        f"no 'info' at {out_dir} — the destination looks like it was cleared while "
        f"the manifest survived. Resuming would skip every task and report success "
        f"having written nothing. Re-run with --no-resume to start over (or delete "
        f"the manifest), or point --dst back at the intended destination.")


def is_systemic(exc: BaseException) -> bool:
    """Would this error recur for every remaining task?

    ``MemoryError`` is the ambiguous one: at stage 2 a single enormous body can
    genuinely exhaust memory while its neighbours are fine. It is treated as
    systemic anyway, because a process that has just hit OOM is not in a state
    worth trusting for the next 40,000 tasks — and if it really was one body,
    the run resumes past it once you exclude or re-scale it.
    """
    if isinstance(exc, _SYSTEMIC_TYPES):
        return True
    return isinstance(exc, OSError) and exc.errno in _SYSTEMIC_ERRNOS


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
    the traceback (so it is not lost on a worker) and a ``systemic`` flag the
    driver's breaker uses to stop immediately.

    ``KeyboardInterrupt`` and ``SystemExit`` derive from ``BaseException``, so
    they are deliberately **not** caught here and still abort the run at once.
    """
    try:
        return worker(key)
    except Exception as exc:                       # noqa: BLE001 - deliberate isolation
        return (key, FAILED, None, {"error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(),
                                    "systemic": is_systemic(exc)})


class FailureBreaker:
    """Stops a stage when failures stop looking like isolated bad bodies.

    Two triggers: a *systemic* exception (see :func:`is_systemic`), which aborts
    at once, and ``max_consecutive`` failures in a row, which catches the slower
    version of the same thing — a misconfiguration or a dying filesystem that
    makes everything fail. A single success resets the streak.

    It records a reason rather than raising inline, so the caller can finish
    applying a batch's successful results before :meth:`check` aborts. Under dask,
    batches complete out of order, so "consecutive" means consecutive in
    *completion* order — a good enough proxy, not a precise statement about the
    task list.
    """

    def __init__(self, max_consecutive: int = 10):
        self.max_consecutive = max_consecutive
        self.consecutive = 0
        self.total = 0
        self.reason: str | None = None

    def success(self) -> None:
        self.consecutive = 0

    def failure(self, key: Any, info: dict | None) -> None:
        info = info or {}
        self.total += 1
        self.consecutive += 1
        if self.reason is not None:
            return
        if info.get("systemic"):
            self.reason = (f"systemic error on task {key} — every remaining task would "
                           f"likely fail the same way: {info.get('error')}")
        elif self.max_consecutive and self.consecutive >= self.max_consecutive:
            self.reason = (f"{self.consecutive} consecutive task failures, most recently "
                           f"on {key}: {info.get('error')}")

    def check(self) -> None:
        """Raise :class:`StageAborted` if a trigger fired."""
        if self.reason is not None:
            raise StageAborted(
                f"{self.reason}. Stopping the stage rather than failing every remaining "
                f"task; {self.total} failure(s) recorded so far. Progress is durable — "
                f"fix the cause and re-run to resume (failed tasks are retried).")


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
