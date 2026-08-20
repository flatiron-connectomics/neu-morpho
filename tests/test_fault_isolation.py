"""Per-task fault isolation, and the asymmetry it deliberately preserves.

Stage 2 (one task = one body) isolates failures: bodies are independent, so
skipping a bad one costs exactly that body and is recorded. Stage 1 (one task =
one block) does NOT, because stage 2 aggregates across blocks — a silently
skipped block truncates every body passing through it and erases any body wholly
inside it, while the output still looks complete.
"""

import importlib
import json
import os

import numpy as np
import pytest

from neu_morpho.config import MeshConfig, OutputConfig, SkeletonConfig
from neu_morpho.ops import meshify, skeletonize_segments
from neu_morpho.ops._progress import FAILED, guarded, is_complete


def _write_seg_zarr(path, vol):
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _three_body_volume():
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:28, 4:10, 4:10] = 7
    vol[4:28, 14:20, 14:20] = 8
    vol[4:28, 24:30, 24:30] = 9
    return vol


SKEL_CFG = dict(anisotropy=(8.0, 8.0, 8.0), block_shape=(16, 16, 16), const=30.0,
                dust_threshold=0, postprocess_dust_nm=0.0, postprocess_tick_nm=0.0)
MESH_CFG = dict(mesh_scale=0, block_shape=(16, 16, 16), num_lods=2, decimation_fraction=1.0)


# --------------------------------------------------------------------------- #
# The resume trap
# --------------------------------------------------------------------------- #
def test_is_complete_excludes_failed_where_is_done_does_not(tmp_path):
    """A 'failed' record must be retried, not treated as done.

    blockrun's Manifest.is_done tests key PRESENCE, so resuming on is_done
    would permanently skip exactly the tasks that need retrying.
    """
    from blockrun import Manifest

    path = str(tmp_path / "m.jsonl")
    m = Manifest(path)
    m.reset()
    m.record("g", [(1, "written"), (2, FAILED), (3, "empty")])
    m.close()

    m2 = Manifest(path).load()
    assert m2.is_done("g", 2) is True                 # the trap
    assert is_complete(m2, "g", 1) is True
    assert is_complete(m2, "g", 2) is False           # ...which is_complete avoids
    assert is_complete(m2, "g", 3) is True
    assert is_complete(m2, "g", 99) is False          # never seen


def test_guarded_converts_exceptions_and_passes_success_through():
    def ok(key):
        return (key, "written", {"a": 1}, {})

    def boom(key):
        raise RuntimeError("kaboom")

    assert guarded(ok, 5) == (5, "written", {"a": 1}, {})
    key, status, metrics, info = guarded(boom, 5)
    assert (key, status, metrics) == (5, FAILED, None)
    assert "kaboom" in info["error"] and "RuntimeError" in info["error"]
    assert "Traceback" in info["traceback"]


# --------------------------------------------------------------------------- #
# Stage 2 isolates
# --------------------------------------------------------------------------- #
def test_skeleton_fuse_failure_is_isolated_and_retried(tmp_path, monkeypatch):
    mod = importlib.import_module("neu_morpho.ops.skeletonize_segments")

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    cfg = SkeletonConfig(**SKEL_CFG)

    real = mod.fuse_body
    calls = {"n": 0}

    def flaky(fragments, cfg_, body_id=None, stats=None):
        if body_id == 8:                       # one poisoned body
            calls["n"] += 1
            raise RuntimeError("synthetic fusion failure")
        return real(fragments, cfg_, body_id=body_id, stats=stats)

    monkeypatch.setattr(mod, "fuse_body", flaky)
    s1 = skeletonize_segments(src, out, cfg, client=None)

    # the run completed, the other bodies are on disk, the failure is recorded
    assert s1["failed_bodies"] == [8]
    assert s1["status_counts"].get(FAILED) == 1
    assert s1["status_counts"].get("written") == 2
    assert os.path.exists(os.path.join(s1["out_dir"], "7"))
    assert os.path.exists(os.path.join(s1["out_dir"], "9"))
    assert not os.path.exists(os.path.join(s1["out_dir"], "8"))

    rows = [json.loads(ln) for ln in open(s1["failures_path"])]
    assert rows[0]["body_id"] == 8 and "synthetic fusion failure" in rows[0]["error"]

    # resume retries ONLY the failed body, and it succeeds once unpoisoned
    monkeypatch.setattr(mod, "fuse_body", real)
    s2 = skeletonize_segments(src, out, cfg, client=None)
    assert s2["num_bodies_fused"] == 1                # not 3 — the others stayed done
    assert s2["failed_bodies"] == []
    assert os.path.exists(os.path.join(s2["out_dir"], "8"))
    assert s2["status_counts"].get(FAILED) is None    # the record was superseded


def test_mesh_assemble_failure_is_isolated_and_retried(tmp_path, monkeypatch):
    mod = importlib.import_module("neu_morpho.ops.meshify")

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    cfg = MeshConfig(**MESH_CFG)

    real = mod.assemble_body
    seen = {"n": 0}

    def flaky(frags, cfg_):
        # bodies are assembled in sorted id order, so the 2nd call is body 8
        seen["n"] += 1
        if seen["n"] == 2:
            raise RuntimeError("synthetic assemble failure")
        return real(frags, cfg_)

    monkeypatch.setattr(mod, "assemble_body", flaky)
    m1 = meshify(src, out, cfg, mesh_voxel_size=(8, 8, 8), client=None)
    assert m1["failed_bodies"] == [8]
    assert m1["status_counts"].get(FAILED) == 1
    assert m1["status_counts"].get("written") == 2        # others still produced
    assert not os.path.exists(os.path.join(m1["out_dir"], "8"))

    monkeypatch.setattr(mod, "assemble_body", real)
    m2 = meshify(src, out, cfg, mesh_voxel_size=(8, 8, 8), client=None)
    assert m2["num_bodies_assembled"] == 1                # only the failure is retried
    assert m2["failed_bodies"] == []
    assert os.path.exists(os.path.join(m2["out_dir"], "8"))


def test_failed_body_metrics_are_not_written(tmp_path, monkeypatch):
    """A failed body must not leave half-written metrics behind."""
    mod = importlib.import_module("neu_morpho.ops.skeletonize_segments")
    from neu_morpho.metrics_db import MetricsDB

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    db_path = str(tmp_path / "m.db")

    def flaky(fragments, cfg_, body_id=None, stats=None):
        raise RuntimeError("all bodies fail")

    monkeypatch.setattr(mod, "fuse_body", flaky)
    skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG), db_path=db_path, client=None)

    db = MetricsDB(db_path)
    rows = db.con.execute("SELECT COUNT(*) FROM bodies WHERE cable_length_nm IS NOT NULL")
    assert rows.fetchone()[0] == 0
    db.close()


# --------------------------------------------------------------------------- #
# The failure breaker
# --------------------------------------------------------------------------- #
def test_is_systemic_classification():
    import errno as E

    from neu_morpho.ops._progress import is_systemic

    assert is_systemic(MemoryError())
    assert is_systemic(ImportError("no module"))
    assert is_systemic(OSError(E.ENOSPC, "No space left on device"))
    assert is_systemic(OSError(E.EDQUOT, "Disk quota exceeded"))
    # a one-off bad body, or a transient read — isolate these, do not abort
    assert not is_systemic(ValueError("degenerate mesh"))
    assert not is_systemic(RuntimeError("kimimaro barfed"))
    assert not is_systemic(OSError(E.EIO, "transient read error"))


def test_guarded_flags_systemic_errors():
    def oom(key):
        raise MemoryError()

    def odd(key):
        raise ValueError("weird body")

    assert guarded(oom, 1)[3]["systemic"] is True
    assert guarded(odd, 1)[3]["systemic"] is False


def test_breaker_trips_on_consecutive_failures_and_resets_on_success():
    from neu_morpho.ops._progress import FailureBreaker, StageAborted

    b = FailureBreaker(max_consecutive=3)
    for i in range(2):
        b.failure(i, {"error": "boom"})
    b.check()                       # 2 < 3, still going
    b.success()                     # a success resets the streak
    for i in range(2):
        b.failure(i, {"error": "boom"})
    b.check()                       # streak restarted, still 2

    b.failure(99, {"error": "boom"})
    with pytest.raises(StageAborted, match="3 consecutive"):
        b.check()
    assert b.total == 5             # total counts every failure, streak or not


def test_breaker_trips_immediately_on_systemic():
    from neu_morpho.ops._progress import FailureBreaker, StageAborted

    b = FailureBreaker(max_consecutive=1000)
    b.failure(7, {"error": "MemoryError: ", "systemic": True})
    with pytest.raises(StageAborted, match="systemic"):
        b.check()


def test_breaker_can_be_disabled():
    from neu_morpho.ops._progress import FailureBreaker

    b = FailureBreaker(max_consecutive=0)
    for i in range(50):
        b.failure(i, {"error": "boom"})
    b.check()                       # never trips
    assert b.total == 50


def test_stage_aborts_after_consecutive_failures_but_keeps_diagnostics(tmp_path, monkeypatch):
    """The breaker must not cost us the record of what already happened."""
    mod = importlib.import_module("neu_morpho.ops.skeletonize_segments")
    from neu_morpho.ops._progress import StageAborted

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))

    def always_fail(*a, **k):
        raise ValueError("everything is broken")

    monkeypatch.setattr(mod, "fuse_body", always_fail)
    with pytest.raises(StageAborted, match="2 consecutive"):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG),
                             max_consecutive_failures=2, client=None)

    # written in `finally`, so the abort still leaves the tracebacks behind
    path = os.path.join(str(tmp_path / "out"), "failures.skel.jsonl")
    rows = [json.loads(ln) for ln in open(path)]
    assert len(rows) == 2 and "everything is broken" in rows[0]["error"]

    # and the failures are recorded as retryable, not as done
    from blockrun import Manifest
    m = Manifest(os.path.join(str(tmp_path / "out"), "progress.skel.jsonl")).load()
    assert not is_complete(m, "skel-fuse", rows[0]["body_id"])


def test_systemic_error_aborts_on_the_first_body(tmp_path, monkeypatch):
    mod = importlib.import_module("neu_morpho.ops.skeletonize_segments")
    from neu_morpho.ops._progress import StageAborted

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    seen = {"n": 0}

    def oom(*a, **k):
        seen["n"] += 1
        raise MemoryError()

    monkeypatch.setattr(mod, "fuse_body", oom)
    with pytest.raises(StageAborted, match="systemic"):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG),
                             max_consecutive_failures=1000, client=None)
    assert seen["n"] == 1          # stopped at the first body, not all three


def test_successes_in_the_aborting_batch_still_get_their_metrics(tmp_path, monkeypatch):
    """A trip must not leave a body's skeleton on disk without its DB metrics."""
    mod = importlib.import_module("neu_morpho.ops.skeletonize_segments")
    from neu_morpho.metrics_db import MetricsDB
    from neu_morpho.ops._progress import StageAborted

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    db_path = str(tmp_path / "m.db")
    real = mod.fuse_body

    def fail_first(fragments, cfg_, body_id=None, stats=None):
        if body_id == 7:
            raise ValueError("first body fails")
        return real(fragments, cfg_, body_id=body_id, stats=stats)

    monkeypatch.setattr(mod, "fuse_body", fail_first)
    # trips on the very first failure, but bodies 8 and 9 already succeeded
    with pytest.raises(StageAborted):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG), db_path=db_path,
                             max_consecutive_failures=1, client=None)

    db = MetricsDB(db_path)
    written = {r[0] for r in db.con.execute(
        "SELECT body_id FROM bodies WHERE cable_length_nm IS NOT NULL")}
    db.close()
    skel_dir = os.path.join(out.volume_dir(), out.skeleton_dir)   # inside the volume
    on_disk = {int(f) for f in os.listdir(skel_dir) if f.isdigit()}
    assert on_disk == written, "a skeleton on disk without metrics in the DB"


# --------------------------------------------------------------------------- #
# Stage 1 does NOT isolate — the asymmetry is the point
# --------------------------------------------------------------------------- #
def test_stage1_block_failure_still_aborts(tmp_path, monkeypatch):
    """Block failures must stay loud: stage 2 aggregates across blocks, so a
    silently skipped block yields a truncated body that still looks complete."""
    mod = importlib.import_module("neu_morpho.ops.skeletonize_segments")

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))

    def boom(*a, **k):
        raise RuntimeError("block read failure")

    monkeypatch.setattr(mod, "skeletonize_block", boom)
    with pytest.raises(RuntimeError, match="block read failure"):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG),
                             stages=("skel-chunk",), client=None)


def test_stage1_mesh_block_failure_still_aborts(tmp_path, monkeypatch):
    mod = importlib.import_module("neu_morpho.ops.meshify")

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))

    def boom(*a, **k):
        raise RuntimeError("block mesh failure")

    monkeypatch.setattr(mod, "mesh_block", boom)
    with pytest.raises(RuntimeError, match="block mesh failure"):
        meshify(src, out, MeshConfig(**MESH_CFG), mesh_voxel_size=(8, 8, 8),
                stages=("chunk",), client=None)


def test_stage1_progress_before_an_abort_is_durable(tmp_path, monkeypatch):
    """A crash must not cost the blocks already recorded."""
    mod = importlib.import_module("neu_morpho.ops.skeletonize_segments")

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    cfg = SkeletonConfig(**SKEL_CFG)

    real = mod.skeletonize_block
    seen = {"n": 0}

    def boom_after_three(*a, **k):
        seen["n"] += 1
        if seen["n"] > 3:
            raise RuntimeError("dies partway")
        return real(*a, **k)

    monkeypatch.setattr(mod, "skeletonize_block", boom_after_three)
    with pytest.raises(RuntimeError):
        skeletonize_segments(src, out, cfg, stages=("skel-chunk",), client=None)

    assert seen["n"] == 4          # 3 succeeded and were recorded, the 4th raised

    retried = {"n": 0}

    def counting(*a, **k):
        retried["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "skeletonize_block", counting)
    resumed = skeletonize_segments(src, out, cfg, stages=("skel-chunk",), client=None)
    # 8 blocks total; the 3 recorded before the crash are not redone
    assert retried["n"] == 5
    assert sum(resumed["chunk_counts"].values()) == 8
