"""End-to-end: synthetic volume -> skel-chunk -> skel-fuse -> precomputed skeletons.

The load-bearing case is a body spanning several blocks: its fragments must weld
into ONE component covering the full extent (that is what ``fix_borders`` +
``join_close_components`` buy us), not a pile of disconnected block-length stubs.
"""

import os
from dataclasses import replace

import numpy as np
import pytest

from em_seg_morpho.config import OutputConfig, SkeletonConfig
from em_seg_morpho.ops.skeletonize_segments import skeletonize_segments
from em_seg_morpho.skeleton import fuse_body, skeletonize_block


def _write_seg_zarr(path, vol):
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _load(out_dir, body):
    from osteoid import Skeleton
    with open(os.path.join(out_dir, str(body)), "rb") as f:
        return Skeleton.from_precomputed(f.read(), segid=body)


VS = (8.0, 8.0, 8.0)


def _volume():
    # 96 x 32 x 32 at 8 nm, blocks of 32 along z -> 3 blocks
    vol = np.zeros((96, 32, 32), np.uint64)
    vol[:, 14:18, 14:18] = 7          # rod spanning all 3 blocks
    vol[40:48, 4:9, 4:9] = 9          # blob wholly inside block 1
    return vol


def _cfg():
    return SkeletonConfig(anisotropy=VS, skeleton_scale=0, block_shape=(32, 32, 32),
                          const=30.0, dust_threshold=0,
                          postprocess_dust_nm=0.0, postprocess_tick_nm=0.0)


def test_skeletonize_end_to_end(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))
    db_path = str(tmp_path / "metrics.db")

    summary = skeletonize_segments(src, out, _cfg(), db_path=db_path, client=None)
    out_dir = summary["out_dir"]

    assert os.path.exists(os.path.join(out_dir, "info"))
    assert summary["num_blocks"] == 3
    assert summary["status_counts"].get("written") == 2

    # the spanning body produced one fragment per block...
    assert len(os.listdir(os.path.join(summary["chunked_dir"], "7"))) == 3
    assert len(os.listdir(os.path.join(summary["chunked_dir"], "9"))) == 1

    # ...and fused into a single connected skeleton over the full z extent.
    rod = _load(out_dir, 7)
    assert len(rod.components()) == 1
    z = rod.vertices[:, 2]                      # stored xyz -> z is column 2
    assert z.min() <= 8 and z.max() >= 744      # 95 voxels * 8 nm = 760
    assert rod.cable_length() > 700

    blob = _load(out_dir, 9)
    bz = blob.vertices[:, 2]
    assert 300 <= bz.min() and bz.max() <= 400  # block 1: z voxels 40..48 -> 320..376 nm


def test_metrics_land_in_db(tmp_path):
    from em_seg_morpho.metrics_db import MetricsDB

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))
    db_path = str(tmp_path / "metrics.db")
    skeletonize_segments(src, out, _cfg(), db_path=db_path, client=None)

    db = MetricsDB(db_path)
    row = db.con.execute(
        "SELECT cable_length_nm, n_branches, n_tips FROM bodies WHERE body_id=7").fetchone()
    db.close()
    assert row is not None
    cable, branches, tips = row
    assert cable > 700                          # the FUSED length, not one block's
    assert branches == 0 and tips == 2          # an unbranched rod


def test_allowlist_restricts_bodies(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))
    summary = skeletonize_segments(src, out, _cfg(), allowlist=[7], client=None)

    assert os.path.exists(os.path.join(summary["out_dir"], "7"))
    assert not os.path.exists(os.path.join(summary["out_dir"], "9"))
    assert not os.path.exists(os.path.join(summary["chunked_dir"], "9"))


def test_resume_skips_done_and_stage2_reuses_fragments(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _volume())
    out = OutputConfig(dst=str(tmp_path / "out"))

    first = skeletonize_segments(src, out, _cfg(), stages=("skel-chunk",), client=None)
    assert first["chunk_counts"].get("written") == 3
    assert not os.path.exists(os.path.join(first["out_dir"], "7"))

    # stage 2 alone reuses the fragments already on disk
    second = skeletonize_segments(src, out, _cfg(), stages=("skel-fuse",), client=None)
    assert second["num_bodies_fused"] == 2
    assert os.path.exists(os.path.join(second["out_dir"], "7"))

    # a third full run has nothing left to do
    third = skeletonize_segments(src, out, _cfg(), client=None)
    assert third["num_bodies_fused"] == 0


def test_fuse_body_welds_seams_not_just_concatenates():
    """Guard the property the whole design rests on: fragments become one tree."""
    from em_seg_morpho.skeleton import fuse_body, skeletonize_block

    cfg = _cfg()
    vol = _volume()
    frags = [skeletonize_block(vol[z0:z0 + 32], (z0, 0, 0), cfg)[7] for z0 in (0, 32, 64)]
    assert len(frags) == 3

    fused = fuse_body(frags, cfg, body_id=7)
    assert fused.id == 7
    assert len(fused.components()) == 1
    # welding adds the seam edges, so the fused cable exceeds the sum of the parts
    assert fused.cable_length() > sum(f.cable_length() for f in frags)


def _split_body():
    """One label, two blobs ~400 nm apart — a segmentation split, not a seam."""
    vol = np.zeros((32, 32, 96), np.uint64)
    vol[8:24, 8:24, 4:20] = 7
    vol[8:24, 8:24, 70:86] = 7
    return vol


def test_default_join_does_not_bridge_a_segmentation_split():
    """The default radius is seam-scale: distant pieces stay apart, and stay PRESENT.

    An unbounded join would connect them with one straight edge between the
    nearest vertex pair — inventing hundreds of nm of cable that no biology
    produced, and inflating cable_length_nm for exactly the bodies whose
    segmentation is least trustworthy.
    """
    from em_seg_morpho.skeleton import fuse_body, join_radius_nm, skeletonize_block

    cfg = _cfg()
    assert join_radius_nm(cfg) == 16.0                  # 2 x 8 nm voxel
    frags = list(skeletonize_block(_split_body(), (0, 0, 0), cfg).values())

    fused = fuse_body(frags, cfg, body_id=7)
    assert len(fused.components()) == 2                 # kept, just disconnected
    bounded_cable = fused.cable_length()

    unbounded = fuse_body(frags, replace(cfg, join_radius_nm=float("inf")), body_id=7)
    assert len(unbounded.components()) == 1
    assert unbounded.cable_length() > 1.5 * bounded_cable   # the invented edge


def test_join_radius_zero_defers_to_postprocess():
    """radius=0 skips the explicit join; postprocess's own join still welds seams."""
    from em_seg_morpho.skeleton import fuse_body, skeletonize_block

    cfg = replace(_cfg(), join_radius_nm=0)
    vol = _volume()
    frags = [skeletonize_block(vol[z0:z0 + 32], (z0, 0, 0), cfg)[7] for z0 in (0, 32, 64)]

    fused = fuse_body(frags, cfg, body_id=7)
    assert len(fused.components()) == 1                 # seams still welded
    assert fused.cable_length() > 700
    # ...but it still refuses to bridge a genuine split
    split = list(skeletonize_block(_split_body(), (0, 0, 0), cfg).values())
    assert len(fuse_body(split, cfg, body_id=7).components()) == 2


def _arbor_with_twigs():
    """A trunk with short lateral twigs and a detached speck of the same label."""
    vol = np.zeros((96, 48, 48), np.uint64)
    vol[4:92, 22:26, 22:26] = 7                      # trunk
    for z in (20, 40, 60, 80):                       # short twigs -> tick fodder
        vol[z:z + 2, 26:34, 23:25] = 7
    vol[50:52, 40:42, 40:42] = 7                     # detached speck -> dust fodder
    return vol


def _arbor_frags(cfg):
    vol = _arbor_with_twigs()
    return [skeletonize_block(vol[z0:z0 + 32], (z0, 0, 0), cfg)[7] for z0 in range(0, 96, 32)]


def test_default_tick_threshold_runs():
    """Regression: the shipped defaults must actually survive postprocess.

    kimimaro's compiled remove_ticks rejects int64 edges / float64 vertices, and
    both arise normally (skeletonize returns int64 edges; join_close_components
    concatenates more). Every earlier test set tick=0, which short-circuits
    remove_ticks entirely — so the shipped tick default was never executed.
    """
    # 32 nm voxels so the trunk clears the 1500 nm dust default
    cfg = SkeletonConfig(anisotropy=(32.0, 32.0, 32.0), block_shape=(32, 32, 32),
                         const=30.0, dust_threshold=0)
    assert (cfg.postprocess_dust_nm, cfg.postprocess_tick_nm) == (1500.0, 3000.0)

    fused = fuse_body(_arbor_frags(cfg), cfg, body_id=7)      # must not raise
    assert fused is not None and len(fused.vertices) > 0
    assert fused.edges.dtype == np.uint32 and fused.vertices.dtype == np.float32


def test_remove_loops_breaks_the_edge_dtype():
    """Why fuse_body inlines postprocess's steps instead of calling it.

    postprocess is dust -> loops -> join -> ticks, and ``remove_loops`` rebuilds
    edges as **int64**. ``remove_ticks`` reaches compiled code declaring a
    ``uint32_t`` buffer, so it raises — not always (it depends on the arbor), but
    often enough that the shipped tick default is unusable. Inlining lets
    normalize_dtypes run between those two steps.

    If this starts failing, kimimaro has fixed the dtype upstream and the
    inlining could be reconsidered.
    """
    import kimimaro
    from kimimaro.post import remove_dust, remove_loops

    from em_seg_morpho.skeleton import join_radius_nm, normalize_dtypes

    cfg = replace(_cfg(), postprocess_dust_nm=0.0, postprocess_tick_nm=1000.0)
    frags = [skeletonize_block(_volume()[z0:z0 + 32], (z0, 0, 0), cfg)[7]
             for z0 in range(0, 96, 32)]
    clean = normalize_dtypes(
        kimimaro.join_close_components(frags, radius=join_radius_nm(cfg)).consolidate())
    assert clean.edges.dtype == np.uint32 and clean.vertices.dtype == np.float32

    # the step that breaks it, in isolation
    assert remove_loops(remove_dust(clean.clone(), 0.0)).edges.dtype == np.int64

    # ...and the resulting failure through the stock entry point
    with pytest.raises(ValueError, match="Buffer dtype mismatch"):
        kimimaro.postprocess(clean.clone(), dust_threshold=0.0, tick_threshold=1000.0)

    # our path handles the same input
    fused = fuse_body(frags, cfg, body_id=7)
    assert fused is not None and fused.edges.dtype == np.uint32


def test_fuse_body_matches_kimimaro_postprocess():
    """fuse_body inlines postprocess's steps to measure them — pin the equivalence.

    Only comparable with tick=0, the one configuration stock postprocess survives
    (see above); that still covers dust, loop removal and the radius join.
    """
    import kimimaro

    from em_seg_morpho.skeleton import (fuse_body, join_radius_nm, normalize_dtypes,
                                        skeletonize_block)

    cfg = replace(_cfg(), postprocess_dust_nm=200.0, postprocess_tick_nm=0.0)
    frags = _arbor_frags(cfg)

    ours = fuse_body(frags, cfg, body_id=7)

    joined = kimimaro.join_close_components(frags, radius=join_radius_nm(cfg))
    theirs = kimimaro.postprocess(normalize_dtypes(joined.consolidate()),
                                  dust_threshold=cfg.postprocess_dust_nm,
                                  tick_threshold=cfg.postprocess_tick_nm)

    assert len(ours.vertices) == len(theirs.vertices)
    assert ours.cable_length() == pytest.approx(theirs.cable_length(), rel=1e-6)


def test_tick_removal_prunes_twigs():
    """The inlined tick path does what it should: fewer tips, less cable."""
    cfg = replace(_cfg(), postprocess_dust_nm=0.0)
    frags = _arbor_frags(cfg)

    keep = fuse_body(frags, replace(cfg, postprocess_tick_nm=0.0), body_id=7)
    prune = fuse_body(frags, replace(cfg, postprocess_tick_nm=1000.0), body_id=7)

    assert len(prune.terminals()) < len(keep.terminals())
    assert prune.cable_length() < keep.cable_length()


def test_fusion_stats_attribute_what_was_dropped():
    from em_seg_morpho.skeleton import fusion_stats_summary

    cfg = replace(_cfg(), postprocess_dust_nm=200.0, postprocess_tick_nm=1000.0)
    frags = _arbor_frags(cfg)

    stats = {}
    fuse_body(frags, cfg, body_id=7, stats=stats)

    assert stats["n_fragments"] == 3
    assert stats["seam_join_comps_merged"] >= 2          # 3 block fragments -> 1 trunk
    assert stats["dust_comps_dropped"] >= 1              # the detached speck
    assert stats["dust_cable_dropped_nm"] > 0
    assert stats["tick_branches_removed"] >= 1           # the lateral twigs
    assert stats["tick_cable_removed_nm"] > 0
    assert not stats["deleted"]

    summary = fusion_stats_summary([stats])
    assert summary["n_bodies"] == 1 and summary["bodies_deleted"] == 0
    assert 0 < summary["dropped_cable_fraction"] < 1
    # nothing dropped when both thresholds are off
    off = {}
    fuse_body(frags, replace(cfg, postprocess_dust_nm=0.0, postprocess_tick_nm=0.0),
              body_id=7, stats=off)
    assert off["dust_comps_dropped"] == 0 and off["tick_cable_removed_nm"] == 0


def test_op_reports_fusion_stats(tmp_path):
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _arbor_with_twigs())
    out = OutputConfig(dst=str(tmp_path / "out"))
    stats_path = str(tmp_path / "fusion.jsonl")

    cfg = replace(_cfg(), block_shape=(32, 32, 32), postprocess_dust_nm=200.0,
                  postprocess_tick_nm=1000.0)
    summary = skeletonize_segments(src, out, cfg, fusion_stats_path=stats_path, client=None)

    fs = summary["fusion_stats"]
    assert fs["n_bodies"] == 1
    assert fs["dust_comps_dropped"] >= 1 and fs["tick_branches_removed"] >= 1
    assert fs["inferred_cable_fraction"] > 0        # the seam edges are inferred

    import json
    rows = [json.loads(ln) for ln in open(stats_path)]
    assert len(rows) == 1 and rows[0]["body_id"] == 7


def test_dust_threshold_deletion_is_reported_not_silent(tmp_path):
    """A body shorter than postprocess_dust_nm is deleted — surface it as "dust"."""
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[8:24, 14:18, 14:18] = 7          # ~128 nm of cable, well under the threshold
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), vol)
    out = OutputConfig(dst=str(tmp_path / "out"))

    cfg = replace(_cfg(), postprocess_dust_nm=1500.0)
    summary = skeletonize_segments(src, out, cfg, client=None)

    assert summary["status_counts"].get("dust") == 1
    assert summary["status_counts"].get("written", 0) == 0
    assert not os.path.exists(os.path.join(summary["out_dir"], "7"))

    # with the threshold off, the same body survives
    out2 = OutputConfig(dst=str(tmp_path / "out2"))
    s2 = skeletonize_segments(src, out2, _cfg(), client=None)
    assert s2["status_counts"].get("written") == 1
