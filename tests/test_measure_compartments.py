"""The compartment join: exact joint counts, label-0 kept, verified against totals."""

import numpy as np
import pytest

from neu_morpho.measure.compartments import (joint_counts, semantic_label_names,
                                             somatic_labels, sweep_compartments,
                                             verify_compartments)
from neu_morpho.measure.driver import sweep_volumes
from neu_morpho.metrics_db import MetricsDB


def _write(path, vol, dtype="uint64"):
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, dtype,
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


# --------------------------------------------------------------------------- #
# the pure joint count
# --------------------------------------------------------------------------- #
def test_joint_counts_keeps_label_zero():
    """Unlabelled tissue is a real category, and keeping it is what makes
    `sum over labels == total` an identity rather than an approximation."""
    seg = np.array([[[1, 1, 1, 2]]], dtype=np.uint64)
    sem = np.array([[[3, 5, 0, 3]]], dtype=np.uint32)
    got = joint_counts(seg, sem)
    assert got == {(1, 3): 1, (1, 5): 1, (1, 0): 1, (2, 3): 1}
    assert sum(v for (b, _), v in got.items() if b == 1) == 3      # body 1's total


def test_joint_counts_drops_background_segmentation():
    seg = np.array([[[0, 0, 7]]], dtype=np.uint64)
    sem = np.array([[[3, 5, 5]]], dtype=np.uint32)
    assert joint_counts(seg, sem) == {(7, 5): 1}


def test_joint_counts_survives_a_large_body_id():
    """The packed key is `body << 4 | label`; a real id is ~1e9, nowhere near overflow."""
    big = np.uint64(2 ** 40 + 12345)
    seg = np.full((1, 1, 3), big, dtype=np.uint64)
    sem = np.array([[[3, 3, 5]]], dtype=np.uint32)
    assert joint_counts(seg, sem) == {(int(big), 3): 2, (int(big), 5): 1}


def test_joint_counts_refuses_an_out_of_range_label():
    seg = np.ones((1, 1, 1), dtype=np.uint64)
    sem = np.array([[[99]]], dtype=np.uint32)
    with pytest.raises(ValueError, match="exceeds"):
        joint_counts(seg, sem)


def test_joint_counts_refuses_misaligned_blocks():
    with pytest.raises(ValueError, match="not aligned"):
        joint_counts(np.ones((2, 2, 2), np.uint64), np.ones((2, 2, 3), np.uint32))


# --------------------------------------------------------------------------- #
# label naming, read from the source rather than guessed
# --------------------------------------------------------------------------- #
def _write_semantic_props(root, mapping):
    from neu_vol import location
    location.write_json(root, {"@type": "neuroglancer_multiscale_volume",
                               "segment_properties": "segment_properties"}, "info")
    location.write_json(root,
                        {"@type": "neuroglancer_segment_properties",
                         "inline": {"ids": [str(k) for k in mapping],
                                    "properties": [{"id": "label", "type": "label",
                                                    "values": list(mapping.values())}]}},
                        "segment_properties", "info")


def test_labels_come_from_the_source(tmp_path):
    root = str(tmp_path / "sem")
    _write_semantic_props(root, {1: "neuropil", 3: "nucleus", 5: "soma"})
    assert semantic_label_names(root) == {1: "neuropil", 3: "nucleus", 5: "soma"}
    assert somatic_labels(root) == [3, 5]          # nucleus, soma — in that order


def test_a_missing_somatic_label_raises(tmp_path):
    """Guessing the integers would produce a plausible non-somatic volume that
    excluded the wrong thing."""
    root = str(tmp_path / "sem")
    _write_semantic_props(root, {1: "neuropil", 2: "fiber-bundle"})
    with pytest.raises(ValueError, match="no label named"):
        somatic_labels(root)


# --------------------------------------------------------------------------- #
# end to end, against the volume sweep's totals
# --------------------------------------------------------------------------- #
def _pair(tmp_path):
    seg = np.zeros((32, 32, 32), np.uint64)
    seg[4:12, 4:12, 4:12] = 100
    seg[20:28, 6:10, 6:10] = 200
    seg[12:20, 24:28, 24:28] = 300      # crosses the z-block boundary at 16
    sem = np.zeros((32, 32, 32), np.uint32)
    sem[4:8, 4:12, 4:12] = 3            # half of body 100 is nucleus
    sem[8:12, 4:12, 4:12] = 5           # the other half soma
    sem[20:28, 6:10, 6:10] = 1          # body 200 all neuropil
    return (_write(str(tmp_path / "seg.zarr"), seg),
            _write(str(tmp_path / "sem.zarr"), sem, dtype="uint32"), seg, sem)


def test_end_to_end_matches_the_volume_totals(tmp_path):
    seg_spec, sem_spec, seg, _ = _pair(tmp_path)
    db = str(tmp_path / "m.sqlite")
    sweep_volumes(seg_spec, db, voxel_size=(8, 8, 8), block=16, client=None)
    out = sweep_compartments(seg_spec, sem_spec, db, sem_shape=(32, 32, 32),
                             block=16, client=None)
    assert out["n_bodies"] == 3

    d = MetricsDB(db, read_only=True)
    rows = {(int(b), int(l)): int(n) for b, l, n in d.con.execute(
        "SELECT body_id, label, voxel_count FROM body_compartments")}
    assert rows[(100, 3)] == 4 * 8 * 8          # nucleus half
    assert rows[(100, 5)] == 4 * 8 * 8          # soma half
    assert rows[(200, 1)] == 8 * 4 * 4
    assert rows[(300, 0)] == 8 * 4 * 4          # no semantic label there
    d.close()

    v = verify_compartments(db)
    assert v["n_mismatched"] == 0 and v["voxels_missing"] == 0


def test_verify_names_the_shortfall_when_blocks_are_missing(tmp_path):
    """The point of the check: a filtered pass that lost a block is reported, not silent."""
    seg_spec, sem_spec, _, _ = _pair(tmp_path)
    db = str(tmp_path / "m.sqlite")
    sweep_volumes(seg_spec, db, voxel_size=(8, 8, 8), block=16, client=None)
    # deliberately withhold the block containing body 200
    keep_blocks = [b for b in np.ndindex(2, 2, 2) if b != (1, 0, 0)]
    sweep_compartments(seg_spec, sem_spec, db, sem_shape=(32, 32, 32), block=16,
                       blocks=keep_blocks, client=None)
    v = verify_compartments(db)
    assert v["n_mismatched"] == 1
    assert v["voxels_missing"] == 128
    assert v["worst"][0][0] == 200


def test_resume_does_not_double_count(tmp_path):
    seg_spec, sem_spec, _, _ = _pair(tmp_path)
    db = str(tmp_path / "m.sqlite")
    kw = dict(sem_shape=(32, 32, 32), block=16, client=None)
    sweep_compartments(seg_spec, sem_spec, db, **kw)
    second = sweep_compartments(seg_spec, sem_spec, db, resume=True, **kw)
    assert second["processed"] == 0
    d = MetricsDB(db, read_only=True)
    assert d.con.execute("SELECT voxel_count FROM body_compartments "
                         "WHERE body_id=100 AND label=3").fetchone()[0] == 256
    d.close()


def test_every_block_stage_is_visible_to_the_reporter(tmp_path):
    """A stage that records progress and reports none is indistinguishable from a hang.

    The reporter used to iterate a hardcoded ("index", "sweep"), so the compartment pass
    wrote its total and its blocks and displayed neither. Both now derive from
    BLOCK_STAGES, and this asserts the registry and the tables agree.
    """
    db_path = str(tmp_path / "m.sqlite")
    d = MetricsDB(db_path)
    try:
        for stage, table in MetricsDB.BLOCK_STAGES.items():
            d._ensure_compartments()          # creates compartment_progress
            assert d._has_table(table) or table == "compartment_progress", \
                f"{stage} has no progress table {table}"
        assert set(d.stage_counts()) == set(MetricsDB.BLOCK_STAGES)
    finally:
        d.close()


def test_a_compartment_run_shows_up_in_progress(tmp_path, capsys):
    seg_spec, sem_spec, _, _ = _pair(tmp_path)
    db = str(tmp_path / "m.sqlite")
    sweep_compartments(seg_spec, sem_spec, db, sem_shape=(32, 32, 32), block=16,
                       client=None)

    from neu_morpho import cli
    rc = cli.main(["measure", "progress", db])
    out = capsys.readouterr().out
    assert rc == 0
    assert "compartments" in out
    assert "8/8" in out or "100.0%" in out        # total was recorded, so a fraction
