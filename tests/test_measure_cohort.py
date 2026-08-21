"""Cohort loading, validation and the nested-cohort join.

Two properties here are worth more than the rest: a **duplicate body id is refused**,
because it doubles that body's weight in any aggregate taken after a join and nothing
downstream can see it; and a **float id column is refused**, because a float64 body id
above 2**53 is silently rounded.
"""

import numpy as np
import pandas as pd
import pytest

from neu_morpho.measure.cohort import (Cohort, cohort_from_table, load_cohort,
                                       membership, save_cohort)


def _table(ids, **cols):
    return pd.DataFrame({"body_id": np.asarray(ids, dtype=np.int64), **cols})


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_a_duplicate_body_id_is_refused():
    """It would double that body's weight in every aggregate after a join."""
    with pytest.raises(ValueError, match="duplicate body id"):
        cohort_from_table(_table([1, 2, 2, 3]), name="c", dataset="d")


def test_the_error_names_the_offenders():
    with pytest.raises(ValueError, match=r"e\.g\. \[7\]"):
        cohort_from_table(_table([7, 7]), name="c", dataset="d")


def test_a_float_id_column_is_refused():
    """A float64 id above 2**53 is silently rounded — the same hazard that makes
    parquet the default over csv for nullable integer columns elsewhere."""
    table = pd.DataFrame({"body_id": [1.0, 2.0]})
    with pytest.raises(ValueError, match="silently rounded"):
        cohort_from_table(table, name="c", dataset="d")


def test_a_table_without_the_id_column_says_what_it_has():
    table = pd.DataFrame({"bodyid": [1, 2], "instance": ["a", "b"]})
    with pytest.raises(ValueError, match="no 'body_id' column"):
        cohort_from_table(table, name="c", dataset="d")


def test_negative_ids_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        cohort_from_table(_table([1, -2]), name="c", dataset="d")


def test_large_uint64_ids_survive():
    """Body ids are uint64 in precomputed, so an id past 2**53 must round-trip."""
    big = 2**60 + 12345
    c = cohort_from_table(_table([big]), name="c", dataset="d")
    assert c.ids == (big,)


# --------------------------------------------------------------------------- #
# attributes are the point
# --------------------------------------------------------------------------- #
def test_a_cohort_keeps_the_attributes_that_selected_it():
    """So results can be grouped by cell type later without re-fetching DVID, and so
    the reason a body qualified is not thrown away."""
    table = _table([10, 20], instance=["Tm2_A2(L)", "CAm(R)_truncated"],
                   is_truncated=[False, True])
    c = cohort_from_table(table, name="non_noise", dataset="megaphragma")
    frame = c.to_frame()
    assert list(frame.columns[:2]) == ["dataset", "cohort"]
    assert frame["instance"].tolist() == ["Tm2_A2(L)", "CAm(R)_truncated"]
    assert frame["is_truncated"].tolist() == [False, True]


def test_a_bare_id_list_still_makes_a_usable_cohort():
    c = load_cohort([3, 1, 2], name="ids", dataset="d")
    assert sorted(c.ids) == [1, 2, 3] and c.attrs is None
    # it measures identically; it just cannot be grouped by anything
    assert c.to_frame()["body_id"].tolist() == list(c.ids)


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #
def test_parquet_round_trip_keeps_ids_and_attrs(tmp_path):
    table = _table([5, 6], instance=["a", "b"])
    c = cohort_from_table(table, name="complete", dataset="megaphragma")
    path = save_cohort(c, str(tmp_path / "c.parquet"))
    back = load_cohort(path, dataset="megaphragma")

    assert back.ids == c.ids
    assert back.name == "c"                       # from the file stem
    assert "instance" in back.attrs.columns


def test_an_id_file_loads_through_the_existing_allowlist_reader(tmp_path):
    p = tmp_path / "ids.csv"
    p.write_text("body_id\n11\n22\n33\n")
    c = load_cohort(str(p), dataset="megaphragma")
    assert sorted(c.ids) == [11, 22, 33]


def test_a_csv_with_attributes_is_read_as_a_table(tmp_path):
    p = tmp_path / "cohort.csv"
    p.write_text("body_id,instance\n11,foo\n22,bar\n")
    c = load_cohort(str(p), dataset="megaphragma")
    assert sorted(c.ids) == [11, 22]
    assert c.attrs is not None and "instance" in c.attrs.columns


# --------------------------------------------------------------------------- #
# nested cohorts
# --------------------------------------------------------------------------- #
def test_membership_is_long_form_so_a_body_can_be_in_both():
    """The complete-cell cohort is a SUBSET of the non-noise one, and a body in both
    has identical measurements — so it is measured once and joined twice."""
    a = Cohort("non_noise", "megaphragma", (1, 2, 3, 4))
    b = Cohort("complete", "megaphragma", (2, 4))
    m = membership([a, b])

    assert len(m) == 6
    assert set(m[m.cohort == "complete"].body_id) == {2, 4}
    # body 2 appears in both, once per cohort
    assert (m.body_id == 2).sum() == 2


def test_filtering_a_measurement_table_to_a_cohort_is_a_join():
    """The reason the two cohorts do not mean two measurement runs."""
    rows = pd.DataFrame({"dataset": "megaphragma", "body_id": [1, 2, 3, 4],
                         "variant": "all", "cable_length_nm": [10.0, 20.0, 30.0, 40.0]})
    m = membership([Cohort("complete", "megaphragma", (2, 4))])
    joined = rows.merge(m, on=["dataset", "body_id"], how="inner")

    assert joined["body_id"].tolist() == [2, 4]
    assert joined["cable_length_nm"].sum() == 60.0


def test_membership_of_nothing_is_an_empty_frame_with_the_right_columns():
    m = membership([])
    assert list(m.columns) == ["dataset", "body_id", "cohort"] and len(m) == 0
