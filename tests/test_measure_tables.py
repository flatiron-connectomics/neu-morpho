"""tables.py: variants, properties parsing, comparison — no kernel needed."""

import numpy as np
import pytest

pytest.importorskip("pandas")

from neu_morpho.measure import tables
from neu_morpho.metrics_db import MetricsDB


def _db(tmp_path, *, with_compartments=True):
    path = str(tmp_path / "m.sqlite")
    db = MetricsDB(path)
    vox = 32.0 ** 3
    # body 1: 1000 voxels, 300 nucleus, 200 soma. body 2: 500 voxels, no compartments.
    db.apply_counts_block("0_0_0", {1: 1000, 2: 500}, vox)
    db.update_body(1, cable_length_nm=10_000.0)
    db.update_body(2, cable_length_nm=5_000.0)
    if with_compartments:
        db.apply_compartment_block("0_0_0", {(1, 1): 500, (1, 3): 300, (1, 5): 200,
                                            (2, 1): 500})
    db.close()
    return path


def test_variants_subtract_the_right_compartments(tmp_path):
    df = tables.load_bodies(_db(tmp_path), voxel_nm=32.0).set_index("body_id")
    vox_um3 = 32.0 ** 3 / 1e9
    assert df.loc[1, "volume_um3"] == pytest.approx(1000 * vox_um3)
    assert df.loc[1, "nucleus_um3"] == pytest.approx(300 * vox_um3)
    assert df.loc[1, "volume_um3_all"] == pytest.approx(1000 * vox_um3)
    assert df.loc[1, "volume_um3_minus_nucleus"] == pytest.approx(700 * vox_um3)
    assert df.loc[1, "volume_um3_minus_soma_nucleus"] == pytest.approx(500 * vox_um3)
    # body 2 has no somatic compartments, so every variant is its total
    assert df.loc[2, "volume_um3_minus_soma_nucleus"] == pytest.approx(500 * vox_um3)


def test_diameter_is_area_equivalent_from_v_over_l(tmp_path):
    df = tables.load_bodies(_db(tmp_path)).set_index("body_id")
    vol_nm3 = 1000 * 32.0 ** 3
    want = 2 * np.sqrt((vol_nm3 / 10_000.0) / np.pi)
    assert df.loc[1, "diameter_nm_all"] == pytest.approx(want)


def test_no_compartment_table_yields_only_the_all_variant(tmp_path):
    """A dataset with no semantic masks must still load, with the variants it has."""
    df = tables.load_bodies(_db(tmp_path, with_compartments=False))
    assert "volume_um3_all" in df.columns
    assert "volume_um3_minus_nucleus" not in df.columns
    assert "diameter_nm_minus_soma_nucleus" not in df.columns


def test_bodies_never_measured_are_dropped(tmp_path):
    """voxel_count DEFAULT 0 cannot distinguish 'not measured' from 'measured empty'."""
    path = _db(tmp_path)
    db = MetricsDB(path)
    db.update_body(999, cable_length_nm=1.0)      # a skel-only row, no voxels
    db.close()
    df = tables.load_bodies(path)
    assert 999 not in set(df["body_id"])


def test_zero_cable_gives_nan_not_inf(tmp_path):
    path = _db(tmp_path)
    db = MetricsDB(path)
    db.apply_counts_block("1_0_0", {7: 100}, 32.0 ** 3)
    db.close()
    df = tables.load_bodies(path).set_index("body_id")
    assert np.isnan(df.loc[7, "diameter_nm_all"])


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #
def _props(root, sub, obj):
    from neu_vol import location
    location.write_json(root, obj, sub, "info")


def test_tags_become_namespaced_columns(tmp_path):
    root = str(tmp_path / "src")
    _props(root, "tags_property",
           {"@type": "neuroglancer_segment_properties",
            "inline": {"ids": ["1", "2"],
                       "properties": [{"id": "tags", "type": "tags",
                                       "tags": ["superclass:ol", "somaSide:L",
                                                "somaSide:R", "cropped"],
                                       "values": [[0, 1], [0, 2, 3]]}]}})
    df = tables.load_segment_properties(root, "tags_property").set_index("body_id")
    assert df.loc[1, "superclass"] == "ol" and df.loc[1, "somaSide"] == "L"
    assert df.loc[2, "somaSide"] == "R"
    assert df.loc[2, "tag"] == "cropped"        # a bare tag keeps its own column


def test_several_sources_are_merged_on_body_id(tmp_path):
    """Published datasets split properties across sources with differing id sets."""
    root = str(tmp_path / "src")
    _props(root, "a", {"@type": "neuroglancer_segment_properties",
                       "inline": {"ids": ["1", "2"],
                                  "properties": [{"id": "type", "type": "label",
                                                  "values": ["Tm2", "L1"]}]}})
    _props(root, "b", {"@type": "neuroglancer_segment_properties",
                       "inline": {"ids": ["2", "3"],
                                  "properties": [{"id": "syn_pre", "type": "number",
                                                  "values": [5, 9]}]}})
    df = tables.load_segment_properties(root, "a", "b").set_index("body_id")
    assert set(df.index) == {1, 2, 3}
    assert df.loc[1, "type"] == "Tm2"
    assert df.loc[3, "syn_pre"] == 9
    assert df["type"].isna().loc[3]


# --------------------------------------------------------------------------- #
# summarise / compare / histogram
# --------------------------------------------------------------------------- #
def test_compare_is_a_ratio_of_statistics_not_a_statistic_of_ratios():
    import pandas as pd
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"x": [1.0, 1.0, 1.0]})
    out = tables.compare(a, b, ["x"], labels=("a", "b")).set_index("column")
    assert out.loc["x", "ratio"] == pytest.approx(2.0)      # median 2 / median 1


def test_summarize_groups_and_orders_by_size():
    import pandas as pd
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 10.0]})
    out = tables.summarize(df, ["v"], by="g")
    assert list(out["g"]) == ["a", "b"]                     # largest group first
    assert out.loc[0, "v_p50"] == pytest.approx(2.0)


def test_log_histogram_conserves_weight():
    import pandas as pd
    df = pd.DataFrame({"d": [1.0, 50.0, 5000.0], "w": [1.0, 2.0, 4.0]})
    h = tables.log_histogram(df, "d", lo=10.0, hi=1000.0, nbins=8, weight="w")
    assert h["weight"].sum() == pytest.approx(7.0)          # under+over included
    assert h.iloc[0]["weight"] == pytest.approx(1.0)        # underflow
    assert h.iloc[-1]["weight"] == pytest.approx(4.0)       # overflow
