import os

import pytest

from em_seg_morpho.allowlist import load_allowlist
from em_seg_morpho import fragments as F


def test_load_allowlist_none_means_all():
    assert load_allowlist(None) is None


def test_load_allowlist_iterable():
    assert load_allowlist([1, 2, "3"]) == {1, 2, 3}


def test_load_allowlist_csv(tmp_path):
    p = tmp_path / "ids.csv"
    p.write_text("id,size\n1001,50\n1002,40\n\n# comment\n1003,\n")
    assert load_allowlist(str(p)) == {1001, 1002, 1003}


def test_load_allowlist_empty_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("id\n")
    with pytest.raises(ValueError):
        load_allowlist(str(p))


def test_fragment_paths_and_list_bodies(tmp_path):
    d = str(tmp_path / "chunked")
    assert F.list_bodies(d) == []                     # missing dir -> empty
    # fragment path layout: <chunked>/<body>/<iz>_<iy>_<ix>.<fmt>
    assert F.fragment_path(d, 42, (1, 2, 3)).endswith("42/1_2_3.drc")
    for body in (5, 100, 7):
        os.makedirs(F.body_dir(d, body), exist_ok=True)
    os.makedirs(os.path.join(d, "not_a_body"), exist_ok=True)   # ignored
    assert F.list_bodies(d) == [5, 7, 100]            # sorted ints only
