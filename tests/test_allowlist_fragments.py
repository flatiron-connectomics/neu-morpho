import os

import pytest

from neu_morpho.allowlist import load_allowlist
from neu_morpho import fragments as F


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
    # fragment path layout: <chunked>/h<body % FANOUT, 3 hex>/<body>/<iz>_<iy>_<ix>.<fmt>
    assert F.fragment_path(d, 42, (1, 2, 3)).endswith("h02a/42/1_2_3.drc")
    for body in (5, 100, 7):
        os.makedirs(F.body_dir(d, body), exist_ok=True)
    os.makedirs(os.path.join(d, "not_a_body"), exist_ok=True)   # ignored
    assert F.list_bodies(d) == [5, 7, 100]            # sorted ints only
    assert F.count_bodies(d) == 3


def test_bodies_sharing_a_bucket_are_all_found(tmp_path):
    d = str(tmp_path / "chunked")
    same = [3, 3 + F.FANOUT, 3 + 7 * F.FANOUT]        # one bucket, three bodies
    for body in same + [4]:
        os.makedirs(F.body_dir(d, body), exist_ok=True)
    assert len(os.listdir(d)) == 2                    # two buckets, not four entries
    assert F.list_bodies(d) == sorted(same + [4])
    assert F.count_bodies(d) == 4


def test_a_bucket_never_parses_as_a_body_id(tmp_path):
    # "100" is a valid 3-hex bucket AND a body id; the prefix keeps them apart.
    assert not F.bucket_name(256).isdigit()
    assert F.count_bodies(str(tmp_path / "missing")) is None


def test_a_store_in_the_old_flat_layout_is_refused_not_read_as_empty(tmp_path):
    # Old stores put body dirs at the top level. Reading only the buckets would see no
    # bodies, and a resumed stage 2 would assemble nothing and report success.
    d = tmp_path / "chunked"
    (d / "42").mkdir(parents=True)
    (d / "42" / "0_0_0.drc").write_bytes(b"x")
    with pytest.raises(F.OldFragmentLayout, match="42"):
        F.list_bodies(str(d))
    with pytest.raises(F.OldFragmentLayout):
        F.count_bodies(str(d))
