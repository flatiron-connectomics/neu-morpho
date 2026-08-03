"""Stage-1 tracer selection: units, parity of shape, and the isotropy guard."""
from __future__ import annotations

import numpy as np
import pytest

from em_seg_morpho import skeleton
from em_seg_morpho.config import SkeletonConfig


def test_default_tracer_is_neutu():
    """Pin the default: which tracer runs by default is a decision, not an accident.

    Every other test here passes ``tracer=`` explicitly, so nothing else would
    notice this flipping — and the two tracers produce different skeletons.
    """
    assert SkeletonConfig().tracer == "neutu"
    assert SkeletonConfig().neutu_cost == "voxel"


def test_driver_default_tracer_matches_the_config_default():
    """The driver always passes --tracer explicitly, so the two can drift apart.

    If they do, ``SkeletonConfig()`` and a bare driver invocation quietly produce
    different skeletons from the same command.
    """
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "run_morpho_slurm.py"
    spec = importlib.util.spec_from_file_location("_driver_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod._parse_args(["--src", "x", "--dst", "y", "--work-dir", "z"])
    assert args.tracer == SkeletonConfig().tracer
    assert args.neutu_cost == SkeletonConfig().neutu_cost


def _two_body_block(n=48, r=4):
    """One block holding two labelled tubes."""
    m = np.zeros((n, n, n), dtype=np.uint64)
    zz, yy, xx = np.indices(m.shape)
    m[(zz >= 4) & (zz < n - 4) & ((yy - 12) ** 2 + (xx - 12) ** 2 <= r * r)] = 7
    m[(zz >= 4) & (zz < n - 4) & ((yy - 34) ** 2 + (xx - 34) ** 2 <= r * r)] = 9
    return np.asfortranarray(m)


def _cfg(**kw):
    return SkeletonConfig(anisotropy=(32.0, 32.0, 32.0), dust_threshold=0, **kw)


@pytest.mark.parametrize("tracer", ["kimimaro", "neutu"])
def test_both_tracers_return_both_bodies_in_nm(tracer):
    m = _two_body_block()
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer=tracer))
    assert set(out) == {7, 9}
    for body_id, skel in out.items():
        v = np.asarray(skel.vertices, float)
        # nm, not voxels: a 48-voxel block at 32 nm spans ~1536 nm
        assert v[:, 0].max() > 200.0, f"{tracer}: vertices look like voxels, not nm"
        assert v.max() <= 48 * 32.0 + 1e-3


def test_neutu_radii_are_in_nm_not_voxels():
    """The conversion that would silently publish radii 32x too small.

    kimimaro is given anisotropy and returns nm; neutu_trace works in voxels, so
    radii must be scaled here. Positions would still be correct if this were
    missed, so no geometric test would catch it.
    """
    m = _two_body_block(r=4)
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu"))
    radii = np.concatenate([np.asarray(s.radii, float) for s in out.values()])
    # a radius-4-voxel tube at 32 nm/voxel is ~128 nm, nowhere near 4
    assert radii.max() > 60.0, f"radii look like voxels: max {radii.max():.2f}"
    assert radii.max() < 32.0 * 8


def test_block_origin_is_applied_by_both_tracers():
    m = _two_body_block()
    origin = (10, 20, 30)
    for tracer in ("kimimaro", "neutu"):
        at0 = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer=tracer))
        off = skeleton.skeletonize_block(m, origin, _cfg(tracer=tracer))
        d0 = np.asarray(at0[7].vertices, float).min(axis=0)
        d1 = np.asarray(off[7].vertices, float).min(axis=0)
        shift = np.asarray(origin, float) * 32.0
        assert np.allclose(d1 - d0, shift, atol=1e-3), f"{tracer} lost the origin"


def test_neutu_tracer_rejects_anisotropic_voxels():
    m = _two_body_block()
    cfg = SkeletonConfig(tracer="neutu", anisotropy=(40.0, 32.0, 32.0))
    with pytest.raises(ValueError, match="isotropic"):
        skeleton.skeletonize_block(m, (0, 0, 0), cfg)


def test_unknown_tracer_raises():
    m = _two_body_block()
    with pytest.raises(ValueError, match="unknown tracer"):
        skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="nope"))


@pytest.mark.parametrize("cost", ["voxel", "edge"])
def test_neutu_cost_mode_reaches_the_tracer(cost):
    """cfg.neutu_cost must actually change the routing, not just be accepted.

    A config field that is silently ignored is worse than none — it reads as enabled.
    """
    m = _two_body_block()
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu",
                                                        neutu_cost=cost))
    assert set(out) == {7, 9}
    for skel in out.values():
        v = np.asarray(skel.vertices, float)
        assert len(v) and v[:, 0].max() > 200.0        # still nm, still traced


def test_neutu_cost_is_validated_not_silently_ignored():
    m = _two_body_block()
    with pytest.raises(ValueError, match="voxel|edge"):
        skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu",
                                                      neutu_cost="nonsense"))
