#!/usr/bin/env python
"""Choose the bodies for the skeletonization benchmark and write a manifest.

These bodies are **regression fixtures**, not a statistical sample: their job is
to expose porting bugs in a NeuTu-style skeletonizer, so the set is deliberately
weighted toward the cases that break things rather than toward what is typical.

Two axes, both read from the production run's ``metrics.db`` (no volume access):

``max_radius_nm``    thickness. Thick bodies are the problematic ones — blobs and
                     somata are where radius convention and node placement diverge
                     most between tools — so **half the set is thick** (>= p90).
``cable_length_nm``  size, varied within each thickness band so the set is not all
                     large arbors.

Bodies 6308993 and 18052382 are pinned: every number in
``docs/skeletonization-comparison.md`` is measured on them, and dropping them
would strand that evidence.

Bounding boxes come from the local stage-1 skeleton fragments
(``<work_dir>/skel_chunked/<body>/*.skel``, zyx nm), which is free — the pipeline
already wrote them. They matter because **NeuTu aborts above 1,073,741,824 voxels
of bounding box** (not voxel count), so a sparse arbor with a wide bbox trips it;
candidates above ``--max-bbox-vox`` are rejected here rather than at run time.

    python scripts/pick_benchmark_bodies.py --work-dir <dir> --out bodies.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3

import numpy as np

# NeuTu's own limit (ONEGIGA in the source); the default cap leaves ~35% headroom.
NEUTU_BBOX_LIMIT = 1_073_741_824

PINNED = [6308993, 18052382]

# (name, cable percentile window, max_radius percentile window). Half the bands
# are thick (max_r >= p90); the thick ones vary in size, since a thick blob on a
# short stub and one on a large arbor fail differently.
BANDS = [
    ("thick-huge",     (0.90, 1.00), (0.995, 1.001)),
    ("thick-large",    (0.80, 0.95), (0.97, 0.995)),
    ("thick-medium",   (0.40, 0.65), (0.95, 0.99)),
    ("thick-small",    (0.05, 0.30), (0.90, 0.97)),
    ("typical-large",  (0.85, 0.97), (0.55, 0.75)),
    ("typical-medium", (0.40, 0.60), (0.40, 0.60)),
    ("typical-small",  (0.05, 0.25), (0.35, 0.55)),
    ("thin-large",     (0.85, 0.97), (0.05, 0.20)),
    ("thin-medium",    (0.40, 0.60), (0.02, 0.12)),
    ("thin-small",     (0.05, 0.20), (0.02, 0.12)),
]


def skel_bbox_vox(skel_dir: str, body: int, voxel_nm: float):
    """Tight bbox in scale voxels from a body's stage-1 skeleton fragments.

    The skeleton lies inside the body, so this under-estimates by roughly the
    local radius; the caller pads. Returns ``None`` if the body has no fragments.
    """
    from osteoid import Skeleton

    d = os.path.join(skel_dir, str(body))
    if not os.path.isdir(d):
        return None
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name in os.listdir(d):
        with open(os.path.join(d, name), "rb") as f:
            v = Skeleton.from_precomputed(f.read(), segid=int(body)).vertices
        if len(v):
            lo = np.minimum(lo, v.min(0))
            hi = np.maximum(hi, v.max(0))
    if not np.isfinite(lo).all():
        return None
    return lo / voxel_nm, hi / voxel_nm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True,
                    help="production run work dir (holds metrics.db + skel_chunked/)")
    ap.add_argument("--out", required=True, help="manifest JSON to write")
    ap.add_argument("--voxel-nm", type=float, default=32.0, help="voxel size at --scale")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--margin-vox", type=int, default=8,
                    help="pad the skeleton bbox; must exceed the largest radius in voxels")
    ap.add_argument("--max-bbox-vox", type=int, default=700_000_000,
                    help=f"reject candidates above this (NeuTu aborts at {NEUTU_BBOX_LIMIT:,})")
    ap.add_argument("--per-band-probe", type=int, default=9,
                    help="candidates whose bbox is measured per band")
    a = ap.parse_args()

    skel_dir = os.path.join(a.work_dir, "skel_chunked")
    con = sqlite3.connect(f"file:{a.work_dir}/metrics.db?mode=ro", uri=True)
    rows = con.execute(
        "SELECT body_id, cable_length_nm, max_radius_nm, n_mesh_components, mesh_area_nm2 "
        "FROM bodies WHERE mesh_verts IS NOT NULL").fetchall()
    arr = np.array(rows, dtype=float)
    bid = arr[:, 0].astype(np.int64)
    cable, maxr, ncomp, area = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    # percentile rank, not value, so the bands are density-based
    crank = np.argsort(np.argsort(cable)) / (len(cable) - 1)
    rrank = np.argsort(np.argsort(maxr)) / (len(maxr) - 1)

    def describe(i, band, lo, hi):
        return {
            "body_id": int(bid[i]), "band": band,
            "cable_length_nm": float(cable[i]), "cable_pctl": round(float(crank[i]), 3),
            "max_radius_nm": float(maxr[i]), "max_radius_pctl": round(float(rrank[i]), 3),
            "n_mesh_components": int(ncomp[i]),
            # mean cross-sectional radius, tube approximation A = 2*pi*r*L. Unlike
            # max_radius this reflects the processes rather than one blob.
            "mean_radius_nm": round(float(area[i] / (2 * np.pi * cable[i])), 1),
            "bbox_lo_vox": [int(v) for v in np.floor(lo)],
            "bbox_hi_vox": [int(v) for v in np.ceil(hi)],
            "bbox_vox": int(np.prod(np.ceil(hi) - np.floor(lo))),
        }

    picked: dict[int, dict] = {}
    for body in PINNED:
        i = int(np.nonzero(bid == body)[0][0])
        lo, hi = skel_bbox_vox(skel_dir, body, a.voxel_nm)
        picked[body] = describe(i, "pinned", np.maximum(0, lo - a.margin_vox),
                                hi + a.margin_vox)

    for band, (c0, c1), (r0, r1) in BANDS:
        sel = np.nonzero((crank >= c0) & (crank < c1) & (rrank >= r0) & (rrank < r1))[0]
        sel = np.array([i for i in sel if int(bid[i]) not in picked], dtype=int)
        if not len(sel):
            print(f"{band:16s} EMPTY BAND — no body matches")
            continue
        # probe evenly across the band by cable rank, then take the MEDIAN bbox:
        # the smallest would systematically pick compact bodies, and compactness
        # is a third axis this set should not silently collapse
        probe = sel[np.argsort(crank[sel])][np.linspace(
            0, len(sel) - 1, min(a.per_band_probe, len(sel))).astype(int)]
        cands = []
        for i in probe:
            bb = skel_bbox_vox(skel_dir, int(bid[i]), a.voxel_nm)
            if bb is None:
                continue
            lo = np.maximum(0, bb[0] - a.margin_vox)
            hi = bb[1] + a.margin_vox
            vol = float(np.prod(hi - lo))
            if vol <= a.max_bbox_vox:
                cands.append((vol, int(i), lo, hi))
        if not cands:
            print(f"{band:16s} no candidate under --max-bbox-vox")
            continue
        cands.sort(key=lambda t: t[0])
        vol, i, lo, hi = cands[len(cands) // 2]
        picked[int(bid[i])] = describe(i, band, lo, hi)

    out = sorted(picked.values(), key=lambda d: -d["max_radius_nm"])
    n_thick = sum(1 for d in out if d["max_radius_pctl"] >= 0.90)
    manifest = {
        "scale": a.scale, "voxel_nm": a.voxel_nm, "margin_vox": a.margin_vox,
        "source_work_dir": a.work_dir, "n_bodies": len(out), "n_thick": n_thick,
        "bodies": out,
    }
    with open(a.out, "w") as f:
        json.dump(manifest, f, indent=1)

    hdr = (f"{'band':16s} {'body':>10s} {'cable_nm':>10s} {'p':>5s} {'max_r':>6s} {'p':>5s} "
           f"{'mean_r':>7s} {'ncomp':>6s} {'bbox_vox':>14s}")
    print(hdr)
    print("-" * len(hdr))
    for d in out:
        print(f"{d['band']:16s} {d['body_id']:10d} {d['cable_length_nm']:10,.0f} "
              f"{d['cable_pctl']:5.2f} {d['max_radius_nm']:6.0f} {d['max_radius_pctl']:5.2f} "
              f"{d['mean_radius_nm']:7.1f} {d['n_mesh_components']:6d} {d['bbox_vox']:14,d}")
    print(f"\n{len(out)} bodies, {n_thick} thick (max_radius >= p90) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
