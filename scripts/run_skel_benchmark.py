#!/usr/bin/env python
"""Run NeuTu and kimimaro over the benchmark bodies and score both.

The NeuTu SWCs this writes are the **regression target** for the Python
NeuTu-style skeletonizer described in ``docs/skeletonization-plan.md``; the
kimimaro results are the incumbent, so a port that regresses is visible.

**Everything is in voxel units**, matching the 2026-07-30 reference outputs:
NeuTu's SWC is already in mask voxel coordinates, and kimimaro is run with
``anisotropy=(1,1,1)`` and ``const`` converted to voxels, so its vertices and
radii come back in voxels too. ``skelmetrics`` rasterises into the mask array, so
it needs voxel units regardless. Multiply by ``voxel_nm`` to report nm.

Scoring always passes ``edges``. Sphere-stamping vertices instead of sweeping
capsules along edges reversed the tool ranking once already — see the Corrections
section of ``docs/skeletonization-comparison.md``.

Bodies run in parallel, largest first (the tail is one big body, so starting it
last would idle the pool). Each body is isolated: a failure is recorded and the
rest continue.

    python scripts/run_skel_benchmark.py --manifest bodies.json --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from neu_morpho import neutu_io, skelmetrics          # noqa: E402

NEUTU_BIN = os.path.expanduser("~/miniforge3/envs/managed_neutu/bin/neutu")

# kimimaro production, as configured in SkeletonConfig. `const` is physical (nm)
# there; here it is voxels because anisotropy is (1,1,1).
KIMI_PRODUCTION = {"scale": 1.5, "const_nm": 150.0,
                   "pdrf_scale": 100000, "pdrf_exponent": 4}


def run_one(rec: dict, out_dir: str, voxel_nm: float, neutu_bin: str,
            minlen: int, kimi: bool, port: bool = True,
            port_kw: dict | None = None) -> dict:
    body = rec["body_id"]
    res = {"body_id": body, "band": rec["band"], "methods": {}}
    mask = np.load(rec["mask_path"])
    res["mask_voxels"] = int(mask.sum())
    res["mask_shape"] = list(mask.shape)

    # -- NeuTu ------------------------------------------------------------
    swc = os.path.join(out_dir, "neutu_swc", f"b{body}_neutu_L{minlen}.swc")
    os.makedirs(os.path.dirname(swc), exist_ok=True)
    tmp = os.path.join(out_dir, "_tmp", str(body))
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    try:
        r = neutu_io.run_neutu(mask, swc, neutu=neutu_bin,
                               params={"minimalLength": minlen}, workdir=tmp)
        res["methods"][f"neutu_L{minlen}"] = _score(
            mask, r["zyx"], r["radius"], r["edges"], time.time() - t0, voxel_nm,
            path=swc)
    except Exception as e:
        res["methods"][f"neutu_L{minlen}"] = {"error": f"{type(e).__name__}: {e}",
                                              "seconds": round(time.time() - t0, 1)}

    # -- the Python port (neu_morpho.neutu_trace) -----------------------
    if port:
        npz = os.path.join(out_dir, "port", f"b{body}_port.npz")
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        t0 = time.time()
        try:
            from neu_morpho import neutu_trace

            s = neutu_trace.skeletonize(mask, **(port_kw or {}))
            v = np.asarray(s.vertices, dtype=float)
            rad = np.asarray(s.radii, dtype=float)
            e = np.asarray(s.edges, dtype=int)
            np.savez_compressed(npz, vertices=v, radii=rad, edges=e)
            t_trace = time.time() - t0
            res["methods"]["port_neutu_cost"] = _score(
                mask, v, rad, e, t_trace, voxel_nm, path=npz)
        except Exception as e:
            v = None
            res["methods"]["port_neutu_cost"] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:],
                "seconds": round(time.time() - t0, 1)}

        # steps 2+3 on top of the same trace, so the node reduction is isolated
        if v is not None:
            for name, fn in (("port_region_sampled", "region_sample"),
                             ("port_simplified", "simplify")):
                t1 = time.time()
                try:
                    from neu_morpho import swc_simplify

                    sv, sr, se = getattr(swc_simplify, fn)(v, rad, e)
                    out = os.path.join(out_dir, "port", f"b{body}_{fn}.npz")
                    np.savez_compressed(out, vertices=sv, radii=sr, edges=se)
                    res["methods"][name] = _score(
                        mask, sv, sr, se, t_trace + (time.time() - t1), voxel_nm,
                        path=out)
                except Exception as e:
                    res["methods"][name] = {
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-1500:],
                        "seconds": round(time.time() - t1, 1)}

    # -- kimimaro production ----------------------------------------------
    if kimi:
        npz = os.path.join(out_dir, "kimimaro", f"b{body}_kimi_production.npz")
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        t0 = time.time()
        try:
            import kimimaro
            tp = {"scale": KIMI_PRODUCTION["scale"],
                  "const": KIMI_PRODUCTION["const_nm"] / voxel_nm,   # nm -> voxels
                  "pdrf_scale": KIMI_PRODUCTION["pdrf_scale"],
                  "pdrf_exponent": KIMI_PRODUCTION["pdrf_exponent"]}
            skels = kimimaro.skeletonize(mask.astype(np.uint64), anisotropy=(1, 1, 1),
                                         object_ids=[1], teasar_params=tp,
                                         dust_threshold=0, progress=False)
            s = skels.get(1)
            if s is None:
                raise RuntimeError("kimimaro returned no skeleton")
            v = np.asarray(s.vertices, dtype=float)
            rad = np.asarray(getattr(s, "radius", getattr(s, "radii")), dtype=float)
            e = np.asarray(s.edges, dtype=int)
            np.savez_compressed(npz, vertices=v, radii=rad, edges=e)
            res["methods"]["kimimaro_production"] = _score(
                mask, v, rad, e, time.time() - t0, voxel_nm, path=npz)
        except Exception as e:
            res["methods"]["kimimaro_production"] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:],
                "seconds": round(time.time() - t0, 1)}
    return res


def _score(mask, zyx, radii, edges, seconds, voxel_nm, path="", reference=None):
    """Score one skeleton. ``reference`` is (zyx, radii, edges) to agree with.

    Coverage and spill are recorded but are **not** the objective — coverage is
    confounded by branch count and rewards inventing neurites, and on a dense
    segmentation spill cannot separate reclaiming a false split from trespassing.
    ``agree_*`` is what to optimise; see ``skelmetrics.agreement``.
    """
    s = skelmetrics.score(mask, zyx, radii, edges)          # edges: never omit
    d = skelmetrics.radius_vs_edt(mask, zyx, radii)
    ag = {}
    if reference is not None:
        a = skelmetrics.agreement(zyx, radii, edges, *reference)
        ag = {f"agree_{k}": round(v, 4) for k, v in a.items()}
    return {**ag,
        "nodes": s["nodes"], "edges": int(len(edges)),
        "coverage": round(s["coverage"], 4), "spill": round(s["spill"], 4),
        "nodes_per_1k_covered": round(s["nodes_per_1k_covered"], 3),
        "max_radius_nm": round(float(np.max(radii)) * voxel_nm, 1),
        "median_radius_nm": round(float(np.median(radii)) * voxel_nm, 1),
        "mask_max_inscribed_nm": round(d["mask_max_inscribed"] * voxel_nm, 1),
        "radius_median_error_vox": round(d["median_error"], 3),
        "frac_over_2vox": round(d["frac_over_2vox"], 4),
        "seconds": round(seconds, 1), "path": path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--neutu-bin", default=NEUTU_BIN)
    ap.add_argument("--minlen", type=int, default=10, help="NeuTu minimalLength (voxels)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-kimimaro", action="store_true",
                    help="skip kimimaro — it is by far the slow half")
    ap.add_argument("--no-port", action="store_true",
                    help="skip neu_morpho.neutu_trace")
    ap.add_argument("--port-scale", type=float, default=None,
                    help="invalidation scale for the port (default: NeuTu's 1.0)")
    ap.add_argument("--port-const", type=float, default=None,
                    help="invalidation const in VOXELS for the port. NeuTu uses 2 "
                         "(maskExpansionRadius); larger values compensate for our "
                         "weaker target selection — see docs/skeletonization-plan.md")
    ap.add_argument("--port-min-length", type=float, default=None,
                    help="minimalLength in voxels (default: NeuTu's 10)")
    ap.add_argument("--only", help="comma-separated body ids")
    a = ap.parse_args()

    with open(a.manifest) as f:
        man = json.load(f)
    bodies = [r for r in man["bodies"] if "mask_path" in r]
    if a.only:
        keep = {int(v) for v in a.only.split(",")}
        bodies = [r for r in bodies if r["body_id"] in keep]
    # largest first: the pool's tail is one big body
    bodies.sort(key=lambda r: -r["mask_voxels"])
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"{len(bodies)} bodies, {a.workers} workers, NeuTu minlen={a.minlen}"
          f"{', kimimaro skipped' if a.no_kimimaro else ''}")
    results, t_start = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        port_kw = {k: v for k, v in (("scale", a.port_scale),
                                     ("const", a.port_const),
                                     ("min_length", a.port_min_length))
                   if v is not None}
        futs = {ex.submit(run_one, r, a.out_dir, man["voxel_nm"], a.neutu_bin,
                          a.minlen, not a.no_kimimaro, not a.no_port,
                          port_kw): r["body_id"]
                for r in bodies}
        for fut in as_completed(futs):
            body = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"body_id": body, "error": f"{type(e).__name__}: {e}"}
            results.append(r)
            done = len(results)
            bits = []
            for name, m in r.get("methods", {}).items():
                bits.append(f"{name}={'ERR' if 'error' in m else f'''{100*m['coverage']:.0f}%/{m['nodes']}n/{m['seconds']}s'''}")
            print(f"[{done}/{len(bodies)}] {body} " + "  ".join(bits), flush=True)

    results.sort(key=lambda r: -r.get("mask_voxels", 0))
    out = {"manifest": a.manifest, "voxel_nm": man["voxel_nm"], "scale": man["scale"],
           "neutu_minlen": a.minlen, "kimimaro_production": KIMI_PRODUCTION,
           "port_params": port_kw,
           "wall_seconds": round(time.time() - t_start, 1), "results": results}
    p = os.path.join(a.out_dir, "results.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{out['wall_seconds']}s total -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
