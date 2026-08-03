#!/usr/bin/env python
"""Compare method columns across one or more ``run_skel_benchmark.py`` result files.

    python scripts/summarize_skel_benchmark.py a/results.json b/results.json \\
        --methods kimimaro_production neutu_L10 port_simplified

Methods are looked up across all given files, so a run that skipped kimimaro can
be compared against one that included it. Ratios are against ``--baseline``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--methods", nargs="+", default=[
        "kimimaro_production", "neutu_L10", "port_simplified"])
    ap.add_argument("--baseline", default="neutu_L10")
    a = ap.parse_args()

    merged: dict[int, dict] = {}
    for path in a.results:
        for r in json.load(open(path))["results"]:
            slot = merged.setdefault(r["body_id"], {"band": r.get("band", ""),
                                                    "methods": {}})
            for name, m in r.get("methods", {}).items():
                if "error" not in m:
                    slot["methods"].setdefault(name, m)

    methods = [m for m in a.methods
               if any(m in s["methods"] for s in merged.values())]
    w = max(12, max(len(m) for m in methods) + 1)
    print(f"{'body':>9s} {'band':15s} " + "".join(f"|{m:>{w}s} " for m in methods))
    print(f"{'':9s} {'':15s} " + "".join(f"|{'fill  nodes':>{w}s} " for _ in methods))
    print("-" * (26 + (w + 2) * len(methods)))

    acc: dict[str, list] = {m: [] for m in methods}
    for body, s in sorted(merged.items(), key=lambda kv: -kv[1]["methods"].get(
            a.baseline, {}).get("nodes", 0)):
        row = f"{body:9d} {s['band']:15s} "
        for m in methods:
            d = s["methods"].get(m)
            if d is None:
                row += f"|{'-':>{w}s} "
                continue
            acc[m].append(d)
            row += f"|{100*d['coverage']:6.0f}% {d['nodes']:{w-8}d} "
        print(row)

    print("-" * (26 + (w + 2) * len(methods)))
    med = lambda m, k: np.median([x[k] for x in acc[m]]) if acc[m] else float("nan")
    print(f"{'median':>25s} " + "".join(
        f"|{100*med(m,'coverage'):6.0f}% {med(m,'nodes'):{w-8}.0f} " for m in methods))
    print(f"{'median spill':>25s} " + "".join(
        f"|{100*med(m,'spill'):6.0f}% {'':{w-8}s} " for m in methods))
    print(f"{'total seconds':>25s} " + "".join(
        f"|{sum(x['seconds'] for x in acc[m]):7.0f} {'':{w-8}s} " for m in methods))

    if a.baseline in acc and acc[a.baseline]:
        print(f"\nvs {a.baseline}:")
        base = {id(None): None}
        for m in methods:
            if m == a.baseline:
                continue
            pairs = [(s["methods"][m], s["methods"][a.baseline])
                     for s in merged.values()
                     if m in s["methods"] and a.baseline in s["methods"]]
            if not pairs:
                continue
            nr = np.array([p[0]["nodes"] / max(1, p[1]["nodes"]) for p in pairs])
            fd = np.array([p[0]["coverage"] - p[1]["coverage"] for p in pairs])
            print(f"  {m:24s} nodes {np.median(nr):5.2f}x   "
                  f"fill {100*np.median(fd):+5.1f} pts   "
                  f"ahead on {int((fd > 0).sum())}/{len(fd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
