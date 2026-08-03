#!/usr/bin/env python
"""Compare the port against NeuTu and against what ships today, per body.

Four panels per body: the NeuTu reference, ``neutu_trace`` + ``swc_simplify`` at
NeuTu's own settings, kimimaro production (the incumbent), and a disagreement
panel showing **both** directions — NeuTu cable the port lacks, and port cable
NeuTu lacks.

Both directions matter and they mean different things. Absent cable is a branch we
failed to trace. Added cable is a branch we invented — which on these data is
often boundary convolution inside a noisy bulb being read as an arbor, so it is
not automatically a win either.

Note what is deliberately *not* plotted: coverage. It is confounded by branch
count, so a figure ranking methods by how much of the mask they fill rewards
whichever one invents the most neurites. See docs/skeletonization-comparison.md.

    python scripts/plot_skeleton_comparison.py --bodies 6308993 45892915 --out fig.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from em_seg_morpho import neutu_io, skelmetrics                              # noqa: E402

# Categorical slots 1-3 of the reference palette, which validate all-pairs in both
# modes (worst CVD dE 9.2, normal-vision 24.0 — checked with the dataviz validator,
# not by eye). Aqua sits under 3:1 on this surface, so the relief rule applies:
# every panel carries a visible direct label.
C_NEUTU, C_KIMI, C_PORT = "#2a78d6", "#eb6834", "#1baf7a"

# The disagreement panel's two highlights, over a grey port skeleton. Violet, not
# the port's own green: red-vs-green is the classic CVD failure and the validator
# WARNs on it (deutan dE 6.9). Violet against red passes clean (dE 22.7). It is
# also not slot-1 blue, which would put the same swatch in the legend as "NeuTu"
# with a different meaning.
C_MISS, C_EXTRA = "#e34948", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#d3d2cd"
SURFACE = "#fcfcfb"


def _plane(shape, axis):
    return tuple(s for k, s in enumerate(shape) if k != axis)


def _best_axis(shape):
    """Project down the axis leaving the squarest picture.

    Always projecting along z squashes an elongated body: 45892915 is
    391 x 764 x 385, so a z-projection is 2:1 while a y-projection is ~1:1.
    """
    return min(range(3), key=lambda k: max(_plane(shape, k)) / min(_plane(shape, k)))


def _npz(path):
    d = np.load(path)
    return (np.asarray(d["vertices"], float), np.asarray(d["radii"], float),
            np.asarray(d["edges"], int))


def _tips(v, e):
    e = np.asarray(e).reshape(-1, 2)
    if not len(e):
        return 0
    return int((np.bincount(e.ravel(), minlength=len(v)) == 1).sum())


def _draw(ax, mip, v, e, colour, axis, lw=0.55):
    from matplotlib.collections import LineCollection

    cols = [k for k in range(3) if k != axis]
    ax.imshow(mip.T, cmap="Greys", alpha=0.16, origin="lower",
              interpolation="nearest", vmin=0, vmax=1.6)
    if len(e):
        seg = np.stack([v[e[:, 0]][:, cols], v[e[:, 1]][:, cols]], axis=1)
        ax.add_collection(LineCollection(seg, colors=colour, linewidths=lw))
    ax.set_xlim(0, mip.shape[0])
    ax.set_ylim(0, mip.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", nargs="+", type=int, required=True)
    ap.add_argument("--benchmark", default="/path/to/scratch/"
                                           "morpho-skel-benchmark/2026-07-31-wide")
    ap.add_argument("--port-dir", default=None,
                    help="dir holding b<body>_simplify.npz "
                         "(default: <benchmark>/fixed/port)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-vox", type=float, default=6.0,
                    help="centreline distance above which cable counts as absent")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.spatial import cKDTree

    port_dir = a.port_dir or f"{a.benchmark}/fixed/port"
    n = len(a.bodies)
    fig, ax = plt.subplots(n, 4, figsize=(19, 5.0 * n), facecolor=SURFACE,
                           squeeze=False)

    for row, body in enumerate(a.bodies):
        mask = np.load(f"{a.benchmark}/masks/body_{body}_mask.npy")
        axis = _best_axis(mask.shape)
        cols = [k for k in range(3) if k != axis]
        mip = mask.max(axis=axis)

        zn, rn, par, nid = neutu_io.read_swc(
            f"{a.benchmark}/neutu_swc/b{body}_neutu_L10.swc")
        en = neutu_io.swc_edges(par, nid)
        vp, rp, ep = _npz(f"{port_dir}/b{body}_simplify.npz")
        vk, rk, ek = _npz(f"{a.benchmark}/kimimaro/b{body}_kimi_production.npz")
        ag = skelmetrics.agreement(vp, rp, ep, zn, rn, en)

        panels = [("NeuTu  (reference)", zn, rn, en, C_NEUTU, None),
                  ("port  (neutu_trace + swc_simplify)", vp, rp, ep, C_PORT, ag),
                  ("kimimaro production  (ships today)", vk, rk, ek, C_KIMI, None)]
        for col, (name, v, r, e, c, agr) in enumerate(panels):
            _draw(ax[row, col], mip, v, e, c, axis)
            ax[row, col].set_title(name, fontsize=10.5, color=INK, pad=8)
            txt = f"{len(v):,} nodes\n{_tips(v, e):,} tips"
            if agr is not None:
                txt += (f"\ntips {agr['tip_ratio']:.2f}x NeuTu"
                        f"\ncable {agr['cable_ratio']:.2f}x NeuTu")
            ax[row, col].text(0.02, 0.03, txt, transform=ax[row, col].transAxes,
                              va="bottom", fontsize=8.5, color=INK2,
                              family="monospace")

        # panel 4: disagreement, both directions
        pn, _ = skelmetrics.sweep(zn, rn, en)
        dv, dr, de = vp, rp, ep
        pp, _ = skelmetrics.sweep(dv, dr, de)
        miss = cKDTree(pp).query(pn)[0] > a.tol_vox          # NeuTu has it, we don't
        extra = cKDTree(pn).query(pp)[0] > a.tol_vox         # we have it, NeuTu doesn't
        axm = ax[row, 3]
        axm.imshow(mip.T, cmap="Greys", alpha=0.16, origin="lower",
                   interpolation="nearest", vmin=0, vmax=1.6)
        if len(de):
            seg = np.stack([dv[de[:, 0]][:, cols], dv[de[:, 1]][:, cols]], axis=1)
            axm.add_collection(LineCollection(seg, colors=MUTED, linewidths=0.5))
        axm.scatter(pp[extra][:, cols[0]], pp[extra][:, cols[1]], s=1.2,
                    c=C_EXTRA, linewidths=0, zorder=3)
        axm.scatter(pn[miss][:, cols[0]], pn[miss][:, cols[1]], s=1.2,
                    c=C_MISS, linewidths=0, zorder=4)
        axm.set_xlim(0, mip.shape[0])
        axm.set_ylim(0, mip.shape[1])
        axm.set_title(f"port vs NeuTu (> {a.tol_vox:.0f} vox)",
                      fontsize=10.5, color=INK, pad=8)
        axm.text(0.02, 0.03,
                 f"{100*miss.mean():.1f}% of NeuTu cable absent\n"
                 f"{100*extra.mean():.1f}% of its cable added",
                 transform=axm.transAxes, va="bottom", fontsize=8.5, color=INK2,
                 family="monospace")

        ax[row, 0].set_ylabel(f"body {body}\n{int(mask.sum()):,} voxels   "
                              f"(projected along {'zyx'[axis]})",
                              fontsize=10, color=INK)

    for r_ in range(n):
        for c in range(4):
            ax[r_, c].set_xticks([])
            ax[r_, c].set_yticks([])
            ax[r_, c].set_aspect("equal")
            for sp in ax[r_, c].spines.values():
                sp.set_color("#dedddb")

    fig.legend(handles=[mpatches.Patch(color=C_NEUTU, label="NeuTu"),
                        mpatches.Patch(color=C_PORT, label="port"),
                        mpatches.Patch(color=C_KIMI,
                                       label="kimimaro production"),
                        mpatches.Patch(color=C_MISS, label="NeuTu cable the port lacks"),
                        mpatches.Patch(color=C_EXTRA, label="port cable NeuTu lacks")],
               loc="lower center", ncol=5, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Skeletons at NeuTu's own settings — the two largest thick bodies",
                 fontsize=13.5, color=INK, y=0.997)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.07,
                        wspace=0.05, hspace=0.10)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=125, facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
