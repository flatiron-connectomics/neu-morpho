#!/usr/bin/env python
"""Zoomed slab overlays: NeuTu and the port on the same axes, over the mask.

**Full-body projections cannot answer geometric questions.** They compress the
whole depth of a bulb into one image, so overlapping processes turn to mush, and a
route that strays 16 voxels — 535 nm, a real error — is a few pixels wide. Every
figure in this investigation up to now has had that limit.

This renders a thin **slab** (a few voxels either side of the cut plane) around a
chosen point, so structure inside a bulb stays legible, and it draws both skeletons
in one panel so the difference in route is visible directly rather than inferred
from two side-by-side pictures.

    python scripts/plot_zoom_overlay.py --bodies 6308993 45892915 --out zoom.png

Zoom targets default to the bulkiest neighbourhood (by local mask occupancy) and
the thinnest, so each body gets one bulb view and one ordinary-neurite view.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from neu_morpho import neutu_io                                       # noqa: E402

# Validated categorical slots 1 and 3 (blue / aqua): all-pairs CVD dE 9.2,
# normal-vision 24.0. Aqua is under 3:1 on this surface, so both series carry
# direct labels, which is the relief rule.
C_NEUTU, C_PORT = "#2a78d6", "#1baf7a"
C_MASK = "#dcdbd6"
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"


def _npz(path):
    d = np.load(path)
    return (np.asarray(d["vertices"], float), np.asarray(d["radii"], float),
            np.asarray(d["edges"], int).reshape(-1, 2))


def _draw_slab(ax, mask, skels, centre, axis, half, slab):
    """Mask slab in grey with both skeletons over it, clipped to the slab."""
    from matplotlib.collections import LineCollection

    cols = [k for k in range(3) if k != axis]
    lo = max(0, centre[axis] - slab)
    hi = min(mask.shape[axis], centre[axis] + slab + 1)
    sl = [slice(None)] * 3
    sl[axis] = slice(lo, hi)
    img = mask[tuple(sl)].max(axis=axis)

    rgb = np.ones(img.T.shape + (3,))
    rgb[img.T] = [int(C_MASK[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    ax.imshow(rgb, origin="lower", interpolation="nearest")

    for (v, r, e), colour, label in skels:
        if not len(e):
            continue
        # keep only edges with both ends inside the slab, so the overlay shows
        # structure at this depth rather than everything projected through it
        inside = (v[:, axis] >= lo) & (v[:, axis] < hi)
        keep = inside[e[:, 0]] & inside[e[:, 1]]
        if not keep.any():
            continue
        seg = np.stack([v[e[keep, 0]][:, cols], v[e[keep, 1]][:, cols]], axis=1)
        ax.add_collection(LineCollection(seg, colors=colour, linewidths=1.4,
                                         label=label))
    ax.set_xlim(centre[cols[0]] - half, centre[cols[0]] + half)
    ax.set_ylim(centre[cols[1]] - half, centre[cols[1]] + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", nargs="+", type=int, required=True)
    ap.add_argument("--benchmark", default="/path/to/scratch/"
                                           "morpho-skel-benchmark/2026-07-31-wide")
    ap.add_argument("--port-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=int, default=45, help="zoom half-width, voxels")
    ap.add_argument("--slab", type=int, default=6, help="slab half-thickness, voxels")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter

    port_dir = a.port_dir or f"{a.benchmark}/preserve/port"
    n = len(a.bodies)
    fig, ax = plt.subplots(n, 2, figsize=(11.5, 5.6 * n), facecolor=SURFACE,
                           squeeze=False)

    for row, body in enumerate(a.bodies):
        mask = np.load(f"{a.benchmark}/masks/body_{body}_mask.npy")
        zn, rn, par, nid = neutu_io.read_swc(
            f"{a.benchmark}/neutu_swc/b{body}_neutu_L10.swc")
        en = neutu_io.swc_edges(par, nid)
        vp, rp, ep = _npz(f"{port_dir}/b{body}_simplify.npz")
        skels = [((zn, rn, en), C_NEUTU, "NeuTu"), ((vp, rp, ep), C_PORT, "port")]

        occ = uniform_filter(mask.astype(np.float32), size=25, mode="constant")
        bulb = np.unravel_index(int(np.argmax(occ)), occ.shape)
        # thinnest place that still holds skeleton, for an ordinary-neurite view
        thin_occ = np.where(mask, occ, np.inf)
        idx = [np.clip(np.round(vp[:, k]).astype(int), 0, mask.shape[k] - 1)
               for k in range(3)]
        at_nodes = thin_occ[idx[0], idx[1], idx[2]]
        thin = tuple(int(v) for v in vp[int(np.argmin(at_nodes))])

        for col, (centre, what) in enumerate([(bulb, "bulb"), (thin, "thin neurite")]):
            axis = 0
            _draw_slab(ax[row, col], mask, skels, tuple(int(c) for c in centre),
                       axis, a.half, a.slab)
            ax[row, col].set_title(
                f"{what} — {2*a.slab+1} voxel slab at z={int(centre[0])}",
                fontsize=10.5, color=INK, pad=8)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            ax[row, col].set_aspect("equal")
            for sp in ax[row, col].spines.values():
                sp.set_color("#dedddb")
        ax[row, 0].set_ylabel(f"body {body}", fontsize=11, color=INK)

    fig.legend(handles=[mpatches.Patch(color=C_NEUTU, label="NeuTu"),
                        mpatches.Patch(color=C_PORT, label="port"),
                        mpatches.Patch(color=C_MASK, label="segment (this slab)")],
               loc="lower center", ncol=3, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.006))
    fig.suptitle("Route comparison in a thin slab — full projections cannot show this",
                 fontsize=13, color=INK, y=0.998)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.075,
                        wspace=0.06, hspace=0.12)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
