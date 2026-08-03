#!/usr/bin/env python
"""Show what the compensating invalidation constant costs, per body.

``neutu_trace`` defaults to ``const=8`` rather than NeuTu's 2, which matches
NeuTu's branch count but is suspected of invalidating genuinely separate neurites
that run close together. The suspicion comes from one number — B->A p90, the
distance from NeuTu's skeleton to ours at the 90th percentile, which roughly
doubles on the two largest thick bodies. This renders what that number is
actually made of.

Per body, four panels: the NeuTu reference, the port at NeuTu's own const=2, the
port at the shipped const=8, and an overlay locating the cable NeuTu traced that
const=8 does not reproduce.

    python scripts/plot_const_comparison.py --bodies 6308993 45892915 --out fig.png

Needs matplotlib, deliberately not a package dependency.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from em_seg_morpho import neutu_io, skelmetrics                              # noqa: E402

# Categorical slots 1-3 of the reference palette, which validate all-pairs in
# both modes; plus the reserved status red for the "not reproduced" highlight.
# Verified with the skill's validator rather than by eye: worst all-pairs CVD
# dE 9.2, normal-vision 24.0. Aqua sits under 3:1 on this surface, so the relief
# rule applies -- every panel carries a visible direct label.
C_NEUTU, C_C2, C_C8 = "#2a78d6", "#eb6834", "#1baf7a"
C_MISS = "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#c9c8c3"
SURFACE = "#fcfcfb"


def _load(body, mask_dir, swc_dir, cache_dir):
    """Mask + the three skeletons, tracing only what is not already cached.

    Body 6308993 takes ~60 s to trace, so results are cached per (body, const)
    and reused across runs.
    """
    from em_seg_morpho import neutu_trace, swc_simplify

    mask = np.load(os.path.join(mask_dir, f"body_{body}_mask.npy"))
    z, r, par, nid = neutu_io.read_swc(os.path.join(swc_dir, f"b{body}_neutu_L10.swc"))
    out = {"neutu": (z, r, neutu_io.swc_edges(par, nid))}
    os.makedirs(cache_dir, exist_ok=True)
    for tag, const in (("c2", neutu_trace.NEUTU_CONST),
                       ("c8", neutu_trace.INVALIDATION_CONST)):
        p = os.path.join(cache_dir, f"b{body}_const{const:g}.npz")
        if os.path.exists(p):
            d = np.load(p)
            out[tag] = (d["vertices"], d["radii"], d["edges"])
            continue
        print(f"  tracing body {body} at const={const:g} …", flush=True)
        s = neutu_trace.skeletonize(mask, const=const)
        v, rr, e = swc_simplify.simplify(np.asarray(s.vertices, float),
                                         np.asarray(s.radii, float),
                                         np.asarray(s.edges, int))
        np.savez_compressed(p, vertices=v, radii=rr, edges=e)
        out[tag] = (v, rr, e)
    return mask, out


def _best_axis(shape):
    """Project down the axis that leaves the squarest picture.

    Always projecting along z squashes an elongated body into a sliver — body
    45892915 is 391 x 764 x 385, so a z-projection is 2:1 while a y-projection is
    almost square.
    """
    return min(range(3), key=lambda k: max(_plane(shape, k)) / min(_plane(shape, k)))


def _plane(shape, axis):
    return tuple(s for k, s in enumerate(shape) if k != axis)


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None,
                    help="where traced skeletons are cached (default: "
                         "<benchmark>/const_compare)")
    ap.add_argument("--miss-vox", type=float, default=6.0,
                    help="NeuTu cable further than this from the port counts as "
                         "not reproduced")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    n = len(a.bodies)
    fig, ax = plt.subplots(n, 4, figsize=(19, 5.0 * n), facecolor=SURFACE,
                           squeeze=False)

    for row, body in enumerate(a.bodies):
        mask, m = _load(body, f"{a.benchmark}/masks", f"{a.benchmark}/neutu_swc",
                        a.cache or f"{a.benchmark}/const_compare")
        axis = _best_axis(mask.shape)
        mip = mask.max(axis=axis)
        cols = [k for k in range(3) if k != axis]
        (zn, rn, en), (z2, r2, e2), (z8, r8, e8) = m["neutu"], m["c2"], m["c8"]

        ag2 = skelmetrics.agreement(z2, r2, e2, zn, rn, en)
        ag8 = skelmetrics.agreement(z8, r8, e8, zn, rn, en)

        for col, (name, (v, e, c), ag) in enumerate([
                ("NeuTu  (reference)", (zn, en, C_NEUTU), None),
                ("port, const=2  (NeuTu's own value)", (z2, e2, C_C2), ag2),
                ("port, const=8  (shipped default)", (z8, e8, C_C8), ag8)]):
            _draw(ax[row, col], mip, v, e, c, axis)
            ax[row, col].set_title(name, fontsize=10.5, color=INK, pad=8)
            tips = int((np.bincount(e.ravel(), minlength=len(v)) == 1).sum()) if len(e) else 0
            txt = f"{len(v):,} nodes\n{tips:,} tips"
            if ag is not None:
                txt += f"\ntips {ag['tip_ratio']:.2f}x NeuTu\nB->A p90 {ag['b_to_a_p90']:.1f} vox"
            # bottom-left: a short panel would otherwise run this into the title
            ax[row, col].text(0.02, 0.03, txt, transform=ax[row, col].transAxes,
                              va="bottom", fontsize=8.5, color=INK2,
                              family="monospace")

        # panel 4: where does const=8 fail to reproduce NeuTu's cable?
        pn, _ = skelmetrics.sweep(zn, rn, en)
        p8, _ = skelmetrics.sweep(z8, r8, e8)
        d, _ = cKDTree(p8).query(pn)
        miss = d > a.miss_vox
        axm = ax[row, 3]
        axm.imshow(mip.T, cmap="Greys", alpha=0.16, origin="lower",
                   interpolation="nearest", vmin=0, vmax=1.6)
        if len(e8):
            from matplotlib.collections import LineCollection
            seg = np.stack([z8[e8[:, 0]][:, cols], z8[e8[:, 1]][:, cols]], axis=1)
            axm.add_collection(LineCollection(seg, colors=MUTED, linewidths=0.5))
        axm.scatter(pn[miss][:, cols[0]], pn[miss][:, cols[1]], s=1.4, c=C_MISS,
                    linewidths=0, zorder=3)
        axm.set_xlim(0, mip.shape[0])
        axm.set_ylim(0, mip.shape[1])
        axm.set_title(f"NeuTu cable const=8 misses (> {a.miss_vox:.0f} vox)",
                      fontsize=10.5, color=INK, pad=8)
        axm.text(0.02, 0.03,
                 f"{100*miss.mean():.1f}% of NeuTu cable\nnot reproduced",
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
                        mpatches.Patch(color=C_C2, label="port, const=2"),
                        mpatches.Patch(color=C_C8, label="port, const=8"),
                        mpatches.Patch(color=C_MISS,
                                       label="NeuTu cable the port does not reproduce")],
               loc="lower center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("What the compensating invalidation constant costs "
                 "on the two largest thick bodies",
                 fontsize=13.5, color=INK, y=0.997)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.07,
                        wspace=0.05, hspace=0.10)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=125, facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
