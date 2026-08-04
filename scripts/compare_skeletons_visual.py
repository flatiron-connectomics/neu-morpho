#!/usr/bin/env python
"""Side-by-side visual + numeric comparison of skeletons on the same mask.

Produces, per method: the skeleton coloured by vertex radius, the rendered tube
against the segment as a max projection, and a single z-slice through the thickest
point — which is where radius error is easiest to see. Also a method-independent
local-thickness map, so you can judge what the segmentation can support before
blaming a skeletonizer for what the mask doesn't contain.

    python scripts/compare_skeletons_visual.py body_123_mask.npy out.png \\
        --swc neutu=b123_neutu.swc --npz kimimaro=b123_kimi.npz --voxel-nm 32

``--swc NAME=PATH``  an SWC file (NeuTu, or anything else that writes SWC)
``--npz NAME=PATH``  npz with ``vertices`` (zyx), ``radii``, ``edges``

Needs matplotlib, which is deliberately not a package dependency — run this from
an environment that has it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from em_seg_morpho import neutu_io, skelmetrics          # noqa: E402

# sequential blue ramp; the lightest steps are dropped because they vanish
# against the grey segment silhouette
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
C_SEG, C_TUBE = "#d6d5d1", "#2a78d6"
INK, INK2 = "#0b0b0b", "#52514e"


def _load(spec, kind):
    # rsplit, not split: method names legitimately contain '=' ("minlen=10")
    name, path = spec.rsplit("=", 1)
    if kind == "swc":
        zyx, r, par, nid = neutu_io.read_swc(path)
        return name, (zyx, r, neutu_io.swc_edges(par, nid))
    d = np.load(path)
    return name, (np.asarray(d["vertices"], float),
                  np.asarray(d["radii"], float),
                  np.asarray(d["edges"], int))


def _paint(ax, base, over, extent=None):
    """Segment in grey; tube painted over it in blue. Grey left showing = missed."""
    h, w = base.shape
    img = np.ones((w, h, 3))
    for layer, col in ((base, C_SEG), (over, C_TUBE)):
        img[layer.T] = [int(col[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    ax.imshow(img, origin="lower", interpolation="nearest")
    ax.set_xlim(*(extent[:2] if extent else (0, h)))
    ax.set_ylim(*(extent[2:] if extent else (0, w)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mask")
    ap.add_argument("out")
    ap.add_argument("--swc", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--npz", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--voxel-nm", type=float, default=32.0)
    ap.add_argument("--zoom", type=int, default=80)
    a = ap.parse_args()

    import edt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.patches as mpatches

    cmap = LinearSegmentedColormap.from_list("radius", RAMP)
    m = np.load(a.mask).astype(bool)
    methods = dict([_load(s, "swc") for s in a.swc] + [_load(s, "npz") for s in a.npz])
    if not methods:
        ap.error("give at least one --swc or --npz")

    e = edt.edt(m.astype(np.uint8), anisotropy=(1, 1, 1), black_border=False)
    pk = np.unravel_index(int(np.argmax(e)), e.shape)
    thick = e.max(axis=0) * a.voxel_nm
    del e
    mip, zsl = m.max(axis=0), int(pk[0])
    ext = (pk[1] - a.zoom, pk[1] + a.zoom, pk[2] - a.zoom, pk[2] + a.zoom)
    rmax = max(r.max() for _, r, _ in methods.values())

    n = len(methods)
    fig, ax = plt.subplots(n + 1, 3, figsize=(16.5, 4.9 * (n + 1)), facecolor="white",
                           squeeze=False)

    im = ax[0, 0].imshow(np.ma.masked_where(thick.T == 0, thick.T), cmap=cmap,
                         origin="lower", interpolation="nearest")
    ax[0, 0].set_ylabel("the segment itself", fontsize=11, color=INK)
    ax[0, 0].set_title("local thickness — largest sphere that fits", fontsize=10.5,
                       color=INK2, pad=9)
    ax[0, 1].imshow(np.ma.masked_where(thick.T == 0, thick.T), cmap=cmap,
                    origin="lower", interpolation="nearest")
    ax[0, 1].set_xlim(ext[0], ext[1]); ax[0, 1].set_ylim(ext[2], ext[3])
    ax[0, 1].set_title(f"zoom: thickest point, {thick.max():.0f} nm radius",
                       fontsize=10.5, color=INK2, pad=9)
    ax[0, 2].axis("off")
    cb0 = fig.colorbar(im, ax=ax[0, 2], fraction=0.35, pad=0.02)
    cb0.set_label("segment thickness (nm)", fontsize=9, color=INK2)
    cb0.ax.tick_params(labelsize=8, colors=INK2)

    for row, (name, (zyx, radii, edges)) in enumerate(methods.items(), start=1):
        s = skelmetrics.score(m, zyx, radii, edges)
        tube = skelmetrics.rasterize(m.shape, zyx, radii, edges)

        ax[row, 0].imshow(mip.T, cmap="Greys", alpha=0.28, origin="lower",
                          interpolation="nearest", vmin=0, vmax=1.6)
        if len(edges):
            seg = np.stack([zyx[edges[:, 0]][:, [1, 2]], zyx[edges[:, 1]][:, [1, 2]]], 1)
            lc = LineCollection(seg, cmap=cmap, norm=Normalize(0, rmax), linewidths=1.1)
            lc.set_array(radii[edges].mean(1))
            ax[row, 0].add_collection(lc)
        ax[row, 0].set_xlim(0, mip.shape[0]); ax[row, 0].set_ylim(0, mip.shape[1])
        ax[row, 0].set_ylabel(name, fontsize=11, color=INK)

        _paint(ax[row, 1], mip, tube.max(axis=0))
        _paint(ax[row, 2], m[zsl], tube[zsl], extent=ext)
        del tube

        ax[row, 0].text(0.02, 0.975, f"{s['nodes']:,} nodes\nmax r = "
                        f"{radii.max()*a.voxel_nm:.0f} nm",
                        transform=ax[row, 0].transAxes, va="top", fontsize=9,
                        color=INK2, family="monospace")
        ax[row, 1].text(0.02, 0.975, f"fills {100*s['coverage']:.0f}% of the segment\n"
                        f"{100*s['spill']:.0f}% of the tube spills outside",
                        transform=ax[row, 1].transAxes, va="top", fontsize=9,
                        color=INK2, family="monospace")
        if row == 1:
            ax[1, 1].set_title("rendered tube vs segment — max projection",
                               fontsize=10.5, color=INK2, pad=9)
            ax[1, 2].set_title(f"single z-slice through thickest point (z={zsl})",
                               fontsize=10.5, color=INK2, pad=9)
        print(f"{name:28s} nodes={s['nodes']:6d}  fills={100*s['coverage']:5.1f}%  "
              f"spill={100*s['spill']:5.1f}%  "
              f"nodes/1k covered={s['nodes_per_1k_covered']:.2f}")

    for r_ in range(n + 1):
        for c in range(3):
            if not (r_ == 0 and c == 2):
                ax[r_, c].set_xticks([]); ax[r_, c].set_yticks([])
                ax[r_, c].set_aspect("equal")
                for sp in ax[r_, c].spines.values():
                    sp.set_color("#dedddb")

    fig.subplots_adjust(left=0.035, right=0.915, top=0.955, bottom=0.045,
                        wspace=0.04, hspace=0.06)
    cax = fig.add_axes([0.932, 0.30, 0.011, 0.30])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                            norm=Normalize(0, rmax * a.voxel_nm)), cax=cax)
    cb.set_label("vertex radius (nm)", fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.legend(handles=[mpatches.Patch(color=C_SEG, label="segment"),
                        mpatches.Patch(color=C_TUBE,
                                       label="skeleton rendered as a tube — "
                                             "grey showing through = not filled")],
               loc="lower center", ncol=2, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.001))
    fig.suptitle(f"{a.mask.rsplit('/', 1)[-1]} — {int(m.sum()):,} voxels "
                 f"at {a.voxel_nm:.0f} nm/vox", fontsize=13.5, color=INK, y=0.998)
    fig.savefig(a.out, dpi=115, facecolor="white")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
