#!/usr/bin/env python
"""Interactive 3D view of a body's published mesh and skeleton, one HTML per body.

Reads back what the pipeline actually *shipped* — the precomputed volume, local or
``s3://`` — rather than any intermediate, so what you see is what neuroglancer sees.

    python scripts/view_body_3d.py --volume s3://bucket/path 12345 67890
    python scripts/view_body_3d.py --volume /path/to/segmentation --lod 0 12345

The skeleton is drawn as **conical frusta**: one truncated cone per edge, its two end
radii taken from the two node radii. That is the honest rendering of what the format
stores — a radius per vertex, linearly interpolated along the edge — so a radius that
does not fit inside the mesh is visible as the tube poking through the surface.

Mesh and skeleton are separate traces: click either legend entry to toggle it.

Reading and geometry live in ``em_seg_morpho.readback``; this script is only the
plotly presentation and the CLI. Needs plotly — ``pip install -e '.[viz]'``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from em_seg_morpho.readback import (                              # noqa: E402
    frustum_mesh, read_body_mesh, read_body_skeleton)

# Palette slots 1 and 2 of the validated categorical ramp: a colourblind-safe pair.
MESH_COLOR = "#2a78d6"
SKEL_COLOR = "#eb6834"


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def build_figure(body_id, mesh, skel, *, sides, mesh_opacity):
    import plotly.graph_objects as go

    traces, notes = [], []
    if mesh is not None:
        mv, mf, lod = mesh
        traces.append(go.Mesh3d(
            x=mv[:, 0], y=mv[:, 1], z=mv[:, 2],
            i=mf[:, 0], j=mf[:, 1], k=mf[:, 2],
            color=MESH_COLOR, opacity=mesh_opacity, flatshading=True,
            name=f"mesh (LOD {lod})", showlegend=True, hoverinfo="name"))
        notes.append(f"mesh LOD {lod}: {len(mv):,} verts / {len(mf):,} tris")

    if skel is not None:
        sv, se, sr = skel
        fv, ff = frustum_mesh(sv, se, sr, sides=sides)
        traces.append(go.Mesh3d(
            x=fv[:, 0], y=fv[:, 1], z=fv[:, 2],
            i=ff[:, 0], j=ff[:, 1], k=ff[:, 2],
            color=SKEL_COLOR, opacity=1.0, flatshading=True,
            name="skeleton (radius)", showlegend=True, hoverinfo="name"))
        notes.append(f"skeleton: {len(sv):,} nodes / {len(se):,} edges, "
                     f"radius {sr.min():.0f}–{sr.max():.0f} nm "
                     f"(median {np.median(sr):.0f})")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"body {body_id}<br><sub>{'  ·  '.join(notes)}</sub>",
        scene=dict(aspectmode="data",                 # nm are nm on every axis
                   xaxis_title="x (nm)", yaxis_title="y (nm)", zaxis_title="z (nm)"),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(itemsizing="constant"))
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("body_ids", nargs="+", type=int)
    ap.add_argument("--volume", required=True,
                    help="precomputed volume root (local path or s3://...)")
    ap.add_argument("--out-dir", default=".", help="where the HTML files go")
    ap.add_argument("--lod", type=int, default=None,
                    help="mesh LOD to draw (default: coarsest available = fastest)")
    ap.add_argument("--sides", type=int, default=8,
                    help="polygon sides per frustum; 4 is much lighter, 16 smoother")
    ap.add_argument("--mesh-opacity", type=float, default=0.35)
    ap.add_argument("--mesh-dir", default="mesh")
    ap.add_argument("--skeleton-dir", default="skeleton")
    ap.add_argument("--no-mesh", action="store_true", help="skeleton only")
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    rc = 0
    for body in a.body_ids:
        skel = read_body_skeleton(a.volume, body, a.skeleton_dir)
        mesh = (None if a.no_mesh
                else read_body_mesh(a.volume, body, a.lod, a.mesh_dir))
        if skel is None and mesh is None:
            print(f"{body:>12}  NOT FOUND — neither {a.skeleton_dir}/{body} nor "
                  f"{a.mesh_dir}/{body} exists in {a.volume}")
            rc = 1
            continue
        if skel is None:
            print(f"{body:>12}  warning: no skeleton, drawing mesh only")
        if mesh is None and not a.no_mesh:
            print(f"{body:>12}  warning: no mesh, drawing skeleton only")

        fig = build_figure(body, mesh, skel, sides=a.sides,
                           mesh_opacity=a.mesh_opacity)
        out = os.path.join(a.out_dir, f"body_{body}_3d.html")
        # Self-contained: these get copied off the cluster, and a CDN reference
        # would leave them blank wherever there is no network.
        fig.write_html(out, include_plotlyjs=True, full_html=True)
        print(f"{body:>12}  -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
