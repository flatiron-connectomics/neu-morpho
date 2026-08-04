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

Needs plotly, which is deliberately not a package dependency (it sits in the
``compare`` extra) — run this from an environment that has it.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from em_volume_tools import location                              # noqa: E402

# Palette slots 1 and 2 of the validated categorical ramp: a colourblind-safe pair.
MESH_COLOR = "#2a78d6"
SKEL_COLOR = "#eb6834"


# --------------------------------------------------------------------------- #
# reading the published volume
# --------------------------------------------------------------------------- #
def load_skeleton(volume: str, body_id: int, skel_dir: str = "skeleton"):
    """``(vertices_xyz_nm, edges, radii)`` for one body, or None if absent.

    ``Skeleton.from_precomputed`` returns vertices in the order the format stores
    them, which is **xyz** — the same order the mesh vertices come back in, so the
    two overlay without a flip. (In memory elsewhere in this package they are zyx;
    see the zyx/xyz invariant in CLAUDE.md.)
    """
    from osteoid import Skeleton

    blob = location.read_bytes(volume, skel_dir, str(int(body_id)))
    if blob is None:
        return None
    skel = Skeleton.from_precomputed(blob, segid=int(body_id))
    radii = np.asarray(getattr(skel, "radii", None), dtype=float) \
        if getattr(skel, "radii", None) is not None else None
    if radii is None or len(radii) != len(skel.vertices):
        raise ValueError(
            f"body {body_id}: skeleton has no usable 'radius' attribute "
            f"({0 if radii is None else len(radii)} radii for "
            f"{len(skel.vertices)} vertices)")
    return (np.asarray(skel.vertices, dtype=float),
            np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2), radii)


def load_mesh(volume: str, body_id: int, lod: int | None, mesh_dir: str = "mesh"):
    """``(vertices_xyz_nm, faces, lod)`` for one LOD, or None if the body has no mesh.

    ``vol2mesh.multires.read_object_mesh`` parses the ``.index`` manifest, and it
    opens files — so for an object store the three objects are staged into a temp
    directory and handed to it. Reusing its parser beats reimplementing the multires
    manifest, which is exactly the kind of format handling that fails silently.
    """
    from vol2mesh import multires

    data = location.read_bytes(volume, mesh_dir, str(int(body_id)))
    index = location.read_bytes(volume, mesh_dir, f"{int(body_id)}.index")
    if data is None or index is None:
        return None
    info = location.read_bytes(volume, mesh_dir, "info")

    with tempfile.TemporaryDirectory(prefix="view3d-") as tmp:
        with open(os.path.join(tmp, str(int(body_id))), "wb") as f:
            f.write(data)
        with open(os.path.join(tmp, f"{int(body_id)}.index"), "wb") as f:
            f.write(index)
        if info is not None:
            with open(os.path.join(tmp, "info"), "wb") as f:
                f.write(info)
        parsed = multires.read_object_mesh(tmp, int(body_id))

    frags = parsed.get("fragments") or []
    if not frags:
        return None
    available = sorted({int(f["lod"]) for f in frags})
    want = available[-1] if lod is None else int(lod)   # default: coarsest = cheapest
    if want not in available:
        raise ValueError(f"body {body_id}: LOD {want} not present; have {available}")

    # Fragments are octree cells; concatenating means shifting each one's face
    # indices by the vertices already emitted.
    verts, faces, offset = [], [], 0
    for f in frags:
        if int(f["lod"]) != want:
            continue
        v = np.asarray(f["vertices_xyz"], dtype=float)
        t = np.asarray(f["faces"], dtype=np.int64).reshape(-1, 3)
        verts.append(v)
        faces.append(t + offset)
        offset += len(v)
    if not verts:
        return None
    return np.concatenate(verts), np.concatenate(faces), want


# --------------------------------------------------------------------------- #
# skeleton -> conical frusta
# --------------------------------------------------------------------------- #
def frustum_mesh(vertices_xyz, edges, radii, sides: int = 8):
    """One truncated cone per edge; returns ``(vertices, faces)`` for a Mesh3d.

    Each edge contributes two rings of ``sides`` points — radius ``r[u]`` at one end
    and ``r[v]`` at the other — joined by a triangle strip. Rings are built in a
    plane perpendicular to the edge, from an arbitrary but stable orthonormal basis;
    the tube is therefore correct in cross-section at every node, and tapers linearly
    between nodes exactly as the format's per-vertex radius implies.

    Adjacent edges are *not* stitched to each other. At a junction the cones simply
    overlap, which renders correctly for a solid surface and avoids inventing a
    joint geometry the data does not specify.
    """
    v = np.asarray(vertices_xyz, dtype=float)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    r = np.asarray(radii, dtype=float)
    if not len(e):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    p0, p1 = v[e[:, 0]], v[e[:, 1]]
    r0, r1 = r[e[:, 0]], r[e[:, 1]]
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    keep = length > 0
    p0, p1, r0, r1, d, length = (a[keep] for a in (p0, p1, r0, r1, d, length))
    if not len(d):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    axis = d / length[:, None]
    # A reference direction that is never parallel to the axis, chosen per edge.
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(axis), 1))
    ref[np.abs(axis[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    a_hat = np.cross(axis, ref)
    a_hat /= np.linalg.norm(a_hat, axis=1)[:, None]
    b_hat = np.cross(axis, a_hat)

    theta = 2.0 * np.pi * np.arange(sides) / sides
    cos_t, sin_t = np.cos(theta)[None, :, None], np.sin(theta)[None, :, None]
    disc = cos_t * a_hat[:, None, :] + sin_t * b_hat[:, None, :]      # (E, sides, 3)
    ring0 = p0[:, None, :] + r0[:, None, None] * disc
    ring1 = p1[:, None, :] + r1[:, None, None] * disc
    verts = np.concatenate([ring0, ring1], axis=1).reshape(-1, 3)     # (E*2*sides, 3)

    k = np.arange(sides)
    kn = (k + 1) % sides
    quad_a = np.stack([k, kn, sides + k], axis=1)
    quad_b = np.stack([kn, sides + kn, sides + k], axis=1)
    tri = np.concatenate([quad_a, quad_b], axis=0)                    # (2*sides, 3)
    base = (np.arange(len(p0)) * 2 * sides)[:, None, None]
    faces = (tri[None, :, :] + base).reshape(-1, 3)
    return verts, faces


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
        skel = load_skeleton(a.volume, body, a.skeleton_dir)
        mesh = None if a.no_mesh else load_mesh(a.volume, body, a.lod, a.mesh_dir)
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
