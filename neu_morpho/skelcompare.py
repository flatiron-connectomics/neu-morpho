"""Compare skeletonization methods (kimimaro vs skeletor) on identical bodies.

Each synthetic body is produced as BOTH a voxel mask (for kimimaro) and a mesh
(for skeletor, via vol2mesh), so methods are compared apples-to-apples. All
skeletons are returned in one frame — **xyz nm** — with common metrics and SWC
export for side-by-side visual inspection (napari / navis / neuroglancer).

Methods are wrapped defensively: a method that fails records the error instead of
sinking the whole comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Synthetic bodies -> labeled mask (body id 1), isotropic by default
# --------------------------------------------------------------------------- #
def _grid(shape):
    zz, yy, xx = np.indices(shape)
    return zz, yy, xx


def _cyl_z(shape, z0, z1, cy, cx, r):
    zz, yy, xx = _grid(shape)
    return (zz >= z0) & (zz < z1) & ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r)


def _sphere(shape, cz, cy, cx, r):
    zz, yy, xx = _grid(shape)
    return (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def rod(length=100, r=3, pad=8):
    s = (length + 2 * pad, 2 * (r + pad), 2 * (r + pad))
    c = s[1] // 2
    m = _cyl_z(s, pad, pad + length, c, c, r)
    return m.astype(np.uint64), (8.0, 8.0, 8.0)


def rod_with_bulb(length=100, r=3, bulb_r=12, pad=8):
    s = (length + 2 * pad, 2 * (bulb_r + pad), 2 * (bulb_r + pad))
    c = s[1] // 2
    m = _cyl_z(s, pad, pad + length, c, c, r) | _sphere(s, pad + length, c, c, bulb_r)
    return m.astype(np.uint64), (8.0, 8.0, 8.0)


def rod_with_cavity(length=100, r=3, bulb_r=14, cav_r=8, pad=8):
    """Bulbous swelling with an internal void — a 'residual nucleus' topology stressor."""
    s = (length + 2 * pad, 2 * (bulb_r + pad), 2 * (bulb_r + pad))
    c = s[1] // 2
    solid = _cyl_z(s, pad, pad + length, c, c, r) | _sphere(s, pad + length, c, c, bulb_r)
    m = solid & ~_sphere(s, pad + length, c, c, cav_r)
    return m.astype(np.uint64), (8.0, 8.0, 8.0)


def y_branch(length=80, r=3, pad=8, spread=30):
    s = (length + spread + 2 * pad, 2 * (spread + pad), 2 * (spread + pad))
    c = s[1] // 2
    zz, yy, xx = _grid(s)
    stem = _cyl_z(s, pad, pad + length, c, c, r)
    # two diagonal arms from the stem tip
    t = (zz - (pad + length)).clip(0)
    arm1 = (yy - (c + t) >= -r) & (yy - (c + t) <= r) & (np.abs(xx - c) <= r) & (zz >= pad + length)
    arm2 = (yy - (c - t) >= -r) & (yy - (c - t) <= r) & (np.abs(xx - c) <= r) & (zz >= pad + length)
    return (stem | arm1 | arm2).astype(np.uint64), (8.0, 8.0, 8.0)


SHAPES = {"rod": rod, "rod_with_bulb": rod_with_bulb,
          "rod_with_cavity": rod_with_cavity, "y_branch": y_branch}


# --------------------------------------------------------------------------- #
# mask -> trimesh (xyz nm), matching the meshing pipeline's coordinate space
# --------------------------------------------------------------------------- #
def mask_to_trimesh(mask_zyx, voxel_size_zyx):
    from vol2mesh import Mesh
    import trimesh

    box = np.array([[0, 0, 0], [s * v for s, v in zip(mask_zyx.shape, voxel_size_zyx)]])
    m = Mesh.from_label_volume(mask_zyx, box, labels=[1], ensure_halo=True, progress=False)[1]
    return trimesh.Trimesh(vertices=m.vertices_zyx[:, ::-1], faces=m.faces)  # zyx -> xyz


# --------------------------------------------------------------------------- #
# Real-body loaders (compare on actual data, where imperfect-seg noise lives)
# --------------------------------------------------------------------------- #
def body_from_volume(seg_spec, body_id, bbox_zyx, voxel_size_zyx):
    """Load a real body's mask + mesh from a segmentation crop (both methods).

    ``seg_spec`` is an neu-vol backend spec opened at the meshing scale;
    ``bbox_zyx`` = (z0, y0, x0, z1, y1, x1) in that scale's voxels.
    """
    from neu_vol.backends.base import open_backend

    z0, y0, x0, z1, y1, x1 = bbox_zyx
    region = open_backend(seg_spec).read_region((slice(z0, z1), slice(y0, y1), slice(x0, x1)))
    mask = (region == body_id).astype(np.uint64)
    return mask, mask_to_trimesh(mask, voxel_size_zyx), tuple(voxel_size_zyx)


def mesh_from_file(path):
    """Load a mesh (obj/ply/stl/…) for skeletor-only comparison (no kimimaro mask)."""
    import trimesh
    return trimesh.load(path, process=False)


# --------------------------------------------------------------------------- #
# Result container + methods -> (name, vertices_xyz, edges, radii, seconds)
# --------------------------------------------------------------------------- #
@dataclass
class SkelResult:
    name: str
    vertices: np.ndarray | None = None      # (N, 3) xyz nm
    edges: np.ndarray | None = None         # (M, 2)
    radii: np.ndarray | None = None
    seconds: float = 0.0
    error: str = ""
    extra: dict = field(default_factory=dict)


def run_kimimaro(mask_zyx, voxel_size_zyx, *, name="kimimaro",
                 teasar_params=None, dust_threshold=0):
    import kimimaro
    tp = teasar_params or {"scale": 1.5, "const": 150, "pdrf_scale": 100000, "pdrf_exponent": 4}
    t0 = time.time()
    try:
        skels = kimimaro.skeletonize(mask_zyx, anisotropy=voxel_size_zyx, object_ids=[1],
                                     teasar_params=tp, dust_threshold=dust_threshold, progress=False)
        s = skels.get(1)
        if s is None:
            return SkelResult(name, error="no skeleton", seconds=time.time() - t0)
        verts = np.asarray(s.vertices)[:, ::-1]     # zyx nm -> xyz nm
        return SkelResult(name, verts, np.asarray(s.edges),
                          getattr(s, "radius", getattr(s, "radii", None)), time.time() - t0)
    except Exception as e:
        return SkelResult(name, error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)


def run_skeletor(mesh, method, *, name=None, per_component=False, post=(), **kwargs):
    import skeletor as sk
    import trimesh
    name = name or f"skeletor.{method}"
    fn = getattr(sk.skeletonize, f"by_{method}")
    parts = mesh.split(only_watertight=False) if per_component else [mesh]
    t0 = time.time()
    try:
        verts_all, edges_all, radii_all = [], [], []
        for part in parts:
            skel = fn(part, progress=False, **kwargs)
            for step in post:                       # e.g. ("clean_up",), ("remove_bristles",)
                skel = getattr(sk.post, step)(skel, mesh=part)
            off = sum(len(v) for v in verts_all)
            nv = len(skel.vertices)
            verts_all.append(np.asarray(skel.vertices))
            edges_all.append(np.asarray(skel.edges) + off)
            rad = getattr(skel, "radius", None)
            try:
                rad = np.asarray(rad, dtype=float).reshape(-1)
                if rad.shape[0] != nv:
                    rad = np.zeros(nv)
            except Exception:
                rad = np.zeros(nv)
            radii_all.append(rad)
        verts = np.vstack(verts_all)
        edges = np.vstack(edges_all) if any(len(e) for e in edges_all) else np.empty((0, 2), int)
        return SkelResult(name, verts, edges, np.concatenate(radii_all), time.time() - t0,
                          extra={"n_mesh_components": len(parts)})
    except Exception as e:
        return SkelResult(name, error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)


# --------------------------------------------------------------------------- #
# Metrics + SWC
# --------------------------------------------------------------------------- #
def skeleton_stats(res: SkelResult, twig_nm=2000.0) -> dict:
    if res.error or res.vertices is None or len(res.edges) == 0:
        return {"error": res.error or "empty"}
    import networkx as nx

    v, e = res.vertices, res.edges
    G = nx.Graph()
    G.add_nodes_from(range(len(v)))
    lengths = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    for (a, b), L in zip(e, lengths):
        G.add_edge(int(a), int(b), length=float(L))
    deg = dict(G.degree())
    tips = [n for n, d in deg.items() if d == 1]
    branch = [n for n, d in deg.items() if d >= 3]
    # twig = leaf->nearest branchpoint path shorter than twig_nm (noise proxy)
    bset = set(branch)
    twigs = 0
    for leaf in tips:
        acc, cur, prev = 0.0, leaf, None
        while deg.get(cur, 0) <= 2 or cur == leaf:
            nbrs = [n for n in G.neighbors(cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            acc += G[cur][nxt]["length"]
            prev, cur = cur, nxt
            if cur in bset:
                break
        if acc < twig_nm:
            twigs += 1
    return {"n_verts": len(v), "n_tips": len(tips), "n_branch": len(branch),
            "n_twigs<%dnm" % int(twig_nm): twigs, "n_components": nx.number_connected_components(G),
            "cable_um": round(float(lengths.sum()) / 1000, 1), "sec": round(res.seconds, 2),
            **res.extra}


def _display_mesh(mesh, max_faces=30000):
    """Decimate a mesh just for display (translucent surface), if it's large."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(max_faces)
    except Exception:
        return mesh


def _skeleton_traces(res: SkelResult, color: str):
    import networkx as nx
    import plotly.graph_objects as go

    v, e = res.vertices, res.edges
    xs, ys, zs = [], [], []
    for a, b in e:
        xs += [v[a, 0], v[b, 0], None]
        ys += [v[a, 1], v[b, 1], None]
        zs += [v[a, 2], v[b, 2], None]
    G = nx.Graph(); G.add_nodes_from(range(len(v)))
    G.add_edges_from([(int(a), int(b)) for a, b in e])
    deg = dict(G.degree())
    bp = [n for n, d in deg.items() if d >= 3]
    tp = [n for n, d in deg.items() if d == 1]
    return [
        go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=4),
                     name="skeleton", showlegend=False),
        go.Scatter3d(x=v[bp, 0], y=v[bp, 1], z=v[bp, 2], mode="markers",
                     marker=dict(size=4, color="red"), name="branch", showlegend=False),
        go.Scatter3d(x=v[tp, 0], y=v[tp, 1], z=v[tp, 2], mode="markers",
                     marker=dict(size=3, color="black"), name="tip", showlegend=False),
    ]


def visualize(results, mesh, path, title=""):
    """Write a self-contained interactive HTML: one 3D panel per method.

    Each panel shows the translucent mesh + that method's skeleton, with branch
    points (red) and tips (black) marked, and tips/branch/twig counts in the
    title — so convolution (extra red/black points) is visible at a glance. Open
    ``path`` in a browser (no server needed); rotate/zoom each panel.
    """
    import math
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    valid = [r for r in results if not r.error and r.vertices is not None and len(r.edges)]
    if not valid:
        return None
    disp = _display_mesh(mesh)
    mv, mf = disp.vertices, disp.faces

    n = len(valid)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    titles = []
    for r in valid:
        st = skeleton_stats(r)
        tw = next((v for k, v in st.items() if k.startswith("n_twigs")), "")
        titles.append(f"{r.name}  (tips={st.get('n_tips')} branch={st.get('n_branch')} "
                      f"comp={st.get('n_components')} twigs={tw})")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        specs=[[{"type": "scene"}] * cols for _ in range(rows)],
                        horizontal_spacing=0.01, vertical_spacing=0.06)
    for idx, r in enumerate(valid):
        row, col = idx // cols + 1, idx % cols + 1
        fig.add_trace(go.Mesh3d(x=mv[:, 0], y=mv[:, 1], z=mv[:, 2],
                                i=mf[:, 0], j=mf[:, 1], k=mf[:, 2], color="lightgray",
                                opacity=0.15, hoverinfo="skip", showlegend=False), row=row, col=col)
        for t in _skeleton_traces(r, "royalblue"):
            fig.add_trace(t, row=row, col=col)
    fig.update_scenes(aspectmode="data")
    fig.update_layout(title=title, height=380 * rows, margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def write_swc(path, res: SkelResult):
    """Write a generic SWC (forest supported): id type x y z radius parent."""
    import networkx as nx

    v, e = res.vertices, res.edges
    try:
        r = np.asarray(res.radii, dtype=float).reshape(-1)
        if r.shape[0] != len(v):
            r = np.ones(len(v))
    except Exception:
        r = np.ones(len(v))
    G = nx.Graph()
    G.add_nodes_from(range(len(v)))
    G.add_edges_from([(int(a), int(b)) for a, b in e])
    parent = {}
    order = []
    for comp in nx.connected_components(G):
        root = min(comp, key=lambda n: G.degree(n))     # a leaf if any
        for a, b in nx.bfs_edges(G, root):
            parent.setdefault(a, parent.get(a, -1))
            parent[b] = a
        parent.setdefault(root, -1)
        order += list(nx.bfs_tree(G, root).nodes)
    remap = {n: i + 1 for i, n in enumerate(order)}
    with open(path, "w") as f:
        f.write(f"# {res.name}\n")
        for n in order:
            p = parent.get(n, -1)
            f.write(f"{remap[n]} 0 {v[n,0]:.2f} {v[n,1]:.2f} {v[n,2]:.2f} "
                    f"{float(r[n]):.2f} {remap[p] if p in remap else -1}\n")
