"""Distances ALONG a skeleton from a reference region, for points attached to it.

The question: for each synapse on a neuron, how much cable lies between it and a reference
compartment — a neuropil ROI the neuron passes through, standing in for a compartment the
dataset does not label per cell. The answer is a signed path length: positive outside the
region, **negative inside it** (depth from the nearest boundary crossing), in nm.

Four steps, each a pure function, so every one is testable without a store:

1. :func:`bridge_gaps` — reconnect pieces separated by small breaks. Bodies here are
   genuinely fragmented, and a synapse on a piece not connected to the compartment has no
   path length at all. Bridges are straight edges between the nearest vertices of two
   pieces, added only below ``max_gap_nm``, and **flagged**: a bridge is cable nobody
   traced (a 400 nm join once doubled a body's cable length), so every result says how
   much of its path crossed one.
2. :meth:`neu_lib.Skeleton.split_at` — a vertex where each edge crosses the region
   boundary, so distances start at the surface rather than at the last vertex inside.
3. :func:`attach_points` — each synapse to the nearest point on an **edge**, not the
   nearest vertex: vertices can sit hundreds of nm apart. The snap distance is reported,
   since a synapse sits on the membrane, up to a radius off the centreline, and a large
   snap can mean it landed on the wrong nearby branch.
4. :func:`path_distances` — one multi-source shortest-path pass from every boundary
   vertex at once, which also handles a neuron entering the region more than once.

**Nothing is dropped.** A synapse whose piece cannot reach the boundary even after bridging
is ``unreachable``; a body whose skeleton never crosses the region is ``no_compartment``.
Dropping either would bias the distribution toward whatever is easy to reach.

Normalisation (subtracting a per-cell minimum, scaling by a maximum) is deliberately NOT
done here — it is an aggregation choice, and the table keeps absolute distances.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from neu_lib import InsideFn, Skeleton

#: Per-synapse status values.
OK = "ok"
UNREACHABLE = "unreachable"
NO_COMPARTMENT = "no_compartment"
NO_SKELETON = "no_skeleton"
STATUSES = (OK, UNREACHABLE, NO_COMPARTMENT, NO_SKELETON)

#: Floor for an edge weight in the sparse graph. scipy reads an explicit zero as "no edge",
#: so a zero-length edge (which skeletons do contain) would silently disconnect the graph.
_MIN_EDGE_NM = 1e-6

#: Which pieces of in-region cable count as the reference compartment.
REGION_PIECES = ("largest", "all")

#: Point-to-edge pairs evaluated per chunk in :func:`attach_points` (bounds its memory).
_ATTACH_PAIRS = 2_000_000


# --------------------------------------------------------------------------- #
# the graph
# --------------------------------------------------------------------------- #
def _lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if not len(edges):
        return np.zeros(0)
    return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)


def skeleton_graph(vertices: np.ndarray, edges: np.ndarray):
    """Symmetric ``csr_matrix`` adjacency weighted by edge length.

    Two scipy behaviours are silent here and both are handled: an explicit zero weight is
    *no* edge (floored at ``_MIN_EDGE_NM``), and duplicate entries are *summed* on
    conversion (duplicates are removed first, so a doubly-listed edge is not twice as long).
    Self-loops are dropped.
    """
    from scipy.sparse import coo_matrix

    v = np.asarray(vertices, dtype=np.float64)
    e = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(e, axis=0) if len(e) else e
    w = np.maximum(_lengths(v, e), _MIN_EDGE_NM)
    n = len(v)
    return coo_matrix((np.concatenate([w, w]),
                       (np.concatenate([e[:, 0], e[:, 1]]),
                        np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()


def components(skeleton: Skeleton) -> tuple[int, np.ndarray]:
    """``(n_components, label_per_vertex)``. An isolated vertex is its own component."""
    from scipy.sparse.csgraph import connected_components

    n, labels = connected_components(
        skeleton_graph(skeleton.vertices_zyx_nm, skeleton.edges), directed=False)
    return int(n), labels


# --------------------------------------------------------------------------- #
# 1. bridging small breaks
# --------------------------------------------------------------------------- #
def bridge_gaps(skeleton: Skeleton, max_gap_nm: float) -> tuple[Skeleton, np.ndarray]:
    """Join the skeleton's pieces wherever two come within ``max_gap_nm`` of each other.

    Returns ``(skeleton, is_bridge)``: the original vertices and edges, unchanged and in
    order, with bridge edges appended, and one bool per edge marking the bridges.

    Each piece is joined to the others by the shortest links that connect them (a minimum
    spanning tree over pieces, using the nearest vertex pair between each two), keeping
    only links no longer than ``max_gap_nm``. That is the same result as
    ``kimimaro.join_close_components``' greedy closest-pair merge, without its vertex
    reordering — the vertex indices here must stay stable — and with the bridges marked
    rather than indistinguishable from traced cable. Never adds a cycle.

    ``max_gap_nm = 0`` bridges nothing. Candidate pairs come from a KD-tree query within
    ``max_gap_nm``, so cost grows with vertex density times the gap; fine at the gaps
    this is for (up to ~a micron), wasteful for an "join everything" gap.
    """
    if max_gap_nm < 0:
        raise ValueError(f"max_gap_nm must be >= 0, got {max_gap_nm}")
    verts = skeleton.vertices_zyx_nm
    edges = skeleton.edges
    flags = np.zeros(len(edges), dtype=bool)

    def unchanged():
        return (Skeleton(verts.copy(), edges.copy(),
                         None if skeleton.radii_nm is None else skeleton.radii_nm.copy(),
                         skeleton.name), flags)

    if len(verts) < 2 or max_gap_nm == 0:
        return unchanged()
    n_comp, label = components(skeleton)
    if n_comp == 1:
        return unchanged()

    from scipy.spatial import cKDTree

    v = verts.astype(np.float64)
    pairs = cKDTree(v).query_pairs(r=float(max_gap_nm), output_type="ndarray")
    if len(pairs):
        pairs = pairs[label[pairs[:, 0]] != label[pairs[:, 1]]]
    if not len(pairs):
        return unchanged()

    # The shortest vertex pair for each pair of pieces.
    d = np.linalg.norm(v[pairs[:, 1]] - v[pairs[:, 0]], axis=1)
    ca, cb = label[pairs[:, 0]], label[pairs[:, 1]]
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    order = np.lexsort((d, hi, lo))
    lo, hi, d, pairs = lo[order], hi[order], d[order], pairs[order]
    first = np.ones(len(lo), dtype=bool)
    first[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    lo, hi, d, pairs = lo[first], hi[first], d[first], pairs[first]

    # Kruskal over pieces.
    parent = np.arange(n_comp)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    chosen = []
    for i in np.argsort(d, kind="stable"):
        a, b = find(int(lo[i])), find(int(hi[i]))
        if a != b:
            parent[a] = b
            chosen.append(i)
    bridges = pairs[np.asarray(chosen, dtype=np.int64)]
    return (Skeleton(verts.copy(), np.concatenate([edges, bridges]),
                     None if skeleton.radii_nm is None else skeleton.radii_nm.copy(),
                     skeleton.name),
            np.concatenate([flags, np.ones(len(bridges), dtype=bool)]))


# --------------------------------------------------------------------------- #
# 3. attaching points
# --------------------------------------------------------------------------- #
def attach_points(skeleton: Skeleton, points_zyx_nm: Any) -> dict[str, np.ndarray]:
    """The nearest point on any edge, for each point.

    Returns ``{"edge", "t", "position", "snap_nm"}``: the edge index, the fraction along it
    from ``edges[:, 0]`` to ``edges[:, 1]``, the attached position (zyx nm) and the distance
    to it. Exact (every edge is tested, in bounded chunks) rather than a nearest-vertex
    prefilter, which can pick the wrong edge when edges are long. Vertices with no edge
    cannot carry a path length and are not attachment targets.
    """
    pts = np.asarray(points_zyx_nm, dtype=np.float64).reshape(-1, 3)
    v = skeleton.vertices_zyx_nm.astype(np.float64)
    e = skeleton.edges
    if not len(e):
        raise ValueError(f"{skeleton!r} has no edges to attach points to")
    a = v[e[:, 0]]
    seg = v[e[:, 1]] - a
    seg_len2 = np.einsum("ij,ij->i", seg, seg)
    safe = np.where(seg_len2 > 0, seg_len2, 1.0)

    edge = np.zeros(len(pts), dtype=np.int64)
    t = np.zeros(len(pts))
    dist2 = np.zeros(len(pts))
    step = max(1, _ATTACH_PAIRS // len(e))
    for s in range(0, len(pts), step):
        p = pts[s:s + step]
        rel = p[:, None, :] - a[None, :, :]
        tt = np.clip(np.einsum("pmk,mk->pm", rel, seg) / safe, 0.0, 1.0)
        tt[:, seg_len2 == 0] = 0.0
        off = rel - tt[:, :, None] * seg[None, :, :]
        dd = np.einsum("pmk,pmk->pm", off, off)
        best = np.argmin(dd, axis=1)
        rows = np.arange(len(p))
        edge[s:s + step], t[s:s + step], dist2[s:s + step] = best, tt[rows, best], dd[rows, best]
    position = a[edge] + t[:, None] * seg[edge]
    return {"edge": edge, "t": t, "position": position, "snap_nm": np.sqrt(dist2)}


# --------------------------------------------------------------------------- #
# 4. path lengths
# --------------------------------------------------------------------------- #
def path_distances(skeleton: Skeleton, sources: Sequence[int],
                   is_bridge: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Shortest path length from the nearest of ``sources``, for every vertex.

    Returns ``{"distance_nm", "n_bridges", "bridge_nm", "source"}`` per vertex: the
    bridge edges on that shortest path and their length, and which source vertex it starts
    from. Unreachable vertices get ``inf``, zero bridges and source ``-1``.
    """
    from scipy.sparse.csgraph import dijkstra

    v = skeleton.vertices_zyx_nm.astype(np.float64)
    e = skeleton.edges
    n = len(v)
    sources = np.asarray(sources, dtype=np.int64)
    n_bridges = np.zeros(n, dtype=np.int64)
    bridge_nm = np.zeros(n)
    if not len(sources):
        return {"distance_nm": np.full(n, np.inf), "n_bridges": n_bridges,
                "bridge_nm": bridge_nm, "source": np.full(n, -1, dtype=np.int64)}
    # min_only=True also returns which source each vertex was reached from.
    dist, pred, nearest = dijkstra(skeleton_graph(v, e), directed=False, indices=sources,
                                    min_only=True, return_predecessors=True)

    if is_bridge is not None and np.any(is_bridge):
        lengths = _lengths(v, e)
        bridge_len = {}
        for (i, j), length in zip(e[is_bridge].tolist(), lengths[is_bridge].tolist()):
            bridge_len[(min(i, j), max(i, j))] = length
        reachable = np.flatnonzero(np.isfinite(dist) & (pred >= 0))
        for vtx in reachable[np.argsort(dist[reachable], kind="stable")].tolist():
            p = int(pred[vtx])
            extra = bridge_len.get((min(p, vtx), max(p, vtx)))
            n_bridges[vtx] = n_bridges[p] + (extra is not None)
            bridge_nm[vtx] = bridge_nm[p] + (extra or 0.0)
    source = np.where(np.isfinite(dist), nearest, -1).astype(np.int64)
    return {"distance_nm": dist, "n_bridges": n_bridges, "bridge_nm": bridge_nm,
            "source": source}


# --------------------------------------------------------------------------- #
# the whole measurement, one body
# --------------------------------------------------------------------------- #
def compartment_graph(skeleton: Skeleton, inside: InsideFn, *, max_gap_nm: float,
                      tolerance_nm: float = 8.0, region_pieces: str = "largest",
                      merge_nm: float = 0.0) -> dict[str, Any]:
    """The bridged, boundary-split skeleton and which of its cable is the compartment.

    The graph :func:`synapse_distances` measures on, exposed so a renderer can draw exactly
    the cable that was treated as the compartment. Returns ``skeleton`` (bridged and split),
    per-edge ``is_bridge`` / ``in_region`` / ``in_compartment``, per-vertex ``inside`` /
    ``boundary`` / ``in_compartment``, the ``sources`` (boundary vertices of the
    compartment), edge ``lengths``, and the piece counts. See :func:`synapse_distances` for
    ``region_pieces`` and ``merge_nm``.
    """
    if region_pieces not in REGION_PIECES:
        raise ValueError(f"region_pieces is one of {REGION_PIECES}, not {region_pieces!r}")
    if merge_nm < 0:
        raise ValueError(f"merge_nm must be >= 0, got {merge_nm}")
    n_before, _ = components(skeleton)
    bridged, is_bridge = bridge_gaps(skeleton, max_gap_nm)
    n_after, _ = components(bridged)

    split, inside_v, boundary = bridged.split_at(inside, tolerance_nm=tolerance_nm)
    # split_at's edge layout: uncrossed edges in order, then the inside halves, then the
    # outside halves of the crossing edges — so the bridge flags follow the same way.
    e = bridged.edges
    crossing = inside_v[e[:, 0]] ^ inside_v[e[:, 1]]
    split_bridge = np.concatenate([is_bridge[~crossing], is_bridge[crossing],
                                   is_bridge[crossing]])

    v = split.vertices_zyx_nm.astype(np.float64)
    lengths = _lengths(v, split.edges)
    edge_inside = inside_v[split.edges[:, 0]] & inside_v[split.edges[:, 1]]

    # The in-region pieces, and which of them is the compartment.
    from scipy.sparse.csgraph import connected_components

    in_edges = split.edges[edge_inside]
    _, piece = connected_components(skeleton_graph(v, in_edges), directed=False)
    traced = np.where(split_bridge[edge_inside], 0.0, lengths[edge_inside])
    piece_cable = np.bincount(piece[in_edges[:, 0]], weights=traced,
                              minlength=int(piece.max()) + 1 if len(piece) else 0)
    has_cable = np.zeros(len(piece_cable), dtype=bool)
    has_cable[piece[in_edges[:, 0]]] = True
    if region_pieces == "largest" and has_cable.any():
        chosen = np.zeros(len(piece_cable), dtype=bool)
        chosen[int(np.argmax(np.where(has_cable, piece_cable, -1.0)))] = True
    else:
        chosen = has_cable
    if region_pieces == "largest" and merge_nm > 0 and has_cable.sum() > 1:
        chosen = _absorb_nearby(split, inside_v, piece, has_cable, chosen, merge_nm)
    in_compartment = inside_v & chosen[piece] if len(piece) else inside_v
    edge_compartment = edge_inside & in_compartment[split.edges[:, 0]]
    sources = np.flatnonzero(boundary & in_compartment)

    return {"skeleton": split, "bridged": bridged, "is_bridge_bridged": is_bridge,
            "is_bridge": split_bridge, "in_region": edge_inside,
            "in_compartment": edge_compartment, "inside": inside_v, "boundary": boundary,
            "vertex_in_compartment": in_compartment, "sources": sources, "lengths": lengths,
            "n_components": n_before, "n_components_bridged": n_after,
            "n_region_pieces": int(has_cable.sum()), "n_compartment_pieces": int(chosen.sum())}


def synapse_distances(skeleton: Skeleton, points_zyx_nm: Any, inside: InsideFn, *,
                      max_gap_nm: float, tolerance_nm: float = 8.0,
                      region_pieces: str = "largest", merge_nm: float = 0.0
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Signed path length from the reference compartment, for each point on one body.

    **The compartment is a connected piece of in-region cable, not the region itself.**
    Inside the region the skeleton falls into pieces (connected runs of cable between
    boundary crossings). ``region_pieces="largest"`` — the default — takes only the piece
    with the most cable; ``"all"`` takes every piece. The default exists because of a
    measured failure: a coarse ROI brushes neighbouring arbor, and on 14 of 70 Megaphragma
    T4s the DENDRITE dipped into the lobula ROI for 0.1-4 µm of cable, right at the medulla
    border. With ``"all"``, most of those cells' synapses were measured from the dips rather
    than from the ~20 µm axon terminal — short distances, nothing raised. The cost of
    ``"largest"``: a terminal that weaves across the boundary splits into several pieces and
    only the biggest counts, so a weave before it adds its length to every distance — up to
    18 µm on the same T4s, where up to 13 pieces made one terminal.

    ``merge_nm`` (with ``"largest"``) grows the compartment from the largest piece by
    absorbing, repeatedly, every other piece within that much cable of it — so a weaving
    terminal is one compartment again while a dip at the far end of the neurite is not.
    The T4 data do not separate cleanly by this distance (pieces sit at 0-22 µm from the
    largest), so it is a parameter to sweep, not a constant.

    Returns ``(per_point, summary)``. ``per_point`` holds arrays, one entry per point:
    ``distance_nm`` (negative inside the compartment; NaN unless ``status == "ok"``),
    ``inside`` (in the compartment), ``in_roi`` (in the region at all), ``snap_nm``, ``n_bridges``, ``bridge_nm``, ``on_bridge`` (the point
    attached to a bridge edge itself), ``status``, ``position`` (the attached point, zyx
    nm), ``radius_nm`` (the skeleton's radius there, interpolated; NaN without radii — so
    ``snap_nm`` can be read against the neurite's own calibre) and ``entry`` (the boundary
    crossing the distance was measured from, zyx nm). ``summary`` describes the body's
    graph: pieces before and after bridging, bridges added and their length, total,
    in-region and compartment cable, how many in-region pieces there are, boundary
    crossings (all, and into the compartment), and how many crossings any point was
    measured from
    (``n_entries_used`` — more than one means the region is entered at several places, which
    is worth looking at before pooling).
    """
    if region_pieces not in REGION_PIECES:
        raise ValueError(f"region_pieces is one of {REGION_PIECES}, not {region_pieces!r}")
    if merge_nm < 0:
        raise ValueError(f"merge_nm must be >= 0, got {merge_nm}")
    pts = np.asarray(points_zyx_nm, dtype=np.float64).reshape(-1, 3)
    g = compartment_graph(skeleton, inside, max_gap_nm=max_gap_nm, tolerance_nm=tolerance_nm,
                          region_pieces=region_pieces, merge_nm=merge_nm)
    split, bridged, is_bridge = g["skeleton"], g["bridged"], g["is_bridge_bridged"]
    split_bridge, edge_inside, edge_compartment = (g["is_bridge"], g["in_region"],
                                                   g["in_compartment"])
    inside_v, boundary, sources, lengths = (g["inside"], g["boundary"], g["sources"],
                                            g["lengths"])
    n_before, n_after = g["n_components"], g["n_components_bridged"]
    e = bridged.edges
    v = split.vertices_zyx_nm.astype(np.float64)

    summary = {
        "n_vertices": int(len(skeleton.vertices_zyx_nm)),
        "n_components": n_before,
        "n_components_bridged": n_after,
        "n_bridges": int(is_bridge.sum()),
        "bridge_nm_total": float(_lengths(bridged.vertices_zyx_nm.astype(np.float64),
                                          e[is_bridge]).sum()),
        "bridge_nm_max": float(_lengths(bridged.vertices_zyx_nm.astype(np.float64),
                                        e[is_bridge]).max()) if is_bridge.any() else 0.0,
        "cable_nm": float(lengths[~split_bridge].sum()),
        "cable_inside_nm": float(lengths[~split_bridge & edge_inside].sum()),
        "compartment_cable_nm": float(lengths[~split_bridge & edge_compartment].sum()),
        "n_region_pieces": g["n_region_pieces"],
        "n_compartment_pieces": g["n_compartment_pieces"],
        "region_pieces": region_pieces,
        "merge_nm": float(merge_nm),
        "n_boundary_crossings": int(boundary.sum()),
        "n_compartment_crossings": int(len(sources)),
        "max_gap_nm": float(max_gap_nm),
    }

    n = len(pts)
    out = {"distance_nm": np.full(n, np.nan), "inside": np.zeros(n, dtype=bool),
           "in_roi": np.zeros(n, dtype=bool),
           "snap_nm": np.full(n, np.nan), "n_bridges": np.zeros(n, dtype=np.int64),
           "bridge_nm": np.zeros(n), "on_bridge": np.zeros(n, dtype=bool),
           "status": np.full(n, OK, dtype=object), "position": np.full((n, 3), np.nan),
           "radius_nm": np.full(n, np.nan), "entry": np.full((n, 3), np.nan)}
    summary["n_entries_used"] = 0
    if not n:
        return out, summary
    if not len(split.edges):
        out["status"][:] = UNREACHABLE
        return out, summary

    att = attach_points(split, pts)
    out["snap_nm"], out["position"] = att["snap_nm"], att["position"]
    ea, eb = split.edges[att["edge"], 0], split.edges[att["edge"], 1]
    out["inside"] = edge_compartment[att["edge"]]
    out["in_roi"] = edge_inside[att["edge"]]
    out["on_bridge"] = split_bridge[att["edge"]]
    if split.radii_nm is not None:
        r = split.radii_nm.astype(np.float64)
        out["radius_nm"] = r[ea] + att["t"] * (r[eb] - r[ea])
    if not len(sources):
        out["status"][:] = NO_COMPARTMENT
        return out, summary

    paths = path_distances(split, sources, split_bridge)
    d, nb, bn = paths["distance_nm"], paths["n_bridges"], paths["bridge_nm"]
    length = lengths[att["edge"]]
    t = att["t"]
    via_a = d[ea] + t * length
    via_b = d[eb] + (1.0 - t) * length
    use_a = via_a <= via_b
    dist = np.where(use_a, via_a, via_b)
    partial = np.where(use_a, t, 1.0 - t) * length
    on = out["on_bridge"]
    out["n_bridges"] = np.where(use_a, nb[ea], nb[eb]) + on
    out["bridge_nm"] = np.where(use_a, bn[ea], bn[eb]) + np.where(on, partial, 0.0)

    ok = np.isfinite(dist)
    entry = np.where(use_a, paths["source"][ea], paths["source"][eb])
    out["entry"][ok] = v[entry[ok]]
    summary["n_entries_used"] = int(len(np.unique(entry[ok])))
    out["status"][~ok] = UNREACHABLE
    out["distance_nm"] = np.where(ok, np.where(out["inside"], -dist, dist), np.nan)
    out["n_bridges"] = np.where(ok, out["n_bridges"], 0)
    out["bridge_nm"] = np.where(ok, out["bridge_nm"], 0.0)
    return out, summary


def _absorb_nearby(skeleton: Skeleton, inside_v: np.ndarray, piece: np.ndarray,
                   has_cable: np.ndarray, chosen: np.ndarray, merge_nm: float) -> np.ndarray:
    """Grow ``chosen`` by every piece within ``merge_nm`` of cable, until nothing is added."""
    from scipy.sparse.csgraph import dijkstra

    graph = skeleton_graph(skeleton.vertices_zyx_nm, skeleton.edges)
    chosen = chosen.copy()
    while True:
        members = np.flatnonzero(inside_v & chosen[piece])
        d = dijkstra(graph, directed=False, indices=members, min_only=True)
        nearest = np.full(len(chosen), np.inf)
        np.minimum.at(nearest, piece[inside_v], d[inside_v])
        add = has_cable & ~chosen & (nearest <= merge_nm)
        if not add.any():
            return chosen
        chosen |= add


# --------------------------------------------------------------------------- #
# many bodies, as tables
# --------------------------------------------------------------------------- #
def synapse_distance_table(skeletons: Mapping[int, Skeleton], synapses: Any,
                           inside: InsideFn | Mapping[int, InsideFn], *, frame: Any,
                           max_gap_nm: float,
                           body_column: str = "post_body", prefix: str = "post_",
                           tolerance_nm: float = 8.0, region_pieces: str = "largest",
                           merge_nm: float = 0.0, dataset: Optional[str] = None):
    """Per-synapse and per-body tables over a set of bodies.

    ``synapses`` is any table with ``{prefix}z/y/x`` columns and ``body_column`` — e.g.
    neu-mark's ``connections`` filtered to the partners wanted, where ``post_*`` is the
    postsynaptic site on the body being measured. Its coordinates are in ``frame``'s voxels
    (scale 0 for DVID). **Every input column is carried through**, so ``pre_body`` or a
    partner cell type stays on each row.

    ``inside`` is one region for every body (an ROI), or a mapping ``{body: inside}`` for
    a compartment each cell marks itself (:func:`near_points_region` over its own outputs).
    A body the mapping lacks gets ``status == "no_compartment"``.

    Returns ``(rows, bodies)``. ``rows`` adds ``distance_nm`` (signed; negative inside the
    compartment — see :func:`synapse_distances` on ``region_pieces``), ``inside``,
    ``in_roi``, ``snap_nm``, ``n_bridges``, ``bridge_nm``, ``on_bridge``,
    ``status``, ``radius_nm``, ``attached_z/y/x_nm``, ``entry_z/y/x_nm`` (the boundary
    crossing measured from) and the ``max_gap_nm`` / ``region_pieces`` / ``merge_nm`` it
    was measured with. A body in
    the table with no skeleton gets ``status == "no_skeleton"`` rather than vanishing.
    ``bodies`` holds :func:`synapse_distances`' summary per body.
    """
    import pandas as pd

    table = synapses.reset_index(drop=True)
    cols = [f"{prefix}{a}" for a in ("z", "y", "x")]
    missing = [c for c in [*cols, body_column] if c not in table.columns]
    if missing:
        raise KeyError(f"synapse table has no column(s) {missing}")
    coords_nm = frame.to_nm(table[cols].to_numpy(dtype=np.float64))
    body_of = table[body_column].astype("int64").to_numpy()

    n = len(table)
    res = {"distance_nm": np.full(n, np.nan), "inside": np.zeros(n, dtype=bool),
           "in_roi": np.zeros(n, dtype=bool), "snap_nm": np.full(n, np.nan), "n_bridges": np.zeros(n, dtype=np.int64),
           "bridge_nm": np.zeros(n), "on_bridge": np.zeros(n, dtype=bool),
           "status": np.full(n, NO_SKELETON, dtype=object),
           "position": np.full((n, 3), np.nan), "radius_nm": np.full(n, np.nan),
           "entry": np.full((n, 3), np.nan)}
    summaries = []
    for body in dict.fromkeys(body_of.tolist()):
        rows = np.flatnonzero(body_of == body)
        skel = skeletons.get(int(body))
        if skel is None:
            summaries.append({"body_id": int(body), "status": NO_SKELETON,
                              "n_synapses": int(len(rows))})
            continue
        region = inside.get(int(body)) if isinstance(inside, Mapping) else inside
        if region is None:
            res["status"][rows] = NO_COMPARTMENT
            summaries.append({"body_id": int(body), "status": NO_COMPARTMENT,
                              "n_synapses": int(len(rows))})
            continue
        per, summary = synapse_distances(skel, coords_nm[rows], region,
                                         max_gap_nm=max_gap_nm, tolerance_nm=tolerance_nm,
                                         region_pieces=region_pieces, merge_nm=merge_nm)
        for k in res:
            res[k][rows] = per[k]
        statuses = pd.Series(per["status"]).value_counts()
        summaries.append({"body_id": int(body), "status": OK, "n_synapses": int(len(rows)),
                          **{f"n_{s}": int(statuses.get(s, 0))
                             for s in (OK, UNREACHABLE, NO_COMPARTMENT)},
                          **summary})

    out = table.copy()
    if dataset is not None:
        out.insert(0, "dataset", dataset)
    for k in ("distance_nm", "inside", "in_roi", "snap_nm", "radius_nm", "n_bridges",
              "bridge_nm", "on_bridge"):
        out[k] = res[k]
    out["status"] = pd.Series(res["status"], dtype="string")
    for i, a in enumerate(("z", "y", "x")):
        out[f"attached_{a}_nm"] = res["position"][:, i]
    for i, a in enumerate(("z", "y", "x")):
        out[f"entry_{a}_nm"] = res["entry"][:, i]
    out["max_gap_nm"] = float(max_gap_nm)
    out["region_pieces"] = region_pieces
    out["merge_nm"] = float(merge_nm)
    bodies = pd.DataFrame(summaries)
    if dataset is not None and len(bodies):
        bodies.insert(0, "dataset", dataset)
    return out, bodies


# --------------------------------------------------------------------------- #
# regions
# --------------------------------------------------------------------------- #
def dvid_roi_region(source: Any, rois: Sequence[str] | str, *,
                    voxel_size_nm: Sequence[float],
                    origin_nm: Sequence[float] = (0.0, 0.0, 0.0)):
    """``(inside, roi_volume)`` for the union of named DVID ROIs.

    ``source`` is a ``dvid://`` URL or spec for any instance on the node. ``voxel_size_nm``
    / ``origin_nm`` are the node's scale-0 grid — the frame its synapse coordinates are in
    (``neu_draw.sources.volume_frame(volume, 0)`` for the exported segmentation).
    """
    from neu_lib import mask_predicate
    from neu_vol import dvid

    spec = dvid.parse_url(source) if isinstance(source, str) else source
    names = [rois] if isinstance(rois, str) else list(rois)
    vol = dvid.roi_volume(spec, names, voxel_size_nm=voxel_size_nm, origin_nm=origin_nm)
    return mask_predicate(vol.mask(), vol.frame), vol


def near_points_region(points_zyx_nm: Any, radius_nm: float) -> InsideFn:
    """``inside()`` for "within ``radius_nm`` of any of these points".

    For a compartment the cell itself marks, where no ROI separates it: T5 dendrites AND
    axon terminals both lie in the Megaphragma lobula (90% of each cell's cable is inside
    ``LO(L)``), so a lobula ROI cannot tell them apart, but the cell's OWN presynaptic sites
    sit on its terminal. A T-bar is on the membrane, a radius or so off the centreline, so
    ``radius_nm`` must reach the skeleton from there; gaps between T-bars along the terminal
    then become separate pieces, which ``merge_nm`` rejoins. A few outputs elsewhere (on a
    dendrite) make small pieces that ``region_pieces="largest"`` leaves out.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points_zyx_nm, dtype=np.float64).reshape(-1, 3)
    if radius_nm <= 0:
        raise ValueError(f"radius_nm must be positive, got {radius_nm}")
    if not len(pts):
        return lambda q: np.zeros(len(np.asarray(q)), dtype=bool)
    tree = cKDTree(pts)

    def inside(q: np.ndarray) -> np.ndarray:
        d, _ = tree.query(np.asarray(q, dtype=np.float64).reshape(-1, 3), k=1,
                          distance_upper_bound=float(radius_nm))
        return np.isfinite(d)

    return inside


def volume_roi_region(volume: str, labels: Sequence[int | str] | int | str, *,
                      level: int):
    """``(inside, piece)`` for labels of a precomputed ROI volume, read whole at ``level``.

    Labels may be ids or names; names resolve through the volume's own
    ``segment_properties`` and are never guessed. Pick a coarse level — the whole level is
    read into memory.
    """
    from neu_lib import mask_predicate
    from neu_vol import read_piece

    wanted = [labels] if isinstance(labels, (int, str, np.integer)) else list(labels)
    if any(isinstance(lab, str) for lab in wanted):
        from .compartments import semantic_label_names

        by_name = {name: i for i, name in semantic_label_names(volume).items()}
        unknown = [lab for lab in wanted if isinstance(lab, str) and lab not in by_name]
        if unknown:
            raise KeyError(f"{volume} has no label named {unknown}; it has "
                           f"{sorted(by_name)}")
        wanted = [by_name[lab] if isinstance(lab, str) else int(lab) for lab in wanted]
    piece = read_piece(volume, level=level)
    return mask_predicate(np.isin(piece.array, np.asarray(wanted)), piece.frame), piece
