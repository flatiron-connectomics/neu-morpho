"""Radius-aware node reduction — NeuTu's two post-processing passes, ported.

Step 1 (``neutu_trace``) gets the *centreline* right but leaves a dense, roughly
per-voxel node set. NeuTu's node economy comes from two passes that run after
tracing, and both are **tool-independent**: they take any skeleton as
``(vertices, radii, edges)`` and return a smaller one, so they apply to kimimaro's
output just as well.

:func:`region_sample`
    ``ZSwcGenerator::createSwcByRegionSampling``
    (``NeuTu/neurolabi/gui/zswcgenerator.cpp:196``). Walk nodes in order of
    decreasing radius; a kept node suppresses every not-yet-kept node strictly
    inside its own ball. Node spacing ends up proportional to local thickness —
    dense in thin neurites, sparse through a blob.

:func:`optimal_downsample`
    ``ZSwcResampler::optimalDownsample``
    (``NeuTu/neurolabi/gui/swc/zswcresampler.cpp:90``). Iterate to a fixpoint,
    dropping a node when its ball is contained in a neighbour's, when the two
    overlap significantly, or when interpolating parent↔child reproduces it in
    both position and radius. A radius-aware Douglas–Peucker; it smooths radii as
    a side effect.

Units are whatever the caller uses, as long as positions and radii share them.

Two deliberate departures from NeuTu
------------------------------------
**Topology is preserved by graph contraction, not by re-chaining.**
``createSwcByRegionSampling`` consumes a ``ZVoxelArray`` — a single traced path —
and re-emits the survivors as a linear chain. Our input is already a tree, so
chaining would destroy branch structure. Instead the kept nodes inherit the
original connectivity: kept neighbours stay joined, and each connected clump of
dropped nodes is replaced by a star joining the kept nodes it used to link.

**The suppression sweep is a forward mark, not an O(n²) scan.** NeuTu compares
each node against every larger kept node. Marking forward from each kept node
through a KD-tree is equivalent — a node is dropped iff some kept node of
larger-or-equal radius contains it — and turns O(n²) into O(n log n), which
matters at the ~50 K nodes step 1 produces on a large body.
"""

from __future__ import annotations

import numpy as np

# NeuTu's ZSwcResampler defaults (zswcresampler.cpp:14-16).
RADIUS_SCALE = 1.2
DISTANCE_SCALE = 2.0

# Exempt branch points and tips from region-sampling suppression. Defaults ON:
# a global region_sample destroys branch structure the trace got right, which is a
# defect of our generalisation rather than a tuning choice. Measured on body
# 45892915: raw trace 66 branch points / mean degree 3.02 / max 4, matching NeuTu
# exactly; without this, 44 / 3.52 / 7. Cable ratio also improves (1.09x -> 1.05x)
# for the cost of ~2 nodes. See region_sample.
PRESERVE_JUNCTIONS = True


# --------------------------------------------------------------------------- #
# step 2: radius-adaptive node placement
# --------------------------------------------------------------------------- #
def region_sample(vertices, radii, edges, preserve_junctions=PRESERVE_JUNCTIONS):
    """Drop every node that sits strictly inside a larger kept node's ball.

    Returns ``(vertices, radii, edges)`` reindexed onto the survivors.

    ``preserve_junctions`` exempts branch points (degree ≥ 3) and tips (degree 1)
    from suppression. **This is where NeuTu and this port diverge structurally.**
    ``createSwcByRegionSampling`` consumes one traced *path* — a linear chain — so
    it can never merge across a junction. Applied globally to a tree, as here, a
    ball centred on a fat node beside a junction can swallow the junction and its
    neighbours, collapsing several branch points into one hub.

    Measured on body 45892915: the raw trace has **66 branch points, mean degree
    3.02, max 4 — identical to NeuTu**. After a global region_sample it is 44
    branch points, mean degree 3.52, max 7. The branch topology is not lost in
    tracing; it is lost here.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=float)
    r = np.asarray(radii, dtype=float)
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    n = len(v)
    if n == 0:
        return v, r, e

    protected = np.zeros(n, dtype=bool)
    if preserve_junctions and len(e):
        deg = np.bincount(e.ravel(), minlength=n)
        protected = (deg >= 3) | (deg == 1)

    tree = cKDTree(v)
    kept = np.zeros(n, dtype=bool)
    dropped = np.zeros(n, dtype=bool)

    # Descending radius. NeuTu sorts by -value, so the largest ball claims first
    # and the node spacing that falls out is proportional to local thickness.
    for i in np.argsort(-r, kind="stable"):
        if dropped[i]:
            continue
        kept[i] = True
        if r[i] <= 0:
            continue
        near = tree.query_ball_point(v[i], r[i])
        if not near:
            continue
        near = np.asarray(near)
        near = near[near != i]
        if not len(near):
            continue
        # strict '<' -- NeuTu uses `dist < prevVoxel.value()`, and a node exactly
        # on the ball surface is kept
        inside = near[np.linalg.norm(v[near] - v[i], axis=1) < r[i]]
        inside = inside[~kept[inside] & ~protected[inside]]
        dropped[inside] = True

    return _contract(v, r, e, kept)


def _contract(v, r, e, kept):
    """Reindex onto ``kept``, rewiring through the dropped nodes.

    Kept neighbours keep their edge. Each connected clump of dropped nodes is
    replaced by a star joining the kept nodes it used to connect — for a tree
    that is exactly the ``k-1`` edges needed to reconnect ``k`` neighbours
    without inventing a cycle.
    """
    n = len(v)
    new_index = np.full(n, -1, dtype=int)
    new_index[kept] = np.arange(int(kept.sum()))

    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in e:
        adj[a].append(b)
        adj[b].append(a)

    out = set()
    for a, b in e:
        if kept[a] and kept[b]:
            out.add((min(new_index[a], new_index[b]), max(new_index[a], new_index[b])))

    seen = np.zeros(n, dtype=bool)
    for start in range(n):
        if kept[start] or seen[start]:
            continue
        # flood the clump of dropped nodes, collecting the kept nodes on its rim
        stack, clump, rim = [start], [], set()
        seen[start] = True
        while stack:
            u = stack.pop()
            clump.append(u)
            for w in adj[u]:
                if kept[w]:
                    rim.add(w)
                elif not seen[w]:
                    seen[w] = True
                    stack.append(w)
        if len(rim) < 2:
            continue
        rim_list = sorted(rim)
        hub = max(rim_list, key=lambda k: r[k])
        for other in rim_list:
            if other == hub:
                continue
            out.add((min(new_index[hub], new_index[other]),
                     max(new_index[hub], new_index[other])))

    edges_out = (np.array(sorted(out), dtype=int) if out
                 else np.zeros((0, 2), dtype=int))
    return v[kept], r[kept], edges_out


# --------------------------------------------------------------------------- #
# step 3: radius-aware simplification to a fixpoint
# --------------------------------------------------------------------------- #
def optimal_downsample(vertices, radii, edges, *, radius_scale=RADIUS_SCALE,
                       distance_scale=DISTANCE_SCALE, max_rounds=100):
    """Iterate NeuTu's ``suboptimalDownsample`` until no node is removed."""
    v = np.asarray(vertices, dtype=float).copy()
    r = np.asarray(radii, dtype=float).copy()
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(v) == 0:
        return v, r, e

    parent, children, roots = _root_forest(len(v), e, r)
    alive = np.ones(len(v), dtype=bool)
    weight = np.ones(len(v), dtype=float)

    for _ in range(max_rounds):
        removed = 0
        # children-before-parents, so a node is judged against a parent that has
        # already absorbed everything below it this round
        for tn in _postorder(roots, children):
            if not alive[tn] or parent[tn] < 0:
                continue
            p = parent[tn]
            if not alive[p]:
                continue
            option = _decide(tn, p, v, r, parent, children, alive,
                             radius_scale, distance_scale)
            if option is None:
                continue
            if option == "child":
                v[p], r[p] = v[tn], r[tn]
            elif option == "weighted":
                w = weight[tn] + weight[p]
                v[p] = (v[tn] * weight[tn] + v[p] * weight[p]) / w
                r[p] = (r[tn] * weight[tn] + r[p] * weight[p]) / w
                weight[p] += 1.0
            for c in children[tn]:
                parent[c] = p
                children[p].add(c)
            children[p].discard(tn)
            children[tn] = set()
            alive[tn] = False
            removed += 1
        if removed == 0:
            break

    idx = np.flatnonzero(alive)
    remap = np.full(len(v), -1, dtype=int)
    remap[idx] = np.arange(len(idx))
    out = {(min(remap[c], remap[parent[c]]), max(remap[c], remap[parent[c]]))
           for c in idx if parent[c] >= 0 and alive[parent[c]]}
    edges_out = (np.array(sorted(out), dtype=int) if out
                 else np.zeros((0, 2), dtype=int))
    return v[idx], r[idx], edges_out


def _decide(tn, p, v, r, parent, children, alive, radius_scale, distance_scale):
    """Which merge (if any) ``suboptimalDownsample`` would apply to ``tn``.

    Returns ``"parent"``, ``"child"``, ``"weighted"``, or ``None``.
    """
    d = float(np.linalg.norm(v[tn] - v[p]))
    if r[tn] + d <= r[p]:                       # isWithin(tn, parent)
        return "parent"
    if r[p] + d <= r[tn]:                       # isWithin(parent, tn)
        return "child"

    tn_cont = _is_continuation(tn, parent, children, alive)
    p_cont = _is_continuation(p, parent, children, alive)
    if tn_cont or p_cont:
        if d < r[tn] or d < r[p]:               # hasSignificantOverlap
            return "child" if not _live_children(tn, children, alive) else "weighted"

    if tn_cont and d < r[tn] + r[p]:            # isInterRedundant, master=parent
        kids = _live_children(tn, children, alive)
        if len(kids) != 1:
            return None
        c = kids[0]
        d1 = float(np.linalg.norm(v[tn] - v[p]))
        d2 = float(np.linalg.norm(v[tn] - v[c]))
        if d1 + d2 <= 0:
            return None
        lam = d2 / (d1 + d2)
        pos = v[p] * lam + v[c] * (1 - lam)
        rad = r[p] * lam + r[c] * (1 - lam)
        if float(np.linalg.norm(v[tn] - pos)) * distance_scale < rad:
            if r[tn] * radius_scale > rad and r[tn] < rad * radius_scale:
                return "parent"
    return None


def _live_children(tn, children, alive):
    return [c for c in children[tn] if alive[c]]


def _is_continuation(tn, parent, children, alive):
    """Degree 2: has a parent and exactly one child (``Swc_Tree_Node_Is_Continuation``)."""
    return parent[tn] >= 0 and len(_live_children(tn, children, alive)) == 1


def _root_forest(n, e, r):
    """Root each connected component at its largest-radius node; BFS for parents."""
    from collections import deque

    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in e:
        adj[a].append(b)
        adj[b].append(a)

    parent = np.full(n, -1, dtype=int)
    children: list[set] = [set() for _ in range(n)]
    seen = np.zeros(n, dtype=bool)
    roots = []
    for node in np.argsort(-np.asarray(r), kind="stable"):
        if seen[node]:
            continue
        roots.append(int(node))
        seen[node] = True
        q = deque([int(node)])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    parent[w] = u
                    children[u].add(w)
                    q.append(w)
    return parent, children, roots


def _postorder(roots, children):
    """Children before parents, iteratively (these trees get deep)."""
    order, stack = [], [(int(r_), False) for r_ in roots]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        stack.append((node, True))
        for c in children[node]:
            stack.append((int(c), False))
    return order


def simplify(vertices, radii, edges, *, preserve_junctions=PRESERVE_JUNCTIONS, **kw):
    """The full reduction: region-sample, then downsample.

    Branch *count* is not this module's problem — it is settled during tracing, by
    ``neutu_trace``'s un-invalidated-length test. A geometric post-hoc pruner used
    to live here and was removed; see docs/skeletonization-plan.md "Removed".
    """
    v, r, e = region_sample(vertices, radii, edges,
                            preserve_junctions=preserve_junctions)
    return optimal_downsample(v, r, e, **kw)
