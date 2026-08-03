"""NeuTu-style TEASAR skeletonization.

NeuTu and kimimaro are both TEASAR, and the measured quality difference
(``docs/skeletonization-comparison.md``) comes from a few specific choices rather
than a different algorithm. This module implements the NeuTu variant directly.

**This is a reimplementation, not a copy of ``kimimaro.trace``.** The algorithm
proper lives in C++ — ``dijkstra3d`` for the distance and parent fields,
``kimimaro.skeletontricks`` for target selection and rolling-ball invalidation —
and those are *imported*, unchanged, so this runs at kimimaro's speed. What is
written out here is only the ~40 lines of orchestration around them. Importing
``kimimaro.trace.trace`` and swapping the cost was not an option: it computes the
cost inline with no hook, so injecting one means monkeypatching
``compute_pdrf`` globally, which changes behaviour for every other caller in the
process. Copying it was the alternative, but most of what would be copied is
machinery NeuTu does not have (see below) plus a standing obligation to track
upstream drift.

How this differs from kimimaro
------------------------------
**Path cost.** ``1/(1 + r²)`` — NeuTu's ``Stack_Voxel_Weight_I``
(``NeuTu/neurolabi/c/tz_stack_graph.c:205``), purely local. kimimaro uses
``pdrf_scale·(1 − DBF/dbf_max^1.01)^exponent``, normalized by the object's
*global* maximum radius, plus a small DAF term as a "trickle of gradient so open
spaces don't collapse". NeuTu has no such term and none is added here.

**Invalidation.** ``EDT + 2`` voxels, NeuTu's own value
(``maskExpansionRadius``, ``gui/zspgrowparser.cpp:296``), against kimimaro's
production ``1.5·DBF + 4.69``.

**No soma mode.** kimimaro detects somata and specially handles the root. NeuTu
has no equivalent, and Megaphragma is ~97% somaless.

**No ``fix_branching``.** kimimaro's default re-runs dijkstra per path
(``railroad``) to pull branch points toward the true divergence, at a large cost.
NeuTu's ``ZSpGrowParser`` walks a single parent field, which is what this does.

Radii are the inscribed ``DBF[vertex]``, as kimimaro reports them. Matching
NeuTu's radius convention is deliberately *not* done here — see step 4 of
``docs/skeletonization-plan.md``, and note that NeuTu's ``−0.5`` correction
(``AdjustedDistanceWeight``, ``gui/zstackskeletonizer.cpp:82``) is already
established as harmful at these radii.

THE UNIT TRAP
-------------
``1/(1 + r²)`` is **not scale-invariant** — the ``1`` is implicitly one voxel
squared. NeuTu computes it on a uint16 squared-distance map in *voxels*, where a
thin neurite (r≈1) and a thick one (r≈13) differ by a factor of ~80. Evaluate the
same expression on a DBF in *nanometres* at 32 nm/voxel and that body spans a
factor of ~157 instead, with every weight down around 1e-3 to 1e-6. Same formula,
different skeleton.

So everything here works in **voxels**, and :func:`skeletonize` returns voxel
coordinates and voxel radii; scale to nm afterwards. This costs nothing on
isotropic data, and NeuTu's own EDT is not anisotropy-aware anyway (anisotropy
enters only through dijkstra step lengths), so there is no faithful anisotropic
version to port.

Licence
-------
``kimimaro`` and ``dijkstra3d`` are GPL-3.0-or-later and are imported here, as
they already are elsewhere in this package — so em-seg-morpho is a GPL-combined
work and must be distributed under GPL-3.0-or-later. No kimimaro source is
copied.
"""

from __future__ import annotations

import numpy as np

# NeuTu's own invalidation, in voxels: ball radius EDT + 2 at each path voxel
# (`maskExpansionRadius = 2.0`, gui/zspgrowparser.cpp:296, applied via
# `DistanceWeight(v) = sqrt(v)` on the squared-distance map).
INVALIDATION_SCALE = 1.0
NEUTU_CONST = 2.0

# We default to NeuTu's own value. An earlier revision defaulted to 8.0 as
# "compensation for weaker target selection"; that was wrong, and the story is
# worth keeping because the evidence for it looked strong.
#
# Two bugs made NeuTu's const=2 look like it over-branched by 4x:
#   1. `_uninvalidated_length` measured uint32 paths with np.diff, which underflows
#      on any decreasing coordinate and returned ~2.7e11 -- so every
#      `>= min_length` test passed and branch rejection never happened at all.
#   2. The extraction loop had no progress guarantee, so it re-extracted one
#      identical path until it hit max_paths (see the comment at that site).
#
# With both fixed, const=2 reproduces NeuTu directly -- tip ratio 0.88-1.00, cable
# 1.05-1.14 -- and const=8 becomes far too aggressive (tip ratio 0.40-0.50,
# B->A p90 7.9-9.5, i.e. real cable deleted). The lesson: a parameter that
# "compensates" for a mechanism you have not verified is usually masking a bug.
INVALIDATION_CONST = NEUTU_CONST
# NeuTu's `minimalLength`, the branch-rejection threshold, in voxels.
MIN_LENGTH = 10.0
# Consecutive rejected branches before extraction stops. NeuTu stops on the
# FIRST one (`isPathAvailable = false`, gui/zspgrowparser.cpp:317) — it can,
# because `extractLongestPath` picks the target maximising un-invalidated
# length, so the best remaining branch being short means every remaining branch
# is short.
#
# We pick targets by max-DAF (`CachedTargetFinder`), which is a much weaker
# proxy, and **stopping on that basis truncates live arbor**. Measured against
# NeuTu, p90 distance from NeuTu's skeleton to ours — i.e. what we fail to
# cover — with patience 3 vs disabled:
#
#     body 35668783   2.40 -> 74.71 voxels
#     body 16104493   2.17 -> 17.03
#     body 45813451   3.11 -> 14.05
#
# So this defaults to OFF. The un-invalidated-length selector that would make
# it sound was implemented, measured not to help, and removed; the numbers and
# the method are in docs/skeletonization-plan.md and commit 83d1356.
PATIENCE = None

# How the path cost is applied. "voxel" uses dijkstra3d's per-voxel field, whose
# effective edge cost is d*f(destination). "edge" builds the graph explicitly and
# uses NeuTu's symmetric d*[f(u)+f(v)] via scipy. They agree only for uniform step
# lengths; measured inside real bulbs the per-voxel routes cost ~10% more under
# NeuTu's own cost and never matched (0/16). See docs/skeletonization-plan.md.
COST = "voxel"

# NeuTu's `skeletonRadius`, the cap on the skeleton-tube radius used for the path
# mask (gui/zspgrowparser.cpp:297). The mask radius is min(EDT + 1, this).
SKELETON_RADIUS = 3.0

# Truncate each new path where it meets the existing skeleton, instead of running
# it back to the root. This is what places branch points ON the centreline; see
# _truncate_at_skeleton. NeuTu always does it.
ATTACH_AT_SKELETON = True


def neutu_pdrf(DBF: np.ndarray) -> np.ndarray:
    """NeuTu's local path cost ``1/(1 + r²)``, on a **voxel-unit** DBF.

    Thick regions (large ``r``) get a small weight and are therefore cheap, which
    is what pulls the path onto the centreline.

    ``DBF`` is expected with background already set to ``inf``, and **background
    must come out as ``inf``, not 0**. The naive expression does the opposite:
    ``1/(1 + inf²)`` is 0, the *cheapest* weight there is, and ``dijkstra3d``
    takes the weight field alone with no separate mask — so paths route straight
    through empty space. kimimaro avoids this only incidentally, because its
    ``(1 − DBF·M)^exponent`` sends background to ``+inf``. Both
    ``test_neutu_pdrf_forbids_background`` and
    ``test_skeleton_stays_inside_the_tube`` cover it. NeuTu has no equivalent
    hazard: it never puts background voxels in the graph at all.

    See THE UNIT TRAP in the module docstring before passing anything in nm.
    """
    DBF = np.asarray(DBF)
    background = ~np.isfinite(DBF)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.empty(DBF.shape, dtype=np.float32, order="F")
        np.multiply(DBF, DBF, out=out)      # r²
        out += np.float32(1.0)
        np.reciprocal(out, out=out)         # 1/(1+r²)
    out[background] = np.inf                # forbidden, not free
    return np.asfortranarray(out)


def skeletonize(mask_zyx, *, scale: float = INVALIDATION_SCALE,
                const: float | None = None, min_length: float = MIN_LENGTH,
                patience: int | None = PATIENCE, cost: str = COST,
                attach_at_skeleton: bool = ATTACH_AT_SKELETON,
                dust_threshold: int = 0, connectivity: int = 26, max_paths=None):
    """Skeletonize a binary mask the way NeuTu does. Voxel units throughout.

    Returns an ``osteoid.Skeleton`` whose ``vertices`` are zyx voxel coordinates
    and whose ``radii`` are voxel radii, or an empty skeleton for an empty mask.

    **Every connected component is traced separately**, which is not an
    optimisation but a correctness requirement. TEASAR grows from one root, and
    both the parent field and the rolling-ball invalidation are confined to that
    root's component — so a single-root trace covers exactly one component and
    silently reports the rest as nothing. Bodies here are genuinely fragmented
    (``docs/skeletonization-comparison.md``), and the root comes from
    ``first_label``, i.e. whichever voxel happens to be first in memory order.
    Measured on body 6308993: 7 components, the largest holding 96.9% of the
    voxels, and ``first_label`` landing on one holding 3.06% — which is exactly
    the coverage a single-root trace achieved.

    ``dust_threshold`` drops components smaller than this many voxels. The
    default keeps everything, since a speck still yields a valid one-vertex
    skeleton and dropping cable is the pipeline's decision, not this function's.

    """
    import cc3d
    from osteoid import Skeleton

    if const is None:
        const = INVALIDATION_CONST
    labels = np.asfortranarray(np.asarray(mask_zyx).astype(np.uint8))
    if not labels.any():
        return Skeleton()

    cc, n_comp = cc3d.connected_components(labels, connectivity=connectivity,
                                           return_N=True)
    if n_comp == 0:
        return Skeleton()
    boxes = cc3d.statistics(cc)["bounding_boxes"]
    counts = np.bincount(cc.reshape(-1), minlength=n_comp + 1)

    pieces = []
    for label in range(1, n_comp + 1):
        if counts[label] < dust_threshold:
            continue
        box = boxes[label]
        # Pad by one voxel of guaranteed background: with the crop flush against
        # the component, black_border=False would read the out-of-bounds side as
        # non-background and inflate the EDT at the face.
        lo = [max(0, s.start - 1) for s in box]
        hi = [min(labels.shape[k], box[k].stop + 1) for k in range(3)]
        sub = np.asfortranarray(
            (cc[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == label).astype(np.uint8))
        skel = _trace_component(sub, scale=scale, const=const,
                                min_length=min_length, patience=patience,
                                cost=cost, attach_at_skeleton=attach_at_skeleton,
                                max_paths=max_paths)
        if len(np.asarray(skel.vertices)) == 0:
            continue
        skel.vertices = np.asarray(skel.vertices, dtype=np.float32) + np.array(
            lo, dtype=np.float32)
        pieces.append(skel)

    if not pieces:
        return Skeleton()
    return Skeleton.simple_merge(pieces).consolidate()


def _trace_component(labels, *, scale, const, min_length=MIN_LENGTH,
                     patience=PATIENCE, cost=COST,
                     attach_at_skeleton=ATTACH_AT_SKELETON, max_paths=None,
                     dbf=None):
    """TEASAR over a single connected component. See :func:`skeletonize`."""
    import dijkstra3d
    import edt
    import kimimaro.skeletontricks
    from osteoid import Skeleton

    if not labels.any():
        return Skeleton()
    if dbf is None:
        dbf = edt.edt(labels, anisotropy=(1, 1, 1), black_border=False, order="F")
    DBF = np.asfortranarray(dbf)

    root = _find_root(labels, DBF)
    if root is None:
        return Skeleton()

    # DBF is reused as the invalidation radius field, where 0 would mean "never
    # invalidates"; inf also marks background as untraversable for the cost.
    DBF = kimimaro.skeletontricks.zero2inf(DBF)

    # Distance-from-root field, used only to choose targets (farthest first).
    # NeuTu's cost has no DAF term, unlike kimimaro's.
    DAF, farthest = dijkstra3d.euclidean_distance_field(
        labels, root, anisotropy=(1, 1, 1), return_max_location=True)
    DAF = kimimaro.skeletontricks.inf2zero(DAF)
    target_finder = kimimaro.skeletontricks.CachedTargetFinder(labels, DAF)
    del DAF

    w = neutu_pdrf(DBF)
    if cost == "edge":
        pred, inside_idx, pos_map, src_c = _edge_weighted_parents(labels, w, root)

        def path_to(t):
            t = tuple(int(v) for v in t)
            tf = int(pos_map[t[0] + labels.shape[0]
                             * (t[1] + labels.shape[1] * t[2])])
            if tf < 0:
                return np.zeros((0, 3), dtype=np.uint32)
            # The source's predecessor is -9999, identical to "unreachable". Reading
            # it as failure returned an EMPTY path, which invalidates nothing, so
            # `remaining` never fell and the loop spun toward max_paths -- two hours
            # of CPU on one body before this was caught.
            if tf == src_c:
                return np.array([t], dtype=np.uint32)
            if pred[tf] < 0:
                return np.zeros((0, 3), dtype=np.uint32)
            return _path_from_predecessors(pred, inside_idx, labels.shape, tf)
    else:
        parents = dijkstra3d.parental_field(w, root)

        def path_to(t):
            return dijkstra3d.path_from_parents(parents, tuple(int(v) for v in t))

    path_mask = (np.zeros(labels.shape, dtype=bool)
                 if attach_at_skeleton else None)
    accepted_verts = np.zeros((0, 3), dtype=np.uint32)
    paths = []
    remaining = int(np.count_nonzero(labels))
    if max_paths is None:
        max_paths = remaining
    pending = [farthest]                    # the extremal point is always a target
    misses = 0
    while (remaining > 0 or pending) and len(paths) < max_paths:
        target = pending.pop() if pending else target_finder.find_target(labels)
        path = path_to(target)
        if not len(path):
            # No path to this target: retire the voxel anyway, or find_target keeps
            # handing it back and the loop makes no progress.
            labels[tuple(int(v) for v in target)] = 0
            remaining -= 1
            misses += 1
            if patience is not None and misses >= patience:
                break
            continue
        keep = True
        if min_length > 0:
            keep = _uninvalidated_length(path, labels) >= min_length
        if path_mask is not None:
            path = _truncate_at_skeleton(path, path_mask, accepted_verts)
        if remaining > 0:
            # Invalidate even for a rejected branch: the territory is covered
            # either way, and skipping it would spin on the same target.
            invalidated, labels = \
                kimimaro.skeletontricks.roll_invalidation_ball_inside_component(
                    labels, DBF, scale, const, (1, 1, 1), path)
            # THE PROGRESS GUARANTEE, and it is load-bearing.
            # roll_invalidation_ball_inside_component erases voxels *around* the
            # path but never the path's own voxels (verified: 307 of 307 left
            # valid), and CachedTargetFinder does not remember what it has already
            # returned. So the farthest valid voxel stays valid, find_target hands
            # back the same target forever, and the loop re-extracts one identical
            # path until it hits max_paths. kimimaro escapes this only because its
            # default fix_branching=True rewrites `parents` to 0 along each path,
            # which reroutes the next railroad() call; this port uses
            # parental_field, so it has no such escape and must retire the path
            # explicitly.
            zs, ys, xs = path[:, 0], path[:, 1], path[:, 2]
            remaining -= int(np.count_nonzero(labels[zs, ys, xs]))
            labels[zs, ys, xs] = 0
            remaining -= invalidated
        if keep and len(path):
            paths.append(path)
            if path_mask is not None:
                _stamp_path_mask(path_mask, path, DBF)
                accepted_verts = np.concatenate([accepted_verts, path], axis=0)
            misses = 0
        else:
            misses += 1
            if patience is not None and misses >= patience:
                break                       # NeuTu's global stop; see the docstring

    if not paths:
        return Skeleton()
    skel = Skeleton.simple_merge([Skeleton.from_path(p) for p in paths]).consolidate()
    v = skel.vertices.flatten().astype(np.uint32)
    skel.radii = DBF[v[::3], v[1::3], v[2::3]]
    return skel


def _uninvalidated_length(path, labels):
    """Geodesic length of the path's tail that is still un-invalidated.

    NeuTu's branch test (`ZSpGrowParser::pathLength(idx, masked=true)`, walking
    back from the endpoint to the first already-covered voxel and subtracting).
    **This, not node spacing, is what makes NeuTu's skeletons small.** Measured on
    the benchmark: after the reduction passes the port already places *fewer*
    nodes per unit cable than NeuTu, but traced 10.5× more cable — 5,022 tips
    against NeuTu's 116 on body 6308993.

    The test rejects on *new territory covered*, not on geometric length, and
    that distinction is the point: a short branch reaching into unclaimed volume
    survives, a long one shadowing ground already covered does not. Pruning
    twigs by length afterwards is strictly worse — swept on real bodies, it hits
    NeuTu's node count only at 45–65% fill where NeuTu holds 64–81%.

    ``path`` runs root-end first, so the un-invalidated stretch is the **tail**:
    take the trailing run of still-valid voxels and measure it.

    **The float cast is load-bearing.** ``dijkstra3d`` returns paths as ``uint32``,
    so ``np.diff`` on the raw array underflows at every step where a coordinate
    decreases and yields ~2**32 — making this return ~2.7e11 and every
    ``>= min_length`` test pass. That silently disabled branch rejection.
    """
    path = np.asarray(path)
    valid = labels[path[:, 0], path[:, 1], path[:, 2]] != 0
    rev = valid[::-1]
    if not rev[0]:                       # endpoint already covered -> nothing new
        return 0.0
    n_tail = len(rev) if rev.all() else int(np.argmin(rev))
    if n_tail < 2:
        return 0.0
    tail = path[len(path) - n_tail:].astype(np.float64)
    return float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum())


def _stamp_path_mask(path_mask, path, DBF, cap=SKELETON_RADIUS):
    """Mark an accepted path into the skeleton tube, radius ``min(EDT+1, cap)``.

    NeuTu's ``m_pathMask``: ``path.sample(ballStack, DistanceWeight)`` then
    ``addValue(1.0)`` then ``minimizeValue(skeletonRadius=3)``
    (``gui/zspgrowparser.cpp:333-341``).
    """
    r = np.minimum(DBF[path[:, 0], path[:, 1], path[:, 2]] + 1.0, cap)
    shape = path_mask.shape
    for k in np.unique(np.maximum(0, np.round(r).astype(int))):
        sel = np.round(r).astype(int) == k
        g = np.arange(-k, k + 1)
        dz, dy, dx = np.meshgrid(g, g, g, indexing="ij")
        ball = (dz ** 2 + dy ** 2 + dx ** 2) <= k * k
        oz, oy, ox = dz[ball], dy[ball], dx[ball]
        zz = (path[sel, 0][:, None] + oz[None, :]).ravel()
        yy = (path[sel, 1][:, None] + oy[None, :]).ravel()
        xx = (path[sel, 2][:, None] + ox[None, :]).ravel()
        ok = ((zz >= 0) & (zz < shape[0]) & (yy >= 0) & (yy < shape[1])
              & (xx >= 0) & (xx < shape[2]))
        path_mask[zz[ok], yy[ok], xx[ok]] = True


def _truncate_at_skeleton(path, path_mask, attach_to=None):
    """Cut a path where it first meets the existing skeleton, walking from the tip.

    **This is what puts branch points on the centreline.** NeuTu's
    ``extractPath`` walks from the endpoint toward the root and breaks at the first
    ``m_pathMask`` voxel, so a new branch stops where it meets what is already
    there rather than running back to the root
    (``gui/zspgrowparser.cpp:51-62``).

    Extracting every path to the root instead — which is what this port did, and
    what the module docstring wrongly described as NeuTu's behaviour — makes branch
    points fall wherever the *dijkstra tree* diverges. In a bulb that is nowhere
    near the centre, which is the visible difference from NeuTu: NeuTu's branches
    come off the main trunk near the middle of the bulb, ours came off wherever the
    parent field happened to split.

    ``path`` is root-first, so "walking from the tip" means taking the **last**
    masked index and keeping the suffix from it.

    **Truncation alone disconnects the skeleton**, which is why ``attach_to`` is
    not optional. The mask is a *tube* of radius up to 3, so the cut voxel is
    generally inside the tube without being a vertex of the existing skeleton —
    nothing is shared, ``consolidate`` merges nothing, and the branch comes back as
    a separate component. (``test_branches_are_traced`` caught exactly this: a Y
    with no degree-3 vertex.) NeuTu gets connectivity from ``wholeTree->merge``;
    here the nearest existing vertex is prepended, so the two paths share it and
    the branch point lands on the centreline.
    """
    if not len(path):
        return path
    hits = path_mask[path[:, 0], path[:, 1], path[:, 2]]
    if not hits.any():
        return path
    cut = path[int(np.flatnonzero(hits)[-1]):]
    if attach_to is None or not len(attach_to):
        return cut
    from scipy.spatial import cKDTree

    _, j = cKDTree(np.asarray(attach_to, dtype=float)).query(
        np.asarray(cut[0], dtype=float))
    anchor = np.asarray(attach_to[int(j)], dtype=cut.dtype).reshape(1, 3)
    if np.array_equal(anchor[0], cut[0]):
        return cut
    return np.concatenate([anchor, cut], axis=0)


def _edge_weighted_parents(labels, weight, root):
    """Parent field under NeuTu's **symmetric edge** cost ``d·[f(u) + f(v)]``.

    ``dijkstra3d`` cannot express this — every one of its entry points takes a
    per-voxel field, so the edge cost it minimises is effectively
    ``Σ dᵢ·f(vᵢ₊₁)``. Those two agree only when every step has the same length:
    ``Σ dᵢ(fᵢ + fᵢ₊₁)`` telescopes to ``2·Σ dᵢfᵢ₊₁`` plus fixed endpoint terms
    exactly when ``dᵢ`` is constant. A tube satisfies that; a bulb, with 1, √2 and
    √3 interleaved, does not. **Measured inside real bulbs: 0 of 16 paths matched,
    and our per-voxel routes cost ~10% more under NeuTu's own cost, straying up to
    12 voxels.**

    So this builds the 26-connected graph over the component explicitly and hands it
    to ``scipy.sparse.csgraph.dijkstra``, which does take edge weights. Roughly 26M
    edges and ~320 MB of CSR for a 10⁶-voxel component — affordable, C-implemented,
    and no new build dependency.

    Returns ``(predecessor, inside, pos, src)`` in the component's compact index
    space; use :func:`_path_from_predecessors` to walk it. **``src`` must be
    returned and checked separately**: scipy marks the source with ``-9999``, the
    same sentinel it uses for unreachable nodes, so a source-is-target query is
    indistinguishable from a failure unless you know which index the source is.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    shape = labels.shape
    flat = labels.reshape(-1, order="F")
    inside = np.flatnonzero(flat != 0)
    pos = np.full(flat.size, -1, dtype=np.int64)
    pos[inside] = np.arange(len(inside))
    f = np.asarray(weight).reshape(-1, order="F")[inside].astype(np.float64)
    cz, cy, cx = _unflatten_np(inside, shape)

    # 13 of the 26 offsets; the transpose supplies the other half
    offsets = [(dz, dy, dx) for dz in (0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) > (0, 0, 0)]
    us, vs, ws = [], [], []
    for dz, dy, dx in offsets:
        nz, ny, nx = cz + dz, cy + dy, cx + dx
        ok = ((nz < shape[0]) & (ny >= 0) & (ny < shape[1])
              & (nx >= 0) & (nx < shape[2]))
        if not ok.any():
            continue
        nbr = pos[nz[ok] + shape[0] * (ny[ok] + shape[1] * nx[ok])]
        good = nbr >= 0
        if not good.any():
            continue
        u = np.flatnonzero(ok)[good]
        v = nbr[good]
        d = float(np.sqrt(dz * dz + dy * dy + dx * dx))
        us.append(u)
        vs.append(v)
        ws.append(d * (f[u] + f[v]))

    n = len(inside)
    if not us:
        return np.full(n, -9999, dtype=np.int64), inside, pos, -1
    u = np.concatenate(us)
    v = np.concatenate(vs)
    w = np.concatenate(ws)
    g = coo_matrix((w, (u, v)), shape=(n, n)).tocsr()
    g = g + g.T                                  # undirected
    src = int(pos[root[0] + shape[0] * (root[1] + shape[1] * root[2])])
    # directed=True: the graph is already symmetric from g + g.T, and letting scipy
    # symmetrise again is 2x slower for the same answer (0.05s vs 0.11s measured).
    _, pred = dijkstra(g, directed=True, indices=src, return_predecessors=True)
    return pred.astype(np.int64), inside, pos, src


def _path_from_predecessors(pred, inside, shape, target_flat):
    """Walk a scipy predecessor array back to the source; returns zyx uint32."""
    cur = int(target_flat)
    out = [cur]
    while True:
        nxt = int(pred[cur])
        if nxt < 0:
            break
        out.append(nxt)
        cur = nxt
        if len(out) > len(inside):               # cycle guard
            break
    z, y, x = _unflatten_np(inside[np.array(out[::-1])], shape)
    return np.stack([z, y, x], axis=1).astype(np.uint32)


def _unflatten_np(flat, shape):
    """Fortran-order flat index -> (z, y, x), vectorised."""
    nz, ny, _ = shape
    z = flat % nz
    rest = flat // nz
    return z, rest % ny, rest // ny


def _find_root(labels, dbf=None):
    """TEASAR root: the point farthest (geodesically) from the **thickest** voxel.

    NeuTu's two-pass root selection (``m_rebase``, on in our config): seed at
    ``Stack_Max(tmpdist)`` — the largest distance-transform value — grow, take the
    longest path, and re-seed at its far end
    (``gui/zstackskeletonizer.cpp:341-359``). The structure here already matched;
    only the seed did not. We used ``first_label``, i.e. whichever voxel came first
    in memory order.

    Why it matters: the root determines the entire parent tree, so it determines
    every path and every branch point. An arbitrary seed makes the whole skeleton
    depend on array layout, which is also why it was worth fixing regardless of
    whether it closes the gap to NeuTu — it removes a dependence on something
    meaningless.

    Pass the **raw** EDT, before ``zero2inf``: afterwards background is ``inf`` and
    the argmax lands outside the object.
    """
    import dijkstra3d
    import kimimaro.skeletontricks

    if dbf is not None and np.isfinite(dbf).any():
        seed = np.unravel_index(int(np.argmax(dbf)), dbf.shape)
        seed = tuple(int(v) for v in seed)
        if not labels[seed]:                    # degenerate: fall back
            seed = kimimaro.skeletontricks.first_label(labels)
    else:
        seed = kimimaro.skeletontricks.first_label(labels)
    if seed is None:
        return None
    _, farthest = dijkstra3d.euclidean_distance_field(
        labels, seed, anisotropy=(1, 1, 1), return_max_location=True)
    return farthest
