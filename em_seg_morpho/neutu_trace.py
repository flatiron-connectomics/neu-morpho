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
# So this defaults to OFF. `select="uninvalidated"` has its own, sound, stop.
PATIENCE = None

# Target selection. "daf" takes the farthest still-valid voxel from the root
# (kimimaro's CachedTargetFinder). "uninvalidated" is NeuTu's real rule — the
# boundary voxel whose not-yet-invalidated tail is longest, stopping once the best
# remaining is shorter than min_length.
#
# The default is "daf" because measured over the 12-body benchmark neither
# dominates and "daf" is closer on counts: tips 1.01x vs 1.22x, cable 1.07x vs
# 1.13x, A->B 0.83 vs 0.93, B->A p90 2.44 vs 2.41. Ten of twelve bodies are
# near-identical. On the two bulb-heavy bodies "uninvalidated" recovers much more
# of NeuTu's branch structure (p90 7.5 -> 4.2) while adding ~2.7x the tips; those
# bulbs are segmentation noise, so which is "right" is not decidable from the
# data. See docs/skeletonization-plan.md.
SELECT = "daf"


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
                patience: int | None = PATIENCE, select: str = SELECT,
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
                                select=select, max_paths=max_paths)
        if len(np.asarray(skel.vertices)) == 0:
            continue
        skel.vertices = np.asarray(skel.vertices, dtype=np.float32) + np.array(
            lo, dtype=np.float32)
        pieces.append(skel)

    if not pieces:
        return Skeleton()
    return Skeleton.simple_merge(pieces).consolidate()


def _trace_component(labels, *, scale, const, min_length=MIN_LENGTH,
                     patience=PATIENCE, select=SELECT, max_paths=None, dbf=None):
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

    root = _find_root(labels)
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

    parents = dijkstra3d.parental_field(neutu_pdrf(DBF), root)

    if select == "uninvalidated":
        return _trace_by_uninvalidated_length(
            labels, DBF, parents, scale, const, min_length, max_paths)

    paths = []
    remaining = int(np.count_nonzero(labels))
    if max_paths is None:
        max_paths = remaining
    pending = [farthest]                    # the extremal point is always a target
    misses = 0
    while (remaining > 0 or pending) and len(paths) < max_paths:
        target = pending.pop() if pending else target_finder.find_target(labels)
        path = dijkstra3d.path_from_parents(parents, target)
        keep = len(path) > 0
        if keep and min_length > 0:
            keep = _uninvalidated_length(path, labels) >= min_length
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
        if keep:
            paths.append(path)
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


def _trace_by_uninvalidated_length(labels, DBF, parents, scale, const, min_length,
                                   max_paths):
    """NeuTu's real target rule: extract the branch covering the most new ground.

    ``extractLongestPath`` (``gui/zspgrowparser.cpp:214``) picks the endpoint
    maximising ``pathLength(idx, masked=True)`` — the geodesic length of the tail
    that is not yet invalidated — and extraction **stops** once the best remaining
    branch is shorter than ``minimalLength`` (``isPathAvailable = false``, line
    317). The stop is sound here in a way it is not under max-DAF selection: if the
    *best* remaining branch is too short, every remaining branch is.

    The argmax is over the whole valid set each round, which is why the cheap
    substitutes were tried first. It is affordable via pointer doubling: the
    per-voxel geodesic length from the root is computed once, and the nearest
    invalidated ancestor is found with ``log2(depth)`` vectorised gathers.
    """
    import dijkstra3d
    import kimimaro.skeletontricks
    from osteoid import Skeleton

    from scipy import ndimage

    shape = labels.shape
    flat = labels.reshape(-1, order="F")

    # Work in the component's own index space, not the crop's. The parent chains
    # never leave the component, and a crop is mostly background: on body 16104493
    # that is 27,750 voxels instead of 2.6 million, a ~94x cut in the per-round
    # doubling cost.
    inside = np.flatnonzero(flat != 0)
    if not len(inside):
        return Skeleton()
    pos = np.full(flat.size, -1, dtype=np.int64)
    pos[inside] = np.arange(len(inside))
    parent_full = decode_parents(parents)
    parent = np.where(parent_full[inside] >= 0, pos[parent_full[inside]], -1)
    parent[parent < 0] = -1
    glen = _chain_length_from_root_compact(parent, inside, shape)

    # NeuTu scores only BOUNDARY voxels as endpoints (`m_fgArray`, voxels adjacent
    # to a barrier), computed once from the original object. Scoring interior
    # voxels too invents branches it would never pick: a voxel deep inside a bulb
    # can have a long un-invalidated tail without being a tip of anything.
    obj = labels != 0
    interior = ndimage.binary_erosion(obj, ndimage.generate_binary_structure(3, 1))
    is_boundary = (obj & ~interior).reshape(-1, order="F")[inside]

    paths = []
    if max_paths is None:
        max_paths = len(inside)

    while len(paths) < max_paths:
        valid = flat[inside] != 0
        if not (valid & is_boundary).any():
            break
        covered = _first_invalid_ancestor_length(parent, glen, valid)
        score = np.where(valid & is_boundary, glen - covered, -np.inf)
        best = int(np.argmax(score))
        if not np.isfinite(score[best]) or score[best] < min_length:
            break                            # NeuTu's global stop
        z, y, x = (int(v) for v in _unflatten(np.int64(inside[best]), shape))
        path = dijkstra3d.path_from_parents(parents, (z, y, x))
        if not len(path):
            break
        _, labels = kimimaro.skeletontricks.roll_invalidation_ball_inside_component(
            labels, DBF, scale, const, (1, 1, 1), path)
        labels[path[:, 0], path[:, 1], path[:, 2]] = 0      # progress guarantee
        paths.append(path)

    if not paths:
        return Skeleton()
    skel = Skeleton.simple_merge(
        [Skeleton.from_path(p) for p in paths]).consolidate()
    v = skel.vertices.flatten().astype(np.uint32)
    skel.radii = DBF[v[::3], v[1::3], v[2::3]]
    return skel


def decode_parents(parental_field):
    """``dijkstra3d.parental_field`` -> flat parent index per voxel, -1 at the root.

    The encoding is a **1-based Fortran-order flat index**; 0 means "no parent".
    Not documented, so ``test_decode_parents_gives_adjacent_parents`` asserts every
    decoded parent is 26-adjacent to its child — if dijkstra3d ever changes the
    encoding, that test fails rather than the selection silently going haywire.
    """
    flat = np.asarray(parental_field).reshape(-1, order="F").astype(np.int64) - 1
    return flat                                     # -1 where the field held 0


def _chain_length_from_root_compact(parent, inside, shape):
    """Geometric path length from the root, in the component's index space.

    NeuTu's ``m_workspace->length[]``. Pointer doubling: chains run thousands of
    voxels deep, so a Python walk per voxel is not viable, while doubling costs
    ``log2(depth)`` vectorised gathers.
    """
    idx = np.arange(parent.size, dtype=np.int64)
    jump = np.where(parent >= 0, parent, idx)
    cz, cy, cx = _unflatten(inside, shape)
    pz, py, px = _unflatten(inside[jump], shape)
    acc = np.sqrt(((pz - cz) ** 2 + (py - cy) ** 2
                   + (px - cx) ** 2).astype(np.float64))
    acc[parent < 0] = 0.0
    for _ in range(64):
        nxt = jump[jump]
        acc = acc + acc[jump]
        if np.array_equal(nxt, jump):
            break
        jump = nxt
    return acc


def _unflatten(flat, shape):
    """Fortran-order flat index -> (z, y, x), vectorised."""
    nz, ny, _ = shape
    z = flat % nz
    rest = flat // nz
    return z, rest % ny, rest // ny


def _first_invalid_ancestor_length(parent, glen, valid):
    """``glen`` at each voxel's nearest invalidated ancestor, by pointer doubling.

    Uses the absorbing map ``f(v) = parent(v) if valid(v) else v``: once a chain
    reaches an invalidated voxel it stops moving, so iterating ``f`` to a fixpoint
    lands every voxel on its first invalidated ancestor. ``log2(depth)`` gathers
    instead of a walk per voxel.
    """
    idx = np.arange(parent.size, dtype=np.int64)
    step = np.where(valid & (parent >= 0), parent, idx)     # absorb at invalid/root
    for _ in range(64):
        nxt = step[step]
        if np.array_equal(nxt, step):
            break
        step = nxt
    anc = step
    # a chain that never met an invalidated voxel has covered nothing: length 0
    out = np.where(valid[anc], 0.0, glen[anc])
    return out


def _find_root(labels):
    """TEASAR root: the point farthest (geodesically) from an arbitrary voxel."""
    import dijkstra3d
    import kimimaro.skeletontricks

    seed = kimimaro.skeletontricks.first_label(labels)
    if seed is None:
        return None
    _, farthest = dijkstra3d.euclidean_distance_field(
        labels, seed, anisotropy=(1, 1, 1), return_max_location=True)
    return farthest
