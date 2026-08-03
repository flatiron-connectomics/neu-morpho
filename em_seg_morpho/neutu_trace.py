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

**Invalidation.** NeuTu uses ``EDT + 2`` voxels
(``maskExpansionRadius``, ``gui/zspgrowparser.cpp:296``), against kimimaro's
production ``1.5·DBF + 4.69``. **We default to ``EDT + 8``** — see
:data:`INVALIDATION_CONST`, which explains why that is compensation for a
different target-selection rule rather than a porting error.

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

# What we actually default to, and it is NOT NeuTu's value.
#
# This is **compensation for weaker target selection, not fidelity.** NeuTu picks
# the next branch by maximum un-invalidated length, so it extracts the largest new
# branch first and its invalidation ball consumes the territory that would
# otherwise resurface as many small branches. We pick by max-DAF
# (`CachedTargetFinder`), which can start on a short spur, invalidate little, and
# leave the same ground to be covered by many more paths. A larger ball buys back
# per path what NeuTu's ordering buys by choosing well.
#
# Measured against NeuTu over the 12-body benchmark (median ratios, port:NeuTu):
#
#              const=2      const=8
#   tips        4.07x        1.04x
#   cable         --         1.06x
#   nodes         --         0.75x
#   B->A p90     2.16         2.30      (what we fail to cover -- barely worse)
#
# TWO CAVEATS, because a tuned constant is a liability:
#   1. It is in VOXELS, so it is tied to `skeleton_scale`. At scale 2 (32 nm) this
#      is EDT + 256 nm; at scale 1 it would be EDT + 128 nm and mean something
#      different. **Re-sweep it if the scale changes.**
#   2. 256 nm is large next to a median process radius of ~55-72 nm, so it can
#      invalidate a genuinely separate neurite running parallel within that
#      distance. This shows up as rising B->A p90 on the two largest thick bodies
#      (6308993 3.81 -> 7.23, 45892915 3.74 -> 7.60) and is the strongest argument
#      for doing the target-selection rewrite instead.
#
# Pass `const=NEUTU_CONST` for the mechanically faithful behaviour.
INVALIDATION_CONST = 8.0
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
# So this defaults to OFF. It only becomes usable alongside target selection by
# un-invalidated length; see docs/skeletonization-plan.md.
PATIENCE = None


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
                const: float = INVALIDATION_CONST, min_length: float = MIN_LENGTH,
                patience: int | None = PATIENCE, dust_threshold: int = 0,
                connectivity: int = 26, max_paths=None):
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
                     patience=PATIENCE, max_paths=None, dbf=None):
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
    """
    valid = labels[path[:, 0], path[:, 1], path[:, 2]] != 0
    rev = valid[::-1]
    if not rev[0]:                       # endpoint already covered -> nothing new
        return 0.0
    n_tail = len(rev) if rev.all() else int(np.argmin(rev))
    if n_tail < 2:
        return 0.0
    tail = path[len(path) - n_tail:]
    return float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum())


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
