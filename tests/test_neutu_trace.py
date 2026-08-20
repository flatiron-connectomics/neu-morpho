"""Tests for the NeuTu-style tracer.

The load-bearing one is :func:`test_per_voxel_weights_match_neutu_edge_cost` —
the substitution the plan flags as assumed-but-unverified.
``dijkstra3d.parental_field`` takes **per-voxel** weights, whereas NeuTu's cost is
the symmetric **edge** form ``d·[f(v₁) + f(v₂)]``. For a path with uniform step
length these telescope to the same argmin up to constant endpoint terms, but 3D
diagonal moves have mixed step lengths (1, √2, √3), so the equivalence is not
free and is checked against an independent reference implementation.
"""

from __future__ import annotations

import heapq

import numpy as np
import pytest

from neu_morpho import neutu_trace


def _tube(length=40, r=3, pad=4, bend=False):
    """A cylinder along z, optionally bent, as a Fortran-order uint8 mask."""
    s = (length + 2 * pad, 2 * (r + pad) + (length // 2 if bend else 0), 2 * (r + pad))
    zz, yy, xx = np.indices(s)
    cy, cx = r + pad, r + pad
    off = np.clip(zz - pad, 0, None) // 2 if bend else 0
    m = ((zz >= pad) & (zz < pad + length) &
         ((yy - cy - off) ** 2 + (xx - cx) ** 2 <= r * r))
    return np.asfortranarray(m.astype(np.uint8))


def _weights(mask):
    import edt
    import kimimaro.skeletontricks

    dbf = edt.edt(mask, anisotropy=(1, 1, 1), black_border=False, order="F")
    return neutu_trace.neutu_pdrf(kimimaro.skeletontricks.zero2inf(dbf))


def test_neutu_pdrf_values():
    dbf = np.asfortranarray(np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32))
    p = neutu_trace.neutu_pdrf(dbf)
    assert p[0, 0, 0] == pytest.approx(0.5)          # 1/(1+1)
    assert p[0, 0, 1] == pytest.approx(0.2)          # 1/(1+4)
    assert p[0, 0, 2] == pytest.approx(0.1)          # 1/(1+9)
    assert p.dtype == np.float32


def test_neutu_pdrf_forbids_background():
    """Background must be inf. 1/(1+inf²) is 0 — the *cheapest* weight there is,
    which would make empty space free to cross."""
    dbf = np.asfortranarray(np.array([[[1.0, np.inf]]], dtype=np.float32))
    p = neutu_trace.neutu_pdrf(dbf)
    assert np.isinf(p[0, 0, 1])
    assert p[0, 0, 0] < p[0, 0, 1]


def _neutu_edge_cost_path(weights, src, dst):
    """Reference Dijkstra using NeuTu's symmetric edge cost ``d·[f(u)+f(v)]``.

    Deliberately plain, slow and obviously correct — it exists to check the fast
    per-voxel path, so sharing any code with it would defeat the purpose.
    26-connected, matching dijkstra3d.
    """
    shape = weights.shape
    offs = [(dz, dy, dx)
            for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)]
    step = {o: float(np.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)) for o in offs}

    best = {src: 0.0}
    prev: dict = {src: None}
    pq = [(0.0, src)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        fu = weights[u]
        for o in offs:
            v = (u[0] + o[0], u[1] + o[1], u[2] + o[2])
            if not all(0 <= v[k] < shape[k] for k in range(3)):
                continue
            fv = weights[v]
            if not np.isfinite(fv):             # background is forbidden
                continue
            nd = d + step[o] * (fu + fv)
            if nd < best.get(v, float("inf")) - 1e-12:
                best[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in prev:
        return None
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return np.array(path[::-1], dtype=int)


@pytest.mark.parametrize("bend", [False, True])
def test_per_voxel_weights_match_neutu_edge_cost(bend):
    """The substitution the plan says to confirm rather than assume.

    Not asserting identical paths a priori: with mixed 3D step lengths the two
    costs are not exactly proportional, so ties could break differently. What
    must hold is that the per-voxel route is no worse *under NeuTu's own cost*
    than NeuTu's optimum. (In practice both tubes come out bit-identical.)
    """
    import dijkstra3d

    m = _tube(length=30, r=3, bend=bend)
    w = _weights(m)
    inside = np.argwhere(m > 0)
    src = tuple(int(v) for v in inside[0])
    dst = tuple(int(v) for v in inside[-1])

    ours = np.asarray(dijkstra3d.path_from_parents(dijkstra3d.parental_field(w, src), dst))
    ref = _neutu_edge_cost_path(w, src, dst)
    assert ref is not None and len(ours) > 0

    def neutu_cost(path):
        d = np.linalg.norm(np.diff(path, axis=0), axis=1)
        f = w[path[:, 0], path[:, 1], path[:, 2]]
        return float(np.sum(d * (f[:-1] + f[1:])))

    c_ours, c_ref = neutu_cost(ours), neutu_cost(ref)
    assert c_ref <= c_ours + 1e-9                      # ref is the true optimum
    assert c_ours <= c_ref * 1.02, (
        f"per-voxel weighting chose a materially worse path under NeuTu's own "
        f"cost: {c_ours:.6g} vs optimum {c_ref:.6g}")


def test_skeleton_stays_inside_the_tube():
    m = _tube(length=40, r=3)
    skel = neutu_trace.skeletonize(m)
    v = np.asarray(skel.vertices).astype(int)
    assert len(v) > 0
    assert v[:, 0].max() - v[:, 0].min() >= 35              # spans the tube
    assert m[v[:, 0], v[:, 1], v[:, 2]].all()               # never leaves it


def test_radii_are_inscribed_not_inflated():
    m = _tube(length=40, r=3)
    skel = neutu_trace.skeletonize(m)
    assert np.asarray(skel.radii).max() <= 3.5 + 1e-6


def test_empty_mask_returns_empty_skeleton():
    skel = neutu_trace.skeletonize(np.zeros((8, 8, 8), dtype=np.uint8))
    assert len(np.asarray(skel.vertices)) == 0


def test_default_invalidation_is_neutus_own_value():
    """The default must stay at NeuTu's EDT+2, not a compensating constant.

    A previous revision defaulted to 8.0 to "compensate for weaker target
    selection". That was masking two bugs (the uint32 length underflow and the
    missing loop progress guarantee); with those fixed, 2.0 reproduces NeuTu and
    8.0 deletes real cable. If this assertion ever fails, check for a
    reintroduced bug before accepting the new constant.
    """
    assert neutu_trace.NEUTU_CONST == 2.0
    assert neutu_trace.INVALIDATION_CONST == neutu_trace.NEUTU_CONST


def test_tighter_invalidation_yields_more_vertices():
    """A smaller invalidation ball leaves more to cover, so more paths."""
    m = _tube(length=40, r=3, bend=True)
    tight = neutu_trace.skeletonize(m, scale=1.0, const=2.0)
    looser = neutu_trace.skeletonize(m, scale=2.0, const=8.0)
    assert len(np.asarray(tight.vertices)) >= len(np.asarray(looser.vertices))


def test_all_connected_components_are_traced():
    """A single-root trace covers exactly one component and silently drops the rest.

    TEASAR grows from one root, and both the parent field and the rolling-ball
    invalidation are confined to that root's component. The root comes from
    ``first_label`` — whichever voxel is first in memory order — so on a
    fragmented body the trace can land on a speck. This reproduces that in
    miniature: a small tube placed *first* in memory order, a large one after.
    Measured on real body 6308993 (7 components, largest 96.9% of voxels) a
    single-root trace covered 3%.
    """
    m = np.zeros((60, 40, 20), dtype=np.uint8)
    m[2:8, 3:9, 3:9] = 1                                    # small, first in z
    zz, yy, xx = np.indices(m.shape)
    m[((zz >= 20) & (zz < 55) & ((yy - 25) ** 2 + (xx - 10) ** 2 <= 16))] = 1
    m = np.asfortranarray(m)

    # min_length=0: this test is about component iteration, not branch rejection.
    # At the default threshold the 6-voxel cube is legitimately dropped for being
    # shorter than minimalLength, which would mask the bug this test guards.
    skel = neutu_trace.skeletonize(m, min_length=0.0)
    v = np.asarray(skel.vertices).astype(int)
    assert len(v) > 0
    in_small = ((v[:, 0] < 15)).sum()
    in_large = ((v[:, 0] >= 15)).sum()
    assert in_small > 0, "the small leading component was not traced"
    assert in_large > 0, "the large component was not traced (single-root bug)"


def test_component_crop_does_not_inflate_radii():
    """Cropping to a component's bbox must pad with background.

    With the crop flush against the component, ``black_border=False`` reads the
    out-of-bounds side as non-background and the EDT at that face is too large.
    """
    m = np.zeros((30, 20, 20), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[((zz >= 0) & (zz < 30) & ((yy - 10) ** 2 + (xx - 10) ** 2 <= 9))] = 1
    skel = neutu_trace.skeletonize(np.asfortranarray(m))
    assert np.asarray(skel.radii).max() <= 3.5 + 1e-6


def test_min_length_rejects_short_spurs_but_keeps_the_trunk():
    """NeuTu's `minimalLength`: reject a branch by the NEW territory it covers.

    A long tube with a 2-voxel bump. The bump is a branch whose un-invalidated
    geodesic tail is ~nothing, so it must not become cable; the trunk must
    survive untouched.
    """
    m = _tube(length=50, r=3, pad=5)
    m = np.asfortranarray(m.copy())
    m[28:30, 12:14, 4] = 1                                  # a tiny nub

    strict = neutu_trace.skeletonize(m, min_length=10.0)
    loose = neutu_trace.skeletonize(m, min_length=0.0)
    assert len(np.asarray(strict.vertices)) <= len(np.asarray(loose.vertices))
    v = np.asarray(strict.vertices)
    assert v[:, 0].max() - v[:, 0].min() >= 40, "the trunk was truncated too"


def test_min_length_is_measured_on_uninvalidated_length_not_geometry():
    """The distinction that makes the criterion work.

    ``_uninvalidated_length`` must measure only the trailing run of still-valid
    voxels, so a long path that mostly retreads covered ground scores low.
    """
    path = np.array([[i, 0, 0] for i in range(10)])
    labels = np.ones((12, 2, 2), dtype=np.uint8)
    assert neutu_trace._uninvalidated_length(path, labels) == pytest.approx(9.0)

    labels_covered = labels.copy()
    labels_covered[:8] = 0                      # only the last 2 are new
    assert neutu_trace._uninvalidated_length(path, labels_covered) == pytest.approx(1.0)

    labels_all = np.zeros((12, 2, 2), dtype=np.uint8)
    assert neutu_trace._uninvalidated_length(path, labels_all) == 0.0


def test_uninvalidated_length_survives_uint32_paths():
    """dijkstra3d returns uint32 paths; np.diff on them underflows.

    Any step where a coordinate DECREASES wraps to ~2**32, so the length came
    back as ~2.7e11 and every `>= min_length` test passed — branch rejection was
    silently disabled. The earlier version of this test used an int64 path and a
    monotonically increasing one, so it missed both halves of the bug.
    """
    labels = np.ones((12, 12, 12), dtype=np.uint8)
    down = np.array([[9, 5, 5], [8, 5, 5], [7, 5, 5], [6, 5, 5]], dtype=np.uint32)
    assert neutu_trace._uninvalidated_length(down, labels) == pytest.approx(3.0)

    diag = np.array([[9, 9, 5], [8, 8, 5], [7, 7, 5]], dtype=np.uint32)
    assert neutu_trace._uninvalidated_length(diag, labels) == pytest.approx(
        2 * np.sqrt(2))


def test_branches_are_traced():
    """A Y should produce a branch point (a degree-3 vertex), not one path."""
    s = (60, 60, 24)
    zz, yy, xx = np.indices(s)
    stem = (zz >= 6) & (zz < 30) & ((yy - 30) ** 2 + (xx - 12) ** 2 <= 9)
    t = np.clip(zz - 30, 0, None)
    arm1 = (zz >= 30) & (zz < 54) & ((yy - 30 - t) ** 2 + (xx - 12) ** 2 <= 9)
    arm2 = (zz >= 30) & (zz < 54) & ((yy - 30 + t) ** 2 + (xx - 12) ** 2 <= 9)
    m = np.asfortranarray((stem | arm1 | arm2).astype(np.uint8))

    skel = neutu_trace.skeletonize(m)
    e = np.asarray(skel.edges)
    degree = np.bincount(e.ravel(), minlength=len(np.asarray(skel.vertices)))
    assert (degree >= 3).any(), "no branch point found on a Y-shaped body"


def test_root_seeds_from_the_thickest_voxel_not_memory_order():
    """NeuTu's rebase: seed at the thickest voxel, not whichever comes first.

    The root determines the whole parent tree, so seeding from `first_label` made
    the skeleton depend on array layout. This builds a body whose thickest part is
    deliberately NOT first in memory order, and checks the seed choice changes the
    root.
    """
    import edt

    thin_z0, thin_z1, blob_z, blob_r = 2, 40, 48, 8
    m = np.zeros((60, 30, 30), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[(zz >= thin_z0) & (zz < thin_z1)
      & ((yy - 15) ** 2 + (xx - 15) ** 2 <= 4)] = 1                        # thin, first
    m[((zz - blob_z) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2)
      <= blob_r ** 2] = 1                                                  # thick, later
    m = np.asfortranarray(m)
    dbf = edt.edt(m, anisotropy=(1, 1, 1), black_border=False, order="F")

    # the sphere's centre is its deepest point, by construction
    thickest = np.unravel_index(int(np.argmax(dbf)), dbf.shape)
    assert thickest[0] == blob_z, "test body is not built as intended"

    from_thick = neutu_trace._find_root(m, dbf)
    from_order = neutu_trace._find_root(m, None)
    assert from_thick is not None and from_order is not None
    # TEASAR's root is the farthest voxel from the seed, so seeding in the blob puts
    # it at the thin cylinder's far tip — a position the fixture fixes, not one
    # read off a previous run.
    assert from_thick[0] == thin_z0


def test_find_root_ignores_zero2inf_background():
    """The raw EDT must be passed: after zero2inf the argmax is background."""
    import edt
    import kimimaro.skeletontricks as skt

    m = _tube(length=30, r=3)
    raw = edt.edt(m, anisotropy=(1, 1, 1), black_border=False, order="F")
    poisoned = skt.zero2inf(raw.copy())
    root = neutu_trace._find_root(m, poisoned)
    assert root is not None
    assert m[tuple(int(v) for v in root)], "root fell outside the object"


def test_attach_at_skeleton_keeps_the_tree_connected():
    """Truncating at the path mask must not fragment the skeleton.

    The mask is a tube of radius up to 3, so the cut voxel is usually not a vertex
    of the existing skeleton. Without prepending an anchor, nothing is shared and
    each branch comes back as its own component.
    """
    s = (60, 60, 24)
    zz, yy, xx = np.indices(s)
    stem = (zz >= 6) & (zz < 30) & ((yy - 30) ** 2 + (xx - 12) ** 2 <= 9)
    t = np.clip(zz - 30, 0, None)
    arm1 = (zz >= 30) & (zz < 54) & ((yy - 30 - t) ** 2 + (xx - 12) ** 2 <= 9)
    arm2 = (zz >= 30) & (zz < 54) & ((yy - 30 + t) ** 2 + (xx - 12) ** 2 <= 9)
    m = np.asfortranarray((stem | arm1 | arm2).astype(np.uint8))

    for attach in (False, True):
        skel = neutu_trace.skeletonize(m, attach_at_skeleton=attach)
        v = np.asarray(skel.vertices)
        e = np.asarray(skel.edges)
        parent = list(range(len(v)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in e:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        assert len({find(i) for i in range(len(v))}) == 1, (
            f"attach_at_skeleton={attach} left the skeleton disconnected")


def test_path_mask_radius_is_capped():
    """The skeleton tube is min(EDT + 1, SKELETON_RADIUS), not the full EDT."""
    import edt

    m = _tube(length=30, r=8)          # fat tube: EDT well above the cap
    dbf = edt.edt(m, anisotropy=(1, 1, 1), black_border=False, order="F")
    pm = np.zeros(m.shape, dtype=bool)
    centre = np.array([[15, 12, 12]], dtype=np.uint32)
    neutu_trace._stamp_path_mask(pm, centre, dbf)
    # a cap of 3 cannot reach 6 voxels away even though the EDT there is large
    assert not pm[15, 12, 12 + 6]
    assert pm[15, 12, 12]


TUBE_RADIUS = 4          # the fixture's radius; bounds below are derived from it


@pytest.mark.parametrize("bend", [False, True])
def test_fix_borders_makes_adjacent_blocks_meet_at_the_face(bend):
    """The property block-first fusion depends on.

    Stage 2 welds fragments with a join bounded to seam scale, because widening it
    invents cable. That only works if both blocks routed a path to the same point on
    the shared face. Cut a tube in two, skeletonize each half independently, and
    compare where each fragment meets the cut.

    Every bound here is derived, not observed:

    - **Each endpoint lies on its block's face**, within one voxel — that is what
      ``fix_borders`` promises, and a vertex is voxel-quantized.
    - **A straight tube must give exactly zero offset.** Both faces expose an
      identical disc, so the contact-area centre is the same point; anything nonzero
      means the target is not a function of the contact area alone.
    - **A bent tube must still land inside the tube**, so the two endpoints cannot be
      further apart than the cross-section they both sit in, ``2 * r``. Adjacent
      blocks do not share a plane — block A ends at z=k, block B starts at z=k+1 — so
      the two discs differ and the residual is genuinely nonzero. Its exact value is
      set by how ``compute_border_targets`` breaks ties between equally-deep voxels,
      which is not a property worth pinning to a number.
    - **``fix_borders`` never makes the seam worse** than not using it.

    The measured values, and what they mean for ``join_radius_nm`` on curved
    processes, are in docs/skeletonization.md — not asserted here, so that a
    routing improvement shows up there rather than as a test failure.
    """
    r = TUBE_RADIUS
    m = _tube(length=60, r=r, pad=4, bend=bend)
    cut = m.shape[0] // 2
    left = np.asfortranarray(m[:cut].copy())
    right = np.asfortranarray(m[cut:].copy())

    def endpoint(mask, face, fix):
        skel = neutu_trace.skeletonize(mask, fix_borders=fix)
        v = np.asarray(skel.vertices, float)
        assert len(v)
        j = int(np.argmin(np.abs(v[:, 0] - face)))
        return v[j], abs(v[j, 0] - face)

    lp, ld = endpoint(left, left.shape[0] - 1, True)
    rp, rd = endpoint(right, 0, True)
    assert ld <= 1.0 and rd <= 1.0, "a fragment did not reach the face"
    off_fix = float(np.linalg.norm(lp[1:] - rp[1:]))

    lp0, _ = endpoint(left, left.shape[0] - 1, False)
    rp0, _ = endpoint(right, 0, False)
    off_no = float(np.linalg.norm(lp0[1:] - rp0[1:]))

    if not bend:
        assert off_fix == 0.0, (
            f"identical contact discs must give the same target, got {off_fix:.2f}")
    assert off_fix <= 2 * r, (
        f"seam offset {off_fix:.2f} exceeds the tube cross-section {2 * r}")
    assert off_fix <= off_no, (
        f"fix_borders made the seam worse: {off_no:.2f} -> {off_fix:.2f}")


def test_border_targets_sit_on_the_faces_and_inside_the_body():
    """Targets must be on a face of the array and inside the object."""
    import cc3d

    m = _tube(length=30, r=4, pad=0)          # pad=0 so the tube touches z faces
    cc = cc3d.connected_components(np.asfortranarray(m), connectivity=26)
    bt = neutu_trace.border_targets(cc)
    assert bt, "no border targets found on a tube that touches the faces"
    for label, pts in bt.items():
        for p in pts:
            z, y, x = (int(v) for v in p)
            assert m[z, y, x], "border target is outside the object"
            on_face = (z in (0, m.shape[0] - 1) or y in (0, m.shape[1] - 1)
                       or x in (0, m.shape[2] - 1))
            assert on_face, f"border target {(z, y, x)} is not on a face"


def test_short_boundary_crossing_stub_survives_min_length():
    """Why per-block min_length does NOT need moving to stage 2.

    The worry was that a branch long overall looks short inside one block, so
    per-block rejection would delete long-range cable. Border targets remove it:
    anything crossing a face has a contact region there, so it gets a MANDATORY
    target exempt from min_length and is traced regardless of its length in this
    block. min_length then only prunes branches wholly interior to the block —
    which is what it is for.

    Geometry: a trunk running near the y-max face with a ~4-voxel spur poking
    through it, comfortably under min_length=10. The spur has to be short *and*
    attached, which is why the trunk sits close to the face — an earlier version put
    the trunk far away, making the spur 12 voxels, and the vacuity guard below
    caught it.
    """
    m = np.zeros((40, 30, 30), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[(zz >= 5) & (zz < 35) & ((yy - 25) ** 2 + (xx - 15) ** 2 <= 9)] = 1   # trunk
    m[(np.abs(zz - 20) <= 1) & (yy >= 25) & ((xx - 15) ** 2 <= 4)] = 1      # spur
    m = np.asfortranarray(m)
    assert m[:, -1, :].any(), "spur does not reach the y-max face"

    def reaches_face(fix):
        skel = neutu_trace.skeletonize(m, fix_borders=fix, min_length=10.0)
        v = np.asarray(skel.vertices, float)
        return bool(len(v)) and bool((v[:, 1] >= m.shape[1] - 2).any())

    assert reaches_face(True), (
        "a boundary-crossing spur was dropped even with fix_borders; per-block "
        "min_length would then delete long-range cable")
    assert not reaches_face(False), (
        "test is vacuous — the spur survives without fix_borders, so it does not "
        "demonstrate the exemption")


def test_keep_single_object_prevents_silent_body_loss():
    """A body shorter than min_length must survive, not vanish.

    NeuTu's keepingSingleObject: if min_length rejects every branch, keep the
    longest. Without it a small body disappears from stage 1 with no record —
    caught by the end-to-end op writing 1 of 2 bodies. That bypasses the pipeline's
    dust reporting, which is where short components are supposed to be accounted
    for.
    """
    m = np.zeros((20, 20, 20), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 9] = 1   # tiny blob
    m = np.asfortranarray(m)

    kept = neutu_trace.skeletonize(m, min_length=50.0, keep_single_object=True)
    dropped = neutu_trace.skeletonize(m, min_length=50.0, keep_single_object=False)
    assert len(np.asarray(kept.vertices)) > 0, "body lost despite keep_single_object"
    assert len(np.asarray(dropped.vertices)) == 0, (
        "test is vacuous — the body survives even with the guard off, so it does "
        "not demonstrate anything")
