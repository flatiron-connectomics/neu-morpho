"""Read a published precomputed volume back — the inverse of :mod:`precomputed`.

``precomputed.py`` only *writes*. This module reads a body's mesh and skeleton back
out of a finished volume, local or ``s3://``, so tooling can inspect what was
actually shipped rather than an intermediate. That distinction is the point: the
encode step is otherwise checked by nothing, and both format bugs this project has
hit were spec-legal files that a writer-only test suite could not see.

**Why this is not in `precomputed.py`.** That module is forbidden from importing
``os`` or calling ``open()`` — a test asserts it, because a single stray one passes
every local test and silently writes nothing to s3. Reading a multires mesh needs
the filesystem: ``vol2mesh.multires.read_object_mesh`` parses the ``.index``
manifest and it opens files, so the objects are staged to a temp directory. Keeping
that here confines the filesystem access to a module that never writes to ``dst``,
and means this package does not own a copy of the multires manifest parser.

No optional dependencies: everything here is numpy plus what the pipeline already
requires. Rendering lives in ``scripts/view_body_3d.py``, which needs the ``viz``
extra.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from neu_vol import location

__all__ = ["read_body_skeleton", "read_body_mesh", "frustum_mesh"]


def read_body_skeleton(volume: str, body_id: int, skeleton_dir: str = "skeleton"):
    """``(vertices_xyz_nm, edges, radii_nm)`` for one body, or ``None`` if absent.

    Vertices come back in the order the format **stores**, which is xyz — matching
    :func:`read_body_mesh`, so the two overlay without a flip. Everything else in
    this package holds vertices zyx in memory; see the zyx/xyz invariant in
    CLAUDE.md, which is what makes this worth stating twice.

    Raises if the skeleton carries no usable ``radius`` attribute rather than
    returning something drawable-but-wrong. **A length check is not enough**: when
    the subresource ``info`` declares no radius attribute, osteoid still hands back
    one value per vertex, filled with the sentinel ``-1``. That is finite and the
    right length, so it survives every structural check and then renders as an
    inverted tube. Negative radii are the signal.
    """
    from osteoid import Skeleton

    blob = location.read_bytes(volume, skeleton_dir, str(int(body_id)))
    if blob is None:
        return None
    skel = Skeleton.from_precomputed(blob, segid=int(body_id))
    vertices = np.asarray(skel.vertices, dtype=float)
    raw = getattr(skel, "radii", None)
    radii = None if raw is None else np.asarray(raw, dtype=float)
    if radii is None or len(radii) != len(vertices):
        raise ValueError(
            f"body {body_id}: skeleton has no usable 'radius' attribute "
            f"({0 if radii is None else len(radii)} radii for "
            f"{len(vertices)} vertices)")
    if len(radii) and not np.isfinite(radii).all():
        raise ValueError(f"body {body_id}: skeleton has non-finite radii")
    if len(radii) and radii.min() < 0:
        raise ValueError(
            f"body {body_id}: skeleton has negative radii (min {radii.min():g}). "
            f"All -1 means {volume}/{skeleton_dir}/info declares no 'radius' "
            f"vertex attribute, so there is nothing to read.")
    return vertices, np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2), radii


def read_body_mesh(volume: str, body_id: int, lod: int | None = None,
                   mesh_dir: str = "mesh"):
    """``(vertices_xyz_nm, faces, lod)`` for one LOD, or ``None`` if absent.

    ``lod`` defaults to the **coarsest** present, which is the cheapest to draw;
    pass 0 for full detail. Fragments are octree cells, so the requested LOD's
    cells are concatenated with their face indices rebased.
    """
    from vol2mesh import multires

    data = location.read_bytes(volume, mesh_dir, str(int(body_id)))
    index = location.read_bytes(volume, mesh_dir, f"{int(body_id)}.index")
    if data is None or index is None:
        return None
    info = location.read_bytes(volume, mesh_dir, "info")

    with tempfile.TemporaryDirectory(prefix="readback-") as tmp:
        with open(os.path.join(tmp, str(int(body_id))), "wb") as f:
            f.write(data)
        with open(os.path.join(tmp, f"{int(body_id)}.index"), "wb") as f:
            f.write(index)
        if info is not None:
            # read_object_mesh reads vertex_quantization_bits from here; without
            # it the parser assumes 16 and would silently misplace vertices if the
            # volume were written with 10.
            with open(os.path.join(tmp, "info"), "wb") as f:
                f.write(info)
        parsed = multires.read_object_mesh(tmp, int(body_id))

    frags = parsed.get("fragments") or []
    if not frags:
        return None
    available = sorted({int(f["lod"]) for f in frags})
    want = available[-1] if lod is None else int(lod)
    if want not in available:
        raise ValueError(f"body {body_id}: LOD {want} not present; have {available}")

    verts, faces, offset = [], [], 0
    for f in frags:
        if int(f["lod"]) != want:
            continue
        v = np.asarray(f["vertices_xyz"], dtype=float)
        verts.append(v)
        faces.append(np.asarray(f["faces"], dtype=np.int64).reshape(-1, 3) + offset)
        offset += len(v)
    if not verts:
        return None
    return np.concatenate(verts), np.concatenate(faces), want


def frustum_mesh(vertices_xyz, edges, radii, sides: int = 8):
    """Skeleton as one truncated cone per edge; returns ``(vertices, faces)``.

    Each edge contributes two rings of ``sides`` points — radius ``r[u]`` at one end,
    ``r[v]`` at the other — joined by a triangle strip, so the surface is correct in
    cross-section at every node and tapers linearly between them, exactly as the
    format's per-vertex radius implies. That makes a radius that does not fit inside
    the body visible as the tube breaking the mesh surface.

    Rings lie in the plane perpendicular to the edge, built from an orthonormal basis
    chosen per edge; the reference direction is swapped near the pole so the cross
    product never degenerates. Zero-length edges are dropped rather than producing
    NaNs.

    Adjacent edges are **not** stitched to each other — at a junction the cones simply
    overlap. That renders correctly for a solid surface and avoids inventing a joint
    geometry the data does not specify.
    """
    v = np.asarray(vertices_xyz, dtype=float)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    r = np.asarray(radii, dtype=float)
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    if not len(e):
        return empty

    p0, p1 = v[e[:, 0]], v[e[:, 1]]
    r0, r1 = r[e[:, 0]], r[e[:, 1]]
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    keep = length > 0
    p0, p1, r0, r1, d, length = (a[keep] for a in (p0, p1, r0, r1, d, length))
    if not len(d):
        return empty

    axis = d / length[:, None]
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
    verts = np.concatenate([ring0, ring1], axis=1).reshape(-1, 3)

    k = np.arange(sides)
    kn = (k + 1) % sides
    tri = np.concatenate([np.stack([k, kn, sides + k], axis=1),
                          np.stack([kn, sides + kn, sides + k], axis=1)], axis=0)
    base = (np.arange(len(p0)) * 2 * sides)[:, None, None]
    return verts, (tri[None, :, :] + base).reshape(-1, 3)
