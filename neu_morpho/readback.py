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

__all__ = ["read_body_skeleton", "read_body_mesh"]


def read_body_skeleton(volume: str, body_id: int, skeleton_dir: str = "skeleton",
                       *, require_radii: bool = True):
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

    ``require_radii=False`` returns ``radii_nm=None`` instead of raising, for a source
    that legitimately publishes **centrelines only**. FlyEM's male-CNS is one: its
    skeleton ``info`` is bare ``{"@type": "neuroglancer_skeletons"}`` with no
    ``vertex_attributes``, so every body comes back all -1. Cable length and topology
    are still there; calibre is not, and has to come from elsewhere. Keep the default
    ``True`` — on a volume this package wrote, missing radii mean something went wrong.
    """
    from osteoid import Skeleton

    blob = location.read_bytes(volume, skeleton_dir, str(int(body_id)))
    if blob is None:
        return None
    skel = Skeleton.from_precomputed(blob, segid=int(body_id))
    vertices = np.asarray(skel.vertices, dtype=float)
    raw = getattr(skel, "radii", None)
    radii = None if raw is None else np.asarray(raw, dtype=float)
    edges = np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2)
    absent = (radii is None or len(radii) != len(vertices)
              or (len(radii) and not np.isfinite(radii).all())
              or (len(radii) and radii.min() < 0))
    if absent and not require_radii:
        return vertices, edges, None
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
            f"vertex attribute, so there is nothing to read. Pass "
            f"require_radii=False for a source that publishes centrelines only.")
    return vertices, edges, radii


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
