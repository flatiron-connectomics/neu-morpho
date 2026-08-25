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
from typing import Any

import numpy as np

from neu_vol import location

__all__ = ["read_body_skeleton", "read_body_mesh", "mesh_transform",
           "skeleton_transform", "UnsupportedSubresource"]


class UnsupportedSubresource(ValueError):
    """A mesh or skeleton subresource is in a form this module cannot decode.

    Separate from "this body has no mesh", which is an ordinary ``None``, because the two
    are otherwise indistinguishable to a caller and mean opposite things: one is data, the
    other is a reader pointed at the wrong format. Anything batching reads should let this
    propagate rather than counting it as a missing body.
    """


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

    Reads **sharded and unsharded** skeletons alike, on the ``sharding`` key of the
    skeleton directory's own ``info``. Simpler than the mesh case: a skeleton is a single
    keyed value, so there is no fragment data sitting outside the index and the sharded
    read is an ordinary lookup.

    Vertices are **physical nm**, mapped through the skeleton ``info``'s ``transform``
    exactly as :func:`read_body_mesh` maps the mesh one. Keeping the two in step is the
    whole point — a source that scales one and not the other puts a neuron's skeleton and
    its own surface in different spaces, which is the failure this pair exists to avoid.
    """
    from osteoid import Skeleton

    raw_info, info = _subresource_info(volume, skeleton_dir)
    _check_supported(info, volume, skeleton_dir, "skeleton")
    sharding = info.get("sharding")
    if sharding is not None:
        blob = _shard_reader(volume, skeleton_dir, sharding).read(int(body_id))
    else:
        blob = location.read_bytes(volume, skeleton_dir, str(int(body_id)))
    if blob is None:
        return None
    skel = Skeleton.from_precomputed(blob, segid=int(body_id))
    vertices = np.asarray(skel.vertices, dtype=float)
    matrix = _transform_matrix(info)
    if matrix is not None:
        vertices = vertices @ matrix[:, :3].T + matrix[:, 3]
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
    scale = _radius_scale(matrix, volume, skeleton_dir, require_radii)
    if scale is None:
        return vertices, edges, None
    return vertices, edges, radii * scale


def _radius_scale(matrix, volume: str, skeleton_dir: str, require_radii: bool):
    """The factor carrying a skeleton's radii into nm alongside its vertices.

    A radius is a length, not a point, so it does not transform the way a vertex does —
    only the linear part applies, and only cleanly when that part is a **uniform** scale.
    That is the shape every real one has (male-CNS's mesh transform is ``diag(16)``), so
    the uniform case is handled and the anisotropic one refuses: scaling the vertices by 16
    and leaving the radii alone returns something whose coordinates and calibre are in
    different units, drawn as a neuron of plausible shape and wrong thickness.

    Returns the factor to multiply by — ``1.0`` where there is nothing to do — or ``None``
    meaning **the radii are unusable and must be dropped**. Under an anisotropic transform
    there is no single correct radius, so this raises when radii were asked for and returns
    ``None`` when ``require_radii=False``. That flag already means "centreline is enough",
    and it is the only honest answer here: a source can be perfectly good for length and
    topology while having no scalar calibre at all. Note ``None`` never means "leave them
    alone" — returning the stored radii unscaled beside nm vertices is the mixed-units bug
    this exists to prevent.
    """
    if matrix is None:
        return 1.0
    linear = matrix[:, :3]
    scales = np.sqrt((linear ** 2).sum(axis=0))
    if np.allclose(linear, np.diag(scales)) and np.allclose(scales, scales[0]):
        return float(scales[0])
    if not require_radii:
        return None
    raise ValueError(
        f"{volume}/{skeleton_dir}/info declares a non-uniform transform "
        f"{matrix.tolist()}, so its radii have no single value in nm. Pass "
        f"require_radii=False for the centreline, and scale the radii yourself if the "
        f"anisotropy is intended.")


#: One :class:`~neu_vol.sharded.ShardReader` per (volume, subresource directory), holding the
#: shard and minishard indexes it has parsed. Those cannot change under a live session for a
#: published volume, and rebuilding them per body costs a pair of range requests each —
#: measured against the male-CNS meshes on GCS, 8 bodies out of one shard took 0.70 s with a
#: fresh reader per body and 0.10 s sharing one. `preshift_bits` is what makes that the normal
#: case rather than a lucky one: it exists to put runs of consecutive ids in the same shard, so
#: a batch of neighbouring bodies concentrates rather than spreading. Same reasoning as
#: `neu_vol.location`'s store cache, and the same tolerance for a race — two threads may
#: build a reader each, which wastes one parse and is otherwise harmless.
_SHARD_READERS: dict[tuple, Any] = {}


def _shard_reader(volume: str, subdir: str, sharding):
    """A cached reader for one sharded subresource — mesh or skeleton alike."""
    import json

    from neu_vol.sharded import ShardReader

    cache_key = (str(volume), str(subdir), json.dumps(dict(sharding), sort_keys=True))
    reader = _SHARD_READERS.get(cache_key)
    if reader is None:
        reader = _SHARD_READERS[cache_key] = ShardReader(volume, sharding, subdir)
    return reader


def _subresource_info(volume: str, subdir: str):
    """``(raw bytes, parsed dict)`` of a subresource ``info``; ``{}`` when there is none."""
    import json

    raw = location.read_bytes(volume, subdir, "info")
    return raw, (json.loads(raw) if raw else {})


#: The ``@type`` each reader here understands. Anything else is a real precomputed format
#: that this module cannot decode — `neuroglancer_legacy_mesh` being the one that exists in
#: the wild, whose per-body manifest is JSON at `<body>:0` naming separate fragment files.
_SUPPORTED_TYPES = {"mesh": "neuroglancer_multilod_draco",
                    "skeleton": "neuroglancer_skeletons"}


def _check_supported(info, volume: str, subdir: str, kind: str) -> None:
    """Refuse a subresource whose format this module cannot read.

    **The point is to fail loudly instead of returning ``None``.** Every unreadable form
    fails the same silent way otherwise: the keys this reader asks for are not the keys the
    format uses, so every body reports absent and an entire volume reads as "holds no
    meshes". That is indistinguishable from a correct answer, and it is what a sharded
    store did here before sharding was supported — the meshes were all present.

    An ``info`` with no ``@type``, or none at all, is allowed through: that is the shape of
    a hand-made or legacy-unsharded directory, which the plain keyed read does handle.
    Only a declared and unrecognised type is refused.
    """
    declared = (info or {}).get("@type")
    expected = _SUPPORTED_TYPES[kind]
    if declared is None or declared == expected:
        return
    extra = ""
    if declared == "neuroglancer_legacy_mesh":
        extra = (" That is the legacy single-resolution mesh format, which stores a JSON "
                 "manifest at '<body>:0' rather than a multi-resolution manifest.")
    raise UnsupportedSubresource(
        f"{volume}/{subdir}/info declares '@type' {declared!r}, but this reader "
        f"understands only {expected!r}.{extra} Refusing rather than reporting every "
        f"body absent, which is what reading it with the wrong format would look like.")


def _transform_matrix(info):
    """A subresource ``transform`` as a ``(3, 4)`` matrix, or ``None`` when it is identity.

    ``None`` rather than the identity so the caller skips the multiply entirely — this
    runs per body, and identity is the case for everything this package writes.
    """
    values = (info or {}).get("transform")
    if values is None:
        return None
    matrix = np.asarray(values, dtype=float).reshape(3, 4)
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    return None if np.array_equal(matrix, identity) else matrix


def _parse_manifest(manifest: bytes):
    """The multi-resolution mesh manifest, split into the pieces a re-emit needs.

    Layout: ``chunk_shape[3]`` and ``grid_origin[3]`` as float32, ``num_lods`` as uint32,
    then ``lod_scales[num_lods]`` and ``vertex_offsets[num_lods][3]`` as float32,
    ``num_fragments_per_lod[num_lods]`` as uint32, and finally per LOD a block of
    ``fragment_positions[3][n]`` followed by ``fragment_offsets[n]`` — the latter being
    byte SIZES despite the name.
    """
    num_lods = int(np.frombuffer(manifest, "<u4", 1, 24)[0])
    scales_end = 28 + num_lods * 4
    offsets_end = scales_end + num_lods * 3 * 4
    counts = np.frombuffer(manifest, "<u4", num_lods, offsets_end)

    pos = offsets_end + num_lods * 4
    blocks, sizes = [], []
    for count in counts:
        start = pos
        pos += int(count) * 3 * 4
        sizes.append(np.frombuffer(manifest, "<u4", int(count), pos).astype(np.int64))
        pos += int(count) * 4
        blocks.append(manifest[start:pos])
    return {
        "head": manifest[:24],
        "num_lods": num_lods,
        "lod_scales": manifest[28:scales_end],
        "vertex_offsets": manifest[scales_end:offsets_end],
        "counts": counts,
        "blocks": blocks,
        "sizes": sizes,
    }


def _single_lod_manifest(parsed, want: int) -> bytes:
    """Re-emit the manifest declaring only LOD ``want``, with the earlier LODs empty.

    The LODs are kept rather than renumbered, and that is the whole subtlety: a fragment
    carries no LOD of its own, its cell size is ``chunk_shape * 2**lod`` from the position
    it occupies in the manifest. Relabelling LOD 3 as LOD 0 to save a few bytes would
    dequantize its vertices against a cell 8x too small and silently shrink the mesh.
    So LODs 0..want-1 stay present and are declared to hold zero fragments, which costs
    nothing on the wire and lets the fragment data for ``want`` be the whole payload.
    """
    counts = np.zeros(want + 1, dtype="<u4")
    counts[want] = parsed["counts"][want]
    return b"".join([
        parsed["head"],
        np.uint32(want + 1).tobytes(),
        parsed["lod_scales"][:(want + 1) * 4],
        parsed["vertex_offsets"][:(want + 1) * 3 * 4],
        counts.tobytes(),
        parsed["blocks"][want],
    ])


def _sharded_object(volume: str, body_id: int, mesh_dir: str, sharding, lod):
    """``(fragment_data, manifest, lod)`` for one body and ONE LOD, or ``None``.

    The two halves an unsharded store keeps as ``<body>`` and ``<body>.index`` are both
    inside a ``.shard`` file here, and only the manifest is an indexed entry. The
    fragment data is **not addressable by key at all**: the spec places it immediately
    before the manifest in the same file, so the whole payload occupies
    ``[manifest_offset - total_size, manifest_offset)``, LODs in order. That is the whole
    reason this needs :class:`~neu_vol.sharded.ShardReader` rather than a keyed read.

    **Only the requested LOD is fetched**, which is the difference between a usable and a
    painful batch: the pyramid is dominated by LOD 0, so the coarsest LOD — the default,
    and the one a whole-neuron view wants — is well under 1% of the bytes. Measured on two
    male-CNS bodies, fetching everything to return the coarsest was a 160x and a 232x
    over-read. The returned manifest is re-emitted to match the truncated payload.

    Returning blobs rather than decoding here is what lets the caller stage them in the
    unsharded layout and hand both cases to the same parser.
    """
    reader = _shard_reader(volume, mesh_dir, sharding)
    found = reader.locate(int(body_id))
    if found is None:
        return None
    shard, offset, _size = found
    manifest = reader.read(int(body_id))
    if manifest is None:
        return None

    parsed = _parse_manifest(manifest)
    present = [i for i, c in enumerate(parsed["counts"]) if int(c)]
    if not present:
        return None
    want = present[-1] if lod is None else int(lod)
    if want not in present:
        raise ValueError(f"body {body_id}: LOD {want} not present; have {present}")

    per_lod = [int(s.sum()) for s in parsed["sizes"]]
    total = sum(per_lod)
    start = offset - total + sum(per_lod[:want])

    # Check the extent before asking for it. A manifest claiming more fragment data than
    # can fit before it puts the start of the read at a negative offset, which the store
    # reports as an unsupported byte range — an error about slice syntax, several layers
    # from the actual problem.
    if total > offset:
        raise ValueError(
            f"body {body_id}: manifest claims {total} bytes of mesh fragment data but "
            f"its own entry starts at byte {offset} of {shard}, so the data cannot "
            f"precede it. {volume}/{mesh_dir} is not a well-formed sharded mesh.")

    data = reader.read_range(shard, start, start + per_lod[want])
    if data is None or len(data) != per_lod[want]:
        raise ValueError(
            f"body {body_id}: LOD {want} fragment data is "
            f"{0 if data is None else len(data)} bytes, manifest says {per_lod[want]}. "
            f"The manifest and the data it points at disagree, so {volume}/{mesh_dir} "
            f"is not a well-formed sharded mesh.")
    return data, _single_lod_manifest(parsed, want), want


def read_body_mesh(volume: str, body_id: int, lod: int | None = None,
                   mesh_dir: str = "mesh"):
    """``(vertices_xyz_nm, faces, lod)`` for one LOD, or ``None`` if absent.

    ``lod`` defaults to the **coarsest** present, which is the cheapest to draw;
    pass 0 for full detail. Fragments are octree cells, so the requested LOD's
    cells are concatenated with their face indices rebased.

    Reads **sharded and unsharded** meshes alike, deciding on the ``sharding`` key of
    the mesh directory's own ``info``. A sharded store has no ``<body>`` object to
    fetch, so the unsharded reader came back ``None`` for every body — indistinguishable
    from a volume that holds no meshes. The sharded path fetches **only the requested
    LOD**; the unsharded one reads the whole object, because there the LODs share a single
    key and there is no range to ask for.

    Vertices are **always physical nm**, per the one-model-space invariant: the stored
    coordinates are mapped through the mesh ``info``'s ``transform``, unconditionally and
    with no opt-out. Every mesh this package writes declares an **identity** transform
    (its vertices are already nm), so this changes nothing about our own output and
    matters only for a foreign source — FlyEM's male-CNS declares ``diag(16)``, whose
    meshes were coming back 16x too small and 16x out of register with the skeletons from
    the same volume, with nothing to say so. :func:`mesh_transform` reports the matrix for
    anyone who needs the stored-model coordinates back.
    """
    from vol2mesh import multires

    raw_info, info = _subresource_info(volume, mesh_dir)
    _check_supported(info, volume, mesh_dir, "mesh")
    sharding = info.get("sharding")

    if sharding is not None:
        staged = _sharded_object(volume, body_id, mesh_dir, sharding, lod)
        if staged is None:
            return None
        # The re-emitted manifest declares only the LOD fetched, so ask the parser for
        # that one rather than for the caller's `lod` — which may have been None.
        data, index, lod = staged
    else:
        data = location.read_bytes(volume, mesh_dir, str(int(body_id)))
        index = location.read_bytes(volume, mesh_dir, f"{int(body_id)}.index")
        if data is None or index is None:
            return None

    with tempfile.TemporaryDirectory(prefix="readback-") as tmp:
        with open(os.path.join(tmp, str(int(body_id))), "wb") as f:
            f.write(data)
        with open(os.path.join(tmp, f"{int(body_id)}.index"), "wb") as f:
            f.write(index)
        if raw_info is not None:
            # read_object_mesh reads vertex_quantization_bits from here; without
            # it the parser assumes 16 and would silently misplace vertices if the
            # volume were written with 10.
            with open(os.path.join(tmp, "info"), "wb") as f:
                f.write(raw_info)
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
    vertices = np.concatenate(verts)
    matrix = _transform_matrix(info)
    if matrix is not None:
        vertices = vertices @ matrix[:, :3].T + matrix[:, 3]
    return vertices, np.concatenate(faces), want


def mesh_transform(volume: str, mesh_dir: str = "mesh"):
    """The mesh ``info``'s stored-model to nm ``transform``, as a ``(3, 4)`` matrix.

    Always a matrix, identity included — unlike the private helper, which returns
    ``None`` for identity to skip work. :func:`read_body_mesh` has already applied this,
    so it is here for reporting what happened and for recovering stored-model
    coordinates from the nm ones.

    **Raises when there is no ``info`` at the given location**, rather than falling back
    to identity. An identity matrix is a perfectly plausible answer, so a mistyped path
    or a ``mesh_dir`` passed where a volume was wanted would otherwise come back as
    "no scaling here" and be believed.
    """
    return _declared_transform(volume, mesh_dir, "mesh", "mesh")


def skeleton_transform(volume: str, skeleton_dir: str = "skeleton"):
    """The skeleton ``info``'s ``transform``, as a ``(3, 4)`` matrix.

    The counterpart to :func:`mesh_transform`, and worth having precisely because the two
    are separate declarations that a source can disagree on.
    """
    return _declared_transform(volume, skeleton_dir, "skeleton", "skeletons")


def _declared_transform(volume: str, subdir: str, kind: str, info_key: str):
    import json

    raw = location.read_bytes(volume, subdir, "info")
    if raw is None:
        raise FileNotFoundError(
            f"no {kind} info at {volume}/{subdir}/info, so there is no transform to "
            f"report. Pass the VOLUME and its {kind} subdirectory separately — the "
            f"subdirectory comes from the volume info's {info_key!r} key.")
    values = json.loads(raw).get("transform")
    if values is None:
        return np.hstack([np.eye(3), np.zeros((3, 1))])
    return np.asarray(values, dtype=float).reshape(3, 4)
