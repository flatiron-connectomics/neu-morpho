"""Write neuroglancer precomputed meshes and skeletons (stage-2 output).

Meshes — thin wrappers over ``vol2mesh.multires``:

  - :func:`write_mesh_info` -> ``multires.write_info`` (the mesh ``info``, once).
  - :func:`write_body_multires` -> per-LOD octree fragments
    (``multires.split_mesh_for_lod``) + ``multires.write_object_mesh``.

Skeletons — the ``neuroglancer_skeletons`` format written directly (via osteoid's
``Skeleton.to_precomputed``, which kimimaro already depends on, so no CloudVolume):

  - :func:`write_skeleton_info` -> the skeleton ``info``, once.
  - :func:`write_body_skeleton` -> one binary blob per body.

**Axis order is the alignment trap.** Model space is physical nm, held *zyx*
in memory (``Mesh.vertices_zyx``, kimimaro vertices). Both precomputed formats
store *xyz*. vol2mesh flips for meshes internally; :func:`encode_skeleton` does
the same flip for skeletons. Get it wrong and skeletons come out mirrored through
the z=x diagonal relative to their meshes — the same class of bug as the dropped
crop origin (see coords.py).

**Every write here goes through neu-vol' kvstore helpers**, never
``open()``, so ``output_dir`` may equally be a local path or an ``s3://`` URL.
vol2mesh makes this possible by separating encoding from writing: we use
``multires.build_info`` / ``encode_multilod_object``, which return a dict and
bytes, and never ``multires.write_info`` / ``write_object_mesh``, which would
write to the filesystem behind our back.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from neu_vol import exists, read_json, write_bytes, write_json

# Lives in neu-vol, which owns a volume's `info`. Imported rather than wrapped so
# there is one implementation, and re-exported here because this is where callers have
# always found it — linking the subresources after the seg stage is this package's job
# even though writing the `info` key is not.
from neu_vol.ops.subresources import link_subresources  # noqa: F401

from .config import MeshConfig

# Written for every body, and declared in the skeleton ``info``. The binary must
# carry exactly these attributes, in this order, so we normalize every skeleton
# to them rather than trusting whatever the producer happened to attach.
#
# float32 ONLY — see :func:`_check_vertex_attributes`. osteoid also attaches a
# uint8 ``vertex_types`` (SWC type codes) by default; it is deliberately not
# emitted, because neuroglancer refuses to render it and kimimaro leaves it all
# zeros for us anyway (no soma detection).
SKELETON_VERTEX_ATTRIBUTES = [
    {"id": "radius", "data_type": "float32", "num_components": 1},
]

# The precomputed spec permits int8/uint8/int16/uint16/int32/uint32 here, but
# neuroglancer uploads skeleton vertex attributes as WebGL vertex attributes and
# its shader path only handles float32. A non-float32 attribute produces a
# spec-legal file that the viewer rejects outright with
# "Data type not supported by WebGL: UINT8", taking the whole layer with it.
WEBGL_SKELETON_ATTRIBUTE_DTYPES = frozenset({"float32"})

IDENTITY_TRANSFORM = [1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0]


VALID_QUANTIZATION_BITS = (10, 16)      # neuroglancer multiresolution mesh spec


def quantization_step_nm(cfg: MeshConfig, chunk_shape_xyz: Sequence[float]) -> float:
    """Position resolution (nm) at the COARSEST LOD — the one that bites.

    Draco spreads ``2**bits`` steps across each octree cell, and a LOD-l cell is
    ``chunk * 2**l`` wide, so the step doubles per LOD.
    """
    coarsest_cell = max(chunk_shape_xyz) * (2 ** max(0, cfg.num_lods - 1))
    return coarsest_cell / (2 ** cfg.draco_quantization_bits)


def check_quantization(cfg: MeshConfig, chunk_shape_xyz: Sequence[float],
                       voxel_size_zyx: Sequence[float], *, min_ratio: float = 4.0) -> None:
    """Fail fast if Draco quantization would collapse triangles at a coarse LOD.

    When the coarsest LOD's quantization step approaches the voxel size,
    decimated triangles quantize to zero area and Draco rejects the *whole body*
    with "All triangles are degenerate" — which on real data hit 29% of bodies at
    10 bits, scale 2. Cheap to check up front; expensive to discover after an
    hour of meshing.
    """
    if cfg.draco_quantization_bits not in VALID_QUANTIZATION_BITS:
        raise ValueError(
            f"draco_quantization_bits must be one of {VALID_QUANTIZATION_BITS} "
            f"(neuroglancer spec), got {cfg.draco_quantization_bits}. Intermediate "
            "values encode fine but neuroglancer will not read them.")

    step = quantization_step_nm(cfg, chunk_shape_xyz)
    finest = min(voxel_size_zyx)
    if step * min_ratio > finest:
        raise ValueError(
            f"Draco quantization step at the coarsest LOD is {step:.3g} nm, too close to "
            f"the {finest:.3g} nm voxel — triangles will collapse and bodies will fail to "
            f"encode. Raise draco_quantization_bits (currently "
            f"{cfg.draco_quantization_bits}; 16 is the max the spec allows), or reduce "
            f"num_lods ({cfg.num_lods}) or block_shape ({tuple(cfg.block_shape)}).")


def write_mesh_info(output_dir: str, cfg: MeshConfig, *, transform=None,
                    lod_scale_multiplier: float = 1.0) -> None:
    """Write the multi-resolution mesh ``info`` (once per output volume).

    ``build_info``, not ``multires.write_info`` — the latter writes to the local
    filesystem, which would silently bypass an ``s3://`` destination.
    """
    from vol2mesh import multires

    info = multires.build_info(vertex_quantization_bits=cfg.draco_quantization_bits,
                               transform=transform,
                               lod_scale_multiplier=lod_scale_multiplier)
    write_json(output_dir, info, "info")


def write_body_multires(output_dir: str, body_id: int, mesh, cfg: MeshConfig,
                        *, chunk_shape_xyz: Sequence[float], grid_origin_xyz: Sequence[float]) -> int:
    """Write one body's multi-resolution mesh; returns fragments written (0 if empty).

    LOD 0 is the assembled mesh; each coarser LOD decimates by
    ``cfg.lod_decimation_factor`` and is octree-partitioned by
    ``multires.split_mesh_for_lod``. ``encode_multilod_object`` produces the
    ``<body>`` data file and ``<body>.index`` manifest (unsharded multires format).
    Vertices are in nm (model space); ``info`` transform is identity.
    """
    from vol2mesh import multires

    fragments_by_lod: dict[int, dict] = {}
    lod_mesh = mesh
    for lod in range(cfg.num_lods):
        if lod > 0:
            try:
                lod_mesh.simplify(1.0 / cfg.lod_decimation_factor)   # coarsen (mutates)
            except Exception:
                pass
        fragments_by_lod[lod] = multires.split_mesh_for_lod(lod_mesh, list(chunk_shape_xyz), lod)

    data_bytes, index_bytes, counts = multires.encode_multilod_object(
        fragments_by_lod, list(chunk_shape_xyz), list(grid_origin_xyz),
        vertex_quantization_bits=cfg.draco_quantization_bits)
    if not data_bytes:
        return 0

    # Data before index: the index is what makes a body discoverable, so writing
    # it last means a torn write leaves an unreferenced blob rather than an index
    # pointing at data that is not there yet.
    write_bytes(output_dir, data_bytes, str(int(body_id)))
    write_bytes(output_dir, index_bytes, f"{int(body_id)}.index")
    return sum(counts)


# --------------------------------------------------------------------------- #
# Skeletons
# --------------------------------------------------------------------------- #
def volume_exists(volume_dir: str) -> bool:
    """True if a precomputed ``info`` is present at ``volume_dir``.

    The cheap liveness test for a destination — one request, and it works for
    object stores where there is no directory to stat. Used to catch a manifest
    that has outlived the data it describes (see the CLI's resume guard).
    """
    return exists(volume_dir, "info")


def _check_vertex_attributes(attrs: Sequence[dict]) -> None:
    """Reject attribute dtypes neuroglancer cannot render.

    The file would be perfectly spec-legal; the viewer just fails the whole layer
    with "Data type not supported by WebGL: <TYPE>". Catching it at write time
    beats discovering it in the browser.
    """
    bad = [a for a in attrs if a.get("data_type") not in WEBGL_SKELETON_ATTRIBUTE_DTYPES]
    if bad:
        names = ", ".join(f"{a.get('id')!r}={a.get('data_type')!r}" for a in bad)
        raise ValueError(
            f"skeleton vertex attributes must be float32 for neuroglancer to render "
            f"them; got {names}. The precomputed spec allows integer types, but the "
            f"viewer uploads these as WebGL vertex attributes and rejects the layer "
            f'with "Data type not supported by WebGL". Cast the attribute to float32 '
            f"or drop it.")


def write_skeleton_info(output_dir: str, *, transform: Sequence[float] | None = None,
                        vertex_attributes: Sequence[dict] | None = None) -> None:
    """Write the ``neuroglancer_skeletons`` ``info`` (once per output volume).

    The transform is **identity**: vertices are already physical nm in the same
    model space as the meshes, so neuroglancer must not re-scale them.
    """
    attrs = list(vertex_attributes or SKELETON_VERTEX_ATTRIBUTES)
    _check_vertex_attributes(attrs)
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": list(transform if transform is not None else IDENTITY_TRANSFORM),
        "vertex_attributes": attrs,
    }
    write_json(output_dir, info, "info")


def encode_skeleton(skeleton) -> bytes:
    """Encode a global-nm **zyx** skeleton as precomputed **xyz** bytes.

    Normalizes the vertex attributes to :data:`SKELETON_VERTEX_ATTRIBUTES` so the
    blob always matches the ``info`` — including *dropping* osteoid's default
    uint8 ``vertex_types``, which the viewer cannot render. Radii kimimaro did not
    supply are left as its ``-1`` sentinel.
    """
    from osteoid import Skeleton

    _check_vertex_attributes(SKELETON_VERTEX_ATTRIBUTES)
    verts_zyx = np.asarray(skeleton.vertices, dtype=np.float32).reshape(-1, 3)
    verts_xyz = np.ascontiguousarray(verts_zyx[:, ::-1])          # <-- the flip
    out = Skeleton(vertices=verts_xyz,
                   edges=np.asarray(skeleton.edges, dtype=np.uint32).reshape(-1, 2),
                   segid=getattr(skeleton, "id", None))
    n = len(verts_xyz)
    radius = getattr(skeleton, "radius", None)
    if radius is not None and len(radius) == n:
        out.radius = np.asarray(radius, dtype=np.float32)
    # exactly what info declares, in that order — to_precomputed walks this list
    out.extra_attributes = [dict(a) for a in SKELETON_VERTEX_ATTRIBUTES]
    return out.to_precomputed()


def write_body_skeleton(output_dir: str, body_id: int, skeleton) -> int:
    """Write one body's skeleton blob; returns the vertex count (0 if empty)."""
    if skeleton is None or len(skeleton.vertices) == 0:
        return 0
    data = encode_skeleton(skeleton)
    # One atomic key write — the previous tmp+os.replace guarded against a torn
    # blob, which a single kvstore write already rules out.
    write_bytes(output_dir, data, str(int(body_id)))
    return len(skeleton.vertices)
