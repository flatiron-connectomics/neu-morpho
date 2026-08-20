"""Read each pyramid level's shape and voxel size from the source metadata.

The coordinate contract (coords.py) insists that every scale be described by its
**own voxel size in nm**, never by an assumed ``2**scale`` factor — real pyramids
are anisotropic and non-standard downsample factors are common. That only helps
if the number actually comes from the data, so this reads it:

- **precomputed**: each entry of ``info["scales"]`` carries its own ``resolution``
  (xyz nm) and ``size`` (xyz voxels).
- **zarr / OME-NGFF**: each multiscale dataset carries its own ``scale``
  coordinate transformation.

Scales are ordered finest-first, so index 0 is full resolution and matches
``scale_index`` in a precomputed spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ScaleInfo:
    """One pyramid level, in canonical zyx."""

    index: int
    shape: tuple[int, int, int]              # voxels (z, y, x)
    voxel_size: tuple[float, float, float]   # nm (z, y, x)
    key: str = ""                            # precomputed scale key, if any

    def factor_from(self, finest: "ScaleInfo") -> tuple[float, float, float]:
        """Full-res voxels per voxel of this scale (NOT assumed to be 2**index)."""
        return tuple(self.voxel_size[a] / finest.voxel_size[a] for a in range(3))


def read_scales(spec: str | Mapping[str, Any]) -> list[ScaleInfo]:
    """All pyramid levels of a source, finest first. Raises if metadata is absent."""
    from neu_vol.source_metadata import detect_backend

    spec = {"path": spec} if isinstance(spec, str) else dict(spec)
    kv = _kvstore(spec)
    # detect_backend takes a *location*, not a spec — handing it the spec dict
    # loses the kvstore driver and tensorstore then refuses to open it.
    backend = spec.get("backend") or detect_backend(kv)

    if backend == "neuroglancer_precomputed":
        return _precomputed_scales(kv)
    if backend in ("zarr3", "zarr2"):
        return _zarr_scales(kv)
    raise ValueError(
        f"cannot read scale metadata for backend {backend!r}; "
        "pass voxel sizes explicitly instead")


def _kvstore(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Directory kvstore for a spec (with a trailing slash, so keys append)."""
    from neu_vol.location import spec_kvstore

    kv = dict(spec_kvstore(spec))
    if "path" in kv and not str(kv["path"]).endswith("/"):
        kv["path"] = str(kv["path"]) + "/"
    return kv


def _read_key(kv: Mapping[str, Any], key: str) -> bytes | None:
    """Read one metadata key through ``location``, never ``ts.KvStore.open`` directly.

    Opening here was a fourth store-opening path that skipped both of the things
    ``location`` exists to guarantee: the **per-prefix store cache**, so every call paid
    a fresh open (visible as two S3 credential-provider log lines and a round trip per
    ``read_scales``), and **``ensure_credentials``**, which makes it a latent 403 in any
    process that has not otherwise bootstrapped (invariant 8). Same class of bug as the
    one already fixed in ``source_metadata._read_key``.
    """
    from neu_vol.location import read_bytes

    return read_bytes(kv, key)


def _precomputed_scales(kv: Mapping[str, Any]) -> list[ScaleInfo]:
    raw = _read_key(kv, "info")
    if raw is None:
        raise ValueError(f"no precomputed 'info' at {kv}")
    scales = json.loads(raw)["scales"]
    # finest first; `scale_index` in a tensorstore spec indexes this same order
    ordered = sorted(scales, key=lambda s: tuple(s["resolution"]))
    return [
        ScaleInfo(index=i,
                  shape=tuple(int(v) for v in s["size"][::-1]),          # xyz -> zyx
                  voxel_size=tuple(float(v) for v in s["resolution"][::-1]),
                  key=str(s.get("key", "")))
        for i, s in enumerate(ordered)
    ]


def _zarr_scales(kv: Mapping[str, Any]) -> list[ScaleInfo]:
    from neu_vol.location import join

    raw = _read_key(kv, "zarr.json")
    if raw is None:
        raise ValueError(f"no 'zarr.json' at {kv} (zarr v2 OME groups not supported here)")
    meta = json.loads(raw)
    ome = meta.get("attributes", {}).get("ome")
    if meta.get("node_type") != "group" or not ome:
        raise ValueError("source is a bare zarr array, not an OME multiscale group; "
                         "pass voxel sizes explicitly")
    ms = ome["multiscales"][0]
    axes = ms["axes"]
    spatial = [i for i, a in enumerate(axes) if a.get("type") == "space"]

    out = []
    for i, ds in enumerate(ms["datasets"]):
        scale = next(t["scale"] for t in ds["coordinateTransformations"] if t["type"] == "scale")
        sub = _read_key(join(kv, ds["path"] + "/"), "zarr.json")
        shape = json.loads(sub)["shape"] if sub else None
        out.append(ScaleInfo(
            index=i,
            shape=tuple(int(shape[a]) for a in spatial) if shape else (0, 0, 0),
            voxel_size=tuple(float(scale[a]) for a in spatial),
            key=str(ds["path"])))
    return out


def scale_spec(spec: str | Mapping[str, Any], scale_index: int) -> dict:
    """A read spec pinned to one pyramid level.

    precomputed selects the level with ``scale_index``; zarr addresses the level's
    subgroup by path, so the two need different treatment.

    **Always go through this rather than hand-writing a spec.** The key is
    ``scale_index``, and an unrecognised one is *silently ignored*: a spec carrying
    ``{"scale": 2}`` opens at full resolution and reports the scale-0 shape, so the
    coordinates you pass are then interpreted 4x too fine and read the wrong place
    entirely. That fails as empty data, not as an error.
    """
    from neu_vol.source_metadata import detect_backend
    from neu_vol.location import join

    spec = {"path": spec} if isinstance(spec, str) else dict(spec)
    kv = _kvstore(spec)
    backend = spec.get("backend") or detect_backend(kv)

    if backend == "neuroglancer_precomputed":
        return {"backend": "neuroglancer_precomputed", "kvstore": kv, "scale_index": scale_index}
    scales = read_scales(spec)
    if not 0 <= scale_index < len(scales):
        raise IndexError(f"scale {scale_index} out of range (source has {len(scales)})")
    return {"backend": backend or "zarr3", "kvstore": join(kv, scales[scale_index].key)}


def describe(spec: str | Mapping[str, Any]) -> str:
    """Human-readable pyramid listing — print this before committing to a run."""
    scales = read_scales(spec)
    lines = [f"{'scale':>5}  {'shape (z,y,x)':>24}  {'voxel nm (z,y,x)':>22}  key"]
    for s in scales:
        lines.append(f"{s.index:>5}  {str(s.shape):>24}  "
                     f"{str(tuple(round(v, 3) for v in s.voxel_size)):>22}  {s.key}")
    return "\n".join(lines)
