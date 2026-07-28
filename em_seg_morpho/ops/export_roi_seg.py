"""Copy the ROI's segmentation into the output as a precomputed volume.

Meshes and skeletons are only inspectable against the labels they came from, so
a run that produced them should be able to produce the matching segmentation
too. This is a thin wrapper over ``em_volume_tools.extract_roi`` (crop view ->
multiscale materialize), which already computes the ROI's **physical offset** —
the part that matters here. The copy carries ``voxel_offset`` for its origin, so
neuroglancer places it at its true global position and it lands in the same
physical-nm space as the meshes and skeletons (coords.py). Without that the
labels would sit at the origin while the meshes sat tens of microns away.

By default the exported region is **expanded to whole blocks**, matching the
region that was actually meshed — stage 1 keeps intersecting blocks whole
(roi.py), so the meshes cover more than the literal ROI and an un-expanded copy
would be missing labels around the edges.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# The only precomputed profile em-volume-tools ships; the name is about where it
# is usually written, not a requirement to be on S3.
PRECOMPUTED_PROFILE = "s3-neuroglancer"


def block_align(roi: Sequence[int], block_shape: Sequence[int],
                shape: Sequence[int]) -> tuple[int, ...]:
    """Expand an ROI to whole blocks of the global grid, clipped to the volume."""
    lo = [int(roi[a]) // block_shape[a] * block_shape[a] for a in range(3)]
    hi = [min(int(shape[a]),
              math.ceil(int(roi[3 + a]) / block_shape[a]) * block_shape[a]) for a in range(3)]
    return tuple(lo + hi)


def export_roi_seg(
    seg_spec: dict | str,
    out_dir: str,
    *,
    roi: Sequence[int] | str | None,
    voxel_size: Sequence[float],
    block_shape: Sequence[int] = (256, 256, 256),
    align_to_blocks: bool = True,
    multiscale: bool = True,
    encoding: str | None = "compressed_segmentation",
    client: Any | None = None,
    npartitions: int | None = None,
    delete_existing: bool = False,
    resume: bool = True,
) -> dict:
    """Write the ROI's labels to ``out_dir`` as a neuroglancer-precomputed volume.

    ``roi`` is in the voxels of the scale ``seg_spec`` is opened at, the same
    convention the other ops use; ``None`` copies the whole volume (rarely what
    you want). Returns the extract summary plus the region actually written.
    """
    from em_volume_tools.backends.base import open_backend
    from em_volume_tools.ops.roi import extract_roi

    from .. import roi as _roi

    shape = open_backend(seg_spec if isinstance(seg_spec, dict)
                         else {"path": seg_spec}).shape
    region = _roi.clip_to_shape(_roi.parse_roi(roi), shape)
    if region is None:
        region = (0, 0, 0, *shape)
    elif align_to_blocks:
        region = block_align(region, block_shape, shape)

    start, stop = list(region[:3]), list(region[3:])
    n_vox = math.prod(b - a for a, b in zip(start, stop))
    logger.info("exporting segmentation %s:%s (%.1f Mvox) -> %s",
                tuple(start), tuple(stop), n_vox / 1e6, out_dir)

    summary = extract_roi(
        seg_spec, out_dir, start, stop, voxel_size,
        kind="segmentation",              # mode (not mean) downsampling for labels
        encoding=encoding,
        profile=PRECOMPUTED_PROFILE,
        multiscale=multiscale,
        name="segmentation",
        client=client, npartitions=npartitions,
        delete_existing=delete_existing, resume=resume,
    )
    return {**summary, "region": region, "n_voxels": n_vox, "out_dir": out_dir}
