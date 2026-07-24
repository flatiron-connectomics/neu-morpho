"""Mesh a segmentation across segments, orchestrated with em-blockrun.

Shape of the pipeline (see docs/DESIGN.md):

    ids   = segments.iter_segment_ids(seg_spec, ...)
    boxes = segments.segment_bounding_boxes(seg_spec, ids)
    precomputed.write_mesh_info(out_dir, mesh_cfg)          # once
    with start_dask(...) as client:                        # or client=None (serial)
        block_map(ids, mesh_one, client=client,            # key = segment id
                  on_result=lambda r: manifest.record(0, r))

Each ``mesh_one(seg_id)`` (picklable; reopens the seg volume from a spec on the
worker) reads the segment's bbox mask at the meshing LOD, meshes it whole or
chunked+stitched, writes the multi-res object, and returns ``(seg_id, status)``.
Resume filters out segment ids already in the manifest. Empty segments record
``empty``; per-segment failures are candidates for the (deferred) em-blockrun
fault-isolation so one bad segment doesn't sink the run.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

from em_blockrun import Manifest, block_map

from ..config import MeshConfig, OutputConfig
from .. import mesh as _mesh
from .. import precomputed as _precomputed
from .. import segments as _segments


def _mesh_one(seg_id: int, *, seg_spec: dict, boxes: dict, out_dir: str,
              mesh_cfg: MeshConfig) -> tuple[int, str]:
    """Picklable per-segment worker: read mask -> mesh -> write. Returns (id, status)."""
    # from em_volume_tools.backends.base import open_backend
    # backend = open_backend(seg_spec)                      # seg volume at meshing LOD
    # box = boxes[seg_id]
    # mask = (backend.read_region(box_to_slices(box)) == seg_id)
    # if not mask.any(): return (seg_id, "empty")
    # if _mesh.should_chunk(mask.shape, mesh_cfg):
    #     blocks, bboxes = read_blocks(backend, box, mesh_cfg.chunk_shape)
    #     m = _mesh.mesh_from_blocks(blocks, bboxes, mesh_cfg)
    # else:
    #     m = _mesh.mesh_from_mask(mask, box, mesh_cfg)
    # _precomputed.write_segment_mesh(out_dir, seg_id, m, mesh_cfg, ...)
    # return (seg_id, "written")
    raise NotImplementedError("wire read_region + mesh + write_segment_mesh")


def mesh_segments(
    seg_spec: dict,
    out: OutputConfig,
    mesh_cfg: MeshConfig | None = None,
    *,
    segment_ids: Sequence[int] | None = None,
    client: Any | None = None,
    npartitions: int | None = None,
    resume: bool = True,
) -> dict:
    """Generate multi-resolution meshes for every segment (skeleton is separate)."""
    mesh_cfg = mesh_cfg or MeshConfig()
    ids = list(segment_ids) if segment_ids is not None else _segments.iter_segment_ids(
        seg_spec, min_voxels=mesh_cfg.min_segment_voxels)
    boxes = _segments.segment_bounding_boxes(seg_spec, ids)

    out_dir = out.dst.rstrip("/") + "/" + out.mesh_dir
    _precomputed.write_mesh_info(out_dir, mesh_cfg)

    manifest = Manifest(out.progress_path)
    manifest.load() if resume else manifest.reset()
    todo = [s for s in ids if not manifest.is_done(0, s)] if resume else list(ids)

    worker = functools.partial(_mesh_one, seg_spec=seg_spec, boxes=boxes,
                               out_dir=out_dir, mesh_cfg=mesh_cfg)
    try:
        block_map(todo, worker, client=client, npartitions=npartitions,
                  on_result=lambda res: manifest.record(0, res))
    finally:
        manifest.close()
    return {"out_dir": out_dir, "num_segments": len(ids), "processed": len(todo),
            "status_counts": manifest.counts()}
