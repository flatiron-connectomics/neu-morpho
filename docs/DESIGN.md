# em-seg-morpho — Design

Generate per-segment **meshes** and **skeletons** from a segmentation volume, in
neuroglancer-precomputed format. Reuses the shared substrate:
`em-blockrun` (dask local/SLURM + resumable manifest) and `em-volume-tools`
(segmentation-array I/O). Meshes via `vol2mesh`, skeletons via `kimimaro`.

## Dependencies & the reason for the split

Kept separate from `em-volume-tools` because meshing brings heavy, orthogonal
deps (`vol2mesh` from flyem-forge → DracoPy/marching-cubes; `kimimaro` → cc3d/
dijkstra3d). Both consumers share `em-blockrun`.

- **vol2mesh** (flyem-forge, `0.2.post13`): multi-resolution Draco meshes.
  **Verified** at scaffold time — the `vol2mesh.multires` submodule is present.
  It needs `DracoPy` (PyPI; conda `vol2mesh` does *not* pull it in — added to
  `pypi-dependencies`). Relevant API:
  - `Mesh.from_binary_vol(mask_zyx, fullres_box_zyx, method=...)` — single mask.
  - `Mesh.from_binary_blocks(blocks, boxes_zyx, stitch=True)` — **chunked meshing
    + boundary stitching** (this is the large-segment path; we don't hand-roll it).
  - `Mesh.simplify(fraction)`, `laplacian_smooth`, `concatenate_meshes`.
  - `multires.write_info(...)`, `multires.write_object_mesh(out, seg_id,
    fragments, chunk_shape_xyz, grid_origin_xyz, vertex_quantization_bits,
    lod_scales)`, `multires.split_mesh_for_lod(mesh, chunk_shape_xyz, lod)`,
    `multires.encode_multilod_object(...)` — the neuroglancer multi-res writer.
- **kimimaro** (PyPI only, 5.8.4): `skeletonize(labels, teasar_params,
  anisotropy, object_ids, dust_threshold, ...)`; `join_close_components` for
  stitching chunked skeletons. → precomputed skeleton format.

## Pipeline

1. **Enumerate segments** (`segments.py`): the set of segment IDs to process and
   their **bounding boxes**. Bbox source is an open question — a label/spatial
   index if the source has one, else accumulate per-label bboxes by scanning
   blocks. Skipping background (0) and applying a min-size filter.
2. **Per-segment mesh task** (`ops/mesh_segments.py`), mapped over segment IDs by
   `em-blockrun.block_map` (Manifest keyed by **segment id**, resumable):
   - read the segment's bbox region from the seg volume → binary mask
     (`em-volume-tools` backend / crop view),
   - mesh (marching cubes at the configured LOD → simplify → Draco → multi-res),
   - write mesh fragment(s) + index in neuroglancer multi-resolution mesh format,
   - record status (`written` / `empty` / later `failed`) in the manifest.
3. **Large segments → chunked meshing + stitching** (`chunked_mesh.py`): when a
   bbox's binary mask would exceed a memory budget (the OOM we hit), tile the
   bbox into blocks (`em-blockrun.iter_blocks`), mesh each block's mask fragment
   independently (bounded memory), then **stitch** — weld shared boundary
   vertices (via 1-voxel overlap / deterministic boundaries) into one watertight
   mesh before Draco/multi-res encoding. Verify whether `vol2mesh`'s multi-res
   path already does block-based generation + stitching, or we implement it.
4. **Skeletons** (`ops/skeletonize_segments.py`, `skeleton.py`): per-segment (or
   chunked for large) `kimimaro` TEASAR → precomputed skeleton format.

## Config (all parameters, dataclasses in `config.py`)

- **Starting LOD / scale**: default **scale 2** (coarser, smaller, faster);
  scale 0 = highest detail. Whether scale-0 detail is worth the size/time is
  data-dependent — **must be a config parameter** (default 2). Plus number of
  LOD levels.
- Mesh decimation/simplification fraction; Draco quantization bits.
- **Chunked-meshing threshold**: bbox voxel count (or mask bytes) above which to
  switch to chunked+stitch; the per-block chunk shape.
- Output: sharded vs unsharded precomputed mesh; output path (local / s3).
- Skeleton: kimimaro TEASAR params (scale, const, dust threshold, etc.).

## Open questions (resolve as we build)

- **Bounding-box source** per segment (label index vs. scan). Biggest unknown.
- ~~Does vol2mesh do chunked meshing + stitching for us?~~ **Resolved:** yes —
  `Mesh.from_binary_blocks(..., stitch=True)`. `chunked_mesh.py` just decides
  when to chunk, reads the segment's blocks (bounded memory each, at the meshing
  LOD), and calls it. (Meshing at scale ≥1 already cuts mask memory ~8×/level,
  which alone avoids most of the OOM.)
- Sharded vs unsharded mesh/skeleton output (neuroglancer supports both).
- Exact LOD-fragment generation (how many LODs, per-LOD simplification) —
  `split_mesh_for_lod` + `write_object_mesh`/`encode_multilod_object` are the
  pieces; tune against real data.
- **Fault isolation** (deferred from em-blockrun): a meshing run over millions of
  segments *will* have per-segment failures (bad geometry, a chunk that still
  OOMs); we want to record `failed` and continue, then resume-retry. Meshing is
  the concrete driver for adding per-task fault isolation to `em-blockrun`.

## Module layout (scaffold)

```
em_seg_morpho/
├── segments.py       # enumerate segment ids + bounding boxes
├── mesh.py           # single-segment meshing via vol2mesh
├── chunked_mesh.py   # chunked meshing + stitching for large segments
├── skeleton.py       # kimimaro skeletonization
├── precomputed.py    # write neuroglancer multires mesh / skeleton formats
├── config.py         # MeshConfig / SkeletonConfig
└── ops/
    ├── mesh_segments.py        # orchestrate meshing across segments (em-blockrun)
    └── skeletonize_segments.py
```
