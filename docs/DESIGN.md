# em-seg-morpho — Design

Generate per-body **meshes** and **skeletons** from a segmentation volume, in
neuroglancer-precomputed format. Reuses `em-blockrun` (dask local/SLURM +
resumable manifest) and `em-volume-tools` (segmentation-array I/O). Meshes via
`vol2mesh`, skeletons via `kimimaro`.

Kept separate from `em-volume-tools` because meshing brings heavy, orthogonal
deps. Motivation vs. the older mesh-n-bone: its CGAL mesh/skeleton path **drops
disconnected components → false body splits**; vol2mesh + kimimaro capture full
bodies (all components).

## Meshing is block-first and two-stage

Modeled on the mesh-n-bone pipeline (verified from its specimen3 configs): iterate
**blocks**, not bodies. This never builds a whole-object binary mask (the OOM), and
reads each block exactly once.

**Stage 1 — chunk** (`block_map` over non-empty blocks; manifest group `"chunk"`,
key = block index): read a block once at the meshing scale, mesh **all present (or
allowlisted) labels together** via `Mesh.from_label_volume(block, fullres_box,
labels=…, ensure_halo=True)`, per-block simplify (keeps stage-2 light), and write
one fragment per `(body, block)` under `<chunked>/<body>/<iz>_<iy>_<ix>.drc`.

**Stage 2 — assemble** (`block_map` over bodies discovered by listing `<chunked>/`;
group `"assemble"`, key = body id): gather a body's fragments, `concatenate_meshes`
→ `stitch_adjacent_faces` (welds block boundaries, keeps **all** components) →
`multires.write_object_mesh`.

One manifest, two groups (block-index tuples vs. scalar body ids — the generalized
em-blockrun keys). Resume skips done blocks/bodies; running only stage 2 reuses
fragments on disk (mesh-n-bone's `reuse_existing_chunked`).

**Occupancy prefilter** (`occupancy.py`): read a coarse scale once and skip empty
blocks (mesh-n-bone: ~85% empty). With an allowlist, "occupied" = `isin(coarse,
allowlist)` so only blocks containing a target body are chunked.

**Body allowlist** (`allowlist.py`): mesh only listed ids (`from_label_volume(labels=)`),
falling back to **all labels** when none is given.

## Coordinate contract (alignment) — `coords.py`

mesh-n-bone hit **mesh↔skeleton offset** bugs (segment 231668). Root cause:
meshes and skeletons landing in different coordinate spaces. The contract that
prevents it, verified by `tests/test_alignment.py`:

- **One model space for everything: physical nanometers** (full-res world, zyx),
  with **identity** neuroglancer `info` transforms (mesh *and* skeleton).
- Expressed via each scale's **voxel size (nm)** — never an assumed `2**scale`
  pyramid factor — so anisotropic / non-standard pyramids stay aligned. (This is
  the hardening against the "pretty substantial modifications for our not-clean
  dataset" experience.)
- Mesh: `from_label_volume(block, physical_box(region, voxel_size_mesh))` →
  vertices in nm (verified: a cube at mesh-voxels [4:12] lands at nm [32:96]).
- Skeleton: run kimimaro with `anisotropy = voxel_size` (vertices become physical
  nm, crop-local), then `skeleton_to_physical(v, crop_origin_nm)` — the crop
  origin is the piece whose omission is the 231668 offset.
- Multires octree: `chunk_shape` / `grid_origin` in nm; `grid_origin = 0`.

## vol2mesh API used (verified; `multires` needs `DracoPy` from PyPI)

`Mesh.from_label_volume(vol, fullres_box, labels=, ensure_halo=True)`,
`Mesh.simplify(fraction)`, `concatenate_meshes`, `Mesh.stitch_adjacent_faces`,
`Mesh.serialize`/`from_file`; `multires.write_info` / `write_object_mesh` /
`split_mesh_for_lod`.

## Skeletons (separate op, stays per-body)

**kimimaro** is the chosen skeletonizer — validated as the clear winner on *real*
Megaphragma neurons (cleaner than skeletor `by_wavefront`/`by_teasar`, which
convolve bulbs and/or drop branches). skeletor stays available only via the
comparison harness (`skelcompare.py` + `scripts/compare_skeletons.py`: same body
→ mask + mesh → methods → metrics + interactive 3D HTML). Optional small mask
open/close (`SkeletonConfig.mask_opening_iters`/`mask_closing_iters`, default 0)
can tame convolution from imperfect segmentation — keep tiny (a voxel at a coarse
skeleton scale is large and can sever thin processes / merge dense arbors).

Skeletonization is per-body from a crop; the crop **bbox comes from the metrics
DB** (below), not pre-known values.

## Per-body metrics database (`metrics_db.py`, `ops/index_segments.py`)

A SQLite DB, one row per body, is the join point for per-body outputs and removes
the "need a body's bbox before we can crop it" dependency.

- **Index scan** (`index_segments`): a block-map reduction over the segmentation —
  each block reports per-label bbox + voxel count; the single-writer driver
  merges into the DB (min/max bbox, summed counts) **atomically per block**
  (bbox+count and the block's done-marker commit together), so it's exact, covers
  all bodies, and resumes without double-counting. Bbox stored in full-res voxels.
- **Enrichment**: meshing/skeletonization `update_body(...)` their columns
  (mesh area/verts/components; cable length, branches, tips, max radius).
- **Consumers**: `crop_at_scale(body_id, factor, margin)` gives skeletonization
  its per-body crop; `bodies_by_size` / `write_allowlist` generate the meshing
  allowlist by size — closing the loop (index → size-filter → allowlist → mesh).

## Config (`config.py`)

- `MeshConfig`: `mesh_scale` (default 2; 0 = full detail), `fullres_factor`,
  `block_shape`, `decimation_fraction`, `num_lods`, `draco_quantization_bits`,
  `fragment_format`.
- `SkeletonConfig`, `OutputConfig`.

## Open questions

- Exact **LOD-fragment generation** for the multires write (`split_mesh_for_lod`
  + `write_object_mesh`; how many LODs, per-LOD decimation) — the one
  `NotImplementedError` left; tune against real data.
- **Fixed-edge simplification** in stage 1 (preserve block boundaries) — vol2mesh
  `simplify` options TBD.
- Fragment store on **object storage** (currently ceph filesystem).
- Scale/voxel-size wiring: caller passes `seg_spec` at the meshing scale +
  `occupancy_spec` at a coarse scale, with voxel sizes — keeps the op
  format-agnostic. Could auto-derive from source metadata later.
- **Fault isolation** (deferred from em-blockrun): a meshing run over many bodies
  *will* have per-item failures; add `failed`-and-continue to em-blockrun.

## Module layout

```
em_seg_morpho/
├── config.py         # MeshConfig / SkeletonConfig / OutputConfig
├── allowlist.py      # load body allowlist (or None = all)
├── occupancy.py      # coarse-scale -> non-empty block indices
├── fragments.py      # per-(body,block) fragment store (write / list_bodies / read)
├── coords.py         # coordinate contract: physical-nm space (mesh↔skeleton alignment)
├── mesh.py           # mesh_block (stage 1) + assemble_body (stage 2)
├── precomputed.py    # write_mesh_info + write_body_multires
├── skeleton.py       # kimimaro (per-body; vertices -> global nm; optional mask open/close)
├── metrics_db.py     # SQLite per-body metrics (bbox/count/volume + enrichment)
├── skelcompare.py    # skeletonization comparison harness (kimimaro vs skeletor) + 3D viz
└── ops/
    ├── meshify.py         # two-stage meshing orchestration via em-blockrun
    └── index_segments.py  # parallel scan -> per-body metrics DB (bbox/count)
```
