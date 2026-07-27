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

## Skeletons — block-first, two-stage (`ops/skeletonize_segments.py`)

**kimimaro** is the chosen skeletonizer — validated as the clear winner on *real*
Megaphragma neurons (cleaner than skeletor `by_wavefront`/`by_teasar`, which
convolve bulbs and/or drop branches). skeletor stays available only via the
comparison harness (`skelcompare.py` + `scripts/compare_skeletons.py`: same body
→ mask + mesh → methods → metrics + interactive 3D HTML). Optional small mask
open/close (`SkeletonConfig.mask_opening_iters`/`mask_closing_iters`, default 0)
can tame convolution from imperfect segmentation — keep tiny (a voxel at a coarse
skeleton scale is large and can sever thin processes / merge dense arbors).

Skeletonization mirrors meshing rather than cropping per body:

- **Stage 1 `skel-chunk`** — `block_map` over occupancy-filtered blocks;
  `skeletonize_block` runs `kimimaro.skeletonize(..., fix_borders=True,
  anisotropy=voxel_size)` over all (allowlisted) labels in the block at once,
  shifts vertices to global nm by the block origin, and writes per-(body, block)
  fragments to the fragment store.
- **Stage 2 `skel-fuse`** — `block_map` over bodies discovered by listing the
  fragment dirs; `fuse_body` runs `join_close_components` (welds the block seams
  — `fix_borders` put each fragment's endpoint at the centre of the contact area,
  so seam endpoints are ~a voxel apart) then `postprocess` (drops dust and short
  ticks), writes the precomputed skeleton, and reports metrics to the DB.

**Why block-first**, given skeletons were originally planned per-body: the OOM
hazard in the crop approach is the bounding-box **extent**, not the voxel count.
A sparse arbor has few voxels but a huge bbox, and it is the dense crop array that
blows up. A block is bounded by construction. A body wholly inside one block
yields a single fragment, so fusion is a no-op for it and only large/spanning
bodies see real welding. The per-body crop path (`skeletonize_body`, fed by
`metrics_db.crop_at_scale`) is kept for one-offs and the comparison harness.

Verified empirically against kimimaro 5.8.4 before building: a rod split across
three blocks fuses back to **one** component at full extent, and a blob inside one
block is untouched.

### Skeleton output format

Written directly as `neuroglancer_skeletons` (`precomputed.write_skeleton_info` /
`write_body_skeleton`) via osteoid's `Skeleton.to_precomputed` — osteoid is
already a kimimaro dependency, so no CloudVolume (which we avoid for writing).
`info` declares an identity transform and the `radius` + `vertex_types` vertex
attributes; every blob is normalized to exactly those, so the two can't drift.

**The zyx→xyz flip is part of the alignment contract.** Model space is nm held
*zyx* in memory (`Mesh.vertices_zyx`, kimimaro vertices), but both precomputed
formats *store* xyz. vol2mesh flips internally for meshes; `encode_skeleton` does
it for skeletons. Skipping it yields skeletons mirrored through the z=x diagonal
relative to their meshes — the same class of bug as the dropped crop origin, and
tested the same way (`tests/test_skeleton_precomputed.py`).

## Per-body metrics database (`metrics_db.py`, `ops/index_segments.py`)

A SQLite DB, one row per body, is the join point for per-body outputs and removes
the "need a body's bbox before we can crop it" dependency.

- **Index scan** (`index_segments`): a block-map reduction over the segmentation —
  each block reports per-label bbox + voxel count; the single-writer driver
  merges into the DB (min/max bbox, summed counts) **atomically per block**
  (bbox+count and the block's done-marker commit together), so it's exact, covers
  all bodies, and resumes without double-counting. Bbox stored in full-res voxels.
- **Enrichment**: `update_body(...)` sets a stage's columns (mesh area / verts /
  components; cable length, branches, tips, max radius) — upserting, so a body the
  index scan never saw still gets its metrics. `skel-fuse` writes its metrics from
  the driver, which stays the DB's sole writer.
- **Consumers**: `crop_at_scale(body_id, factor, margin)` gives the per-body crop
  path its crop; `bodies_by_size` / `write_allowlist` generate the meshing
  allowlist by size — closing the loop (index → size-filter → allowlist → mesh).

## Config (`config.py`)

- `MeshConfig`: `mesh_scale` (default 2; 0 = full detail), `fullres_factor`,
  `block_shape`, `decimation_fraction`, `num_lods`, `draco_quantization_bits`,
  `fragment_format`.
- `SkeletonConfig`: `anisotropy` (the skeleton scale's voxel size in nm — there is
  no separate voxel-size argument), `block_shape`, TEASAR params, `fix_borders`,
  and the stage-2 fusion knobs `join_radius_nm` / `postprocess_dust_nm` /
  `postprocess_tick_nm` (all in nm, since vertices are nm).
- `OutputConfig`.

## Open questions

- **`join_radius_nm` default is unbounded** (igneous' behaviour): every component
  of a body is joined into one tree. That guarantees one skeleton per body, but
  bridges genuinely-separate pieces with a long straight edge that inflates
  `cable_length_nm`. Bounding it to a few voxels would weld only block seams.
  Decide against real data.
- **Fixed-edge simplification** in stage 1 (preserve block boundaries) — vol2mesh
  `simplify` options TBD.
- **Meshing DB enrichment** is still unwired: the `assemble` worker should return
  mesh area / verts / components and the driver `update_body` them, the way
  `skel-fuse` now does.
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
├── fragments.py      # per-(body,block) fragment store, meshes (.drc) + skeletons (.skel)
├── coords.py         # coordinate contract: physical-nm space (mesh↔skeleton alignment)
├── mesh.py           # mesh_block (stage 1) + assemble_body (stage 2)
├── precomputed.py    # mesh info/multires + skeleton info/blob (zyx->xyz flip lives here)
├── skeleton.py       # kimimaro: skeletonize_block (stage 1), fuse_body + metrics (stage 2)
├── metrics_db.py     # SQLite per-body metrics (bbox/count/volume + enrichment)
├── skelcompare.py    # skeletonization comparison harness (kimimaro vs skeletor) + 3D viz
└── ops/
    ├── meshify.py              # two-stage meshing orchestration via em-blockrun
    ├── skeletonize_segments.py # two-stage skeletonization (skel-chunk / skel-fuse)
    └── index_segments.py       # parallel scan -> per-body metrics DB (bbox/count)
```
