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
  fragment dirs; `fuse_body` welds the seams, writes the precomputed skeleton,
  and reports metrics to the DB.

### How fusion joins, and why the radius is bounded (measured)

`join_close_components` is greedy single-linkage over components: build a KD-tree
per component, take the globally nearest vertex pair, add **one straight edge**
between those two existing vertices, merge, repeat until nothing is closer than
`radius`. It adds no new vertices, and it joins nearest *vertex* to nearest
*vertex* — not endpoint to endpoint — so a join can land mid-branch and create a
spurious branch point. Components left unjoined are **kept, disconnected**; the
only thing that deletes a component is the dust threshold.

`fuse_body` therefore runs two different joins:

1. An explicit `join_close_components(radius=join_radius_nm)` whose only job is
   the **block seams**. `fix_borders` puts both fragments' endpoints at the centre
   of the contact area, so a seam is ~1 voxel wide; the default radius is
   `2 x max(anisotropy)`, just enough to reach across it.
2. `postprocess`, which internally runs its own join with `restrict_by_radius=True`
   — connect two pieces only where the gap is smaller than the sum of their local
   radii, i.e. their cross-sections nearly touch. This is the principled criterion
   for repairing a segmentation gap, and it is always on.

Measured on synthetic cases (`tests/test_skeletonize_e2e.py` pins both):

| case | `radius=inf` | default (16 nm) | `radius=0` (postprocess only) |
|---|---|---|---|
| rod split by the block grid (~1 voxel seam) | 1 comp, 760 nm | 1 comp, 760 nm | 1 comp, 760 nm |
| one label, two blobs 400 nm apart | **1 comp, 858 nm** | 2 comps, 416 nm | 2 comps, 416 nm |

The unbounded join more than doubles that body's cable length by inventing a
442 nm edge — and it does so precisely for the bodies whose segmentation is least
trustworthy. Hence the seam-scale default. `join_radius_nm=0` skips the explicit
join entirely and still welds seams via postprocess; `float("inf")` restores the
join-everything behaviour if a single connected tree per body is what you want.

**The dust threshold deletes small bodies.** `postprocess_dust_nm` (kimimaro's
1500 nm default) drops every component shorter than itself — and it is applied to
one body's components, so a body whose *entire* skeleton is shorter than that
vanishes. The op reports those as status **`dust`**, distinct from `empty` (no
fragments at all), so the count is visible rather than silent.

### Why fusion inlines postprocess instead of calling it

`fuse_body` reproduces `postprocess`'s sequence (dust → loops → radius-restricted
join → ticks) rather than calling it, for two reasons.

1. **Measurement.** Each step is profiled, so the run reports what was thrown
   away and what was inferred (below).
2. **Tick removal does not otherwise work here.** kimimaro's compiled
   `create_distance_graph`, reached from `remove_ticks`, declares `float` /
   `uint32_t` buffers. `kimimaro.post.remove_loops` rebuilds edges as **int64**,
   and `postprocess` runs it immediately before `remove_ticks` — so stock
   `kimimaro.postprocess` raises `ValueError: Buffer dtype mismatch` for some
   arbors at any non-zero `tick_threshold`, however clean its input was. (It is
   arbor-dependent, not universal: a rod split across three blocks fails, the
   twiggy test arbor happens to survive.) Inlining lets `normalize_dtypes` run
   between those two steps. `tests/test_skeletonize_e2e.py` pins the upstream bug
   and our equivalence to `postprocess` at `tick=0`; if the bug test starts
   failing, kimimaro has fixed it upstream.

### Measuring what was dropped

`skeletonize_segments` returns a `fusion_stats` block (and, with
`fusion_stats_path`, per-body JSONL): components and cable deleted by dust,
branches and cable pruned by ticks, bodies deleted outright, bodies left
multi-component, and how much cable each join *added* —
`inferred_cable_fraction` is the share of the skeleton that is inferred across
gaps rather than measured from the segmentation.

`scripts/sweep_postprocess.py` sweeps both thresholds over real bodies (from a
`skel-chunk` fragment dir, or straight from a volume + metrics DB) or synthetic
ones, and prints that table — the row where `drop%` climbs steeply is where the
threshold has started eating morphology rather than noise. On the synthetic set
(bodies of only ~0.8–1.5 µm cable) dust=1000 already deletes 3 of 5 bodies and
dust=1500 deletes 4 — a scale artifact of tiny test bodies, but the mechanism is
real: the threshold is an absolute length, so it interacts with body size, and
it must be chosen against the actual body-size distribution.

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
- **Enrichment**: `update_body(...)` sets a stage's columns — upserting, so a body
  the index scan never saw still gets its metrics. Both stage-2 workers return
  metrics alongside their status and the **driver** writes them, staying the DB's
  sole writer.
  - `assemble` -> `mesh_area_nm2`, `mesh_verts`, `n_mesh_components`, measured on
    the LOD-0 mesh before `write_body_multires` decimates it in place.
    `n_mesh_components` doubles as QC: block fragments only merge once
    `stitch_adjacent_faces` welds their coincident boundary vertices, so a
    spanning body still reporting >1 means the stitch did not take (a genuinely
    split body also reports >1 — the count alone cannot tell them apart).
  - `skel-fuse` -> `cable_length_nm`, `n_branches`, `n_tips`, `max_radius_nm`.
  - Under an `roi`, metrics describe the **truncated** body, not the whole one.
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

## Running a job (`examples/run_morpho_slurm.py`, `configs/`)

One driver runs `index -> allowlist -> mesh -> skel` (`--stages` picks a subset)
against a dask cluster from `em_blockrun.start_dask`, or in-process with
`--serial`. See the README for invocations.

- **ROI (`roi.py`)** — `--roi z0,y0,x0,z1,y1,x1` filters blocks of the **global**
  grid rather than re-tiling a sub-volume. Block indices, regions and nm
  coordinates are therefore identical in an ROI run and the eventual full run, so
  a trial is a *prefix*: widening the ROI reuses the fragments and manifest
  entries already written. Blocks intersecting the ROI are kept **whole**, never
  clipped — clipping would make a block's content depend on the ROI, and the same
  block index would then mean different data in two runs, so resume would reuse
  the wrong fragment. A body straddling the ROI edge is truncated until its
  neighbouring blocks run.
- **Scales (`scales.py`)** — the driver takes scale *integers* and reads each
  level's real voxel size from the precomputed `info` / zarr OME metadata.
  `ScaleInfo.factor_from` gives true per-axis factors, which are routinely not
  `2**index`: a pyramid that halves x/y but leaves z alone has factor `(1,2,2)`.
  Feeding the coords contract an assumed factor is precisely how meshes and
  skeletons end up in different spaces.
- **Cluster sizing (`configs/`)** — `dask-slurm-any.yaml` sets **no**
  `--constraint`. Rusty's `gen` partition mixes rome (128c/1 TB), icelake
  (64c/1 TB) and genoa (96c/1.5 TB); pinning to genoa would drop ~2/3 of eligible
  nodes to buy memory this workload does not use, because the peak is one
  **block** (a 256³ uint64 block is 128 MB) rather than one body. Sized to fit
  the smallest node so the job is eligible everywhere.

## Fault policy — asymmetric by design (`ops/_progress.py`)

Task granularity differs per stage, and so does what a failure costs:

| stage | one task is | on failure |
|---|---|---|
| `chunk` / `skel-chunk` | one **block** | raise, abort the stage |
| `assemble` / `skel-fuse` | one **body** | record `failed`, continue |

**Stage 2 isolates** because bodies are independent: skipping a bad one costs
exactly that body, is recorded in the manifest, is logged with its traceback to
`dst/failures.{mesh,skel}.jsonl`, and is retried on the next run. One pathological
body cannot kill hour 11 of a 50k-body run.

**Stage 1 does not**, and that is the point. Stage 2 **aggregates across blocks**,
so a silently skipped block does not leave a hole in one block — it truncates
every body passing through it, and erases outright any body lying wholly inside
it, while the output still looks complete. Stage 2 cannot tell "this body had 3
fragments" from "it had 4 but one block died". A crash you notice; a truncated
neuron you may not. Resume already makes relaunching a chunk run cheap: progress
is recorded per block as results stream back, so only in-flight work is lost.

(This differs from em-volume-tools' conversion, where each block writes an
independent output chunk and isolation is straightforwardly safe. The difference
is the aggregating second stage, which is why the question was deferred from
em-blockrun rather than answered there.)

**The resume trap.** `Manifest.is_done` tests key *presence*, so a recorded
`failed` reads as done and would never be retried — the tasks most needing a
retry would be skipped forever. Ops resume on `_progress.is_complete`, which
excludes `failed`, never on `is_done`.

The driver logs failed body ids per stage and **exits non-zero**, so a scripted
pipeline does not mistake a partial result for a clean one.

## Open questions

- **Multi-component bodies are now possible** (the seam-scale join default keeps
  genuinely-split pieces apart). Decide against real Megaphragma data whether
  that is what you want downstream, and whether `postprocess_dust_nm` should be
  lowered so small bodies are not deleted.
- **Fixed-edge simplification** in stage 1 (preserve block boundaries) — vol2mesh
  `simplify` options TBD.
- **Mesh surface area is measured on the decimated mesh.** `decimation_fraction`
  defaults to 0.1 in stage 1, so `mesh_area_nm2` reflects the simplified geometry,
  not the marching-cubes original. Fine as a relative size measure across bodies;
  do not read it as an absolute membrane area without checking the bias against
  `decimation_fraction=1.0` on a sample.
- Fragment store on **object storage** (currently ceph filesystem).
- Scale/voxel-size wiring: caller passes `seg_spec` at the meshing scale +
  `occupancy_spec` at a coarse scale, with voxel sizes — keeps the op
  format-agnostic. Could auto-derive from source metadata later.
- **`Manifest.is_done` still treats `failed` as done** in em-blockrun itself. We
  work around it locally with `_progress.is_complete`, but em-blockrun's own
  docstring invites a `failed` status, so the next consumer will hit the same
  trap. Worth fixing upstream (an `ignore_statuses` argument) rather than in each
  caller.

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
├── roi.py            # restrict a run to part of the volume, on the global grid
├── scales.py         # per-level shape + true voxel size from source metadata
├── skelcompare.py    # skeletonization comparison harness (kimimaro vs skeletor) + 3D viz
└── ops/
    ├── meshify.py              # two-stage meshing orchestration via em-blockrun
    ├── skeletonize_segments.py # two-stage skeletonization (skel-chunk / skel-fuse)
    ├── index_segments.py       # parallel scan -> per-body metrics DB (bbox/count)
    └── _progress.py            # per-group manifest tallies

examples/run_morpho_slurm.py  # the driver: index -> allowlist -> mesh -> skel
configs/dask-{local,slurm-any}.yaml
scripts/sweep_postprocess.py  # standalone: choose dust/tick thresholds from data
```
