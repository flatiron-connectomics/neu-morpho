# NeuTu vs kimimaro: skeletons and vertex radii

Measured 2026-07-30 on specimen 3, scale 2 (32 nm/voxel). Two bodies, identical
input masks, both tools scored on the same question.

**Visual version: [`skeletonization-comparison.html`](skeletonization-comparison.html)**
— same findings with the rendered figures inline; open it in a browser. Regenerate
with `scripts/build_comparison_report.py` (add `--embed` for a single self-contained
file to share).

Reusable code from that session is now `em_seg_morpho/neutu_io.py`,
`em_seg_morpho/skelmetrics.py` and `scripts/compare_skeletons_visual.py`.
Reference outputs:
`/path/to/scratch/morpho-skel-benchmark/2026-07-30-specimen3/`.

---

## What "correct" means here

**Fill, not inscribe.** These segmentations are imperfect, so the mask is *not*
ground truth and agreement with its distance transform is not the target. The
operative definition of a good radius in this project is the one that makes the
rendered tube fill the segment, because the immediate goal is good-enough
visualization.

That definition is pragmatic, not rigorous — there is no calibrated notion of
correctness behind it, and it is the *wrong* definition if these radii are ever
reused for volume or surface-area measurement. Whenever this trade-off comes up,
say which definition you mean before optimising against either.

## How it was measured

Rasterise the skeleton as a **tapered tube** — connected frusta between parent and
child, which is what an SWC viewer draws — and compare to the mask:

- **fill** — fraction of segment voxels inside the tube (higher better)
- **spill** — fraction of tube volume outside the segment (lower better)

Neither tool reaches 100% and none can: a chain of circular cross-sections cannot
fill an irregular one. Read the numbers against each other, not against perfection.

## Results

### body 18052382 — 206,659 voxels, no soma

| Tool | Setting | Nodes | Fill | Spill | Max radius | Time |
|---|---|---:|---:|---:|---:|---:|
| NeuTu | minlen=2 | 2,193 | **78%** | 14% | 442 nm | 1.6 s |
| NeuTu | minlen=10 | 1,759 | 68% | 14% | 442 nm | 1.3 s |
| NeuTu | minlen=40 | 1,329 | 50% | 14% | 442 nm | 1.3 s |
| kimimaro | production — scale 1.5, const 150 nm | 4,417 | 68% | 8% | 317 nm | 25 s |
| kimimaro | relaxed — scale 1.0, const 2 vox | 10,177 | **89%** | 13% | 317 nm | 231 s |

### body 6308993 — 1,016,526 voxels, nucleated

| Tool | Setting | Nodes | Fill | Spill | Max radius | Time |
|---|---|---:|---:|---:|---:|---:|
| NeuTu | minlen=10 | 1,431 | **73%** | 11% | 543 nm | 16 s |
| kimimaro | production — scale 1.5, const 150 nm | 5,705 | 39% | 4% | 345 nm | ~9 min |

On the larger arbor NeuTu fills nearly twice as much of the segment from a quarter
of the nodes. That is the result the plan is built on.

**Node economy** is the other real difference: NeuTu spends ~1 node per 2.9 voxels
of cable where kimimaro spends ~1 per 1.5.

**The radius conventions differ, and that is the whole story on radius.** kimimaro
sets `radii = DBF[vertex]` — the exact inscribed radius, correct at the node and
systematically small wherever a cross-section isn't round. NeuTu's are larger, which
fills better. Visible in the z-slice panels: NeuTu's circle fills the cross-section,
kimimaro's leaves a rim all the way around.

The counterweight: NeuTu reports a **543 nm** max radius on body 6308993 where the
largest sphere fitting anywhere in that mask is **345 nm**, and ~13% of its nodes are
>2 voxels above the local distance-transform value. Fine for rendering; a real error
if reused for measurement.

## What the segmentation can support

Method-independent, from the masks themselves:

| | body 6308993 | body 18052382 |
|---|---|---|
| volume | 33.3 µm³ | 6.8 µm³ |
| bbox | 35 × 20 × 21 µm | 9 × 10 × 8 µm |
| max inscribed radius | 345 nm (10.77 vox) | 317 nm (9.90 vox) |
| median process radius | 72 nm (2.24 vox) | 55 nm (1.73 vox) |
| enclosed cavities | 693 voxels total, largest ≈3 vox equiv. radius | — |

Both are thin-process arbors. The nucleus **is** part of the segment (not a cavity),
per the person who looked at the EM. The soma region nonetheless carries no thick
core in the mask — max inscribed radius anywhere is 345 nm — so no skeletonizer can
recover a soma-scale radius from it. That is a statement about segmentation quality,
not anatomy. `scripts/compare_skeletons_visual.py` renders this thickness map first
so it can be checked before blaming a skeletonizer.

## How NeuTu differs, mechanically

All in the NeuTu tree; entry point `neurolabi/gui/zstackskeletonizer.cpp:490`. It is
TEASAR, the same family as kimimaro (there is even a `neurolabi/c/teasar.c`), so the
differences are in five specific places:

1. **Path cost is local, not normalized.** `Stack_Voxel_Weight_I`
   (`c/tz_stack_graph.c:205`) is `d/(1+r₁²) + d/(1+r₂²)` on the *squared* EDT.
   kimimaro's `compute_pdrf` (`kimimaro/trace.py:315`) is
   `pdrf_scale·(1 − DBF/dbf_max^1.01)^exponent`, normalized by the object's global
   max radius. **Untested and probably irrelevant here** — the effect needs
   `dbf_max ≫ neurite radius`, and these masks span only ~4.8×.
2. **Radius-adaptive node placement.** `createSwcByRegionSampling`
   (`gui/zswcgenerator.cpp:196`): sort path voxels by decreasing radius, greedily drop
   any voxel inside a kept larger voxel's ball. Nodes end up spaced by local radius.
3. **Radius-aware simplification to a fixpoint.** `ZSwcResampler::optimalDownsample`
   (`gui/swc/zswcresampler.cpp:90`) — drop a node if its ball is contained in a
   neighbour's, or if interpolating parent↔child reproduces it in both position and
   radius. A radius-aware Douglas–Peucker.
4. **Tighter invalidation, length-based pruning.** Invalidation radius is `EDT + 2`
   voxels (`gui/zspgrowparser.cpp:344`) vs kimimaro's `1.5·DBF + 4.7`; branches are
   rejected by *un-invalidated geodesic length* < `minimalLength` rather than by
   post-hoc tick removal. This early termination is why NeuTu takes 1.3 s where
   kimimaro at comparable invalidation takes 231 s.
5. **Radius convention** — `AdjustedDistanceWeight`
   (`gui/zstackskeletonizer.cpp:82`) is `max(0.1, √v − 0.5)`.

## Corrections — things established as wrong

Recorded so they are not re-derived.

- **Sphere-stamping reversed the ranking.** The first version of the fill metric
  stamped isolated spheres at vertices instead of sweeping capsules along edges. That
  penalised sparse-node methods and produced a table showing kimimaro ahead. It is
  wrong. `skelmetrics.sweep` exists to prevent a repeat — always pass `edges`.
- **Do not port NeuTu's `−0.5` radius correction.** It was recommended early as a
  cheap win. At these radii it removes 42–58% of ball volume (`r=2` keeps 42%,
  `r=1.5` keeps 30%), and it accounts for ~9 points of fill deficit. Harmful here.
- **`const = 150 nm` is not aggressive.** At 32 nm/vox that is 4.69 voxels, not the
  18.75 assumed from the `(8,8,8)` placeholder in `SkeletonConfig.anisotropy`. The
  earlier "erases every side branch" warning was wrong.
- **"This body has no soma" overstepped.** It was inferred from mask geometry alone.
  The correct statement is that the *mask* has no thick region.

## Operating the NeuTu CLI

Built in a separate conda env (`managed_neutu`); it cannot be installed alongside
this package — the conda recipe pins hdf5 1.8.18, jansson 2.7, libpng 1.6.28 and
libdvid-cpp, and the SWIG bindings predate Python 3.12. Shell out instead; see
`em_seg_morpho.neutu_io.run_neutu`.

```
neutu --command --config cfg.json body.sobj --skeletonize
```

- **Argument order is load-bearing.** genelib's `Process_Arguments` **segfaults**
  when a positional input is followed by any valued option — `[<input:string> ...]`
  (`gui/zcommandline.cpp:1658`) is greedy. Valued options must precede the
  positional. Bare flags after it are fine. `-o out.swc` after the input crashes;
  set `output` in the config JSON instead.
- **Bounding-box limit.** Aborts above `ONEGIGA` = 1,073,741,824 voxels of *bbox*,
  not voxel count. Body 6308993 at 449 M cleared it with 2.4× headroom.
- **Radius saturation.** The distance map is uint16 *squared* distance, so radii clip
  at ~256 voxels. Irrelevant at 32 nm/vox here; recheck if you go coarser.
- **Cost.** 16 s and 7.8 GB peak on the 449 M-voxel bbox.
- **Licence.** NeuTu is GPL. Shelling out to a separate binary is fine; linking is not.

## Scope — what this does not establish

- **Two bodies, one specimen**, both thin-process arbors (median diameter
  110–140 nm). Nothing here is tested on genuinely thick structure.
- **No ground truth.** Everything is scored against imperfect masks.
- **The ranking already reversed once**, when a modelling error in the metric was
  fixed. Treat it as provisional until the benchmark is wider — see step 0 of
  `skeletonization-plan.md`.
- **kimimaro relaxed was never run on body 6308993** (production alone took ~9 min on
  that bbox), so the best-coverage setting is unmeasured on the larger body.
