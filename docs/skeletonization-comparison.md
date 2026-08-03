# NeuTu vs kimimaro: skeletons and vertex radii

Specimen 3, scale 2 (32 nm/voxel). Two measurement rounds:

- **2026-07-30** — two bodies, in depth. Source of the mechanism analysis below.
- **2026-07-31** — widened to **12 bodies**, half of them thick, plus the first
  Python port. This round **corrected two conclusions from the first**; see
  [Corrections](#corrections--things-established-as-wrong) before relying on
  anything here.

Figures from the 2-body round: [`images/fig_18052382.png`](images/fig_18052382.png),
[`images/fig_6308993.png`](images/fig_6308993.png),
[`images/fig_6308993_thickness.png`](images/fig_6308993_thickness.png). An HTML
report used to wrap these; it ranked methods by **fill %**, which the Corrections
below establish as not a score, so it was deleted rather than left to mislead — see
`docs/skeletonization-plan.md` "Removed".

Reusable code: `em_seg_morpho/neutu_io.py`, `em_seg_morpho/skelmetrics.py`,
`em_seg_morpho/neutu_trace.py`, and under `scripts/`,
`pick_benchmark_bodies.py` → `export_benchmark_masks.py` → `run_skel_benchmark.py`
(the benchmark pipeline) plus `compare_skeletons_visual.py`.

Reference outputs:
`/path/to/scratch/morpho-skel-benchmark/2026-07-30-specimen3/` (2 bodies)
and `.../2026-07-31-wide/` (12 bodies, masks + NeuTu SWCs + baselines).

---

## Production already emits vertex radii

Worth stating up front, because it is easy to believe otherwise. The published
volume carries a spec-conformant `radius` vertex attribute:

    info:  "vertex_attributes": [{"id":"radius","data_type":"float32","num_components":1}]
    6308993: 9,303 vertices, 1 float32 attr/vertex, radius 32.0–390.6 nm, no -1 sentinels

Radii looked absent before 2026-07-28 because of a *viewer-level* failure, fixed
in `2dc7434`: the declared `vertex_types` attribute was **uint8**, and
neuroglancer rejects the entire layer with "Data type not supported by WebGL:
UINT8". The file was spec-legal and unloadable, so nothing rendered at all.

The open problem is therefore the **values**, not the attribute — see
[radius](#what-the-segmentation-can-support) below.

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
of the nodes. That is the result the plan was built on — **and the 12-body round
below does not reproduce the size of that gap.** Read on before quoting it.

### 12 bodies (2026-07-31)

Bodies chosen by `scripts/pick_benchmark_bodies.py` from the production run's
`metrics.db`, stratified on `max_radius_nm` × `cable_length_nm` so that **half
are thick** (≥ p90 radius), spanning 5 K–1 M voxels. The two bodies above are
pinned into the set; both reproduce their 07-30 numbers to within ~1%, the
difference being a slightly more generous crop.

| | kimimaro production | NeuTu minlen=10 | port |
|---|---:|---:|---:|
| median nodes | 3,195 | 912 | **686** |
| median tip ratio vs NeuTu | — | 1.00× | **1.01×** |
| median cable ratio vs NeuTu | — | 1.00× | 1.07× |
| median centreline dist. to NeuTu | — | — | 0.83 vox |
| median radius error | 0.00 vox | −0.50 vox | 0.00 vox |
| total time, 12 bodies | 1,703 s | **83 s** | 99 s |

The port reproduces NeuTu's branching from **NeuTu's own `scale=1, const=2,
minimalLength=10`** — no tuned constant. Per-body tip ratio 0.87–1.16×, cable
1.03–1.24×. Radii remain exact inscribed radii (median error 0.00 voxels).

**A previous revision of this table reported `const=8` and a 1.04× tip ratio.**
That was withdrawn: it compensated for two bugs (a `uint32` underflow that
disabled branch rejection entirely, and a missing loop progress guarantee), and
with those fixed `const=8` deletes real cable. The full account is in step 5 of
`skeletonization-plan.md`, kept because the wrong conclusion was well-evidenced
and survived a 12-body validation.

**Fill is deliberately absent from that table** — see below.

### Read fill and spill as diagnostics, not as scores

**Fill is confounded by branch count** — more branches fill more, so it rewards a
skeletonizer that invents neurites. This was learned the hard way: the port was
reported as "+3.2 points of fill, ahead on 10 of 12 bodies", and that framing is
**withdrawn**. Per unit cable NeuTu is *more* fill-efficient (0.46 vs 0.21 fill
per 1k cable on body 45813451), and the port got its higher number by carrying
5–10× the tip count. On body 4978519 the port was strictly worse: 10× the tips
and *less* fill.

**Spill is not an error signal here either.** The segmentation is dense — every
voxel belongs to some segment, and 0.0% of spill lands on background — and many
segments are one neuron incorrectly split, so a tube crossing into a neighbour is
often reclaiming its own neurite. A `spill_by_neighbour_size` metric was built to
test that — grading spill by how large the neighbour entered is, on the assumption
that most false splits are small fragments — and **it could not settle it**: moving
the fragment/large boundary one bin swings "spill into large neighbours" from 13%
to 1%, because nearly every neighbour involved sits in that band. It and the
`nbrsize` arrays that fed it were deleted; see `docs/skeletonization-plan.md`
"Removed".

**Optimise against `skelmetrics.agreement` instead** — bidirectional centreline
distance to a reference skeleton, plus node/tip/cable ratios. There is not enough
ground truth to do better, so NeuTu is the reference because it behaves well
enough, not because it is right.

**NeuTu's fill advantage is real but modest and not uniform**: +3 points at the
median, ahead on 9 of 12 bodies, and *behind* on body 79347718 (55% vs 66%). The
two-body result (73% vs 39%) is not representative.

**Node economy is the large, consistent win — and it scales with thickness**,
which is why thick bodies are the ones worth caring about:

| band | body | kimimaro nodes | NeuTu nodes | ratio |
|---|---|---:|---:|---:|
| thick-huge | 45892915 | 5,633 | 442 | **12.7×** |
| pinned (thick) | 6308993 | 11,867 | 1,383 | 8.6× |
| thick-medium | 45813451 | 1,896 | 229 | 8.3× |
| thick-small | 4978519 | 288 | 42 | 6.9× |
| thin-large | 43230132 | 5,863 | 3,858 | 1.5× |
| thin-medium | 79347718 | 2,108 | 1,622 | 1.3× |

Median 3.0× overall. The thin bodies barely differ; the thick ones differ by an
order of magnitude.

**Node economy** is the other real difference: NeuTu spends ~1 node per 2.9 voxels
of cable where kimimaro spends ~1 per 1.5.

**The radius conventions differ**, and the 2-body round concluded that was "the
whole story on radius". ~~It is not~~ — see the correction below. kimimaro sets
`radii = DBF[vertex]`, the exact inscribed radius, correct at the node and
systematically small wherever a cross-section isn't round.

NeuTu's radii are **bimodal**, and the median goes the *opposite* way from what
the z-slice panels suggest. Across 12 bodies, against each mask's own distance
transform (`skelmetrics.radius_vs_edt`, in voxels):

| | NeuTu median error | NeuTu frac >2 vox over | kimimaro |
|---|---:|---:|---|
| 8 of 12 bodies | **−0.50** | 0–3% | 0.00 / 0% by construction |
| thick bodies | +0.09 … +0.77 | 7–27% | 0.00 / 0% |

The **−0.50** is `AdjustedDistanceWeight` (`gui/zstackskeletonizer.cpp:82`,
`max(0.1, √v − 0.5)`) landing exactly as written: NeuTu's typical node reports a
radius *half a voxel smaller* than the inscribed sphere. What inflates the
headline maximum is a minority tail on thick bodies:

| body | NeuTu max radius | largest sphere that fits | nodes >2 vox over |
|---|---:|---:|---:|
| 45892915 | 852 nm | 354 nm | 27% |
| 6308993 | 720 nm | 359 nm | 12% |
| 4978519 | 336 nm | 288 nm | 17% |

Reporting 2.4× the largest radius the mask can support is a real error if these
are reused for measurement, and it is concentrated exactly where the segmentation
is thickest. Where it comes from is **not** the radius formula — that subtracts
half a voxel — so it must enter at the region-sampling or resampling step, which
is worth pinning down before porting either (steps 2–3 of the plan).

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

- **"NeuTu's larger radii are what fills better" is wrong** (established
  2026-07-31, 12 bodies). NeuTu's *median* radius error is **−0.50 voxels** — it
  reports radii **smaller** than the inscribed sphere, not larger, because
  `AdjustedDistanceWeight` subtracts half a voxel exactly as its source says.
  Only a 0–27% tail overshoots. Two independent lines of evidence say fill is
  driven by **path coverage and node placement**, not radius: (a) NeuTu fills
  more than kimimaro while reporting a smaller median radius, and (b) the Python
  port reaches **92% fill using kimimaro's exact inscribed radii**, well past
  NeuTu's 71%. Do not tune radius to chase fill — it is the wrong lever, and it
  would trade a measurable quantity for one that is not.

- **A single-root TEASAR trace covers exactly one connected component.** Not a
  claim from the earlier round, but the same class of silent error and it cost
  real time. The parent field and the rolling-ball invalidation are both confined
  to the root's component, and the root comes from `first_label` — whichever
  voxel is first in memory order. On body 6308993 that landed on a component
  holding 3.06% of the voxels and the trace reported 3% coverage. These bodies
  are genuinely fragmented, so **any** skeletonizer added here must iterate
  components; `neutu_trace.skeletonize` does, and
  `test_all_connected_components_are_traced` holds it to that.

- **`1/(1 + r²)` makes background the cheapest voxel in the volume.** NeuTu never
  puts background in its graph, so it has no such hazard; `dijkstra3d` takes a
  weight field with no separate mask, and `1/(1 + inf²)` is **0**. Paths then cut
  straight through empty space. kimimaro is safe only incidentally — its
  `(1 − DBF·M)^exponent` sends background to `+inf`. Any new cost function has to
  set background to `inf` explicitly.

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

- **One specimen, 12 bodies.** Wide enough to have overturned two conclusions
  from the 2-body round, not wide enough to be a statistical claim. The set is
  built as **regression fixtures** — deliberately weighted toward thick,
  problematic bodies — so its medians are not the medians of the dataset.
- **No ground truth.** Everything is scored against imperfect masks. "Fill" is a
  pragmatic target for visualization and the wrong one for measurement.
- **The ranking has now reversed twice** — once when sphere-stamping was fixed,
  once when the radius conclusion was corrected. Both times the error was in the
  *model of the question*, not the data. Treat any single-round conclusion here
  as provisional.
- **kimimaro relaxed is still unmeasured on the large bodies.** Deliberately
  dropped from the 12-body round: it exists to answer whether NeuTu's advantage
  is robust, and 231 s on a 200 K-voxel body made it the dominant cost. If the
  ranking is ever in question again, this is the missing arm.
- **Everything is at scale 2.** No claim about finer scales, where the docs'
  fragmentation finding says component counts rise sharply.
