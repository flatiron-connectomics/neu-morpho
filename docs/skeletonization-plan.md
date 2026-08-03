# Plan: NeuTu-quality skeletons in em-seg-morpho

Background and evidence: `docs/skeletonization-comparison.md`. Read that first —
in particular the "Corrections" section, which lists conclusions already established
as wrong.

**Status. All five steps are done, and the output now matches NeuTu's.**
`em_seg_morpho/neutu_trace.py` (tracing) plus `em_seg_morpho/swc_simplify.py`
(node reduction), scored by `skelmetrics.agreement` over the 12-body benchmark:

| median ratio, port : NeuTu | |
|---|---:|
| tips | **1.04×** |
| cable | 1.06× |
| nodes | 0.75× |
| centreline distance A→B / B→A | 0.82 / 0.78 voxels |

Sub-voxel centreline agreement, equivalent branching, fewer nodes. Against
kimimaro production — what ships today — it is roughly **2× fewer nodes and
10× faster**.

**Do not read the fill/spill numbers in the comparison doc as a score.** Fill is
confounded by branch count, so it rewards inventing neurites; an earlier revision
of this plan claimed a fill win that was exactly that artefact. See the
Corrections section there.

**One mechanism is still not NeuTu's** — target selection — and it is papered
over by a tuned constant (`const=8` rather than NeuTu's 2). See step 5.

Step 4 closed with no code. What remains is **wiring it into the pipeline** —
see "Integration" at the bottom, which is not a small change.

**Step 5 was the whole node-count story, and this plan under-rated it** — it was
filed as "only if runtime demands it". It is the opposite: it is the single
largest lever on node count, and no amount of post-hoc simplification substitutes
for it. See step 5 for the measurement.

**Goal.** Skeletons that fill the segment well, with few nodes and radii good enough
for visualization. Two things have been re-scoped since this was written:

- **Radius attributes already ship.** Production emits a spec-conformant float32
  `radius` per vertex and has since `2dc7434`. The goal is better *values*, not a
  new attribute.
- **Radius is not the fill lever.** The original framing ("NeuTu's larger radii
  fill better") is measured wrong; NeuTu's median radius is *smaller* than the
  inscribed sphere. Fill comes from path coverage and node placement. Tuning
  radius to chase fill trades a measurable quantity for one that is not.

**Two routes.** Preferred: reimplement NeuTu's method in Python. Fallback: shell out
to the NeuTu binary from its own environment as an optional pipeline stage
(`em_seg_morpho.neutu_io.run_neutu` already does this).

---

## Why the reimplementation is mostly parameterization

This is the finding that makes the preferred route cheap. `kimimaro/trace.py:148-176`
is structured as:

```python
PDRF    = compute_pdrf(dbf_max, pdrf_scale, pdrf_exponent, DBF, DAF, DAF[target])
parents = dijkstra3d.parental_field(PDRF, root, voxel_graph=voxel_graph)
paths   = compute_paths(root, labels, DBF, target_finder, parents, scale, const, ...)
skel.radii = DBF[verts[::3], verts[1::3], verts[2::3]]
```

Mapping NeuTu onto it:

| NeuTu behaviour | Change |
|---|---|
| cost `d/(1+r²)` | **one line** — `PDRF = 1.0 / (1.0 + DBF**2)` |
| invalidation `EDT + 2 vox` | **zero lines** — already expressible as `scale=1.0, const=2` |
| radius convention | one line at `skel.radii` |
| `minimalLength` early termination | edit inside `compute_paths`, which is plain Python |
| radius-adaptive placement + resampler | new numpy post-processing, tool-independent |

**The Dijkstra never leaves C++.** `dijkstra3d` and `skeletontricks` do the heavy
work unchanged, so runtime should be kimimaro's. NeuTu's length-based termination may
make it *faster* than kimimaro is today at comparable invalidation (1.3 s vs 231 s).
Confirmed: 164 s vs kimimaro's 1,703 s over the 12-body set.

~~Vendor `kimimaro.trace.trace`~~ — **reimplemented instead.** Importing and
calling it is impossible (it computes the cost inline with no hook, so injecting
one means monkeypatching `compute_pdrf` process-globally), but copying it turned
out to be the wrong alternative: once soma mode and `fix_branching` are dropped —
NeuTu has neither — what remains is ~40 lines of orchestration around the C++
primitives, which are *imported* either way. Reimplementing gets the same
readable-and-pinned property with no upstream-drift obligation. It does not change
the licence position: `kimimaro` and `dijkstra3d` are GPL-3.0-or-later and are
imported regardless, so em-seg-morpho is a GPL-combined work.

---

## Step 0 — widen the benchmark first ✅ done 2026-07-31

12 bodies (target was ~10), **half of them thick**, at
`/path/to/scratch/morpho-skel-benchmark/2026-07-31-wide/`: masks, NeuTu
reference SWCs, kimimaro baselines, `results.json`, and `bodies.json` recording
the selection. Pipeline: `scripts/pick_benchmark_bodies.py` →
`export_benchmark_masks.py` → `run_skel_benchmark.py`.

Two changes from the plan as written, both deliberate:

- **kimimaro *relaxed* was dropped.** It exists to test whether NeuTu's advantage
  is robust; that question was settled independently, and relaxed was the
  dominant cost (231 s on a 200 K-voxel body, never completed on a 1 M one).
- **The set is weighted toward thick bodies, not representative.** These are
  regression fixtures for catching porting bugs, and thick bodies are where the
  tools diverge most — the node-count ratio runs 12.7× on thick bodies and 1.3×
  on thin ones.

It earned its keep: it overturned the radius conclusion, and it caught the
connected-component bug in step 1 that unit tests on synthetic tubes did not.

## Step 1 — NeuTu-style cost, everything else kimimaro ✅ done 2026-07-31

`em_seg_morpho/neutu_trace.py`, with `tests/test_neutu_trace.py` (11 tests).

The `parental_field` question the plan said to confirm rather than assume: **the
per-voxel substitution is exact here.** Checked against an independent, slow
Dijkstra using NeuTu's symmetric edge cost `d·[f(v₁) + f(v₂)]` on straight and
bent synthetic tubes — the paths come out **bit-identical**, ratio 1.000000
(`test_per_voxel_weights_match_neutu_edge_cost`). The telescoping argument holds
despite mixed 3D step lengths.

Result, as predicted: fill well past kimimaro, node count still dense.

| | kimimaro production | NeuTu minlen=10 | step 1 |
|---|---:|---:|---:|
| median fill | 68% | 71% | **92%** |
| median nodes | 3,195 | **912** | 7,947 |
| total time, 12 bodies | 1,703 s | 83 s | 164 s |

**Two traps found, neither visible without a test.** Both are written up in the
comparison doc's Corrections; in short: `1/(1+r²)` makes background the *cheapest*
voxel in the volume (`1/(1+inf²)` = 0) so paths cut through empty space; and a
single-root trace covers exactly one connected component, which on body 6308993
meant 3% coverage.

**A unit trap to preserve.** `1/(1 + r²)` is not scale-invariant — the `1` is one
voxel squared. Feed it a DBF in nm at 32 nm/voxel and the same body's weights span
a factor of 157 instead of 80. `neutu_trace` therefore works in voxels and raises
rather than accepting an anisotropy it cannot honour. NeuTu's EDT is not
anisotropy-aware either, so there is no faithful anisotropic version to port.

## Step 2 — radius-adaptive node placement ← **the whole remaining gap**

Port `createSwcByRegionSampling` (`gui/zswcgenerator.cpp:196`): sort path voxels by
decreasing radius, greedily drop any within a kept larger voxel's ball. NeuTu's
implementation is O(n²); use a KD-tree.

This is **tool-independent post-processing** — it applies to any SWC, so it is worth
having even if step 1 is abandoned. Expected: ~2× fewer nodes at equal fill.

**Done 2026-07-31** — `swc_simplify.region_sample`, with
`tests/test_swc_simplify.py`. **7,947 → 3,304 median nodes (2.4×, up to 5.0× on
thick bodies) for 1 point of fill.** Better than the ~2× expected, and the
reduction is largest exactly where node counts hurt: body 45892915 went 28,180 →
5,598.

Two departures from NeuTu, both forced by our input being a tree:

- `createSwcByRegionSampling` consumes a `ZVoxelArray` — one traced path — and
  re-emits survivors as a **linear chain**. Ours is already a tree, so chaining
  would destroy branch structure. The kept nodes inherit the original
  connectivity instead: kept neighbours stay joined, and each clump of dropped
  nodes becomes a star joining the kept nodes it used to link. Tests assert the
  result stays connected, acyclic, and keeps its branch points.
- NeuTu's suppression is O(n²) against every larger kept node. Marking **forward**
  from each kept node through a KD-tree is equivalent — a node is dropped iff some
  kept node of larger-or-equal radius contains it — and is what makes 50,000
  nodes tractable.

**Where NeuTu's over-large radii enter is still unidentified — but it is not
here.** This pass *selects* nodes and never interpolates, so radii stay exactly
`DBF[vertex]`: measured median error 0.00 voxels, 0% of nodes over by >2, max
radius equal to the largest sphere the mask admits. NeuTu's inflation must come
from something else in its pipeline; we do not reproduce it and should not.

## Step 3 — radius-aware simplification ✅ done 2026-07-31

`swc_simplify.optimal_downsample`, a faithful port of
`ZSwcResampler::optimalDownsample` (`gui/swc/zswcresampler.cpp:90`) including
`suboptimalDownsample`, `isInterRedundant`, and the `isWithin` /
`hasSignificantOverlap` predicates, iterated to a fixpoint. NeuTu's defaults
carry over: `m_radiusScale = 1.2`, `m_distanceScale = 2.0`.

**3,304 → 2,480 median nodes (a further 1.3×) for ~1 point of fill.** Smaller than
step 2, as expected — step 2 has already removed the redundancy this pass targets.
It earns its place on the long thin arbors, where it is the larger of the two
(body 42074060: 8,842 → 6,125).

The one thing to know if this is ever revisited: NeuTu's merge options
(`MERGE_W_PARENT` / `MERGE_W_CHILD` / `MERGE_WEIGHTED_AVERAGE`) do move node
positions, unlike step 2 — so this is the pass that *could* drift radii away from
the distance transform. Measured, it does not: radii still show 0.00 median error
and 0% over by >2 voxels.

## Step 4 — radius convention ✅ closed with no code

This step assumed NeuTu's radii are larger and that this is what fills better.
Both halves are measured wrong (see Corrections). NeuTu's *median* radius error
is **−0.50 voxels** — smaller than the inscribed sphere — and step 1 reaches 92%
fill using kimimaro's exact inscribed radii, well past NeuTu's 71%.

So there is no fill argument for changing the convention, and the inscribed
radius has the large advantage of meaning something: it is the largest sphere
that fits, which survives being reused for measurement. **Recommendation: keep
`radii = DBF[vertex]` and do not sweep a scale factor.**

The remaining worry was that steps 2–3 would *silently* inherit NeuTu's
inflation when they merge nodes. **Measured after step 3, on every body: median
error 0.00 voxels, 0% of nodes over by more than 2, and a maximum radius equal to
the largest sphere the mask admits.** Step 2 selects rather than interpolates, and
step 3's merges do not drift. So this step is finished, and the radii the pipeline
would publish are exact inscribed radii — reusable for measurement, not just
rendering.

**Do not port the `−0.5` correction** — established harmful, and now confirmed as
the source of NeuTu's negative median error.

## Step 5 — length-based branch termination ✅ done 2026-07-31 — **the main lever**

Filed here as "only if runtime demands it". That was wrong, and it cost a detour
through steps 2–3 before the diagnosis was made. **This is the single largest
determinant of node count.**

`neutu_trace._uninvalidated_length` + the `min_length` parameter, ported from
`ZSpGrowParser::pathLength(idx, masked=true)` and the
`if (length < minLength) isPathAvailable = false` test at
`gui/zspgrowparser.cpp:317`.

### The diagnosis, because the obvious answer was wrong

After steps 2–3 the port sat at 2.4× NeuTu's nodes, and the natural assumption
was that it placed them too densely. **It did not.** Per 100 voxels of cable:

| body | NeuTu | port | |
|---|---:|---:|---|
| 6308993 | 16.6 | 10.3 | port sparser |
| 18166095 | 43.3 | 19.1 | port sparser |
| 45813451 | 14.3 | 9.1 | port sparser |

The gap was **entirely cable** — 3–13× more of it, and **5,022 tips against
NeuTu's 116** on body 6308993. Node economy was never the problem; branch count
was. Anyone re-opening this should measure `nodes / cable` and `tip count` first.

### Why the criterion has to be un-invalidated length

Pruning short twigs geometrically afterwards is **strictly dominated** — the two
approaches, swept to comparable node counts on real bodies:

| body | post-hoc twig pruning | in-extraction `minimalLength` | NeuTu |
|---|---|---|---|
| 45813451 | 198 nodes → 65% fill | 352 → **77%** | 229 → 74% |
| 35668783 | 264 → 45% | 534 → **72%** | 272 → 64% |
| 45892915 | 709 → 64% | 1,611 → **80%** | 442 → 79% |

The test is on *new territory reached*, not geometric length: a short branch into
unclaimed volume survives, a long one shadowing already-covered ground does not.
Length alone cannot express that, which is why `swc_simplify.prune_twigs` exists
but is off by default.

### It is not a tuning knob

`min_length` of 5, 10, 20 and 40 give near-identical output (352/352/352 nodes on
body 45813451). Un-invalidated length is bimodal — a branch either covers real
volume or nearly none — so the threshold separates cleanly instead of trading
off. NeuTu's default of 10 is fine; do not sweep it.

### Target selection is still not NeuTu's, and that is the open item

NeuTu picks the next target by **maximum un-invalidated length**
(`extractLongestPath`); we pick max-DAF via kimimaro's `CachedTargetFinder`. Two
consequences, and the second is the one that cost real effort to find:

**The global stop is unusable with max-DAF.** NeuTu stops on the first short
branch (`isPathAvailable = false`) because its selector guarantees the best
remaining branch is the one it just tested. Ours does not, and stopping anyway
truncates live arbor — B→A p90, i.e. what we fail to cover, on three bodies:

| body | reject-and-continue | stop after 3 misses |
|---|---:|---:|
| 35668783 | 2.40 | **74.71** |
| 16104493 | 2.17 | 17.03 |
| 45813451 | 3.11 | 14.05 |

So `patience` defaults to `None` (off).

**Selection order, not the invalidation radius, drives branch count.** With
NeuTu's own `const=2` the port carried a **4.07× median tip ratio**. That is not a
porting error in the invalidation — NeuTu's radius really is `EDT + 2`
(`maskExpansionRadius = 2.0`, `DistanceWeight(v) = sqrt(v)`, both confirmed in
source). It is that NeuTu extracts the largest new branch first, so one
invalidation ball consumes territory that otherwise resurfaces as many small
branches.

**The shipped workaround is `const = 8`**, which buys per path what NeuTu's
ordering buys by choosing well. Validated on all 12 bodies:

| median ratio, port : NeuTu | const=2 | const=8 |
|---|---:|---:|
| tips | 4.07× | **1.04×** |
| cable | — | 1.06× |
| nodes | — | 0.75× |
| B→A p90 | 2.16 | 2.30 |

Per-body tip ratio lands in 0.87–1.23×. See `INVALIDATION_CONST` for the two
caveats — the constant is in voxels so it is tied to `skeleton_scale`, and at
256 nm it can invalidate a genuinely separate neurite running parallel, which
shows up as B→A p90 roughly doubling on the two largest thick bodies.

**Whether to do the rewrite** (target selection by un-invalidated length, via
pointer-jumping over the parent field, after which the global stop becomes sound)
now rests on removing the tuned constant and recovering that thick-body p90 — not
on branch count, which `const=8` already matches.

## Fallback — the NeuTu plugin

If steps 1–3 stall, `em_seg_morpho.neutu_io.run_neutu` is ready. Costs: a GPL binary
out of process, a second conda environment, per-body subprocess and file I/O, and the
argument-order workaround. Parallelises fine per body (disBatch on Rusty).

Keep the NeuTu reference SWCs from step 0 either way — they are how the
reimplementation gets validated.

---

## Integration — the remaining work

Not yet done, and not a drop-in. Production skeletonization is **block-first**:
stage 1 skeletonizes each 256³ block with kimimaro's `fix_borders=True` so
fragments meet at seams, stage 2 fuses each body's fragments
(`join_close_components` + `postprocess`). `neutu_trace.skeletonize` is
**whole-body** — it computes an EDT and a parent field over one connected
component at a time, which is exactly the per-body cropping the block-first
design exists to avoid (see `SkeletonConfig`: the OOM risk is bounding-box
*extent*, not voxel count).

Two routes, in increasing order of work:

1. **`swc_simplify` alone, in stage 2.** The reduction passes are
   tool-independent — they take any `(vertices, radii, edges)`. Applying them to
   the *current* kimimaro fusion output would capture a large part of the node
   win with no change to tracing and no memory risk. Cheapest useful step.
2. **Replace the tracer.** Needs a block-first story for `min_length`, and that
   is the hard part: un-invalidated length is measured against the *whole*
   component, so a branch crossing a block boundary looks short in each block
   independently. Rejecting per block would delete exactly the long-range cable
   the pipeline cares most about. Either trace per body where extent allows, or
   defer branch rejection to stage 2 on the fused skeleton — where it is
   post-hoc, and therefore the weaker geometric form.

## Watch out for

- **Do not optimise fill or spill.** Fill rewards inventing branches — the thing
  we are trying to remove — and on a dense segmentation spill cannot distinguish
  reclaiming a false split from trespassing. Optimise `skelmetrics.agreement`
  against NeuTu. Both mistakes were made in this branch; see the comparison doc.
- **Always pass `edges` to `skelmetrics`.** Scoring vertex spheres instead of swept
  capsules reversed the tool ranking once already.
- **Don't port the `−0.5` radius correction.** Established harmful at these radii.
- **Iterate connected components.** A single-root TEASAR trace covers exactly one,
  and these bodies are genuinely fragmented. It fails as a plausible-looking low
  coverage number, not as an error.
- **Set background to `inf` in any new cost function.** `dijkstra3d` takes a
  weight field with no mask, so a cost that sends background to 0 makes empty
  space the cheapest thing to cross.
- **`anisotropy=(8,8,8)` in `SkeletonConfig` is a placeholder.** Scale 2 is
  32 nm/voxel. `const` is in nm; get the conversion right before comparing settings.
  (The driver overrides it from the source metadata — the published radii are
  correct — but the default on its own would be wrong by 4×.)
- **NeuTu's EDT is not anisotropy-aware** (anisotropy enters only via Dijkstra step
  lengths). Fine for isotropic data; a real problem otherwise.
- **The mask is not ground truth.** Check the thickness map before attributing a bad
  radius to the skeletonizer.
