# Plan: NeuTu-quality skeletons in em-seg-morpho

Background and evidence: `docs/skeletonization-comparison.md`. Read that first —
in particular the "Corrections" section, which lists conclusions already established
as wrong.

**Status. All five steps are done, at NeuTu's own parameter values.**
`em_seg_morpho/neutu_trace.py` (tracing) plus `em_seg_morpho/swc_simplify.py`
(node reduction), scored by `skelmetrics.agreement` over the 12-body benchmark:

| median ratio, port : NeuTu | |
|---|---:|
| tips | **1.01×** (0.87–1.16 per body) |
| cable | 1.07× |
| nodes | 0.75× |
| centreline distance A→B / B→A | 0.83 / 0.79 voxels |

Sub-voxel centreline agreement and matched branching, from **NeuTu's own
`scale=1, const=2, minimalLength=10`** — no tuned constant, no target-selection
rewrite. Against kimimaro production, what ships today, roughly **2× fewer nodes
and 10× faster**.

**Two withdrawn conclusions**, both from this branch, both worth reading before
trusting any number here:

- **Fill/spill are not scores.** Fill is confounded by branch count, so it rewards
  inventing neurites. An earlier revision claimed a fill win that was exactly that
  artefact. See the comparison doc's Corrections.
- **The `const=8` "compensation" was masking two bugs.** An earlier revision
  defaulted to `const=8` and argued at length that it compensated for
  target-selection ordering. Both the number and the mechanism were wrong — see
  step 5.

**No mechanism is now known to be missing.** Target selection is still max-DAF
rather than NeuTu's max-un-invalidated-length, but at these settings that no
longer costs measurable agreement, so the rewrite has no justification. What
remains is **integration** (see below).

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

### ~~Selection order drives branch count~~ — WITHDRAWN, it was two bugs

An earlier revision recorded that NeuTu's own `const=2` gave a **4.07× median tip
ratio**, concluded that selection *ordering* was responsible, and shipped
`const=8` as compensation with a table showing it reaching 1.04×. **All of that
was an artefact.** Two bugs, both silent:

1. **`_uninvalidated_length` measured `uint32` paths with `np.diff`**, which
   underflows on any decreasing coordinate and returned ~2.7e11. Every
   `>= min_length` test passed, so **branch rejection never ran at all**. The unit
   test missed it by using an int64, monotonically-increasing path.
2. **The extraction loop had no progress guarantee.**
   `roll_invalidation_ball_inside_component` erases voxels *around* a path but
   never the path's own voxels (measured: 307 of 307 left valid), and
   `CachedTargetFinder` does not remember what it returned — so `find_target`
   handed back the same voxel forever and the loop re-extracted one identical path
   until it hit `max_paths`. kimimaro escapes this only because its default
   `fix_branching=True` rewrites `parents` along each path; a `parental_field` port
   has no such escape and must retire the path explicitly.

With both fixed, **NeuTu's `const=2` reproduces NeuTu** (tip 1.01× median) and
`const=8` over-prunes (tip 0.40–0.50, real cable deleted). The default is back to
`NEUTU_CONST`, with a test pinning it and a comment saying to look for a
reintroduced bug if it ever needs raising.

The `patience` finding above still stands — it was measured before these fixes but
re-checked after, and a global stop on max-DAF selection still truncates arbor.

### Target selection by un-invalidated length — implemented, and it is a trade

`select="uninvalidated"` in `neutu_trace` is NeuTu's actual rule: score every
boundary voxel by the geodesic length of its not-yet-invalidated tail, take the
argmax, and stop once the best remaining falls under `minimalLength`. Two things
made it affordable, both of which I had wrong at first:

- **`dijkstra3d.parental_field` is decodable** — a 1-based Fortran flat index, 0 at
  the root. `decode_parents` reads it and
  `test_decode_parents_gives_adjacent_parents` pins the encoding, since it is
  undocumented.
- **Work in the component's index space, not the crop's.** The per-round
  first-invalidated-ancestor search is pointer doubling over an absorbing map,
  `log2(depth)` vectorised gathers. Run over the full crop it took 139 s on body
  45813451; compacted to the component's 27,750 voxels it takes 5.6 s — on par
  with max-DAF. **No Cython needed**; the earlier "this needs a C extension"
  concern was about the wrong array.

Measured over all 12 bodies, medians (port : NeuTu):

| | `select="daf"` (default) | `select="uninvalidated"` |
|---|---:|---:|
| tips | **1.01×** | 1.22× |
| cable | **1.07×** | 1.13× |
| A→B | **0.83** | 0.93 |
| B→A p90 | 2.44 | **2.41** |

Near-identical on ten bodies. On the two bulb-heavy ones the trade is large and
runs both ways:

| body | daf | uninvalidated |
|---|---|---|
| 6308993 | 1.01× tips, p90 **7.51** | **2.78×** tips, p90 **4.18** |
| 45892915 | 1.16× tips, p90 **7.52** | **2.61×** tips, p90 **4.50** |

The figure settled what the medians could not, and **against** the new selector.
`figures/selector_comparison.png`: it drops NeuTu-cable-absent from 13.6% to
**2.8%** — so the p90 gain was real — but at **31.5% of its own cable added**, and
that addition is overwhelmingly boundary convolution inside the bulbs. It was not
choosing branches better; it was tracing nearly everything, including the third of
its cable NeuTu deliberately rejects. On data where the bulbs are segmentation
noise, that is the failure mode, not the fix.

**So the code was removed** (it lived at commit `83d1356`), on the same grounds as
the greedy selector: ~120 lines, a public `select` parameter nobody should pass,
and a dependency on `parental_field`'s undocumented encoding. Everything needed to
rebuild it is here — the encoding is a 1-based Fortran flat index with 0 at the
root, and the per-round search must run in the component's index space, not the
crop's.

Two hypotheses for the extra branches were tested and **refuted**: restricting
candidates to boundary voxels (NeuTu's `m_fgArray`) changed nothing (1.93× →
1.93×), and NeuTu's effective threshold is *lower* than ours (8, since
`minLength -= maskExpansionRadius`), which would give it more branches, not fewer.
### What is actually left: three differences in the growth, none of them the cost

Read from source rather than assumed. **The weight function already matches ours**:
`Stack_Voxel_Weight_I` (`c/tz_stack_graph.c:205`) is `d/(1+v₁) + d/(1+v₂)` where
`v` is the *squared* distance from `Stack_Bwdist_L_U16P` — i.e. `d/(1+r²)`, which
is what `neutu_pdrf` computes. So "port NeuTu's growth" is not about the cost.

What does differ, in decreasing order of suspicion:

1. **Root selection — NeuTu re-seeds (`m_rebase`, on in our config).** It seeds at
   the *thickest* voxel (`Stack_Max(tmpdist)`), grows, extracts the longest path,
   then **re-seeds at that path's first point and grows again**
   (`zstackskeletonizer.cpp:341-359`). Ours seeds at the geodesically farthest
   point from `first_label`, an arbitrary voxel. Both land on an extremal tip, but
   not the same one — and the root determines the whole parent tree, hence every
   path. Small code change, plausibly large effect.
2. **Edge form — MEASURED, and it does matter.** NeuTu's cost is symmetric over the
   edge; `parental_field` takes a per-voxel field. On tubes they are bit-identical,
   which is what `test_per_voxel_weights_match_neutu_edge_cost` asserts — but a tube
   is precisely where they must agree, since one route dominates. Inside real bulbs
   (61³ crops on the two thick bodies, 16 root→target pairs against the slow
   reference Dijkstra):

   | | tubes | bulbs |
   |---|---|---|
   | identical paths | bit-identical | **0 / 16** |
   | our cost ÷ NeuTu's optimum | 1.000000 | **median 1.10, max 1.13** |
   | max deviation from reference route | 0 | **up to 12.1 voxels** |

   The two costs differ *only* through mixed 3D step lengths:
   `Σ dᵢ(fᵢ + fᵢ₊₁)` telescopes to `2·Σ dᵢfᵢ₊₁` plus fixed endpoint terms **when all
   steps are equal**. With 1, √2 and √3 interleaved it does not, and where routes are
   near-equal that flips the winner. So the tube test proves less than it appears to
   — do not read it as validating the substitution in general.

   Caveat on scope: both crops landed at occupancy 1.00, i.e. *solid* thick cores,
   not the convoluted tufts where the figure shows disagreement. A crop in a
   convoluted region is still owed before claiming this covers the tufts.

   **Fixing it needs an edge-weighted Dijkstra, which `dijkstra3d` cannot do** — all
   its entry points take a per-voxel `data` field. `scipy.sparse.csgraph.dijkstra`
   can, with `return_predecessors=True`: build the 26-connected graph over the
   component explicitly with weights `d·(f(u)+f(v))`. For a 10⁶-voxel component that
   is ~26M edges, roughly 320 MB in CSR — affordable, C-implemented, and **no new
   build dependency**, so still no case for Cython.
3. **Distance-map quantisation.** NeuTu's is uint16 *integer squared* distance; we
   square a float EDT. At bulb radii (r≈10, r²≈100) that is ~1% relative, so this
   is the weakest candidate — but it is a one-line experiment.

Start with 1 and 2; both are cheap, and 2 is a pure measurement.

**Also tried and removed: greedy path reselection.** Reframing the rule as a
greedy set-cover over the paths from a `min_length=0` pass measured far worse
(tip 5.9× and 21.9×) because only *accepted* paths invalidate there, whereas the
sequential pass invalidates on rejected ones too — so territory stayed uncovered
and far too many paths cleared the threshold.

**The generalisable lesson**: a parameter that compensates for a mechanism you have
not directly verified is usually masking a bug. The compensating constant survived
several rounds of scrutiny, a 12-body validation, and a figure, because every
measurement was downstream of the same two defects.

## Fallback — the NeuTu plugin

If steps 1–3 stall, `em_seg_morpho.neutu_io.run_neutu` is ready. Costs: a GPL binary
out of process, a second conda environment, per-body subprocess and file I/O, and the
argument-order workaround. Parallelises fine per body (disBatch on Rusty).

Keep the NeuTu reference SWCs from step 0 either way — they are how the
reimplementation gets validated.

---

## Integration — the remaining work

**Wanted for the production run: `cost="edge"` exposed in `SkeletonConfig`**
(requested 2026-08-03, conditional on the 12-body numbers holding up and the
rendered figure looking right). Deliberately *not* added yet: `SkeletonConfig`
configures the kimimaro block-first stages, and `neutu_trace` is not wired into
them, so the field would be a knob nothing reads. Add it with the tracer swap.

**Check memory before setting it.** `cost="edge"` builds an explicit 26-connected
graph, so peak RSS scales with edge count: 1.9 → 2.9 GB on body 45892915
(568 K voxels), extrapolating to ~4–5 GB on a 10⁶-voxel component. At 48 dask
workers that is ~240 GB, so it needs either a voxel-count cap that falls back to
`"voxel"`, or fewer workers. This may be moot under block-first, where per-body
components inside a 256³ block are far smaller than whole-body ones — but it has
not been measured there.

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
