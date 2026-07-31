# Plan: NeuTu-quality skeletons in em-seg-morpho

Background and evidence: `docs/skeletonization-comparison.md`. Read that first —
in particular the "Corrections" section, which lists conclusions already established
as wrong.

**Goal.** Skeletons that fill the segment well, with few nodes and radii good enough
for visualization. NeuTu currently does this better than kimimaro at production
settings (73% fill / 1,431 nodes vs 39% / 5,705 on body 6308993).

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

Vendor `kimimaro.trace.trace` and `compute_paths` into this package rather than
monkeypatching, so the behaviour is readable and pinned.

---

## Step 0 — widen the benchmark first

**Do this before writing any skeletonizer code.** The current conclusion rests on two
bodies from one specimen, and the metric already reversed once. Pick ~10 bodies from
`configs/largest_20k_bodies.csv` spanning a range of size and thickness, export masks
at scale 2, and record for each: NeuTu SWC (the regression target), kimimaro at
production and relaxed settings, and `skelmetrics.score` for all.

Store masks and reference SWCs on ceph, not in the repo. Cheap, and it de-risks
everything below — if NeuTu's advantage is body-dependent, better to know now.

## Step 1 — NeuTu-style cost, everything else kimimaro

Vendor `trace()`, swap in `PDRF = 1/(1 + DBF**2)`, run with `scale=1.0, const=2`.

Verify against the NeuTu references from step 0. One thing to check rather than
assume: `parental_field` takes **per-voxel** weights, whereas NeuTu's cost is the
symmetric edge form `d·[f(v₁) + f(v₂)]`. These should differ only by a constant
factor and boundary terms, leaving the argmin path unchanged — but confirm it with a
test on a synthetic tube before trusting it on real data.

Expected: centreline close to NeuTu's, node count still kimimaro-like (dense),
fill improved over kimimaro production.

## Step 2 — radius-adaptive node placement

Port `createSwcByRegionSampling` (`gui/zswcgenerator.cpp:196`): sort path voxels by
decreasing radius, greedily drop any within a kept larger voxel's ball. NeuTu's
implementation is O(n²); use a KD-tree.

This is **tool-independent post-processing** — it applies to any SWC, so it is worth
having even if step 1 is abandoned. Expected: ~2× fewer nodes at equal fill.

## Step 3 — radius-aware simplification

Port `ZSwcResampler::optimalDownsample` (`gui/swc/zswcresampler.cpp:90`): iterate to a
fixpoint, dropping a node when its ball is contained in a neighbour's or when
interpolating parent↔child reproduces it in position (within `radius/2`) and radius
(within 1.2×). Smooths radii as a side effect.

## Step 4 — radius convention

Move from inscribing to filling. kimimaro's `radii = DBF[vertex]` under-fills
non-circular cross-sections; NeuTu's larger radii fill better but overshoot the true
maximum (543 nm reported vs 345 nm available on body 6308993).

Neither is obviously right. Sweep a scale factor against `skelmetrics.score` on the
step-0 benchmark and pick the fill/spill trade you want. **Record the choice and its
rationale** — a radius tuned for filling is not a measurement.

## Step 5 — length-based branch termination

Only if runtime demands it. Add NeuTu's `minimalLength` test on *un-invalidated
geodesic length* to the vendored `compute_paths`, terminating extraction early
instead of running until everything is invalidated.

## Fallback — the NeuTu plugin

If steps 1–3 stall, `em_seg_morpho.neutu_io.run_neutu` is ready. Costs: a GPL binary
out of process, a second conda environment, per-body subprocess and file I/O, and the
argument-order workaround. Parallelises fine per body (disBatch on Rusty).

Keep the NeuTu reference SWCs from step 0 either way — they are how the
reimplementation gets validated.

---

## Watch out for

- **Always pass `edges` to `skelmetrics`.** Scoring vertex spheres instead of swept
  capsules reversed the tool ranking once already.
- **Don't port the `−0.5` radius correction.** Established harmful at these radii.
- **`anisotropy=(8,8,8)` in `SkeletonConfig` is a placeholder.** Scale 2 is
  32 nm/voxel. `const` is in nm; get the conversion right before comparing settings.
- **NeuTu's EDT is not anisotropy-aware** (anisotropy enters only via Dijkstra step
  lengths). Fine for isotropic data; a real problem otherwise.
- **The mask is not ground truth.** Check the thickness map before attributing a bad
  radius to the skeletonizer.
