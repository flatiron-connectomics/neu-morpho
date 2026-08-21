# How `measure` was calibrated

Every default in `neu_morpho/measure/sweep.py` came from a measurement against real data:
a large published Drosophila CNS segmentation (165,122 bodies, 8 nm isotropic at level 0,
sharded `compressed_segmentation`) and our own published sample3. Locations are
deliberately not recorded here; get them from the site config. This records what was measured, so the numbers
are not re-derived and the defaults are not changed by guess.

Two conclusions were reached and then **reversed by measurement**; both are kept below,
because the wrong version was well-argued and someone will arrive at it again.

## The measurement: `V/L`

A body's voxel count gives its volume. Dividing by the cable length of the centreline
through it gives the **mean cross-sectional area**. Binning the voxels by nearest
centreline node makes the same pass a *length-weighted distribution* of local
cross-section.

Validated end to end on one body: volume **40.31 µm³** against **41.47 µm³** from the
mesh divergence theorem — two unrelated methods agreeing to **2.8%** — and median diameter
**231 nm** against the `ds7` reference's 222 nm (ratio 1.04). The histogram conserves cable
exactly, which is what the under/overflow bins are for.

## Why not the alternatives

**Their published skeletons carry no usable calibre.**

| source | verdict |
|---|---|
| neuroglancer skeletons | `info` is a bare `{"@type": "neuroglancer_skeletons"}` — no `vertex_attributes`, every radius the `-1` sentinel |
| SWC `ds7` (64 nm) | **excellent: 1.00–1.02** against `V/L` for medium/thick — but ~1% of bodies |
| SWC `ds63` (512 nm) | unusable: lattice-quantised, every body's median landing on exactly 256 nm, overshooting `V/L` by 1.08–4.84 (median ~1.9) |

The generation is in the SWC header (`ds_intv`, `downresLevel`); sampling 72 bodies
stratified by synapse count found **71 of 72 were `ds63`**. Their neuroglancer skeletons
sit on a 512 nm lattice — edge lengths are exactly `512·√1, √2, √3, 2, √5, √6, √8, 3`,
i.e. steps between voxel centres — where ours are on 32 nm.

**A mesh ray estimator is worse.** Casting rays in the cross-sectional plane and taking
the polygon area overshoots by **1.14–1.29×** even after gating out rays that escape
through the meshes' ~5% boundary edges and clipping per-section outliers. Before those
fixes it was 1.39–1.68×. The mesh volume itself is fine — stable to ~1% with the origin at
the centroid, and the translation sensitivity is a free per-body leak error bar — but the
*calibre* is not competitive with `ds7` radii, so it does not earn its complexity.

**Re-tracing is unnecessary.** Counting voxels is I/O-bound where skeletonization is
compute-bound: ~1 s per 512³ block against 225 s per 256³ block, a ~300× difference.

## The three performance facts

Measured at 32 nm on that segmentation (64³ chunks at every level, `preshift_bits=9`):

**Read 512³ blocks.**

| block | s/block | Mvox/s |
|---|---|---|
| 64³ | 0.123 | 2.1 |
| 128³ | 0.108 | 19.3 |
| 256³ | 0.264 | 63.6 |
| **512³** | 0.348 | **385.9** |

Per-block time is nearly flat, so it is fixed overhead and bigger is better. 512³ is also
exactly the `preshift_bits=9` shard locality group, which is why the jump from 256³ is 6×
and why a further jump to 1024³ is unlikely. One block is ~1.07 GB as uint64.

**Drop zeros before counting.** These volumes are ~94% empty, so `block[block != 0]` then
`np.unique` takes **0.08 s** where `np.unique` on the raw block takes **1.39 s**.

**Never `np.isin` to select a cohort inside a block** — **20 s**, twenty times the read
cost. Count every label and select cohort rows afterwards from the small result table.

## Scale: 32 nm, and coarsening *over*estimates

The obvious worry — that a coarse voxel count undercounts thin processes through partial
volume — is **backwards**. Segmentation downsampling is by *mode*, so a thin process that
wins a 2×2×2 vote takes the whole coarse voxel: it dilates rather than erodes.

Volume relative to 8 nm, on 4 µm boxes around thin/medium/thick regions:

| region | 16 nm | 32 nm | 64 nm |
|---|---|---|---|
| thin | 0.99 | 1.03 | 1.10 |
| medium | 1.00 | 1.02 | 1.07 |
| thick | 1.00 | 1.01 | 1.03 |

And the recovered radius is nearly scale-invariant: thin 61→64 nm, medium 110→113, thick
171→173 across the whole 8→64 nm range. So **32 nm carries 1–3% bias and needs no
correction**, and finer buys nothing.

## The ROI filter

That dataset publishes a major-compartment ROI volume at 256 nm labelling `CentralBrain`,
`Optic(L)`, `Optic(R)`, `CV` and `VNC`. Brain (labels 1–3) is **67.1%** of the tissue but only **8.0%** of the blocks
(9,466 of 118,404 at 512³/32 nm, or 12,096 dilated by one) — because the tissue is sparse
in its own bounding box: the VNC stretches the box along one axis while the brain is a
compact blob. Dilating by one block is insurance, not cosmetics: a false-positive block
costs one read that finds nothing, a false negative silently truncates a body.

Cost with that filter: **~3.6 min on 48 workers** for per-body counts, ~1.1 h with
nearest-node binning.

## `V/L` is a tube measure

For a sphere of radius R whose centreline crosses it, `V/L = (4/3)πR³ / 2R = (2/3)πR²`, so
the recovered radius is `R·√(2/3) = 0.816R` — and that is before any departure from
sphericity or any question of how much centreline the skeletonizer put inside a soma.

Measured against `ds7` radii, the ratio degrades monotonically with thickness: **p10 0.81,
p50 1.09, p90 0.70**, consistently across boxes from 16 to 33 µm and whether centred on
the thickest node or the median one. So the thick-tail compression is inherent, not
box-clipping.

**Somata are excluded by design** in the variants that matter, so the regime where this
bites is the regime being removed. On an unexcluded body, read the thick tail as "`V/L`
over a blob", never as a diameter. `blob_signal` reports the share of cable where the
voxels are spread more widely than a tube can explain.

### The reversal worth remembering

`blob_signal`'s first version compared a node's mean voxel-to-node distance against its
`V/L` radius, and reported **86% of a thin arbor as blob-like**. A voxel's distance to its
node has an *axial* component from the node spacing as well as a radial one, so the
comparison must be against a cylinder's expectation, `√(r²/2 + s²/12)`. At 181 nm node
spacing and a 115 nm radius the axial term dominates — that is the ordinary case, not a
corner one. Corrected, the same body reads 0.235 / 0.093 / 0.022 at tolerance
1.25 / 1.5 / 2.0.

## Nearest-node assignment is robust to coarse centrelines

The worry was that 512 nm node spacing would misattribute voxels between branches passing
close together. Validated by taking a fine (`ds7`, 181 nm median edge) centreline and
subsampling it to ~800 nm:

| | correlation with `ds7` radii | cable-weighted p10/p50/p90 |
|---|---|---|
| fine, 181 nm | 0.835 | 48 / 105 / 280 |
| coarse, ~800 nm | **0.855** | 50 / 107 / 267 |

Slightly *better* coarse, and distributions agreeing to a few nm. **Coarse node spacing is
not a source of error here**, which matters because the production centrelines are 512 nm.
Per-node values are noisy (±40%), so this is an instrument for distributions, not for
individual nodes.

## Cable length, and the other reversal

Their cable length is measurable from their published skeletons, but a 512 nm trace cannot
follow finer curvature. Resampling *our* 32 nm skeletons to 512 nm spacing — the
achievable direction, since coarsening Megaphragma destroys its masks outright — retains
**93.9%** of the cable. So their lengths are low by ~6% from sampling alone. That is a
floor, not the whole story: a coarser trace also follows a different path.

The reversal: their skeletons were **generated by NeuTu**, which `neu_morpho.neutu_trace`
reimplements and already matches to sub-voxel centreline agreement, and their `min_length`
is recorded in every SWC header. So a matched comparison is closer to hand than the raw
512-vs-32 nm gap suggests.

## Bins

Log-spaced, because the two datasets differ by decades. Linear bins over 0–10 µm put the
*entire* Megaphragma diameter distribution into bins 0–2 of 32; log bins over 16 nm–10 µm
give 22% per bin at 32 bins and 14% at 48. The bin count and range are **not yet settled**
— the range should come from pooled percentiles across both datasets, chosen once, fixed
thereafter, and recorded by name in every row, since a histogram whose edges are implied
or vary between files is worse than none.

Under and overflow bins are not optional: they are what makes `hist.sum() == cable`
exact, so the histogram is a decomposition of the cable rather than a lossy view, and the
total is the check that catches a wrong range.

## Open

- **Box-edge effects on `V/L`** — a body's volume extending past a patch boundary while
  its centreline does not, or the reverse. Not yet quantified.
- **Bin range and count**, per above.
- **Megaphragma must be measured identically.** Its published skeletons carry inscribed
  radii; switching to `V/L` for male-CNS means recomputing Megaphragma the same way, or
  the comparison is minor-axis on one side and area-equivalent on the other.
- Cross-sections are genuinely non-circular — median aspect ratio ~1.4 by one estimate —
  so the *inscribed* radius tracks the minor axis and is blind to the major one. `V/L` is
  an area-equivalent measure and so does not share that bias, which is a reason to prefer
  it beyond mere availability.
