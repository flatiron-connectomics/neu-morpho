# How `measure` was calibrated

Every default in `neu_morpho/measure/sweep.py` came from a measurement against real data:
a large published Drosophila CNS segmentation (165,122 bodies, 8 nm isotropic at level 0,
sharded `compressed_segmentation`) and our own smaller published Megaphragma segmentation
(7 isotropic levels from 8 nm). Neither is named or located here — get both from the site
config. This records what was measured, so the numbers are not re-derived and the defaults
are not changed by guess.

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

**Neither dataset's published skeletons carry a usable calibre DISTRIBUTION, and that is
what forces `V/L` on both sides.** Theirs is the obvious case; ours is the one that had to
be measured, and it is the reason "measure both identically" resolves the way it does.

Our own inscribed radii come from a distance transform on the skeletonization lattice, so
they are **floored at one voxel** and quantised above it. Measured over 345.9 µm of cable
from five median-sized bodies, cable-weighted:

| | value |
|---|---|
| cable at exactly 32.0 nm — the one-voxel floor | **23.9%** |
| distinct radius values below 64 nm | **7** (32.0, 38.6, 45.3, 55.4, 61.1, 63.5, 64.0) |
| cable-weighted p10 / p25 / p50 / p75 / p90 | 32.0 / 45.3 / **45.3** / 90.5 / 146.6 nm |

p25 and p50 are the same number. Half the cable is thinner than 48 nm, which is where the
quantisation is worst — so a histogram built from these radii would show lattice steps and
report them as biology. Same failure mode as the 512 nm `ds63` radii, three octaves finer.

`V/L` escapes it by averaging: at ~157 nm node spacing and ~45 nm radius a node owns ~30
voxels, so the quantum is ~1/30 of an *area* rather than one voxel of radius — ~1.7%
resolution instead of a 21% jump from 32 to 38.6 nm. This is why `V/L` is not merely the
best available option for the dataset that lacks radii; it is the only estimator that can
carry a distribution on **either** side.

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

**The same ladder was then run on our own dataset, and it must be — the bias is a property
of whoever built the pyramid, not of the geometry alone.** Two segmentations downsampled by
different implementations do not share a bias, so matching the physical voxel size does not
match the error, and each side has to be measured. Four median-cable bodies, three 4 µm
boxes each centred on a node at that body's radius p10 / p50 / p90, volume relative to 8 nm:

| | 16 nm | 32 nm | 64 nm |
|---|---|---|---|
| thin | 1.002–1.004 | 1.008–1.017 | 1.022–1.030 |
| medium | 1.003–1.006 | 1.009–1.018 | 0.997–1.036 |
| thick | 1.002–1.003 | 1.007–1.011 | 1.022–1.032 |

**~1% at 32 nm** — *smaller* than the other dataset's 1–3%, despite our neurites being
thinner in absolute terms (cable-weighted median radius 45 nm against ~105 nm). The
prediction that a thinner neurite at a fixed voxel size must suffer more was wrong, which
is the pyramid-provenance point above. So **32 nm is settled for both sides and no 16 nm
re-index is needed.** The one 0.997 is a 659-voxel box at 64 nm, i.e. small-sample noise.

**Do not read a body's own volume out of a pipeline work dir without checking the stage
ran.** A `--stages skel` run still writes `metrics.db` with a `voxel_count` column and a
planned index scale in `run_plan.json`, but every count is 0 and every bbox NULL, because
the index stage was never in `--stages`. The planned scale is not evidence the scan happened.

## Block selection: no ROI filter, and a derived one for the second pass

That dataset publishes a major-compartment ROI volume at 256 nm labelling `CentralBrain`,
`Optic(L)`, `Optic(R)`, `CV` and `VNC`. Brain (labels 1–3) is **67.1%** of the tissue but
only **8.0%** of the blocks (9,466 of 118,404 at 512³/32 nm, or 12,096 dilated by one) —
because the tissue is sparse in its own bounding box. That filter looked like a 10× saving.
(Treat the dilated figure as an upper bound: it was measured while `roi_block_mask` grew the
set with a *periodic* `np.roll`, so some of those blocks were wrap-around from the opposite
face. Since fixed; the undilated count is unaffected.)

**It is not safe, for two independent reasons, and neither is visible in the output.**

1. **Many somata with nuclei lie OUTSIDE the shells.** Filtering blocks to the shells reads
   a neuron's arbor and misses its soma — while the *skeleton* still covers both. `V/L` then
   divides a truncated volume by a complete cable length and **under-reads calibre on
   exactly the bodies that have somata**, with every count, every manifest entry and every
   row looking complete. Invariant 5's shape.
2. **The ROI array does not span the segmentation.** Both are anchored at origin zero, but
   the ROI's extent is 341 × 430 × 742 µm against the segmentation's 1077 × 627 × 753 µm.
   So two thirds of z is not "outside the brain", it is *unlabelled* — a filter built from
   it silently drops whatever is there.

Patching it with a dilated union of soma-points and the nuclei segmentation would work, but
the dilation radius is a guess and a guess too small truncates silently. So there is no ROI
filter. Instead:

**Pass 1 reads every block.** 118,404 blocks at the measured 0.348 s/block is ~15 min of
wall clock on 48 workers. The filter was buying 15 min → 1.5 min and now costs correctness.
This pass is `ops/index_segments.py` — per-label bbox, voxel count and volume, block-mapped
and resumable — pointed at the volume with `block_shape=(512,512,512)` and `roi=None`.

**Pass 2 uses the bboxes pass 1 produced.** Nearest-node binning is ~10× the cost of
counting, so it runs over the union of the *selected cohort's* bounding boxes: a block set
that is exact by construction rather than an approximation of one, and smaller than any ROI
would have given. Brain-versus-VNC then becomes a post-hoc filter on body position, which
also classifies bodies rather than blocks — so a neuron straddling the boundary is a
decision instead of a truncation.

## The compartment volumes share the segmentation's grid

The semantic masks (nuclei / soma / neuropil / fiber-bundle) and the nuclei instance
segmentation are both 16 nm at their level 0, origin zero, same voxel size as the
segmentation's level 1. Shapes differ only by padding — at 32 nm, semantic
33,664 × 19,584 × 23,522 against the segmentation's 33,644 × 19,580 × 23,522.

So a **joint per-block read at matching voxel indices is valid**, clipped to the
segmentation's shape, with no resampling and no offset. That is what makes
compartment-resolved volume for every body a by-product of pass 1 — count `(body,
semantic_label)` pairs instead of bodies — rather than a per-body soma-point fetch. Check
the padding rather than assuming it: the grids agree here, and nothing guarantees the next
pair will.

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

## An inscribed radius under-reads area by the ASPECT RATIO, and it is ~1.5

Measured whole-body on 300 bodies, comparing `V/L` against the cable-weighted mean area
implied by the same skeletons' published inscribed radii:

| | value |
|---|---|
| area ratio, p10 / p50 / p90 | 1.38 / **1.54** / 1.79 |
| radius ratio at the median | 1.24 |
| correlation with the body's median radius | **+0.23** |
| area ratio by median-radius bin (<40 / 40-50 / 50-70 / >70 nm) | 1.45 / 1.52 / 1.58 / 1.57 |

**Flat in thickness.** The obvious explanation — that eccentricity worsens as a process
thins, which would have reconciled this with the `ds7` result below — is wrong, and the
weak correlation runs the *other* way.

What does explain it: for an ellipse with semi-axes `a >= b` the inscribed radius is `b`
and the area is `pi*a*b`, so `area / (pi * r_inscribed^2)` **is exactly the aspect ratio**.
So 1.54 says the median cross-section has an aspect ratio of ~1.54 — against an
independent estimate of ~1.4 from a different method. Two unrelated routes, one number.

Note the quantisation floor pushes this the *conservative* way: 24% of cable is reported at
the one-voxel minimum, which for genuinely thinner regions over-states the inscribed radius
and so shrinks the ratio. The true figure is at least this large.

**Consequence, and it is the load-bearing one:** mixing an inscribed radius on one side of
a comparison with `V/L` on the other biases the areas by ~1.5x — far more than any voxel-size
or cable-sampling effect in this document. This is the strongest reason the two datasets must
both be measured by `V/L`, over and above the quantisation argument.

**It also puts the `ds7` 1.00-1.02 agreement in question rather than confirming it.** Both
are inscribed radii, so they should not disagree by 30%. The difference in method is that
`ds7` was compared inside 4 um boxes while this is whole-body — and box-edge effects on
`V/L` are still unquantified (a body's volume crossing the box edge while its cable is
clipped biases the ratio directly). Resolve that before quoting either number.

## First complete result, and how to read it

Both datasets measured by `V/L` at 32 nm: volume from a per-block voxel sweep, cable from
each dataset's own published skeletons. `all` variant — no compartment exclusion yet.

| | n | cable µm (p10/p50/p90) | volume µm³ | `V/L` diameter nm |
|---|---|---|---|---|
| larger dataset, all | 165,121 | 215 / **600** / 2374 | 41.6 / **98.9** / 410.9 | 365 / **455** / 662 |
| smaller dataset, size-selected | 19,989 | 15.2 / **71.3** / 228.4 | 0.4 / **2.0** / 5.9 | 122 / **185** / 288 |

Median ratios: **volume 48.5x, cable 8.4x, diameter 2.46x.** Those are consistent —
`8.4 x 2.46^2 = 50.9` against 48.5, the residual being medians-of-ratios versus
ratios-of-medians — which is the internal check that the decomposition is sound. So the
volume difference is *mostly arbor extent, not calibre*: ~8x more cable and ~2.5x thicker.

**Calibre is near-invariant across cell classes while arbor size is not**, in the larger
dataset: median `V/L` diameter spans only 426-479 nm across intrinsic classes (optic 457,
central 426, VNC 479, visual projection 436) while median cable spans 392-1792 µm, a 4.6x
range. A systematic measurement artefact would more plausibly track size, so this is mild
evidence the measure is behaving. The thinnest class measured is sensory, at 384 nm.

**Two corrections both push the volume ratio DOWN, and neither is applied yet:**

- **The cohorts are selected on different principles.** The smaller dataset's list is the
  *largest N bodies*, so it is truncated from below and its median volume is higher than a
  curation-selected set's would be. Correcting this makes the ratio **larger**.
- **Compartments are not excluded.** Cell bodies are ~21.7% of neuronal tissue in the
  larger dataset (below), while the smaller dataset's cohort is overwhelmingly anucleate —
  which is the biological point of the comparison. So its volumes carry a somatic term the
  other's mostly lack, and removing it takes 48.5x to roughly 38x. Somata also inflate
  `V/L` diameter, in the same direction.

Do not quote 48.5x as a result. It is the `all`-variant, mismatched-cohort figure, and both
known corrections move it.

## The compartment split, per body — and which variant is the right control

Measured by joining each segmentation block against the aligned semantic block and
counting `(body, label)` pairs. Share of the body list's OWN voxels (not of tissue):

| neuropil | fiber-bundle | soma | nucleus | glia | other |
|---|---|---|---|---|---|
| 64.64% | 11.73% | 11.40% | 9.82% | 2.01% | 0.41% |

Cell bodies are **21.2% of the bodies' own volume**, which independently reproduces the
21.7% tissue-based figure from a different computation. The ~2% of neuron voxels labelled
`glia` is semantic-mask/segmentation disagreement, and 0.07% carry no label at all.

**Per-body nucleus fraction: p10/p50/p90 = 0.004 / 0.162 / 0.258.** The spread is the
point — a population average would have hidden it, and the p10 tail is bodies whose soma
lies outside the volume.

| variant | vol µm³ p50 | `V/L` diam p50 | vs the smaller dataset |
|---|---|---|---|
| `all` | 98.9 | 455 nm | 48.5x vol, 2.46x diam |
| **`minus_nucleus`** | 82.5 | 412 nm | **40.5x vol, 2.23x diam** |
| `minus_soma_nucleus` | 70.8 | 378 nm | 34.7x vol, 2.05x diam |

**`minus_nucleus` is the matched control, not `minus_soma_nucleus`.** The smaller
dataset's neurons are *denucleated but not asomatic* — they lack nuclei and still have
cell bodies — so stripping the whole cell body from one side while leaving it on the other
over-corrects. Getting this wrong moves the volume ratio by 6 units and the diameter ratio
by 0.18, in the direction that flatters the hypothesis.

**Known bias in the minus-variants, and its direction:** they divide a reduced volume by
the FULL cable length, including the cable running through the soma. So they
*under*-estimate neurite calibre. Fixing it means splitting cable by compartment too —
sampling the semantic volume at each skeleton node, which is cheap and not yet done.

**The occupancy filter was validated, not assumed.** The pass ran on blocks non-empty at
2048 nm of the segmentation itself, dilated by one — 24,366 of 118,404. Because label 0 is
counted, each body's per-label sum must equal the total the unfiltered volume sweep
recorded, and it did for every body but one: **53 voxels missing out of ~1.07e12, in a
single body**. Invariant 6's failure mode is therefore real (a coarse grid did drop an
isolated speck that one block of dilation could not recover) and negligible, and it is
*named* rather than silent. An occupancy grid derived from the segmentation is safe in a
way an anatomical ROI is not: a soma is segmentation, so it is non-zero at any level.

## Compartment fractions, from the semantic masks

The larger dataset publishes **eight** semantic labels, not four: neuropil, fiber-bundle,
nucleus, glia, soma, trachea, muscle, do-not-merge. Global fractions of labelled tissue at
512 nm:

| neuropil | glia | fiber-bundle | soma | nucleus | muscle | trachea | do-not-merge |
|---|---|---|---|---|---|---|---|
| 51.09% | 10.41% | 9.57% | 9.31% | 7.48% | 6.37% | 4.04% | 1.72% |

Cell bodies (soma + nucleus) are **16.8% of all tissue**, or **21.7% of neuronal tissue**
(neuropil + fiber-bundle + soma + nucleus = 77.45%) — muscle, trachea and glia are not in
the body list at all, so which denominator is used matters.

**These fractions are scale-invariant**, so nothing here needs fine reads: three 128 µm
boxes at 64 / 128 / 512 nm gave neuropil 56.69 / 56.70 / 56.79% and soma 5.98 / 5.97 /
5.94%. The mode-downsampling worry — that coarse levels would favour compact somata over
thin neuropil — does not materialise. (Those box figures differ from the global ones
because three boxes are not a representative sample; they establish scale-invariance only.)

## Open

- **Box-edge effects on `V/L`** — a body's volume extending past a patch boundary while
  its centreline does not, or the reverse. Not yet quantified.
- **Bin range and count**, per above. Note `node_radii` returns *radii* while the reported
  quantity is *diameter*: record which the edges are in, or a radius column gets compared
  against a diameter column and a factor of two is reported as biology.
- **Pruning, which is the dominant cable-length term and is larger than sampling.** Our
  production run's tick removed 3.6% of cable (`tick_cable_removed_nm` against `cable_in`),
  and earlier body sets gave 17.4% and 8.2% — the threshold is still unswept. Their
  `min_length` is recorded in every SWC header, so the two are directly matchable. Node
  *spacing*, by contrast, is a ~3% effect on radius: the measured 6.1% cable deficit enters
  `V/L` as an area and so as `1/sqrt(L)` on radius.
- **Whether the fine and coarse SWC generations share a `min_length`.** If they do not, the
  in-dataset cable comparison measures pruning and sampling together and cannot separate
  them, and the pairing has to change.
- **Whether the EM image is zeroed outside the segmentation or outside the neuropil.** If
  the former, a coarse level's non-zero extent is a valid occupancy grid; if the latter it
  has the same soma problem as the ROI shells. Untested, and only an optimisation now that
  pass 1 reads everything.
- Cross-sections are genuinely non-circular — median aspect ratio ~1.4 by one estimate —
  so the *inscribed* radius tracks the minor axis and is blind to the major one. `V/L` is
  an area-equivalent measure and so does not share that bias, which is a reason to prefer
  it beyond mere availability.
