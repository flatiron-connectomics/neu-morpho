"""Per-body morphology measurement, over PUBLISHED output.

An analysis layer, not a pipeline stage: it reads meshes and skeletons that have already
been written and turns them into tables. Deliberately not a `--stages` entry, because it
measures published volumes rather than producing them, and the thing being measured is
often not the thing this pipeline produced.

The driving comparison is Megaphragma against the Drosophila male-CNS — non-somatic
volume, neurite diameter and cable length, controlling for denucleation — and one property
of it shapes the whole interface: **the two cohorts are not measured the same way.**
Male-CNS has `soma` and `nucleus` semantic labels, so its control is a *mask*. Megaphragma
has none, but its nucleated bodies are annotated, so its control is a *cohort selection*
(anucleate cells only). Both are legitimate; the risk is that the difference goes
unrecorded and the two columns get compared as though they were the same measurement.
Hence every row carries an explicit ``variant``, and the table is keyed
``(dataset, body_id, variant)``.

Nothing here imports a store. Reading is `neu_morpho.readback` and `neu_vol`; the region
predicates that turn a compartment into an ``inside()`` are `neu_lib`.
"""

from .compartments import (joint_counts, semantic_label_names,
                           somatic_labels, sweep_compartments,
                           verify_compartments)
from .driver import resolve_keep, sweep_volumes
from .skeletons import sweep_skeletons
from .metrics import (cable_length_nm, diameter_stats, frustum_area_nm2,
                      frustum_volume_nm3, measure_skeleton, topology,
                      weighted_quantile)
from .tables import (compare, load_bodies, load_segment_properties,
                     log_histogram, summarize, write_table)
from .sweep import (DEFAULT_BLOCK, DEFAULT_VOXEL_NM, SweepTotals, bin_to_nodes,
                    blob_signal, blocks_from_mask, cable_shares, count_labels,
                    log_bin_edges, mean_cross_section, node_radii, roi_block_mask,
                    weighted_histogram)

__all__ = [
    # per-skeleton metrics
    "cable_length_nm",
    "diameter_stats",
    "frustum_area_nm2",
    "frustum_volume_nm3",
    "measure_skeleton",
    "topology",
    "weighted_quantile",
    # the block-mapped driver (the only part that opens a store)
    "resolve_keep",
    "sweep_volumes",
    "sweep_skeletons",
    "sweep_compartments",
    "verify_compartments",
    "joint_counts",
    # tidy tables for cohort selection (pandas, from the `measure` extra)
    "load_bodies",
    "load_segment_properties",
    "summarize",
    "compare",
    "log_histogram",
    "write_table",
    "semantic_label_names",
    "somatic_labels",
    # the voxel-counting sweep
    "DEFAULT_BLOCK",
    "DEFAULT_VOXEL_NM",
    "SweepTotals",
    "bin_to_nodes",
    "blob_signal",
    "blocks_from_mask",
    "cable_shares",
    "count_labels",
    "log_bin_edges",
    "mean_cross_section",
    "node_radii",
    "roi_block_mask",
    "weighted_histogram",
]
