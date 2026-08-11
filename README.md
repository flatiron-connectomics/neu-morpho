# em-seg-morpho

Segment **morphology** from segmentation volumes: multi-resolution Draco-encoded
**meshes** (via [`vol2mesh`]) and **skeletons** (via [`kimimaro`]), written in
neuroglancer-precomputed format.

- **Orchestration** across segments: [`em-blockrun`](../em-blockrun) (dask
  local/SLURM + resumable manifest).
- **Segmentation-array I/O**: [`em-volume-tools`](../em-volume-tools) (read
  precomputed / zarr / … regions and crop views).
- **Meshes**: `vol2mesh` (marching cubes → simplify → Draco → multi-resolution
  neuroglancer mesh).
- **Skeletons**: `kimimaro` (TEASAR) → precomputed skeleton format.

Large segments — whose binary mask over a big bounding box would OOM (the failure
mode this package is designed around) — are meshed **chunked and stitched**
rather than materialized whole.

## Environment

One conda environment covers this repo and the two below it, each installed
editable. `em-blockrun` and `em-volume-tools` must be **sibling directories**.

```bash
conda activate em-lib
pip install --no-deps -e ../em-blockrun -e ../em-volume-tools -e .
python -m pytest -q
```

Python is pinned to **3.12**: `vol2mesh` and `dvidutils` are only built for py312
on flyem-forge, and they are conda-only (no PyPI equivalent), so the conda
environment — not pip — has to provide them. The combined spec lives one level
up, at `em-libraries/environment.yml`. Skeleton *comparison* tooling
(`skelcompare.py`) needs the `compare` extra: `networkx`, `plotly`, `skeletor`.

## Running it

Installing the package provides the **`em-morpho`** command (equivalently
`python -m em_seg_morpho`):

```bash
em-morpho run ...                  # the pipeline
em-morpho progress <work-dir>      # live per-stage counts
em-morpho run-report <work-dir>    # self-contained HTML summary
```

`run` drives the stages against a dask cluster — local or SLURM, chosen by
`--config`. Every stage is idempotent, so re-running the same command resumes.

### The stages

`--stages` takes a comma-separated subset, run in this order. The default is
`index,mesh,skel`.

| stage | what it does |
| --- | --- |
| `seg` | Copies the ROI's labels out as a precomputed volume, so the meshes and skeletons can be viewed against the segmentation they came from. **Off by default** — it is pure I/O and usually the most expensive stage of the four. |
| `index` | Scans the volume once and records **every body's bounding box and voxel count** in a SQLite metrics DB. |
| `mesh` | Multi-resolution Draco meshes, one per body. |
| `skel` | kimimaro skeletons, one per body. |

**`index` is required before `mesh` or `skel`, and it is the stage whose purpose is
least obvious.** Meshing and skeletonization each crop a body out of the volume by its
bounding box, and there is nowhere to ask for that box: a label's extent is only knowable
by looking at every block it might appear in, which is exactly the scan `index`
performs. The voxel counts it records are the other half — the size filter that builds
the allowlist reads them, so without the DB there is no way to say "only bodies above N
voxels". It writes no volume data, which is why it is easy to mistake for optional.

Once the DB exists you can re-run the later stages without it: `--stages mesh` is the
normal way to redo meshing after a parameter change. `mesh` and `skel` each run in two
phases — fragments per (body, block), then one output assembled per body — so a re-run
of only stage 2 reuses the fragments already on disk instead of recomputing them.

```bash
# 0. look at the pyramid and pick scales from real metadata (no cluster, no writes)
em-morpho run --src /path/to/seg --describe

# 1. one small cube, in-process — the fastest way to see it work end to end
em-morpho run --src ... \
    --dst /path/to/morpho/segmentation --work-dir /path/to/morpho \
    --serial --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh,skel

# 2. the same cube on SLURM, surviving logout, publishing to s3
nohup env PYTHONUNBUFFERED=1 em-morpho run --src ... \
    --dst s3://bucket/sample3/segmentation --work-dir /path/to/morpho \
    --config dask-slurm-example --config ~/my-site.yaml --workers 48 \
    --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh,skel > run.log 2>&1 &
squeue -u "$USER"

# 3. widen or drop --roi for the whole volume; step 2's work is reused
```

`PYTHONUNBUFFERED=1` is the console-script equivalent of `python -u`; without it the
log lags a long run in 8 KB blocks.

`--roi-scale` is **required** whenever `--roi` is given: the same six numbers name a
box four times smaller per pyramid level, and the wrong one silently processes the
wrong region instead of failing.

### Cluster config

`--config` takes a **bundled template name or a path**, and is **repeatable**,
deep-merged left to right. The templates (`dask-local`, used by default, and
`dask-slurm-example`) ship with **em-blockrun**, next to `start_dask`, so every
consumer shares one set. An overlay carries only the keys that differ:

```bash
em-morpho run --config dask-slurm-example --config ~/my-site.yaml ...
```

```yaml
# my-site.yaml — everything else comes from the template
jobqueue:
  slurm:
    account: my-account
    queue: my-partition
    log-directory: /path/with/room
```

Unrecognised keys raise rather than merging silently, and the effective merged config
is written into the run's work dir. Site-specific configs and dataset allowlists are
deliberately not in this repo.

### Two destinations: `--dst` and `--work-dir`

**`--dst` is the published volume** — the precomputed labels with meshes and
skeletons written *inside* it, so a single neuroglancer layer carries all three.
It may be a local path or an `s3://` URL.

```
precomputed://file:///mnt/ceph/.../morpho/segmentation
precomputed://s3://bucket/sample3/segmentation
```

That volume's `info` gets `"mesh": "mesh"` and `"skeletons": "skeleton"` (the
precomputed spec's subdirectory-naming keys), and the copy carries `voxel_offset`
so it lands at its true global position rather than at the origin.

**`--work-dir` is everything else** — `metrics.db`, manifests, failure logs and
the stage-1 fragment stores. It must be an ordinary filesystem path: it holds a
sqlite DB, appended JSONL and a fragment store read back with file I/O, none of
which work over an object store. It defaults to the parent of a local `--dst`,
and is required when `--dst` is remote.

> **`--dst` changed meaning.** It used to be the run root, with the volume at
> `<dst>/segmentation`. It is now the volume itself. An existing invocation
> passing `--dst /mnt/ceph/.../morpho` will publish to `.../morpho` directly
> rather than `.../morpho/segmentation`. Add `/segmentation` to keep the old
> layout — the work dir then defaults to the old run root.

Because bookkeeping no longer lives inside the output, a manifest can outlive the
data it describes: clear `--dst` and a resumed run would skip every task and
report success having written nothing. Each stage refuses to resume when its
manifest records completed work but its output `info` is gone, naming
`--no-resume` as the explicit override.

`--dry-run` reports the plan (scales, voxel sizes, block counts per stage)
without touching anything — worth running before any large job.

**The ROI filters blocks on the global grid**, it does not re-tile. A block's
index, its region and its nm coordinates are the same numbers in a small run and
in the full run, so widening the ROI later reuses the fragments and manifest
entries already on disk instead of redoing them. The caveat: a body straddling
the ROI edge is built only from the blocks inside it, so it stays truncated until
its neighbours are run.

**Scales are integers; voxel sizes are read from the source.** `--mesh-scale 2`
looks up that level's actual resolution in the precomputed `info` (or the zarr
OME metadata) rather than assuming `2**scale`. Real pyramids are often
anisotropic — z frequently isn't downsampled at all — and that assumption is
exactly what misaligns skeletons against meshes.

Cluster sizing lives in `configs/`. `dask-slurm-any.yaml` deliberately sets no
`--constraint`, so jobs land on any free CPU node; the peak memory here is one
*block*, not one body, so the fat nodes buy nothing. See the comments in that
file before changing `--block`.

Status: pipelines implemented and tested; not yet run on production data.
See `docs/DESIGN.md`.

[`vol2mesh`]: https://github.com/janelia-flyem/vol2mesh
[`kimimaro`]: https://github.com/seung-lab/kimimaro

## License

©2026 The Simons Foundation, Inc.

Licensed under the [GNU General Public License v3.0 or later](LICENSE). GPL rather
than a permissive licence because this package imports `kimimaro` and `dijkstra3d`,
both GPL-3.0-or-later, which makes it a combined work — see the Licence note in
`em_seg_morpho/neutu_trace.py`. See [CONTRIBUTING.md](CONTRIBUTING.md) for how
contributions are licensed.
