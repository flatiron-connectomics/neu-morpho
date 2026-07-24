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

```bash
pixi install && pixi run -e dev test
```

Status: scaffolding. See `docs/DESIGN.md`.

[`vol2mesh`]: https://github.com/janelia-flyem/vol2mesh
[`kimimaro`]: https://github.com/seung-lab/kimimaro
