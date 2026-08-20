"""neu-morpho: segment morphology from segmentation volumes.

Generate multi-resolution Draco-encoded **meshes** (via ``vol2mesh``) and
**skeletons** (TEASAR: ``neutu_trace`` by default, ``kimimaro`` for anisotropic
voxels) per segment, written in neuroglancer-precomputed format, orchestrated
across segments with ``blockrun`` and reading segmentation arrays with
``neu-vol``.

Large segments (whose binary mask over a big bounding box would OOM) are meshed
**chunked and stitched** rather than materialized whole. See the README, and
docs/skeletonization.md for the tracer's settings.
"""

__version__ = "0.1.0"
