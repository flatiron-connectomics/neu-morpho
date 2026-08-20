"""neu-morpho: segment morphology from segmentation volumes.

Generate multi-resolution Draco-encoded **meshes** (via ``vol2mesh``) and
**skeletons** (via ``kimimaro``) per segment, written in neuroglancer-precomputed
format, orchestrated across segments with ``blockrun`` and reading segmentation
arrays with ``neu-vol``.

Large segments (whose binary mask over a big bounding box would OOM) are meshed
**chunked and stitched** rather than materialized whole. See docs/DESIGN.md.
"""

__version__ = "0.1.0"
