"""End-to-end: a synthetic labeled volume -> chunk -> assemble -> multires meshes.

Verifies the whole block-first pipeline runs serially, that a cross-block body is
assembled from both of its block fragments, and — via the multires fragment
*grid-cell positions* — that bodies land at the correct nm locations (alignment).
"""

import os

import numpy as np

from em_seg_morpho.config import MeshConfig, OutputConfig
from em_seg_morpho.ops.meshify import meshify


def _write_seg_zarr(path, vol):
    from em_volume_tools.backends.tensorstore import TensorStoreBackend
    from em_volume_tools.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region((slice(0, vol.shape[0]), slice(0, vol.shape[1]), slice(0, vol.shape[2])), vol)


def _lod0_cells(mesh_dir, body):
    from vol2mesh import multires
    res = multires.read_object_mesh(mesh_dir, body)
    return {tuple(int(c) for c in fr["position"]) for fr in res["fragments"] if fr["lod"] == 0}


def test_meshify_end_to_end(tmp_path):
    # voxel 8 nm, block 16 -> LOD-0 octree cell = 16*8 = 128 nm
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100        # block (0,0,0); nm [32:96] -> cell (0,0,0)
    vol[20:28, 20:28, 20:28] = 200     # block (1,1,1); nm [160:224] -> cell (1,1,1)
    vol[12:20, 6:10, 6:10] = 300       # crosses z-block boundary: blocks (0,0,0)+(1,0,0)

    src = str(tmp_path / "seg.zarr")
    _write_seg_zarr(src, vol)
    out = OutputConfig(dst=str(tmp_path / "out"))
    cfg = MeshConfig(mesh_scale=0, block_shape=(16, 16, 16), num_lods=2, decimation_fraction=1.0)

    summary = meshify({"backend": "zarr3", "path": src}, out, cfg,
                      mesh_voxel_size=(8, 8, 8), client=None)

    mesh_dir = summary["out_dir"]
    assert os.path.exists(os.path.join(mesh_dir, "info"))
    for body in (100, 200, 300):
        assert os.path.exists(os.path.join(mesh_dir, str(body)))
        assert os.path.exists(os.path.join(mesh_dir, f"{body}.index"))

    # alignment: LOD-0 fragment cells match each body's nm location
    assert (0, 0, 0) in _lod0_cells(mesh_dir, 100)
    assert (1, 1, 1) in _lod0_cells(mesh_dir, 200)

    # cross-block body: assembled from 2 block fragments, spanning 2 z-cells
    assert len(os.listdir(os.path.join(summary["chunked_dir"], "300"))) == 2
    cells300 = _lod0_cells(mesh_dir, 300)
    assert {(0, 0, 0), (0, 0, 1)} <= cells300      # xyz: spans z = 0 and 1

    assert summary["status_counts"].get("written", 0) >= 3
