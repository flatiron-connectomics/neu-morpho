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


def test_mesh_metrics_land_in_db(tmp_path):
    """Meshing enriches the same per-body DB skeletonization writes to."""
    from em_seg_morpho.metrics_db import MetricsDB

    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 100          # a cube inside one block
    vol[12:20, 6:10, 6:10] = 300         # crosses the z-block boundary

    src = str(tmp_path / "seg.zarr")
    _write_seg_zarr(src, vol)
    out = OutputConfig(dst=str(tmp_path / "out"))
    db_path = str(tmp_path / "metrics.db")
    cfg = MeshConfig(mesh_scale=0, block_shape=(16, 16, 16), num_lods=2, decimation_fraction=1.0)

    meshify({"backend": "zarr3", "path": src}, out, cfg, mesh_voxel_size=(8, 8, 8),
            db_path=db_path, client=None)

    db = MetricsDB(db_path)
    rows = {r[0]: r[1:] for r in db.con.execute(
        "SELECT body_id, mesh_area_nm2, mesh_verts, n_mesh_components FROM bodies")}
    db.close()

    assert set(rows) == {100, 300}
    for body, (area, verts, comps) in rows.items():
        assert area > 0 and verts > 0, (body, area, verts)
        # the spanning body must come out STITCHED into one component, not two
        assert comps == 1, (body, comps)

    # a 64 nm cube: 6 faces of 64^2 = 24576 nm^2, modulo marching-cubes rounding
    assert 0.7 * 24576 < rows[100][0] < 1.4 * 24576


def test_mesh_metrics_are_computed_before_lod_decimation():
    """Metrics describe LOD 0, not whatever the multires writer decimated it to."""
    from em_seg_morpho.mesh import assemble_body, mesh_block, mesh_metrics
    from em_seg_morpho.coords import physical_box

    block = np.zeros((16, 16, 16), np.uint64)
    block[4:12, 4:12, 4:12] = 1
    cfg = MeshConfig(mesh_scale=0, block_shape=(16, 16, 16), decimation_fraction=1.0)
    meshes = mesh_block(block, physical_box((slice(0, 16),) * 3, (8, 8, 8)), cfg)
    mesh = assemble_body([meshes[1]], cfg)

    before = mesh_metrics(mesh)
    mesh.simplify(0.5)                    # what write_body_multires does per LOD
    after = mesh_metrics(mesh)
    assert after["mesh_verts"] < before["mesh_verts"]


def test_count_components_sees_unstitched_fragments():
    """The count is QC: it reports >1 when block fragments failed to weld."""
    from vol2mesh import concatenate_meshes

    from em_seg_morpho.coords import physical_box
    from em_seg_morpho.mesh import count_components, mesh_block

    vol = np.zeros((32, 16, 16), np.uint64)
    vol[4:28, 6:10, 6:10] = 1
    cfg = MeshConfig(mesh_scale=0, block_shape=(16, 16, 16), decimation_fraction=1.0)
    frags = []
    for z0 in (0, 16):
        box = physical_box((slice(z0, z0 + 16), slice(0, 16), slice(0, 16)), (8, 8, 8))
        frags.append(mesh_block(vol[z0:z0 + 16], box, cfg)[1])

    merged = concatenate_meshes(frags)
    assert count_components(merged) == 2       # concatenated only
    merged.stitch_adjacent_faces()
    assert count_components(merged) == 1       # welded
