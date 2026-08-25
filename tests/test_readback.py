"""Reading a published volume back.

The round-trip tests are the reason this module exists in the package rather than in
a script: ``precomputed.py`` writes both formats and nothing else verified that what
it writes can be read back as what went in. A writer-only suite cannot see that,
because a wrong-but-consistent encoding passes every check it makes.
"""

import numpy as np
import pytest

from neu_morpho.config import MeshConfig
from neu_morpho.readback import read_body_mesh, read_body_skeleton


# --------------------------------------------------------------------------- #
# round trips: precomputed.write_* -> readback.read_*
# --------------------------------------------------------------------------- #
def _skeleton(n=12):
    from osteoid import Skeleton

    zyx = np.stack([np.arange(n, dtype=np.float32) * 40.0,
                    np.full(n, 100.0, np.float32),
                    np.full(n, 200.0, np.float32)], axis=1)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.uint32)
    skel = Skeleton(vertices=zyx, edges=edges, segid=7)
    skel.radii = (np.arange(n, dtype=np.float32) + 5.0)
    return skel


def test_skeleton_round_trip_preserves_geometry_and_radii(tmp_path):
    from neu_morpho import precomputed

    vol = str(tmp_path / "segmentation")
    precomputed.write_skeleton_info(vol + "/skeleton")
    skel = _skeleton()
    precomputed.write_body_skeleton(vol + "/skeleton", 7, skel)

    v, e, r = read_body_skeleton(vol, 7)
    assert len(v) == len(skel.vertices) and len(e) == len(skel.edges)
    np.testing.assert_allclose(r, np.asarray(skel.radii, float), rtol=0, atol=1e-4)
    # The writer flips zyx -> xyz on the way out, so the vertex that was written
    # z-varying must come back x-varying. Getting this wrong mirrors the skeleton
    # through the z=x diagonal and is invisible in any per-vertex count.
    written_zyx = np.asarray(skel.vertices, float)
    np.testing.assert_allclose(v[:, 0], written_zyx[:, 2], atol=1e-4)   # x <- x
    np.testing.assert_allclose(v[:, 2], written_zyx[:, 0], atol=1e-4)   # z <- z


def test_read_body_skeleton_is_none_when_absent(tmp_path):
    vol = str(tmp_path / "segmentation")
    from neu_morpho import precomputed
    precomputed.write_skeleton_info(vol + "/skeleton")
    assert read_body_skeleton(vol, 999) is None


def test_read_body_skeleton_rejects_the_no_radius_sentinel(tmp_path):
    """A skeleton whose info declares no radius attribute must raise.

    osteoid does not report this as missing — it returns one value per vertex filled
    with ``-1``. Right length, finite, and physically impossible, so only a sign
    check catches it. This test failed against a length-only guard, which is exactly
    the hole it exists to hold shut.
    """
    from osteoid import Skeleton

    from neu_morpho import precomputed

    vol = str(tmp_path / "segmentation")
    # info WITHOUT the radius attribute, and a body encoded to match
    precomputed.write_skeleton_info(vol + "/skeleton", vertex_attributes=[])
    n = 4
    skel = Skeleton(
        vertices=np.stack([np.arange(n, dtype=np.float32) * 10.0,
                           np.zeros(n, np.float32), np.zeros(n, np.float32)], axis=1),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.uint32),
        segid=3)
    skel.radii = None
    precomputed.write_body_skeleton(vol + "/skeleton", 3, skel)
    with pytest.raises(ValueError, match="negative radii"):
        read_body_skeleton(vol, 3)


def _cube_mesh(cfg):
    from neu_morpho.coords import physical_box
    from neu_morpho.mesh import assemble_body, mesh_block

    block = np.zeros((24, 24, 24), np.uint64)
    block[6:18, 6:18, 6:18] = 1
    meshes = mesh_block(block, physical_box((slice(0, 24),) * 3, (8, 8, 8)), cfg)
    return assemble_body([meshes[1]], cfg)


def test_mesh_round_trip_returns_the_written_geometry(tmp_path):
    from neu_morpho import precomputed

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    mesh = _cube_mesh(cfg)
    written = np.asarray(mesh.vertices_zyx, float)[:, ::-1]        # zyx -> xyz

    vol = str(tmp_path / "segmentation")
    chunk_xyz, origin_xyz = [192.0] * 3, [0.0, 0.0, 0.0]
    precomputed.write_mesh_info(vol + "/mesh", cfg)
    n = precomputed.write_body_multires(vol + "/mesh", 5, mesh, cfg,
                                        chunk_shape_xyz=chunk_xyz,
                                        grid_origin_xyz=origin_xyz)
    assert n > 0

    got = read_body_mesh(vol, 5, lod=0)
    assert got is not None
    v, f, lod = got
    assert lod == 0 and len(v) and len(f)
    assert f.max() < len(v), "face index out of range after fragment concatenation"
    # Draco is lossy by quantization, not by displacement: the decoded corner must
    # land on the written one to within the quantization step, not merely nearby.
    assert np.allclose(v.min(axis=0), written.min(axis=0), atol=2.0)
    assert np.allclose(v.max(axis=0), written.max(axis=0), atol=2.0)


def test_read_body_mesh_defaults_to_the_coarsest_lod(tmp_path):
    from neu_morpho import precomputed

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0,
                     num_lods=3)
    vol = str(tmp_path / "segmentation")
    precomputed.write_mesh_info(vol + "/mesh", cfg)
    precomputed.write_body_multires(vol + "/mesh", 5, _cube_mesh(cfg), cfg,
                                    chunk_shape_xyz=[192.0] * 3,
                                    grid_origin_xyz=[0.0, 0.0, 0.0])
    default_lod = read_body_mesh(vol, 5)[2]
    assert default_lod == max(
        read_body_mesh(vol, 5, lod=l)[2] for l in range(cfg.num_lods)
        if read_body_mesh(vol, 5, lod=l) is not None)
    with pytest.raises(ValueError, match="not present"):
        read_body_mesh(vol, 5, lod=99)


def test_read_body_mesh_is_none_when_absent(tmp_path):
    from neu_morpho import precomputed

    cfg = MeshConfig(mesh_scale=0)
    vol = str(tmp_path / "segmentation")
    precomputed.write_mesh_info(vol + "/mesh", cfg)
    assert read_body_mesh(vol, 999) is None


def test_require_radii_false_returns_a_centreline(tmp_path):
    """A source may legitimately publish centrelines only — FlyEM's male-CNS skeleton
    `info` is a bare {"@type": "neuroglancer_skeletons"} with no vertex_attributes, so
    every body's radii arrive as the -1 sentinel. Cable length and topology survive;
    calibre does not, and the caller has to know that rather than get -1 silently.
    """
    import numpy as np
    from osteoid import Skeleton as Osteoid

    from neu_morpho import readback

    vertices = np.array([[0.0, 0, 0], [100.0, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.uint32)
    blob = Osteoid(vertices=vertices, edges=edges,
                   radii=np.full(2, -1.0, dtype=np.float32),
                   vertex_types=np.zeros(2, dtype=np.uint8)).to_precomputed()
    root = tmp_path / "vol"
    (root / "skeleton").mkdir(parents=True)
    (root / "skeleton" / "7").write_bytes(blob)

    with pytest.raises(ValueError, match="require_radii=False"):
        readback.read_body_skeleton(str(root), 7)

    v, e, r = readback.read_body_skeleton(str(root), 7, require_radii=False)
    assert r is None
    assert len(v) == 2 and e.tolist() == [[0, 1]]


# --------------------------------------------------------------------------- #
# sharded meshes, and the info transform
# --------------------------------------------------------------------------- #

def _pack_mesh_shard(directory, spec, body_id, data, manifest):
    """Write a one-body sharded mesh the way the format lays one out.

    A test-only writer, and necessarily an independent one: the fragment data is **not an
    indexed entry**, so tensorstore's sharded driver cannot produce this file and
    ``sharded.write_all`` is no help. That independence is the point — it means the reader
    is being checked against the layout rather than against itself.

    Layout is ``[shard index][fragment data][manifest][minishard index]``, with the
    fragment data immediately before the manifest because that is the only way the
    manifest's offsets can address it.
    """
    import gzip
    import os

    import numpy as np
    from neu_vol import sharded

    shard, minishard = sharded.shard_location(body_id, spec)
    index_length = (1 << int(spec["minishard_bits"])) * 16
    stored = gzip.compress(manifest) if spec["data_encoding"] == "gzip" else manifest
    payload = data + stored

    # One entry, so every delta is its own absolute value. Offsets in both index levels
    # are relative to the END of the shard index, which is where the data section starts.
    flat = np.array([body_id, len(data), len(stored)], dtype="<u8").tobytes()
    if spec["minishard_index_encoding"] == "gzip":
        flat = gzip.compress(flat)

    table = np.zeros((1 << int(spec["minishard_bits"]), 2), dtype="<u8")
    table[minishard] = (len(payload), len(payload) + len(flat))

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, shard), "wb") as f:
        f.write(table.tobytes())
        f.write(payload)
        f.write(flat)


def _write_unsharded_mesh(tmp_path, cfg, body=5, transform=None):
    """The existing round-trip fixture: one cube meshed and written unsharded."""
    from neu_morpho import precomputed

    vol = str(tmp_path / "segmentation")
    precomputed.write_mesh_info(vol + "/mesh", cfg, transform=transform)
    n = precomputed.write_body_multires(vol + "/mesh", body, _cube_mesh(cfg), cfg,
                                        chunk_shape_xyz=[192.0] * 3,
                                        grid_origin_xyz=[0.0, 0.0, 0.0])
    assert n > 0
    return vol


def _reshard(vol, tmp_path, body=5, **spec_kw):
    """Copy an unsharded mesh into a sharded one, byte for byte."""
    import json

    from neu_vol import location, sharded

    spec = sharded.sharding_spec(**{"shard_bits": 2, "minishard_bits": 1, **spec_kw})
    data = location.read_bytes(vol, "mesh", str(body))
    manifest = location.read_bytes(vol, "mesh", f"{body}.index")
    assert data is not None and manifest is not None

    out = str(tmp_path / "sharded")
    info = json.loads(location.read_bytes(vol, "mesh", "info"))
    info["sharding"] = spec
    location.write_bytes(out + "/mesh", json.dumps(info).encode(), "info")
    _pack_mesh_shard(out + "/mesh", spec, body, data, manifest)
    return out, spec


def test_a_sharded_mesh_reads_back_as_the_same_geometry(tmp_path):
    """The bug this fixes: a sharded store has no `<body>` object, so the unsharded
    reader returned None for every body — indistinguishable from a volume with no meshes.
    """
    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    plain = read_body_mesh(vol, 5, lod=0)
    assert plain is not None

    out, _ = _reshard(vol, tmp_path)
    got = read_body_mesh(out, 5, lod=0)
    assert got is not None, "sharded mesh read back as absent"
    np.testing.assert_array_equal(got[0], plain[0])
    np.testing.assert_array_equal(got[1], plain[1])
    assert got[2] == plain[2]


@pytest.mark.parametrize("encoding", ["gzip", "raw"])
def test_a_sharded_mesh_reads_under_either_data_encoding(tmp_path, encoding):
    """`data_encoding` compresses the manifest but not the fragment data, so the
    manifest's byte offsets are into the STORED bytes. Getting that backwards puts the
    fragment read at the wrong place under gzip only."""
    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    plain = read_body_mesh(vol, 5, lod=0)

    out, _ = _reshard(vol, tmp_path, data_encoding=encoding,
                      minishard_index_encoding=encoding)
    got = read_body_mesh(out, 5, lod=0)
    assert got is not None
    np.testing.assert_array_equal(got[0], plain[0])


def test_the_packed_shard_is_readable_by_tensorstore_too(tmp_path):
    """Validates the fixture itself, and with it the shard placement: if tensorstore's
    own driver finds the manifest at the key we hashed it to, the addressing is right."""
    from neu_vol import sharded

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    out, spec = _reshard(vol, tmp_path)

    from neu_vol import location
    expected = location.read_bytes(vol, "mesh", "5.index")
    assert sharded.read_one(out + "/mesh", spec, 5) == expected


def test_a_sharded_mesh_missing_a_body_is_none(tmp_path):
    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    out, _ = _reshard(vol, tmp_path)
    assert read_body_mesh(out, 999, lod=0) is None


def test_a_manifest_pointing_past_its_fragment_data_raises(tmp_path):
    """Truncated fragment data must not decode into a partial mesh. The manifest states
    the total, so a short read is detectable and is a corrupt store, not a missing body."""
    import json

    from neu_vol import location, sharded

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=1)
    data = location.read_bytes(vol, "mesh", "5")
    manifest = location.read_bytes(vol, "mesh", "5.index")

    out = str(tmp_path / "truncated")
    info = json.loads(location.read_bytes(vol, "mesh", "info"))
    info["sharding"] = spec
    location.write_bytes(out + "/mesh", json.dumps(info).encode(), "info")
    _pack_mesh_shard(out + "/mesh", spec, 5, data[:-64], manifest)

    with pytest.raises(ValueError, match="not a well-formed sharded mesh"):
        read_body_mesh(out, 5, lod=0)


def test_the_info_transform_is_applied_so_vertices_are_nm(tmp_path):
    """A foreign source may store model coordinates and declare the scale to nm in its
    `transform`. FlyEM's male-CNS declares diag(16); ignoring it returns meshes 16x too
    small and 16x out of register with the skeletons from the same volume, silently."""
    from neu_morpho.readback import mesh_transform

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    plain = read_body_mesh(_write_unsharded_mesh(tmp_path, cfg), 5, lod=0)

    scaled_dir = tmp_path / "scaled"
    transform = [4, 0, 0, 0,
                 0, 4, 0, 0,
                 0, 0, 4, 0]
    scaled = _write_unsharded_mesh(scaled_dir, cfg, transform=transform)
    got = read_body_mesh(scaled, 5, lod=0)

    np.testing.assert_allclose(got[0], plain[0] * 4)
    np.testing.assert_array_equal(mesh_transform(scaled)[:, :3], np.eye(3) * 4)


def test_mesh_transform_refuses_a_location_with_no_info(tmp_path):
    """Identity is a plausible answer, so a mistyped path must not come back as one."""
    from neu_morpho.readback import mesh_transform

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    with pytest.raises(FileNotFoundError, match="no mesh info"):
        mesh_transform(vol + "/mesh")           # volume and subdirectory conflated


def test_an_identity_transform_leaves_our_own_output_untouched(tmp_path):
    """Everything this package writes is already nm with an identity transform, so the
    transform step must be exactly a no-op there."""
    from neu_morpho.readback import mesh_transform

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    np.testing.assert_array_equal(mesh_transform(vol),
                                  np.hstack([np.eye(3), np.zeros((3, 1))]))
    assert read_body_mesh(vol, 5, lod=0) is not None


def test_a_targeted_lod_read_equals_reading_the_whole_pyramid(tmp_path):
    """The sharded path fetches only the requested LOD's byte range and re-emits a
    manifest to match. Every LOD must come back exactly as the unsharded reader — which
    parses the whole object — returns it, or the range arithmetic is off by a fragment.
    """
    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0,
                     num_lods=3)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    out, _ = _reshard(vol, tmp_path)

    checked = 0
    for lod in range(cfg.num_lods):
        plain = read_body_mesh(vol, 5, lod=lod)
        if plain is None:
            continue
        got = read_body_mesh(out, 5, lod=lod)
        assert got is not None, f"LOD {lod} missing from the sharded copy"
        np.testing.assert_array_equal(got[0], plain[0])
        np.testing.assert_array_equal(got[1], plain[1])
        assert got[2] == plain[2] == lod
        checked += 1
    assert checked > 1, "fixture produced one LOD, so this proved nothing"

    # And the default — coarsest present — must agree too, without being told which.
    np.testing.assert_array_equal(read_body_mesh(out, 5)[0], read_body_mesh(vol, 5)[0])
    assert read_body_mesh(out, 5)[2] == read_body_mesh(vol, 5)[2]


def test_a_truncated_manifest_keeps_the_lod_numbering(tmp_path):
    """A fragment's cell size is `chunk_shape * 2**lod` from where it sits in the
    manifest, so re-emitting LOD 2 as LOD 0 would dequantize against a cell 4x too small.
    The re-emitted manifest must therefore still declare the earlier, empty LODs."""
    from neu_morpho.readback import _parse_manifest, _single_lod_manifest

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0,
                     num_lods=3)
    vol = _write_unsharded_mesh(tmp_path, cfg)

    from neu_vol import location
    parsed = _parse_manifest(location.read_bytes(vol, "mesh", "5.index"))
    present = [i for i, c in enumerate(parsed["counts"]) if int(c)]
    want = present[-1]

    rebuilt = _parse_manifest(_single_lod_manifest(parsed, want))
    assert rebuilt["num_lods"] == want + 1, "earlier LODs were dropped, not emptied"
    assert list(rebuilt["counts"][:want]) == [0] * want
    assert int(rebuilt["counts"][want]) == int(parsed["counts"][want])
    assert rebuilt["head"] == parsed["head"], "chunk_shape/grid_origin must survive"
    assert rebuilt["blocks"][want] == parsed["blocks"][want]


def test_the_shard_indexes_are_parsed_once_per_volume_not_once_per_body(tmp_path):
    """Reading a batch must not re-parse the shard index per body: `preshift_bits` puts
    runs of consecutive ids in one shard precisely so a batch concentrates there."""
    from neu_morpho import readback

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    out, _ = _reshard(vol, tmp_path)

    readback._SHARD_READERS.clear()
    read_body_mesh(out, 5, lod=0)
    first = len(readback._SHARD_READERS)
    for _ in range(5):
        read_body_mesh(out, 5, lod=0)
    assert first == 1 and len(readback._SHARD_READERS) == 1


# --------------------------------------------------------------------------- #
# sharded skeletons
# --------------------------------------------------------------------------- #

def _write_skeleton_volume(tmp_path, body=7, transform=None):
    from neu_morpho import precomputed

    vol = str(tmp_path / "segmentation")
    precomputed.write_skeleton_info(vol + "/skeleton", transform=transform)
    precomputed.write_body_skeleton(vol + "/skeleton", body, _skeleton())
    return vol


def _reshard_skeleton(vol, tmp_path, body=7, **spec_kw):
    """A sharded skeleton is an ordinary keyed entry, so tensorstore can write one."""
    import json

    from neu_vol import location, sharded

    spec = sharded.sharding_spec(**{"shard_bits": 2, "minishard_bits": 1, **spec_kw})
    blob = location.read_bytes(vol, "skeleton", str(body))
    assert blob is not None

    out = str(tmp_path / "sharded")
    info = json.loads(location.read_bytes(vol, "skeleton", "info"))
    info["sharding"] = spec
    location.write_bytes(out + "/skeleton", json.dumps(info).encode(), "info")
    sharded.write_all(out + "/skeleton", spec, [(body, blob)])
    return out, spec


def test_a_sharded_skeleton_reads_back_as_the_same_geometry(tmp_path):
    cfg_vol = _write_skeleton_volume(tmp_path)
    plain = read_body_skeleton(cfg_vol, 7)
    out, _ = _reshard_skeleton(cfg_vol, tmp_path)

    got = read_body_skeleton(out, 7)
    assert got is not None, "sharded skeleton read back as absent"
    for a, b in zip(got, plain):
        np.testing.assert_array_equal(a, b)


def test_a_sharded_skeleton_missing_a_body_is_none(tmp_path):
    vol = _write_skeleton_volume(tmp_path)
    out, _ = _reshard_skeleton(vol, tmp_path)
    assert read_body_skeleton(out, 999) is None


def test_the_skeleton_info_transform_is_applied_to_vertices_and_radii(tmp_path):
    """Vertices and radii must move together. Scaling one and not the other returns a
    neuron of plausible shape and wrong calibre, which no shape or length check sees."""
    from neu_morpho.readback import skeleton_transform

    plain = read_body_skeleton(_write_skeleton_volume(tmp_path), 7)
    scaled = _write_skeleton_volume(tmp_path / "scaled",
                                    transform=[3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0])
    got = read_body_skeleton(scaled, 7)

    np.testing.assert_allclose(got[0], plain[0] * 3)
    np.testing.assert_array_equal(got[1], plain[1])
    np.testing.assert_allclose(got[2], plain[2] * 3)
    np.testing.assert_array_equal(skeleton_transform(scaled)[:, :3], np.eye(3) * 3)


def test_a_non_uniform_skeleton_transform_refuses_to_invent_a_radius(tmp_path):
    """There is no single radius under an anisotropic scale, so returning one would be a
    guess presented as a measurement."""
    scaled = _write_skeleton_volume(tmp_path,
                                    transform=[2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0])
    with pytest.raises(ValueError, match="non-uniform transform"):
        read_body_skeleton(scaled, 7)
    # The centreline itself is still well defined, so asking for one must still work.
    v, e, r = read_body_skeleton(scaled, 7, require_radii=False)
    assert r is None and len(v) and len(e)


def test_the_skeleton_transform_matches_the_mesh_one_on_our_own_output(tmp_path):
    """Both identity, which is what keeps a body's skeleton and its surface in one space."""
    from neu_morpho.readback import mesh_transform, skeleton_transform

    cfg = MeshConfig(mesh_scale=0, block_shape=(24, 24, 24), decimation_fraction=1.0)
    vol = _write_unsharded_mesh(tmp_path, cfg)
    from neu_morpho import precomputed
    precomputed.write_skeleton_info(vol + "/skeleton")
    np.testing.assert_array_equal(mesh_transform(vol), skeleton_transform(vol))
