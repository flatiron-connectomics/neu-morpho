#!/usr/bin/env python
"""Export per-body boolean masks for the skeletonization benchmark.

Reads the manifest written by ``pick_benchmark_bodies.py``, crops each body out
of the segmentation at ``--scale``, and writes ``body_<id>_mask.npy`` (bool, zyx)
plus an updated manifest carrying the true crop and voxel count.

**The crop bbox comes from the body's stage-1 mesh fragments, not the manifest's
skeleton bbox.** The manifest's bbox is a selection heuristic — it is derived from
skeleton fragments, and stage 1 skeletonization drops per-block components below
``dust_threshold`` (50 voxels), so a body's small detached specks have no skeleton
and lie outside it. Meshing applies no such threshold, so the mesh bbox is the
body's true extent. Verified on body 6308993: the mesh bbox reproduces the
448,672,289-voxel extent recorded in ``docs/skeletonization.md``,
whereas the skeleton bbox does not.

Masks go to ceph, not the repo — a single one of these is 450 MB.

    python scripts/export_benchmark_masks.py --manifest bodies.json \\
        --src <seg path> --out-dir /mnt/ceph/.../masks
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np


def mesh_bbox_vox(chunked_dir: str, body: int, voxel_nm: float):
    """True bbox in scale voxels from a body's stage-1 mesh fragments (zyx nm)."""
    from vol2mesh import Mesh

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n = 0
    for p in glob.glob(os.path.join(chunked_dir, str(int(body)), "*.drc")):
        v = Mesh.from_file(p).vertices_zyx
        n += 1
        if len(v):
            lo = np.minimum(lo, v.min(0))
            hi = np.maximum(hi, v.max(0))
    if not n or not np.isfinite(lo).all():
        return None
    return lo / voxel_nm, hi / voxel_nm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--src", required=True, help="segmentation source path")
    ap.add_argument("--out-dir", required=True, help="where masks go (use ceph)")
    ap.add_argument("--work-dir", help="production work dir holding chunked/ "
                                       "(default: the manifest's source_work_dir)")
    ap.add_argument("--margin-vox", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from neu_vol import read_scales, scale_spec
    from neu_vol.backends.base import open_backend
    from neu_vol.logs import quiet_store_logs

    with open(a.manifest) as f:
        man = json.load(f)
    work_dir = a.work_dir or man["source_work_dir"]
    chunked = os.path.join(work_dir, "chunked")
    os.makedirs(a.out_dir, exist_ok=True)

    with quiet_store_logs():
        scales = read_scales(a.src)
        si = scales[man["scale"]]
        vox = si.voxel_size[0]
        if abs(vox - man["voxel_nm"]) > 1e-6:
            raise SystemExit(f"manifest says {man['voxel_nm']} nm/vox but scale "
                             f"{man['scale']} is {si.voxel_size} nm")
        backend = open_backend(scale_spec(a.src, si.index))

        print(f"scale {si.index}: shape {si.shape} @ {si.voxel_size} nm")
        hdr = (f"{'body':>10s} {'extent(zyx)':>20s} {'bbox_vox':>14s} {'mask_vox':>12s} "
               f"{'fill%':>6s} {'MB':>7s} {'s':>6s}")
        print(hdr)
        print("-" * len(hdr))

        for rec in man["bodies"]:
            body = rec["body_id"]
            out = os.path.join(a.out_dir, f"body_{body}_mask.npy")
            if os.path.exists(out) and not a.overwrite:
                print(f"{body:10d}  exists, skipping (--overwrite to redo)")
                continue

            bb = mesh_bbox_vox(chunked, body, vox)
            if bb is None:
                print(f"{body:10d}  NO MESH FRAGMENTS — skipped")
                rec["error"] = "no mesh fragments"
                continue
            lo = np.maximum(0, np.floor(bb[0]).astype(int) - a.margin_vox)
            hi = np.minimum(np.array(si.shape, dtype=int),
                            np.ceil(bb[1]).astype(int) + a.margin_vox + 1)

            t0 = time.time()
            region = backend.read_region(tuple(slice(int(l), int(h))
                                               for l, h in zip(lo, hi)))
            mask = np.ascontiguousarray(region == body)
            del region
            dt = time.time() - t0
            n = int(mask.sum())
            if n == 0:
                print(f"{body:10d}  EMPTY CROP — body id absent from the region")
                rec["error"] = "empty crop"
                continue

            # A body voxel on the crop face means the bbox was too tight and the
            # arbor is being truncated. Faces flush with the volume bound are fine.
            touched = []
            for ax in range(3):
                if lo[ax] > 0 and mask.take(0, axis=ax).any():
                    touched.append(f"-{'zyx'[ax]}")
                if hi[ax] < si.shape[ax] and mask.take(-1, axis=ax).any():
                    touched.append(f"+{'zyx'[ax]}")
            if touched:
                print(f"{body:10d}  WARNING truncated at {','.join(touched)}")
                rec["truncated_faces"] = touched

            np.save(out, mask)
            ext = (hi - lo).tolist()
            rec.update(crop_lo_vox=lo.tolist(), crop_hi_vox=hi.tolist(),
                       crop_vox=int(np.prod(hi - lo)), mask_voxels=n,
                       mask_path=out, read_seconds=round(dt, 1))
            print(f"{body:10d} {str(ext):>20s} {np.prod(hi-lo):14,.0f} {n:12,d} "
                  f"{100*n/np.prod(hi-lo):6.2f} {os.path.getsize(out)/1e6:7.1f} {dt:6.1f}")

    man["mask_dir"] = a.out_dir
    man["src"] = a.src
    with open(a.manifest, "w") as f:
        json.dump(man, f, indent=1)
    ok = sum(1 for r in man["bodies"] if "mask_path" in r)
    print(f"\n{ok}/{len(man['bodies'])} masks -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
