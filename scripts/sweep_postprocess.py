"""Measure what postprocess throws away, across dust/tick thresholds.

The defaults kimimaro ships (dust 1500 nm, tick 3000 nm) are large enough to
delete small bodies outright, and there is no way to pick them from first
principles — so measure. For each threshold pair this reports the components and
cable removed, the bodies deleted entirely, and how much of the joined cable was
*inferred* rather than measured.

    # real data (bodies come from the metrics DB built by index_segments):
    pixi run -e dev python scripts/sweep_postprocess.py \
        --volume /mnt/ceph/.../seg --backend neuroglancer_precomputed --scale 2 \
        --voxel 32,32,32 --db metrics.db --limit 50

    # already-chunked fragments from a skel-chunk run (no re-skeletonization):
    pixi run -e dev python scripts/sweep_postprocess.py --chunked out/skel_chunked

    # synthetic arbors, no data needed:
    pixi run -e dev python scripts/sweep_postprocess.py --synthetic

``--dust`` / ``--tick`` take comma-separated nm values. The row where
``drop%`` starts climbing steeply is where the threshold has begun eating real
morphology rather than noise.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from em_seg_morpho.config import SkeletonConfig
from em_seg_morpho.skeleton import fuse_body, fusion_stats_summary, skeletonize_block

DUSTS = [0.0, 250.0, 500.0, 1000.0, 1500.0, 3000.0]
TICKS = [0.0, 1000.0, 3000.0]


def _parse_floats(s):
    return [float(x) for x in s.split(",")]


# --------------------------------------------------------------------------- #
# Fragment sources
# --------------------------------------------------------------------------- #
def frags_from_chunked(chunked_dir, cfg, limit=None):
    """Reuse fragments a skel-chunk run already wrote."""
    from em_seg_morpho import fragments as _frag

    bodies = _frag.list_bodies(chunked_dir)
    if limit:
        bodies = bodies[:limit]
    for body in bodies:
        yield body, _frag.read_body_skel_fragments(chunked_dir, body, cfg.fragment_format)


def frags_from_volume(seg_spec, db_path, cfg, limit, min_voxels):
    """Skeletonize each body's blocks straight from the volume, via the metrics DB."""
    from em_volume_tools.backends.base import open_backend

    from em_seg_morpho.metrics_db import MetricsDB

    be = open_backend(seg_spec)
    shape = be.shape
    factor = tuple(int(round(a / b)) for a, b in zip(cfg.anisotropy, cfg.anisotropy))
    db = MetricsDB(db_path)
    bodies = db.bodies_by_size(min_voxels=min_voxels, limit=limit)
    for body in bodies:
        crop = db.crop_at_scale(body, factor_zyx=factor, margin_vox=cfg.bbox_margin_vox,
                                clip_shape=shape)
        if crop is None:
            continue
        z0, y0, x0, z1, y1, x1 = crop
        # walk the body's bbox on the block grid, so this matches what skel-chunk does
        out = []
        bz, by, bx = cfg.block_shape
        for zs in range(z0 // bz * bz, z1, bz):
            for ys in range(y0 // by * by, y1, by):
                for xs in range(x0 // bx * bx, x1, bx):
                    region = (slice(zs, min(zs + bz, shape[0])),
                              slice(ys, min(ys + by, shape[1])),
                              slice(xs, min(xs + bx, shape[2])))
                    blk = be.read_region(region)
                    if not (blk == body).any():
                        continue
                    got = skeletonize_block(blk, (zs, ys, xs), cfg, allowlist={body})
                    if body in got:
                        out.append(got[body])
        if out:
            yield body, out
    db.close()


def noisy_arbor(length=140, r=3, pad=8, n_twigs=6, twig_len=6, n_specks=4, seed=0):
    """A branched process with short side twigs and detached specks.

    The clean shapes in ``skelcompare.SHAPES`` have neither, so they cannot show
    what dust/tick thresholds do. Twigs stand in for boundary roughness in an
    imperfect segmentation (what ``tick_threshold`` prunes); specks stand in for
    detached debris carrying the body's label (what ``dust_threshold`` deletes).
    """
    rng = np.random.default_rng(seed)
    w = 2 * (pad + 16)
    s = (length + 2 * pad, w, w)
    c = w // 2
    zz, yy, xx = np.indices(s)
    m = ((yy - c) ** 2 + (xx - c) ** 2 <= r ** 2) & (zz >= pad) & (zz < pad + length)

    for i in range(n_twigs):                     # short lateral twigs off the trunk
        z = pad + int((i + 0.5) * length / n_twigs)
        ln = twig_len + int(rng.integers(0, 3))
        m |= ((np.abs(zz - z) <= 1) & (xx == c) &
              (yy >= c + r) & (yy < c + r + ln))

    for i in range(n_specks):                    # detached debris with the same label
        z = pad + int((i + 0.5) * length / n_specks)
        oy = c + 10 + int(rng.integers(0, 4))
        m |= ((np.abs(zz - z) <= 1) & (np.abs(yy - oy) <= 1) & (np.abs(xx - c) <= 1))

    return m.astype(np.uint64)


def frags_synthetic(cfg):
    """Synthetic bodies, block-split the way skel-chunk would split them."""
    from em_seg_morpho.skelcompare import SHAPES

    bodies = {name: fn()[0] for name, fn in SHAPES.items()}
    bodies["noisy_arbor"] = noisy_arbor()
    for name, mask in bodies.items():
        vol = (np.asarray(mask) > 0).astype(np.uint64)
        bz = cfg.block_shape[0]
        frags = []
        for z0 in range(0, vol.shape[0], bz):
            blk = vol[z0:z0 + bz]
            if not blk.any():
                continue
            got = skeletonize_block(blk, (z0, 0, 0), cfg, allowlist={1})
            if 1 in got:
                frags.append(got[1])
        if frags:
            yield name, frags


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("fragment source (pick one)")
    src.add_argument("--chunked", help="a skel-chunk fragment dir")
    src.add_argument("--volume", help="segmentation path")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--backend", default="zarr3")
    p.add_argument("--scale", type=int, default=None, help="precomputed scale_index")
    p.add_argument("--voxel", default="8,8,8", help="skeleton-scale voxel size, nm zyx")
    p.add_argument("--block", default="256,256,256")
    p.add_argument("--db", help="metrics DB (required with --volume)")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--min-voxels", type=int, default=0)
    p.add_argument("--dust", type=_parse_floats, default=DUSTS)
    p.add_argument("--tick", type=_parse_floats, default=TICKS)
    p.add_argument("--jsonl", help="also write per-body rows for the default thresholds")
    args = p.parse_args()

    voxel = tuple(_parse_floats(args.voxel))
    block = tuple(int(x) for x in args.block.split(","))
    base = SkeletonConfig(anisotropy=voxel, block_shape=block)

    if args.chunked:
        source = list(frags_from_chunked(args.chunked, base, args.limit))
    elif args.volume:
        if not args.db:
            p.error("--volume requires --db (build it with ops.index_segments)")
        spec = {"backend": args.backend, "path": args.volume}
        if args.scale is not None:
            spec["scale_index"] = args.scale
        source = list(frags_from_volume(spec, args.db, base, args.limit, args.min_voxels))
    elif args.synthetic:
        source = list(frags_synthetic(base))
    else:
        p.error("pick one of --chunked / --volume / --synthetic")

    print(f"{len(source)} bodies, voxel={voxel} nm, block={block}\n")
    hdr = (f"{'dust nm':>8}{'tick nm':>9}{'bodies':>8}{'deleted':>9}{'multi':>7}"
           f"{'dustC':>7}{'ticks':>7}{'cable in µm':>13}{'out µm':>10}"
           f"{'drop%':>8}{'inferred%':>11}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for tick in args.tick:
        for dust in args.dust:
            cfg = replace(base, postprocess_dust_nm=dust, postprocess_tick_nm=tick)
            per_body = []
            for _body, frags in source:
                st = {}
                fuse_body(frags, cfg, stats=st)
                per_body.append(st)
            s = fusion_stats_summary(per_body)
            print(f"{dust:8.0f}{tick:9.0f}{s['n_bodies']:8d}{s['bodies_deleted']:9d}"
                  f"{s['bodies_multi_component']:7d}{s['dust_comps_dropped']:7d}"
                  f"{s['tick_branches_removed']:7d}{s['cable_joined_nm']/1000:13.1f}"
                  f"{s['cable_out_nm']/1000:10.1f}{100*s['dropped_cable_fraction']:8.2f}"
                  f"{100*s['inferred_cable_fraction']:11.2f}")
            rows.append((dust, tick, s))

    if args.jsonl:
        import json
        cfg = base
        with open(args.jsonl, "w") as f:
            for body, frags in source:
                st = {"body_id": body}
                fuse_body(frags, cfg, stats=st)
                f.write(json.dumps(st) + "\n")
        print(f"\nper-body rows -> {args.jsonl}")

    print("\ndustC = components deleted by the dust threshold; ticks = side branches pruned.")
    print("drop%     = (dust + tick cable) / cable entering fusion.")
    print("inferred% = cable ADDED by the two joins / cable entering fusion —")
    print("            edges inferred across gaps, not measured from the segmentation.")


if __name__ == "__main__":
    main()
