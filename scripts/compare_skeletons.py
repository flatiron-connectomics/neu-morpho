"""Compare kimimaro vs skeletor skeletonization — synthetic bodies or real data.

    # synthetic bodies:
    pixi run -e dev python scripts/compare_skeletons.py [--shapes rod,rod_with_bulb,...]

    # a real body crop (both methods; needs the seg volume + the body's bbox):
    pixi run -e dev python scripts/compare_skeletons.py \
        --volume /mnt/ceph/.../seg --backend neuroglancer_precomputed --scale 2 \
        --body 231668 --bbox z0,y0,x0,z1,y1,x1 --voxel 32,32,32

    # a real mesh file (skeletor methods only):
    pixi run -e dev python scripts/compare_skeletons.py --mesh body231668.obj

Per body it prints a metrics table and writes to --out/<body>/:
  - comparison.html : self-contained interactive 3D — one panel per method, the
    translucent mesh + skeleton with branch points (red) and tips (black); open
    in a browser, rotate/zoom (convolution = extra red/black points).
  - <method>.swc    : also loadable in napari/navis.
Edit METHODS / INV to sweep parameters. --no-viz skips the HTML.
"""

from __future__ import annotations

import argparse
import os

from em_seg_morpho.skelcompare import (
    SHAPES, body_from_volume, mask_to_trimesh, mesh_from_file,
    run_kimimaro, run_skeletor, skeleton_stats, visualize, write_swc,
)

INV = 120  # teasar invalidation distance (nm); ~ a few x local radius. TUNE per data.
# (label, runner(mask, mesh, vs) -> SkelResult, needs_mask)
METHODS = [
    ("kimimaro",          lambda mask, mesh, vs: run_kimimaro(mask, vs, name="kimimaro"), True),
    ("wavefront",         lambda mask, mesh, vs: run_skeletor(mesh, "wavefront", per_component=True, step_size=1), False),
    ("wavefront+cleanup", lambda mask, mesh, vs: run_skeletor(mesh, "wavefront", per_component=True, step_size=1, post=("clean_up",)), False),
    ("teasar(whole)",     lambda mask, mesh, vs: run_skeletor(mesh, "teasar", per_component=False, inv_dist=INV), False),
    ("teasar(percomp)",   lambda mask, mesh, vs: run_skeletor(mesh, "teasar", per_component=True, inv_dist=INV), False),
    ("teasar+bristles",   lambda mask, mesh, vs: run_skeletor(mesh, "teasar", per_component=True, inv_dist=INV, post=("remove_bristles",)), False),
]
COLS = ["n_verts", "n_tips", "n_branch", "n_components", "cable_um", "sec"]


def _run_body(name, mask, mesh, vs, outdir, viz=True):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== {name}  (mesh {len(mesh.vertices)} verts, {mesh.body_count} components) ===")
    hdr = f"{'method':22}" + "".join(f"{c:>13}" for c in COLS) + "   twigs   note"
    print(hdr); print("-" * len(hdr))
    results = []
    for label, runner, needs_mask in METHODS:
        if needs_mask and mask is None:
            print(f"{label:22}{'(no mask)':>13}")
            continue
        res = runner(mask, mesh, vs)
        res.name = label
        results.append(res)
        st = skeleton_stats(res)
        if "error" in st:
            print(f"{label:22}{'ERROR':>13}   {st['error'][:50]}")
            continue
        row = f"{label:22}" + "".join(f"{st.get(c,''):>13}" for c in COLS)
        twig = next((v for k, v in st.items() if k.startswith("n_twigs")), "")
        print(row + f"{twig:>8}   mesh_comps={st.get('n_mesh_components','')}")
        write_swc(os.path.join(outdir, f"{label.replace('/', '_')}.swc"), res)
    if viz:
        html = visualize(results, mesh, os.path.join(outdir, "comparison.html"), title=name)
        if html:
            print(f"  -> {html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--out", default="skel_compare_out")
    ap.add_argument("--mesh", help="skeletonize a real mesh file (skeletor only)")
    ap.add_argument("--volume", help="segmentation path (real body: both methods)")
    ap.add_argument("--backend", default="neuroglancer_precomputed")
    ap.add_argument("--scale", type=int, default=0, help="precomputed scale_index for --volume")
    ap.add_argument("--body", type=int)
    ap.add_argument("--bbox", help="z0,y0,x0,z1,y1,x1 (in --scale voxels)")
    ap.add_argument("--voxel", default="8,8,8", help="z,y,x voxel size (nm) at --scale")
    ap.add_argument("--no-viz", action="store_true", help="skip the interactive HTML")
    args = ap.parse_args()
    viz = not args.no_viz

    if args.mesh:
        _run_body(os.path.basename(args.mesh), None, mesh_from_file(args.mesh),
                  tuple(float(x) for x in args.voxel.split(",")), os.path.join(args.out, "mesh"), viz)
    elif args.volume:
        spec = {"backend": args.backend, "path": args.volume}
        if args.backend == "neuroglancer_precomputed":
            spec["scale_index"] = args.scale
        bbox = tuple(int(x) for x in args.bbox.split(","))
        vs = tuple(float(x) for x in args.voxel.split(","))
        mask, mesh, vs = body_from_volume(spec, args.body, bbox, vs)
        _run_body(f"body{args.body}", mask, mesh, vs, os.path.join(args.out, f"body{args.body}"), viz)
    else:
        for shape in args.shapes.split(","):
            shape = shape.strip()
            mask, vs = SHAPES[shape]()
            _run_body(shape, mask, mask_to_trimesh(mask, vs), vs, os.path.join(args.out, shape), viz)
    print(f"\nOutputs under {args.out}/ : comparison.html (interactive) + per-method .swc")


if __name__ == "__main__":
    main()
