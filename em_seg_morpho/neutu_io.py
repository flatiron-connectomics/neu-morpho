"""Talk to NeuTu: the ``.sobj`` sparse-mask format, SWC, and the headless CLI.

Needed by **both** routes under consideration (see ``docs/skeletonization-plan.md``):

- the *plugin* route shells out to NeuTu and reads back its SWC;
- the *reimplementation* route needs NeuTu's SWCs as the regression target that a
  Python NeuTu-style skeletonizer is diffed against.

``.sobj`` is NeuTu's ``ZObject3dScan`` on-disk form — run-length encoded along x,
which is why a 449 MB dense mask becomes a 1.1 MB file. There is no header and no
version field; the format is, all ``int32`` little-endian
(``NeuTu/neurolabi/gui/zobject3dscan.cpp:872``, ``zobject3dstripe.cpp:177``)::

    nstripes
    per stripe:  z, y, nseg, then 2*nseg values (x_start, x_end)   # INCLUSIVE

CLI gotcha, verified the hard way: NeuTu's argument parser (genelib
``Process_Arguments``) **segfaults** when a positional input is followed by any
option that takes a value — the ``[<input:string> ...]`` entry in the spec
(``NeuTu/neurolabi/gui/zcommandline.cpp:1658``) is greedy. Valued options must come
*before* the positional input. Bare flags after it are fine. :func:`run_neutu`
builds the argv in the safe order.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np


# --------------------------------------------------------------------------- #
# .sobj
# --------------------------------------------------------------------------- #
def write_sobj(mask_zyx: np.ndarray, path: str) -> int:
    """Write a boolean ``[z,y,x]`` mask as NeuTu ``.sobj``. Returns the stripe count.

    Vectorised: one global run-length pass, then a single buffer assembly. A
    ~10^6-voxel body (~10^5 stripes) takes well under a second; the naive
    per-stripe loop took minutes.
    """
    m = np.asarray(mask_zyx).astype(bool)
    _, ny, nx = m.shape
    rows = m.reshape(-1, nx)
    ridx = np.nonzero(rows.any(axis=1))[0]          # flat (z*ny + y) per stripe
    sub = rows[ridx]

    p = np.zeros((sub.shape[0], nx + 2), dtype=np.int8)
    p[:, 1:-1] = sub
    d = np.diff(p, axis=1)
    sr, sx = np.nonzero(d == 1)                     # run starts (row, x)
    er, ex = np.nonzero(d == -1)
    if not np.array_equal(sr, er):
        raise RuntimeError("run-length encoding desynchronised")
    ex = ex - 1                                     # inclusive end

    nseg = np.bincount(sr, minlength=sub.shape[0])
    nstripes = len(ridx)
    buf = np.empty(nstripes * 3 + 2 * len(sx) + 1, dtype="<i4")
    buf[0] = nstripes

    off = 1 + np.concatenate(([0], np.cumsum(nseg * 2 + 3)[:-1]))
    buf[off] = ridx // ny                           # z
    buf[off + 1] = ridx % ny                        # y
    buf[off + 2] = nseg
    within = np.arange(len(sx)) - np.repeat(
        np.concatenate(([0], np.cumsum(nseg)[:-1])), nseg)
    segpos = off[sr] + 3 + 2 * within
    buf[segpos] = sx
    buf[segpos + 1] = ex

    buf.tofile(path)
    return nstripes


# --------------------------------------------------------------------------- #
# SWC
# --------------------------------------------------------------------------- #
def read_swc(path: str):
    """Parse an SWC file.

    Returns ``(zyx, radius, parent, node_id)``. **Note the axis flip**: SWC columns
    are ``x y z`` but everything in this package is ``zyx``, so the returned
    positions are already reordered to index the mask directly as
    ``mask[z, y, x]``.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            rows.append((int(p[0]), float(p[2]), float(p[3]), float(p[4]),
                         float(p[5]), int(p[6])))
    if not rows:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros(0, int), np.zeros(0, int))
    node_id = np.array([r[0] for r in rows], dtype=int)
    zyx = np.array([[r[3], r[2], r[1]] for r in rows], dtype=float)   # x,y,z -> z,y,x
    radius = np.array([r[4] for r in rows], dtype=float)
    parent = np.array([r[5] for r in rows], dtype=int)
    return zyx, radius, parent, node_id


def swc_edges(parent: np.ndarray, node_id: np.ndarray) -> np.ndarray:
    """``(N,2)`` index pairs into the node arrays, dropping roots (parent ``-1``)."""
    idx = {v: k for k, v in enumerate(node_id)}
    e = [(k, idx[q]) for k, q in enumerate(parent) if q in idx]
    return np.array(e, dtype=int) if e else np.zeros((0, 2), dtype=int)


def cable_length(zyx: np.ndarray, edges: np.ndarray) -> float:
    if len(edges) == 0:
        return 0.0
    return float(np.linalg.norm(zyx[edges[:, 0]] - zyx[edges[:, 1]], axis=1).sum())


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
DEFAULT_SKELETONIZE = {
    "downsampleInterval": [0, 0, 0],
    "minimalLength": 10,          # in VOXELS at the scale of the mask you pass in
    "finalMinimalLength": 0,
    "maximalDistance": 50,
    "keepingSingleObject": True,
    "rebase": True,
    "fillingHole": False,
}


def run_neutu(mask_zyx: np.ndarray, out_swc: str, *, neutu: str = "neutu",
              params: dict | None = None, workdir: str | None = None,
              timeout: int = 3600) -> dict:
    """Skeletonize a mask with the NeuTu CLI. Returns the parsed SWC.

    ``neutu`` must be on PATH in an environment where NeuTu is built (it is *not*
    installable alongside this package — see ``docs/skeletonization-comparison.md``
    for why), so in practice this is invoked via a wrapper that activates that
    environment.

    Two limits worth knowing before calling this on a large body:

    - NeuTu aborts above ``ONEGIGA`` (1,073,741,824) voxels of **bounding box** —
      not voxel count. A sparse arbor with a wide bbox trips it.
    - Its distance map is uint16 *squared* distance, so radii saturate at ~256
      voxels. Irrelevant at 32 nm/vox for these data; check if you go coarser.
    """
    cfg = dict(DEFAULT_SKELETONIZE)
    cfg.update(params or {})
    tmp = workdir or tempfile.mkdtemp(prefix="neutu-")
    sobj = os.path.join(tmp, "body.sobj")
    cfgp = os.path.join(tmp, "cfg.json")

    write_sobj(mask_zyx, sobj)
    with open(cfgp, "w") as f:
        json.dump({"command": "skeletonize", "output": out_swc,
                   "skeletonize": cfg}, f)

    # Valued options BEFORE the positional input -- see module docstring.
    argv = [neutu, "--command", "--config", cfgp, sobj, "--skeletonize"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(out_swc):
        raise RuntimeError(
            f"neutu failed (rc={proc.returncode}). argv={argv}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}")

    zyx, radius, parent, node_id = read_swc(out_swc)
    return {"zyx": zyx, "radius": radius, "parent": parent, "node_id": node_id,
            "edges": swc_edges(parent, node_id)}
