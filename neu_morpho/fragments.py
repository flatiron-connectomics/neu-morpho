"""Intermediate per-(body, block) fragments (stage 1 -> stage 2 handoff).

Stage 1 writes one fragment per body per block under
``<chunked_dir>/<body_id>/<iz>_<iy>_<ix>.<fmt>``. Stage 2 discovers bodies by
listing the directory (so no separate bounding-box index is needed) and gathers
each body's fragments to assemble. Filesystem-backed for now (ceph); an
object-store layout can come later.

Meshes (``.drc``) and skeletons (``.skel``) share the layout, each under its own
chunked dir. Skeleton fragments are stored in the precomputed encoding purely
because it is a compact binary that osteoid can already read and write — but
unlike the *output* blobs they hold **zyx** vertices in global nm, since that is
the model space stage 2 fuses in. The zyx->xyz flip happens once, at output
(precomputed.encode_skeleton).
"""

from __future__ import annotations

import glob
import os
from typing import Sequence


def body_dir(chunked_dir: str, body_id: int) -> str:
    return os.path.join(chunked_dir, str(int(body_id)))


def fragment_path(chunked_dir: str, body_id: int, block_index: Sequence[int], fmt: str = "drc") -> str:
    iz, iy, ix = block_index
    return os.path.join(body_dir(chunked_dir, body_id), f"{iz}_{iy}_{ix}.{fmt}")


def write_fragment(chunked_dir: str, body_id: int, block_index: Sequence[int], mesh, fmt: str = "drc") -> None:
    path = fragment_path(chunked_dir, body_id, block_index, fmt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.serialize(path)                                  # fmt inferred from extension


def list_bodies(chunked_dir: str) -> list[int]:
    if not os.path.isdir(chunked_dir):
        return []
    return sorted(int(d) for d in os.listdir(chunked_dir) if d.isdigit())


def read_body_fragments(chunked_dir: str, body_id: int, fmt: str = "drc") -> list:
    """Load all of a body's fragment meshes (``vol2mesh.Mesh``)."""
    from vol2mesh import Mesh

    paths = sorted(glob.glob(os.path.join(body_dir(chunked_dir, body_id), f"*.{fmt}")))
    return [Mesh.from_file(p) for p in paths]


# --------------------------------------------------------------------------- #
# Skeleton fragments (zyx vertices, global nm — see the module docstring)
# --------------------------------------------------------------------------- #
def write_skel_fragment(chunked_dir: str, body_id: int, block_index: Sequence[int],
                        skeleton, fmt: str = "skel") -> None:
    path = fragment_path(chunked_dir, body_id, block_index, fmt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(skeleton.to_precomputed())
    os.replace(tmp, path)          # atomic: a killed worker leaves no half fragment


def read_body_skel_fragments(chunked_dir: str, body_id: int, fmt: str = "skel") -> list:
    """Load all of a body's skeleton fragments (``osteoid.Skeleton``, zyx nm)."""
    from osteoid import Skeleton

    paths = sorted(glob.glob(os.path.join(body_dir(chunked_dir, body_id), f"*.{fmt}")))
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(Skeleton.from_precomputed(f.read(), segid=int(body_id)))
    return out
