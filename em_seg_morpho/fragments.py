"""Intermediate per-(body, block) mesh fragments (stage 1 -> stage 2 handoff).

Stage 1 writes one fragment per body per block under
``<chunked_dir>/<body_id>/<iz>_<iy>_<ix>.<fmt>``. Stage 2 discovers bodies by
listing the directory (so no separate bounding-box index is needed) and gathers
each body's fragments to assemble. Filesystem-backed for now (ceph); an
object-store layout can come later.
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
