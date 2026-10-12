"""Intermediate per-(body, block) fragments (stage 1 -> stage 2 handoff).

Stage 1 writes one fragment per body per block under
``<chunked_dir>/h<bucket>/<body_id>/<iz>_<iy>_<ix>.<fmt>``, where ``bucket`` is
``body_id % FANOUT`` in three hex digits. Stage 2 discovers bodies by listing the
buckets (so no separate bounding-box index is needed) and gathers each body's
fragments to assemble. Filesystem-backed for now (ceph); an object-store layout can
come later.

**The buckets exist because ceph handles wide directories badly.** One directory per
body directly under ``chunked_dir`` put 568,165 entries in one directory on a run that
meshed every segment of an agglomeration (2026-10-12): listing it hung for minutes and
the run left 658k files of ~800 bytes. Bucketing caps a directory at ``bodies / 4096``
entries; it does NOT reduce the file count, which only packing fragments per block
would. The ``h`` prefix keeps a bucket name from ever parsing as a body id, which is
how :func:`list_bodies` tells the old flat layout from this one.

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


#: Buckets per fragment store. 4096 keeps a million-body run to ~250 entries per directory
#: and costs a small run nothing, since a bucket is only created when written to.
FANOUT = 4096


class OldFragmentLayout(RuntimeError):
    """The store holds body directories at its top level, from before bucketing."""


def bucket_name(body_id: int) -> str:
    return f"h{int(body_id) % FANOUT:03x}"


def body_dir(chunked_dir: str, body_id: int) -> str:
    return os.path.join(chunked_dir, bucket_name(body_id), str(int(body_id)))


def _buckets(chunked_dir: str) -> list[str]:
    """The bucket directories, refusing a store written in the old flat layout.

    Refusing rather than reading both: an old store's bodies would otherwise be
    invisible to stage 2, so a resumed run would assemble nothing and report success.
    """
    names = os.listdir(chunked_dir)
    old = next((n for n in names if n.isdigit()), None)
    if old is not None:
        raise OldFragmentLayout(
            f"{chunked_dir} holds body directories at its top level (e.g. {old}/), the "
            f"layout before fragment stores were bucketed. Delete the store and rerun "
            f"stage 1 (it is scratch), or finish the run with the neu-morpho that wrote it.")
    return [os.path.join(chunked_dir, n) for n in names
            if n.startswith("h") and os.path.isdir(os.path.join(chunked_dir, n))]


def count_bodies(chunked_dir: str) -> int | None:
    """How many bodies have fragments, from one listing per bucket -- never by walking
    into the body directories, which would be a millions-of-inodes traversal. ``None``
    if the store does not exist."""
    if not os.path.isdir(chunked_dir):
        return None
    return sum(sum(1 for n in os.listdir(b) if n.isdigit()) for b in _buckets(chunked_dir))


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
    return sorted(int(d) for b in _buckets(chunked_dir) for d in os.listdir(b) if d.isdigit())


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
