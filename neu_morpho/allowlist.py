"""Body allowlist handling.

Mesh only the given body IDs (mesh-n-bone used e.g. ``largest_20k_bodies.csv``),
falling back to **all labels** when no allowlist is provided.
"""

from __future__ import annotations

import os
from typing import Iterable


def load_allowlist(source: str | Iterable[int] | None) -> set[int] | None:
    """Return a set of allowed body IDs, or ``None`` meaning "all labels".

    ``source`` may be a path to a text/CSV file (one integer id per line, or a
    single ``id`` column), an iterable of ids, or ``None``.
    """
    if source is None:
        return None
    if not isinstance(source, str):
        return {int(x) for x in source}
    ids: set[int] = set()
    with open(os.path.expanduser(source)) as f:
        for line in f:
            tok = line.strip().split(",")[0].strip()
            if not tok or not tok.lstrip("-").isdigit():
                continue                     # skip header / blank lines
            ids.add(int(tok))
    if not ids:
        raise ValueError(f"no body IDs parsed from allowlist {source!r}")
    return ids
