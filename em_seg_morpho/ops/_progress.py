"""Per-group manifest tallies.

``Manifest.counts()`` aggregates every group in the file, so a two-stage op that
reports it conflates "blocks chunked" with "bodies assembled" — the totals then
exceed the block count and mean nothing. Each stage reports its own group.
"""

from __future__ import annotations

from typing import Any


def group_counts(manifest: Any, group: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in manifest.done_keys(group):
        status = manifest.status(group, key)
        out[status] = out.get(status, 0) + 1
    return out
