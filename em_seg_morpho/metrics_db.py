"""Per-body metrics database (SQLite).

One row per body, progressively enriched: the index scan fills bbox / voxel_count
/ volume; meshing and skeletonization update their columns. Bbox is stored in
**full-resolution voxel coords** (canonical, matching the mesh space); consumers
convert to whatever scale they read at.

Correctness on resume: the index applies each block's contribution AND marks that
block done in a **single SQLite transaction**, so a crash can't double-count a
summed metric (voxel_count) — either both land or neither. The driver is the sole
writer (block-map workers only compute per-block partials).
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Mapping

_BBOX = ["z0", "y0", "x0", "z1", "y1", "x1"]
# enrichment columns other stages may set
_EXTRA = ["cable_length_nm", "n_branches", "n_tips", "max_radius_nm",
          "mesh_area_nm2", "mesh_verts", "n_mesh_components"]


class MetricsDB:
    def __init__(self, path: str):
        self.con = sqlite3.connect(path)
        self.con.execute("PRAGMA journal_mode=WAL")
        cols = ", ".join(f"{c} INTEGER" for c in _BBOX)
        extra = ", ".join(f"{c} REAL" for c in _EXTRA)
        self.con.execute(
            f"CREATE TABLE IF NOT EXISTS bodies (body_id INTEGER PRIMARY KEY, {cols}, "
            f"voxel_count INTEGER DEFAULT 0, volume_nm3 REAL DEFAULT 0, {extra})")
        self.con.execute("CREATE TABLE IF NOT EXISTS index_progress (block TEXT PRIMARY KEY)")
        self.con.commit()

    # -- index scan (reduction) -------------------------------------------
    def done_blocks(self) -> set[str]:
        return {r[0] for r in self.con.execute("SELECT block FROM index_progress")}

    def reset_index(self) -> None:
        self.con.execute("DELETE FROM bodies")
        self.con.execute("DELETE FROM index_progress")
        self.con.commit()

    def apply_index_block(self, block_key: str,
                          partials: Mapping[int, tuple], voxel_volume_nm3: float) -> None:
        """Merge one block's {body_id: (z0,y0,x0,z1,y1,x1,count)} and mark it done, atomically.

        bbox merges by min/max; voxel_count and volume accumulate. No-op if the
        block was already applied (idempotent).
        """
        cur = self.con.cursor()
        cur.execute("BEGIN")
        try:
            if cur.execute("SELECT 1 FROM index_progress WHERE block=?", (block_key,)).fetchone():
                cur.execute("COMMIT")
                return
            for body_id, (z0, y0, x0, z1, y1, x1, count) in partials.items():
                cur.execute(
                    "INSERT INTO bodies (body_id, z0,y0,x0,z1,y1,x1, voxel_count, volume_nm3) "
                    "VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(body_id) DO UPDATE SET "
                    "z0=min(z0,excluded.z0), y0=min(y0,excluded.y0), x0=min(x0,excluded.x0), "
                    "z1=max(z1,excluded.z1), y1=max(y1,excluded.y1), x1=max(x1,excluded.x1), "
                    "voxel_count=voxel_count+excluded.voxel_count, "
                    "volume_nm3=volume_nm3+excluded.volume_nm3",
                    (int(body_id), z0, y0, x0, z1, y1, x1, int(count), count * voxel_volume_nm3))
            cur.execute("INSERT INTO index_progress (block) VALUES (?)", (block_key,))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    # -- queries -----------------------------------------------------------
    def get_bbox(self, body_id: int) -> tuple | None:
        r = self.con.execute(
            f"SELECT {','.join(_BBOX)} FROM bodies WHERE body_id=?", (int(body_id),)).fetchone()
        return tuple(r) if r else None

    def crop_at_scale(self, body_id: int, factor_zyx: Sequence[int], *,
                      margin_vox: int = 0, clip_shape: Sequence[int] | None = None) -> tuple | None:
        """A body's bbox at a read scale, as (z0,y0,x0,z1,y1,x1) voxels (+ margin).

        ``factor_zyx`` = full-res voxels per read-scale voxel (e.g. 2**read_scale).
        This is the crop skeletonization/meshing reads for one body — from the DB,
        so no pre-known bbox is needed.
        """
        bb = self.get_bbox(body_id)
        if bb is None:
            return None
        fz, fy, fx = factor_zyx
        lo = [bb[0] // fz, bb[1] // fy, bb[2] // fx]
        hi = [-(-bb[3] // fz), -(-bb[4] // fy), -(-bb[5] // fx)]   # ceil
        lo = [max(0, v - margin_vox) for v in lo]
        hi = [v + margin_vox for v in hi]
        if clip_shape is not None:
            hi = [min(h, int(s)) for h, s in zip(hi, clip_shape)]
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

    def bodies_by_size(self, min_voxels: int = 0, limit: int | None = None) -> list[int]:
        q = "SELECT body_id FROM bodies WHERE voxel_count>=? ORDER BY voxel_count DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [r[0] for r in self.con.execute(q, (int(min_voxels),))]

    def write_allowlist(self, path: str, min_voxels: int = 0, limit: int | None = None) -> int:
        ids = self.bodies_by_size(min_voxels, limit)
        with open(path, "w") as f:
            f.write("body_id\n")
            f.writelines(f"{i}\n" for i in ids)
        return len(ids)

    # -- enrichment (mesh / skeleton stages, via the single-writer driver) --
    def update_body(self, body_id: int, **cols) -> None:
        if not cols:
            return
        assignments = ", ".join(f"{k}=?" for k in cols)
        self.con.execute(
            f"INSERT INTO bodies (body_id) VALUES (?) ON CONFLICT(body_id) DO UPDATE SET {assignments}",
            (int(body_id), *cols.values()))
        self.con.commit()

    def to_csv(self, path: str) -> None:
        import csv
        cur = self.con.execute("SELECT * FROM bodies ORDER BY body_id")
        names = [d[0] for d in cur.description]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(names)
            w.writerows(cur)

    def commit(self) -> None:
        self.con.commit()

    def close(self) -> None:
        self.con.commit()
        self.con.close()
