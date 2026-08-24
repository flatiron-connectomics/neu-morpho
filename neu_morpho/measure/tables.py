"""Turn a measure DB into tidy tables, for cohort selection and comparison.

The notebook-facing half of `measure`: DataFrame in, DataFrame out, no printing and no
plotting, so the interesting logic is testable without a kernel. A notebook should be a
thin driver over this, not the place the derivations live.

pandas and pyarrow arrive with ``neu-morpho[measure]`` and are imported INSIDE functions,
so ``import neu_morpho.measure`` still costs nothing — the same rule the CLI follows.

The variant vocabulary is the load-bearing part. ``all`` is the whole body;
``minus_nucleus`` removes the nucleus only; ``minus_soma_nucleus`` removes the whole cell
body. **Which one matches depends on the other cohort**, and getting it wrong is a silent
bias rather than an error: comparing against a denucleated-but-not-asomatic cohort calls
for ``minus_nucleus``, and using ``minus_soma_nucleus`` there strips a compartment from one
side that is still present on the other.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

#: Compartment names as published by the semantic source this was built against. Passed
#: in rather than assumed — a different dataset numbers its labels differently, and
#: `compartments.semantic_label_names` reads the real mapping from the source.
DEFAULT_COMPARTMENTS = {0: "unlabelled", 1: "neuropil", 2: "fiber_bundle", 3: "nucleus",
                        4: "glia", 5: "soma", 6: "trachea", 7: "muscle",
                        8: "do_not_merge"}

#: variant -> the compartments subtracted from the total.
VARIANTS = {"all": (),
            "minus_nucleus": ("nucleus",),
            "minus_soma_nucleus": ("nucleus", "soma")}


def load_bodies(db_path: str, *, voxel_nm: float = 32.0,
                dataset: Optional[str] = None,
                compartments: Optional[Mapping[int, str]] = None):
    """One row per body, with volume, cable, per-compartment volume and the variants.

    Reads only — opens the DB read-only, so this is safe against a run in flight.
    Bodies with no voxel count are dropped: they were never measured, which the
    ``voxel_count DEFAULT 0`` column cannot distinguish from "measured as empty".
    """
    import pandas as pd

    from ..metrics_db import MetricsDB

    names = dict(DEFAULT_COMPARTMENTS if compartments is None else compartments)
    db = MetricsDB(db_path, read_only=True)
    try:
        bodies = pd.read_sql_query(
            "SELECT body_id, voxel_count, volume_nm3, cable_length_nm, n_tips, "
            "n_branches, max_radius_nm FROM bodies WHERE voxel_count > 0", db.con)
        has_comp = db._has_table("body_compartments")
        comp = (pd.read_sql_query(
            "SELECT body_id, label, voxel_count FROM body_compartments", db.con)
            if has_comp else None)
    finally:
        db.close()

    vox = float(voxel_nm) ** 3
    out = bodies.copy()
    out["volume_um3"] = out["volume_nm3"] / 1e9
    out["cable_um"] = out["cable_length_nm"] / 1e3

    if comp is not None and len(comp):
        wide = (comp.pivot_table(index="body_id", columns="label",
                                 values="voxel_count", aggfunc="sum", fill_value=0)
                .rename(columns=lambda c: f"{names.get(int(c), f'label_{int(c)}')}_um3"))
        wide = wide * vox / 1e9
        out = out.merge(wide, on="body_id", how="left")
        for col in wide.columns:
            out[col] = out[col].fillna(0.0)

    for variant, drop in VARIANTS.items():
        cols = [f"{d}_um3" for d in drop if f"{d}_um3" in out.columns]
        if len(cols) != len(drop):
            continue                        # that compartment was not measured
        vol = out["volume_um3"] - (out[cols].sum(axis=1) if cols else 0.0)
        out[f"volume_um3_{variant}"] = vol
        out[f"diameter_nm_{variant}"] = _diameter_nm(vol, out["cable_um"])

    if dataset is not None:
        out.insert(0, "dataset", str(dataset))
    return out


def _diameter_nm(volume_um3, cable_um):
    """Area-equivalent diameter from V/L. NaN where there is no cable, never inf.

    NOTE the minus-variants divide a REDUCED volume by the FULL cable length, including
    the cable running through the soma, so they under-read neurite calibre. Splitting
    cable by compartment is the fix and is not implemented.
    """
    import numpy as np
    import pandas as pd

    vol_nm3 = pd.Series(volume_um3).astype(float) * 1e9
    cable_nm = pd.Series(cable_um).astype(float) * 1e3
    area = vol_nm3.where(cable_nm > 0) / cable_nm.where(cable_nm > 0)
    return 2.0 * np.sqrt(area / np.pi)


def _slug(tag: str) -> str:
    """A tag turned into a column-name-safe suffix."""
    import re

    return re.sub(r"[^0-9a-zA-Z]+", "_", tag).strip("_").lower() or "tag"


def load_segment_properties(base: str, *sources: str):
    """Wide table of every inline property across one or more properties sources.

    ``tags`` become two kinds of column: a boolean per bare tag, and, for the
    ``namespace:value`` convention that published sources use, one categorical column
    per namespace. A namespace carrying several values for one body keeps the first and
    the whole list lands in ``<namespace>_all``, because a multi-valued facet silently
    reduced to one value is how a column starts lying.
    """
    import pandas as pd

    from neu_vol import location
    from neu_vol.logs import quiet_reads

    frames = []
    for sub in sources:
        # Reading a properties source opens a store, and on a public bucket the
        # credential chain logs at ERROR severity before succeeding anonymously. From a
        # notebook's point of view THIS is the entry point, so it filters itself rather
        # than expecting the caller to remember; `neu_vol.logs.reads_quiet = False`
        # opts out.
        with quiet_reads():
            inline = location.read_json(base, sub, "info")["inline"]
        ids = [int(i) for i in inline["ids"]]
        # Columns are accumulated and the frame built ONCE. Assigning them one at a time
        # fragments the block manager and pandas warns per assignment — which at 482 tags
        # is hundreds of warnings and a genuinely slow frame, not just noise.
        cols: dict[str, Any] = {"body_id": ids}
        for prop in inline["properties"]:
            kind, pid = prop.get("type"), prop.get("id", "value")
            if kind == "tags":
                vocab = prop["tags"]
                rows = [[vocab[j] for j in row] for row in prop["values"]]
                # TWO passes. Collecting namespaces while building the columns leaves a
                # namespace first seen on a later row short by however many rows preceded
                # it, and pandas then rejects the frame — which is the good outcome; a
                # silently misaligned column would attribute tags to the wrong bodies.
                parsed, bare = [], []
                for row in rows:
                    seen: dict[str, list[str]] = {}
                    plain: list[str] = []
                    for tag in row:
                        ns, sep, val = tag.partition(":")
                        if sep:
                            seen.setdefault(ns, []).append(val)
                        else:
                            plain.append(tag)
                    parsed.append(seen)
                    bare.append(plain)
                for ns in sorted({k for s in parsed for k in s}):
                    col = [tuple(s.get(ns, ())) for s in parsed]
                    cols[ns] = [v[0] if v else None for v in col]
                    if any(len(v) > 1 for v in col):
                        # TUPLES, not lists. A list-valued column is unhashable, so
                        # `nunique`, `groupby` and `value_counts` all raise on it — which
                        # makes the column unusable for exactly the grouping this table
                        # exists to support.
                        cols[f"{ns}_all"] = col
                if any(bare):
                    # A vocabulary with no namespace convention (`side-r`, `fragment`)
                    # gets one BOOLEAN column per tag. Keeping only the first, as a
                    # namespace column does, silently discards the rest — and the tags
                    # are not mutually exclusive, so there is no first to prefer.
                    cols[f"{pid}_all"] = [tuple(b) for b in bare]
                    for tag in sorted({t for b in bare for t in b}):
                        cols[f"{pid}_{_slug(tag)}"] = [tag in b for b in bare]
            else:
                cols[pid] = prop["values"]
        frames.append(pd.DataFrame(cols))

    if not frames:
        raise ValueError("no properties sources given")
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="body_id", how="outer")
    return out


def merge_properties(bodies, props, *, how: str = "left"):
    """Join properties onto a bodies table, keeping boolean columns BOOLEAN.

    Use this rather than ``bodies.merge(props, on='body_id')``. A plain merge widens any
    ``bool`` column to ``object`` wherever a body has no properties row, and then ``~`` on
    that column is **arithmetic, not logical**: ``~True`` is ``-2``, ``~False`` is ``-1``.
    A bare ``~`` raises on the NaN, which is survivable — but the *defensive* idiom,
    ``~col.fillna(False)``, does not raise and yields integers, so the resulting mask does
    not mean what it reads as. Measured on real data: three stacked exclusions left the
    row count unchanged at 16,858 where the correct answer was 6,570.

    A body with no properties row gets ``False`` for every flag, which is the right
    reading — an absent record asserts no tags.
    """
    import pandas as pd

    flags = [c for c in props.columns
             if c != "body_id" and pd.api.types.is_bool_dtype(props[c])]
    out = bodies.merge(props, on="body_id", how=how)
    for col in flags:
        out[col] = out[col].fillna(False).astype(bool)
    return out


def summarize(df, columns: Sequence[str], *, by: Optional[str] = None,
              quantiles: Sequence[float] = (0.1, 0.5, 0.9)):
    """Percentile table for ``columns``, optionally grouped by a column.

    Percentiles rather than mean and sd, because every distribution here is heavy-tailed
    by orders of magnitude and a mean says more about the largest body than about a
    typical one.
    """
    import pandas as pd

    cols = [c for c in columns if c in df.columns]
    if by is None:
        rows = {"n": len(df)}
        for c in cols:
            for q in quantiles:
                rows[f"{c}_p{int(q * 100)}"] = df[c].quantile(q)
        return pd.DataFrame([rows])
    out = []
    for key, part in df.groupby(by, dropna=False):
        row = {by: key, "n": len(part)}
        for c in cols:
            for q in quantiles:
                row[f"{c}_p{int(q * 100)}"] = part[c].quantile(q)
        out.append(row)
    return pd.DataFrame(out).sort_values("n", ascending=False).reset_index(drop=True)


def compare(left, right, columns: Sequence[str], *,
            labels: Sequence[str] = ("left", "right"),
            statistic: str = "median"):
    """One row per column: each cohort's statistic and the ratio between them.

    The ratio is of the statistics, not the statistic of the ratios — the two bodies of
    data are unpaired, so a per-body ratio does not exist.
    """
    import pandas as pd

    def stat(frame, col):
        series = frame[col].dropna()
        if not len(series):
            return float("nan")
        return series.median() if statistic == "median" else getattr(series, statistic)()

    rows = []
    for c in columns:
        if c not in left.columns or c not in right.columns:
            continue
        a, b = stat(left, c), stat(right, c)
        rows.append({"column": c, labels[0]: a, labels[1]: b,
                     "ratio": (a / b) if b else float("nan")})
    return pd.DataFrame(rows)


def log_histogram(df, column: str, *, lo: float, hi: float, nbins: int = 48,
                  weight: Optional[str] = None):
    """Log-spaced histogram as a DataFrame, with under/overflow rows.

    The catch-alls make the weight total exact, so the table is a decomposition rather
    than a lossy view — and a fat overflow row is the signal that the range is wrong.
    """
    import numpy as np
    import pandas as pd

    from .sweep import log_bin_edges, weighted_histogram

    edges = log_bin_edges(lo, hi, nbins)
    values = df[column].to_numpy(dtype=float)
    weights = (np.ones(len(values)) if weight is None
               else df[weight].to_numpy(dtype=float))
    good = np.isfinite(values) & np.isfinite(weights)
    counts = weighted_histogram(values[good], weights[good], edges)
    lows = np.concatenate([[-np.inf], edges])
    highs = np.concatenate([edges, [np.inf]])
    return pd.DataFrame({"lo": lows, "hi": highs, "weight": counts})


def write_table(df, path: str) -> str:
    """Write parquet, which unlike csv round-trips a nullable integer column."""
    df.to_parquet(path, index=False)
    return path
