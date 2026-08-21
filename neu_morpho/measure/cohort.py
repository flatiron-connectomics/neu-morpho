"""Which bodies to measure, and why each one qualified.

A cohort is a **table**, not just a list of ids: ``body_id`` plus whatever attributes the
selection actually read. That extra column or two is what lets results be grouped by cell
type later without re-fetching anything, and it is the record of *why* a body is in — which
a bare id list throws away.

**Selection does not happen here.** This module loads, validates and joins cohorts; it
does not decide membership. The two datasets in the driving comparison select by entirely
different means — Megaphragma by DVID ``instance`` string suffixes, male-CNS by FlyEM's own
annotations — and neither has anything to do with measuring a neuron. Keeping selection
upstream also keeps it *reproducible*: a cohort file is a fixed record, where a live query
would quietly measure a different set of bodies next month as proofreading moves on.

The Megaphragma cohorts are built from `neu-mark select-bodies` (synapse count) plus
`neu-mark bodies` (the ``instance`` annotations), then filtered:

- **noise** — ``irrelevant`` / ``block`` / ``chunk`` / ``unknown`` — always dropped.
- **``glia``** — dropped for a neuron comparison, kept in the data for anyone else.
- **``NCL`` / ``nucleus``** — dropped, because Megaphragma's denucleation control *is* the
  cohort: with no semantic labels to mask a nucleus out, the comparison is restricted to
  anucleate cells.
- **``fragment`` / ``truncated``** — dropped from the *complete-cell* cohort only. A
  fragment or a truncated cell has a perfectly measurable calibre but no meaningful total
  volume or cable length, so it belongs in a diameter comparison and not a volume one.

Which gives two cohorts, the second a subset of the first. **A body in both has identical
measurements**, so it is measured once and membership is carried separately — see
:func:`membership`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class Cohort:
    """A named set of bodies to measure, with the attributes that selected them."""

    name: str
    dataset: str
    ids: tuple[int, ...]
    attrs: Any = None            # a DataFrame keyed by body_id, or None

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, body_id: object) -> bool:
        return int(body_id) in set(self.ids)          # noqa: PLR6201 - tuple is the API

    def __repr__(self) -> str:
        cols = "" if self.attrs is None else f", {len(self.attrs.columns)} attrs"
        return f"Cohort({self.name!r}, {self.dataset!r}, {len(self.ids)} bodies{cols})"

    def to_frame(self):
        """The cohort as a DataFrame, synthesised from the ids when it has no attrs."""
        import pandas as pd

        if self.attrs is not None:
            out = self.attrs.copy()
        else:
            out = pd.DataFrame({"body_id": list(self.ids)})
        out.insert(0, "cohort", self.name)
        out.insert(0, "dataset", self.dataset)
        return out


def _check_ids(ids: Sequence[int], where: str) -> tuple[int, ...]:
    """Integer, non-negative and **unique**.

    Uniqueness is not pedantry: a duplicated ``body_id`` silently doubles that body's
    weight in every aggregate computed after a join, and nothing downstream can see it.
    """
    out = [int(i) for i in ids]
    if any(i < 0 for i in out):
        raise ValueError(f"{where}: body ids must be non-negative")
    seen = set()
    dupes = {i for i in out if i in seen or seen.add(i)}
    if dupes:
        sample = sorted(dupes)[:5]
        raise ValueError(
            f"{where}: {len(dupes)} duplicate body id(s), e.g. {sample}. A duplicate "
            f"doubles that body's weight in any aggregate taken after a join, invisibly.")
    return tuple(out)


def cohort_from_table(table: Any, *, name: str, dataset: str,
                      id_column: str = "body_id") -> Cohort:
    """A :class:`Cohort` from a DataFrame carrying ``body_id`` and any attributes."""
    if id_column not in table.columns:
        raise ValueError(
            f"cohort table has no {id_column!r} column; it has {list(table.columns)}")
    col = table[id_column]
    # A float column silently loses precision above 2**53, and body ids are uint64 in
    # precomputed. Refuse rather than truncate — the same hazard that makes parquet the
    # default over csv for nullable integer columns elsewhere in the suite.
    if str(col.dtype).startswith("float"):
        raise ValueError(
            f"{id_column!r} is {col.dtype}; body ids must be an integer dtype, since a "
            f"float64 id above 2**53 is silently rounded")
    ids = _check_ids(col.tolist(), where="cohort table")
    return Cohort(name=name, dataset=dataset, ids=ids, attrs=table.reset_index(drop=True))


def load_cohort(source: Any, *, name: Optional[str] = None,
                dataset: Optional[str] = None, id_column: str = "body_id") -> Cohort:
    """Load a cohort from a parquet/csv **table**, a plain id file, or an iterable.

    A table keeps its attribute columns; an id file or iterable produces a cohort with
    ``attrs=None``, which measures identically and simply cannot be grouped by anything.
    ``name`` defaults to the file stem so a cohort is labelled without being told twice.
    """
    import os

    if isinstance(source, str):
        stem = os.path.splitext(os.path.basename(source))[0]
        name = name or stem
        if source.endswith((".parquet", ".pq")):
            import pandas as pd

            return cohort_from_table(pd.read_parquet(source), name=name,
                                     dataset=dataset or "", id_column=id_column)
        if source.endswith(".csv"):
            import pandas as pd

            table = pd.read_csv(source)
            if id_column in table.columns and len(table.columns) > 1:
                return cohort_from_table(table, name=name, dataset=dataset or "",
                                         id_column=id_column)
        # a bare list of ids, one per line — the existing allowlist format
        from ..allowlist import load_allowlist

        ids = load_allowlist(source)
        return Cohort(name=name, dataset=dataset or "",
                      ids=_check_ids(sorted(ids or ()), where=source))

    if hasattr(source, "columns"):                    # already a DataFrame
        return cohort_from_table(source, name=name or "cohort",
                                 dataset=dataset or "", id_column=id_column)

    return Cohort(name=name or "cohort", dataset=dataset or "",
                  ids=_check_ids(list(source), where="cohort"))


def save_cohort(cohort: Cohort, path: str) -> str:
    """Write a cohort as parquet, so it is its own provenance record."""
    frame = cohort.to_frame()
    if path.endswith(".csv"):
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def membership(cohorts: Iterable[Cohort]) -> Any:
    """Long-form ``(dataset, body_id, cohort)`` for several cohorts.

    One row per body per cohort it belongs to. This is how nested cohorts stay cheap: the
    complete-cell cohort is a **subset** of the non-noise one, and a body in both has
    identical measurements, so it is measured once and joined twice rather than measured
    twice. Filtering a comparison to complete cells is then a join, not a rerun.
    """
    import pandas as pd

    frames = []
    for cohort in cohorts:
        frames.append(pd.DataFrame({
            "dataset": cohort.dataset,
            "body_id": list(cohort.ids),
            "cohort": cohort.name,
        }))
    if not frames:
        return pd.DataFrame({"dataset": [], "body_id": [], "cohort": []})
    return pd.concat(frames, ignore_index=True)
