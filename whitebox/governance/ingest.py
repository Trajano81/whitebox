"""Schema snapshot + revision diff for new-data ingestion.

When new data arrives we compare its schema to the known variables and report
which variables are new (need review), missing (dropped), or changed
(dtype/cardinality drift, needing re-review).
"""
import pandas as pd


def schema_snapshot(data, columns=None):
    """Capture {col: {dtype, n_unique}} for the given (or all) columns."""
    cols = list(columns) if columns is not None else list(data.columns)
    snap = {}
    for c in cols:
        if c not in data.columns:
            continue
        s = data[c]
        snap[c] = {
            "dtype": "numeric" if pd.api.types.is_numeric_dtype(s.dtype) else "categorical",
            "n_unique": int(s.nunique(dropna=True)),
        }
    return snap


def diff_schema(old_snap, new_snap):
    """Diff two schema snapshots into new / missing / changed variables."""
    old = set(old_snap)
    new = set(new_snap)
    added = sorted(new - old)
    missing = sorted(old - new)
    changed = []
    for c in sorted(old & new):
        o, n = old_snap[c], new_snap[c]
        if o["dtype"] != n["dtype"]:
            changed.append({"name": c, "reason": "dtype", "old": o, "new": n})
        elif o["n_unique"] != n["n_unique"]:
            changed.append({"name": c, "reason": "cardinality", "old": o, "new": n})
    return {"new": added, "missing": missing, "changed": changed}
