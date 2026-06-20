"""Profile a variable's levels and detect data-quality issues.

Used at first ingest (raw variables are messy: blanks, NA, ".", odd characters)
and during the review of derived variables.
"""
import re

import numpy as np
import pandas as pd

# Tokens treated as "missing" when found in a categorical column.
DEFAULT_MISSING_TOKENS = [
    "", "NA", "N/A", "na", "n/a", ".", "null", "NULL", "NaN", "nan", "None", "none",
]

# Characters beyond letters, digits, space, underscore and hyphen are "odd".
_ODD_CHAR_RE = re.compile(r"[^0-9A-Za-z _\-]")

# Default thresholds for raising review flags.
HIGH_MISSING_PCT = 20.0
HIGH_CARDINALITY = 50


def normalize_missing(series, tokens=None):
    """Boolean mask: True where the value is missing (NaN/None or a missing token).

    Comparison is case-insensitive and ignores surrounding whitespace.
    """
    token_set = {str(t).strip().lower() for t in (tokens if tokens is not None else DEFAULT_MISSING_TOKENS)}

    def _is_missing(v):
        if v is None:
            return True
        try:
            if isinstance(v, float) and np.isnan(v):
                return True
        except TypeError:
            pass
        return str(v).strip().lower() in token_set

    return series.map(_is_missing)


def profile_variable(series, name=None, missing_tokens=None):
    """Return a profile dict describing levels and data-quality flags."""
    n = len(series)
    miss_mask = normalize_missing(series, missing_tokens)
    n_missing = int(miss_mask.sum())
    missing_pct = round(100.0 * n_missing / n, 2) if n else 0.0
    is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
    nonmiss = series[~miss_mask]
    n_unique = int(nonmiss.nunique())

    odd_tokens = []
    top_levels = []
    if not is_numeric:
        for val in nonmiss.astype(str).unique():
            if _ODD_CHAR_RE.search(val) or val != val.strip():
                odd_tokens.append(val)
        counts = nonmiss.astype(str).value_counts().head(20)
        top_levels = [
            {"label": label, "count": int(cnt), "pct": round(100.0 * cnt / n, 2) if n else 0.0}
            for label, cnt in counts.items()
        ]

    flags = []
    if missing_pct >= HIGH_MISSING_PCT:
        flags.append("high_missingness")
    if odd_tokens:
        flags.append("odd_tokens")
    if not is_numeric and n_unique > HIGH_CARDINALITY:
        flags.append("high_cardinality")

    return {
        "name": name,
        "dtype": "numeric" if is_numeric else "categorical",
        "n": n,
        "n_missing": n_missing,
        "missing_pct": missing_pct,
        "n_unique": n_unique,
        "top_levels": top_levels,
        "odd_tokens": odd_tokens[:50],
        "flags": flags,
    }
