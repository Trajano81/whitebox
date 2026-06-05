"""Editable cleaning / banding proposals.

`propose_cleaning` describes a variable and proposes a mapping (categorical) or a
banding (numeric) that a reviewer can edit or accept directly. `apply_cleaning`
materializes the (possibly edited) proposal into a cleaned column.
"""
import numpy as np
import pandas as pd

from .profiler import normalize_missing, profile_variable

MISSING_LABEL = "Missing"
OTHER_LABEL = "Other"

DEFAULT_N_BINS = 10


def propose_cleaning(
    series,
    name=None,
    method="by_bins",
    n_bins=DEFAULT_N_BINS,
    missing_tokens=None,
    rare_threshold=0.01,
):
    """Propose an editable cleaning for `series`.

    Categorical -> a `level_map` (original label -> cleaned/grouped label) with
    missing tokens folded to `Missing` and rare levels to `Other`.
    Numeric -> a banding (`method` in {"by_bins", "by_magnitude"}, default 10 bins)
    with editable `edges` + `labels`.
    """
    prof = profile_variable(series, name=name, missing_tokens=missing_tokens)
    if prof["dtype"] == "categorical":
        return _propose_categorical(series, prof, missing_tokens, rare_threshold)
    return _propose_numeric(series, prof, method, n_bins, missing_tokens)


def _propose_categorical(series, prof, missing_tokens, rare_threshold):
    miss = normalize_missing(series, missing_tokens)
    n = len(series)
    level_map = {}
    counts = series[~miss].astype(str).value_counts()
    for label, cnt in counts.items():
        if n and (cnt / n) < rare_threshold:
            level_map[label] = OTHER_LABEL
        else:
            level_map[label] = label.strip()  # trim whitespace oddities
    for label in series[miss].astype(str).unique():
        level_map[label] = MISSING_LABEL
    return {
        "kind": "categorical_cleaning",
        "missing_label": MISSING_LABEL,
        "level_map": level_map,
        "profile": prof,
    }


def _band_edges_by_bins(values, n_bins):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(values, qs))
    return edges


def _band_edges_by_magnitude(values, n_bins):
    v = values[~np.isnan(values)]
    v = v[v > 0]
    if v.size == 0:
        return _band_edges_by_bins(values, n_bins)
    lo = int(np.floor(np.log10(v.min())))
    hi = int(np.ceil(np.log10(v.max())))
    edges = np.power(10.0, np.arange(lo, hi + 1))
    # Bookend with the actual min/max so all points fall inside a band.
    edges = np.unique(np.concatenate(([v.min()], edges, [v.max()])))
    return edges


def _propose_numeric(series, prof, method, n_bins, missing_tokens):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    if method == "by_magnitude":
        edges = _band_edges_by_magnitude(values, n_bins)
    else:
        method = "by_bins"
        edges = _band_edges_by_bins(values, n_bins)
    edges = [float(e) for e in edges]
    labels = [f"[{edges[i]:.4g}, {edges[i + 1]:.4g})" for i in range(len(edges) - 1)]
    return {
        "kind": "numeric_banding",
        "method": method,
        "n_bins": n_bins,
        "edges": edges,
        "labels": labels,
        "profile": prof,
    }


def apply_cleaning(data, name, proposal):
    """Materialize a (possibly edited) proposal into a cleaned Series of labels.

    Returns the cleaned Series (the caller writes it back and re-encodes). For
    categorical cleaning, maps original labels via `level_map`. For numeric
    banding, cuts on `edges` into the band `labels`.
    """
    series = data[name]
    kind = proposal.get("kind")
    if kind == "categorical_cleaning":
        level_map = proposal["level_map"]
        missing_label = proposal.get("missing_label", MISSING_LABEL)
        return series.astype(str).map(lambda v: level_map.get(v, missing_label if v == "nan" else v))
    if kind == "numeric_banding":
        edges = proposal["edges"]
        labels = proposal.get("labels")
        cut = pd.cut(
            pd.to_numeric(series, errors="coerce"),
            bins=edges,
            labels=labels,
            include_lowest=True,
            duplicates="drop",
        )
        return cut.astype("object").where(cut.notna(), MISSING_LABEL)
    raise ValueError(f"unknown proposal kind {kind!r}")
