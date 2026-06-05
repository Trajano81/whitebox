"""Whitebox MCP tool logic (pure functions over a Session; no MCP dependency).

Each function validates inputs and returns JSON-serializable structured data, or
raises ToolError with an actionable, next-step message. server.py wraps these for
FastMCP. Keeping the logic here makes every tool unit-testable without MCP.
"""
import os

# Default cap for list pagination.
DEFAULT_LIMIT = 50


class ToolError(Exception):
    """Raised for tool-level errors; the message should suggest a next step."""


class Session:
    """Holds the current Whitebox instance for a server session."""

    def __init__(self):
        self.wb = None

    def require(self):
        if self.wb is None:
            raise ToolError(
                "no active session: call whitebox_load_session(data_path, model_path, "
                "feature_names, weight_col) first"
            )
        return self.wb


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _safe_path(path):
    """Resolve a path and reject anything that does not exist (basic sanitation)."""
    if not isinstance(path, str) or not path:
        raise ToolError("path must be a non-empty string")
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        raise ToolError(f"file not found: {path}")
    return resolved


def _read_table(path):
    import pandas as pd

    p = _safe_path(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith((".csv", ".txt")):
        return pd.read_csv(p)
    raise ToolError(f"unsupported data format for {path}; use .csv or .parquet")


def _variable_meta(wb, name):
    rec = wb.encoder.registry.get(name, {})
    return {
        "name": name,
        "origin": rec.get("origin"),
        "review_status": rec.get("review_status"),
        "dtype": rec.get("dtype"),
        "created_by": rec.get("created_by"),
        "sources": rec.get("sources"),
        "flags": (rec.get("data_quality") or {}).get("flags", []),
    }


# ----------------------------------------------------------------------
# session / introspection
# ----------------------------------------------------------------------
def load_session(session, data_path, model_path, feature_names, weight_col,
                 actuals_col=None, link_fn="poisson"):
    import xgboost as xgb

    from ..core import Whitebox

    data = _read_table(data_path)
    mp = _safe_path(model_path)
    model = xgb.Booster()
    try:
        model.load_model(mp)
    except Exception as e:  # noqa: BLE001
        raise ToolError(
            f"could not load xgboost model from {model_path}: {e}. "
            "Provide an xgboost model saved with Booster.save_model (.json/.ubj)."
        )
    missing = [c for c in feature_names if c not in data.columns]
    if missing:
        raise ToolError(f"feature_names not in data: {missing}")
    wb = Whitebox(
        data=data,
        model=model,
        weight_col=weight_col,
        actuals_col=actuals_col,
        feature_names=list(feature_names),
        link_fn=link_fn,
        verbose=False,
    )
    session.wb = wb
    return {
        "loaded": True,
        "n_rows": int(len(data)),
        "feature_names": list(feature_names),
        "variables": wb.encoder.list_variables(),
    }


def list_variables(session, status=None, origin=None, limit=DEFAULT_LIMIT, offset=0):
    wb = session.require()
    names = wb.encoder.list_variables(status=status, origin=origin)
    total = len(names)
    page = names[offset:offset + limit]
    return {
        "total_count": total,
        "count": len(page),
        "offset": offset,
        "items": [_variable_meta(wb, n) for n in page],
        "has_more": offset + limit < total,
        "next_offset": offset + limit if offset + limit < total else None,
    }


def describe_variable(session, name):
    wb = session.require()
    if name not in wb.encoder.registry:
        raise ToolError(
            f"unknown variable {name!r}; call whitebox_list_variables to see available names"
        )
    meta = _variable_meta(wb, name)
    meta["mapping"] = wb.category_mappings.get(name)
    return meta


def get_category_mapping(session, name):
    wb = session.require()
    if name not in wb.category_mappings:
        raise ToolError(f"{name!r} has no category mapping (it may be numeric or unknown)")
    return {"name": name, "mapping": wb.category_mappings[name]}


# ----------------------------------------------------------------------
# derived variables + status
# ----------------------------------------------------------------------
def add_group(session, name, source, level_map, default=None):
    wb = session.require()
    try:
        wb.add_group(name, source, level_map, default=default)
    except ValueError as e:
        raise ToolError(str(e))
    return _variable_meta(wb, name)


def add_combination(session, name, sources, sep="_x_"):
    wb = session.require()
    try:
        wb.add_combination(name, sources, sep=sep)
    except ValueError as e:
        raise ToolError(str(e))
    return _variable_meta(wb, name)


def set_variable_status(session, name, status):
    wb = session.require()
    try:
        wb.set_status(name, status)
    except ValueError as e:
        raise ToolError(str(e))
    return _variable_meta(wb, name)


def remove_variable(session, name):
    wb = session.require()
    try:
        wb.remove_variable(name)
    except ValueError as e:
        raise ToolError(str(e))
    return {"removed": name}


# ----------------------------------------------------------------------
# governance
# ----------------------------------------------------------------------
def ingest_data(session, data_path):
    wb = session.require()
    new_data = _read_table(data_path)
    return wb.ingest(new_data)


def profile_variable(session, name):
    wb = session.require()
    if name not in wb.data.columns:
        raise ToolError(f"unknown variable {name!r}")
    return wb.profile(name)


def propose_cleaning(session, name, method="by_bins", n_bins=10):
    wb = session.require()
    if name not in wb.data.columns:
        raise ToolError(f"unknown variable {name!r}")
    return wb.propose_cleaning(name, method=method, n_bins=n_bins)


def apply_cleaning(session, name, proposal):
    wb = session.require()
    try:
        wb.apply_cleaning(name, proposal)
    except (ValueError, KeyError) as e:
        raise ToolError(f"could not apply cleaning to {name!r}: {e}")
    return _variable_meta(wb, name)


def sync_variables(session):
    wb = session.require()
    return {"variables": wb.encoder.list_variables(), "trainable": wb.trainable_variables()}


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def _save_fig_html(fig, out_dir, filename):
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    os.makedirs(out_dir, exist_ok=True)
    html = file_html(fig, CDN, filename)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write(html)
    return path


def univariate_plot(session, var, out_dir="whitebox_report_exports", shap=True,
                    glm=False, actuals=False, weight=True):
    wb = session.require()
    if var not in wb.plottable_variables():
        raise ToolError(
            f"{var!r} is not plottable; call whitebox_list_variables for available names"
        )
    fig = wb.univariate_plot(
        var, shap=shap, glm=glm, actuals=actuals, weight=weight, show=False
    )
    path = _save_fig_html(fig, out_dir, f"univariate_{var}.html")
    return {"variable": var, "html_path": path}


def bivariate_plot(session, var1, var2, out_dir="whitebox_report_exports"):
    wb = session.require()
    plottable = wb.plottable_variables()
    for v in (var1, var2):
        if v not in plottable:
            raise ToolError(f"{v!r} is not plottable; call whitebox_list_variables")
    fig = wb.bivariate_plot(var1, var2, shap=True, show=False)
    path = _save_fig_html(fig, out_dir, f"bivariate_{var1}_x_{var2}.html")
    return {"var1": var1, "var2": var2, "html_path": path}
