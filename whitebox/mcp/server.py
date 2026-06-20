"""FastMCP (stdio) server exposing whitebox tools.

Run with: python -m whitebox.mcp.server

The tool logic lives in tools.py; this module only wires it to FastMCP, applies
tool annotations, and converts ToolError into actionable in-result errors. Per the
stdio contract, nothing is logged to stdout (FastMCP handles transport).
"""
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from . import tools as T

mcp = FastMCP("whitebox_mcp")
_session = T.Session()


def _wrap(fn, *args, **kwargs):
    try:
        return fn(_session, *args, **kwargs)
    except T.ToolError as e:
        return {"isError": True, "message": str(e)}


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": True})
def whitebox_load_session(data_path: str, model_path: str, feature_names: List[str],
                          weight_col: str, actuals_col: Optional[str] = None,
                          link_fn: str = "poisson") -> Dict[str, Any]:
    """Load a dataset (.csv/.parquet) and an xgboost model, building a Whitebox session."""
    return _wrap(T.load_session, data_path, model_path, feature_names, weight_col,
                 actuals_col=actuals_col, link_fn=link_fn)


@mcp.tool(annotations={"readOnlyHint": True})
def whitebox_list_variables(status: Optional[str] = None, origin: Optional[str] = None,
                            limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """List variables (raw + derived) with governance metadata; paginated."""
    return _wrap(T.list_variables, status=status, origin=origin, limit=limit, offset=offset)


@mcp.tool(annotations={"readOnlyHint": True})
def whitebox_describe_variable(name: str) -> Dict[str, Any]:
    """Describe a variable: origin, status, dtype, sources, mapping."""
    return _wrap(T.describe_variable, name)


@mcp.tool(annotations={"readOnlyHint": True})
def whitebox_get_category_mapping(name: str) -> Dict[str, Any]:
    """Return the {code: label} mapping for a categorical variable."""
    return _wrap(T.get_category_mapping, name)


@mcp.tool(annotations={"idempotentHint": True})
def whitebox_add_group(name: str, source: str, level_map: Dict[str, str],
                       default: Optional[str] = None) -> Dict[str, Any]:
    """Create a grouped derived variable (pending_review). Name must be a unique slug."""
    return _wrap(T.add_group, name, source, level_map, default=default)


@mcp.tool(annotations={"idempotentHint": True})
def whitebox_add_combination(name: str, sources: List[str], sep: str = "_x_") -> Dict[str, Any]:
    """Create a combined derived variable from 2+ sources (pending_review)."""
    return _wrap(T.add_combination, name, sources, sep=sep)


@mcp.tool()
def whitebox_set_variable_status(name: str, status: str) -> Dict[str, Any]:
    """Set a variable's review status (pending_review/ready_to_model/needs_cleaning/excluded)."""
    return _wrap(T.set_variable_status, name, status)


@mcp.tool(annotations={"destructiveHint": True})
def whitebox_remove_variable(name: str) -> Dict[str, Any]:
    """Delete a derived variable, freeing its primary-key name."""
    return _wrap(T.remove_variable, name)


@mcp.tool(annotations={"readOnlyHint": True})
def whitebox_ingest_data(data_path: str) -> Dict[str, Any]:
    """Diff a new dataset's schema vs the current variables: new/missing/changed."""
    return _wrap(T.ingest_data, data_path)


@mcp.tool(annotations={"readOnlyHint": True})
def whitebox_profile_variable(name: str) -> Dict[str, Any]:
    """Profile a variable: missingness, odd tokens, cardinality, flags."""
    return _wrap(T.profile_variable, name)


@mcp.tool(annotations={"readOnlyHint": True})
def whitebox_propose_cleaning(name: str, method: str = "by_bins", n_bins: int = 10) -> Dict[str, Any]:
    """Propose an editable cleaning (categorical level_map) or banding (numeric, default 10 bins)."""
    return _wrap(T.propose_cleaning, name, method=method, n_bins=n_bins)


@mcp.tool()
def whitebox_apply_cleaning(name: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a (possibly edited) cleaning/banding proposal to a variable."""
    return _wrap(T.apply_cleaning, name, proposal)


@mcp.tool(annotations={"readOnlyHint": True})
def whitebox_sync_variables() -> Dict[str, Any]:
    """Return the current global variable namespace + the trainable manifest."""
    return _wrap(T.sync_variables)


@mcp.tool()
def whitebox_univariate_plot(var: str, out_dir: str = "whitebox_report_exports",
                             shap: bool = True, glm: bool = False, actuals: bool = False,
                             weight: bool = True) -> Dict[str, Any]:
    """Render a univariate plot to an HTML file; returns its path."""
    return _wrap(T.univariate_plot, var, out_dir=out_dir, shap=shap, glm=glm,
                 actuals=actuals, weight=weight)


@mcp.tool()
def whitebox_bivariate_plot(var1: str, var2: str,
                            out_dir: str = "whitebox_report_exports") -> Dict[str, Any]:
    """Render a bivariate plot to an HTML file; returns its path."""
    return _wrap(T.bivariate_plot, var1, var2, out_dir=out_dir)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
