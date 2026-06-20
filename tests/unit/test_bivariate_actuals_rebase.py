"""Regression: bivariate actuals + rebase must not explode the chart.

When a var2 level has zero (or NaN) actuals in the base var1 band, the
engine used to divide by that zero base, producing NaN/inf series that
poisoned the shared y-range and left lines unrendered. The engine now
mirrors the univariate guard: fall back to base=1 and emit a warning.
"""
import numpy as np

import whitebox.engines.bokeh_engine as bokeh_engine


def _renderer_y_values(fig):
    for r in fig.renderers:
        ds = getattr(r, "data_source", None)
        if ds is not None and "y" in ds.data:
            yield np.asarray(ds.data["y"], dtype=float)


def test_bivariate_actuals_rebase_returns_finite_figure(wb_clean, monkeypatch):
    monkeypatch.setattr(bokeh_engine, "show", lambda p: None)

    # numeric var1 banded, categorical var2 (and the reverse) both exercise
    # the sparse-actuals rebase path.
    for var1, var2 in [("region", "vehicle_value"), ("vehicle_value", "region")]:
        fig = wb_clean.bivariate_plot(
            var1, var2, shap=True, actuals=True, rebase=True, show=False
        )
        assert fig is not None

        yr = fig.y_range
        assert yr.start is not None and yr.end is not None
        assert np.isfinite(yr.start) and np.isfinite(yr.end)

        for yv in _renderer_y_values(fig):
            if len(yv):
                assert not np.isinf(yv).any()


def test_bivariate_actuals_rebase_legend_has_separator(wb_clean, monkeypatch):
    from bokeh.models import Legend

    monkeypatch.setattr(bokeh_engine, "show", lambda p: None)
    fig = wb_clean.bivariate_plot(
        "region", "vehicle_value", shap=True, actuals=True, rebase=True, show=False
    )

    labels = []
    for lay in fig.right:
        if isinstance(lay, Legend):
            for item in lay.items:
                value = item.label["value"] if isinstance(item.label, dict) else item.label
                labels.append(str(value))

    actuals_labels = [l for l in labels if "Actuals" in l]
    assert actuals_labels, "expected at least one actuals legend entry"
    # Label must be "Actuals: var2=..." and never the malformed "Actualsvar2=...".
    for label in actuals_labels:
        assert "Actuals: " in label
