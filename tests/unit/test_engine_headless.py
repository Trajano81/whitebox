"""Headless rendering: show=False returns a figure without calling Bokeh show (spec-005)."""
import whitebox.engines.bokeh_engine as bokeh_engine


def test_univariate_show_false_does_not_call_show(wb, monkeypatch):
    calls = []
    monkeypatch.setattr(bokeh_engine, "show", lambda p: calls.append(p))
    fig = wb.univariate_plot("region", shap=True, weight=True, show=False)
    assert fig is not None
    assert calls == []  # show() was not invoked


def test_univariate_show_true_calls_show(wb, monkeypatch):
    calls = []
    monkeypatch.setattr(bokeh_engine, "show", lambda p: calls.append(p))
    wb.univariate_plot("region", shap=True, weight=True, show=True)
    assert len(calls) == 1


def test_bivariate_show_false_headless(wb_clean, monkeypatch):
    calls = []
    monkeypatch.setattr(bokeh_engine, "show", lambda p: calls.append(p))
    fig = wb_clean.bivariate_plot("age", "region", shap=True, show=False)
    assert fig is not None
    assert calls == []
