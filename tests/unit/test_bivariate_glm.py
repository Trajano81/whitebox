"""Bivariate GLM path honors glm_var_map (regression for the demo notebook)."""
import numpy as np
import pandas as pd
import pytest

import whitebox.engines.bokeh_engine as bokeh_engine


def _attach_glm(wb):
    n = len(wb.data)
    # GLM export uses different column names than the GBM features.
    wb.glm_df = pd.DataFrame(
        {
            "age_band": np.linspace(-0.1, 0.1, n),
            "territory": np.zeros(n),
        }
    )
    wb.glm_var_map = {"age": "age_band", "region": "territory"}


def test_bivariate_glm_uses_var_map(wb_clean, monkeypatch):
    monkeypatch.setattr(bokeh_engine, "show", lambda p: None)
    _attach_glm(wb_clean)
    # var1='age' must resolve to glm_df['age_band'] via glm_var_map.
    fig = wb_clean.bivariate_plot("age", "region", shap=True, glm=True, show=False)
    assert fig is not None


def test_bivariate_glm_missing_column_actionable(wb_clean, monkeypatch):
    monkeypatch.setattr(bokeh_engine, "show", lambda p: None)
    _attach_glm(wb_clean)
    # vehicle_value has no glm column and no mapping -> actionable error.
    with pytest.raises(ValueError) as e:
        wb_clean.bivariate_plot("vehicle_value", "region", shap=True, glm=True, show=False)
    assert "glm_var_map" in str(e.value)
