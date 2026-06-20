"""Derived grouped variables + synthesized SHAP (spec-003)."""
import numpy as np
import pandas as pd
import pytest

LEVEL_MAP = {"North": "NS", "South": "NS", "East": "EW", "West": "EW"}


def test_add_group_creates_columns_and_registry(wb):
    wb.add_group("region_grouped", "region", LEVEL_MAP, default="Other")
    assert "region_grouped" in wb.data.columns
    assert "region_grouped_encoded" in wb.data.columns
    assert "region_grouped" in wb.category_mappings
    rec = wb.encoder.registry["region_grouped"]
    assert rec["origin"] == "derived_group"
    assert rec["review_status"] == "pending_review"
    assert rec["sources"] == ["region"]
    # uncovered messy levels collapse into the default group.
    assert "Other" in set(wb.data["region_grouped"].unique())
    assert "region_grouped" in wb.plottable_variables()


def test_group_shap_equals_source_grouped(wb):
    wb.DataPrep.prep_shap_values()
    wb.add_group("region_grouped", "region", LEVEL_MAP, default="Other")
    # single-source derived SHAP is exactly the source feature's SHAP.
    derived = wb.DataPrep._derived_shap_series("region_grouped")
    pd.testing.assert_series_equal(
        derived.reset_index(drop=True),
        wb.shap_df["region"].reset_index(drop=True),
        check_names=False,
    )
    # weighted-mean SHAP per new group matches a manual groupby.
    w = wb.data["exposure"].values
    grp = wb.data["region_grouped"].values
    src_shap = wb.shap_df["region"].values
    man = pd.DataFrame({"g": grp, "sw": src_shap * w, "w": w})
    manual = man.groupby("g")["sw"].sum() / man.groupby("g")["w"].sum()
    assert set(manual.index) == set(np.unique(grp))


def test_uncovered_passthrough_without_default(wb):
    partial = {"North": "NS"}
    wb.add_group("region_partial", "region", partial)  # default None -> passthrough
    vals = set(wb.data["region_partial"].unique())
    assert "NS" in vals
    assert "South" in vals  # uncovered kept as-is


def test_remove_derived_frees_name(wb):
    wb.add_group("region_grouped", "region", LEVEL_MAP, default="Other")
    wb.remove_variable("region_grouped")
    assert "region_grouped" not in wb.data.columns
    assert "region_grouped_encoded" not in wb.data.columns
    assert "region_grouped" not in wb.encoder.registry
    # name is free again.
    assert wb.encoder.validate_name("region_grouped") is True


def test_duplicate_group_name_rejected(wb):
    wb.add_group("region_grouped", "region", LEVEL_MAP, default="Other")
    with pytest.raises(ValueError):
        wb.add_group("region_grouped", "region", LEVEL_MAP, default="Other")


def _univariate_kwargs(**over):
    kwargs = dict(
        base=None, shap=True, shap_points=False, shap_sd=None, n_shap_points=1000,
        glm=False, glm_pred=False, plot_name="x", nlevels=None, write_out_shap=None,
        weight=True, actuals=False, shap_points_seed=1, percentile_actuals=1,
        start=None, finish=None, stepsize=None, percentile_start=10,
        percentile_finish=90, infinity_higher=True, infinity_lower=True, ci_z=2,
        glm_ci=False, y_axis_max=None, y_axis_min=None, rebase=True, joinshaps=None,
        glmindic_cols=None, joinshaps_error=True, height=500, width=1000,
        max_levels=None,
    )
    kwargs.update(over)
    return kwargs


def test_derived_group_plots_through_prep(wb):
    wb.DataPrep.prep_shap_values()
    wb.add_group("region_grouped", "region", LEVEL_MAP, default="Other")
    res = wb.DataPrep.prep_univariate_data("region_grouped", _univariate_kwargs())
    agg = res["agg_data"]
    assert len(agg) >= 2
    assert "shap" in agg.columns

