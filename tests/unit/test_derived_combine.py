"""Derived combined variables + summed SHAP (spec-003)."""
import pandas as pd
import pytest


def test_add_combination_creates_columns_and_labels(wb):
    wb.add_combination("region_x_states", ["region", "mexican_states"])
    assert "region_x_states" in wb.data.columns
    assert "region_x_states_encoded" in wb.data.columns
    rec = wb.encoder.registry["region_x_states"]
    assert rec["origin"] == "derived_combine"
    assert rec["sources"] == ["region", "mexican_states"]
    assert rec["sep"] == "_x_"
    # combined labels join sources with the safe separator.
    sample = wb.data["region_x_states"].iloc[0]
    assert "_x_" in sample


def test_combine_shap_is_sum_of_sources(wb):
    wb.DataPrep.prep_shap_values()
    wb.add_combination("region_x_states", ["region", "mexican_states"])
    derived = wb.DataPrep._derived_shap_series("region_x_states")
    expected = wb.shap_df["region"] + wb.shap_df["mexican_states"]
    pd.testing.assert_series_equal(
        derived.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_combine_requires_two_sources(wb):
    with pytest.raises(ValueError):
        wb.add_combination("just_one", ["region"])
