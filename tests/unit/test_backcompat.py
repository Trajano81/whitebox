"""Backward compatibility: the legacy pre-encoded path matches internal encoding (spec-001)."""
import numpy as np
import pandas as pd

from whitebox import Whitebox


def _make_wb(data, model, feature_names, category_mappings=None):
    return Whitebox(
        data=data,
        model=model,
        weight_col="exposure",
        actuals_col="claim_count",
        feature_names=feature_names,
        link_fn="poisson",
        category_mappings=category_mappings,
        verbose=False,
    )


def test_legacy_preencoded_matches_internal(messy_df, trained_model, feature_names):
    # New way: raw object categoricals, encoded internally.
    wb_new = _make_wb(messy_df.copy(), trained_model, feature_names)

    # Legacy way: caller pre-built {col}_encoded columns; mapping reconstructed.
    legacy = messy_df.copy()
    for col in ["region", "mexican_states"]:
        legacy[f"{col}_encoded"] = pd.Categorical(legacy[col]).codes
    wb_old = _make_wb(legacy, trained_model, feature_names)

    for col in ["region", "mexican_states"]:
        np.testing.assert_array_equal(
            wb_new.data[f"{col}_encoded"].values,
            wb_old.data[f"{col}_encoded"].values,
        )

    # SHAP values come out identical because the model inputs are identical.
    wb_new.DataPrep.prep_shap_values()
    wb_old.DataPrep.prep_shap_values()
    pd.testing.assert_frame_equal(wb_new.shap_df, wb_old.shap_df)


def test_explicit_category_mappings_honored(messy_df, trained_model, feature_names):
    # Provide an explicit mapping that matches the categorical code order.
    cat = pd.Categorical(messy_df["region"])
    mapping = {"region": {int(c): lab for c, lab in enumerate(cat.categories)}}
    wb = _make_wb(messy_df.copy(), trained_model, feature_names, category_mappings=mapping)
    np.testing.assert_array_equal(wb.data["region_encoded"].values, cat.codes)
    assert wb.category_mappings["region"] == mapping["region"]
