"""Unit tests for the internal Encoder (spec-001)."""
import numpy as np
import pandas as pd

from whitebox.encoding import Encoder


def test_auto_encode_matches_categorical_codes(messy_df, feature_names):
    enc = Encoder(messy_df, feature_names, verbose=False)
    enc.auto_encode()
    for col in ["region", "mexican_states"]:
        expected = pd.Categorical(messy_df[col]).codes
        assert (messy_df[f"{col}_encoded"].values == expected).all()


def test_auto_encode_registers_pending_review(messy_df, feature_names):
    enc = Encoder(messy_df, feature_names, verbose=False)
    enc.auto_encode()
    # Every raw feature is registered, and none is auto-trainable.
    for col in feature_names:
        assert enc.registry[col]["origin"] == "raw"
        assert enc.registry[col]["review_status"] == "pending_review"
    assert enc.trainable_variables() == []


def test_mapping_roundtrip_decode(messy_df, feature_names):
    enc = Encoder(messy_df, feature_names, verbose=False)
    enc.auto_encode()
    codes = messy_df["region_encoded"]
    labels = enc.decode("region", codes)
    # decode(encode(x)) == x for non-missing rows.
    nonmissing = codes >= 0
    assert (labels[nonmissing].values == messy_df["region"][nonmissing].values).all()


def test_reconstruct_mapping_when_encoded_exists(messy_df, feature_names):
    # Pre-build the encoded column externally (legacy path).
    df = messy_df.copy()
    cat = pd.Categorical(df["region"])
    df["region_encoded"] = cat.codes
    enc = Encoder(df, feature_names, verbose=False)
    enc.auto_encode()
    # Mapping reconstructed from the pair, codes unchanged.
    assert enc.category_mappings["region"]
    assert (df["region_encoded"].values == cat.codes).all()


def test_nan_maps_to_negative_one(feature_names):
    df = pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0],
            "vehicle_value": [10.0, 20.0, 30.0],
            "region": ["North", None, "South"],
            "mexican_states": ["CDMX", "Jalisco", "Sonora"],
        }
    )
    enc = Encoder(df, feature_names, verbose=False)
    enc.auto_encode()
    assert df["region_encoded"].iloc[1] == -1


def test_model_input_column_prefers_encoded(messy_df, feature_names):
    enc = Encoder(messy_df, feature_names, verbose=False)
    enc.auto_encode()
    # categorical -> encoded series; numeric -> original series.
    assert enc.model_input_column("region").name == "region_encoded"
    np.testing.assert_array_equal(
        enc.model_input_column("age").values, messy_df["age"].values
    )
