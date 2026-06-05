"""Shared pytest fixtures for whitebox tests.

Provides a small, deliberately messy synthetic dataset (missing tokens, odd
characters, a high-cardinality categorical) used to exercise encoding and
governance, plus a trained XGBoost model on it.
"""
import numpy as np
import pandas as pd
import pytest

MEXICAN_STATES = [
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
    "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango", "Guanajuato",
    "Guerrero", "Hidalgo", "Jalisco", "Mexico", "Michoacan", "Morelos",
    "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla", "Queretaro", "Quintana Roo",
    "San Luis Potosi", "Sinaloa", "Sonora", "Tabasco", "Tamaulipas", "Tlaxcala",
    "Veracruz", "Yucatan", "Zacatecas", "CDMX",
]


@pytest.fixture
def messy_df():
    """A small messy dataset: clean numerics, a categorical with missing tokens
    and odd characters, and a 32-level mexican_states column."""
    rng = np.random.RandomState(42)
    n = 600
    region = rng.choice(["North", "South", "East", "West"], n).astype(object)
    # Inject messy missing tokens and odd chars into ~10% of region.
    region[rng.choice(n, 30, replace=False)] = ""
    region[rng.choice(n, 20, replace=False)] = "NA"
    region[rng.choice(n, 10, replace=False)] = "."
    region[rng.choice(n, 10, replace=False)] = " North "  # whitespace oddity

    df = pd.DataFrame(
        {
            "age": rng.normal(45, 15, n).clip(18, 85),
            "vehicle_value": rng.lognormal(10, 0.5, n).clip(5000, 150000),
            "region": region,
            "mexican_states": rng.choice(MEXICAN_STATES, n).astype(object),
            "exposure": rng.uniform(0.5, 1.0, n),
        }
    )
    # Poisson-ish claim counts.
    rate = 0.1 * np.exp(-0.01 * (df["age"] - 40))
    df["claim_count"] = rng.poisson(rate * df["exposure"]).astype(float)
    return df


@pytest.fixture
def feature_names():
    return ["age", "vehicle_value", "region", "mexican_states"]


def _encode_for_training(df, feature_names):
    """Encode categoricals with pd.Categorical codes (matches Encoder.auto_encode)."""
    cols = {}
    for c in feature_names:
        if df[c].dtype == object or str(df[c].dtype) == "category":
            cols[c] = pd.Categorical(df[c]).codes
        else:
            cols[c] = df[c].values
    return pd.DataFrame(cols, index=df.index)


@pytest.fixture
def trained_model(messy_df, feature_names):
    import xgboost as xgb

    X = _encode_for_training(messy_df, feature_names)
    dtrain = xgb.DMatrix(
        X,
        label=messy_df["claim_count"],
        weight=messy_df["exposure"],
        feature_names=feature_names,
    )
    params = {"objective": "count:poisson", "max_depth": 3, "eta": 0.3, "verbosity": 0}
    return xgb.train(params, dtrain, num_boost_round=15)


@pytest.fixture
def wb(messy_df, trained_model, feature_names):
    from whitebox import Whitebox

    return Whitebox(
        data=messy_df.copy(),
        model=trained_model,
        weight_col="exposure",
        actuals_col="claim_count",
        feature_names=feature_names,
        link_fn="poisson",
        verbose=False,
    )


@pytest.fixture
def clean_df():
    """A larger, clean dataset (no messy tokens) for plotting-engine tests."""
    rng = np.random.RandomState(7)
    n = 1500
    df = pd.DataFrame(
        {
            "age": rng.normal(45, 15, n).clip(18, 85),
            "vehicle_value": rng.lognormal(10, 0.5, n).clip(5000, 150000),
            "region": rng.choice(["North", "South", "East", "West"], n).astype(object),
            "vehicle_type": rng.choice(["Sedan", "SUV", "Truck"], n).astype(object),
            "exposure": rng.uniform(0.5, 1.0, n),
        }
    )
    rate = 0.1 * np.exp(-0.01 * (df["age"] - 40))
    df["claim_count"] = rng.poisson(rate * df["exposure"]).astype(float)
    return df


@pytest.fixture
def wb_clean(clean_df):
    import xgboost as xgb

    from whitebox import Whitebox

    fn = ["age", "vehicle_value", "region", "vehicle_type"]
    X = _encode_for_training(clean_df, fn)
    dtrain = xgb.DMatrix(
        X, label=clean_df["claim_count"], weight=clean_df["exposure"], feature_names=fn
    )
    model = xgb.train(
        {"objective": "count:poisson", "max_depth": 3, "eta": 0.3, "verbosity": 0},
        dtrain,
        num_boost_round=20,
    )
    return Whitebox(
        data=clean_df.copy(),
        model=model,
        weight_col="exposure",
        actuals_col="claim_count",
        feature_names=fn,
        link_fn="poisson",
        verbose=False,
    )
