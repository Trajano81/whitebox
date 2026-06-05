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
