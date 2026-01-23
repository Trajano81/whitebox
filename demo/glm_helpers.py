"""
GLM simulation helpers for MintyPython demo.

Since Emblem model exports are proprietary, this module provides simulated
GLM relativities that demonstrate MintyPython's GLM comparison features.
"""

import numpy as np
import pandas as pd


def create_glm_relativities(data, feature_names):
    """
    Create simulated GLM relativities for demo purposes.

    This function generates a DataFrame with GLM factor relativities that
    simulate what would be extracted from an Emblem model export via scorepyon.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing feature values
    feature_names : list
        List of feature names to create relativities for

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per feature containing log-relativities
    """
    glm_df = pd.DataFrame(index=data.index)

    # Age relativities (log scale, younger = higher risk)
    # GLM uses log link, so these are log-relativities
    if 'age' in feature_names:
        age_min = data['age'].min()
        age_max = data['age'].max()
        # Linear interpolation: lowest age = log(1.2), highest age = log(0.8)
        log_rel_min = np.log(1.2)  # ~0.182 for lowest age
        log_rel_max = np.log(0.8)  # ~-0.223 for highest age
        # Normalize age to [0, 1] then interpolate
        age_normalized = (data['age'] - age_min) / (age_max - age_min)
        glm_df['age'] = log_rel_min + age_normalized * (log_rel_max - log_rel_min)

    # Vehicle value relativities (log scale)
    if 'vehicle_value' in feature_names:
        value_centered = np.log(data['vehicle_value']) - np.log(30000)  # Base at 30k
        glm_df['vehicle_value'] = 0.08 * value_centered

    # Years licensed relativities (log scale, more experience = lower risk)
    if 'years_licensed' in feature_names:
        exp_centered = data['years_licensed'] - 10  # Base at 10 years
        glm_df['years_licensed'] = -0.018 * exp_centered

    # Region relativities (categorical, log scale)
    if 'region' in feature_names:
        # Map numeric codes to log-relativities
        # Assuming: 0=East, 1=North, 2=South, 3=West (alphabetical encoding)
        region_effects = {0: -0.05, 1: 0.0, 2: 0.10, 3: 0.05}  # North as base
        glm_df['region'] = data['region'].map(region_effects)

    # Vehicle type relativities (categorical, log scale)
    if 'vehicle_type' in feature_names:
        # Assuming: 0=SUV, 1=Sedan, 2=Sports, 3=Truck (alphabetical encoding)
        vtype_effects = {0: 0.10, 1: 0.0, 2: 0.40, 3: -0.05}  # Sedan as base
        glm_df['vehicle_type'] = data['vehicle_type'].map(vtype_effects)

    return glm_df


def create_glm_predictions(data, glm_df, exposure_col='exposure'):
    """
    Create GLM predictions from relativities.

    Parameters
    ----------
    data : pd.DataFrame
        Original data with exposure
    glm_df : pd.DataFrame
        DataFrame with log-relativities from create_glm_relativities
    exposure_col : str
        Name of the exposure column

    Returns
    -------
    pd.Series
        GLM predictions (claim frequency)
    """
    # Sum log-relativities
    log_prediction = glm_df.sum(axis=1)

    # Add intercept (base rate)
    base_rate_log = np.log(0.12)  # Same base rate as in synthetic data
    log_prediction = log_prediction + base_rate_log

    # Convert to response scale (exponentiate) and multiply by exposure
    prediction = np.exp(log_prediction) * data[exposure_col]

    return prediction


def get_glm_data_for_mintypython(data, feature_names):
    """
    Convenience function to get GLM data formatted for MintyPython.

    Parameters
    ----------
    data : pd.DataFrame
        The input data
    feature_names : list
        List of feature names

    Returns
    -------
    tuple
        (glm_df, glm_predictions)
        - glm_df: DataFrame with log-relativities for each feature
        - glm_predictions: Series with GLM predicted claim counts
    """
    glm_df = create_glm_relativities(data, feature_names)
    glm_preds = create_glm_predictions(data, glm_df)

    return glm_df, glm_preds


def create_category_mapping_dict(category_mappings):
    """
    Create a mapping dictionary suitable for MintyPython's mapping_dict parameter.

    This helps label categorical variables in plots with their actual names
    instead of numeric codes.

    Parameters
    ----------
    category_mappings : dict
        Dictionary from prepare_data_for_mintypython with
        {column: {code: label}} structure

    Returns
    -------
    dict
        Mapping dictionary for MintyPython
    """
    mapping_dict = {}
    for col, code_to_label in category_mappings.items():
        # MintyPython expects {column: {code: label}}
        mapping_dict[col] = code_to_label
    return mapping_dict


if __name__ == '__main__':
    # Test GLM helpers
    from synthetic_data import get_demo_data_and_model

    data, model, feature_names, category_mappings = get_demo_data_and_model(n_samples=1000)

    print("Testing GLM helpers...")
    glm_df, glm_preds = get_glm_data_for_mintypython(data, feature_names)

    print(f"\nGLM DataFrame columns: {list(glm_df.columns)}")
    print(f"\nGLM relativities summary:")
    print(glm_df.describe())

    print(f"\nGLM predictions summary:")
    print(f"Mean: {glm_preds.mean():.4f}")
    print(f"Std: {glm_preds.std():.4f}")
    print(f"Min: {glm_preds.min():.4f}")
    print(f"Max: {glm_preds.max():.4f}")

    print(f"\nActual claim_count mean: {data['claim_count'].mean():.4f}")
