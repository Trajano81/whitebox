"""
Synthetic data generation for MintyPython demo.

This module generates synthetic insurance-like data for demonstrating
MintyPython's visualization capabilities.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


def generate_synthetic_data(n_samples=10000, random_state=42):
    """
    Generate synthetic insurance data with realistic feature relationships.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with features, exposure, and claim counts
    """
    np.random.seed(random_state)

    # Continuous features
    age = np.random.normal(45, 15, n_samples).clip(18, 85)
    vehicle_value = np.random.lognormal(10, 0.5, n_samples).clip(5000, 150000)
    years_licensed = ((age - 18) * np.random.uniform(0.3, 1.0, n_samples)).clip(0, 60)

    # Categorical features
    regions = ['North', 'South', 'East', 'West']
    vehicle_types = ['Sedan', 'SUV', 'Truck', 'Sports']

    region = np.random.choice(regions, n_samples, p=[0.25, 0.30, 0.25, 0.20])
    vehicle_type = np.random.choice(vehicle_types, n_samples, p=[0.40, 0.30, 0.20, 0.10])

    # Exposure (in years, 0.5 to 1.0)
    exposure = np.random.uniform(0.5, 1.0, n_samples)

    # Calculate claim frequency based on feature relationships
    base_rate = 0.12

    # Age effect: younger drivers have higher risk
    age_effect = np.exp(-0.02 * (age - 25))
    age_effect = age_effect.clip(0.5, 2.0)

    # Vehicle type effect
    vehicle_type_effects = {'Sedan': 1.0, 'SUV': 1.1, 'Truck': 0.95, 'Sports': 1.5}
    vehicle_effect = np.array([vehicle_type_effects[v] for v in vehicle_type])

    # Region effect
    region_effects = {'North': 1.0, 'South': 1.1, 'East': 0.95, 'West': 1.05}
    region_effect = np.array([region_effects[r] for r in region])

    # Vehicle value effect (higher value = slightly higher risk)
    value_effect = 1.0 + 0.1 * np.log(vehicle_value / 30000)
    value_effect = value_effect.clip(0.8, 1.3)

    # Experience effect (more years licensed = lower risk)
    experience_effect = np.exp(-0.02 * years_licensed)
    experience_effect = experience_effect.clip(0.6, 1.5)

    # Calculate lambda for Poisson distribution
    lambda_param = (base_rate * age_effect * vehicle_effect * region_effect *
                   value_effect * experience_effect * exposure)

    # Generate claim counts
    claim_count = np.random.poisson(lambda_param)

    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'vehicle_value': vehicle_value,
        'years_licensed': years_licensed,
        'region': region,
        'vehicle_type': vehicle_type,
        'exposure': exposure,
        'claim_count': claim_count
    })

    return data


def train_xgboost_model(data, feature_names, weight_col='exposure',
                        target_col='claim_count', random_state=42, **xgb_params):
    """
    Train an XGBoost model with Poisson objective for claim frequency.

    Parameters
    ----------
    data : pd.DataFrame
        Training data
    feature_names : list
        List of feature column names
    weight_col : str
        Name of weight/exposure column
    target_col : str
        Name of target column
    random_state : int
        Random seed
    **xgb_params : dict
        Additional XGBoost parameters

    Returns
    -------
    xgb.Booster
        Trained XGBoost model
    """
    # Prepare data for XGBoost
    X = data[feature_names].copy()

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    y = data[target_col]
    weights = data[weight_col]

    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y, weight=weights, feature_names=feature_names)

    # Default parameters for Poisson regression
    default_params = {
        'objective': 'count:poisson',
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': random_state,
        'verbosity': 0
    }
    default_params.update(xgb_params)

    # Train model
    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=100,
        verbose_eval=False
    )

    return model


def _is_string_dtype(dtype):
    """Check if dtype is string-like (object, string, or StringDtype)."""
    dtype_str = str(dtype)
    return dtype_str in ['object', 'string'] or 'str' in dtype_str.lower()


def encode_categoricals(data, columns=None, optimize=False, verbose=True):
    """
    Prepare DataFrame for modeling: encode categoricals and optionally optimize numerics.

    - Always encodes object/string columns to integer codes (int8/int16)
    - Always preserves original values in {col}_original columns as category dtype
    - Optionally downcasts numeric columns (int64->int8/16/32, float64->float32)

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    columns : list, optional
        Specific categorical columns to encode. If None, encodes all object/string columns.
    optimize : bool, default False
        If True, also downcasts numeric columns for memory efficiency.
    verbose : bool, default True
        If True, prints encoding and optimization summary.

    Returns
    -------
    pd.DataFrame
        Data with encoded columns and {col}_original columns
    dict
        Mapping of {column: {code: label}}
    """
    data_encoded = data.copy()
    category_mappings = {}
    encoded_columns = []
    numeric_optimized = []

    initial_memory = data.memory_usage(deep=True).sum()

    # Determine categorical columns to encode (object, string, StringDtype)
    if columns is None:
        columns = [col for col in data.columns if _is_string_dtype(data[col].dtype)]

    # Validate columns exist
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Process all columns
    for col in data_encoded.columns:
        dtype = data_encoded[col].dtype
        dtype_str = str(dtype)

        # ALWAYS encode object/string columns
        if col in columns and _is_string_dtype(dtype):
            # Save original values as category dtype
            data_encoded[f'{col}_original'] = data[col].astype('category')
            # Encode to integer codes
            cat_series = data[col].astype('category')
            n_categories = len(cat_series.cat.categories)
            code_dtype = 'int8' if n_categories <= 127 else 'int16'
            data_encoded[col] = cat_series.cat.codes.astype(code_dtype)
            # Store mapping
            category_mappings[col] = dict(enumerate(cat_series.cat.categories))
            encoded_columns.append(col)

        # ONLY optimize numerics if optimize=True
        elif optimize and dtype_str in ['int64', 'int32']:
            original_dtype = dtype
            data_encoded[col] = pd.to_numeric(data_encoded[col], downcast='integer')
            if data_encoded[col].dtype != original_dtype:
                numeric_optimized.append(f"{col}: {original_dtype} -> {data_encoded[col].dtype}")

        elif optimize and dtype_str == 'float64':
            original_dtype = dtype
            data_encoded[col] = pd.to_numeric(data_encoded[col], downcast='float')
            if data_encoded[col].dtype != original_dtype:
                numeric_optimized.append(f"{col}: {original_dtype} -> {data_encoded[col].dtype}")

    final_memory = data_encoded.memory_usage(deep=True).sum()
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100

    if verbose:
        if encoded_columns:
            print(f"Encoded {len(encoded_columns)} categorical column(s): {encoded_columns}")
            print(f"Original values preserved in: {[f'{col}_original' for col in encoded_columns]}")
        if numeric_optimized:
            print(f"Optimized {len(numeric_optimized)} numeric column(s):")
            for change in numeric_optimized:
                print(f"  {change}")
        print(f"Memory: {initial_memory/1024:.1f}KB -> {final_memory/1024:.1f}KB ({memory_reduction:.1f}% reduction)")

    return data_encoded, category_mappings


def prepare_data_for_mintypython(data, feature_names, optimize=False, verbose=True):
    """
    Prepare data DataFrame for use with MintyPython.
    Encodes categorical variables and optionally optimizes numeric dtypes.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data with categorical columns
    feature_names : list
        List of feature column names
    optimize : bool, default False
        If True, also downcasts numeric columns for memory efficiency
    verbose : bool, default True
        If True, prints log messages about encoding and optimization.

    Returns
    -------
    pd.DataFrame
        Data with encoded categorical variables (and {col}_original columns)
    dict
        Mapping of categorical values to codes
    """
    # Validate feature columns exist
    missing_features = [col for col in feature_names if col not in data.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")

    # Get categorical columns from feature_names (object, string, or StringDtype)
    cat_cols = [col for col in feature_names if _is_string_dtype(data[col].dtype)]

    return encode_categoricals(data, columns=cat_cols, optimize=optimize, verbose=verbose)


def get_demo_data_and_model(n_samples=10000, random_state=42):
    """
    Convenience function to get synthetic data and trained model.

    Parameters
    ----------
    n_samples : int
        Number of samples
    random_state : int
        Random seed

    Returns
    -------
    tuple
        (data, model, feature_names, category_mappings)
    """
    # Generate data
    data = generate_synthetic_data(n_samples=n_samples, random_state=random_state)

    # Define features
    feature_names = ['age', 'vehicle_value', 'years_licensed', 'region', 'vehicle_type']

    # Prepare for model training
    data_encoded, category_mappings = prepare_data_for_mintypython(data, feature_names)

    # Train model
    model = train_xgboost_model(
        data_encoded,
        feature_names,
        weight_col='exposure',
        target_col='claim_count',
        random_state=random_state
    )

    return data_encoded, model, feature_names, category_mappings


if __name__ == '__main__':
    # Test data generation
    data, model, feature_names, mappings = get_demo_data_and_model()
    print(f"Generated {len(data)} samples")
    print(f"Features: {feature_names}")
    print(f"Category mappings: {mappings}")
    print(f"\nData summary:")
    print(data.describe())
    print(f"\nClaim count distribution:")
    print(data['claim_count'].value_counts().sort_index().head(10))
