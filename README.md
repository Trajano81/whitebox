# Whitebox

Model-agnostic visualization library for interpreting ML models. Whitebox generates interactive plots showing SHAP values, model predictions, actuals, and GLM relativities to help interpret and compare model behavior.

## Installation

### With Poetry (Recommended)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Whitebox
poetry install
```

### With pip

```bash
pip install whitebox
```

### Development Mode

```bash
git clone https://github.com/your-repo/whitebox.git
cd whitebox
poetry install
```

## Quick Start

```python
from whitebox import Whitebox

# Initialize with your model and data
wb = Whitebox(
    data=df,
    model=xgb_model,
    weight_col='exposure',
    actuals_col='claim_count',
    feature_names=['age', 'region_encoded', 'vehicle_type_encoded']
)

# Generate univariate plot
wb.univariate_plot('age', shap=True, glm=True)

# Generate bivariate plot
wb.bivariate_plot('age', 'region_encoded', shap=True)
```

## Features

- **SHAP Values**: Visualize SHAP values for any tree-based model (XGBoost, LightGBM)
- **GLM Comparison**: Overlay GLM relativities from Emblem model exports
- **Interactive Plots**: Bokeh-powered interactive plots with tooltips
- **Bivariate Analysis**: Two-way interaction plots
- **Model Comparison**: Compare multiple models side-by-side
- **Categorical Support**: Display "Label(code)" format for categorical variables

## Categorical Variable Encoding

For categorical variables, use the `_encoded` suffix convention:

```python
# Create encoded columns
df['region_encoded'] = df['region'].map({'North': 0, 'South': 1, 'East': 2, 'West': 3})

# Define mappings
category_mappings = {
    'region': {0: 'North', 1: 'South', 2: 'East', 3: 'West'}
}

# Initialize Whitebox with mappings
wb = Whitebox(
    data=df,
    model=model,
    feature_names=['region_encoded'],
    category_mappings=category_mappings,
    ...
)
```

## Documentation

See the [demo notebook](demo/demo_notebook.ipynb) for comprehensive examples.

## License

MIT
