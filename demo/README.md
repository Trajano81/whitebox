# Whitebox Demo

This folder contains a self-contained demo that demonstrates all Whitebox features using synthetic insurance-like data.

## Quick Start with Poetry (Recommended)

### 1. Install Poetry

If you haven't installed Poetry yet:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Configure Poetry for In-Project Virtual Environment

```bash
# Navigate to the project root directory
cd /path/to/whitebox

# Configure Poetry to create .venv in project folder (optional but recommended)
poetry config virtualenvs.in-project true --local
```

### 3. Install Dependencies

```bash
# Install all dependencies (creates .venv/ in project folder)
poetry install

# Install dev dependencies (includes jupyter and ipykernel)
poetry install --with dev
```

### 4. Create Jupyter Kernel

```bash
# Create a Jupyter kernel for this environment
poetry run python -m ipykernel install --user --name=whitebox-demo --display-name="Whitebox Demo"
```

### 5. Run the Demo Notebook

**Option A: Using Jupyter in browser**
```bash
# Activate shell first, then run jupyter
poetry shell
jupyter notebook demo/demo_notebook.ipynb

# Or run directly with poetry run
poetry run jupyter notebook demo/demo_notebook.ipynb
```

**Option B: Using VS Code (Recommended)**
1. Open VS Code in the project folder:
   ```bash
   code /path/to/whitebox
   ```
2. Install the "Jupyter" extension if not already installed
3. Open `demo/demo_notebook.ipynb`
4. Click "Select Kernel" in the top right corner
5. Choose "Whitebox Demo" from the kernel list
6. Run the cells with `Shift+Enter`

> **Tip:** If "Whitebox Demo" doesn't appear, run the kernel installation command from Step 4 and restart VS Code.

## Alternative: Traditional Virtual Environment

### 1. Create and Activate Virtual Environment

**On macOS/Linux:**
```bash
# Navigate to the project root directory
cd /path/to/whitebox

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Navigate to the project root directory
cd \path\to\whitebox

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 2. Install Whitebox

```bash
# Install the package in development mode
pip install -e .

# Install Jupyter notebook support
pip install jupyter ipykernel
```

### 3. Create a Jupyter Kernel

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=whitebox-demo --display-name="Whitebox Demo"
```

### 4. Run the Demo Notebook

```bash
jupyter notebook demo/demo_notebook.ipynb
```

**Using VS Code:** Open `demo_notebook.ipynb` and select "Whitebox Demo" from the kernel picker.

### Deactivating the Virtual Environment

```bash
deactivate
```

## What's Included

### Files

- **`synthetic_data.py`**: Module that generates synthetic insurance data and trains an XGBoost model
- **`glm_helpers.py`**: Module that simulates GLM relativities for demonstration
- **`demo_notebook.ipynb`**: Jupyter notebook demonstrating all Whitebox methods

### Demo Coverage

The notebook demonstrates:

1. **Data Generation**: Creating synthetic insurance data with realistic feature relationships
2. **Model Training**: Training XGBoost models with Poisson objective for claim frequency
3. **Univariate Plots**:
   - Basic SHAP visualization
   - SHAP with standard deviation bands
   - GLM relativities overlay
   - Actuals and weights display
   - Custom binning options
   - Categorical variables
4. **Bivariate Plots**:
   - Two continuous variables interaction
   - Continuous x categorical interaction
   - With SHAP and GLM overlays
5. **Model Comparison**: Comparing multiple models using the `compare()` method
6. **Configuration**: Customizing colors, labels, and line widths
7. **Plot Engines**: Both Bokeh (interactive) and Matplotlib (static) engines

## Categorical Variable Encoding

When working with categorical variables, Whitebox supports displaying labels in "Label(code)" format (e.g., "North(0)", "South(1)"). To enable this:

### 1. Create encoded columns with `_encoded` suffix

```python
# Original categorical column
df['region'] = ['North', 'South', 'East', 'West', ...]

# Create encoded version with _encoded suffix
df['region_encoded'] = df['region'].map({
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3
})
```

### 2. Define category mappings

```python
category_mappings = {
    'region': {0: 'North', 1: 'South', 2: 'East', 3: 'West'},
    'vehicle_type': {0: 'Sedan', 1: 'SUV', 2: 'Truck', 3: 'Sports'}
}
```

### 3. Pass mappings to Whitebox

```python
wb = Whitebox(
    data=df,
    model=model,
    feature_names=['age', 'region_encoded', 'vehicle_type_encoded', ...],
    category_mappings=category_mappings,
    weight_col='exposure',
    actuals_col='claim_count'
)
```

### 4. Plot with automatic label display

```python
# Labels will show as "North(0)", "South(1)", etc.
wb.univariate_plot('region_encoded', shap=True)
```

## Synthetic Data Description

The generated dataset simulates auto insurance claims with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Continuous | Driver age (18-85), normally distributed around 45 |
| `vehicle_value` | Continuous | Vehicle value, log-normally distributed |
| `years_licensed` | Continuous | Years the driver has been licensed |
| `region` | Categorical | Geographic region (North, South, East, West) |
| `vehicle_type` | Categorical | Type of vehicle (Sedan, SUV, Truck, Sports) |
| `exposure` | Continuous | Policy exposure in years (0.5-1.0) |
| `claim_count` | Response | Number of claims (Poisson distributed) |

The claim frequency is modeled with realistic relationships:
- Younger drivers have higher claim rates
- Sports cars have higher claim rates
- Regional variations exist
- Higher vehicle values correlate with slightly higher claim rates

## GLM Integration Note

This demo uses simulated GLM relativities created by `glm_helpers.py`. The simulated relativities mirror realistic GLM factor structures and demonstrate how Whitebox displays GLM comparisons.

You can integrate GLM coefficients from any source (R, Python statsmodels, etc.) by creating a DataFrame with factor relativities and passing it via the `glm_df` parameter. Use `glm_var_map` to map GBM variable names to GLM column names if they differ.

## Troubleshooting

### ImportError: No module named 'whitebox'
Make sure you installed the package:
```bash
# With Poetry
poetry install

# Or with pip
pip install -e .
```

### Plots not displaying in Jupyter
For Bokeh plots, ensure you have the Bokeh extension loaded:
```python
from bokeh.io import output_notebook
output_notebook()
```

### Missing dependencies
With Poetry:
```bash
poetry install
```

With pip:
```bash
pip install -e .
```
