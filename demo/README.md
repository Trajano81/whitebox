# MintyPython Demo

This folder contains a self-contained demo that demonstrates all MintyPython features using synthetic insurance-like data.

## Quick Start

### 1. Create and Activate Virtual Environment

We recommend using a virtual environment to avoid conflicts with other packages.

**On macOS/Linux:**
```bash
# Navigate to the project root directory
cd /path/to/plotting_minty

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Navigate to the project root directory
cd \path\to\plotting_minty

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt when the environment is activated.

### 2. Install MintyPython and Dependencies

With the virtual environment activated, install the package:

```bash
# Install mintypython with all dependencies (from project root)
pip install -e .

# Install Jupyter notebook support
pip install jupyter ipykernel
```

### 3. Create a Jupyter Kernel

Register the virtual environment as a Jupyter kernel so you can select it in notebooks:

```bash
# Create Jupyter kernel
python -m ipykernel install --user --name=mintypython-demo --display-name="MintyPython Demo"
```

### 4. Run the Demo Notebook

```bash
# From the project root
jupyter notebook demo/demo_notebook.ipynb

# OR from the demo directory
cd demo
jupyter notebook demo_notebook.ipynb
```

**Using VS Code:** Open `demo_notebook.ipynb` and select "MintyPython Demo" from the kernel picker in the top right.

### Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## What's Included

### Files

- **`synthetic_data.py`**: Module that generates synthetic insurance data and trains an XGBoost model
- **`glm_helpers.py`**: Module that simulates GLM relativities for demonstration (since Emblem exports are proprietary)
- **`demo_notebook.ipynb`**: Jupyter notebook demonstrating all MintyPython methods

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

Since Emblem model exports are proprietary, this demo uses simulated GLM relativities created by `glm_helpers.py`. The simulated relativities mirror realistic GLM factor structures and demonstrate how MintyPython displays GLM comparisons.

If you have actual Emblem exports (`.emb` files), you can use them with MintyPython by passing the `emb_model_export` parameter instead of the simulated `glm_df`.

## Troubleshooting

### ImportError: No module named 'mintypython'
Make sure you installed the package in development mode:
```bash
pip install -e ..  # From demo directory
# OR
pip install -e .   # From project root
```

### Plots not displaying in Jupyter
For Bokeh plots, ensure you have the Bokeh extension loaded:
```python
from bokeh.io import output_notebook
output_notebook()
```

### Missing dependencies
The main `setup.py` includes all required dependencies. If you encounter missing packages:
```bash
pip install -e ..[dev]  # If dev extras are defined
# OR install individually
pip install xgboost shap bokeh matplotlib seaborn pandas numpy
```
