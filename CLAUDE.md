# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MintyPython (Model INTerpretation with pYthon) is a visualization library for comparing GLM (Generalized Linear Model) and GBM (Gradient Boosting Machine) models. It generates interactive plots showing SHAP values, model predictions, actuals, and GLM relativities to help interpret and compare model behavior.

## Build and Development Commands

```bash
# Install the package in development mode
pip install -e .

# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=mintypython

# Format code
black mintypython/
```

## Architecture

### Core Components

**mintypython class** (`mintypython/Mintypython.py`):
- Main entry point - orchestrates data preparation and plotting
- Accepts XGBoost or LightGBM models along with data, weights, and optional Emblem model exports
- Supports multiple link functions: poisson, gamma, tweedie, logistic, identity
- Key methods: `univariate_plot()`, `bivariate_plot()`, `compare()`

**Data_prep class** (`mintypython/Data_prep.py`):
- Handles all data transformation for plotting
- Computes SHAP values via the `shap` library (lazy evaluation with caching to `shap_df`)
- Creates variable bandings for continuous features
- Prepares GLM indication data from Emblem model exports via `scorepyon`
- Key methods: `prep_univariate_data()`, `prep_bivariate_data()`, `prep_shap_values()`, `create_bandings()`

### Plot Engine Pattern

Plot engines implement `Plot_interface` (abstract base class) and provide rendering:
- **Bokeh_plot**: Default engine, produces interactive HTML plots with tooltips and legends
- **Matplotlib_plot**: Alternative static plot engine (univariate only)

Engines are instantiated per-plot and receive prepared data from `Data_prep`.

### External Dependencies

- **scorepyon**: Scores GLM models and extracts factor relativities from Emblem exports
- **bidipy**: Reads `.fac` files for variable label mappings
- **shap**: Computes SHAP values for tree-based models

## Key Patterns

- Link functions transform model outputs (stored as `link_fn` and `_scorepyon_link_fn`)
- The `config` dict on mintypython controls plot aesthetics (colors, labels, line widths)
- Variable mappings between GBM and GLM names handled via `emb_gbm_map` parameter
- SHAP values can be combined across variables using `joinshaps` parameter
