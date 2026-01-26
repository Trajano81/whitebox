# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Whitebox is a model-agnostic visualization library for comparing GLM (Generalized Linear Model) and GBM (Gradient Boosting Machine) models. It generates interactive plots showing SHAP values, model predictions, actuals, and GLM relativities to help interpret and compare model behavior.

## Build and Development Commands

```bash
# Install with Poetry (recommended)
poetry install

# Or install in development mode with pip
pip install -e .

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=whitebox

# Format code
poetry run black whitebox/
```

## Architecture

### Core Components

**Whitebox class** (`whitebox/core.py`):
- Main entry point - orchestrates data preparation and plotting
- Accepts XGBoost or LightGBM models along with data, weights, and optional Emblem model exports
- Supports multiple link functions: poisson, gamma, tweedie, logistic, identity
- Key methods: `univariate_plot()`, `bivariate_plot()`, `compare()`

**DataPrep class** (`whitebox/data_prep.py`):
- Handles all data transformation for plotting
- Computes SHAP values via the `shap` library (lazy evaluation with caching to `shap_df`)
- Creates variable bandings for continuous features
- Prepares GLM indication data from Emblem model exports via `scorepyon`
- Key methods: `prep_univariate_data()`, `prep_bivariate_data()`, `prep_shap_values()`, `create_bandings()`

### Plot Engine Pattern

Plot engines implement `PlotEngine` (abstract base class in `whitebox/engines/base.py`) and provide rendering:
- **BokehEngine**: Default engine, produces interactive HTML plots with tooltips and legends
- **MatplotlibEngine**: Alternative static plot engine (univariate only)

Engines are instantiated per-plot and receive prepared data from `DataPrep`.

### External Dependencies

- **scorepyon**: Scores GLM models and extracts factor relativities from Emblem exports
- **bidipy**: Reads `.fac` files for variable label mappings
- **shap**: Computes SHAP values for tree-based models

## Key Patterns

- Link functions transform model outputs (stored as `link_fn` and `_scorepyon_link_fn`)
- The `config` dict on Whitebox controls plot aesthetics (colors, labels, line widths)
- Variable mappings between GBM and GLM names handled via `emb_gbm_map` parameter
- SHAP values can be combined across variables using `joinshaps` parameter
- Categorical variables use `_encoded` suffix convention with `category_mappings` for display

## Git Commit Guidelines

- Do NOT include the Claude co-author signature in commits (no `Co-Authored-By: Claude` line)
