# Spec 003: Derived variables (group + combine) with synthesized SHAP

- **Status:** approved
- **Phase:** 3
- **Depends on:** spec-001, spec-002

## Problem
Users need to derive new variables from existing ones (group a categorical's levels;
combine 2+ variables) inside the package: internally encoded, directly plottable, and with
an exposed `{name}_encoded` column for manual retrain. The current model was not trained on
them, so their SHAP is synthesized from source-feature SHAP.

## Requirements
1. `Encoder.add_group(name, source, level_map, default=None)`: creates `data[name]` (human
   labels via `data[source].map(level_map)`, uncovered -> `default` or passthrough with a
   verbose warning) and `data[name+'_encoded']`; registers `derived_group` with
   `sources=[source]`, status `pending_review`.
2. `Encoder.add_combination(name, sources, sep="_x_")`: combined human label joins source
   labels with `_x_` (e.g. `North_x_SUV`); creates `name` + `name_encoded`; registers
   `derived_combine` with `sources`, status `pending_review`. Warn if combined cardinality
   > ~50.
3. `Encoder.remove_derived(name)`: drops `{name}`/`{name}_encoded` columns, the registry
   entry, and `category_mappings[name]`; frees the primary-key name.
4. Derived variables are plottable by name in univariate and bivariate plots.
5. SHAP synthesis: single source -> source feature's SHAP; multiple sources -> row-wise sum
   of source SHAPs. Grouping along new levels falls out of the existing `groupby("x_axis")`.
6. `Whitebox.add_group`, `Whitebox.add_combination`, `Whitebox.plottable_variables()`.

## Design
- `whitebox/encoding.py`: `add_group`, `add_combination`, `remove_derived` (name validation
  from spec-001).
- `whitebox/data_prep.py`:
  - `_derived_shap_series(name)`: single -> `shap_df[source]`; multi -> `shap_df[sources].sum(1)`.
  - `prep_univariate_data`: validation -> `plottable_variables()`; SHAP block branches on
    `encoder.is_derived(var_name)` -> `_derived_shap_series`.
  - `prep_bivariate_data`: relax two-var validation; SHAP assignment uses
    `_derived_shap_series(var1)` when derived. GLM toggle degrades gracefully.

## Acceptance criteria
- `add_group`/`add_combination` create the columns + registry entry + `category_mappings[name]`.
- `univariate_plot(name, shap=True)` returns a figure; aggregated SHAP equals a manual
  `groupby` (group) or sum of source SHAPs (combine).
- Uncovered-level `default` behavior; cardinality warning on combine.
- `remove_derived` drops columns + mapping + frees the name.
- Name primary-key uniqueness + slug rule enforced (collision/invalid -> ValueError).

## Test plan
- `tests/unit/test_derived_group.py`, `tests/unit/test_derived_combine.py`,
  and uniqueness/lifecycle cases in `tests/unit/test_variable_manager.py`.

## Out of scope
UI creation flow (spec-005), cross-modeler propagation (spec-004).
