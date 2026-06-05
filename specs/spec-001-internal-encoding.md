# Spec 001: Internal encoding + unified variable registry

- **Status:** approved
- **Phase:** 1
- **Depends on:** none

## Problem
Encoding lives outside the package today (the demo's `prepare_data_for_whitebox` /
`encode_categoricals` build `{col}_encoded` columns and `category_mappings` before
`Whitebox` is constructed). Modifying or deriving variables forces hand-built encoded
columns. We want the package to own encoding internally, as the only path.

## Requirements
1. A new `whitebox/encoding.py` with an `Encoder` class that owns categorical encoding
   and a unified variable registry (one record per variable, raw or derived; name is the
   primary key).
2. `Encoder.auto_encode()` detects object/category/string features in `feature_names` and
   creates `{col}_encoded` via `pd.Categorical(series).cat.codes`, setting
   `category_mappings[col] = dict(enumerate(cat.categories))`. If `{col}_encoded` already
   exists, reconstruct the mapping from the pair instead of overwriting. NaN -> code -1.
3. Each raw feature is registered with `origin="raw"`, `review_status="pending_review"`
   (never auto `ready_to_model`), and `dtype` in {categorical, numeric}.
4. The variable name is a strict slug `^[A-Za-z][A-Za-z0-9_]*$`, must not end in
   `_encoded`, and must be unique across feature_names + existing columns + registry +
   `{name}_encoded`. Violations raise `ValueError` with an actionable message.
5. `Encoder.model_input_column(col)` returns `{col}_encoded` if present else `{col}`,
   centralizing the logic duplicated in core.py and data_prep.py.
6. `Whitebox` exposes `self.category_mappings` as the SAME dict object the encoder owns;
   backward compatibility preserved for callers passing `category_mappings=` or pre-built
   `{col}_encoded` columns.
7. `Whitebox` is picklable: `__getstate__`/`__setstate__` drop non-picklable `link_fn`,
   `_ci_fn`, `explainer` and rebuild link functions from `link_fn_str` on load.

## Design
- `whitebox/encoding.py`:
  - `class Encoder`: holds `data` (ref to Whitebox.data), `feature_names`,
    `category_mappings` (owned dict), `registry` (dict keyed by name).
  - Methods: `auto_encode()`, `encode_column(col, mapping=None)`, `model_input_column(col)`,
    `validate_name(name)`, `is_derived(name)`, `source_features(name)`, `decode(col, codes)`,
    `list_variables(status=None, origin=None)`, `trainable_variables()`. (Derived-var
    methods `add_group`/`add_combination`/`set_status`/`remove_derived` land in spec-002/003
    but the registry shape and slug/uniqueness validation live here.)
- `whitebox/core.py`:
  - Add `_resolve_feature_names()` (read names off the xgb/lgb model without building the
    DMatrix). Build `self.encoder` and call `auto_encode()` BEFORE
    `_create_model_specific_variables()` (currently line 173) so encoded columns exist for
    the DMatrix in `_get_xgb_params` (~660, switch to `encoder.model_input_column`).
  - Refactor link-fn setup (lines 179-206) into `_build_link_functions()`; call from
    `__init__` and `__setstate__`.
  - Replace `self.DataPrep.process_categoricals()` (line 211) with a deprecated shim calling
    `self.encoder.auto_encode()`.
  - `fac_mapping` (214-217): `mapping_dict if mapping_dict is not None else encoder.category_mappings`.
  - Alias `self.category_mappings = self.encoder.category_mappings`.
- `whitebox/data_prep.py`: `prep_shap_values` builds SHAP input via
  `encoder.model_input_column(col)`.

## Acceptance criteria
- `auto_encode` produces codes equal to `pd.Categorical(s).cat.codes` (test_encoder).
- Reconstruction path when `{col}_encoded` pre-exists (test_encoder).
- NaN -> -1 (test_encoder).
- Slug + uniqueness validation raises on bad/colliding names (test covered with spec-003 too).
- Backward compat: passing `category_mappings` + `{col}_encoded` yields identical
  `prep_univariate_data` agg output (test_backcompat).
- `pickle.dumps/loads(wb)` round-trips and `link_fn` works after load (test_pickle).

## Test plan
- `tests/unit/test_encoder.py`, `tests/unit/test_backcompat.py`, `tests/unit/test_pickle.py`.

## Out of scope
Derived variables (spec-003), governance ingest/profiling (spec-002), report (spec-005).
