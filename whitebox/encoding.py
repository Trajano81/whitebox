"""Internal categorical encoding and the unified variable registry.

The Encoder is the single owner of categorical encoding for a Whitebox instance.
It auto-detects categorical features and creates `{col}_encoded` integer columns
internally (so callers no longer have to build them externally), owns the
`category_mappings` dict, and holds a unified registry with one record per
variable (raw or derived). The variable name is the registry primary key.

Naming convention preserved from the rest of the package:
- original column keeps human-readable values (e.g. 'region')
- encoded column has the '_encoded' suffix (e.g. 'region_encoded')
- category_mappings maps {col: {code: label}}
"""
import re

import numpy as np
import pandas as pd

# Strict slug for variable names we generate/accept (derived variables). A name
# becomes a real column, its `{name}_encoded` column, a registry primary key,
# and part of export filenames, so it must be filesystem/identifier safe.
SLUG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Default separator joining source labels of a combined variable, e.g.
# "North_x_SUV". Space/special-char free so values survive CSV/parquet/filenames.
DEFAULT_COMBINE_SEP = "_x_"

VALID_STATUSES = ("pending_review", "ready_to_model", "needs_cleaning", "excluded")


class Encoder:
    """Owns categorical encoding + the unified variable registry for a Whitebox."""

    def __init__(self, data, feature_names, category_mappings=None, verbose=True):
        self.data = data
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.verbose = verbose
        # category_mappings is owned here and aliased by Whitebox.
        self.category_mappings = {}
        if category_mappings:
            self.category_mappings.update(category_mappings)
        # Unified registry: one record per variable (raw or derived), keyed by name.
        self.registry = {}

    # ------------------------------------------------------------------
    # dtype helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_categorical(series):
        """True for object / category / string dtype columns (non-numeric)."""
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_numeric_dtype(dtype):
            return False
        return dtype == object or pd.api.types.is_string_dtype(dtype)

    @staticmethod
    def _encode_series(series):
        """Encode a categorical series to int codes via pandas Categorical.

        Returns (codes_series, {code: label}). NaN maps to code -1 (pandas default).
        Codes are deterministic for a fixed set of categories, which keeps internal
        encoding aligned with training-time `pd.Categorical(...).cat.codes`.
        """
        cat = pd.Categorical(series)
        codes = pd.Series(np.asarray(cat.codes), index=series.index).astype("int64")
        mapping = {int(code): label for code, label in enumerate(cat.categories)}
        return codes, mapping

    # ------------------------------------------------------------------
    # registry helpers
    # ------------------------------------------------------------------
    def _register_raw(self, name, dtype):
        if name in self.registry:
            self.registry[name]["dtype"] = dtype
            return
        self.registry[name] = {
            "origin": "raw",
            "created_by": "data",
            "review_status": "pending_review",
            "dtype": dtype,
            "data_quality": {},
            "proposal": None,
        }

    def is_derived(self, name):
        rec = self.registry.get(name)
        return bool(rec) and rec["origin"] in ("derived_group", "derived_combine")

    def source_features(self, name):
        rec = self.registry.get(name)
        return list(rec.get("sources", [])) if rec else []

    def list_variables(self, status=None, origin=None):
        names = []
        for name, rec in self.registry.items():
            if status is not None and rec["review_status"] != status:
                continue
            if origin is not None and rec["origin"] != origin:
                continue
            names.append(name)
        return names

    def trainable_variables(self):
        """Names whose review_status is 'ready_to_model' (the retrain manifest)."""
        return self.list_variables(status="ready_to_model")

    # ------------------------------------------------------------------
    # name validation (primary key); used by derived-variable creation
    # ------------------------------------------------------------------
    def validate_name(self, name):
        """Validate a NEW derived variable name: strict slug, no reserved suffix,
        and globally unique across features / columns / registry / `{name}_encoded`.
        Raises ValueError with an actionable message on violation.
        """
        if not isinstance(name, str) or not SLUG_RE.match(name):
            raise ValueError(
                f"invalid variable name {name!r}: must match ^[A-Za-z][A-Za-z0-9_]*$ "
                "(start with a letter; letters, digits, underscores only; no spaces or special chars)"
            )
        if name.endswith("_encoded"):
            raise ValueError(
                f"invalid variable name {name!r}: must not end in '_encoded' (reserved suffix)"
            )
        reserved = set(self.feature_names) | set(self.data.columns) | set(self.registry)
        if name in reserved or f"{name}_encoded" in self.data.columns:
            raise ValueError(
                f"variable {name!r} already exists; choose a unique name or delete the existing one first"
            )
        return True

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def encode_column(self, col, mapping=None):
        """Encode a single column into `{col}_encoded`, registering its mapping.

        If `mapping` ({code: label}) is provided, the column is encoded via the
        inverse (label -> code). Otherwise codes are derived from the data.
        Returns the int-coded Series.
        """
        encoded_col = f"{col}_encoded"
        if mapping is not None:
            label_to_code = {label: code for code, label in mapping.items()}
            codes = self.data[col].map(label_to_code)
            self.category_mappings[col] = dict(mapping)
        else:
            codes, built = self._encode_series(self.data[col])
            self.category_mappings[col] = built
        self.data[encoded_col] = codes
        return self.data[encoded_col]

    def auto_encode(self):
        """Detect categorical features and create their `{col}_encoded` columns.

        Scenarios per feature column:
        - `{col}_encoded` already exists: reconstruct the mapping from the pair if
          missing (legacy/manual path), do not overwrite.
        - categorical WITH a provided mapping: encode using it (label -> code).
        - categorical WITHOUT a mapping: auto-encode via pandas Categorical codes.
        - numeric: register only, no encoding.

        Every raw feature is registered as `pending_review` (never auto-trainable).
        """
        processed = []
        for col in self.feature_names:
            if col not in self.data.columns:
                continue
            encoded_col = f"{col}_encoded"

            if encoded_col in self.data.columns:
                if col not in self.category_mappings:
                    pairs = self.data[[col, encoded_col]].dropna().drop_duplicates()
                    self.category_mappings[col] = {
                        int(code): label
                        for code, label in zip(pairs[encoded_col], pairs[col])
                    }
                self._register_raw(col, dtype="categorical")
                processed.append(col)
            elif self._is_categorical(self.data[col]):
                if col in self.category_mappings:
                    self.encode_column(col, mapping=self.category_mappings[col])
                else:
                    self.encode_column(col)
                self._register_raw(col, dtype="categorical")
                processed.append(col)
            else:
                self._register_raw(col, dtype="numeric")

        if self.verbose and processed:
            print(f"Encoded {len(processed)} categorical column(s): {processed}")
        return self.category_mappings

    # ------------------------------------------------------------------
    # lookups used by the model / plotting layers
    # ------------------------------------------------------------------
    def model_input_column(self, col):
        """Series to feed the model/SHAP for `col`: `{col}_encoded` if present else `{col}`."""
        encoded_col = f"{col}_encoded"
        if encoded_col in self.data.columns:
            return self.data[encoded_col]
        return self.data[col]

    def decode(self, col, codes):
        """Map integer codes back to labels for `col` (for axis labels)."""
        mapping = self.category_mappings.get(col, {})
        if hasattr(codes, "map"):
            return codes.map(lambda c: mapping.get(c, c))
        return [mapping.get(c, c) for c in codes]
