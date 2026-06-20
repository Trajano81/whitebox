"""Variable governance: ingestion diff, profiling, editable cleaning/banding
proposals, and the validation-gate statuses.

Nothing is trainable automatically: every variable (raw or derived) starts as
``pending_review`` and only becomes ``ready_to_model`` via an explicit, validated
status transition after review.
"""
from .profiler import (
    DEFAULT_MISSING_TOKENS,
    normalize_missing,
    profile_variable,
)
from .proposals import (
    MISSING_LABEL,
    OTHER_LABEL,
    apply_cleaning,
    propose_cleaning,
)
from .ingest import diff_schema, schema_snapshot

__all__ = [
    "DEFAULT_MISSING_TOKENS",
    "normalize_missing",
    "profile_variable",
    "propose_cleaning",
    "apply_cleaning",
    "MISSING_LABEL",
    "OTHER_LABEL",
    "schema_snapshot",
    "diff_schema",
]
