"""Collaborative shared variable registry.

`registry.json` lives on a dedicated ref (the orphan branch `variables-registry`)
and is the global, machine-owned source of truth for variable names and their
governance state. This package provides:

- `validate`: a pure-python validator (no network) used by CI and a pre-commit hook
  to enforce the global primary-key (unique names) and governance invariants.
- `sync`: local read/write of the registry file plus a gated push to the registry
  ref. Outward-facing pushes are off by default (offline no-op) and only run when
  WHITEBOX_REGISTRY_SYNC=1 and a remote is configured.
"""
from .validate import validate_registry, validate_registry_file
from .sync import (
    REGISTRY_FILE,
    REGISTRY_REF,
    export_record,
    load_registry,
    push_variable,
    save_registry,
    sync_enabled,
    write_variable,
)

__all__ = [
    "validate_registry",
    "validate_registry_file",
    "REGISTRY_FILE",
    "REGISTRY_REF",
    "export_record",
    "load_registry",
    "save_registry",
    "sync_enabled",
    "write_variable",
    "push_variable",
]
