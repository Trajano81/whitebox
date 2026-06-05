"""Pure-python registry validator (no network).

Enforces the global primary key (unique variable names) and the governance
invariant that a ``ready_to_model`` record carries the required review fields and
no unresolved data-quality flags. Used by CI (registry-check.yml) and the
pre-commit hook, so it must not import heavy/optional dependencies.
"""
import json
import sys

REQUIRED_FIELDS = ("name", "origin", "created_by", "review_status", "dtype")
VALID_STATUSES = ("pending_review", "ready_to_model", "needs_cleaning", "excluded")


def validate_registry(records):
    """Validate a list of registry records. Returns a list of error strings
    (empty list means valid)."""
    errors = []
    seen = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"record {i} is not an object")
            continue
        name = rec.get("name")
        if not name:
            errors.append(f"record {i} is missing 'name'")
            continue
        if name in seen:
            errors.append(
                f"duplicate variable name '{name}' (records {seen[name]} and {i}); "
                "the variable name is a global primary key"
            )
        seen[name] = i
        for field in REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"variable '{name}' is missing required field '{field}'")
        status = rec.get("review_status")
        if status is not None and status not in VALID_STATUSES:
            errors.append(
                f"variable '{name}' has invalid review_status '{status}'; "
                f"valid: {VALID_STATUSES}"
            )
        if status == "ready_to_model":
            flags = (rec.get("data_quality") or {}).get("flags") or []
            if flags:
                errors.append(
                    f"variable '{name}' is ready_to_model but has unresolved "
                    f"data_quality flags {flags}"
                )
    return errors


def validate_registry_file(path):
    """Validate a registry.json file. Returns a list of error strings."""
    with open(path) as f:
        data = json.load(f)
    records = data["variables"] if isinstance(data, dict) else data
    return validate_registry(records)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else "registry.json"
    try:
        errors = validate_registry_file(path)
    except FileNotFoundError:
        print(f"registry file not found (nothing to validate): {path}")
        return 0
    if errors:
        print(f"registry validation FAILED ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"registry OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
