"""Registry validator: global primary key + governance invariants (spec-004)."""
from whitebox.registry import validate_registry


def _rec(name, **over):
    base = {
        "name": name,
        "origin": "raw",
        "created_by": "data",
        "review_status": "pending_review",
        "dtype": "categorical",
    }
    base.update(over)
    return base


def test_valid_registry_has_no_errors():
    records = [_rec("region"), _rec("age", dtype="numeric")]
    assert validate_registry(records) == []


def test_duplicate_names_rejected():
    records = [_rec("region"), _rec("region")]
    errors = validate_registry(records)
    assert any("duplicate variable name 'region'" in e for e in errors)


def test_missing_required_field_rejected():
    bad = _rec("region")
    del bad["dtype"]
    errors = validate_registry([bad])
    assert any("missing required field 'dtype'" in e for e in errors)


def test_invalid_status_rejected():
    errors = validate_registry([_rec("region", review_status="bogus")])
    assert any("invalid review_status" in e for e in errors)


def test_ready_to_model_with_flags_rejected():
    rec = _rec(
        "region",
        review_status="ready_to_model",
        data_quality={"flags": ["high_missingness"]},
    )
    errors = validate_registry([rec])
    assert any("ready_to_model but has unresolved" in e for e in errors)
