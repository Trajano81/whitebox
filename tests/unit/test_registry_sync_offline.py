"""Registry sync is offline-safe and never pushes without explicit opt-in (spec-004)."""
import os

from whitebox.registry import (
    export_record,
    load_registry,
    push_variable,
    sync_enabled,
    write_variable,
)


def _rec():
    return {
        "origin": "derived_group",
        "created_by": "user",
        "review_status": "pending_review",
        "dtype": "categorical",
        "kind": "group",
        "sources": ["region"],
        "level_map": {"North": "NS"},
    }


def test_export_record_carries_governance_fields():
    out = export_record("region_grp", _rec(), created_at="2026-06-05")
    assert out["name"] == "region_grp"
    assert out["origin"] == "derived_group"
    assert out["sources"] == ["region"]
    assert out["created_at"] == "2026-06-05"


def test_write_variable_roundtrips(tmp_path):
    path = str(tmp_path / "registry.json")
    write_variable("region_grp", _rec(), path=path)
    records = load_registry(path)
    assert [r["name"] for r in records] == ["region_grp"]
    # Upsert replaces, never duplicates.
    write_variable("region_grp", _rec(), path=path)
    records = load_registry(path)
    assert len(records) == 1


def test_push_is_noop_when_sync_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("WHITEBOX_REGISTRY_SYNC", raising=False)
    assert sync_enabled() is False
    path = str(tmp_path / "registry.json")
    result = push_variable("region_grp", _rec(), path=path, repo_dir=str(tmp_path))
    assert result["pushed"] is False
    # The local registry was still written (validate + save), just not pushed.
    assert load_registry(path)[0]["name"] == "region_grp"


def test_whitebox_sync_variable_offline(wb, tmp_path):
    wb.add_group("region_grp", "region", {"North": "NS", "South": "NS"}, default="Other")
    result = wb.sync_variable(
        "region_grp",
        registry_path=str(tmp_path / "registry.json"),
        repo_dir=str(tmp_path),
    )
    assert result["pushed"] is False
    assert result["record"]["name"] == "region_grp"
