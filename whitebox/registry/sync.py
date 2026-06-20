"""Local read/write of the shared registry plus a GATED push to the registry ref.

The push side is OFF by default: it is a no-op unless WHITEBOX_REGISTRY_SYNC=1 is
set (and a remote is configured). This keeps outward-facing actions opt-in. The
local write/validate side always works offline so the namespace can be maintained
and tested without a network.
"""
import json
import os
import subprocess

from .validate import validate_registry

REGISTRY_FILE = "registry.json"
REGISTRY_REF = "variables-registry"

# Fields copied from an Encoder registry record into the shared registry.json.
_EXPORT_FIELDS = (
    "origin", "created_by", "review_status", "dtype", "data_quality",
    "kind", "sources", "level_map", "sep",
)


def sync_enabled():
    """True only when the modeler has explicitly opted into remote sync."""
    return os.environ.get("WHITEBOX_REGISTRY_SYNC", "") == "1"


def export_record(name, rec, created_at=None):
    """Serialize an Encoder registry entry into a registry.json record."""
    out = {"name": name}
    for field in _EXPORT_FIELDS:
        if field in rec:
            out[field] = rec[field]
    if created_at is not None:
        out["created_at"] = created_at
    return out


def load_registry(path=REGISTRY_FILE):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data["variables"] if isinstance(data, dict) else data


def save_registry(path, records):
    with open(path, "w") as f:
        json.dump({"variables": records}, f, indent=2, sort_keys=True)
        f.write("\n")


def _upsert(records, record):
    out = [r for r in records if r.get("name") != record["name"]]
    out.append(record)
    out.sort(key=lambda r: r.get("name", ""))
    return out


def write_variable(name, rec, path=REGISTRY_FILE, created_at=None):
    """Write/update a variable record in the local registry.json, validating the
    whole file (global uniqueness) before saving. No git, always offline-safe."""
    records = load_registry(path)
    record = export_record(name, rec, created_at=created_at)
    records = _upsert(records, record)
    errors = validate_registry(records)
    if errors:
        raise ValueError("registry validation failed: " + "; ".join(errors))
    save_registry(path, records)
    return record


def _has_remote(repo_dir):
    try:
        out = subprocess.run(
            ["git", "remote"], cwd=repo_dir, capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def push_variable(name, rec, path=REGISTRY_FILE, repo_dir=".", created_at=None):
    """Full sync: write locally then commit+push to the registry ref.

    GATED: returns {"pushed": False, "reason": ...} (an offline no-op) unless
    WHITEBOX_REGISTRY_SYNC=1 and a git remote is configured. This is the only path
    that performs an outward-facing action, and it stays opt-in.
    """
    record = write_variable(name, rec, path=path, created_at=created_at)
    if not sync_enabled():
        return {"pushed": False, "reason": "sync disabled (set WHITEBOX_REGISTRY_SYNC=1)", "record": record}
    if not _has_remote(repo_dir):
        return {"pushed": False, "reason": "no git remote configured", "record": record}
    # Commit the registry change on the current ref. Pushing to the dedicated
    # registry ref and the CI fan-out are handled by the workflow / a deliberate
    # operator action; we intentionally do not force-push here.
    subprocess.run(["git", "add", path], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore(variables): sync {name}"],
        cwd=repo_dir,
        check=True,
    )
    return {"pushed": True, "reason": "committed registry change", "record": record}
