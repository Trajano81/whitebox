# Spec 004: Collaborative shared variable registry

- **Status:** approved
- **Phase:** 4
- **Depends on:** spec-001, spec-002, spec-003

## Problem
With multiple modelers on separate branches, a new variable must propagate so everyone's
namespace stays globally unique and they can see each other's variables, regardless of which
branch they work on.

## Requirements
1. `registry.json` lives on a dedicated ref (orphan branch `variables-registry`), NOT merged
   through feature branches. It mirrors the unified registry record (spec-001) + `created_at`.
2. Machine-owned invariant: `registry.json` is written only by automation, never by hand,
   making branch updates always conflict-free.
3. A validator (`whitebox/registry/validate.py`) fails on duplicate `name` keys or a
   `ready_to_model` record missing required review fields. Wired as a pre-commit hook and a CI
   check (`.github/workflows/registry-check.yml`).
4. Local sync (`whitebox/registry/sync.py`): on the two trigger events (a modeler creates a
   variable; a new variable is detected from new raw data) it pulls the ref, re-validates the
   global name, writes the record, commits, and pushes to `variables-registry`
   (retry-on-conflict). Client-side auto-pull refreshes just `registry.json`.
5. Server-side fan-out (`.github/workflows/registry-propagate.yml`): triggers on a change to
   the registry ref via path filter `paths: ['registry.json']`, declares
   `permissions: contents: write`, uses the Actions-provided `GITHUB_TOKEN` (NO stored token),
   enumerates active branches, and writes/pushes `registry.json` into each branch's remote.
6. The fan-out and any remote push are gated behind a config flag; with no remote configured,
   only local registry read/write + validation run. Outward-facing pushes require explicit
   authorization (a `human-attention-needed/` file) before first use.

## Design
- `whitebox/registry/validate.py` (pure-python, no network) for use in CI + pre-commit.
- `whitebox/registry/sync.py` for the local pull/validate/write/push + auto-pull, behind a
  `WHITEBOX_REGISTRY_SYNC=1`-style flag; default off in tests.
- GitHub workflows under `.github/workflows/`.

## Acceptance criteria
- Validator catches duplicate names and bad `ready_to_model` records (unit test, no network).
- Registry round-trips: write a record -> read back -> namespace reflects it (local, no push).
- Sync module never pushes when the flag/remote is absent (test asserts it is a no-op offline).

## Test plan
- `tests/unit/test_registry_validate.py`, `tests/unit/test_registry_sync_offline.py`.

## Out of scope
Actually executing remote pushes / GitHub Actions in CI here (needs a live remote; gated and
deferred to a human-authorized step).
