---
name: whitebox-dev
description: Spec-driven workflow for developing the whitebox library (internal encoding, variable governance, derived variables, collaborative registry, Streamlit report, MCP server). Use when implementing, reviewing, or extending whitebox features. Documents the model-tiering guidance, the whitebox_mcp tools, the governance rules, and the commit/PR conventions.
---

# Whitebox development workflow

Whitebox is a model-agnostic visualization library (GLM vs GBM, SHAP, predictions,
GLM relativities). It now owns categorical encoding internally, governs variables
through a review gate, supports derived variables, a Streamlit report, an MCP
server, and a shared variable registry.

## Spec-driven flow

1. Write or update a spec under `specs/spec-NNN-*.md` (use `specs/TEMPLATE.md`)
   BEFORE implementing. Get it approved.
2. Implement against the spec. Match the surrounding code style.
3. Add unit tests under `tests/unit/` (and `tests/e2e/` for app/MCP smoke, marked
   `@pytest.mark.e2e`). Run `poetry run pytest -m "not e2e"`.
4. Commit per spec with Conventional Commits (`feat`/`refactor`/`test`/`docs`/`chore`
   + scope, e.g. `feat(encoding): ...`). Reference the spec id.
   - Project rule: NO `Co-Authored-By: Claude` line in commits.
   - Writing-style rule: NO em-dashes in commits, comments, or docs.
5. One PR per spec. PR body references the spec and acceptance criteria.
6. Keep `specs/PROGRESS.md` checkboxes current.

## Model-tiering guidance (pick the tier per task)

- Opus: advanced design, tricky encoding/SHAP work, refactors, spec authoring,
  report/MCP architecture.
- Sonnet: operative implementation against an approved spec, wiring, routine tests.
- Haiku: conventional commits, changelog entries, trivial chores.

(Named tiered agent files are intentionally NOT shipped; they only cost tokens when
invoked and the structure above already steers any subagent. Add them later if the
workflow proves repetitive.)

## Core rules to preserve

- Encoding is internal and the only path: `whitebox/encoding.py` `Encoder` owns
  `category_mappings` and the unified registry (one record per variable, raw or
  derived; the name is the primary key). Use `encoder.model_input_column(col)` for
  model/SHAP inputs. Codes come from `pd.Categorical(...).cat.codes`.
- Variable name is a strict slug `^[A-Za-z][A-Za-z0-9_]*$`, never ending in
  `_encoded`, globally unique. Derived-variable creation validates this.
- Governance gate: every variable starts `pending_review`. Only an explicit,
  validated transition reaches `ready_to_model` (blocked while `needs_cleaning`).
  Only `ready_to_model` variables are in the trainable manifest.
- Derived variables (group/combine) are real, plottable, and carry an
  `{name}_encoded` column for manual retrain; their SHAP is synthesized from source
  features (single = source SHAP; multiple = sum).
- The shared `registry.json` is machine-owned (never hand-edited) and lives on the
  `variables-registry` ref, not feature branches. Remote pushes are gated
  (`WHITEBOX_REGISTRY_SYNC=1` + a remote); the CI fan-out uses the Actions
  `GITHUB_TOKEN` (no stored token).

## whitebox_mcp tools (for agent-driven development)

Server: `python -m whitebox.mcp.server` (registered in `.mcp.json`). Tools:
`whitebox_load_session`, `whitebox_list_variables`, `whitebox_describe_variable`,
`whitebox_get_category_mapping`, `whitebox_add_group`, `whitebox_add_combination`,
`whitebox_set_variable_status`, `whitebox_remove_variable`, `whitebox_ingest_data`,
`whitebox_profile_variable`, `whitebox_propose_cleaning`, `whitebox_apply_cleaning`,
`whitebox_sync_variables`, `whitebox_univariate_plot`, `whitebox_bivariate_plot`.

## Optional extras

- Report: `pip install 'whitebox[report]'` (Streamlit). Launch via `wb.launch_report()`.
- MCP: `pip install 'whitebox[mcp]'`.

## Async decisions

When a genuine decision/credential/external dependency blocks work, drop a numbered
file in `human-attention-needed/` with the question, options, and recommended
default, and continue other unblocked work.
