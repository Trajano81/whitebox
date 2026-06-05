# Build progress tracker

The autonomous (ralph-loop) build reads this each iteration to know what is done
and what is next. Update the checkboxes and notes as work completes. Source of
truth for the design is the approved plan at
`/Users/kmiloaparicio/.claude/plans/reflective-growing-shell.md`.

Legend: [ ] todo, [~] in progress, [x] done, [!] blocked (see human-attention-needed/).

## Phase 0 - scaffolding
- [x] specs/ + TEMPLATE + PROGRESS
- [x] tests/ layout (unit/, e2e/) + pytest markers
- [x] pyproject deps (report/mcp extras, pytest-playwright) + Bokeh re-pin <3.5
- [x] .venv_agent/.env.example
- [x] human-attention-needed/ (+ done/) + README
- [x] .gitignore updates
- [x] spec stubs 001-007 written
- [x] Phase 0 committed (a4b094c)

## Phase 1 - internal encoding (spec-001)
- [x] whitebox/encoding.py (Encoder + unified registry)
- [x] core.py wiring (_resolve_feature_names, encoder, auto_encode before model vars, __getstate__/__setstate__ incl x_data)
- [x] data_prep.py uses encoder.model_input_column (prep_shap_values + process_categoricals shim)
- [x] demo/synthetic_data.py de-externalized (deprecate encode helpers, get_demo_data_and_model returns raw)
- [x] tests/unit/test_encoder.py, test_backcompat.py, test_pickle.py
- [x] tests green + committed

## Phase 2 - variable governance (spec-002)
- [x] whitebox/governance/ (ingest diff, profiler, proposals by_bins/by_magnitude, status gate)
- [x] Encoder.set_status / set_profile; Whitebox profile/propose_cleaning/apply_cleaning/set_status/ingest
- [x] tests/unit/test_governance.py, test_variable_manager.py
- [x] tests green (24 unit) + committed

## Phase 3 - derived variables (spec-003)
- [x] add_group / add_combination / remove_derived on Encoder + Whitebox; plottable_variables
- [x] data_prep _derived_shap_series + univariate/bivariate validation + SHAP wiring
- [x] tests/unit/test_derived_group.py, test_derived_combine.py
- [x] tests green (33 unit) + committed

## Phase 4 - collaborative registry (spec-004)
- [x] whitebox/registry/validate.py (pure-python; dup names + ready_to_model invariants) + CLI
- [x] whitebox/registry/sync.py (write/validate local; push GATED offline no-op) + Whitebox.sync_variable
- [x] .github/workflows/registry-check.yml + registry-propagate.yml (path-filtered, token-free GITHUB_TOKEN)
- [x] .pre-commit-config.yaml runs the validator
- [x] tests/unit/test_registry_validate.py, test_registry_sync_offline.py (42 unit) + committed
- Note: registry.json itself NOT committed to feature branches (lives on variables-registry ref); the
  ref + remote fan-out are a gated, human-authorized step (see human-attention-needed when ready to enable).

## Phase 5 - Streamlit report (spec-005)
- [x] headless engine (show kwarg threaded; guarded show(p) x3; guarded output_notebook)
- [x] whitebox/report/launcher.py + app.py (one-way, two-way, data-review, variable-manager)
- [x] Whitebox.launch_report
- [x] tests/unit/test_engine_headless.py, test_report_launcher.py + tests/e2e/test_report_e2e.py (skips w/o streamlit)
- [x] tests green (46 unit, 1 skip) + committed

## Phase 6 - MCP server (spec-006)
- [x] whitebox/mcp/tools.py (pure logic, testable) + server.py (FastMCP stdio wiring) + .mcp.json
- [x] whitebox/mcp/evaluation.xml (read-only Q/A over the demo)
- [x] tests/unit/test_mcp_tools.py + tests/e2e/test_mcp_e2e.py (skips without mcp)
- [x] server py_compile OK; tests green (54 unit, 3 skip) + committed

## Phase 7 - dev guidance skill (spec-007)
- [x] .claude/skills/whitebox-dev/SKILL.md (spec-driven flow + model-tiering + MCP tools + governance rules)
- [x] committed

## Notes / blockers
- All phases 0-7 implemented; `python -m pytest -m "not e2e"` is green (54 unit, 3 skipped: streamlit/mcp
  extras not installed locally; e2e tests skip without them).
- Outward-facing steps deferred for human authorization: see
  human-attention-needed/001-enable-remote-registry-and-prs.md (push branch / open PRs, create the
  variables-registry ref, enable CI fan-out, turn on live sync). All are gated and offline-safe by default.
