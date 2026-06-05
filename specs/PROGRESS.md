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
- [ ] add_group / add_combination on Encoder + Whitebox
- [ ] data_prep _derived_shap_series + plot wiring
- [ ] tests/unit/test_derived_group.py, test_derived_combine.py
- [ ] tests green + committed

## Phase 4 - collaborative registry (spec-004)
- [ ] registry.json schema + validator + pre-commit hook
- [ ] sync (pull/validate/push to variables-registry ref) - local side
- [ ] .github/workflows/registry-check.yml + registry-propagate.yml (path-filtered, token-free)
- [ ] tests green + committed

## Phase 5 - Streamlit report (spec-005)
- [ ] headless engine (show=False guards)
- [ ] whitebox/report/launcher.py + app.py (one-way, two-way, data-review, variable-manager)
- [ ] tests/unit/test_engine_headless.py + tests/e2e/test_report_e2e.py
- [ ] tests green + committed

## Phase 6 - MCP server (spec-006)
- [ ] whitebox/mcp/server.py (tools incl governance/ingest) + .mcp.json
- [ ] whitebox/mcp/evaluation.xml (10 Q/A)
- [ ] tests/unit/test_mcp_tools.py + tests/e2e/test_mcp_e2e.py
- [ ] tests green + committed

## Phase 7 - dev guidance skill (spec-007)
- [ ] .claude/skills/whitebox-dev/SKILL.md (workflow + model-tiering guidance)
- [ ] committed

## Notes / blockers
- (none yet)
