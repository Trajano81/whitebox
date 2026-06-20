# Spec 006: MCP server `whitebox_mcp`

- **Status:** approved
- **Phase:** 6
- **Depends on:** spec-001, spec-002, spec-003

## Problem
Expose whitebox capabilities as well-designed MCP tools so agents can drive it, following the
`mcp-builder` skill best practices.

## Requirements
1. Python FastMCP over stdio. `whitebox/mcp/server.py`, server name `whitebox_mcp`.
2. Tools (snake_case, `whitebox_` prefix, Pydantic inputs, annotations, actionable errors,
   json|markdown response_format):
   - `whitebox_load_session`, `whitebox_list_variables` (paginated), `whitebox_describe_variable`,
     `whitebox_get_category_mapping`.
   - `whitebox_add_group`, `whitebox_add_combination`, `whitebox_set_variable_status`,
     `whitebox_remove_variable`.
   - Governance: `whitebox_ingest_data`, `whitebox_profile_variable`, `whitebox_propose_cleaning`,
     `whitebox_apply_cleaning`.
   - Plots: `whitebox_univariate_plot`, `whitebox_bivariate_plot` (return saved HTML path +
     summary stats), `whitebox_launch_report`.
   - `whitebox_sync_variables` (pull latest shared registry; readOnlyHint).
3. Cross-cutting: Pydantic schemas with constraints + examples; structuredContent where useful;
   errors in-result with next-step suggestions; sanitize `data_path`/`model_path` (no traversal);
   never log to stdout (stderr only).
4. `.mcp.json` project registration (command = `python -m whitebox.mcp.server`).
5. `whitebox/mcp/evaluation.xml`: 10 independent, read-only, verifiable Q/A pairs over the demo.

## Acceptance criteria
- `python -m py_compile whitebox/mcp/server.py` passes.
- Each tool function returns valid structured output and actionable errors on bad input
  (unit test calling the tool fns directly).
- e2e MCP session: load -> list -> add_group -> univariate_plot asserts saved HTML + summary.

## Test plan
- `tests/unit/test_mcp_tools.py`, `tests/e2e/test_mcp_e2e.py`.

## Out of scope
The dev guidance skill (spec-007).
