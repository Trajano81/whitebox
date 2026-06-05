# Spec 005: Streamlit localhost report

- **Status:** approved
- **Phase:** 5
- **Depends on:** spec-001, spec-002, spec-003

## Problem
No interactive way to browse variables, review/govern them, create derived variables, or save
visual summaries. Build a Streamlit app launched from the instance.

## Requirements
1. Headless rendering: thread `show: bool = True` into the Bokeh engine; guard
   `if show: show(p)` at the three `show(p)` sites; guard module-level `output_notebook()` in
   try/except. Streamlit passes `show=False`.
2. `whitebox/report/launcher.py` `launch_report(port, export_dir)`: pre-compute `shap_df`, set
   `explainer=None`, pickle `wb` to a temp path, `subprocess.Popen([... streamlit run app.py
   -- <pkl> <export_dir>])`. Clear `ImportError("pip install whitebox[report]")` if streamlit
   missing.
3. `whitebox/report/app.py` pages:
   - One-way: `selectbox(plottable_variables())` + toggles -> `univariate_plot(..., show=False)`.
   - Two-way: two selectboxes -> `bivariate_plot(..., show=False)`.
   - Data review / governance: ingestion revision report; categorized table by status/origin;
     per-variable profile + editable proposal (categorical level_map via `st.data_editor`;
     numeric banding method `by_bins`/`by_magnitude`, bins default 10); status buttons
     (`ready_to_model`/`needs_cleaning`/`excluded`), approval blocked while needs_cleaning.
   - Variable manager: group + combine create `pending_review`; table with recheck/status/delete;
     name validated (slug + uniqueness) inline before submit.
   - Embedding/export: `bokeh.embed.file_html(fig, CDN, title)` -> `st.components.v1.html(...)`;
     "Save summary" writes HTML into `export_dir`; export a manifest of `ready_to_model` vars.

## Acceptance criteria
- `univariate_plot(..., show=False)` returns a figure without calling `show()` (unit test, mock).
- e2e: launch the app, render One-way + Two-way, create a group + a combination, assert HTML
  files appear in `export_dir`.

## Test plan
- `tests/unit/test_engine_headless.py`, `tests/e2e/test_report_e2e.py` (webapp-testing/Playwright).

## Out of scope
MCP (spec-006). Real cross-modeler sync execution (spec-004).
