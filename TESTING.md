# How to test whitebox

This guide covers testing the feature build on `feature/internal-encoding-governance`:
internal encoding, variable governance, derived variables, the collaborative
registry, the Streamlit report, and the MCP server.

All commands assume you are at the repo root:

```bash
cd "/Users/kmiloaparicio/Documents/Projects/Github Repos/whitebox"
```

## 1. Environment setup

The repo uses a local virtual environment at `.venv`. Activate it first:

```bash
source .venv/bin/activate
```

The core library installs by default, but the report and MCP features are
optional extras. Streamlit and mcp are NOT installed by default, so the tests
that need them will skip (and the report/server cannot launch) until you install
the extras.

Install everything needed for full testing:

```bash
# core library, editable
pip install -e .

# Streamlit report extra (enables launch_report and the report e2e test)
pip install -e '.[report]'

# MCP server extra (enables the server and the MCP e2e test)
pip install -e '.[mcp]'

# dev extras (pytest, pytest-cov, pytest-playwright, black, ruff, jupyter)
pip install -e '.[dev]'
```

You can combine extras in one call:

```bash
pip install -e '.[report,mcp,dev]'
```

Notes:
- `pytest-playwright` ships with the `dev` extra. The current e2e tests do not
  drive a browser, but installing it keeps the dev environment complete.
- If you only run unit tests, you do not need the `report` or `mcp` extras; the
  affected tests skip cleanly.

## 2. Unit tests

Run the unit suite (everything except the `e2e` marker):

```bash
source .venv/bin/activate && python -m pytest -m "not e2e" -q
```

Expected result: about 54 tests passing. A few tests skip when the `report`
(streamlit) and `mcp` extras are not installed (roughly 3 skips in a bare
environment). With both extras installed, those skips go away.

The unit tests live under `tests/unit/` and cover each phase:

- `test_encoder.py`, `test_backcompat.py`, `test_pickle.py` (internal encoding)
- `test_governance.py`, `test_variable_manager.py` (variable governance)
- `test_derived_group.py`, `test_derived_combine.py` (derived variables)
- `test_registry_validate.py`, `test_registry_sync_offline.py` (collaborative registry)
- `test_engine_headless.py`, `test_report_launcher.py` (headless engine + report)
- `test_mcp_tools.py` (MCP tool logic)
- `test_smoke.py` (end-to-end build sanity)

Shared fixtures (`messy_df`, `clean_df`, `trained_model`, `wb`, `wb_clean`) live
in `tests/conftest.py`.

Coverage report:

```bash
poetry run pytest --cov=whitebox
```

(or `python -m pytest --cov=whitebox` inside the activated `.venv`).

## 3. End-to-end tests

Run only the e2e tests:

```bash
source .venv/bin/activate && python -m pytest -m e2e
```

These live in `tests/e2e/` and are gated:

- `test_report_e2e.py` skips unless `streamlit` is installed. With it installed,
  `test_report_serves` launches the Streamlit app as a subprocess on a free port
  and verifies the report serves.
- `test_mcp_e2e.py` skips unless the `mcp` package is installed. With it
  installed, `test_server_registers_tools` imports the FastMCP server and
  confirms the `whitebox_*` tools are registered.

So with no extras installed, `python -m pytest -m e2e` reports all e2e tests as
skipped. Install the relevant extra (see section 1) to actually exercise them.

## 4. Manual test of the demo notebook

Execute the notebook headlessly:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute demo/demo_notebook.ipynb
```

Or open it in Jupyter and choose Run All:

```bash
jupyter notebook demo/demo_notebook.ipynb
```

What to look for (the notebook should run top to bottom without errors):

- Internal encoding: the demo passes RAW data (human-readable categoricals) to
  `Whitebox`, and the encoder builds the integer-coded columns internally, for
  example `region_encoded`. There is NO call to the deprecated
  `prepare_data_for_whitebox`; it is only mentioned in comments/markdown as
  removed.
- Governance statuses: variables report a `review_status` (for example
  `pending_review`, then `ready_to_model` after review).
- Derived variables: the notebook builds a grouped variable (for example
  `region_group` via `add_group`) and a combined variable (via
  `add_combination`), then plots them. The derived variable should appear in
  `wb.plottable_variables()` and produce a univariate plot.

## 5. Manual test of the Streamlit report

Install the report extra first:

```bash
source .venv/bin/activate
pip install -e '.[report]'
```

Then build a Whitebox from the demo data/model and launch the report:

```python
from demo.synthetic_data import get_demo_data_and_model
from whitebox import Whitebox

data, model, feature_names = get_demo_data_and_model()

wb = Whitebox(
    data=data,
    model=model,
    weight_col="exposure",
    actuals_col="claim_count",
    feature_names=feature_names,
    link_fn="poisson",
)

wb.launch_report()  # opens the Streamlit app (default port 8501)
```

`launch_report()` precomputes SHAP, pickles the Whitebox to a temp file, and
starts `streamlit run` on the bundled app at `whitebox/report/app.py`. Open the
URL it prints (default http://localhost:8501).

Pages to click (sidebar, "View" radio):

- One-way: pick a variable, render a univariate plot.
- Two-way: pick two variables, render a bivariate plot.
- Data review: profile a variable and propose/apply cleaning or banding
  (`by_bins` / `by_magnitude`).
- Variable manager: create grouped/combined derived variables and set statuses.

When you save a plot from a page, the report writes standalone HTML into the
export directory `whitebox_report_exports/` (created in the current working
directory by default; override with `export_dir=...`).

## 6. Manual test of the MCP server

Install the mcp extra first:

```bash
source .venv/bin/activate
pip install -e '.[mcp]'
```

Run the FastMCP (stdio) server directly:

```bash
python -m whitebox.mcp.server
```

This is a stdio server, so it will sit waiting for an MCP client on
stdin/stdout (nothing is printed to stdout by design). It is wired the same way
in `.mcp.json`:

```json
{
  "mcpServers": {
    "whitebox_mcp": {
      "command": "python",
      "args": ["-m", "whitebox.mcp.server"]
    }
  }
}
```

To interact with it manually, use the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python -m whitebox.mcp.server
```

From the Inspector you can list the registered tools (all prefixed
`whitebox_`, for example `whitebox_load_session`, `whitebox_list_variables`,
`whitebox_describe_variable`, `whitebox_add_group`, `whitebox_univariate_plot`)
and call them.

A set of read-only, verifiable question/answer pairs over the demo dataset lives
at `whitebox/mcp/evaluation.xml`. These are useful for sanity-checking tool
behavior (for example: after `whitebox_load_session` on the demo, the total
variable count is 5 and every raw feature starts at `pending_review`).

## 7. Registry validator

Validate a collaborative registry file:

```bash
source .venv/bin/activate
python -m whitebox.registry.validate registry.json
```

The validator checks for duplicate variable names and `ready_to_model`
invariants. If `registry.json` does not exist, it no-ops cleanly (prints
"registry file not found (nothing to validate)" and exits 0). Note that
`registry.json` is intentionally not committed on feature branches, so this
will commonly no-op until a registry file is present locally.

## 8. Quick Python smoke test

A minimal end-to-end check you can paste into a Python shell (inside the
activated `.venv`):

```python
from demo.synthetic_data import get_demo_data_and_model
from whitebox import Whitebox

data, model, feature_names = get_demo_data_and_model()

wb = Whitebox(
    data=data,
    model=model,
    weight_col="exposure",
    actuals_col="claim_count",
    feature_names=feature_names,
    link_fn="poisson",
    verbose=False,
)

# Internal encoding + SHAP
wb.DataPrep.prep_shap_values()

# Derived variable (grouped) via governance API
wb.add_group(
    "region_group",
    "region",
    {"North": "Cold", "West": "Cold", "South": "Warm", "East": "Warm"},
)
assert "region_group" in wb.plottable_variables()

# Headless univariate plot (no display, returns the figure)
fig = wb.univariate_plot("region_group", show=False)
print("smoke OK:", fig is not None)
```

If this prints `smoke OK: True` with no exceptions, internal encoding, SHAP,
derived variables, and headless plotting are all working.
