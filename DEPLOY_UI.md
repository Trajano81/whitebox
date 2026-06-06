# Deploying / running the Whitebox Streamlit report UI

This guide explains how to launch the interactive Whitebox report, with a focus
on starting it straight from the demo notebook (`demo/demo_notebook.ipynb`).

The report is a small Streamlit app (`whitebox/report/app.py`) that loads a
pickled `Whitebox` instance and gives you four interactive pages: One-way,
Two-way, Data review, and Variable manager. It is designed for local,
interactive use launched from a session (notebook or script).

---

## 1. Prerequisite: install the report extra

Streamlit is **not** installed by default. It lives in an optional extra. If you
call `launch_report()` without it, you get an `ImportError`:

```
the report requires Streamlit. Install it with: pip install 'whitebox[report]'
```

Install it into your virtual environment:

```bash
source .venv/bin/activate
pip install -e '.[report]'
```

The extra is defined in `pyproject.toml` as `report = ["streamlit>=1.30"]`.

If you use Poetry instead of pip, install the extra with:

```bash
poetry install --extras report
```

---

## 2. Launch from the demo notebook (the main path)

The notebook builds a `Whitebox` instance named `wb` earlier on (in the
`Whitebox(...)` cell). The very last code cell contains a commented-out call:

```python
# wb.launch_report()
```

To run the report, uncomment it (capturing the return value is recommended so
you can stop the server later):

```python
info = wb.launch_report()
```

Then open the app in your browser:

```
http://localhost:8501
```

### What actually happens when you call it

`wb.launch_report()` (in `whitebox/core.py`) delegates to
`whitebox/report/launcher.py:launch_report(wb, port=8501, export_dir=None)`,
which does the following:

1. Checks Streamlit is installed (raises the `ImportError` above if not).
2. Resolves the export directory. With the default `export_dir=None` it uses
   `./whitebox_report_exports` (relative to the current working directory) and
   creates it.
3. **Precomputes SHAP** if `wb.shap_df` is `None` (calls
   `wb.DataPrep.prep_shap_values()`), then sets `wb.explainer = None`. The
   TreeExplainer holds ctypes state and is not picklable, so it is dropped. The
   subprocess only ever needs the already-computed SHAP DataFrame.
4. **Pickles `wb`** to a temp file (suffix `.whitebox.pkl`).
5. Starts the server as a **subprocess** via `subprocess.Popen`, effectively
   running:

   ```
   python -m streamlit run <.../report/app.py> --server.port 8501 -- <pkl_path> <export_dir>
   ```

6. Returns a dict:

   ```python
   {"pkl_path": ..., "export_dir": ..., "cmd": [...], "process": <Popen>}
   ```

Because the Streamlit server runs as an independent subprocess, it keeps running
even after the notebook cell finishes. Your notebook stays usable.

### Stopping the server

Keep the returned dict and terminate the process:

```python
info = wb.launch_report()
# ... use the report ...
info["process"].terminate()
```

If you did not keep the handle, kill the Streamlit process from a shell. For
example, on macOS / Linux:

```bash
pkill -f "streamlit run"
# or target the specific port:
lsof -ti :8501 | xargs kill
```

### Custom port / export directory

```python
info = wb.launch_report(port=8600, export_dir="my_exports")
```

This serves on `http://localhost:8600` and writes saved summaries into
`my_exports/`.

---

## 3. Launch from a plain Python script (outside the notebook)

You can build a `wb` from the bundled synthetic demo helper and launch the same
report. Note `get_demo_data_and_model` returns a 3-tuple
`(data, model, feature_names)`.

```python
# run_report.py  (run from the repo root so demo/ is importable)
import sys
sys.path.insert(0, "demo")

from whitebox.core import Whitebox
from synthetic_data import get_demo_data_and_model

data, model, feature_names = get_demo_data_and_model()

wb = Whitebox(
    data=data,
    model=model,
    weight_col="exposure",
    actuals_col="claim_count",
    feature_names=feature_names,
    link_fn="poisson",
    verbose=True,
)

info = wb.launch_report()  # default port 8501, export_dir ./whitebox_report_exports
print("Report running. cmd:", info["cmd"])
print("Open http://localhost:8501")

# The Streamlit server runs as a subprocess and outlives this script only if the
# parent process stays alive. To keep the script running until you stop it:
try:
    info["process"].wait()
except KeyboardInterrupt:
    info["process"].terminate()
```

Run it:

```bash
source .venv/bin/activate
python run_report.py
```

Press Ctrl-C to stop. (In a one-shot script, do not let the process exit
immediately after `launch_report()`, or the subprocess may be cleaned up before
you can use it. `info["process"].wait()` keeps it alive.)

---

## 4. Run the Streamlit app directly (advanced / reproducible)

This mirrors exactly what the launcher does, but you control each step. First
pickle a `Whitebox` yourself (remember to precompute SHAP and drop the
explainer, just like the launcher), then run Streamlit against the saved pickle.

Pickle step:

```python
# make_pickle.py
import pickle, sys
sys.path.insert(0, "demo")

from whitebox.core import Whitebox
from synthetic_data import get_demo_data_and_model

data, model, feature_names = get_demo_data_and_model()
wb = Whitebox(
    data=data, model=model, weight_col="exposure", actuals_col="claim_count",
    feature_names=feature_names, link_fn="poisson",
)
if wb.shap_df is None:
    wb.DataPrep.prep_shap_values()   # precompute SHAP
wb.explainer = None                  # drop the unpicklable TreeExplainer

with open("wb.pkl", "wb") as f:
    pickle.dump(wb, f)
print("wrote wb.pkl")
```

Then launch the app, passing the two positional arguments after `--`:

```bash
python -m streamlit run whitebox/report/app.py --server.port 8501 -- wb.pkl whitebox_report_exports
```

The app reads `sys.argv` directly (`whitebox/report/app.py`):

- The **first** positional arg after `--` is the **pickle path**
  (`PKL_PATH = _ARGV[0]`). The pickle is loaded once and cached with
  `@st.cache_resource`.
- The **second** positional arg after `--` is the **export directory**
  (`EXPORT_DIR = _ARGV[1]`, defaulting to `whitebox_report_exports` if omitted).

The `--` separator is required: it tells Streamlit that everything after it
belongs to your script, not to Streamlit itself.

---

## 5. The four pages

The sidebar radio (`whitebox/report/app.py`) switches between:

- **One-way (univariate)**: pick a variable and toggle SHAP / GLM / Actuals /
  Weight, then **Render** a `wb.univariate_plot(...)`.
- **Two-way (bivariate)**: pick two variables and **Render** a
  `wb.bivariate_plot(var1, var2, shap=True)`.
- **Data review / governance**: a table of every plottable variable with its
  origin, review status, creator, and data-quality flags. You can profile a
  variable, propose and apply cleaning (banding by bins or by magnitude), and
  set its status (`ready_to_model`, `needs_cleaning`, `excluded`). It also shows
  the trainable manifest (the `ready_to_model` variables).
- **Variable manager**: create a group (level remapping of one source), create a
  combination (2+ sources), and promote or delete derived variables.

### Save summary

On the One-way and Two-way pages, after rendering, a **Save summary** button
writes a standalone Bokeh HTML file into `export_dir`
(`whitebox_report_exports/` by default), for example
`univariate_age.html` or `bivariate_age_x_region.html`. These are
self-contained HTML files you can share or archive.

---

## 6. Sharing / deploying beyond localhost

### LAN access

By default Streamlit binds to localhost. To make the report reachable from other
machines on your network, bind to all interfaces:

```bash
python -m streamlit run whitebox/report/app.py \
  --server.address 0.0.0.0 --server.port 8501 \
  -- wb.pkl whitebox_report_exports
```

Security caveat: the report has **no built-in authentication**. Anyone who can
reach the host and port can view your data and use the governance / variable
management controls. Do not expose it directly to the public internet. If you
need remote access, put it behind a reverse proxy that handles auth (for
example nginx with basic auth or an SSO proxy), or restrict access to a VPN.

### Dockerized / long-running deployment

A persistent deployment is more involved than the local/interactive workflow,
because the app needs a pickled `Whitebox` available at the exact path passed as
`argv` (`PKL_PATH`). The app does not build a model itself, it only loads the
pickle. So a long-running deployment should **generate or refresh that pickle as
part of startup** (or on a schedule), then point Streamlit at it.

A minimal container entrypoint would:

1. Build a `Whitebox`, precompute SHAP, set `explainer = None`, and pickle it to
   a known path (see the `make_pickle.py` step above).
2. Run
   `python -m streamlit run whitebox/report/app.py --server.address 0.0.0.0 --server.port 8501 -- /path/to/wb.pkl /path/to/exports`.

Keep in mind the report is designed for local, interactive use launched from a
session. Treat a hosted deployment as a read-only viewer of a pre-built pickle,
and remember the no-auth caveat above.

---

## 7. Troubleshooting

- **`ImportError: the report requires Streamlit ...`**: the optional extra is
  not installed. Run `pip install -e '.[report]'` (or
  `pip install 'whitebox[report]'`).
- **Port already in use**: pass a different port,
  `wb.launch_report(port=8600)`, or `--server.port 8600` when running the app
  directly. You can also free the default port with
  `lsof -ti :8501 | xargs kill`.
- **Blank page / Bokeh figure not rendering**: this is already handled. The app
  embeds figures with `bokeh.embed.file_html(...)` plus
  `streamlit.components.v1.html(...)`, which works with Bokeh 3.x. It
  deliberately avoids `st.bokeh_chart`, which is incompatible with Bokeh 3.x. If
  you see nothing, check the terminal running Streamlit for tracebacks and
  confirm your Bokeh version matches the pin in `pyproject.toml`
  (`bokeh>=3.0.3,<3.5`).
- **`No Whitebox pickle provided.`**: you ran the app directly without the
  pickle path after `--`. Pass it as the first positional argument:
  `... -- wb.pkl whitebox_report_exports`.
