"""Launch the Streamlit report as a subprocess.

The Whitebox instance is pickled to a temp file and passed to the app. SHAP is
precomputed and the (unpicklable) explainer dropped so the subprocess never needs
to recompute it.
"""
import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile


def _require_streamlit():
    if importlib.util.find_spec("streamlit") is None:
        raise ImportError(
            "the report requires Streamlit. Install it with: pip install 'whitebox[report]'"
        )


def launch_report(wb, port=8501, export_dir=None, _spawn=True):
    """Pickle `wb` and start the Streamlit report app.

    Returns the path to the pickle and the launch command. When `_spawn` is False
    the subprocess is not started (used by tests to validate wiring without a
    running server).
    """
    _require_streamlit()

    if export_dir is None:
        export_dir = os.path.join(os.getcwd(), "whitebox_report_exports")
    os.makedirs(export_dir, exist_ok=True)

    # Precompute SHAP so the subprocess only needs the DataFrame, then drop the
    # explainer (TreeExplainer holds ctypes state and is excluded from pickling).
    if wb.shap_df is None:
        wb.DataPrep.prep_shap_values()
    wb.explainer = None

    fd, pkl_path = tempfile.mkstemp(suffix=".whitebox.pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(wb, f)

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(port),
        "--", pkl_path, export_dir,
    ]
    proc = None
    if _spawn:
        proc = subprocess.Popen(cmd)
    return {"pkl_path": pkl_path, "export_dir": export_dir, "cmd": cmd, "process": proc}
