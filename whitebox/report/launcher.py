"""Launch the Streamlit report as a subprocess.

The Whitebox instance is pickled to a temp file and passed to the app. SHAP is
precomputed and the (unpicklable) explainer dropped so the subprocess never needs
to recompute it.

The server runs headless on purpose: that skips Streamlit's first-run interactive
email/onboarding prompt, which would otherwise block on stdin (there is no
terminal when launched from a notebook) and make the subprocess exit before it
binds the port. Because headless mode does not auto-open a browser, this launcher
opens it after the server is reachable.
"""
import importlib.util
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser


def _require_streamlit():
    if importlib.util.find_spec("streamlit") is None:
        raise ImportError(
            "the report requires Streamlit. Install it with: pip install 'whitebox[report]'"
        )


def _port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _wait_until_serving(port, proc, timeout=30):
    """Wait until the port accepts connections. Returns True if serving, False if
    the process died first or the timeout elapsed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if _port_open(port):
            return True
        time.sleep(0.5)
    return _port_open(port)


def launch_report(wb, port=8501, export_dir=None, open_browser=True, _spawn=True):
    """Pickle `wb` and start the Streamlit report app (headless).

    Returns a dict with the pickle path, export dir, launch command, the process,
    the URL, and the path to the captured server log. When `_spawn` is False the
    subprocess is not started (used by tests to validate wiring without a running
    server). Raises RuntimeError if the server exits before it starts serving.
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
    log_path = os.path.join(export_dir, "streamlit_report.log")
    url = f"http://localhost:{port}"
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(port),
        "--server.headless", "true",   # skip the first-run email prompt (no stdin)
        "--", pkl_path, export_dir,
    ]
    proc = None
    if _spawn:
        # Belt and suspenders: also set the env so the prompt is skipped even if
        # the flag is ignored, and capture the child's output for diagnostics.
        env = dict(os.environ, STREAMLIT_SERVER_HEADLESS="true")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

        if not _wait_until_serving(port, proc):
            try:
                with open(log_path) as f:
                    tail = f.read()[-2000:]
            except OSError:
                tail = "(no log captured)"
            raise RuntimeError(
                f"Streamlit report failed to start on port {port} "
                f"(process exit code {proc.poll()}). Server log tail:\n{tail}\n"
                f"Common causes: the port is already in use (pass a different port=), "
                f"or streamlit is not installed in this kernel's environment."
            )
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        print(f"Whitebox report serving at {url} (log: {log_path})")

    return {
        "pkl_path": pkl_path,
        "export_dir": export_dir,
        "cmd": cmd,
        "process": proc,
        "url": url,
        "log_path": log_path,
    }
