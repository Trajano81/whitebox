"""E2E smoke for the Streamlit report (spec-005).

Skipped unless streamlit is installed. Launches the app subprocess on a free
port, waits for it to serve, and asserts it responds, then terminates it.
"""
import importlib.util
import socket
import time
import urllib.request

import pytest

pytestmark = pytest.mark.e2e

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.skipif(not HAS_STREAMLIT, reason="streamlit not installed")
def test_report_serves(wb_clean):
    from whitebox.report.launcher import launch_report

    port = _free_port()
    info = launch_report(wb_clean, port=port)
    proc = info["process"]
    try:
        ok = False
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                    if r.status == 200:
                        ok = True
                        break
            except Exception:
                time.sleep(1)
        assert ok, "Streamlit report did not start serving"
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=15)
