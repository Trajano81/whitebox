"""Report launcher import-guard + wiring (spec-005)."""
import importlib.util

import pytest

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@pytest.mark.skipif(HAS_STREAMLIT, reason="streamlit installed; import-guard path not exercised")
def test_launch_report_requires_streamlit(wb):
    with pytest.raises(ImportError) as exc:
        wb.launch_report()
    assert "whitebox[report]" in str(exc.value)


@pytest.mark.skipif(not HAS_STREAMLIT, reason="streamlit not installed")
def test_launch_report_wires_without_spawn(wb):
    from whitebox.report.launcher import launch_report

    info = launch_report(wb, _spawn=False)
    assert info["pkl_path"].endswith(".whitebox.pkl")
    assert "streamlit" in info["cmd"]
    # Must run headless so the first-run email prompt does not block the subprocess.
    assert "--server.headless" in info["cmd"]
    assert info["cmd"][info["cmd"].index("--server.headless") + 1] == "true"
    # SHAP precomputed and explainer dropped for a clean pickle.
    assert wb.shap_df is not None
    assert wb.explainer is None
