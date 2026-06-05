"""Interactive localhost report (Streamlit).

`launch_report(wb)` pickles the Whitebox instance and starts a Streamlit app with
One-way, Two-way, Data-review/governance and Variable-manager pages. Bokeh figures
are embedded as standalone HTML (avoiding the Bokeh 3.x / st.bokeh_chart conflict).
"""
from .launcher import launch_report

__all__ = ["launch_report"]
