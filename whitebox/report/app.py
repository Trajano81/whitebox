"""Streamlit report app (run as a subprocess by launcher.launch_report).

Usage (set up by the launcher):
    streamlit run app.py -- <whitebox.pkl> <export_dir>

Pages: One-way, Two-way, Data review / governance, Variable manager.
Bokeh figures are embedded as standalone HTML via bokeh.embed.file_html, which
avoids the Bokeh 3.x incompatibility with st.bokeh_chart.
"""
import os
import pickle
import sys

import streamlit as st
import streamlit.components.v1 as components
from bokeh.embed import file_html
from bokeh.resources import CDN

_ARGV = sys.argv[1:]
PKL_PATH = _ARGV[0] if _ARGV else None
EXPORT_DIR = _ARGV[1] if len(_ARGV) > 1 else "whitebox_report_exports"


@st.cache_resource
def load_wb(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def render(fig, key, height=560):
    """Embed a Bokeh figure as standalone HTML and return the HTML string."""
    html = file_html(fig, CDN, key)
    components.html(html, height=height, scrolling=True)
    return html


def save_summary(html, filename):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, "w") as f:
        f.write(html)
    return path


def page_one_way(wb):
    st.header("One-way (univariate)")
    var = st.selectbox("Variable", wb.plottable_variables())
    c1, c2, c3, c4 = st.columns(4)
    shap = c1.checkbox("SHAP", value=True)
    glm = c2.checkbox("GLM", value=False)
    actuals = c3.checkbox("Actuals", value=False)
    weight = c4.checkbox("Weight", value=True)
    if st.button("Render", key="render_one"):
        fig = wb.univariate_plot(
            var, shap=shap, glm=glm, actuals=actuals, weight=weight, show=False
        )
        html = render(fig, var)
        if st.button("Save summary", key="save_one"):
            st.success(f"Saved {save_summary(html, f'univariate_{var}.html')}")


def page_two_way(wb):
    st.header("Two-way (bivariate)")
    options = wb.plottable_variables()
    var1 = st.selectbox("Variable 1", options, key="bv1")
    var2 = st.selectbox("Variable 2", options, index=min(1, len(options) - 1), key="bv2")
    if st.button("Render", key="render_two"):
        fig = wb.bivariate_plot(var1, var2, shap=True, show=False)
        html = render(fig, f"{var1}_x_{var2}")
        if st.button("Save summary", key="save_two"):
            st.success(
                f"Saved {save_summary(html, f'bivariate_{var1}_x_{var2}.html')}"
            )


def page_data_review(wb):
    st.header("Data review / governance")
    rows = []
    for name in wb.plottable_variables():
        rec = wb.encoder.registry.get(name, {})
        rows.append(
            {
                "name": name,
                "origin": rec.get("origin"),
                "status": rec.get("review_status"),
                "created_by": rec.get("created_by"),
                "flags": ", ".join((rec.get("data_quality") or {}).get("flags", [])),
            }
        )
    st.dataframe(rows, use_container_width=True)

    var = st.selectbox("Review variable", wb.plottable_variables(), key="review_var")
    if st.button("Profile + propose", key="profile_btn"):
        prof = wb.profile(var)
        st.json(prof)
        is_numeric = prof["dtype"] == "numeric"
        method = st.selectbox("Banding method", ["by_bins", "by_magnitude"]) if is_numeric else "by_bins"
        n_bins = st.number_input("Number of bins", min_value=2, max_value=50, value=10)
        proposal = wb.propose_cleaning(var, method=method, n_bins=int(n_bins))
        st.write("Proposal (edit then apply):")
        if proposal["kind"] == "categorical_cleaning":
            st.json(proposal["level_map"])
        else:
            st.json({"edges": proposal["edges"], "labels": proposal["labels"]})
        if st.button("Apply cleaning", key="apply_btn"):
            wb.apply_cleaning(var, proposal)
            st.success(f"Applied cleaning to {var}")

    c1, c2, c3 = st.columns(3)
    if c1.button("Mark ready_to_model", key="ready_btn"):
        try:
            wb.set_status(var, "ready_to_model")
            st.success(f"{var} -> ready_to_model")
        except ValueError as e:
            st.error(str(e))
    if c2.button("Mark needs_cleaning", key="clean_btn"):
        wb.set_status(var, "needs_cleaning")
    if c3.button("Exclude", key="excl_btn"):
        wb.set_status(var, "excluded")

    st.subheader("Trainable manifest (ready_to_model)")
    st.write(wb.trainable_variables())


def page_variable_manager(wb):
    st.header("Variable manager")
    st.subheader("Create a group")
    src = st.selectbox("Source", [n for n in wb.plottable_variables()], key="grp_src")
    gname = st.text_input("New variable name", key="grp_name")
    levels = sorted(wb.data[src].astype(str).unique()) if src else []
    mapping = {}
    for lvl in levels[:40]:
        mapping[lvl] = st.text_input(f"{lvl} ->", value=lvl, key=f"grp_{lvl}")
    if st.button("Create group", key="create_grp"):
        try:
            wb.add_group(gname, src, mapping)
            st.success(f"Created group {gname} (pending_review)")
        except ValueError as e:
            st.error(str(e))

    st.subheader("Create a combination")
    sources = st.multiselect("Sources (2+)", wb.plottable_variables(), key="cmb_src")
    cname = st.text_input("New variable name", key="cmb_name")
    if st.button("Create combination", key="create_cmb"):
        try:
            wb.add_combination(cname, sources)
            st.success(f"Created combination {cname} (pending_review)")
        except ValueError as e:
            st.error(str(e))

    st.subheader("Derived variables")
    derived = [n for n in wb.encoder.registry if wb.encoder.is_derived(n)]
    for name in derived:
        rec = wb.encoder.registry[name]
        cols = st.columns([3, 2, 1])
        cols[0].write(f"**{name}** ({rec['origin']}, {rec['review_status']})")
        if cols[1].button("Promote", key=f"prom_{name}"):
            try:
                wb.set_status(name, "ready_to_model")
            except ValueError as e:
                st.error(str(e))
        if cols[2].button("Delete", key=f"del_{name}"):
            wb.remove_variable(name)
            st.rerun()


def main():
    if not PKL_PATH:
        st.error("No Whitebox pickle provided.")
        return
    wb = load_wb(PKL_PATH)
    st.sidebar.title("Whitebox report")
    page = st.sidebar.radio(
        "View", ["One-way", "Two-way", "Data review", "Variable manager"]
    )
    if page == "One-way":
        page_one_way(wb)
    elif page == "Two-way":
        page_two_way(wb)
    elif page == "Data review":
        page_data_review(wb)
    else:
        page_variable_manager(wb)


if __name__ == "__main__" or True:  # Streamlit executes the module top-to-bottom.
    main()
