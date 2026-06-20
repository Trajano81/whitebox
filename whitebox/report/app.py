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


def figure_html(fig, key):
    """Bokeh figure -> standalone HTML string. The figure stretches to the iframe
    width so the right-side legend is not clipped."""
    try:
        fig.sizing_mode = "stretch_width"
    except Exception:
        pass
    return file_html(fig, CDN, key)


def show_html(html, height=640):
    """Embed an HTML string into the page."""
    components.html(html, height=height, scrolling=True)


def _optional_float(text):
    text = (text or "").strip()
    return float(text) if text else None


def save_summary(html, filename):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, "w") as f:
        f.write(html)
    return path


def page_one_way(wb):
    st.header("One-way (univariate)")
    var = st.selectbox("Variable", wb.plottable_variables())

    st.markdown("**Series to show**")
    c1, c2, c3, c4 = st.columns(4)
    shap = c1.checkbox("SHAP", value=True)
    shap_points = c1.checkbox("SHAP points", value=False)
    shap_sd = c2.checkbox("SHAP +/- SD", value=False)
    glm = c2.checkbox("GLM indication", value=False)
    glm_pred = c3.checkbox("GLM prediction", value=False)
    gbm_pred = c3.checkbox("GBM prediction", value=False)
    actuals = c4.checkbox("Actuals", value=False)
    weight = c4.checkbox("Weight", value=True)

    with st.expander("Plot options", expanded=False):
        st.markdown("**Axis main options**")
        oc1, oc2, oc3 = st.columns(3)
        rebase = oc1.checkbox("Rebase", value=True)
        infinity_lower = oc2.checkbox("Infinity lower band", value=True)
        infinity_higher = oc3.checkbox("Infinity higher band", value=True)
        bc1, bc2, bc3 = st.columns(3)
        nlevels = bc1.number_input("Bands (numeric, nlevels; 0 = auto)", 0, 100, 0)
        pstart = bc2.number_input("Percentile start", 0, 100, 10)
        pfinish = bc3.number_input("Percentile finish", 0, 100, 90)
        rc1, rc2, rc3 = st.columns(3)
        start = rc1.text_input("Start (override)", value="")
        finish = rc2.text_input("Finish (override)", value="")
        stepsize = rc3.text_input("Stepsize (override)", value="")
        yc1, yc2 = st.columns(2)
        y_min = yc1.text_input("Y axis min", value="")
        y_max = yc2.text_input("Y axis max", value="")

        st.markdown("**SHAP options**")
        sc1, sc2 = st.columns(2)
        n_shap_points = sc1.number_input("SHAP points sample (n)", 100, 200000, 1000, step=100)
        ci_z = sc2.number_input("SD band width (ci_z)", 0.0, 5.0, 2.0, step=0.5)

        st.markdown("**Labels**")
        plot_name = st.text_input("Plot title", value="Univariate Plot")

    if st.button("Render", key="render_one"):
        kwargs = dict(
            shap=shap, shap_points=shap_points, glm=glm, glm_pred=glm_pred,
            gbm_pred=gbm_pred, actuals=actuals, weight=weight, rebase=rebase,
            infinity_lower=infinity_lower, infinity_higher=infinity_higher,
            percentile_start=int(pstart), percentile_finish=int(pfinish),
            n_shap_points=int(n_shap_points), ci_z=float(ci_z),
            plot_name=plot_name, show=False,
        )
        if shap_sd:
            kwargs["shap_sd"] = True
        if int(nlevels) > 0:
            kwargs["nlevels"] = int(nlevels)
        for name, val in (("start", start), ("finish", finish), ("stepsize", stepsize),
                          ("y_axis_min", y_min), ("y_axis_max", y_max)):
            f = _optional_float(val)
            if f is not None:
                kwargs[name] = f
        try:
            fig = wb.univariate_plot(var, **kwargs)
            st.session_state["one_html"] = figure_html(fig, var)
            st.session_state["one_name"] = f"univariate_{var}.html"
        except Exception as e:  # noqa: BLE001
            st.error(f"Render failed: {e}")

    if st.session_state.get("one_html"):
        show_html(st.session_state["one_html"])
        if st.button("Save summary", key="save_one"):
            st.success(
                f"Saved {save_summary(st.session_state['one_html'], st.session_state['one_name'])}"
            )


def page_two_way(wb):
    st.header("Two-way (bivariate)")
    options = wb.plottable_variables()
    var1 = st.selectbox("Variable 1", options, key="bv1")
    var2 = st.selectbox("Variable 2", options, index=min(1, len(options) - 1), key="bv2")

    st.markdown("**Series to show**")
    c1, c2, c3, c4 = st.columns(4)
    shap = c1.checkbox("SHAP", value=True, key="bv_shap")
    glm = c2.checkbox("GLM indication", value=False, key="bv_glm")
    gbm_pred = c3.checkbox("GBM prediction", value=False, key="bv_gbm")
    actuals = c4.checkbox("Actuals", value=False, key="bv_act")

    with st.expander("Plot options", expanded=False):
        st.markdown("**Axis main options**")
        rebase = st.checkbox("Rebase", value=True, key="bv_rebase")
        st.caption("Bands per variable (numeric; 0 = auto)")
        v1c1, v1c2, v1c3 = st.columns(3)
        nlevels_var1 = v1c1.number_input("var1 nlevels", 0, 100, 0, key="bv_nl1")
        ps1 = v1c2.number_input("var1 percentile start", 0, 100, 1, key="bv_ps1")
        pf1 = v1c3.number_input("var1 percentile finish", 0, 100, 99, key="bv_pf1")
        v2c1, v2c2, v2c3 = st.columns(3)
        nlevels_var2 = v2c1.number_input("var2 nlevels", 0, 100, 0, key="bv_nl2")
        ps2 = v2c2.number_input("var2 percentile start", 0, 100, 1, key="bv_ps2")
        pf2 = v2c3.number_input("var2 percentile finish", 0, 100, 99, key="bv_pf2")

        st.markdown("**Labels**")
        plot_title = st.text_input("Plot title", value="Bivariate plot", key="bv_title")

    if st.button("Render", key="render_two"):
        kwargs = dict(
            shap=shap, glm=glm, gbm_pred=gbm_pred, actuals=actuals, rebase=rebase,
            percentile_start_var1=int(ps1), percentile_finish_var1=int(pf1),
            percentile_start_var2=int(ps2), percentile_finish_var2=int(pf2),
            plot_title=plot_title, show=False,
        )
        if int(nlevels_var1) > 0:
            kwargs["nlevels_var1"] = int(nlevels_var1)
        if int(nlevels_var2) > 0:
            kwargs["nlevels_var2"] = int(nlevels_var2)
        try:
            fig = wb.bivariate_plot(var1, var2, **kwargs)
            st.session_state["two_html"] = figure_html(fig, f"{var1}_x_{var2}")
            st.session_state["two_name"] = f"bivariate_{var1}_x_{var2}.html"
        except Exception as e:  # noqa: BLE001
            st.error(f"Render failed: {e}")

    if st.session_state.get("two_html"):
        show_html(st.session_state["two_html"])
        if st.button("Save summary", key="save_two"):
            st.success(
                f"Saved {save_summary(st.session_state['two_html'], st.session_state['two_name'])}"
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
    if st.button(
        "Profile + propose",
        key="profile_btn",
        help=(
            "Scan the selected variable for data-quality issues (the Profile) and "
            "suggest an editable cleaning plan (the Proposal). Numeric variables are "
            "banded into bins; categorical variables get a level map that folds "
            "missing tokens to 'Missing' and rare levels to 'Other'. Review the "
            "proposal, then use 'Apply cleaning' to materialize it."
        ),
    ):
        prof = wb.profile(var)
        st.json(prof)
        with st.expander("What do these profile fields mean?"):
            st.markdown(
                "- **name**: the variable being profiled.\n"
                "- **dtype**: `numeric` or `categorical` (decides which cleaning is offered).\n"
                "- **n**: total number of rows.\n"
                "- **n_missing**: values counted as missing, real NaN/None plus tokens like "
                "`\"\"`, `NA`, `.`, `null`, `NaN` (case and whitespace insensitive).\n"
                "- **missing_pct**: `n_missing` as a percentage of `n`.\n"
                "- **n_unique**: distinct non-missing values.\n"
                "- **top_levels**: the 20 most frequent categories with count and pct "
                "(empty for numeric variables).\n"
                "- **odd_tokens**: distinct category values with odd characters (anything "
                "outside letters, digits, space, `_`, `-`) or stray whitespace "
                "(empty for numeric variables).\n"
                "- **flags**: review warnings. `high_missingness` (>= 20% missing), "
                "`odd_tokens` (odd characters found), `high_cardinality` (categorical with "
                "> 50 levels). An empty list means no issues were detected."
            )
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
    # Wide layout uses the full window width (the default centered column is too
    # narrow for the plots + legend). Must be the first Streamlit call.
    st.set_page_config(page_title="Whitebox report", layout="wide")
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
