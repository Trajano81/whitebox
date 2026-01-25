# bokeh plot
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, LinearAxis, Range1d, Select, Legend, FixedTicker
import bokeh.io
from bokeh.io import output_notebook, curdoc, push_notebook, show
from bokeh.layouts import row, column
from bokeh.palettes import magma, viridis, cividis, RdYlBu, Category20c, Spectral
from collections import OrderedDict
from .Plot_interface import Plot_interface
from ..libs import *
import pandas as pd
import numpy as np

bokeh.io.reset_output()
bokeh.io.output_notebook()


class Bokeh_plot(Plot_interface):
    def __init__(self, mintypython):
        self.mintypython = mintypython
        pass

    def univariate_plot(self, kwargs, plot_data, var_name):
        """
        Creates a bokeh plot with univariate data

        PARAMETERS
        ----------

        plot_data: dictionary containing 2 dataframes ["agg_data","shap_points"]
                "agg_data" is all the statistics summarized by the var_name, like weight
                "shap_points" is the individual shap points to plot.

        var_name: variable name to plot on the x_axis

        kwargs: is a dictionary containing all the parameters to plot.

        OUTPUT
        ------

        bokeh plot
        """
        # hover tips
        T = [("Name", "$name"), ("X", "@var"), ("Value", "$y")]

        plot_data["agg_data"].index = plot_data["agg_data"].index.astype(str)

        x_axis_values = plot_data["agg_data"].index.values
        # Create numeric indices for x-axis (avoids Bokeh FactorRange bugs)
        x_indices = np.arange(len(x_axis_values))
        x_map = {v: i for i, v in enumerate(x_axis_values)}

        w_avg = plot_data["agg_data"]["weight"].values

        if "base" not in kwargs.keys():
            base = np.argmax(w_avg)
        else:
            if kwargs["base"] is None:
                base = np.argmax(w_avg)
            else:
                if kwargs["base"] in x_axis_values:
                    base = np.argmax(x_axis_values == kwargs["base"])
                else:
                    sample_values = x_axis_values[:3] if len(x_axis_values) > 3 else x_axis_values
                    print(
                        "Warning: The base level you selected is not in the data. "
                        + "dtype: " + str(x_axis_values.dtype) + ", "
                        + "sample values: " + str(list(sample_values))
                    )
                    base = np.argmax(w_avg)

        # create a new plot with numeric x-range
        p = figure(
            height=kwargs["height"],
            width=kwargs["width"],
            tooltips=T,
            x_range=Range1d(-0.5, len(x_axis_values) - 0.5),
        )
        # Configure x-axis with categorical labels at integer positions
        p.xaxis.ticker = FixedTicker(ticks=list(x_indices))
        p.xaxis.major_label_overrides = {i: str(v) for i, v in enumerate(x_axis_values)}

        # title
        p.title.text = kwargs["plot_name"]
        p.title.text_font_size = "14pt"

        # label
        p.xaxis.axis_label = var_name
        p.xaxis.axis_label_text_font_size = "12.5pt"

        # Setting the second y axis range name and range
        p.extra_y_ranges = {"Weight": Range1d(start=0, end=3.0 * max(w_avg))}

        # Adding the second axis to the plot.
        p.add_layout(LinearAxis(y_range_name="Weight", axis_label="Weight"), "right")

        # set y range
        lower = 1
        upper = 0

        # grid background
        p.grid.grid_line_alpha = 0
        p.ygrid.band_fill_color = "grey"
        p.ygrid.band_fill_alpha = 0.1

        # add renderers
        ######################################
        # plot weight
        source = ColumnDataSource(data=dict(var=x_indices, counts=w_avg))
        r_w = p.vbar(
            x="var",
            top="counts",
            width=0.9,
            source=source,
            line_color="white",
            color=self.mintypython.config["colors"]["weight"],
            alpha=0.5,
            y_range_name="Weight",
            name=self.mintypython.config["labels"]["weight"],
        )
        L_items = [(self.mintypython.config["labels"]["weight"], [r_w])]

        # shap points
        if "shap_points" in kwargs.keys():
            if kwargs["shap_points"]:
                avg_shap = plot_data["agg_data"]["shap"]
                x_shap = plot_data["shap_points"]["x_axis"].astype(str)
                if kwargs["rebase"]:
                    plot_data["shap_points"]["shap"] = (
                        plot_data["shap_points"]["shap"] / avg_shap.iloc[base]
                    ).values
                else:
                    plot_data["shap_points"]["shap"] = plot_data["shap_points"][
                        "shap"
                    ].values

                # Add random jitter to categorical indices (avoids Bokeh jitter() bug)
                x_map = {v: i for i, v in enumerate(x_axis_values)}
                x_jittered = np.array([x_map.get(v, 0) for v in x_shap.values]) + np.random.uniform(-0.25, 0.25, len(x_shap))
                source = ColumnDataSource(
                    data=dict(
                        var=x_jittered,
                        shap=plot_data["shap_points"]["shap"].values,
                    )
                )
                r_shap = p.circle(
                    x="var",
                    y="shap",
                    source=source,
                    size=2,
                    alpha=0.8,
                    name=self.mintypython.config["labels"]["shap_points"],
                    color=self.mintypython.config["colors"]["shap_points"],
                )
                L_items.append(
                    (self.mintypython.config["labels"]["shap_points"], [r_shap])
                )

        if "shap_sd" in kwargs.keys() and kwargs["shap_sd"]:
            avg_shap = plot_data["agg_data"]["shap"]
            if kwargs["rebase"]:
                plot_data["agg_data"]["shap_sd"] = (
                    plot_data["agg_data"]["shap_sd"] / avg_shap.iloc[base]
                )
                avg_shap = avg_shap / avg_shap.iloc[base]
            source = ColumnDataSource(
                data=dict(
                    var=x_indices,
                    top=avg_shap + plot_data["agg_data"]["shap_sd"],
                    bottom=avg_shap - plot_data["agg_data"]["shap_sd"],
                )
            )
            sd_shap = p.vbar(
                x="var",
                source=source,
                width=self.mintypython.config["line_width"]["shap_sd"],
                bottom="bottom",
                top="top",
                color=self.mintypython.config["colors"]["shap_sd"],
                name=self.mintypython.config["labels"]["shap_sd"],
                fill_alpha=0.3,
                line_alpha=0,
            )
            L_items.append((self.mintypython.config["labels"]["shap_sd"], [sd_shap]))
        # Lines
        for line in ["shap", "glm", "actuals", "glm_pred", "gbm_pred"]:
            if line in kwargs.keys() and kwargs[line]:
                avg = plot_data["agg_data"][line]
                if kwargs["rebase"]:
                    avgbase = avg.iloc[base]
                    if avgbase == 0:
                        print("Warning: unable to rebase " + line + ".")
                        avgbase = 1
                else:
                    avgbase = 1

                mean = np.average(avg.fillna(0), weights=np.sqrt(w_avg)) / avgbase
                sd = np.sqrt(np.cov(avg.fillna(0), aweights=np.sqrt(w_avg))) / avgbase

                lower = min(lower, np.nanmax([np.nanmin(avg / avgbase), mean - 3 * sd]))
                upper = max(upper, np.nanmin([np.nanmax(avg / avgbase), mean + 3 * sd]))
                source = ColumnDataSource(data=dict(var=x_indices, y=avg / avgbase))
                r_avg = p.line(
                    x="var",
                    y="y",
                    source=source,
                    line_width=self.mintypython.config["line_width"][line],
                    name=self.mintypython.config["labels"][line],
                    color=self.mintypython.config["colors"][line],
                )
                c_avg = p.diamond(
                    x="var",
                    y="y",
                    source=source,
                    size=10,
                    alpha=0.9,
                    name=self.mintypython.config["labels"][line],
                    color=self.mintypython.config["colors"][line],
                )
                L_items.append(
                    (self.mintypython.config["labels"][line], [r_avg, c_avg])
                )

        if "glm_ci" in kwargs.keys() and kwargs["glm_ci"]:
            sup = plot_data["agg_data"]["glm_+2sigma"]
            glm = plot_data["agg_data"]["glm_pred"]
            inf = plot_data["agg_data"]["glm_-2sigma"]

            if kwargs["rebase"]:
                glmbase = glm.iloc[base]
            else:
                glmbase = 1

            lower = min(lower, np.nanmin(inf / glmbase))
            upper = max(upper, np.nanmax(sup / glmbase))

            s_avg = p.line(
                x_indices,
                sup / glmbase,
                line_width=self.mintypython.config["line_width"]["glm_pred"],
                line_dash="dashed",
                name="sup",
                color=self.mintypython.config["colors"]["glm_pred"],
            )
            i_avg = p.line(
                x_indices,
                inf / glmbase,
                line_width=self.mintypython.config["line_width"]["glm_pred"],
                line_dash="dashed",
                name="inf",
                color=self.mintypython.config["colors"]["glm_pred"],
            )
            L_items.append(
                (
                    self.mintypython.config["labels"]["glm_ci"]
                    + str(kwargs["ci_z"])
                    + " \u03C3",
                    [i_avg, s_avg],
                )
            )

        # legend
        legend = Legend(items=L_items)
        p.add_layout(legend, "right")
        p.legend.click_policy = "hide"

        if kwargs["y_axis_max"] != None:
            upper = kwargs["y_axis_max"]

        if kwargs["y_axis_min"] != None:
            lower = kwargs["y_axis_min"]

        lower_ = min(lower, upper)
        upper_ = max(lower, upper)
        p.y_range = Range1d(
            lower_ - 0.5 * (upper_ - lower_), upper_ + 0.2 * (upper_ - lower_)
        )

        p.xaxis.major_label_orientation = 3.14 / 2
        self.figure = p

        if "save_plot" in kwargs.keys() and kwargs["save_plot"] is not None:
            output_file(kwargs["save_plot"] + var_name + ".html", title=var_name)

        show(p)
        return p
    
    def sort_banding_levels(self, data_to_plot_var):
        levels = pd.DataFrame(
            data_to_plot_var.cat.categories.values, columns=["bandings"]
        )
        cat = [float(levels.bandings[i].split(",")[0]) for i in range(len(levels))]
        levels = np.array(
            pd.Series(cat, index=levels.bandings).sort_values().dropna().index
        )
        return levels

    def bivariate_plot(
        self,
        data_to_plot,
        data_to_plot_agg,
        var1,
        var2,
        shap=True,
        glm=False,
        actuals=False,
        gbm_pred=False,
        base=None,
        shap_points=False,
        nlevels_var1=None,
        start_var1=None,
        finish_var1=None,
        stepsize_var1=None,
        percentile_start_var1=1,
        percentile_finish_var1=99,
        nlevels_var2=None,
        start_var2=None,
        finish_var2=None,
        stepsize_var2=None,
        percentile_start_var2=1,
        percentile_finish_var2=99,
        shap_points_seed=1000,
        n_shap_points=10000,
        plot_title="bivariate plot",
        rebase=True,
        glmindic_cols=None,
        joinshaps=None,
        save_plot=None,
        **kwargs
    ):
        if shap_points:
            np.random.seed(shap_points_seed)

            plot_index = np.random.choice(
                data_to_plot.index,
                size=min(n_shap_points, len(data_to_plot)),
                replace=False,
            )
            shap_data = data_to_plot.loc[plot_index]

        w_avg = data_to_plot.groupby("var1").sum()["weight"]

        x1_levels = (
            data_to_plot["var1"].drop_duplicates().sort_values().astype(str).values
        )
        x2_levels = (
            data_to_plot["var2"].drop_duplicates().sort_values().astype(str).values
        )
        # Create numeric indices for x-axis (avoids Bokeh FactorRange bugs)
        x1_indices = np.arange(len(x1_levels))
        x1_map = {v: i for i, v in enumerate(x1_levels)}

        if base is None:
            base = np.argmax(w_avg)
        else:
            if base in x1_levels:
                base = np.argmax(x1_levels == base)
            else:
                sample_values = x1_levels[:3] if len(x1_levels) > 3 else x1_levels
                print(
                    "Warning: The base level you selected is not in the data. "
                    + "dtype: " + str(x1_levels.dtype) + ", "
                    + "sample values: " + str(list(sample_values))
                )
                base = np.argmax(w_avg)

        # hover tips
        T = [("Name", "$name"), ("X", "@var"), ("Value", "$y")]

        # creat a plot with numeric x-range
        p = figure(
            height=kwargs["height"],
            width=kwargs["width"],
            tooltips=T,
            x_range=Range1d(-0.5, len(x1_levels) - 0.5),
        )
        # Configure x-axis with categorical labels at integer positions
        p.xaxis.ticker = FixedTicker(ticks=list(x1_indices))
        p.xaxis.major_label_overrides = {i: str(v) for i, v in enumerate(x1_levels)}

        p.yaxis.axis_label = "Shapley value of " + var1

        # Setting the second y axis range name and range
        p.extra_y_ranges = {
            "Weight": Range1d(start=0, end=3.7 * max(data_to_plot_agg["weight"]))
        }

        # Adding the second axis to the plot.
        p.add_layout(LinearAxis(y_range_name="Weight", axis_label="Weight"), "right")

        # set y range
        # set y range
        lower = 1
        upper = 1.000001

        colormap = dict(zip(x2_levels, cividis(len(x2_levels))))

        legend_item_list = []

        for _, level in enumerate(x2_levels):
            if shap:
                if rebase:
                    shapbase = data_to_plot_agg.xs(level, level="var2")[
                        "shap_avg"
                    ].iloc[base]
                else:
                    shapbase = 1

                x_cat = (
                    data_to_plot_agg.xs(level, level="var2")["shap_avg"]
                    .sort_index()
                    .index
                )
                x = np.array([x1_map.get(str(v), 0) for v in x_cat])
                y = (
                    data_to_plot_agg.xs(level, level="var2")["shap_avg"].sort_index()
                    / shapbase
                )

                lower = min(lower, np.nanmin(y))
                upper = max(upper, np.nanmax(y))

                source = ColumnDataSource(data=dict(var=x, y=y))
                r1 = p.line(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    line_width=1.5,
                    name=var2 + "=" + str(level),
                )
                c1 = p.diamond(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    fill_color="white",
                    size=8,
                )
                legend_item_list.append(
                    ("Wtg Avg Shap: " + var2 + "=" + str(level), [r1, c1])
                )

            if glm:
                if rebase:
                    glmbase = data_to_plot_agg.xs(level, level="var2")["glm_avg"].iloc[
                        base
                    ]
                else:
                    glmbase = 1

                x_cat = (
                    data_to_plot_agg.xs(level, level="var2")["glm_avg"]
                    .sort_index()
                    .index
                )
                x = np.array([x1_map.get(str(v), 0) for v in x_cat])
                y = (
                    data_to_plot_agg.xs(level, level="var2")["glm_avg"].sort_index()
                    / glmbase
                )

                lower = min(lower, np.nanmin(y))
                upper = max(upper, np.nanmax(y))

                source = ColumnDataSource(data=dict(var=x, y=y))
                r1 = p.line(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    line_width=0.5,
                    name=var2 + "=" + str(level),
                )
                c1 = p.square(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    fill_color="white",
                    size=8,
                )
                legend_item_list.append(
                    ("Wtg Avg Glm: " + var2 + "=" + str(level), [r1, c1])
                )
            
            if shap_points:
                # Add random jitter to categorical indices (avoids Bokeh jitter() bug)
                var1_values = shap_data.var1[shap_data.var2 == level].values
                x_jittered = np.array([x1_map.get(str(v), 0) for v in var1_values]) + np.random.uniform(-0.25, 0.25, len(var1_values))
                source = ColumnDataSource(
                    data=dict(
                        var=x_jittered,
                        shap=shap_data.shap[shap_data.var2 == level].values / shapbase,
                    )
                )
                r_shap = p.circle(
                    x="var",
                    y="shap",
                    source=source,
                    size=2,
                    alpha=0.8,
                    name=var2 + "=" + level,
                    color=colormap[level],
                )
                legend_item_list.append(("Shap: " + var2 + "=" + str(level), [r_shap]))

            if actuals:
                if rebase:
                    actbase = data_to_plot_agg.xs(level, level="var2")["actuals"].iloc[
                        base
                    ]
                else:
                    actbase = 1
                xdata_cat = (
                    data_to_plot_agg.xs(level, level="var2")["actuals"]
                    .sort_index()
                    .index
                )
                xdata = np.array([x1_map.get(str(v), 0) for v in xdata_cat])
                ydata = (
                    data_to_plot_agg.xs(level, level="var2")["actuals"].sort_index()
                    / actbase
                )

                lower = min(lower, np.nanmin(ydata))
                upper = max(upper, np.nanmax(ydata))

                source = ColumnDataSource(data=dict(var=xdata, y=ydata))
                r1act = p.line(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    line_width=1.5,
                    name=var2 + "=" + str(level),
                )
                c1act = p.circle(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    fill_color="white",
                    size=8,
                )

                legend_item_list.append(
                    (
                        self.mintypython.config["labels"]["actuals"]
                        + var2
                        + "="
                        + str(level),
                        [r1act, c1act],
                    )
                )

            if gbm_pred:
                if rebase:
                    gbmpredbase = data_to_plot_agg.xs(level, level="var2")[
                        "gbm_pred"
                    ].iloc[base]
                else:
                    gbmpredbase = 1

                xdata_cat = (
                    data_to_plot_agg.xs(level, level="var2")["gbm_pred"]
                    .sort_index()
                    .index
                )
                xdata = np.array([x1_map.get(str(v), 0) for v in xdata_cat])
                ydata = (
                    data_to_plot_agg.xs(level, level="var2")["gbm_pred"].sort_index()
                    / gbmpredbase
                )

                lower = min(lower, np.nanmin(ydata))
                upper = max(upper, np.nanmax(ydata))

                source = ColumnDataSource(data=dict(var=xdata, y=ydata))
                r1gbm = p.line(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    line_width=1.5,
                    name=var2 + "=" + str(level),
                )
                c1gbm = p.triangle(
                    x="var",
                    y="y",
                    source=source,
                    color=colormap[level],
                    fill_color="white",
                    size=8,
                )

                legend_item_list.append(
                    (
                        self.mintypython.config["labels"]["gbm_pred"]
                        + var2
                        + "="
                        + str(level),
                        [r1gbm, c1gbm],
                    )
                )

        p.y_range = Range1d(
            lower - 0.5 * (upper - lower), upper + 0.2 * (upper - lower)
        )

        # plot weight
        w_avg_x = np.array([x1_map.get(str(v), 0) for v in w_avg.index])
        source = ColumnDataSource(data=dict(var=w_avg_x, counts=w_avg))
        p.vbar(
            x="var",
            top="counts",
            width=0.9,
            source=source,
            line_color="white",
            color="orange",
            alpha=0.5,
            y_range_name="Weight",
            name="Weight",
        )

        # title
        p.title.text = plot_title
        p.title.text_font_size = "14pt"

        # legend
        # p.legend.location = 'top_left'
        legend = Legend(items=legend_item_list)
        p.add_layout(legend, "right")
        p.legend.click_policy = "hide"

        # label
        p.xaxis.axis_label = var1
        p.xaxis.axis_label_text_font_size = "13pt"

        # grid background
        p.grid.grid_line_alpha = 0
        p.ygrid.band_fill_color = "grey"
        p.ygrid.band_fill_alpha = 0.1

        p.y_range = Range1d(
            lower - 0.5 * (upper - lower), upper + 0.2 * (upper - lower)
        )

        p.xaxis.major_label_orientation = 3.14 / 2
        self.figure = p

        if save_plot is not None:
            output_file(
                save_plot + var1 + "_x_" + var2 + ".html", title=var1 + "_x_" + var2
            )

        show(p)
        return p
    
    def compare(self, var_name, merged_data, merged_shap_points, **kwargs):
        # hover tips
        T = [("Name", "$name"), ("X", "@var"), ("Value", "$y")]

        merged_data.index = merged_data.index.astype(str)

        x_axis_values = merged_data.index.values
        # Create numeric indices for x-axis (avoids Bokeh FactorRange bugs)
        x_indices = np.arange(len(x_axis_values))
        x_map = {v: i for i, v in enumerate(x_axis_values)}

        w_avg = merged_data["weight@" + kwargs["model_ids"][0]].values

        if "base" not in kwargs.keys():
            base = np.argmax(w_avg)
        else:
            if kwargs["base"] is None:
                base = np.argmax(w_avg)
            else:
                if kwargs["base"] in x_axis_values:
                    base = np.argmax(x_axis_values == kwargs["base"])
                else:
                    sample_values = x_axis_values[:3] if len(x_axis_values) > 3 else x_axis_values
                    print(
                        "Warning: The base level you selected is not in the data. "
                        + "dtype: " + str(x_axis_values.dtype) + ", "
                        + "sample values: " + str(list(sample_values))
                    )
                    base = np.argmax(w_avg)

        # create a new plot with numeric x-range
        p = figure(
            height=kwargs["height"],
            width=kwargs["width"],
            tooltips=T,
            x_range=Range1d(-0.5, len(x_axis_values) - 0.5),
        )
        # Configure x-axis with categorical labels at integer positions
        p.xaxis.ticker = FixedTicker(ticks=list(x_indices))
        p.xaxis.major_label_overrides = {i: str(v) for i, v in enumerate(x_axis_values)}

        # title
        p.title.text = kwargs["plot_name"]
        p.title.text_font_size = "14pt"

        # label
        p.xaxis.axis_label = var_name
        p.xaxis.axis_label_text_font_size = "12.5pt"

        # Setting the second y axis range name and range
        p.extra_y_ranges = {"Weight": Range1d(start=0, end=3.0 * max(w_avg))}

        # Adding the second axis to the plot.
        p.add_layout(LinearAxis(y_range_name="Weight", axis_label="Weight"), "right")

        # set y range
        lower = 1
        upper = 1.000001

        # grid background
        p.grid.grid_line_alpha = 0
        p.ygrid.band_fill_color = "grey"
        p.ygrid.band_fill_alpha = 0.1

        # add renderers
        ######################################
        # plot weight
        source = ColumnDataSource(data=dict(var=x_indices, counts=w_avg))
        r_w = p.vbar(
            x="var",
            top="counts",
            width=0.9,
            source=source,
            line_color="white",
            color=self.mintypython.config["colors"]["weight"],
            alpha=0.5,
            y_range_name="Weight",
            name=self.mintypython.config["labels"]["weight"],
        )
        L_items = [(self.mintypython.config["labels"]["weight"], [r_w])]

        # Colormap
        lbs = []
        for line in ["shap", "glm", "actuals", "glm_pred", "gbm_pred", "shap_points"]:
            if line in kwargs.keys():
                if kwargs[line]:
                    lbs += [line]
        elem = [l + "@" + m for l in lbs for m in kwargs["model_ids"]]
        colormap = dict(zip(elem, cividis(len(elem))))

        # shap points
        if "shap_points" in kwargs.keys():
            if kwargs["shap_points"]:
                for model_id in kwargs["model_ids"]:
                    avg_shap = merged_data["shap@" + model_id]
                    if kwargs["rebase"]:
                        avg_shap_base = avg_shap.iloc[base]
                    else:
                        avg_shap_base = 1

                    merged_shap_points["shap@" + model_id] = (
                        merged_shap_points["shap@" + model_id] / avg_shap_base
                    ).values

                    # Add random jitter to categorical indices (avoids Bokeh jitter() bug)
                    x_vals = merged_shap_points["x_axis"].values
                    x_jittered = np.array([x_map.get(str(v), 0) for v in x_vals]) + np.random.uniform(-0.25, 0.25, len(x_vals))
                    source = ColumnDataSource(
                        data=dict(
                            var=x_jittered,
                            shap=merged_shap_points["shap@" + model_id].values,
                        )
                    )
                    r_shap = p.circle(
                        x="var",
                        y="shap",
                        source=source,
                        size=2,
                        alpha=0.8,
                        name="shap@" + model_id,
                        color=colormap["shap_points@" + model_id],
                    )
                    L_items.append(
                        (
                            self.mintypython.config["labels"]["shap_points"]
                            + " "
                            + model_id,
                            [r_shap],
                        )
                    )

        # Lines
        for line in ["shap", "glm", "actuals", "glm_pred", "gbm_pred"]:
            if line in kwargs.keys() and kwargs[line]:
                for model_id in kwargs["model_ids"]:
                    name = line + "@" + model_id
                    avg = merged_data[name]
                    if kwargs["rebase"]:
                        avgbase = avg.iloc[base]
                    else:
                        avgbase = 1

                    lower = min(lower, np.nanmin(avg / avgbase))
                    upper = max(upper, np.nanmax(avg / avgbase))
                    source = ColumnDataSource(
                        data=dict(var=x_indices, y=avg / avgbase)
                    )
                    r_avg = p.line(
                        x="var",
                        y="y",
                        source=source,
                        line_width=self.mintypython.config["line_width"][line],
                        name=name,
                        color=colormap[name],
                    )
                    c_avg = p.diamond(
                        x="var",
                        y="y",
                        source=source,
                        size=10,
                        alpha=0.9,
                        name=name,
                        color=colormap[name],
                    )
                    L_items.append(
                        (
                            self.mintypython.config["labels"][line] + " " + model_id,
                            [r_avg, c_avg],
                        )
                    )

        # legend
        legend = Legend(items=L_items)
        p.add_layout(legend, "right")
        p.legend.click_policy = "hide"

        if kwargs["y_axis_max"] != None:
            upper = kwargs["y_axis_max"]

        if kwargs["y_axis_min"] != None:
            lower = kwargs["y_axis_min"]

        p.y_range = Range1d(
            lower - 0.5 * (upper - lower), upper + 0.2 * (upper - lower)
        )

        p.xaxis.major_label_orientation = 3.14 / 2
        self.figure = p

        if "save_plot" in kwargs.keys() and kwargs["save_plot"] is not None:
            output_file(kwargs["save_plot"] + var_name + ".html", title=var_name)

        show(p)
        return p
