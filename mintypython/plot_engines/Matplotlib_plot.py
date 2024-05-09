from .Plot_interface import Plot_interface
from ..libs import *
from matplotlib import pyplot as plt
import seaborn as sns
from collections import OrderedDict


class Matplotlib_plot(Plot_interface):
    def __init__(self, mintypython):
        self.mintypython = mintypython
        pass

    def univariate_plot(self, kwargs, plot_data, var_name):
        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(12)

        plt.xticks(rotation=90)
        ax2 = ax.twinx()

        x_axis_values = plot_data["agg_data"].index.values
        x_axis_values = x_axis_values.astype(str)
        w_avg = plot_data["agg_data"]["weight"].values

        ax2.set_ylabel("Weight")

        # set y range
        lower = 1
        upper = 1.000001

        if "base" not in kwargs.keys():
            base = np.argmax(w_avg)
        else:
            if kwargs["base"] is None:
                base = np.argmax(w_avg)
            else:
                if kwargs["base"] in x_axis_values:
                    base = np.argmax(x_axis_values == kwargs["base"])
                else:
                    print(
                        "Warning: The base level you selected is not in:"
                        + str(x_axis_values)
                    )

        # ax2.bar(x_axis_values,w_avg,label=self.mintypython.config["labels"]["weight"], color=self.mintypython.config["colors"]["weight"])
        sns.barplot(
            x=x_axis_values,
            y=w_avg,
            label=self.mintypython.config["labels"]["weight"],
            color=self.mintypython.config["colors"]["weight"],
            ax=ax2,
        )

        # shap points
        if "shap_points" in kwargs.keys() and kwargs["shap_points"]:
            avg_shap = plot_data["agg_data"]["shap"]
            plot_data["shap_points"] = plot_data["shap_points"].sort_values(by="x_axis")
            x_shap = plot_data["shap_points"]["x_axis"]
            x_shap = x_shap.astype(str)
            if kwargs["rebase"]:
                plot_data["shap_points"]["shap"] = (
                    plot_data["shap_points"]["shap"] / avg_shap.iloc[base]
                ).values
            else:
                plot_data["shap_points"]["shap"] = plot_data["shap_points"][
                    "shap"
                ].values

            # sns.stripplot(x = np.concatenate((x_axis_values, x_shap.values)),
            #               y = np.concatenate((np.ones_like(x_axis_values),plot_data["shap_points"]["shap"].values)),
            #               ax = ax,
            #               # size=3,
            #               label=self.mintypython.config["labels"]["shap_points"],
            #               jitter=0.20,
            #               color=self.mintypython.config["colors"]["shap_points"],
            #               size = np.concatenate((np.zeros_like(x_axis_values,dtype=float),np.ones_like(x_shap.values,dtype=float))),
            #               alpha = 0.5)
            sns.stripplot(
                x=x_shap.values,
                y=plot_data["shap_points"]["shap"].values,
                order=x_axis_values,
                ax=ax,
                label=self.mintypython.config["labels"]["shap_points"],
                jitter=0.20,
                color=self.mintypython.config["colors"]["shap_points"],
                size=3,
            )

        if "shap_sd" in kwargs.keys() and kwargs["shap_sd"]:
            avg_shap = plot_data["agg_data"]["shap"]
            shap_sd = plot_data["agg_data"]["shap_sd"]
            if kwargs["rebase"]:
                avgbase = avg_shap.iloc[base]
                if avgbase == 0:
                    print("Warning: unable to rebase shap sd")
                    avgbase = 1
            else:
                avgbase = 1
            ax.errorbar(
                x_axis_values,
                avg_shap / avgbase,
                yerr=shap_sd / avgbase,
                linewidth=0,
                ecolor=self.mintypython.config["colors"]["shap_sd"],
                alpha=0.3,
                elinewidth=self.mintypython.config["line_width"]["shap_sd"] * 5,
                label=self.mintypython.config["labels"]["shap_sd"],
            )

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

                lower = min(lower, min(avg / avgbase))
                upper = max(upper, max(avg / avgbase))
                # ax.plot(x_axis_values, avg / avgbase,
                #     linewidth=self.mintypython.config["line_width"][line],
                #     label=self.mintypython.config["labels"][line],
                #     color=self.mintypython.config["colors"][line],
                #     marker=self.mintypython.config["marker"][line]
                # )
                sns.lineplot(
                    x=x_axis_values,
                    y=avg / avgbase,
                    linewidth=self.mintypython.config["line_width"][line],
                    label=self.mintypython.config["labels"][line],
                    color=self.mintypython.config["colors"][line],
                    marker=self.mintypython.config["marker"][line],
                    ax=ax,
                )

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.set_frame_on(False)

        handles_1, labels_1 = ax.get_legend_handles_labels()
        handles_2, labels_2 = ax2.get_legend_handles_labels()

        handles = handles_2 + handles_1
        labels = labels_2 + labels_1

        by_label = OrderedDict(zip(labels, handles))

        if kwargs["y_axis_min"] is not None:
            ax.set_ylim(bottom=kwargs["y_axis_min"])
        if kwargs["y_axis_max"] is not None:
            ax.set_ylim(top=kwargs["y_axis_max"])

        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            bbox_to_anchor=(1.05, 0.9),
            fontsize=12,
        )

        ax.set_xlabel(var_name, fontsize=14)
        plt.title(kwargs["plot_name"], loc="left", fontsize=16, fontweight="bold")

        plt.show()
        return fig
    
    def bivariate_plot(
        self,
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
    ):
        pass