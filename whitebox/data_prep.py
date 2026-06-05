from .utils import *
import shap
import os
import pandas as pd
from .exceptions import JoinShapNotFoundError


class DataPrep:
    def __init__(self, whitebox):
        self.whitebox = whitebox

    def process_categoricals(self, columns=None):
        """Deprecated shim. Categorical encoding is now owned by ``Whitebox.encoder``
        and runs automatically at construction (``Encoder.auto_encode``). Kept for
        backward compatibility; delegates to the encoder.
        """
        return self.whitebox.encoder.auto_encode()

    def prep_shap_values(self, out_file=""):
        """
        Prepare shapley values and stores in the self.shap_df attribute
        :param out_file: string pickle file path of a pandas dataframe containing the shapley values, if the file not exists it creates calculates
        the shapley values and save as out_file
        """
        # computing shapley values
        self.whitebox.explainer = shap.TreeExplainer(self.whitebox.model)

        if os.path.exists(out_file) == False:
            # Build feature data for SHAP, using encoder-managed _encoded columns.
            shap_data = pd.DataFrame()
            for col in self.whitebox.feature_names:
                shap_data[col] = self.whitebox.encoder.model_input_column(col)

            shap_values = self.whitebox.explainer.shap_values(shap_data)
            shap_values = self.whitebox.link_fn(shap_values)
            self.whitebox.shap_df = pd.DataFrame(
                shap_values,
                columns=self.whitebox.feature_names,  # Keep original names
                index=shap_data.index,
            )
            if out_file != "":
                self.whitebox.shap_df.to_pickle(out_file)
                if self.whitebox.verbose:
                    print("Shapley values for all features saved to :" + out_file)
        else:
            print("Reading shapley from:" + out_file)
            self.whitebox.shap_df = pd.read_pickle(out_file)
    
    def _derived_shap_series(self, name):
        """Synthesize SHAP for a derived variable from its source feature(s):
        a single source uses that feature's SHAP; multiple sources sum row-wise.
        Grouping along the derived levels happens later via groupby on the x-axis.
        """
        if self.whitebox.shap_df is None:
            self.prep_shap_values()
        sources = self.whitebox.encoder.source_features(name)
        if len(sources) == 1:
            return self.whitebox.shap_df[sources[0]]
        return self.whitebox.shap_df[sources].sum(axis=1)

    def prep_univariate_data(self, var_name, kwargs):
        """
        Creates univariate data to plot
        """
        if kwargs["shap"] is not None and (
            var_name not in self.whitebox.plottable_variables()
            and kwargs["joinshaps"] is None
            and kwargs["shap"]
        ):
            raise ValueError(
                "The variable "
                + var_name
                + " is not in the model to plot shaps, the availables variables are:"
                + str(self.whitebox.feature_names)
            )
        shap_data = pd.DataFrame()
        data_to_plot = pd.DataFrame()
        data_to_plot_agg = pd.DataFrame()
        group_by_var = var_name

        glm_data_to_plot = None
        if "glm" in kwargs.keys() and kwargs["glm"]:
            if kwargs["glmindic_cols"] is None:
                if self.whitebox.glm_df is not None:
                    # Apply variable name mapping if provided
                    if self.whitebox.glm_var_map and var_name in self.whitebox.glm_var_map:
                        glm_col = self.whitebox.glm_var_map[var_name]
                    else:
                        glm_col = var_name
                    # Check if the GLM column exists in glm_df
                    if glm_col in self.whitebox.glm_df.columns:
                        kwargs["glmindic_cols"] = [glm_col]
                    else:
                        print(
                            f"The variable '{glm_col}' is not in glm_df. "
                            f"Available columns are: {list(self.whitebox.glm_df.columns)}. "
                            f"Use glm_var_map to map GBM variable names to GLM column names."
                        )
                        kwargs["glm"] = False
                else:
                    print(
                        "No GLM data to display. Provide glm_df parameter in the Whitebox constructor."
                    )
                    kwargs["glm"] = False

            # Process glmindic_cols (handles both explicit and auto-set cases)
            if kwargs["glmindic_cols"] is not None:
                glm_data_to_plot_pts = self.whitebox.glm_df[
                    kwargs["glmindic_cols"]
                ].prod(axis=1)
                data_to_plot["glm"] = glm_data_to_plot_pts
                data_to_plot["glm_wtg"] = (
                    glm_data_to_plot_pts
                    * self.whitebox.data[self.whitebox.weight_col]
                )

        data_to_plot["x_axis"] = self._bin_if_numeric(
            group_by_var,
            kwargs["nlevels"],
            kwargs["start"],
            kwargs["finish"],
            kwargs["stepsize"],
            kwargs["percentile_start"],
            kwargs["percentile_finish"],
            kwargs["infinity_higher"],
            kwargs["infinity_lower"],
            glm_data_to_plot=glm_data_to_plot,
        )

        data_to_plot["weight"] = self.whitebox.data[self.whitebox.weight_col]

        data_to_plot_agg["weight"] = data_to_plot.groupby("x_axis", dropna=False)[
            "weight"
        ].sum()

        if glm_data_to_plot is not None:
            data_to_plot_agg["glm"] = data_to_plot_agg.index.map(
                glm_data_to_plot
            ).astype(float)

        if kwargs["glmindic_cols"] is not None:
            data_to_plot_agg["glm"] = (
                data_to_plot.groupby("x_axis", dropna=False)["glm_wtg"].sum()
                / data_to_plot_agg["weight"]
            )

        # Apply fac_mapping to show labels with encoded values: "Label(code)"
        if self.whitebox.fac_mapping != None and group_by_var in self.whitebox.fac_mapping:
            mapping = self.whitebox.fac_mapping[group_by_var]
            # Create combined label: "Label(code)" supporting both numeric codes and string labels
            combined_mapping = {}
            for code, label in mapping.items():
                combined_label = f"{label}({code})"
                combined_mapping[code] = combined_label        # numeric code (0, 1, 2)
                combined_mapping[str(code)] = combined_label  # string code ("0", "1", "2")
                combined_mapping[label] = combined_label      # string label ("North", "South")
            data_to_plot["x_axis"] = data_to_plot["x_axis"].map(
                lambda x: combined_mapping.get(x, str(x))
            )
            data_to_plot_agg.index = pd.Series(data_to_plot_agg.index).map(
                lambda x: combined_mapping.get(x, str(x))
            )

        if "actuals" in kwargs.keys() and kwargs["actuals"]:
            if self.whitebox.actuals_col is None:
                print(
                    "If you want to plot actuals you must provide a actuals_col in the Whitebox call e.g Whitebox(actuals_col='your actuals column')"
                )
                kwargs["actuals"] = False
            else:
                data_to_plot["actuals"] = self.whitebox.data[
                    self.whitebox.actuals_col
                ]
                data_to_plot_agg["actuals"] = (
                    data_to_plot.groupby("x_axis", dropna=False)["actuals"].sum()
                    * np.sign(data_to_plot_agg["weight"])
                    / data_to_plot_agg["weight"]
                )

        if (
            ("shap" in kwargs.keys() and kwargs["shap"])
            or ("shap_points" in kwargs.keys() and kwargs["shap_points"])
            or ("shap_sd" in kwargs.keys() and kwargs["shap_sd"])
        ):
            if self.whitebox.shap_df is None:
                self.prep_shap_values()
            # weighted shap
            if kwargs["joinshaps"] is None or len(kwargs["joinshaps"]) == 0:
                if self.whitebox.encoder.is_derived(var_name):
                    shap_vals = self._derived_shap_series(var_name).values
                else:
                    shap_vals = self.whitebox.shap_df[var_name].values
            else:
                valid_shaps = self.get_valid_shaps(kwargs["joinshaps"])
                shap_vals = self.whitebox.shap_df[valid_shaps].prod(axis=1).values

            data_to_plot["shap_wtg"] = (
                shap_vals * self.whitebox.data[self.whitebox.weight_col].values
            )

            data_to_plot_agg["shap"] = (
                data_to_plot.groupby("x_axis", dropna=False)["shap_wtg"].sum()
                / data_to_plot_agg["weight"]
            )
            if "shap_sd" in kwargs.keys() and kwargs["shap_sd"]:
                data_to_plot["shap2_wtg"] = (
                    shap_vals
                    * shap_vals
                    * self.whitebox.data[self.whitebox.weight_col].values
                )
                data_to_plot_agg["shap_sd"] = np.sqrt(
                    data_to_plot.groupby("x_axis", dropna=False)["shap2_wtg"].sum()
                    / data_to_plot_agg["weight"]
                    - data_to_plot_agg["shap"] * data_to_plot_agg["shap"]
                ).fillna(0)

        if "shap_points" in kwargs.keys() and kwargs["shap_points"]:
            np.random.seed(kwargs["shap_points_seed"])
            data_to_plot["shap"] = shap_vals
            plot_index = np.random.choice(
                data_to_plot.index,
                size=min(kwargs["n_shap_points"], len(data_to_plot)),
                replace=False,
            )
            shap_data = data_to_plot.loc[plot_index, ["shap", "x_axis"]]
        
        if "glm_pred" in kwargs.keys() and kwargs["glm_pred"]:
            if self.whitebox.glm_preds_col is None:
                print(
                    "To plot GLM predictions, provide glm_preds_col parameter "
                    "in the Whitebox constructor with the column name containing "
                    "pre-computed GLM predictions."
                )
                kwargs["glm_pred"] = False
            else:
                data_to_plot["glm_pred_wgt"] = (
                    self.whitebox.data[self.whitebox.glm_preds_col]
                    * data_to_plot["weight"]
                )

                data_to_plot_agg["glm_pred_wgt"] = data_to_plot.groupby(
                    "x_axis", dropna=False
                )["glm_pred_wgt"].sum()
                data_to_plot_agg["glm_pred"] = (
                    data_to_plot_agg["glm_pred_wgt"] / data_to_plot_agg["weight"]
                )

                if "glm_ci" in kwargs.keys() and kwargs["glm_ci"]:
                    if (
                        self.whitebox.scale_parameter is not None
                        and self.whitebox._ci_fn is not None
                    ):
                        data_to_plot_agg["stdev"] = np.sqrt(
                            self.whitebox._ci_fn(
                                data_to_plot_agg["glm_pred"],
                                data_to_plot_agg["weight"],
                            )
                        )
                        # central limit theorem
                        data_to_plot_agg["glm_+2sigma"] = (
                            data_to_plot_agg["glm_pred"]
                            + kwargs["ci_z"] * data_to_plot_agg["stdev"]
                        )
                        data_to_plot_agg["glm_-2sigma"] = (
                            data_to_plot_agg["glm_pred"]
                            - kwargs["ci_z"] * data_to_plot_agg["stdev"]
                        )
                    else:
                        kwargs["glm_ci"] = False
                        print(
                            "To plot confidence intervals you must pass a scale_parameter and a link_fn to the class construction"
                        )
        if "gbm_pred" in kwargs.keys():
            # If the prediction were already calculated donot calculate again
            if self.whitebox._gbm_predictions is None:
                self.whitebox._gbm_predictions = self.whitebox.model.predict(
                    self.whitebox.x_data
                )
            data_to_plot["gbm_pred_wgt"] = (
                self.whitebox._gbm_predictions * data_to_plot["weight"].values
            )
            data_to_plot_agg["gbm_pred_wgt"] = data_to_plot.groupby(
                "x_axis", dropna=False
            )["gbm_pred_wgt"].sum()
            data_to_plot_agg["gbm_pred"] = (
                data_to_plot_agg["gbm_pred_wgt"] / data_to_plot_agg["weight"]
            )

        return {"shap_points": shap_data, "agg_data": data_to_plot_agg}
    

    def create_bandings(
        self,
        data,
        start=None,
        finish=None,
        stepsize=None,
        nlevels=None,
        percentile_start=10,
        percentile_finish=90,
        infinity_higher=True,
        infinity_lower=True,
        supress_warnings=False,
    ):
        """
        This function creates a banding for the variables to be used in emblem

            :param start: where your banding will start
            :type start: float

            :param finish: where your banding will finish
            :type finish: float

            :param stepsize: the size of steps between the bands
            :type stepsize: float

            :param nlevels: number of levels
            :type nlevels: float

            :param percentile_start: if start is not defined it select this percentile as start
            :type percentile_start: float

            :param percentile_finish: if finish is not defined it select this percentile as finish
            :type percentile_finish: float

            :param infinity_higher:  includes a infinite bound after the finish
            :type infinity_higher: Boolean

            :param infinity_lower:   includes a minus infinite bound before the start
            :type infinity_lower: Boolean

            :param supress_warnings: supress emblem 255 warning
            :type supress_warnings: None
        """

        if start != None and finish != None and stepsize != None and nlevels != None:
            raise AttributeError(
                "You have to choose between stepsize and nlevels when start and finish are defined."
            )
        if nlevels != None and supress_warnings == False:
            if nlevels > 255:
                import warnings

                warnings.warn(
                    "Is good to remember that Emblem suports up to 255 levels:",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if start == None:
            if stepsize != None and nlevels != None:
                if finish == None:
                    start = np.nanpercentile(data, 50) - stepsize * (nlevels / 2)
                else:
                    start = finish - stepsize * nlevels
            else:
                start = np.nanpercentile(data, percentile_start)

        if finish == None:
            if stepsize != None and nlevels != None:
                if start == None:
                    finish = np.nanpercentile(data, 50) + stepsize * (nlevels / 2)
                else:
                    finish = start + stepsize * nlevels
            else:
                finish = np.nanpercentile(data, percentile_finish)

        if stepsize == None:
            if nlevels == None:
                nlevels = 10
            stepsize = (finish - start) / (nlevels - 1)

        cut_points = [start]
        if stepsize != 0:
            cut_points = list(np.arange(start, finish + stepsize, stepsize))

        resp = pd.cut(
            data,
            bins=[-10e10000] * infinity_lower
            + cut_points
            + [10e10000] * infinity_higher,
        )
        if resp.hasnans:
            resp = resp.cat.add_categories("nan").fillna("nan")
        resp = pd.Series(resp).cat.rename_categories(
            lambda x: str.replace(str(x), "(", "")
        )
        return resp.astype(str)
    
    def _partial_dependency(self, bst, X, w, f_id, features, N=255):
        """
            Calculates partial dependency function values for a feature.

        PARAMETERS
        ----------
        \t    bst (object): xgboost model object.
        \t    X (pd dataframe object): Data for calculating.
        \t    w (array like): Weight.
        \t    f_id (int): Position index of the interested feature.
        \t    features (list): Names of the features used in the model, usually X.columns.
        \t    N (int): Maximum number of levels of the feature values to use for calculating.
        """
        X_temp = X.copy()
        n_level = X_temp.iloc[:, f_id].nunique()

        x = X_temp.iloc[:, f_id]
        if np.isnan(x).any():
            print(
                features[f_id],
                " contains missing value, dropped when calculating pd values.",
            )
            x = X_temp.iloc[:, f_id].dropna()

        if n_level <= N:
            grid = np.sort(x.unique())
        else:
            grid = np.quantile(x, np.linspace(0, 1, N))

        y_pred = np.zeros(len(grid))

        for i, val in enumerate(grid):
            X_temp.iloc[:, f_id] = val
            data = self.whitebox.xgb.DMatrix(X_temp, feature_names=features)
            y_pred[i] = np.sum(bst.predict(data, output_margin=False) * w) / w.sum()

        return grid, y_pred

    def _bin_if_numeric(
        self,
        group_by_var,
        nlevel,
        start,
        finish,
        stepsize,
        percentile_start,
        percentile_finish,
        infinity_higher=True,
        infinity_lower=True,
        glm_data_to_plot=None,
    ):
        banded_var = None
        # Identify if the variable is numeric or categorical.
        # Check variable type to decide if we need to create bandings or not.
        if pd.api.types.is_numeric_dtype(self.whitebox.data[group_by_var]):
            nlevels = len(self.whitebox.data[group_by_var].unique())
            if (nlevels > 200 and glm_data_to_plot == None) or (
                nlevel != None or start != None or finish != None or stepsize != None
            ):
                # if is numeric and has a lot of levels bin it
                banded_var = self.create_bandings(
                    self.whitebox.data[group_by_var],
                    nlevels=nlevel,
                    start=start,
                    finish=finish,
                    stepsize=stepsize,
                    percentile_start=percentile_start,
                    percentile_finish=percentile_finish,
                    infinity_higher=infinity_higher,
                    infinity_lower=infinity_lower,
                )
            else:
                banded_var = self.whitebox.data[group_by_var]
        elif (
            pd.api.types.is_categorical_dtype(self.whitebox.data[group_by_var])
            and not self.whitebox.data[group_by_var].cat.ordered
        ):
            # if is category gets the category name
            banded_var = self.whitebox.data[group_by_var].astype(str)
        else:
            # Just get the data
            banded_var = self.whitebox.data[group_by_var]

        # Apply fac_mapping to show labels with encoded values: "Label(code)"
        if (
            self.whitebox.fac_mapping is not None
            and group_by_var in self.whitebox.fac_mapping
        ):
            mapping = self.whitebox.fac_mapping[group_by_var]
            # Create combined label: "Label(code)" supporting both numeric codes and string labels
            combined_mapping = {}
            for code, label in mapping.items():
                combined_label = f"{label}({code})"
                combined_mapping[code] = combined_label        # numeric code (0, 1, 2)
                combined_mapping[str(code)] = combined_label  # string code ("0", "1", "2")
                combined_mapping[label] = combined_label      # string label ("North", "South")
            banded_var = banded_var.map(
                lambda x: combined_mapping.get(x, str(x))
            )

        return banded_var.astype(str)

    def prep_bivariate_data(
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
        plottable = self.whitebox.plottable_variables()
        if var1 not in plottable:
            raise ValueError(
                "The variable "
                + var1
                + " is not available to plot, the available variables are "
                + str(plottable)
            )
        if var2 not in plottable:
            raise ValueError(
                "The variable "
                + var2
                + " is not available to plot, the available variables are "
                + str(plottable)
            )

        data_to_plot = pd.DataFrame()

        data_to_plot["var1"] = self._bin_if_numeric(
            var1,
            nlevels_var1,
            start_var1,
            finish_var1,
            stepsize_var1,
            percentile_start_var1,
            percentile_finish_var1,
        )
        data_to_plot["var2"] = self._bin_if_numeric(
            var2,
            nlevels_var2,
            start_var2,
            finish_var2,
            stepsize_var2,
            percentile_start_var2,
            percentile_finish_var2,
        )

        data_to_plot["weight"] = self.whitebox.data[self.whitebox.weight_col]

        data_to_plot = data_to_plot.reset_index()

        if shap:
            if self.whitebox.shap_df is None:
                self.prep_shap_values()
            if joinshaps is not None:
                data_to_plot["shap"] = (
                    self.whitebox.shap_df[joinshaps]
                    .prod(axis=1)
                    .reset_index(drop=True)
                )
                data_to_plot["shap_wgt"] = (
                    self.whitebox.shap_df[joinshaps].prod(axis=1)
                    * data_to_plot["weight"]
                ).reset_index(drop=True)
            else:
                if self.whitebox.encoder.is_derived(var1):
                    var1_shap = self._derived_shap_series(var1)
                else:
                    var1_shap = self.whitebox.shap_df[var1]
                data_to_plot["shap"] = var1_shap.reset_index(drop=True)
                data_to_plot["shap_wgt"] = (
                    var1_shap.values * data_to_plot["weight"].values
                )
        
        if glm:
            if glmindic_cols is not None:
                data_to_plot["glm_wgt"] = (
                    self.whitebox.glm_df[glmindic_cols].prod(axis=1)
                    * data_to_plot["weight"]
                ).reset_index(drop=True)
            else:
                data_to_plot["glm_wgt"] = (
                    self.whitebox.glm_df[var1] * data_to_plot["weight"]
                ).reset_index(drop=True)

        if actuals:
            act = pd.DataFrame(
                self.whitebox.data[self.whitebox.actuals_col]
            ).reset_index(drop=True)
            data_to_plot["actuals"] = act

        data_to_plot_agg = data_to_plot.groupby(["var1", "var2"], dropna=False).sum()

        if shap:
            data_to_plot_agg["shap_avg"] = (
                data_to_plot_agg["shap_wgt"] / data_to_plot_agg["weight"]
            )
        if glm:
            data_to_plot_agg["glm_avg"] = (
                data_to_plot_agg["glm_wgt"] / data_to_plot_agg["weight"]
            )

        if actuals:
            data_to_plot_agg["actuals"] = (
                data_to_plot_agg["actuals"] / data_to_plot_agg["weight"]
            )

        if gbm_pred:
            # If the prediction were already calculated donot calculate again
            if self.whitebox._gbm_predictions is None:
                self.whitebox._gbm_predictions = self.whitebox.model.predict(
                    self.whitebox.x_data
                )
            data_to_plot["gbm_pred_wgt"] = (
                self.whitebox._gbm_predictions * data_to_plot["weight"]
            )
            data_to_plot_agg["gbm_pred_wgt"] = data_to_plot.groupby(
                ["var1", "var2"], dropna=False
            )["gbm_pred_wgt"].sum()
            data_to_plot_agg["gbm_pred"] = (
                data_to_plot_agg["gbm_pred_wgt"] / data_to_plot_agg["weight"]
            )
        return data_to_plot, data_to_plot_agg
    
    def prep_compare_data(self, var_name, wblist, **kwargs):
        model_ids = kwargs["model_ids"]
        if len(np.unique(model_ids)) != len(model_ids):
            hp = np.unique(model_ids, return_counts=True)
            raise ValueError(
                "Please make sure you do not have duplicated items on model_names:"
                + str(hp[0])
                + " counts:"
                + str(hp[1])
            )

        # Include the calling model (self.whitebox) as the first model
        all_models = [self.whitebox] + wblist
        n = len(all_models)
        plot_data = list()
        agg_data = list()
        shap_points = list()

        for i in range(0, n):
            plot_data += [all_models[i].DataPrep.prep_univariate_data(var_name, kwargs)]
            agg_data += [plot_data[i]["agg_data"]]
            shap_points += [plot_data[i]["shap_points"].reset_index(drop=True)]

        # Merge data from Whitebox objects
        merged_data = agg_data[0].copy()
        merged_data.columns = [col + "@" + model_ids[0] for col in merged_data.columns]
        merged_shap_points = shap_points[0].copy()
        merged_shap_points.columns = [
            col + ("@" + model_ids[0]) * (col != "x_axis")
            for col in merged_shap_points.columns
        ]

        for i in range(1, n):
            agg_data[i] = agg_data[i].rename(
                columns={c: c + "@" + model_ids[i] for c in agg_data[i].columns}
            )
            merged_data = merged_data.merge(
                agg_data[i], left_index=True, right_index=True, how='outer'
            )
            if kwargs["shap_points"]:
                shap_points[i] = shap_points[i].rename(
                    columns={c: c + "@" + model_ids[i] for c in shap_points[i].columns}
                )
                merged_shap_points = merged_shap_points.merge(
                    shap_points[i], on="x_axis", left_index=True, right_index=True
                )
            else:
                merged_shap_points = None

        if kwargs["shap_points"]:
            merged_shap_points.x_axis = merged_shap_points.x_axis.astype(str)

        return merged_data, merged_shap_points

    def get_valid_shaps(self, joinshaps):
        valid_shaps = set(joinshaps).intersection(set(self.whitebox.shap_df.columns))

        if len(valid_shaps) < len(joinshaps):
            missing = set(joinshaps) - valid_shaps
            print(
                str(missing)
                + " do not exist in in the shap_df, their values will not be included in the calculation"
            )

        return list(valid_shaps)