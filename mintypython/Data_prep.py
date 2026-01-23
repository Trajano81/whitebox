from .libs import *
import shap
import os
import pandas as pd
from .exceptions import JoinShapNotFoundError


class Data_prep:
    def __init__(self, mintypython):
        self.mintypython = mintypython

    def prep_glm_df(self):
        glm_data = self.mintypython.data
        mdl_export = self.mintypython.emb_mdl

        block = glm_data.copy()
        concatenator = "_xOx_"
        export_dict = mdl_export
        block_cols = list(block)
        glm_df = pd.DataFrame()

        base = export_dict.pop("Base", 0.0)
        glm_df["Base"] = np.repeat(base, len(block))

        for key, val in export_dict.items():
            if key in block_cols:
                glm_df[key] = block[key].map(val)
            else:
                problems = [
                    col for col in key.split(concatenator) if col not in list(block)
                ]
                if problems:
                    prob_key = key.replace(concatenator, "_x_")
                    print(
                        f"{prob_key} cannot be created from data. Accordingly, the prediction will "
                        f"not include {prob_key} \n"
                    )
                else:
                    intx = tuple(key.split(concatenator))
                    if len(intx) == 2:
                        v1, v2 = intx
                        srs = block[v1].map(str) + "-" + block[v2].map(str)
                    elif len(intx) == 3:
                        v1, v2, v3 = intx
                        srs = (
                            block[v1].map(str)
                            + "-"
                            + block[v2].map(str)
                            + "-"
                            + block[v3].map(str)
                        )
                    else:
                        raise ValueError(
                            "There is likely a 4-way interaction. That is not supported"
                        )
                    glm_df[key] = srs.map(val)

        self.mintypython.glm_df = glm_df

    def prep_shap_values(self, out_file=""):
        """
        Prepare shapley values and stores in the self.shap_df attribute
        :param out_file: string pickle file path of a pandas dataframe containing the shapley values, if the file not exists it creates calculates
        the shapley values and save as out_file
        """
        # computing shapley values
        self.mintypython.explainer = shap.TreeExplainer(self.mintypython.model)

        if os.path.exists(out_file) == False:
            shap_values = self.mintypython.explainer.shap_values(
                self.mintypython.data[self.mintypython.feature_names]
            )
            shap_values = self.mintypython.link_fn(shap_values)
            self.mintypython.shap_df = pd.DataFrame(
                shap_values,
                columns=self.mintypython.feature_names,
                index=self.mintypython.data[self.mintypython.feature_names].index,
            )
            if out_file != "":
                self.mintypython.shap_df.to_pickle(out_file)
                if self.mintypython.verbose:
                    print("Shapley values for all features saved to :" + out_file)
        else:
            print("Reading shapley from:" + out_file)
            self.mintypython.shap_df = pd.read_pickle(out_file)
    
    def prep_univariate_data(self, var_name, kwargs):
        """
        Creates univariate data to plot
        """
        if kwargs["shap"] is not None and (
            var_name not in self.mintypython.feature_names
            and kwargs["joinshaps"] is None
            and kwargs["shap"]
        ):
            raise ValueError(
                "The variable "
                + var_name
                + " is not in the model to plot shaps, the availables variables are:"
                + str(self.mintypython.feature_names)
            )
        shap_data = pd.DataFrame()
        data_to_plot = pd.DataFrame()
        data_to_plot_agg = pd.DataFrame()
        group_by_var = var_name

        glm_data_to_plot = None
        if "glm" in kwargs.keys() and kwargs["glm"]:
            if kwargs["glmindic_cols"] is None:
                # Check if emblem model is is available.
                if self.mintypython.emb_mdl != None:
                    # If the GLM vars are different than GBM but they are linked check if a variable map existis and variable is in the map and uses the glm variable to summarize shap.
                    if self.mintypython.emb_gbm_map != None:
                        if var_name in self.mintypython.emb_gbm_map.keys():
                            group_by_var = self.mintypython.emb_gbm_map[var_name]
                    # Check if var is in the glm
                    if group_by_var in self.mintypython.emb_mdl.keys():
                        # glm_data_to_plot=dict((k, self.mintypython.link_fn(v)) for k, v in self.mintypython.emb_mdl[group_by_var].items())
                        glm_data_to_plot = self.mintypython.emb_mdl[group_by_var]
                    else:
                        print(
                            "The variable "
                            + group_by_var
                            + "is not in the GLM model, if the variable is in the GLM but with a different name, use the parameter emb_gbm_map to map the variables the variable in the glm are:"
                            + str(self.mintypython.emb_mdl.keys())
                        )
                        kwargs["glm"] = False
                else:
                    print(
                        "No GLM model to display, if you want to display model relativities, provide a model export emb_model_export on the mintypython construction"
                    )
            else:
                glm_data_to_plot_pts = self.mintypython.glm_df[
                    kwargs["glmindic_cols"]
                ].prod(axis=1)
                data_to_plot["glm"] = glm_data_to_plot_pts
                data_to_plot["glm_wtg"] = (
                    glm_data_to_plot_pts
                    * self.mintypython.data[self.mintypython.weight_col]
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

        data_to_plot["weight"] = self.mintypython.data[self.mintypython.weight_col]

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

        if self.mintypython.fac_mapping != None and group_by_var in self.mintypython.fac_mapping:
            data_to_plot["x_axis"] = data_to_plot["x_axis"].map(
                self.mintypython.fac_mapping[group_by_var]
            )
            data_to_plot_agg.index = pd.Series(data_to_plot_agg.index).map(
                self.mintypython.fac_mapping[group_by_var]
            )

        if "actuals" in kwargs.keys() and kwargs["actuals"]:
            if self.mintypython.actuals_col is None:
                print(
                    "If you want to plot actuals you must provide a actuals_col in the mintypython call e.g mintypython(actuals_col='your actuals column')"
                )
                kwargs["actuals"] = False
            else:
                data_to_plot["actuals"] = self.mintypython.data[
                    self.mintypython.actuals_col
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
            if self.mintypython.shap_df is None:
                self.prep_shap_values()
            # weighted shap
            if kwargs["joinshaps"] is None or len(kwargs["joinshaps"]) == 0:
                shap_vals = self.mintypython.shap_df[var_name].values
            else:
                valid_shaps = self.get_valid_shaps(kwargs["joinshaps"])
                shap_vals = self.mintypython.shap_df[valid_shaps].prod(axis=1).values

            data_to_plot["shap_wtg"] = (
                shap_vals * self.mintypython.data[self.mintypython.weight_col].values
            )

            data_to_plot_agg["shap"] = (
                data_to_plot.groupby("x_axis", dropna=False)["shap_wtg"].sum()
                / data_to_plot_agg["weight"]
            )
            if "shap_sd" in kwargs.keys() and kwargs["shap_sd"]:
                data_to_plot["shap2_wtg"] = (
                    shap_vals
                    * shap_vals
                    * self.mintypython.data[self.mintypython.weight_col].values
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
            if self.mintypython.glm_preds_col is None:
                # If the prediction were already calculated donot calculate again
                if self.mintypython._glm_predictions is None:
                    # Scorepyon
                    scr_data = self.mintypython.data.copy()
                    for col in scr_data:
                        if str(scr_data[col].dtype) == "category":
                            scr_data[col] = scr_data[col].astype(str)
                    nm = self.mintypython.emb_model_export.split("/")
                    model_name = nm[len(nm) - 1].split(".")[0]
                    if self.mintypython.scorepyon_str:
                        for col in scr_data:
                            scr_data[col] = scr_data[col].astype(str)
                    if self.mintypython._scorepyon_link_fn == "ERROR":
                        raise ValueError(
                            "You must provide a link_fn on the class construction to score a glm"
                        )
                    if self.mintypython.rename_glm != None:
                        inv_map = {v: k for k, v in self.mintypython.rename_glm.items()}
                        scr_data = scr_data.rename(columns=inv_map)
                    self.mintypython._glm_predictions = scp.score_frame(
                        scr_data,
                        self.mintypython.emb_model_export,
                        link_fn=self.mintypython._scorepyon_link_fn,
                        use_labels=self.mintypython.use_labels,
                        reduce_mem=True,
                    )[model_name]
                data_to_plot["glm_pred_wgt"] = (
                    self.mintypython._glm_predictions * data_to_plot["weight"]
                )
            else:
                data_to_plot["glm_pred_wgt"] = (
                    self.mintypython.data[self.mintypython.glm_preds_col]
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
                    self.mintypython.scale_parameter is not None
                    and self.mintypython._ci_fn is not None
                ):
                    data_to_plot_agg["stdev"] = np.sqrt(
                        self.mintypython._ci_fn(
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
            if self.mintypython._gbm_predictions is None:
                self.mintypython._gbm_predictions = self.mintypython.model.predict(
                    self.mintypython.x_data
                )
            data_to_plot["gbm_pred_wgt"] = (
                self.mintypython._gbm_predictions * data_to_plot["weight"].values
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
        return resp
    
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
            data = self.mintypython.xgb.DMatrix(X_temp, feature_names=features)
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
        if pd.api.types.is_numeric_dtype(self.mintypython.data[group_by_var]):
            nlevels = len(self.mintypython.data[group_by_var].unique())
            if (nlevels > 200 and glm_data_to_plot == None) or (
                nlevel != None or start != None or finish != None or stepsize != None
            ):
                # if is numeric and has a lot of levels bin it
                banded_var = self.create_bandings(
                    self.mintypython.data[group_by_var],
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
                banded_var = self.mintypython.data[group_by_var]
        elif (
            pd.api.types.is_categorical_dtype(self.mintypython.data[group_by_var])
            and not self.mintypython.data[group_by_var].cat.ordered
        ):
            # if is category gets the category name
            banded_var = self.mintypython.data[group_by_var].astype(str)
        else:
            # Just get the data
            banded_var = self.mintypython.data[group_by_var]

        return banded_var
    
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
        if var1 not in self.mintypython.feature_names:
            raise ValueError(
                "The variable "
                + var1
                + " is not in the model the availables variables are "
                + str(self.mintypython.feature_names)
            )
        if var2 not in self.mintypython.feature_names:
            raise ValueError(
                "The variable "
                + var2
                + " is not in the model the availables variables are "
                + str(self.mintypython.feature_names)
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

        data_to_plot["weight"] = self.mintypython.data[self.mintypython.weight_col]

        data_to_plot = data_to_plot.reset_index()

        if shap:
            if self.mintypython.shap_df is None:
                self.prep_shap_values()
            if joinshaps is not None:
                data_to_plot["shap"] = (
                    self.mintypython.shap_df[joinshaps]
                    .prod(axis=1)
                    .reset_index(drop=True)
                )
                data_to_plot["shap_wgt"] = (
                    self.mintypython.shap_df[joinshaps].prod(axis=1)
                    * data_to_plot["weight"]
                ).reset_index(drop=True)
            else:
                data_to_plot["shap"] = self.mintypython.shap_df[var1].reset_index(
                    drop=True
                )
                data_to_plot["shap_wgt"] = (
                    self.mintypython.shap_df[var1] * data_to_plot["weight"]
                ).reset_index(drop=True)
        
        if glm:
            if glmindic_cols is not None:
                data_to_plot["glm_wgt"] = (
                    self.mintypython.glm_df[glmindic_cols].prod(axis=1)
                    * data_to_plot["weight"]
                ).reset_index(drop=True)
            else:
                data_to_plot["glm_wgt"] = (
                    self.mintypython.glm_df[var1] * data_to_plot["weight"]
                ).reset_index(drop=True)

        if actuals:
            act = pd.DataFrame(
                self.mintypython.data[self.mintypython.actuals_col]
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
            if self.mintypython._gbm_predictions is None:
                self.mintypython._gbm_predictions = self.mintypython.model.predict(
                    self.mintypython.x_data
                )
            data_to_plot["gbm_pred_wgt"] = (
                self.mintypython._gbm_predictions * data_to_plot["weight"]
            )
            data_to_plot_agg["gbm_pred_wgt"] = data_to_plot.groupby(
                ["var1", "var2"], dropna=False
            )["gbm_pred_wgt"].sum()
            data_to_plot_agg["gbm_pred"] = (
                data_to_plot_agg["gbm_pred_wgt"] / data_to_plot_agg["weight"]
            )
        return data_to_plot, data_to_plot_agg
    
    def prep_compare_data(self, var_name, mintylist, **kwargs):
        model_ids = kwargs["model_ids"]
        if len(np.unique(model_ids)) != len(model_ids):
            hp = np.unique(model_ids, return_counts=True)
            raise ValueError(
                "Please make sure you do not have duplicated itens on mintynames items:"
                + str(hp[0])
                + " counts:"
                + str(hp[1])
            )

        n = len(mintylist)
        plot_data = list()
        agg_data = list()
        shap_points = list()

        for i in range(0, n):
            plot_data += [mintylist[i].Data_prep.prep_univariate_data(var_name, kwargs)]
            agg_data += [plot_data[i]["agg_data"]]
            shap_points += [plot_data[i]["shap_points"].reset_index(drop=True)]

        # Merge data from mintypython objects
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
                agg_data[i], left_index=True, right_index=True
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
        valid_shaps = set(joinshaps).intersection(set(self.mintypython.shap_df.columns))

        if len(valid_shaps) < len(joinshaps):
            missing = set(joinshaps) - valid_shaps
            print(
                str(missing)
                + " do not exist in in the shap_df, their values will not be included in the calculation"
            )

        return list(valid_shaps)