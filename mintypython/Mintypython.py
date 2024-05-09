from .libs import *
from .Data_prep import Data_prep
from bidipy import fac_file
import xgboost as xgb


class mintypython:
    def __init__(
        self,
        data,
        weight_col,
        model=None,
        feature_names=None,
        actuals_col=None,
        verbose=True,
        emb_model_export=None,
        emb_gbm_map=dict(),
        glm_preds_col=None,
        gbm_preds_col=None,
        use_labels=True,
        link_fn=None,
        scorepyon_str=False,
        fac_file_path=None,
        scale_parameter=1,
        tweedie_power=None,
        rename_glm=None,
        mapping_dict=None,
        shap_df=None,
        glm_df=None,
        prep_glm_df=False,
        default_engine="Bokeh",
    ):
        """
            Defines the data to be plotted

            :param data: Data containing all features in the model, response variable and weight.
            :type data: Pandas DataFrame

            :param model: GBM model to calculate shapley values
            :type model: Xgboost or Ligthgbm Booster

            :param weight_col: weight column name in data
            :type weight_col: String

            :param feature_names: if the model is xgboost and the feature names are not available in the model object
            :type feature_names: List

            :param actuals_col: Response column name in data
            :type actuals_col: String

            :param verbose: Flag do display or not messages
            :type verbose: Boolean

            :param emb_model_export: Emblem model export path.
            :type emb_model_export: String

            :param emb_gbm_map: if you used banded variables in emblem but continuous in GBM, use this dictionary to link both {"GBM var":"EMB var"}
            :type emb_gbm_map: String

            :param glm_preds_col: glm prediction column, if not passed mintypython will try to use scorepyon to create the predictions
            :type glm_preds_col: String

            :param gbm_preds_col: gbm prediction column, if not passed the model will be used to score the data
            :type gbm_preds_col: String

            :param use_labels: for scorepyon to score the data
            :type use_labels: Boolean

            :param scorpyon_str: Converts scorpyon data to string before scoring
            :type scorpyon_str: Boolean

            :param link_fn: link function to use if the objective is not present in the model ["poisson":"gamma":"identity","logistic"]
            :type link_fn: String

            :param fac_file_path: If your data does not have labels, give me a fac_file i will make you life easier.
            :type fac_file_path: String

            :param rename_glm: Rename dataframe columns before scoring glm.
            :type rename_glm: dictionary

            :param mapping_dict: relabel levels in data, if a fac file is provided this parameter will be ignored.
            :type mapping_dict: dictionary

            :param shap_df: The shapley values dataframe.
            :type shap_df: pandas dataframe

            :param glm_df: A dataframe containing the indications
            :type glm_df: pandas dataframe

            :param prep_glm_df: If you want to create glm_df from emb_model_export
            :type prep_glm_df: boolean
        
        METHODS
        -------
        univariate_plot
        bivariate_plot
        summary_plot
        get_univariate_data_to_plot
        prep_shap_values
        create_bandings

        _create_model_specific_variables
        _get_lgb_params
        _get_xgb_params
        _create_bokeh_plot
        _partial_dependency
        _bin_if_numeric

        """

        self.scale_parameter = scale_parameter
        self.tweedie_power = tweedie_power
        self.use_labels = use_labels
        self.verbose = verbose
        self.data = data

        self.model = model
        self.emb_model_export = emb_model_export
        self.actuals_col = actuals_col

        self.data[weight_col] = self.data[weight_col].astype("float64")

        if actuals_col != None:
            self.data[actuals_col] = self.data[actuals_col].astype("float64")

        if gbm_preds_col != None:
            self._gbm_predictions = self.data[gbm_preds_col].astype("float64")
        else:
            self._gbm_predictions = None

        if glm_preds_col != None:
            self._glm_predictions = self.data[glm_preds_col].astype("float64")
        else:
            self._glm_predictions = None

        self.weight_col = weight_col
        self.emb_gbm_map = emb_gbm_map
        self.link_fn_str = link_fn
        self.glm_preds_col = glm_preds_col

        self.link_fn = None
        self.feature_names = feature_names
        self.shap_df = shap_df
        self.glm_df = glm_df

        self.fac_mapping = None
        self.scorepyon_str = scorepyon_str
        self.config = {
            "labels": {
                "glm": "GLM indication",
                "shap": "Avg SHAP",
                "shap_points": "SHAP",
                "shap_sd": "SHAP +/- SD",
                "glm_pred": "GLM prediction",
                "gbm_pred": "GBM prediction",
                "actuals": "Actuals",
                "weight": "Weight",
                "glm_ci": "GLM pred +/-",
            },
            "colors": {
                "glm": "green",
                "shap": "#3182bd",
                "shap_sd": "#3182bd",
                "shap_points": "#6baed6",
                "glm_pred": "purple",
                "gbm_pred": "silver",
                "actuals": "red",
                "weight": "orange",
            },
            "line_width": {
                "glm": 2,
                "shap": 2,
                "shap_points": None,
                "glm_pred": 1,
                "gbm_pred": 1,
                "actuals": 1,
                "shap_sd": 0.5,
                "weight": None,
            },
            "marker": {
                "glm": "s",
                "shap": "s",
                "shap_points": "s",
                "glm_pred": "s",
                "gbm_pred": "s",
                "actuals": "s",
                "weight": "s",
            },
        }
        self.rename_glm = rename_glm

        if model is not None:
            # Build model type specific variables
            self._create_model_specific_variables()
        else:
            self.feature_names = list(self.data)

        if self.link_fn_str in ["poisson", "gamma", "tweedie"]:
            self.link_fn = np.exp
            self._scorepyon_link_fn = "log"
            if self.link_fn_str == "poisson":
                self._ci_fn = lambda mean, count=None: self.scale_parameter * mean
            if self.link_fn_str == "gamma":
                self._ci_fn = lambda mean, count=None: self.scale_parameter * mean**2
            if self.link_fn_str == "tweedie":
                self._ci_fn = (
                    lambda mean, count=None: self.scale_parameter
                    * mean**self.tweedie_power
                )
        elif self.link_fn_str == "logistic":
            self.link_fn = lambda x: np.exp(x) / (1 + np.exp(x))
            self._scorepyon_link_fn = "logit"
            self._ci_fn = (
                lambda mean, count=None: self.scale_parameter
                * mean
                * (1 - mean)
                / count
            )
        elif self.link_fn_str == "identity":
            self.link_fn = lambda x: x
            self._scorepyon_link_fn = None
            self._ci_fn = lambda mean, count=None: self.scale_parameter
        else:
            self._scorepyon_link_fn = "ERROR"
            self._ci_fn = None
        # Load relativities from scorepyon
        if emb_model_export != None:
            self.emb_mdl = scp.get_factors(
                [emb_model_export], link_fn=lambda x: x, use_labels=use_labels
            )[0]

            # Check how scorepyon is exporting the relativities.
            k1 = list(self.emb_mdl.keys())[0]
            # if is list, convert to value
            if type(self.emb_mdl[k1]) == list:
                new_dict = dict()
                for var, lvls in self.emb_mdl.items():
                    new_dict[var] = dict()
                    if type(lvls) != list:
                        for lvl, rel in lvls.items():
                            new_dict[var][lvl] = rel[0]
                    else:
                        new_dict[var] = lvls[0]

                self.emb_mdl = new_dict

            if self.rename_glm != None:
                for i in self.rename_glm.keys():
                    if i in self.emb_mdl.keys():
                        self.emb_mdl[self.rename_glm[i]] = self.emb_mdl[i]

        self.Data_prep = Data_prep(self)
        if prep_glm_df:
            self.Data_prep.prep_glm_df()
        # Load fac_file
        if fac_file_path != None:
            self.fac_mapping = fac_file.read_fac(fac_file_path)
            if self.rename_glm != None:
                for i in self.rename_glm.keys():
                    if i in self.emb_mdl.keys():
                        self.fac_mapping[self.rename_glm[i]] = self.fac_mapping[i]
        elif mapping_dict != None:
            self.fac_mapping = mapping_dict
        self.default_engine = default_engine
    
    def univariate_plot(
        self,
        var_name,
        base=None,
        shap=False,
        shap_points=False,
        shap_sd=None,
        n_shap_points=1000,
        glm=False,
        glm_pred=False,
        plot_name="Univariate Plot",
        max_levels=None,
        write_out_shap=None,
        weight=True,
        actuals=False,
        shap_points_seed=1256,
        percentile_actuals=1,
        start=None,
        finish=None,
        stepsize=None,
        nlevels=None,
        percentile_start=10,
        percentile_finish=90,
        infinity_higher=True,
        infinity_lower=True,
        ci_z=2,
        glm_ci=False,
        y_axis_max=None,
        y_axis_min=None,
        rebase=True,
        joinshaps=None,
        glmindic_cols=None,
        engine=None,
        joinshaps_error=True,
        height=500,
        width=1000,
        **kwargs
    ):
        """
        Creates a univariate plot

        :param var_name: Variable name in the GBM model
        :type var_name: string

        :param base: Define a base level
        :type base: string

        :param shap: Plot gbm shappley average
        :type shap: boolean

        :param shap_points: Plot shap points
        :type shap_points: boolean

        :param shap_sd: Plot shap standard deviation bar, default to None, which will be set to be the same as shap
        :type shap_sd: boolean

        :param n_shap_points: Number of points to plot and consider in the calculation
        :type n_shap_points: integer

        :param glm: Plot glm relativities
        :type glm: boolean

        :param glm_pred: Plot glm prediction
        :type glm_pred: boolean

        :param plot_name: plot name
        :type plot_name: string

        :param write_out_shap: Write shap out to reuse it
        :type write_out_shap: string

        :param weight: Plot weight
        :type weight: boolean

        :param glm_pred: Plot glm prediction
        :type glm_pred: boolean

        :param actuals: Plot response
        :type actuals: boolean

        :param  shap_points_seed: Seed to sample shap points
        :type  shap_points_seed: integer

        :param percentile_actuals: Percentile to diplay and do not mess with scale
        :type percentile_actuals: float

        :param save_plot: Path to save plot
        :type save_plot: string

        :param start:  For bandings, where the banding starts. Note if the var is categorical or there is a GLM variable this is ignored.
        :type start: float

        :param finish: For bandings, where the banding finishes. Note if the var is categorical or there is a GLM variable this is ignored
        :type finish: float

        :param stepsize: For bandings, stepsize between bins. Note if the var is categorical or there is a GLM variable this is ignored.
        :type stepsize: float

        :param nlevels: For producing SHAP plots, this is the maximum number of levels to show. Note if the var is categorical or there is a GLM variable this is ignored.
        :type nlevels: integer

        :param percentile_start: For bandings, if start is not defined starts with a percentile of the data. Note if the var is categorical or there is a GLM variable this is ignored.
        :type percentile_start: float

        :param percentile_finish: For bandings, if finish is not defined finishes with a percentile of the data. Note if the var is categorical or there is a GLM variable this is ignored.
        :type percentile_finish: float

        :param infinity_higher: For bandings, creates a +inf band. Note if the var is categorical or there is a GLM variable this is ignored.
        :type infinity_higher: boolean

        :param infinity_lower: For bandings, creates a -inf band. Note if the var is categorical or there is a GLM variable this is ignored.
        :type infinity_lower: boolean

        :param ci_z: For confidence intervals: z in mean+-z*stdev.
        :type ci_z: float

        :param glm_ci: Plot confidence intervals.
        :type glm_ci: float

        :param y_axis_max: Set maximun on the y axis to display
        :type y_axis_max: float

        :param y_axis_min: Set minimun on the y axis to display
        :type y_axis_min: float

        :param rebase: If you want to rescale the values to base level
        :type rebase: boolean

        :param joinshaps: If you want to multiply the shapley values of multiple other variables instead of considering the shap of var_name
        :type joinshaps: list("String")

        :param glmindic_cols: If you want to multiply the indication of multiple other variables instead of considering the indication of only var_name.
        :type glmindic_cols: list("String")

        :param joinshaps_error: Ignore errors on joinshaps
        :type joinshaps_error: boolean

        :param height: Height of your plot returned. Default is 500.
        :type height: int

        :param width: Width of your plot returned. Default is 1000.
        :type width: int
        """
        kwargs["base"] = base
        kwargs["shap"] = shap
        kwargs["shap_points"] = shap_points
        if shap_sd is None:
            shap_sd = shap
        kwargs["shap_sd"] = shap_sd
        kwargs["n_shap_points"] = n_shap_points
        kwargs["glm"] = glm
        kwargs["glm_pred"] = glm_pred
        kwargs["plot_name"] = plot_name
        kwargs["nlevels"] = nlevels
        kwargs["write_out_shap"] = write_out_shap
        kwargs["weight"] = weight
        kwargs["actuals"] = actuals
        kwargs["shap_points_seed"] = shap_points_seed
        kwargs["percentile_actuals"] = percentile_actuals
        kwargs["start"] = start
        kwargs["finish"] = finish
        kwargs["stepsize"] = stepsize
        kwargs["percentile_start"] = percentile_start
        kwargs["percentile_finish"] = percentile_finish
        kwargs["infinity_higher"] = infinity_higher
        kwargs["infinity_lower"] = infinity_lower
        kwargs["ci_z"] = ci_z
        kwargs["glm_ci"] = glm_ci
        kwargs["y_axis_max"] = y_axis_max
        kwargs["y_axis_min"] = y_axis_min
        kwargs["rebase"] = rebase
        kwargs["joinshaps"] = joinshaps
        kwargs["glmindic_cols"] = glmindic_cols
        kwargs["joinshaps_error"] = joinshaps_error
        kwargs["height"] = height
        kwargs["width"] = width

        plot_data = self.Data_prep.prep_univariate_data(var_name, kwargs)

        if engine is None:
            engine = self.default_engine

        if engine == "Bokeh":
            from .plot_engines.Bokeh_plot import Bokeh_plot

            plot_engine = Bokeh_plot(self)
        elif engine == "Matplotlib":
            from .plot_engines.Matplotlib_plot import Matplotlib_plot

            plot_engine = Matplotlib_plot(self)

        plot = plot_engine.univariate_plot(kwargs, plot_data, var_name)
        return plot
    
    def bivariate_plot(
        self,
        var1,
        var2,
        glm=False,
        shap=True,
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
        plot_title="Bivariate plot",
        rebase=True,
        glmindic_cols=None,
        joinshaps=None,
        save_plot=None,
        width=1000,
        height=500,
        **kwargs
    ):
        """
        Creates a bivariate plot, where the x-axis is the var1 and the lines on the plot are the var2

        :param var1: Variable to be displayed on the x-axis
        :type var1: string

        :param var2: Variable to be displayed as lines on the graph
        :type var2: string

        :param actuals: Display Actuals
        :type actuals: boolean

        :param gbm_pred: Display gbm predictions
        :type gbm_pred: boolean

        :param nlevels_var1: For producing SHAP plots, this is the maximum number of levels to show. For the bandings on the var1
        :type nlevels_var1: integer

        :param nlevels_var2: For producing SHAP plots, this is the maximum number of levels to show. For the bandings on the var2
        :type nlevels_var2: integer

        :param start_var1: For bandings, where the banding starts. For the bandings on the var1
        :type start_var1: float

        :param start_var2: For bandings, where the banding starts. For the bandings on the var2
        :type start_var2: float

        :param finish_var1: For bandings, where the banding ends.For the bandings on the var1
        :type finish_var1: float

        :param finish_var2: For bandings, where the banding ends.For the bandings on the var2
        :type finish_var2: float

        :param stepsize_var1: For bandings, stepsize between bins. For the bandings on the var1
        :type stepsize_var1: Float

        :param stepsize_var2: For bandings, stepsize between bins. For the bandings on the var2
        :type stepsize_var2: float

        :param percentile_start_var1: For bandings, if start is not defined starts with a percentile of the data. For the bandings on the var1
        :type percentile_start_var1: Float

        :param percentile_start_var2: For bandings, if start is not defined starts with a percentile of the data. For the bandings on the var2
        :type percentile_start_var2: float

        :param percentile_finish_var1: For bandings, if finish is not defined finishes with a percentile of the data. For the bandings on the var1
        :type percentile_finish_var1: Float

        :param percentile_finish_var2: For bandings, if finish is not defined finishes with a percentile of the data. For the bandings on the var2
        :type percentile_finish_var2: float

        :param shap_points_seed: Seed to sample shap points. For the bandings on the var2
        :type shap_points_seed: integer

        :param n_shap_points: Number of points to plot and consider in the calculation. For the bandings on the var2
        :type n_shap_points: integer

        :param plot_title: Title of the plot
        :type plot_title: string

        :param rebase: If you want to rescale the values to base level
        :type rebase: boolean

        :param height: Height of your plot returned. Default is 500.
        :type height: int

        :param width: Width of your plot returned. Default is 1000.
        :type width: int
        """
        kwargs["height"] = height
        kwargs["width"] = width

        if actuals:
            if self.actuals_col is None:
                print(
                    "If you want to plot actuals you must provide a actuals_col in the mintypython call e.g mintypython(actuals_col='your actuals column')"
                )
                actuals = False

        data_to_plot, data_to_plot_agg = self.Data_prep.prep_bivariate_data(
            var1,
            var2,
            shap,
            glm,
            actuals,
            gbm_pred,
            base,
            shap_points,
            nlevels_var1,
            start_var1,
            finish_var1,
            stepsize_var1,
            percentile_start_var1,
            percentile_finish_var1,
            nlevels_var2,
            start_var2,
            finish_var2,
            stepsize_var2,
            percentile_start_var2,
            percentile_finish_var2,
            shap_points_seed,
            n_shap_points,
            plot_title,
            rebase,
            glmindic_cols,
            joinshaps,
        )
        from .plot_engines.Bokeh_plot import Bokeh_plot

        plot_engine = Bokeh_plot(self)

        plot = plot_engine.bivariate_plot(
            data_to_plot,
            data_to_plot_agg,
            var1,
            var2,
            shap,
            glm,
            actuals,
            gbm_pred,
            base,
            shap_points,
            nlevels_var1,
            start_var1,
            finish_var1,
            stepsize_var1,
            percentile_start_var1,
            percentile_finish_var1,
            nlevels_var2,
            start_var2,
            finish_var2,
            stepsize_var2,
            percentile_start_var2,
            percentile_finish_var2,
            shap_points_seed,
            n_shap_points,
            plot_title,
            rebase,
            glmindic_cols,
            joinshaps,
            save_plot,
            **kwargs
        )

        return plot
    
    def _create_model_specific_variables(self):
        """
        Creates feature_names and objective base on model type.
        """
        mdl_type_str = str(type(self.model))
        if "xgboost" in mdl_type_str:
            self._get_xgb_params()
        elif "lightgbm" in mdl_type_str:
            self._get_lgb_params()
        else:
            raise TypeError(
                "Not a valid booster. Only xgboost and lightgbm are accepted"
            )

    def _get_lgb_params(self):
        """
        Create model specific variables if model is lightgbm
        """
        self.feature_names = self.model.feature_name()

        # If link function was defined on the class construction replaces model objective
        if self.link_fn_str is not None:
            self.model.params["objective"] = self.link_fn_str
        else:
            if "objective" not in self.model.params.keys():
                raise ValueError(
                    "This model object does not contain an objective parameter, in order to calculate shap values you must provide a link function 'link_fn' or embed a objective in the model"
                )
        self.x_data = self.data[self.feature_names]

    def _get_xgb_params(self):
        """
        Creates model especific variables if model is xgboost
        """
        if self.feature_names == None:
            if self.model.feature_names == None:
                raise ValueError(
                    "This xgboost model object does not contain feature names, please provide feature_names in the class construction in the same order they were created in the model"
                )

        if "objective" in self.model.attributes().keys():
            model_objective = self.model.attributes()["objective"]
        else:
            model_objective = None

        xgb_map = {
            "count:poisson": "poisson",
            "reg:gamma": "gamma",
            "reg:tweedie": "tweedie",
            "reg:logistic": "logistic",
            "binary:logistic": "logistic",
            "reg:squarederror": "regression",
        }

        if self.link_fn_str == None:
            if model_objective == None:
                raise ValueError(
                    "This model object does not contain an objective parameter, in order to calculate shap values you must provide a link function 'link_fn' or embed a objective in the model"
                )
            elif model_objective not in xgb_map.keys():
                raise ValueError(
                    "This model objective is not valid only" + str(xgb_map.keys())
                )
            else:
                self.link_fn_str = xgb_map[model_objective]

        self.x_data = xgb.DMatrix(
            self.data[self.feature_names],
            weight=self.data[self.weight_col],
            feature_names=self.feature_names,
        )

    def compare(
        self,
        mintylist,
        var_name,
        mintynames=None,
        base=None,
        shap=False,
        shap_points=False,
        n_shap_points=1000,
        glm=False,
        glm_pred=False,
        plot_name="Univariate Plot",
        max_levels=None,
        write_out_shap=None,
        weight=True,
        actuals=False,
        shap_points_seed=1256,
        percentile_actuals=1,
        start=None,
        finish=None,
        stepsize=None,
        nlevels=None,
        percentile_start=10,
        percentile_finish=90,
        infinity_higher=True,
        infinity_lower=True,
        y_axis_max=None,
        y_axis_min=None,
        rebase=True,
        joinshaps=None,
        glmindic_cols=None,
        joinshaps_error=False,
        height=500,
        width=1000,
        **kwargs
    ):
        """
        Creates plot comparing multiple models

        :param mintlist: list of mintypython instances
        :type mintlist: list

        :param var_name: string Variable name
        :type var_name: string

        :param mintynames: List of model names/identifyers
        :type mintynames: List(String)

        :param base: Define a base level
        :type base: string

        :param shap: Plot gbm shappley average
        :type shap: boolean

        :param shap_points: Plot shap points
        :type shap_points: boolean

        :param n_shap_points: Number of points to plot and consider in the calculation
        :type n_shap_points: integer

        :param glm: Plot glm relativities
        :type glm: boolean

        :param glm_pred: Plot glm prediction
        :type glm_pred: boolean

        :param plot_name: plot name
        :type plot_name: string

        :param write_out_shap: Write shap out to reuse it
        :type write_out_shap: string

        :param weight: Plot weight
        :type weight: boolean

        :param actuals: Plot response
        :type actuals: boolean

        :param  shap_points_seed: Seed to sample shap points
        :type  shap_points_seed: integer

        :param percentile_actuals: Percentile to diplay and do not mess with scale
        :type percentile_actuals: float

        :param start:  For bandings, where the banding starts. Note if the var is categorical or there is a GLM variable this is ignored.
        :type start: float

        :param finish: For bandings, where the banding finishes. Note if the var is categorical or there is a GLM variable this is ignored
        :type finish: float

        :param stepsize: For bandings, stepsize between bins. Note if the var is categorical or there is a GLM variable this is ignored.
        :type stepsize: float

        :param nlevels: For producing SHAP plots, this is the maximum number of levels to show. Note if the var is categorical or there is a GLM variable this is ignored.
        :type nlevels: integer

        :param percentile_start: For bandings, if start is not defined starts with a percentile of the data. Note if the var is categorical or there is a GLM variable this is ignored.
        :type percentile_start: float

        :param percentile_finish: For bandings, if finish is not defined finishes with a percentile of the data. Note if the var is categorical or there is a GLM variable this is ignored.
        :type percentile_finish: float

        :param infinity_higher: For bandings, creates a +inf band. Note if the var is categorical or there is a GLM variable this is ignored.
        :type infinity_higher: boolean

        :param infinity_lower: For bandings, creates a -inf band. Note if the var is categorical or there is a GLM variable this is ignored.
        :type infinity_lower: boolean

        :param y_axis_max: Set maximun on the y axis to display
        :type y_axis_max: float

        :param y_axis_min: Set minimun on the y axis to display
        :type y_axis_min: float

        :param rebase: If you want to rescale the values to base level
        :type rebase: boolean

        :param joinshaps: If you want to multiply the shapley values of multiple other variables instead of considering the shap of only var_name
        :type joinshaps: list("String")

        :param glmindic_cols: If you want to multiply the indication of multiple other variables instead of considering the indication of only var_name.
        :type glmindic_cols: list("String")

        :param joinshaps_error: Throw exception if variable is not in join shaps.
        :type joinshaps_error: boolean

        :param height: Height of your plot returned. Default is 500.
        :type height: int

        :param width: Width of your plot returned. Default is 1000.
        :type width: int
        """

        kwargs["base"] = base
        kwargs["shap"] = shap
        kwargs["shap_points"] = shap_points
        kwargs["n_shap_points"] = n_shap_points
        kwargs["glm"] = glm
        kwargs["glm_pred"] = glm_pred
        kwargs["plot_name"] = plot_name
        kwargs["nlevels"] = nlevels
        kwargs["write_out_shap"] = write_out_shap
        kwargs["weight"] = weight
        kwargs["actuals"] = actuals
        kwargs["shap_points_seed"] = shap_points_seed
        kwargs["percentile_actuals"] = percentile_actuals
        kwargs["start"] = start
        kwargs["finish"] = finish
        kwargs["stepsize"] = stepsize
        kwargs["percentile_start"] = percentile_start
        kwargs["percentile_finish"] = percentile_finish
        kwargs["infinity_higher"] = infinity_higher
        kwargs["infinity_lower"] = infinity_lower
        kwargs["ci_z"] = 2
        kwargs["glm_ci"] = False
        kwargs["y_axis_max"] = y_axis_max
        kwargs["y_axis_min"] = y_axis_min
        kwargs["rebase"] = rebase
        kwargs["joinshaps"] = joinshaps
        kwargs["glmindic_cols"] = glmindic_cols
        kwargs["model_ids"] = mintynames
        kwargs["joinshaps_error"] = joinshaps_error
        kwargs["height"] = height
        kwargs["width"] = width

        merged_data, merged_shap_points = self.Data_prep.prep_compare_data(
            var_name, mintylist, **kwargs
        )

        from .plot_engines.Bokeh_plot import Bokeh_plot

        plot_engine = Bokeh_plot(self)

        plot = plot_engine.compare(var_name, merged_data, merged_shap_points, **kwargs)
        return plot
