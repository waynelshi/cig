"""
Pass 4: Automated Causal Estimation, Refutation, Mediation, and Sensitivity Analysis
via DoWhy, EconML, Statsmodels, and cig_core.
"""

from __future__ import annotations

from typing import Any
import json
import math
import warnings
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from cig import PyCausalEngine


class RefutationResult(BaseModel):
    method: str
    estimated_effect: float
    new_effect: float
    p_value: float | None = None
    passed: bool = Field(
        description="True if the causal estimate survived the sensitivity refutation"
    )


class CausalEstimateSummary(BaseModel):
    treatment: str
    outcome: str
    identification_method: str = Field(
        description="'backdoor', 'frontdoor', or 'instrumental_variable'"
    )
    adjustment_set: list[str]
    estimand_expression: str
    value: float = Field(description="Estimated Average Treatment Effect (ATE)")
    estimator_method: str
    confidence_intervals: list[float] | None = None
    refutations: list[RefutationResult] = Field(default_factory=list)


class MediationSummary(BaseModel):
    treatment: str
    mediator: str
    outcome: str
    confounders: list[str]
    total_effect: float = Field(description="Total Causal Effect (TE)")
    natural_direct_effect: float = Field(
        description="Natural Direct Effect (NDE: T -> Y bypassing M)"
    )
    natural_indirect_effect: float = Field(
        description="Natural Indirect Effect (NIE: T -> M -> Y)"
    )
    proportion_mediated: float = Field(
        description="NIE / TE (ratio of effect routed through mediator)"
    )


class SensitivitySummary(BaseModel):
    treatment: str
    outcome: str
    point_estimate: float
    standard_error: float
    t_statistic: float
    degrees_of_freedom: int
    rv_q: float = Field(
        description="Cinelli-Hazlett Robustness Value to bring point estimate to zero (0.0 to 1.0)"
    )
    rv_alpha: float = Field(
        description="Cinelli-Hazlett Robustness Value to bring 95% confidence interval to include zero"
    )
    e_value: float = Field(
        description="VanderWeele-Ding E-value for unmeasured confounding"
    )
    interpretation: str


class CausalEstimationEngine:
    def __init__(self, engine: PyCausalEngine):
        self.engine = engine

    def _export_to_gml(self) -> str:
        """Converts the cig_core graph into a GML string for DoWhy."""
        graph_data = json.loads(self.engine.export_json())
        lines = ["graph [", "  directed 1"]

        node_map = {}
        for idx, node in enumerate(graph_data["nodes"]):
            name = node["name"]
            node_map[name] = idx
            lines.append(f'  node [ id {idx} label "{name}" ]')

        for src, dst, _ in graph_data["edges"]:
            if src in node_map and dst in node_map:
                lines.append(
                    f"  edge [ source {node_map[src]} target {node_map[dst]} ]"
                )

        lines.append("]")
        return "\n".join(lines)

    def _estimate_econml_native(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: list[str],
        estimator_method: str,
        method_params: dict[str, Any] | None = None,
    ) -> tuple[float, list[float] | None]:
        """Direct EconML execution with proper signature routing."""
        from econml.dml import LinearDML, NonParamDML, CausalForestDML

        Y = df[outcome].values
        T = df[treatment].values
        W = df[adjustment_set].values if adjustment_set else None

        params = method_params or {}

        if "CausalForestDML" in estimator_method:
            X = W
            model = CausalForestDML(
                model_y=params.get(
                    "model_y",
                    RandomForestRegressor(
                        n_estimators=40, max_depth=4, random_state=42
                    ),
                ),
                model_t=params.get(
                    "model_t",
                    RandomForestRegressor(
                        n_estimators=40, max_depth=4, random_state=42
                    ),
                ),
                n_estimators=params.get("n_estimators", 40),
                subforest_size=params.get("subforest_size", 4),
                discrete_treatment=params.get("discrete_treatment", False),
                cv=params.get("cv", 3),
                random_state=42,
            )
            model.fit(Y, T, X=X, W=None)
            ate = float(np.mean(model.effect(X)))
            try:
                lb, ub = model.ate_interval(X=X)
                ci = [float(lb), float(ub)]
            except Exception:
                ci = None
            return ate, ci

        elif "NonParamDML" in estimator_method:
            model = NonParamDML(
                model_y=params.get(
                    "model_y",
                    RandomForestRegressor(
                        n_estimators=40, max_depth=4, random_state=42
                    ),
                ),
                model_t=params.get(
                    "model_t",
                    RandomForestRegressor(
                        n_estimators=40, max_depth=4, random_state=42
                    ),
                ),
                model_final=params.get(
                    "model_final",
                    RandomForestRegressor(
                        n_estimators=40, max_depth=4, random_state=42
                    ),
                ),
                discrete_treatment=params.get("discrete_treatment", False),
                cv=params.get("cv", 3),
                random_state=42,
            )
            model.fit(Y, T, X=None, W=W)
            ate = float(model.ate(X=None))
            return ate, None

        else:  # LinearDML default
            model = LinearDML(
                model_y=params.get(
                    "model_y",
                    GradientBoostingRegressor(
                        n_estimators=40, max_depth=3, random_state=42
                    ),
                ),
                model_t=params.get(
                    "model_t",
                    GradientBoostingRegressor(
                        n_estimators=40, max_depth=3, random_state=42
                    ),
                ),
                discrete_treatment=params.get("discrete_treatment", False),
                cv=params.get("cv", 3),
                random_state=42,
            )
            model.fit(Y, T, X=None, W=W)
            ate = float(model.ate(X=None))
            try:
                lb, ub = model.ate_interval(X=None)
                ci = [float(lb), float(ub)]
            except Exception:
                ci = None
            return ate, ci

    def estimate_effect(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        estimator_method: str = "backdoor.linear_regression",
        method_params: dict[str, Any] | None = None,
        run_refutations: bool = True,
    ) -> CausalEstimateSummary:
        """
        Runs automated identification, estimation (OLS, LinearDML, NonParamDML, CausalForestDML, 2SLS IV), and refutations.
        """
        backdoor_sets = self.engine.find_backdoor_adjustment_sets(treatment, outcome)
        frontdoor_sets = self.engine.find_frontdoor_adjustment_sets(treatment, outcome)
        instruments = self.engine.find_instrumental_variables(treatment, outcome)

        if (
            "iv" in estimator_method.lower()
            or "2sls" in estimator_method.lower()
            or "instrument" in estimator_method.lower()
        ):
            if not instruments:
                raise ValueError(
                    f"No valid Instrumental Variables found in DAG for {treatment} -> {outcome}"
                )
            strategy = "instrumental_variable"
            adjustment_set = [instruments[0]]
            estimator_method = "iv.instrumental_variable"
        elif backdoor_sets:
            strategy = "backdoor"
            adjustment_set = backdoor_sets[0]
        elif frontdoor_sets:
            strategy = "frontdoor"
            adjustment_set = frontdoor_sets[0]
            if "backdoor" in estimator_method:
                estimator_method = "frontdoor.two_stage_regression"
        elif instruments:
            strategy = "instrumental_variable"
            adjustment_set = [instruments[0]]
            estimator_method = "iv.instrumental_variable"
        else:
            raise ValueError(
                f"No valid Backdoor, Frontdoor, or Instrumental Variable strategies found in DAG for {treatment} -> {outcome}"
            )

        gml_graph = self._export_to_gml()
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=gml_graph,
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        ci = None
        if "econml" in estimator_method:
            ate_value, ci = self._estimate_econml_native(
                df=df,
                treatment=treatment,
                outcome=outcome,
                adjustment_set=adjustment_set,
                estimator_method=estimator_method,
                method_params=method_params,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=estimator_method,
                    method_params=method_params,
                    test_significance=None,
                )
            ate_value = float(estimate.value)
            try:
                intervals = estimate.get_confidence_intervals()
                if intervals is not None:
                    ci = [float(intervals[0]), float(intervals[1])]
            except Exception:
                pass

        refutation_results = []
        if run_refutations:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref_estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression"
                    if strategy != "instrumental_variable"
                    else "iv.instrumental_variable",
                    test_significance=None,
                )

            try:
                ref_rand = model.refute_estimate(
                    identified_estimand,
                    ref_estimate,
                    method_name="random_common_cause",
                )
                passed = (
                    abs(ref_rand.new_effect - ate_value) / (abs(ate_value) + 1e-6)
                    < 0.40
                )
                refutation_results.append(
                    RefutationResult(
                        method="random_common_cause",
                        estimated_effect=ate_value,
                        new_effect=float(ref_rand.new_effect),
                        passed=passed,
                    )
                )
            except Exception:
                pass

            try:
                ref_placebo = model.refute_estimate(
                    identified_estimand,
                    ref_estimate,
                    method_name="placebo_treatment_refuter",
                    placebo_type="permute",
                )
                passed = abs(ref_placebo.new_effect) < (abs(ate_value) * 0.35 + 0.20)
                refutation_results.append(
                    RefutationResult(
                        method="placebo_treatment",
                        estimated_effect=ate_value,
                        new_effect=float(ref_placebo.new_effect),
                        p_value=getattr(ref_placebo, "p_value", None),
                        passed=passed,
                    )
                )
            except Exception:
                pass

        return CausalEstimateSummary(
            treatment=treatment,
            outcome=outcome,
            identification_method=strategy,
            adjustment_set=adjustment_set,
            estimand_expression=str(identified_estimand),
            value=ate_value,
            estimator_method=estimator_method,
            confidence_intervals=ci,
            refutations=refutation_results,
        )

    def decompose_mediation(
        self,
        df: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        confounders: list[str] | None = None,
    ) -> MediationSummary:
        """
        Decomposes Total Causal Effect into Natural Direct Effect (NDE) and Natural Indirect Effect (NIE)
        using Pearl's Mediation Formula.
        """
        import statsmodels.api as sm

        w_cols = confounders
        if w_cols is None:
            backdoor_sets = self.engine.find_backdoor_adjustment_sets(
                treatment, outcome
            )
            w_cols = [
                c for c in (backdoor_sets[0] if backdoor_sets else []) if c != mediator
            ]

        stage1_features = [treatment] + w_cols
        X_m = sm.add_constant(df[stage1_features])
        y_m = df[mediator]
        model_m = sm.OLS(y_m, X_m).fit()
        alpha_t = model_m.params[treatment]

        stage2_features = [treatment, mediator] + w_cols
        X_y = sm.add_constant(df[stage2_features])
        y_out = df[outcome]
        model_y = sm.OLS(y_out, X_y).fit()
        beta_t = model_y.params[treatment]
        beta_m = model_y.params[mediator]

        nde = float(beta_t)
        nie = float(alpha_t * beta_m)
        total_effect = float(nde + nie)
        prop_mediated = float(nie / total_effect) if abs(total_effect) > 1e-6 else 0.0

        return MediationSummary(
            treatment=treatment,
            mediator=mediator,
            outcome=outcome,
            confounders=w_cols,
            total_effect=round(total_effect, 4),
            natural_direct_effect=round(nde, 4),
            natural_indirect_effect=round(nie, 4),
            proportion_mediated=round(prop_mediated, 4),
        )

    def compute_sensitivity_analysis(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: list[str] | None = None,
        q: float = 1.0,
        alpha: float = 0.05,
    ) -> SensitivitySummary:
        """
        Computes Cinelli-Hazlett Robustness Values (RV_q, RV_alpha) and VanderWeele-Ding E-value.
        """
        import statsmodels.api as sm

        w_cols = adjustment_set
        if w_cols is None:
            backdoor_sets = self.engine.find_backdoor_adjustment_sets(
                treatment, outcome
            )
            w_cols = backdoor_sets[0] if backdoor_sets else []

        features = [treatment] + w_cols
        X = sm.add_constant(df[features])
        y = df[outcome]
        model = sm.OLS(y, X).fit()

        tau = float(model.params[treatment])
        se = float(model.bse[treatment])
        t_stat = float(model.tvalues[treatment])
        dof = int(model.df_resid)

        abs_t = abs(t_stat)

        f_q = q * abs_t / math.sqrt(dof)
        rv_q = (math.sqrt(f_q**2 + 1.0) - 1.0) / f_q if f_q > 1e-6 else 0.0

        t_crit = stats.t.ppf(1.0 - alpha / 2.0, dof)
        if abs_t > t_crit:
            f_alpha = (abs_t - t_crit) / math.sqrt(dof)
            rv_alpha = (
                (math.sqrt(f_alpha**2 + 1.0) - 1.0) / f_alpha if f_alpha > 1e-6 else 0.0
            )
        else:
            rv_alpha = 0.0

        sd_y = float(df[outcome].std())
        std_effect = abs(tau) / (sd_y + 1e-6)
        rr = math.exp(0.91 * std_effect)
        if rr >= 1.0:
            e_val = rr + math.sqrt(rr * (rr - 1.0))
        else:
            rr_inv = 1.0 / rr
            e_val = rr_inv + math.sqrt(rr_inv * (rr_inv - 1.0))

        interp = (
            f"An unobserved confounder explaining ≥ {rv_q * 100:.1f}% of the residual variance in both "
            f"'{treatment}' and '{outcome}' is required to eliminate the observed effect. "
            f"E-value is {e_val:.2f}."
        )

        return SensitivitySummary(
            treatment=treatment,
            outcome=outcome,
            point_estimate=round(tau, 4),
            standard_error=round(se, 4),
            t_statistic=round(t_stat, 4),
            degrees_of_freedom=dof,
            rv_q=round(rv_q, 4),
            rv_alpha=round(rv_alpha, 4),
            e_value=round(e_val, 4),
            interpretation=interp,
        )
