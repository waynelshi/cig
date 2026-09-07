"""
Pass 5: Level 3 Structural Counterfactual Reasoning Engine.
Implements Judea Pearl's 3-step counterfactual computation:
  1. Abduction: Inverts structural equations to calculate exogenous noise (epsilon).
  2. Action: Mutates the structural causal model via surgical do-interventions.
  3. Prediction: Propagates intervened values downstream through topological order.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm

from cig import PyCausalEngine


class CounterfactualUnitResult(BaseModel):
    factual_inputs: dict[str, float]
    intervention: dict[str, float]
    predicted_factual_outcome: float
    counterfactual_outcome: float
    individual_causal_effect: float = Field(
        description="Y(x') - Y(x) under exact individual background noise"
    )


class CounterfactualSummary(BaseModel):
    treatment: str
    intervention_value: float
    outcome: str
    sample_size: int
    mean_factual_outcome: float
    mean_counterfactual_outcome: float
    average_treatment_effect_on_treated: float = Field(
        description="Average Counterfactual Difference across target population"
    )
    structural_equations: dict[str, list[str]]
    unit_level_samples: list[CounterfactualUnitResult] = Field(default_factory=list)


class StructuralCounterfactualEngine:
    """Computes Pearlian counterfactuals from observational data and unified SCM graphs."""

    def __init__(self, engine: PyCausalEngine):
        self.engine = engine

    def _get_topological_order(self) -> list[str]:
        """Calculates topological ordering of the DAG."""
        import json

        graph_data = json.loads(self.engine.export_json())
        nodes = [node["name"] for node in graph_data["nodes"]]
        in_degree = {n: 0 for n in nodes}
        adj = {n: [] for n in nodes}

        for src, dst, _ in graph_data.get("edges", []):
            if src in adj and dst in in_degree:
                adj[src].append(dst)
                in_degree[dst] += 1

        queue = [n for n in nodes if in_degree[n] == 0]
        topo_order = []

        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for neighbor in adj.get(curr, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Fallback for unconnected or remaining nodes
        for n in nodes:
            if n not in topo_order:
                topo_order.append(n)

        return topo_order

    def _get_parents_map(self) -> dict[str, list[str]]:
        """Maps each variable to its immediate parents (direct causes)."""
        import json

        graph_data = json.loads(self.engine.export_json())
        parents_map = {node["name"]: [] for node in graph_data["nodes"]}

        for src, dst, _ in graph_data.get("edges", []):
            if dst in parents_map and src not in parents_map[dst]:
                parents_map[dst].append(src)

        return parents_map

    def fit_and_compute_counterfactual(
        self,
        df: pd.DataFrame,
        treatment: str,
        intervention_value: float,
        outcome: str,
    ) -> CounterfactualSummary:
        """
        Executes Abduction -> Action -> Prediction for every row in the dataset.
        """
        parents_map = self._get_parents_map()
        topo_order = self._get_topological_order()

        # Step 1: Structural Equation Modeling (SEM Fitting)
        fitted_models: dict[str, Any] = {}
        fitted_equations: dict[str, list[str]] = {}

        for node in topo_order:
            parents = [p for p in parents_map.get(node, []) if p in df.columns]
            fitted_equations[node] = parents
            if parents and node in df.columns:
                X = sm.add_constant(df[parents])
                y = df[node]
                try:
                    model = sm.OLS(y, X).fit()
                    fitted_models[node] = model
                except Exception:
                    pass

        # Step 2: Abduction (Calculate exogenous residuals epsilon for each unit)
        noise_df = pd.DataFrame(index=df.index)
        for node, model in fitted_models.items():
            parents = fitted_equations[node]
            X = sm.add_constant(df[parents])
            predicted_factual = model.predict(X)
            noise_df[node] = df[node] - predicted_factual

        # Step 3: Action & Prediction (Surgical intervention do(treatment = intervention_value))
        cf_df = df.copy()
        cf_df[treatment] = float(intervention_value)

        # Propagate downstream along the topological order
        treatment_idx = topo_order.index(treatment) if treatment in topo_order else -1
        downstream_nodes = (
            topo_order[treatment_idx + 1 :] if treatment_idx != -1 else topo_order
        )

        for node in downstream_nodes:
            if node in fitted_models:
                parents = fitted_equations[node]
                X_cf = sm.add_constant(cf_df[parents], has_constant="add")
                # Re-add inferred background noise term
                pred_cf = fitted_models[node].predict(X_cf)
                if node in noise_df.columns:
                    cf_df[node] = pred_cf + noise_df[node]
                else:
                    cf_df[node] = pred_cf

        factual_y = df[outcome].values
        cf_y = cf_df[outcome].values
        unit_effects = cf_y - factual_y

        # Build individual sample outputs
        sample_units: list[CounterfactualUnitResult] = []
        for i in range(min(5, len(df))):
            factual_inputs = {
                col: float(df[col].iloc[i])
                for col in df.columns
                if col in parents_map.get(outcome, []) or col == treatment
            }
            sample_units.append(
                CounterfactualUnitResult(
                    factual_inputs=factual_inputs,
                    intervention={treatment: float(intervention_value)},
                    predicted_factual_outcome=float(factual_y[i]),
                    counterfactual_outcome=float(cf_y[i]),
                    individual_causal_effect=float(unit_effects[i]),
                )
            )

        return CounterfactualSummary(
            treatment=treatment,
            intervention_value=float(intervention_value),
            outcome=outcome,
            sample_size=len(df),
            mean_factual_outcome=round(float(np.mean(factual_y)), 4),
            mean_counterfactual_outcome=round(float(np.mean(cf_y)), 4),
            average_treatment_effect_on_treated=round(float(np.mean(unit_effects)), 4),
            structural_equations=fitted_equations,
            unit_level_samples=sample_units,
        )
