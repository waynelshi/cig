"""
Pass 6: Active Inference Agent Core.
Coordinates the unified SCM with the Free Energy Principle:
  - Epistemic Value: Directs experiments to eliminate edge ambiguity and reduce entropy.
  - Pragmatic Value: Evaluates intervention policies to satisfy target outcome distributions.
  - Expected Free Energy (EFE): Selects optimal actions balancing exploration and exploitation.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from cig import PyCausalEngine
from cig.passes.pass3_unify import SCMUnificationReport, UnifiedEdge


class EpistemicTarget(BaseModel):
    source: str
    target: str
    edge_type: str
    uncertainty_score: float = Field(
        description="Entropy-based measure of ambiguity or lack of statistical confidence"
    )
    suggested_experiment: str = Field(
        description="Proposed active intervention or randomized trial to resolve the edge"
    )


class PolicyEvaluation(BaseModel):
    policy_name: str
    intervention: dict[str, float]
    target_outcome: str
    expected_outcome: float
    pragmatic_value: float = Field(
        description="Negative divergence from goal distribution (higher is better)"
    )
    epistemic_value: float = Field(
        description="Expected reduction in DAG uncertainty/entropy (higher is better)"
    )
    expected_free_energy: float = Field(
        description="G = - (Pragmatic + Epistemic). Lower indicates the preferred policy."
    )


class ActiveInferencePlan(BaseModel):
    epistemic_targets: list[EpistemicTarget] = Field(default_factory=list)
    evaluated_policies: list[PolicyEvaluation] = Field(default_factory=list)
    optimal_policy: PolicyEvaluation | None = None
    agent_rationale: str


class ActiveInferenceAgent:
    """Active Inference engine operating over a unified Causal Inference Graph."""

    def __init__(self, engine: PyCausalEngine, report: SCMUnificationReport):
        self.engine = engine
        self.report = report

    def score_epistemic_uncertainty(self) -> list[EpistemicTarget]:
        """Identifies edges with high Shannon entropy that require active scientific probing."""
        targets: list[EpistemicTarget] = []

        for edge in self.report.edges:
            # 1. Ambiguous edges represent maximum structural entropy
            if edge.edge_type == "AMBIGUOUS":
                targets.append(
                    EpistemicTarget(
                        source=edge.source,
                        target=edge.target,
                        edge_type=edge.edge_type,
                        uncertainty_score=1.0,
                        suggested_experiment=(
                            f"Run a randomized trial intervening on '{edge.source}' to resolve bidirectional "
                            f"contradiction with '{edge.target}'."
                        ),
                    )
                )
            # 2. Inferred hypotheses have uncertainty proportional to 1 - confidence
            elif edge.edge_type == "INFERRED" and edge.confidence < 0.95:
                # Binary Shannon entropy as uncertainty metric: H(p) = -p log2(p) - (1-p) log2(1-p)
                p = max(0.01, min(0.99, edge.confidence))
                entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
                targets.append(
                    EpistemicTarget(
                        source=edge.source,
                        target=edge.target,
                        edge_type=edge.edge_type,
                        uncertainty_score=round(entropy, 4),
                        suggested_experiment=(
                            f"Perform observational stratification or collect targeted logs on '{edge.source} -> {edge.target}' "
                            f"to validate hypothetical link (prior confidence: {edge.confidence:.2f})."
                        ),
                    )
                )

        targets.sort(key=lambda x: x.uncertainty_score, reverse=True)
        return targets

    def evaluate_policies(
        self,
        candidate_interventions: dict[str, dict[str, float]],
        target_outcome: str,
        desired_value: float,
        df: pd.DataFrame,
        tolerance_sigma: float = 1.0,
    ) -> list[PolicyEvaluation]:
        """
        Evaluates candidate policies by computing Expected Free Energy:
          G = Pragmatic Divergence - Epistemic Information Gain
        """
        evaluations: list[PolicyEvaluation] = []
        epistemic_targets = self.score_epistemic_uncertainty()
        epistemic_nodes = {t.source for t in epistemic_targets} | {
            t.target for t in epistemic_targets
        }

        for name, intervention in candidate_interventions.items():
            treatment_var = next(iter(intervention.keys()))
            val = intervention[treatment_var]

            # 1. Compute Pragmatic Value (Approximation via OLS or Backdoor conditioning)
            parents = [
                e.source
                for e in self.report.edges
                if e.target == target_outcome and e.source in df.columns
            ]
            if treatment_var in df.columns and target_outcome in df.columns:
                corr = df[[treatment_var, target_outcome]].corr().iloc[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                baseline_mean = float(df[target_outcome].mean())
                std_ratio = (
                    float(
                        df[target_outcome].std() / max(0.001, df[treatment_var].std())
                    )
                    if df[treatment_var].std() > 0
                    else 1.0
                )
                pred_y = baseline_mean + (
                    corr * std_ratio * (val - float(df[treatment_var].mean()))
                )
            else:
                pred_y = desired_value

            # Gaussian log-likelihood as negative divergence (Pragmatic utility)
            error = (pred_y - desired_value) ** 2
            pragmatic_utility = -(error / (2 * (tolerance_sigma**2)))

            # 2. Compute Epistemic Value (Information gain if policy perturbs uncertain nodes)
            epistemic_gain = 0.0
            if treatment_var in epistemic_nodes:
                matching_uncertainties = [
                    t.uncertainty_score
                    for t in epistemic_targets
                    if t.source == treatment_var or t.target == treatment_var
                ]
                epistemic_gain = sum(matching_uncertainties)

            # 3. Expected Free Energy: G = - (Pragmatic + Epistemic)
            efe = -(pragmatic_utility + (1.5 * epistemic_gain))

            evaluations.append(
                PolicyEvaluation(
                    policy_name=name,
                    intervention=intervention,
                    target_outcome=target_outcome,
                    expected_outcome=round(pred_y, 3),
                    pragmatic_value=round(pragmatic_utility, 4),
                    epistemic_value=round(epistemic_gain, 4),
                    expected_free_energy=round(efe, 4),
                )
            )

        # Lower EFE is superior
        evaluations.sort(key=lambda x: x.expected_free_energy)
        return evaluations

    def plan(
        self,
        candidate_interventions: dict[str, dict[str, float]],
        target_outcome: str,
        desired_value: float,
        df: pd.DataFrame,
    ) -> ActiveInferencePlan:
        """Executes full active inference cycle: Epistemic diagnosis + Policy arbitration."""
        targets = self.score_epistemic_uncertainty()
        evaluated = self.evaluate_policies(
            candidate_interventions=candidate_interventions,
            target_outcome=target_outcome,
            desired_value=desired_value,
            df=df,
        )

        optimal = evaluated[0] if evaluated else None
        rationale = (
            f"Policy '{optimal.policy_name}' selected with minimum Expected Free Energy (G = {optimal.expected_free_energy:.3f}). "
            f"Balances expected target attainment ({optimal.expected_outcome:.2f} vs goal {desired_value:.2f}) "
            f"with epistemic information gain ({optimal.epistemic_value:.2f}) on unverified DAG components."
            if optimal
            else "No actionable policies provided."
        )

        return ActiveInferencePlan(
            epistemic_targets=targets,
            evaluated_policies=evaluated,
            optimal_policy=optimal,
            agent_rationale=rationale,
        )
