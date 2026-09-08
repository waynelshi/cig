"""
Pass 4B: Heterogeneous Treatment Effect (CATE) & Policy Tree Engine.
Estimates unit-level treatment heterogeneity using Causal Forest DML and
extracts shallow, interpretable decision policy trees for personalized interventions.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.tree import DecisionTreeRegressor, export_text

from cig import PyCausalEngine
from cig.passes.pass4_engine import CausalEstimationEngine


class PolicySegment(BaseModel):
    segment_id: str
    condition: str
    mean_treatment_effect: float
    sample_size: int
    recommended_policy: str


class CATEReport(BaseModel):
    treatment: str
    outcome: str
    population_ate: float
    heterogeneous_features: list[str]
    segments: list[PolicySegment]
    rules_text: str
    mermaid_tree: str


class PolicyTreeEngine:
    """Estimates heterogeneous treatment effects and fits interpretable policy trees."""

    def __init__(self, engine: PyCausalEngine):
        self.engine = engine
        self.estimator = CausalEstimationEngine(engine)

    def fit_policy_tree(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        heterogeneous_features: Sequence[str] | None = None,
        max_depth: int = 2,
        min_samples_leaf: int = 20,
    ) -> CATEReport:
        """Fits CATE using CausalForestDML and distills marginal effects into an interpretable tree."""
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor

        # 1. Identify valid adjustment set
        backdoor_sets = self.engine.find_backdoor_adjustment_sets(treatment, outcome)
        raw_adj = backdoor_sets[0] if backdoor_sets else []
        w_cols = [
            c for c in raw_adj if c in df.columns and c not in (treatment, outcome)
        ]

        # 2. Determine heterogeneous feature set X
        if heterogeneous_features:
            x_cols = [
                c
                for c in heterogeneous_features
                if c in df.columns and c not in (treatment, outcome)
            ]
        else:
            # Auto-detect numeric covariates excluding T and Y
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            x_cols = [c for c in numeric_cols if c not in (treatment, outcome)]

        if not x_cols:
            raise ValueError(
                f"No valid heterogeneous features available in dataframe for {treatment} -> {outcome}"
            )

        # Keep features distinct between X (heterogeneity) and W (pure controls)
        pure_w = [c for c in w_cols if c not in x_cols]

        Y = df[outcome].values
        T = df[treatment].values
        X = df[x_cols].values
        W = df[pure_w].values if pure_w else None

        # 3. Fit CausalForestDML
        # 3. Fit CausalForestDML
        cf = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=40, max_depth=5, random_state=42
            ),
            model_t=RandomForestRegressor(
                n_estimators=40, max_depth=5, random_state=42
            ),
            n_estimators=40,
            subforest_size=4,
            discrete_treatment=False,
            cv=3,
            random_state=42,
        )
        cf.fit(Y, T, X=X, W=W)

        # 4. Predict unit-level treatment effects (tau_i)
        marginal_effects = cf.effect(X)
        pop_ate = float(np.mean(marginal_effects))

        # 5. Distill individual treatment effects into a shallow policy tree
        policy_tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        policy_tree.fit(X, marginal_effects)

        tree_rules = export_text(policy_tree, feature_names=x_cols)

        # 6. Extract leaf segments
        leaf_ids = policy_tree.apply(X)
        segments: list[PolicySegment] = []
        unique_leaves = np.unique(leaf_ids)

        for idx, leaf_val in enumerate(unique_leaves, 1):
            mask = leaf_ids == leaf_val
            sub_effects = marginal_effects[mask]
            mean_eff = float(np.mean(sub_effects))
            size = int(np.sum(mask))

            # Recommendation heuristic based on effect direction
            if mean_eff > 0:
                rec = f"EXPAND {treatment} (Positive causal uplift: +{mean_eff:.2f})"
            elif mean_eff < -10.0:
                rec = (
                    f"RESTRICT {treatment} (Destructive margin erosion: {mean_eff:.2f})"
                )
            else:
                rec = f"HOLD {treatment} baseline (Negligible sensitivity: {mean_eff:.2f})"

            segments.append(
                PolicySegment(
                    segment_id=f"Cohort_{idx}",
                    condition=f"Leaf_Node_{leaf_val}",
                    mean_treatment_effect=round(mean_eff, 4),
                    sample_size=size,
                    recommended_policy=rec,
                )
            )

        # 7. Generate Mermaid flowchart syntax
        mermaid_lines = ["graph TD"]
        tree_ = policy_tree.tree_

        def recurse_mermaid(
            node: int, parent_id: str | None = None, label: str = ""
        ) -> None:
            if tree_.feature[node] != -2:  # Not a leaf
                feat_name = x_cols[tree_.feature[node]]
                thresh = tree_.threshold[node]
                node_id = f"node_{node}"
                mermaid_lines.append(f'  {node_id}["{feat_name} <= {thresh:.2f}?"]')
                if parent_id:
                    mermaid_lines.append(f"  {parent_id} -->|{label}| {node_id}")

                left_child = tree_.children_left[node]
                right_child = tree_.children_right[node]
                recurse_mermaid(left_child, node_id, "Yes")
                recurse_mermaid(right_child, node_id, "No")
            else:
                val = float(tree_.value[node][0][0])
                node_id = f"leaf_{node}"
                rec_label = f"CATE: {val:+.2f}"
                mermaid_lines.append(f'  {node_id}["{rec_label}"]')
                if parent_id:
                    mermaid_lines.append(f"  {parent_id} -->|{label}| {node_id}")

        recurse_mermaid(0)
        mermaid_tree = "\n".join(mermaid_lines)

        return CATEReport(
            treatment=treatment,
            outcome=outcome,
            population_ate=round(pop_ate, 4),
            heterogeneous_features=x_cols,
            segments=segments,
            rules_text=tree_rules,
            mermaid_tree=mermaid_tree,
        )
