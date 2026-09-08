"""
Pass 7: Temporal Causal Invariance & Structural Mechanism Drift Monitor.
Audits structural equations P(Y | PA(Y), E) across environments or time slices
to detect broken physical/business mechanisms and distinguish benign covariate shift
from structural causal regime shifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.linear_model import LinearRegression

from cig import PyCausalEngine


class MechanismDriftResult(BaseModel):
    target: str
    parents: list[str]
    p_value: float
    ks_statistic: float
    is_invariant: bool
    status: str = Field(
        description="'STABLE', 'MECHANISM_SHIFT', or 'UNOBSERVED_INTERVENTION'"
    )
    details: str


class CausalInvarianceReport(BaseModel):
    time_column: str
    total_mechanisms_audited: int
    stable_mechanisms: int
    broken_mechanisms: int
    results: list[MechanismDriftResult]


class CausalInvarianceMonitor:
    """Monitors structural mechanism invariance across temporal partitions."""

    def __init__(self, engine: PyCausalEngine, significance_level: float = 0.05):
        self.engine = engine
        self.significance_level = significance_level

        # Extract parent relationships from Rust engine
        graph_data = json.loads(self.engine.export_json())
        self.parents: dict[str, list[str]] = {}
        for node in graph_data.get("nodes", []):
            self.parents[node["name"]] = []

        for edge in graph_data.get("edges", []):
            src = edge[0] if isinstance(edge, (list, tuple)) else edge["source"]
            dst = edge[1] if isinstance(edge, (list, tuple)) else edge["target"]
            if dst in self.parents:
                self.parents[dst].append(src)

    def audit_temporal_invariance(
        self,
        df: pd.DataFrame,
        time_column: str | None = None,
        target_nodes: Sequence[str] | None = None,
        split_ratio: float = 0.5,
    ) -> CausalInvarianceReport:
        """
        Splits data into reference (E=0) and recent (E=1) temporal windows,
        fits structural regressions Y ~ PA(Y), and runs two-sample Kolmogorov-Smirnov
        tests on residuals to detect structural mechanism drift.
        """
        actual_time_col = time_column
        if time_column and time_column in df.columns:
            sorted_df = df.sort_values(by=time_column).reset_index(drop=True)
        else:
            # Look for common time/index columns
            candidates = [
                "timestamp",
                "created_at",
                "date",
                "time",
                "order_id",
                "idx",
                "id",
            ]
            found = [c for c in candidates if c in df.columns]
            if found:
                actual_time_col = found[0]
                sorted_df = df.sort_values(by=actual_time_col).reset_index(drop=True)
            else:
                actual_time_col = "__index__"
                sorted_df = df.copy().reset_index(drop=True)

        split_idx = int(len(sorted_df) * split_ratio)
        e0 = sorted_df.iloc[:split_idx]
        e1 = sorted_df.iloc[split_idx:]

        targets = target_nodes or [n for n in self.parents if self.parents[n]]
        results: list[MechanismDriftResult] = []

        for target in targets:
            pa = [p for p in self.parents.get(target, []) if p in df.columns]
            if not pa or target not in df.columns:
                continue

            # Fit structural model on reference window E=0
            X_e0 = e0[pa].values
            y_e0 = e0[target].values
            reg = LinearRegression()
            reg.fit(X_e0, y_e0)

            # Compute residuals on both environments
            res_e0 = y_e0 - reg.predict(X_e0)
            X_e1 = e1[pa].values
            y_e1 = e1[target].values
            res_e1 = y_e1 - reg.predict(X_e1)

            # Test null hypothesis: residuals come from identical continuous distribution
            ks_res = stats.ks_2samp(res_e0, res_e1)
            p_val = float(ks_res.pvalue)
            ks_stat = float(ks_res.statistic)

            is_invariant = p_val >= self.significance_level

            if is_invariant:
                status = "STABLE"
                details = f"Mechanism P({target} | {', '.join(pa)}) holds across windows (p={p_val:.4f})."
            else:
                status = "MECHANISM_SHIFT"
                details = (
                    f"Structural breakdown detected! Residual distribution shifted significantly "
                    f"(KS={ks_stat:.3f}, p={p_val:.4e}). External intervention or latent confounder shift."
                )

            results.append(
                MechanismDriftResult(
                    target=target,
                    parents=pa,
                    p_value=round(p_val, 6),
                    ks_statistic=round(ks_stat, 4),
                    is_invariant=is_invariant,
                    status=status,
                    details=details,
                )
            )

        stable_count = sum(1 for r in results if r.is_invariant)
        broken_count = len(results) - stable_count

        return CausalInvarianceReport(
            time_column=actual_time_col,
            total_mechanisms_audited=len(results),
            stable_mechanisms=stable_count,
            broken_mechanisms=broken_count,
            results=results,
        )
