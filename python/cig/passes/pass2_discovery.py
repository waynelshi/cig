"""
Pass 2B: Continuous Data-Driven Causal Discovery Engine.
Discovers causal skeletons and orientations directly from observational tabular data
using partial correlation conditional independence tests (PC-style Fisher's z-transform).
"""

from __future__ import annotations

import itertools
from math import atanh, sqrt
from typing import Sequence
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats


class DiscoveredEdge(BaseModel):
    source: str
    target: str
    confidence: float = Field(
        description="Statistical confidence derived from conditional independence test (1 - p_value)"
    )
    partial_corr: float
    p_value: float
    conditioning_set: list[str] = Field(default_factory=list)
    provenance: str = "data_discovery"


class CausalDiscoveryEngine:
    """Discovers directed causal edges from tabular observations."""

    def __init__(self, significance_level: float = 0.05, max_cond_size: int = 2):
        self.alpha = significance_level
        self.max_cond_size = max_cond_size

    def _partial_corr(
        self, df: pd.DataFrame, x: str, y: str, cond_set: Sequence[str]
    ) -> tuple[float, float]:
        """Calculates partial correlation and p-value via Fisher's z-transform."""
        sub_vars = [x, y] + list(cond_set)
        clean_df = df[sub_vars].dropna()
        n = len(clean_df)

        if n <= len(cond_set) + 3:
            return 0.0, 1.0

        cov = clean_df.cov().values
        # Invert covariance matrix (precision matrix)
        try:
            prec = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            return 0.0, 1.0

        idx_x, idx_y = 0, 1
        denom = sqrt(max(1e-12, prec[idx_x, idx_x] * prec[idx_y, idx_y]))
        r_partial = -prec[idx_x, idx_y] / denom

        # Bound r_partial to avoid float overflow in atanh
        r_partial = max(-0.999999, min(0.999999, r_partial))

        # Fisher's z-transformation
        z = atanh(r_partial)
        dof = n - len(cond_set) - 3
        if dof <= 0:
            return float(r_partial), 1.0

        stat = abs(z) * sqrt(dof)
        p_val = 2.0 * (1.0 - stats.norm.cdf(stat))

        return float(r_partial), float(p_val)

    def discover_edges(
        self,
        df: pd.DataFrame,
        target_columns: Sequence[str] | None = None,
        temporal_or_domain_order: Sequence[str] | None = None,
    ) -> list[DiscoveredEdge]:
        """
        Discovers causal edges among numeric columns in df.
        Uses temporal_or_domain_order to orient edges if provided.
        """
        # Filter for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_columns:
            cols = [c for c in target_columns if c in numeric_cols]
        else:
            cols = numeric_cols

        p = len(cols)
        if p < 2:
            return []

        # Step 1: Initialize fully connected undirected skeleton
        adj_candidates: dict[tuple[str, str], tuple[float, float, list[str]]] = {}
        # Filter out columns with zero variance
        valid_cols = [c for c in cols if df[c].dropna().std() > 1e-12]

        for x, y in itertools.combinations(valid_cols, 2):
            r, p_val = stats.pearsonr(df[x].dropna(), df[y].dropna())
            if p_val < self.alpha:
                adj_candidates[(x, y)] = (float(r), float(p_val), [])

        # Step 2: Prune edges with conditional independence tests
        all_vars = set(cols)
        for x, y in list(adj_candidates.keys()):
            possible_conds = list(all_vars - {x, y})
            eliminated = False

            for cond_size in range(
                1, min(self.max_cond_size + 1, len(possible_conds) + 1)
            ):
                for cond_set in itertools.combinations(possible_conds, cond_size):
                    r_cond, p_cond = self._partial_corr(df, x, y, cond_set)
                    if p_cond >= self.alpha:  # Conditionally independent -> remove edge
                        adj_candidates.pop((x, y), None)
                        eliminated = True
                        break
                if eliminated:
                    break

        # Step 3: Orient edges based on temporal/topological prior or variance heuristics
        order_map = {col: idx for idx, col in enumerate(temporal_or_domain_order or [])}
        discovered: list[DiscoveredEdge] = []

        for (x, y), (r_val, p_val, conds) in adj_candidates.items():
            conf = round(max(0.50, min(0.95, 1.0 - p_val)), 4)

            # Prioritize orientation if ordering is known
            if x in order_map and y in order_map:
                if order_map[x] < order_map[y]:
                    src, dst = x, y
                else:
                    src, dst = y, x
            else:
                # Default heuristic: lower variance variable drives higher variance variable
                if df[x].var() <= df[y].var():
                    src, dst = x, y
                else:
                    src, dst = y, x

            discovered.append(
                DiscoveredEdge(
                    source=src,
                    target=dst,
                    confidence=conf,
                    partial_corr=round(r_val, 4),
                    p_value=round(p_val, 6),
                    conditioning_set=conds,
                    provenance="continuous_discovery",
                )
            )

        return discovered
