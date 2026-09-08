from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cig import PyCausalEngine
from cig.passes.pass4b_cate import PolicyTreeEngine


def test_cate_policy_tree_heterogeneity():
    np.random.seed(42)
    n = 600

    # Two distinct customer tiers:
    # Tier 1 (low spend <= 50): Highly price sensitive, discount boosts volume -> positive revenue effect (+50)
    # Tier 2 (high spend > 50): Price inelastic, discount erodes margin -> negative revenue effect (-120)
    spend = np.random.uniform(10, 100, n)
    discount = np.random.uniform(0.01, 0.25, n)

    effect = np.where(spend <= 50.0, 50.0, -120.0)
    revenue = 500.0 + effect * discount + 2.0 * spend + np.random.normal(0, 5, n)

    df = pd.DataFrame(
        {
            "discount_pct": discount,
            "spend": spend,
            "revenue": revenue,
        }
    )

    engine = PyCausalEngine()
    engine.add_extracted_edge("discount_pct", "revenue", "pricing.sql", None)
    engine.add_extracted_edge("spend", "revenue", "features.py", None)

    policy_engine = PolicyTreeEngine(engine)
    report = policy_engine.fit_policy_tree(
        df=df,
        treatment="discount_pct",
        outcome="revenue",
        heterogeneous_features=["spend"],
        max_depth=1,
        min_samples_leaf=30,
    )

    assert report.population_ate < 0  # Overall negative due to high-spend drag
    assert len(report.segments) == 2

    # Verify that the two cohorts have contrasting treatment effects
    effects = [seg.mean_treatment_effect for seg in report.segments]
    assert min(effects) < -50.0
    assert max(effects) > 20.0
    assert "graph TD" in report.mermaid_tree
