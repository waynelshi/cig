from __future__ import annotations

import numpy as np
import pandas as pd
from cig import PyCausalEngine
from cig.passes.pass5_counterfactual import StructuralCounterfactualEngine


def test_level3_structural_counterfactual_engine():
    engine = PyCausalEngine()
    engine.add_extracted_edge("user_tenure", "discount_pct", None, None)
    engine.add_extracted_edge("user_tenure", "revenue", None, None)
    engine.add_extracted_edge("discount_pct", "revenue", None, None)

    np.random.seed(42)
    N = 500
    user_tenure = np.random.normal(5.0, 1.0, N)
    discount_pct = 0.05 * user_tenure + np.random.normal(0.0, 0.01, N)
    revenue = (
        500.0
        - 100.0 * discount_pct
        + 20.0 * user_tenure
        + np.random.normal(0.0, 2.0, N)
    )

    df = pd.DataFrame(
        {
            "user_tenure": user_tenure,
            "discount_pct": discount_pct,
            "revenue": revenue,
        }
    )

    cf_engine = StructuralCounterfactualEngine(engine)
    summary = cf_engine.fit_and_compute_counterfactual(
        df=df,
        treatment="discount_pct",
        intervention_value=0.0,
        outcome="revenue",
    )

    assert summary.sample_size == N
    assert summary.mean_counterfactual_outcome > summary.mean_factual_outcome
    assert summary.average_treatment_effect_on_treated > 0.0
    assert len(summary.unit_level_samples) > 0
