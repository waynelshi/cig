import numpy as np
import pandas as pd
import pytest
from cig import PyCausalEngine
from cig.passes.pass4_engine import CausalEstimationEngine


def test_pearl_mediation_decomposition():
    np.random.seed(42)
    n = 2000

    # True Structural Model:
    #   W ~ N(0, 1) (Confounder)
    #   T = 0.5 * W + N(0, 1)
    #   M = 1.5 * T + 0.8 * W + N(0, 0.5) (alpha_t = 1.5)
    #   Y = 1.0 * T + 2.0 * M + 1.2 * W + N(0, 0.5) (beta_t = 1.0, beta_m = 2.0)
    # Expected:
    #   NDE = 1.0
    #   NIE = 1.5 * 2.0 = 3.0
    #   TE  = 1.0 + 3.0 = 4.0
    #   Proportion Mediated = 3.0 / 4.0 = 75.0%
    w = np.random.normal(0, 1, n)
    t = 0.5 * w + np.random.normal(0, 1, n)
    m = 1.5 * t + 0.8 * w + np.random.normal(0, 0.5, n)
    y = 1.0 * t + 2.0 * m + 1.2 * w + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"W": w, "T": t, "M": m, "Y": y})

    engine = PyCausalEngine()
    engine.add_extracted_edge("W", "T", None, None)
    engine.add_extracted_edge("W", "M", None, None)
    engine.add_extracted_edge("W", "Y", None, None)
    engine.add_extracted_edge("T", "M", None, None)
    engine.add_extracted_edge("M", "Y", None, None)
    engine.add_extracted_edge("T", "Y", None, None)

    estimator = CausalEstimationEngine(engine)
    summary = estimator.decompose_mediation(
        df=df,
        treatment="T",
        mediator="M",
        outcome="Y",
    )

    assert pytest.approx(summary.natural_direct_effect, abs=0.15) == 1.0
    assert pytest.approx(summary.natural_indirect_effect, abs=0.15) == 3.0
    assert pytest.approx(summary.total_effect, abs=0.20) == 4.0
    assert pytest.approx(summary.proportion_mediated, abs=0.05) == 0.75
    assert "W" in summary.confounders
