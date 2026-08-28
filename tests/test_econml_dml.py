import numpy as np
import pandas as pd
import pytest
from cig import PyCausalEngine
from cig.passes.pass4_engine import CausalEstimationEngine


def test_econml_linear_dml_estimation():
    np.random.seed(42)
    n = 1000

    w1 = np.random.uniform(-2, 2, n)
    w2 = np.random.normal(0, 1, n)
    t = np.sin(w1) + np.exp(w2 / 2.0) + np.random.normal(0, 0.5, n)
    y = 3.0 * t + (w1**2) + w2 + np.random.normal(0, 0.5, n)  # TRUE ATE = 3.0

    df = pd.DataFrame({"w1": w1, "w2": w2, "t": t, "y": y})

    engine = PyCausalEngine()
    engine.add_extracted_edge("w1", "t", None, None)
    engine.add_extracted_edge("w1", "y", None, None)
    engine.add_extracted_edge("w2", "t", None, None)
    engine.add_extracted_edge("w2", "y", None, None)
    engine.add_extracted_edge("t", "y", None, None)

    estimator = CausalEstimationEngine(engine)

    summary_dml = estimator.estimate_effect(
        df=df,
        treatment="t",
        outcome="y",
        estimator_method="backdoor.econml.dml.LinearDML",
        run_refutations=False,
    )

    assert pytest.approx(summary_dml.value, abs=0.35) == 3.0


def test_econml_causal_forest_dml_estimation():
    np.random.seed(42)
    n = 800

    w = np.random.normal(0, 1, n)
    t = 0.6 * w + np.random.normal(0, 0.8, n)
    y = 2.0 * t + 1.2 * w + np.random.normal(0, 0.5, n)  # TRUE ATE = 2.0

    df = pd.DataFrame({"w": w, "t": t, "y": y})

    engine = PyCausalEngine()
    engine.add_extracted_edge("w", "t", None, None)
    engine.add_extracted_edge("w", "y", None, None)
    engine.add_extracted_edge("t", "y", None, None)

    estimator = CausalEstimationEngine(engine)

    summary_cf = estimator.estimate_effect(
        df=df,
        treatment="t",
        outcome="y",
        estimator_method="backdoor.econml.dml.CausalForestDML",
        run_refutations=False,
    )

    assert pytest.approx(summary_cf.value, abs=0.35) == 2.0
