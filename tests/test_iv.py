import numpy as np
import pandas as pd
import pytest
from cig import PyCausalEngine
from cig.passes.pass4_engine import CausalEstimationEngine


def test_instrumental_variable_identification():
    # DAG:
    #   Z -> X (Instrument -> Treatment)
    #   U -> X, U -> Y (Unobserved Confounder U)
    #   X -> Y (Treatment -> Outcome)
    #   Z is d-separated from Y given X in G_underline_X
    engine = PyCausalEngine()
    engine.add_extracted_edge("Z", "X", None, None)
    engine.add_extracted_edge("U", "X", None, None)
    engine.add_extracted_edge("U", "Y", None, None)
    engine.add_extracted_edge("X", "Y", None, None)

    # Backdoor with observable variables fails due to hidden U
    # But Z is a valid instrument!
    assert engine.is_valid_instrument("Z", "X", "Y")
    assert not engine.is_valid_instrument("U", "X", "Y")
    assert not engine.is_valid_instrument("X", "X", "Y")

    ivs = engine.find_instrumental_variables("X", "Y")
    assert "Z" in ivs


def test_end_to_end_2sls_estimation():
    np.random.seed(42)
    n = 2000

    # True structural model with strong unobserved confounding:
    #   Z ~ N(0, 1)  (Instrument: e.g. lottery draft / discount assigned)
    #   U ~ N(0, 1)  (Hidden Confounder: e.g. unobserved consumer intent)
    #   X = 1.2 * Z + 1.5 * U + e_x
    #   Y = 2.0 * X + 2.5 * U + e_y  (TRUE ATE = 2.0)
    z = np.random.normal(0, 1, n)
    u = np.random.normal(0, 1, n)
    x = 1.2 * z + 1.5 * u + np.random.normal(0, 0.5, n)
    y = 2.0 * x + 2.5 * u + np.random.normal(0, 0.5, n)

    # Naive OLS is severely biased because cov(X, U) > 0
    # OLS coefficient would be ~ 2.8 instead of 2.0

    df = pd.DataFrame({"Z": z, "X": x, "Y": y})

    engine = PyCausalEngine()
    engine.add_extracted_edge("Z", "X", None, None)
    engine.add_extracted_edge("X", "Y", None, None)

    estimator = CausalEstimationEngine(engine)
    summary = estimator.estimate_effect(
        df=df,
        treatment="X",
        outcome="Y",
        estimator_method="iv.two_stage_least_squares",
        run_refutations=False,
    )

    assert summary.identification_method == "instrumental_variable"
    assert "Z" in summary.adjustment_set
    # 2SLS recovers the true unconfounded treatment effect (2.0)
    assert pytest.approx(summary.value, abs=0.25) == 2.0
