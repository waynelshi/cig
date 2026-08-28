import numpy as np
import pandas as pd
from cig import PyCausalEngine
from cig.passes.pass4_engine import CausalEstimationEngine


def test_cinelli_hazlett_and_e_values():
    np.random.seed(42)
    n = 1000

    # Synthetic DGP:
    # W -> T, W -> Y
    # T -> Y (True effect = 2.5)
    w = np.random.normal(0, 1, n)
    t = 0.5 * w + np.random.normal(0, 1, n)
    y = 2.5 * t + 1.2 * w + np.random.normal(0, 1, n)

    df = pd.DataFrame({"W": w, "T": t, "Y": y})

    engine = PyCausalEngine()
    engine.add_extracted_edge("W", "T", None, None)
    engine.add_extracted_edge("W", "Y", None, None)
    engine.add_extracted_edge("T", "Y", None, None)

    estimator = CausalEstimationEngine(engine)
    sens = estimator.compute_sensitivity_analysis(df, treatment="T", outcome="Y")

    assert sens.point_estimate > 2.0
    assert sens.rv_q > 0.30  # Substantial robustness against unobserved confounding
    assert sens.rv_alpha > 0.25
    assert sens.e_value > 1.5
    assert sens.degrees_of_freedom == n - 3
