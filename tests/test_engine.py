import numpy as np
import pandas as pd
import pytest
from cig import PyCausalEngine
from cig.passes.pass4_engine import CausalEstimationEngine


def test_end_to_end_causal_estimation():
    np.random.seed(42)
    n_samples = 2000

    # True model:
    #   confounder -> treatment (beta = 0.8)
    #   confounder -> outcome   (beta = 1.5)
    #   treatment  -> outcome   (TRUE CAUSAL EFFECT = 2.5)
    confounder = np.random.normal(10, 2, n_samples)
    treatment = 0.8 * confounder + np.random.normal(0, 1, n_samples)
    outcome = 2.5 * treatment + 1.5 * confounder + np.random.normal(0, 1, n_samples)

    df = pd.DataFrame(
        {
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    # 1. Setup DAG in cig_core
    core = PyCausalEngine()
    core.add_extracted_edge("confounder", "treatment", "schema.sql", None)
    core.add_extracted_edge("confounder", "outcome", "schema.sql", None)
    core.add_extracted_edge("treatment", "outcome", "pipeline.py", None)

    # 2. Run Pass 4 estimation
    estimator = CausalEstimationEngine(core)
    summary = estimator.estimate_effect(
        df=df,
        treatment="treatment",
        outcome="outcome",
        estimator_method="backdoor.linear_regression",
        run_refutations=True,
    )

    # Verify identification found the confounder
    assert summary.identification_method == "backdoor"
    assert "confounder" in summary.adjustment_set

    # Verify estimated ATE is close to ground truth (2.5)
    assert pytest.approx(summary.value, abs=0.15) == 2.5

    # Verify refutations passed
    assert len(summary.refutations) >= 1
    for ref in summary.refutations:
        assert ref.passed
