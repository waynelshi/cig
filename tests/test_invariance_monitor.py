from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cig import PyCausalEngine
from cig.passes.pass7_monitor import CausalInvarianceMonitor


def test_causal_invariance_detection():
    np.random.seed(42)
    n = 600

    # Two temporal windows (E0: 300 samples, E1: 300 samples)
    # Target 1 (revenue): Invariant mechanism revenue = 2.0 * spend + noise across both windows
    spend = np.random.uniform(10, 50, n)
    rev_noise = np.random.normal(0, 1, n)
    revenue = 2.0 * spend + rev_noise

    # Target 2 (churn): Mechanism breaks in E1 (e.g. competitor entry, regime shift)
    support_calls = np.random.uniform(1, 10, n)
    churn = np.zeros(n)
    # E0: churn = 0.5 * calls + noise
    churn[:300] = 0.5 * support_calls[:300] + np.random.normal(0, 0.5, 300)
    # E1: Structural shift! churn = 4.0 * calls + 10.0 + noise
    churn[300:] = 4.0 * support_calls[300:] + 10.0 + np.random.normal(0, 0.5, 300)

    time_idx = np.arange(n)

    df = pd.DataFrame(
        {
            "time_idx": time_idx,
            "spend": spend,
            "revenue": revenue,
            "support_calls": support_calls,
            "churn": churn,
        }
    )

    engine = PyCausalEngine()
    engine.add_extracted_edge("spend", "revenue", "etl.sql", None)
    engine.add_extracted_edge("support_calls", "churn", "crm.py", None)

    monitor = CausalInvarianceMonitor(engine, significance_level=0.01)
    report = monitor.audit_temporal_invariance(df, time_column="time_idx")

    assert report.total_mechanisms_audited == 2

    # Map by target
    res_map = {r.target: r for r in report.results}
    assert res_map["revenue"].is_invariant is True
    assert res_map["revenue"].status == "STABLE"

    assert res_map["churn"].is_invariant is False
    assert res_map["churn"].status == "MECHANISM_SHIFT"
    assert res_map["churn"].p_value < 1e-5
