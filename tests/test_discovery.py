from __future__ import annotations

import numpy as np
import pandas as pd
from cig.passes.pass2_discovery import CausalDiscoveryEngine


def test_continuous_causal_discovery_chain():
    np.random.seed(42)
    n = 1000

    # True DAG: X -> Y -> Z (X and Z are conditionally independent given Y)
    x = np.random.normal(0, 1, n)
    y = 2.0 * x + np.random.normal(0, 0.5, n)
    z = 1.5 * y + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x": x, "y": y, "z": z})

    engine = CausalDiscoveryEngine(significance_level=0.05, max_cond_size=2)
    edges = engine.discover_edges(df, temporal_or_domain_order=["x", "y", "z"])

    edge_tuples = {(e.source, e.target) for e in edges}

    # Verify direct edges recovered and transitive edge (x -> z) pruned
    assert ("x", "y") in edge_tuples
    assert ("y", "z") in edge_tuples
    assert ("x", "z") not in edge_tuples
