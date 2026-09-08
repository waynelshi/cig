from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from cig.passes.pass3_unify import SCMGraphUnification, ExtractedEdge
from cig.artifacts.openlineage_exporter import OpenLineageExporter
from cig.integrations.orchestration import (
    CausalOrchestrationGuard,
    airflow_causal_invariance_operator,
)


def test_openlineage_export(tmp_path: Path):
    unifier = SCMGraphUnification()
    unifier.add_extracted_edges(
        [
            ExtractedEdge(
                source="spend",
                target="revenue",
                provenance="sql",
                expression="spend * 2",
            ),
            ExtractedEdge(
                source="clicks",
                target="revenue",
                provenance="sql",
                expression="clicks * 0.5",
            ),
        ]
    )
    report = unifier.unify()

    event = OpenLineageExporter.to_openlineage_event(report)
    assert event.eventType == "COMPLETE"
    assert len(event.inputs) == 2
    assert len(event.outputs) == 1
    assert event.outputs[0].name == "revenue"

    out_file = tmp_path / "openlineage.json"
    OpenLineageExporter.export_to_file(report, out_path=out_file)
    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert "inputs" in payload
    assert "outputs" in payload


def test_orchestration_guard_and_airflow_hook(tmp_path: Path):
    # Setup mock graph artifact
    graph_dict = {
        "edges": [
            {"source": "x", "target": "y", "edge_type": "EXTRACTED", "confidence": 1.0}
        ],
        "nodes": [{"name": "x"}, {"name": "y"}],
    }
    graph_path = tmp_path / "cig_graph.json"
    graph_path.write_text(json.dumps(graph_dict))

    # Stable dataset
    np.random.seed(42)
    x = np.random.uniform(0, 10, 200)
    y = 2.0 * x + np.random.normal(0, 0.1, 200)
    df_clean = pd.DataFrame({"x": x, "y": y})

    guard = CausalOrchestrationGuard(cig_graph_path=graph_path)
    passed, msg = guard.verify_invariance(df_clean)
    assert passed is True

    # Airflow hook passes
    res = airflow_causal_invariance_operator(df_clean, cig_graph_path=graph_path)
    assert "invariant" in res

    # Degraded dataset (structural regime change halfway)
    y_broken = y.copy()
    y_broken[100:] = 10.0 * x[100:] + 25.0
    df_broken = pd.DataFrame({"x": x, "y": y_broken})

    passed_broken, msg_broken = guard.verify_invariance(df_broken, alpha=0.01)
    assert passed_broken is False
    assert "Breakdown" in msg_broken

    # Airflow hook raises ValueError
    with pytest.raises(ValueError, match="Causal Gate Failure"):
        airflow_causal_invariance_operator(
            df_broken, cig_graph_path=graph_path, alpha=0.01
        )
