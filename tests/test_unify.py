import pytest
import tempfile
import json
from pathlib import Path
from cig import PyCausalEngine
from cig.passes.pass1_lineage import ExtractedEdge
from cig.passes.pass2_semantic import InferredEdge
from cig.passes.pass3_unify import SCMGraphUnification


def test_extracted_overrides_inferred_conflict():
    unifier = SCMGraphUnification()

    # Pass 1 says: price -> revenue
    unifier.add_extracted_edges(
        [
            ExtractedEdge(
                source="price",
                target="revenue",
                confidence=1.0,
                provenance="models/sales.sql",
                expression="price * qty AS revenue",
            )
        ]
    )

    # Pass 2 erroneously hypothesized: revenue -> price
    unifier.add_inferred_edges(
        [
            InferredEdge(
                source="revenue",
                target="price",
                confidence=0.6,
                provenance="notes.md",
                rationale="Speculative hypothesis",
            )
        ]
    )

    report = unifier.unify()
    # EXTRACTED wins: (revenue -> price) is removed, (price -> revenue) retained
    assert report.total_edges == 1
    edge = report.edges[0]
    assert edge.source == "price"
    assert edge.target == "revenue"
    assert edge.edge_type == "EXTRACTED"


def test_ambiguous_inferred_contradiction():
    unifier = SCMGraphUnification()

    # Paper 1: A -> B
    unifier.add_inferred_edges(
        [
            InferredEdge(
                source="ad_spend",
                target="customer_sentiment",
                confidence=0.7,
                provenance="paper_a.pdf",
                rationale="Ads build brand affinity",
            )
        ]
    )

    # Paper 2: B -> A
    unifier.add_inferred_edges(
        [
            InferredEdge(
                source="customer_sentiment",
                target="ad_spend",
                confidence=0.65,
                provenance="paper_b.pdf",
                rationale="Positive sentiment increases ad budget allocation",
            )
        ]
    )

    report = unifier.unify()
    assert report.ambiguous_edges == 2
    for edge in report.edges:
        assert edge.edge_type == "AMBIGUOUS"
        assert "Contradicted by reverse edge" in edge.conflict_notes


def test_god_confounder_detection_and_artifacts():
    unifier = SCMGraphUnification()

    # 'user_age' affects 3 variables -> God Confounder
    unifier.add_extracted_edges(
        [
            ExtractedEdge(
                source="user_age",
                target="income",
                provenance="features.py",
                expression="...",
            ),
            ExtractedEdge(
                source="user_age",
                target="churn_risk",
                provenance="features.py",
                expression="...",
            ),
            ExtractedEdge(
                source="user_age",
                target="ad_clicks",
                provenance="features.py",
                expression="...",
            ),
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        unifier.export_artifacts(out_dir=tmpdir)

        json_file = Path(tmpdir) / "cig_graph.json"
        md_file = Path(tmpdir) / "CIG_REPORT.md"

        assert json_file.exists()
        assert md_file.exists()

        data = json.loads(json_file.read_text())
        assert "user_age" in data["god_confounders"]
        assert data["total_nodes"] == 4
        assert data["total_edges"] == 3
