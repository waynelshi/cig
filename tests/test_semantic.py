import pytest
from cig import PyCausalEngine
from cig.passes.pass2_semantic import SemanticCausalExtractor, InferredEdge


def test_inferred_edge_model():
    edge = InferredEdge(
        source="interest_rate",
        target="borrowing",
        confidence=0.9,
        provenance="macroeconomics_paper.pdf",
        rationale="Cost of credit increases with rate hikes",
        quote="Higher interest rates directly suppress corporate borrowing.",
    )
    assert edge.source == "interest_rate"
    assert edge.target == "borrowing"
    assert edge.edge_type == "INFERRED"
    assert edge.confidence == 0.9


def test_semantic_extractor_and_rust_ingest():
    doc_text = """
    In our empirical market study, changes in price directly impacted overall demand.
    Furthermore, central bank interest rate policies shifted commercial borrowing volumes.
    """
    extractor = SemanticCausalExtractor()
    inferred_edges = extractor.parse_text(doc_text, provenance="market_report.md")

    assert len(inferred_edges) >= 2

    # Ingest into Rust Engine
    engine = PyCausalEngine()
    count = engine.insert_inferred_edges(inferred_edges)
    assert count == len(inferred_edges)
    assert engine.num_edges == count
