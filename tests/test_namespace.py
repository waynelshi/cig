from __future__ import annotations

from cig.passes.namespace import NamespaceResolver
from cig.passes.pass1_lineage import ExtractedEdge
from cig.passes.pass2_semantic import InferredEdge
from cig.passes.pass3_unify import SCMGraphUnification


def test_namespace_resolver_normalization():
    resolver = NamespaceResolver(["revenue", "user_tenure", "discount_pct"])

    assert resolver.resolve("pricing_mart.revenue") == "revenue"
    assert resolver.resolve("`users`.`user_tenure`") == "user_tenure"
    assert resolver.resolve("df['discount_pct']") == "discount_pct"
    assert resolver.resolve("mart_revenue") == "revenue"


def test_unify_collapses_qualified_and_colloquial_names():
    unifier = SCMGraphUnification()

    # Pass 1 produces table-qualified edges
    unifier.add_extracted_edges(
        [
            ExtractedEdge(
                source="pricing_mart.discount_pct",
                target="pricing_mart.revenue",
                expression="price * (1 - discount_pct)",
                provenance="mart.sql",
            )
        ]
    )

    # Pass 2 produces colloquial/naked edge
    unifier.add_inferred_edges(
        [
            InferredEdge(
                source="user_tenure",
                target="revenue",
                confidence=0.80,
                provenance="docs.md",
                rationale="tenure grows revenue",
            )
        ]
    )

    report = unifier.unify()

    # pricing_mart.revenue and revenue should merge into single 'revenue' node
    node_names = {e.target for e in report.edges} | {e.source for e in report.edges}
    assert "pricing_mart.revenue" not in node_names
    assert "revenue" in node_names
    assert "discount_pct" in node_names
    assert "user_tenure" in node_names

    # Check incoming edges to revenue
    revenue_parents = {e.source for e in report.edges if e.target == "revenue"}
    assert revenue_parents == {"discount_pct", "user_tenure"}
