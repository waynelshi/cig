from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from cig.artifacts.graphml_exporter import GraphMLExporter
from cig.passes.pass1_lineage import ExtractedEdge
from cig.passes.pass2_semantic import InferredEdge
from cig.passes.pass3_unify import SCMGraphUnification


def test_graphml_export_and_schema():
    unifier = SCMGraphUnification()

    unifier.add_extracted_edges(
        [
            ExtractedEdge(
                source="user_tenure",
                target="discount_rate",
                provenance="pricing.sql",
                expression="tenure * 0.05",
            ),
            ExtractedEdge(
                source="discount_rate",
                target="revenue",
                provenance="mart.sql",
                expression="price * (1 - discount_rate)",
            ),
        ]
    )

    unifier.add_inferred_edges(
        [
            InferredEdge(
                source="user_tenure",
                target="revenue",
                confidence=0.85,
                provenance="retention.md",
                rationale="Loyal customers spend more",
            )
        ]
    )

    report = unifier.unify()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_graph.graphml"
        generated_path = GraphMLExporter.export_to_file(report, out_path=out_file)

        assert generated_path.exists()
        content = generated_path.read_text(encoding="utf-8")

        # 1. XML Header and Schema verification
        assert '<?xml version="1.0" encoding="UTF-8"?>' in content
        assert "http://graphml.graphdrawing.org/xmlns" in content
        assert '<graph id="SCM_DAG" edgedefault="directed">' in content

        # 2. Parse XML to ensure well-formed document
        root = ET.fromstring(content)
        assert root.tag.endswith("graphml")

        # 3. Node and Edge verification
        assert '<node id="user_tenure">' in content
        assert '<node id="discount_rate">' in content
        assert '<node id="revenue">' in content
        assert 'source="user_tenure" target="discount_rate"' in content
        assert 'source="discount_rate" target="revenue"' in content
