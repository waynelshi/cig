import tempfile
from pathlib import Path
from cig.artifacts.markdown_exporter import SCMMarkdownExporter
from cig.passes.pass1_lineage import ExtractedEdge
from cig.passes.pass2_semantic import InferredEdge
from cig.passes.pass3_unify import SCMGraphUnification


def test_markdown_exporter_generation():
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
                confidence=0.8,
                provenance="retention.md",
                rationale="Loyal customers spend more",
            )
        ]
    )

    report = unifier.unify()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "SCM.md"
        generated_path = SCMMarkdownExporter.export_to_file(report, out_path=out_file)

        assert generated_path.exists()
        content = generated_path.read_text(encoding="utf-8")

        # 1. Mermaid diagram verification
        assert "```mermaid" in content
        assert "graph TD" in content
        assert "user_tenure" in content
        assert "discount_rate" in content
        assert "revenue" in content

        # 2. Markdown summary metrics verification
        assert "| **Total Variables (Nodes)** |" in content
        assert "| **Total Causal Edges** |" in content
        assert "| **Deterministic Extracted Lineages** |" in content
        assert "| **Domain / LLM Inferred Hypotheses** |" in content

        # 3. Unified edge rows verification
        assert "| `user_tenure` | `discount_rate` | **EXTRACTED** | 1.00 |" in content
        assert "| `discount_rate` | `revenue` | **EXTRACTED** | 1.00 |" in content
        assert "| `user_tenure` | `revenue` | **INFERRED** | 0.80 |" in content
