from __future__ import annotations
import tempfile
from pathlib import Path
from cig.passes.pass1_lineage import ExtractedEdge
from cig.passes.pass2_semantic import InferredEdge
from cig.passes.pass3_unify import SCMGraphUnification
from cig.artifacts.visualizer import HTMLDAGVisualizer


def test_html_visualizer_generation_and_pyodide_wasm():
    unifier = SCMGraphUnification()

    unifier.add_extracted_edges(
        [
            ExtractedEdge(
                source="w_confounder",
                target="treatment_x",
                provenance="mart.sql",
                expression="calc(w)",
            ),
            ExtractedEdge(
                source="w_confounder",
                target="outcome_y",
                provenance="mart.sql",
                expression="calc(w)",
            ),
            ExtractedEdge(
                source="treatment_x",
                target="outcome_y",
                provenance="mart.sql",
                expression="t * 2",
            ),
        ]
    )
    unifier.add_inferred_edges(
        [
            InferredEdge(
                source="ad_campaign",
                target="treatment_x",
                confidence=0.85,
                provenance="notes.md",
                rationale="Ads boost adoption",
            )
        ]
    )

    report = unifier.unify()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "cig.html"
        generated_path = HTMLDAGVisualizer.generate_html(report, out_path=out_file)

        assert generated_path.exists()
        content = generated_path.read_text(encoding="utf-8")

        # 1. Verify JSON payload injection
        assert "__CIG_DATA_PLACEHOLDER__" not in content
        assert "treatment_x" in content
        assert "outcome_y" in content

        # 2. Verify Pyodide WebAssembly Runtime bindings
        assert "pyodide.js" in content
        assert "loadPyodide" in content
        assert "tab-pyodide" in content
        assert "wasm-select-t" in content
        assert "wasm-select-y" in content
        assert "btn-run-ate" in content
        assert "runInBrowserEstimation" in content
        assert "generateSyntheticDataset" in content

        # 3. Verify Sensitivity Simulator & Export Handlers
        assert "renderContour()" in content
        assert "exportDAGAsSVG()" in content
        assert "exportContourAsPNG()" in content


def test_html_visualizer_svg_markup_elements():
    """Verify SVG generator templates inside HTML contain valid XML/SVG tags and arrow markers."""
    unifier = SCMGraphUnification()
    unifier.add_extracted_edges(
        [
            ExtractedEdge(
                source="cost",
                target="profit",
                provenance="ops.sql",
                expression="rev - cost",
            ),
        ]
    )
    report = unifier.unify()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "cig.html"
        HTMLDAGVisualizer.generate_html(report, out_path=out_file)
        content = out_file.read_text(encoding="utf-8")

        # Check SVG builder template markers and styling
        assert 'xmlns="http://www.w3.org/2000/svg"' in content
        assert '<marker id="arrow"' in content
        assert 'stroke-dasharray="4,4"' in content
        assert "image/svg+xml;charset=utf-8" in content
