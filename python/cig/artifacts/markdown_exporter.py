"""
Markdown & Mermaid SCM Exporter.
Generates human-readable Markdown documentation and Mermaid DAG diagrams
from the unified Structural Causal Model (SCMUnificationReport).
"""

from pathlib import Path
from typing import Union
from cig.passes.pass3_unify import SCMUnificationReport, UnifiedEdge


class SCMMarkdownExporter:
    """Renders SCMUnificationReport into SCM.md with Mermaid diagrams and summary tables."""

    @staticmethod
    def _sanitize_node_id(name: str) -> str:
        """Sanitizes variable names for Mermaid node identifiers."""
        return (
            name.replace(".", "_").replace("-", "_").replace(" ", "_").replace(":", "_")
        )

    @classmethod
    def generate_mermaid_diagram(cls, report: SCMUnificationReport) -> str:
        """Generates Mermaid flowchart TD diagram from unified edges."""
        lines = ["```mermaid", "graph TD"]

        god_set = set(report.god_confounders or [])

        # 1. Define nodes & labels
        all_nodes = set()
        for e in report.edges:
            all_nodes.add(e.source)
            all_nodes.add(e.target)

        for node in sorted(all_nodes):
            safe_id = cls._sanitize_node_id(node)
            is_god = node in god_set
            display_label = f"⭐ {node} (Hub)" if is_god else node
            lines.append(f'    {safe_id}["{display_label}"]')

        # 2. Define edges with style classes
        for e in report.edges:
            src_id = cls._sanitize_node_id(e.source)
            dst_id = cls._sanitize_node_id(e.target)

            if e.edge_type == "EXTRACTED":
                lines.append(f'    {src_id} -->|"{e.confidence:.2f}"| {dst_id}')
            elif e.edge_type == "INFERRED":
                lines.append(f'    {src_id} -.->|"{e.confidence:.2f}"| {dst_id}')
            elif e.edge_type == "AMBIGUOUS":
                lines.append(f'    {src_id} ==>|"⚠ Conflict"| {dst_id}')

        # 3. Styling classes
        lines.append("")
        lines.append("    %% Styling")
        lines.append(
            "    classDef godNode fill:#382800,stroke:#d29922,stroke-width:2px,color:#f1e05a;"
        )
        lines.append(
            "    classDef default fill:#161b22,stroke:#58a6ff,stroke-width:1px,color:#c9d1d9;"
        )

        if god_set:
            god_ids = [cls._sanitize_node_id(g) for g in god_set if g in all_nodes]
            if god_ids:
                lines.append(f"    class {','.join(god_ids)} godNode;")

        lines.append("```")
        return "\n".join(lines)

    @classmethod
    def generate_markdown(
        cls, report: SCMUnificationReport, title: str = "Structural Causal Model (SCM)"
    ) -> str:
        """Constructs the full Markdown document."""
        ambiguous_edges = [e for e in report.edges if e.edge_type == "AMBIGUOUS"]

        sections: list[str] = [
            f"# {title}",
            "",
            "> Automated causal DAG and lineage summary synthesized by **CIG** (Causal Inference Graph engine).",
            "",
            "## SCM DAG Overview",
            "",
            cls.generate_mermaid_diagram(report),
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Variables (Nodes)** | {report.total_nodes} |",
            f"| **Total Causal Edges** | {report.total_edges} |",
            f"| **Deterministic Extracted Lineages** | {report.extracted_edges} |",
            f"| **Domain / LLM Inferred Hypotheses** | {report.inferred_edges} |",
            f"| **Ambiguous / Conflicted Edges** | {report.ambiguous_edges} |",
            f"| **God Confounders (Hub Nodes)** | {len(report.god_confounders)} |",
            "",
        ]

        # God Confounders Section
        if report.god_confounders:
            sections.extend(
                [
                    "## 👑 God Confounders (High-Degree Hub Nodes)",
                    "",
                    "The following variables exhibit extensive downstream or bidirectional influence across multiple subgraphs, making them critical candidates for backdoor adjustment sets:",
                    "",
                ]
            )
            for gc in sorted(report.god_confounders):
                sections.append(f"- **`{gc}`**")
            sections.append("")

        # Edges Table
        sections.extend(
            [
                "## Unified Causal Relationships",
                "",
                "| Cause (Source) | Effect (Target) | Type | Confidence | Expression / Mechanism | Provenance |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )

        for e in sorted(report.edges, key=lambda x: (x.source, x.target)):
            mech = (
                "<br>".join(e.mechanisms) if e.mechanisms else (e.conflict_notes or "-")
            )
            prov = "<br>".join(e.provenance) if e.provenance else "-"
            mech_escaped = mech.replace("|", "\\|")
            prov_escaped = prov.replace("|", "\\|")
            sections.append(
                f"| `{e.source}` | `{e.target}` | **{e.edge_type}** | {e.confidence:.2f} | {mech_escaped} | {prov_escaped} |"
            )
        sections.append("")

        # Ambiguities & Conflicts Section
        if ambiguous_edges:
            sections.extend(
                [
                    "## ⚠ Ambiguities and Domain Conflicts",
                    "",
                    "The following relationships showed conflicting directions or contradictions between lineage extraction and domain hypotheses:",
                    "",
                    "| Source | Target | Reason / Conflict Notes |",
                    "| :--- | :--- | :--- |",
                ]
            )
            for amb in ambiguous_edges:
                sections.append(
                    f"| `{amb.source}` | `{amb.target}` | {amb.conflict_notes or 'Contradictory directional edges detected'} |"
                )
            sections.append("")

        return "\n".join(sections)

    @classmethod
    def export_to_file(
        cls,
        report: SCMUnificationReport,
        out_path: Union[str, Path] = ".cig/SCM.md",
    ) -> Path:
        """Renders and writes the SCM markdown report to disk."""
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = cls.generate_markdown(report)
        path.write_text(content, encoding="utf-8")
        return path
