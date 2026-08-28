"""
Pass 3: SCM Graph Unification & Conflict Resolution.
Consolidates EXTRACTED and INFERRED edges, detects cycles/contradictions,
flags AMBIGUOUS relationships, and produces .cig artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from pydantic import BaseModel, Field

from cig import PyCausalEngine
from cig.passes.pass1_lineage import ExtractedEdge
from cig.passes.pass2_semantic import InferredEdge


class UnifiedEdge(BaseModel):
    source: str
    target: str
    edge_type: str = Field(description="'EXTRACTED', 'INFERRED', or 'AMBIGUOUS'")
    confidence: float
    provenance: list[str] = Field(default_factory=list)
    mechanisms: list[str] = Field(default_factory=list)
    conflict_notes: str | None = None


class SCMUnificationReport(BaseModel):
    total_nodes: int
    total_edges: int
    extracted_edges: int
    inferred_edges: int
    ambiguous_edges: int
    god_confounders: list[str] = Field(
        description="Hub nodes with high out-degree influencing multiple variables"
    )
    edges: list[UnifiedEdge]


class SCMGraphUnification:
    def __init__(self):
        self.raw_edges: list[ExtractedEdge | InferredEdge] = []
        self.unified_edges: dict[tuple[str, str], UnifiedEdge] = {}

    def add_extracted_edges(self, edges: list[ExtractedEdge]) -> None:
        self.raw_edges.extend(edges)

    def add_inferred_edges(self, edges: list[InferredEdge]) -> None:
        self.raw_edges.extend(edges)

    def unify(self) -> SCMUnificationReport:
        """
        Consolidates edges, resolves conflicts, and produces the unified SCM.
        """
        grouped: dict[tuple[str, str], list[ExtractedEdge | InferredEdge]] = (
            defaultdict(list)
        )

        for e in self.raw_edges:
            grouped[(e.source, e.target)].append(e)

        self.unified_edges.clear()

        # 1. Merge parallel edges going the same direction (A -> B)
        for (u, v), edge_list in grouped.items():
            extracted = [e for e in edge_list if e.edge_type == "EXTRACTED"]
            inferred = [e for e in edge_list if e.edge_type == "INFERRED"]

            provenance = []
            mechanisms = []

            for e in edge_list:
                if isinstance(e, ExtractedEdge):
                    provenance.append(f"[AST/SQL] {e.provenance}")
                    if e.expression:
                        mechanisms.append(f"Formula: {e.expression}")
                elif isinstance(e, InferredEdge):
                    provenance.append(f"[Doc/LLM] {e.provenance}")
                    if e.rationale:
                        mechanisms.append(f"Rationale: {e.rationale}")

            if extracted:
                # Deterministic wins
                self.unified_edges[(u, v)] = UnifiedEdge(
                    source=u,
                    target=v,
                    edge_type="EXTRACTED",
                    confidence=1.0,
                    provenance=list(set(provenance)),
                    mechanisms=list(set(mechanisms)),
                )
            else:
                # Combine multiple inferred confidences: 1 - product(1 - c)
                confidences = [e.confidence for e in inferred]
                combined_conf = 1.0
                for c in confidences:
                    combined_conf *= 1.0 - c
                final_conf = round(1.0 - combined_conf, 3)

                self.unified_edges[(u, v)] = UnifiedEdge(
                    source=u,
                    target=v,
                    edge_type="INFERRED",
                    confidence=final_conf,
                    provenance=list(set(provenance)),
                    mechanisms=list(set(mechanisms)),
                )

        # 2. Detect 2-cycle contradictions: (A -> B) vs (B -> A)
        checked_pairs: set[tuple[str, str]] = set()
        for (u, v), edge_uv in list(self.unified_edges.items()):
            if (v, u) in self.unified_edges and (u, v) not in checked_pairs:
                edge_vu = self.unified_edges[(v, u)]
                checked_pairs.add((u, v))
                checked_pairs.add((v, u))

                # Conflict resolution:
                # If one is EXTRACTED and the other is INFERRED, EXTRACTED overrides
                if edge_uv.edge_type == "EXTRACTED" and edge_vu.edge_type == "INFERRED":
                    del self.unified_edges[(v, u)]
                elif (
                    edge_vu.edge_type == "EXTRACTED" and edge_uv.edge_type == "INFERRED"
                ):
                    del self.unified_edges[(u, v)]
                else:
                    # Both INFERRED -> Contradiction! Mark both as AMBIGUOUS
                    edge_uv.edge_type = "AMBIGUOUS"
                    edge_uv.conflict_notes = (
                        f"Contradicted by reverse edge ({v} -> {u})"
                    )
                    edge_vu.edge_type = "AMBIGUOUS"
                    edge_vu.conflict_notes = (
                        f"Contradicted by reverse edge ({u} -> {v})"
                    )

        # 3. Analyze node out-degrees to identify God Confounders
        out_degrees: dict[str, int] = defaultdict(int)
        all_nodes: set[str] = set()
        for u, v in self.unified_edges.keys():
            all_nodes.add(u)
            all_nodes.add(v)
            out_degrees[u] += 1

        # Nodes that affect 3 or more variables are candidate god confounders
        god_confounders = [node for node, deg in out_degrees.items() if deg >= 3]

        extracted_cnt = sum(
            1 for e in self.unified_edges.values() if e.edge_type == "EXTRACTED"
        )
        inferred_cnt = sum(
            1 for e in self.unified_edges.values() if e.edge_type == "INFERRED"
        )
        ambiguous_cnt = sum(
            1 for e in self.unified_edges.values() if e.edge_type == "AMBIGUOUS"
        )

        return SCMUnificationReport(
            total_nodes=len(all_nodes),
            total_edges=len(self.unified_edges),
            extracted_edges=extracted_cnt,
            inferred_edges=inferred_cnt,
            ambiguous_edges=ambiguous_cnt,
            god_confounders=god_confounders,
            edges=list(self.unified_edges.values()),
        )

    def populate_rust_engine(self, engine: PyCausalEngine) -> int:
        """Loads all unified non-ambiguous edges into the compiled Rust cig_core engine."""
        loaded = 0
        for edge in self.unified_edges.values():
            if edge.edge_type == "EXTRACTED":
                engine.add_extracted_edge(
                    edge.source,
                    edge.target,
                    "; ".join(edge.provenance) if edge.provenance else None,
                    "; ".join(edge.mechanisms) if edge.mechanisms else None,
                )
                loaded += 1
            elif edge.edge_type == "INFERRED":
                engine.add_inferred_edge(
                    edge.source,
                    edge.target,
                    edge.confidence,
                    "; ".join(edge.provenance) if edge.provenance else None,
                    "; ".join(edge.mechanisms) if edge.mechanisms else None,
                )
                loaded += 1
            # Ambiguous edges are intentionally withheld from automated d-sep until resolved
        return loaded

    def export_artifacts(self, out_dir: str | Path = ".cig") -> None:
        """Writes .cig/cig_graph.json and CIG_REPORT.md"""
        report = self.unify()
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Write JSON artifact
        json_path = path / "cig_graph.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        # 2. Write Markdown summary report
        md_path = path / "CIG_REPORT.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# CIG Structural Causal Model Report\n\n")
            f.write(f"- **Total Variables (Nodes):** {report.total_nodes}\n")
            f.write(f"- **Total Causal Connections:** {report.total_edges}\n")
            f.write(f"- **Deterministic (`EXTRACTED`):** {report.extracted_edges}\n")
            f.write(f"- **Hypothesized (`INFERRED`):** {report.inferred_edges}\n")
            f.write(f"- **Contradictions (`AMBIGUOUS`):** {report.ambiguous_edges}\n\n")

            if report.god_confounders:
                f.write("### 🚨 Key Confounder Hubs ('God Nodes')\n")
                for hub in report.god_confounders:
                    f.write(
                        f"- **`{hub}`** (Acts on multiple downstream pathways; high risk of confounding)\n"
                    )
                f.write("\n")

            if report.ambiguous_edges > 0:
                f.write("### ⚠️ Ambiguous / Conflicting Relationships\n")
                for e in report.edges:
                    if e.edge_type == "AMBIGUOUS":
                        f.write(f"- `{e.source}` ⟷ `{e.target}`: {e.conflict_notes}\n")
                f.write("\n")

            f.write("### 📋 Unified Causal Edge Catalog\n\n")
            f.write("| Source | Target | Type | Confidence | Provenance |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for e in report.edges:
                prov = "<br>".join(e.provenance) if e.provenance else "-"
                f.write(
                    f"| `{e.source}` | `{e.target}` | **`{e.edge_type}`** | {e.confidence:.2f} | {prov} |\n"
                )
