"""
GraphML Exporter for cig Structural Causal Models.
Serializes unified SCM graphs to standard GraphML format compatible with
Gephi, Cytoscape, NetworkX, and igraph.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from cig.passes.pass3_unify import SCMUnificationReport


class GraphMLExporter:
    """Exports SCMUnificationReport into valid GraphML XML."""

    GRAPHML_XMLNS = "http://graphml.graphdrawing.org/xmlns"
    XSI_XMLNS = "http://www.w3.org/2001/XMLSchema-instance"
    SCHEMA_LOC = "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"

    @classmethod
    def generate_graphml_xml(cls, report: SCMUnificationReport) -> str:
        """Constructs and returns the GraphML XML document string."""
        root = ET.Element(
            "graphml",
            {
                "xmlns": cls.GRAPHML_XMLNS,
                "xmlns:xsi": cls.XSI_XMLNS,
                "xsi:schemaLocation": cls.SCHEMA_LOC,
            },
        )

        # 1. Define Node Attribute Keys
        ET.SubElement(
            root,
            "key",
            {
                "id": "node_label",
                "for": "node",
                "attr.name": "label",
                "attr.type": "string",
            },
        )
        ET.SubElement(
            root,
            "key",
            {
                "id": "is_god_confounder",
                "for": "node",
                "attr.name": "is_god_confounder",
                "attr.type": "boolean",
            },
        )

        # 2. Define Edge Attribute Keys
        ET.SubElement(
            root,
            "key",
            {
                "id": "edge_type",
                "for": "edge",
                "attr.name": "edge_type",
                "attr.type": "string",
            },
        )
        ET.SubElement(
            root,
            "key",
            {
                "id": "confidence",
                "for": "edge",
                "attr.name": "confidence",
                "attr.type": "double",
            },
        )
        ET.SubElement(
            root,
            "key",
            {
                "id": "provenance",
                "for": "edge",
                "attr.name": "provenance",
                "attr.type": "string",
            },
        )
        ET.SubElement(
            root,
            "key",
            {
                "id": "mechanisms",
                "for": "edge",
                "attr.name": "mechanisms",
                "attr.type": "string",
            },
        )
        ET.SubElement(
            root,
            "key",
            {
                "id": "conflict_notes",
                "for": "edge",
                "attr.name": "conflict_notes",
                "attr.type": "string",
            },
        )

        # 3. Create Graph Element
        graph_elem = ET.SubElement(
            root,
            "graph",
            {
                "id": "SCM_DAG",
                "edgedefault": "directed",
            },
        )

        god_set = set(report.god_confounders or [])
        all_nodes: set[str] = set()

        for edge in report.edges:
            all_nodes.add(edge.source)
            all_nodes.add(edge.target)

        # 4. Populate Nodes
        for node_id in sorted(all_nodes):
            node_elem = ET.SubElement(graph_elem, "node", {"id": node_id})

            d_label = ET.SubElement(node_elem, "data", {"key": "node_label"})
            d_label.text = node_id

            d_god = ET.SubElement(node_elem, "data", {"key": "is_god_confounder"})
            d_god.text = "true" if node_id in god_set else "false"

        # 5. Populate Edges
        for idx, edge in enumerate(report.edges):
            edge_id = f"e{idx}_{edge.source}->{edge.target}"
            edge_elem = ET.SubElement(
                graph_elem,
                "edge",
                {
                    "id": edge_id,
                    "source": edge.source,
                    "target": edge.target,
                },
            )

            d_type = ET.SubElement(edge_elem, "data", {"key": "edge_type"})
            d_type.text = edge.edge_type

            d_conf = ET.SubElement(edge_elem, "data", {"key": "confidence"})
            d_conf.text = f"{edge.confidence:.4f}"

            if edge.provenance:
                d_prov = ET.SubElement(edge_elem, "data", {"key": "provenance"})
                d_prov.text = "; ".join(edge.provenance)

            if edge.mechanisms:
                d_mech = ET.SubElement(edge_elem, "data", {"key": "mechanisms"})
                d_mech.text = "; ".join(edge.mechanisms)

            if edge.conflict_notes:
                d_conf_note = ET.SubElement(
                    edge_elem, "data", {"key": "conflict_notes"}
                )
                d_conf_note.text = edge.conflict_notes

        # Format and indent XML
        ET.indent(root, space="  ", level=0)
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        return xml_declaration + ET.tostring(root, encoding="utf-8").decode("utf-8")

    @classmethod
    def export_to_file(
        cls,
        report: SCMUnificationReport,
        out_path: str | Path = ".cig/cig_graph.graphml",
    ) -> Path:
        """Writes the unified SCM as a GraphML file."""
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xml_text = cls.generate_graphml_xml(report)
        path.write_text(xml_text, encoding="utf-8")
        return path
