"""
OpenLineage SCM Exporter.
Serializes CIG's unified causal DAG into OpenLineage 1.0+ events with custom facets:
  - causalLineage: node roles, edge types (EXTRACTED, INFERRED, DISCOVERED), and confidences.
  - causalMechanism: identified parent-child structural equations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from cig.passes.pass3_unify import SCMUnificationReport


class OpenLineageProducer(BaseModel):
    name: str = "cig-causal-engine"
    version: str = "0.1.0"


class OpenLineageDatasetFacet(BaseModel):
    producer: str = "https://github.com/cig-project/cig"
    schemaURL: str = (
        "https://openlineage.io/spec/facets/1-0-0/CausalLineageDatasetFacet.json"
    )
    edge_type: str
    confidence: float
    provenance: str
    expression: str | None = None


class OpenLineageDataset(BaseModel):
    namespace: str = "cig://scm"
    name: str
    facets: dict[str, Any] = Field(default_factory=dict)


class OpenLineageRunEvent(BaseModel):
    eventType: str = "COMPLETE"
    eventTime: str
    producer: str = "https://github.com/cig-project/cig"
    schemaURL: str = "https://openlineage.io/spec/1-0-5/OpenLineage.json"
    inputs: list[OpenLineageDataset] = Field(default_factory=list)
    outputs: list[OpenLineageDataset] = Field(default_factory=list)


class OpenLineageExporter:
    """Exports SCM unification reports to OpenLineage RunEvent JSON."""

    @classmethod
    def to_openlineage_event(
        cls, report: SCMUnificationReport, job_namespace: str = "cig://governance"
    ) -> OpenLineageRunEvent:
        now_iso = datetime.now(timezone.utc).isoformat()

        # Collect distinct inputs (sources) and outputs (targets)
        inputs_map: dict[str, list[dict]] = {}
        outputs_map: dict[str, list[dict]] = {}

        for edge in report.edges:
            prov_str = (
                ", ".join(edge.provenance)
                if isinstance(edge.provenance, list)
                else str(edge.provenance)
            )
            expr_str = (
                "; ".join(edge.mechanisms)
                if getattr(edge, "mechanisms", None)
                else None
            )

            inputs_map.setdefault(edge.source, []).append(
                {
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "confidence": edge.confidence,
                    "provenance": prov_str,
                    "expression": expr_str,
                }
            )
            outputs_map.setdefault(edge.target, []).append(
                {
                    "source": edge.source,
                    "edge_type": edge.edge_type,
                    "confidence": edge.confidence,
                    "provenance": prov_str,
                    "expression": expr_str,
                }
            )

        inputs = [
            OpenLineageDataset(
                namespace=job_namespace,
                name=node_name,
                facets={
                    "causalDownstream": edges,
                    "isGodConfounder": node_name in report.god_confounders,
                },
            )
            for node_name, edges in inputs_map.items()
        ]

        outputs = [
            OpenLineageDataset(
                namespace=job_namespace,
                name=node_name,
                facets={
                    "causalUpstream": edges,
                    "isGodConfounder": node_name in report.god_confounders,
                },
            )
            for node_name, edges in outputs_map.items()
        ]

        return OpenLineageRunEvent(
            eventType="COMPLETE",
            eventTime=now_iso,
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def export_to_file(
        cls,
        report: SCMUnificationReport,
        out_path: str | Path = ".cig/openlineage_event.json",
    ) -> Path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        event = cls.to_openlineage_event(report)
        p.write_text(event.model_dump_json(indent=2), encoding="utf-8")
        return p
