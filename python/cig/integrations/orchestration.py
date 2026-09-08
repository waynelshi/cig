"""
Orchestration integration hooks for Airflow and Dagster.
Enables pipeline assertions on causal invariance and identification validity.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from cig import PyCausalEngine
from cig.passes.pass7_monitor import CausalInvarianceMonitor


class CausalOrchestrationGuard:
    """Common interface for orchestrator hooks (Airflow / Dagster)."""

    def __init__(self, cig_graph_path: str | Path = ".cig/cig_graph.json"):
        self.cig_graph_path = Path(cig_graph_path)
        if not self.cig_graph_path.exists():
            raise FileNotFoundError(
                f"SCM graph artifact not found at {self.cig_graph_path}"
            )

        self.engine = PyCausalEngine()
        with open(self.cig_graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        for edge in graph_data.get("edges", []):
            if isinstance(edge, (list, tuple)):
                self.engine.add_extracted_edge(edge[0], edge[1], None, None)
            elif edge.get("edge_type") == "EXTRACTED":
                self.engine.add_extracted_edge(
                    edge["source"], edge["target"], None, None
                )
            else:
                self.engine.add_inferred_edge(
                    edge["source"],
                    edge["target"],
                    float(edge.get("confidence", 1.0)),
                    None,
                    None,
                )

    def verify_invariance(
        self,
        df: pd.DataFrame,
        time_col: str | None = None,
        alpha: float = 0.05,
        target_nodes: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Audits structural invariance across data partitions."""
        monitor = CausalInvarianceMonitor(self.engine, significance_level=alpha)
        report = monitor.audit_temporal_invariance(
            df=df, time_column=time_col, target_nodes=target_nodes
        )

        if report.broken_mechanisms > 0:
            broken_details = [
                f"{r.target} (p={r.p_value:.4e})"
                for r in report.results
                if not r.is_invariant
            ]
            msg = (
                f"Causal Mechanism Breakdown! {report.broken_mechanisms} broken mechanism(s): "
                f"{', '.join(broken_details)}"
            )
            return False, msg

        return (
            True,
            f"All {report.total_mechanisms_audited} structural mechanisms invariant.",
        )


def make_dagster_asset_check(
    asset_name: str,
    cig_graph_path: str | Path = ".cig/cig_graph.json",
    time_col: str | None = None,
    alpha: float = 0.05,
):
    """
    Returns a Dagster AssetCheckFunction if dagster is installed.
    Raises ImportError if dagster is not available.
    """
    try:
        from dagster import AssetCheckResult, asset_check
    except ImportError:
        raise ImportError(
            "Dagster is required to use make_dagster_asset_check. Install with: uv add dagster"
        )

    guard = CausalOrchestrationGuard(cig_graph_path)

    @asset_check(asset=asset_name, name=f"{asset_name}_causal_invariance")
    def causal_check(context, asset_value: pd.DataFrame) -> AssetCheckResult:
        passed, description = guard.verify_invariance(
            asset_value, time_col=time_col, alpha=alpha
        )
        return AssetCheckResult(passed=passed, description=description)

    return causal_check


def airflow_causal_invariance_operator(
    df: pd.DataFrame,
    cig_graph_path: str | Path = ".cig/cig_graph.json",
    time_col: str | None = None,
    alpha: float = 0.05,
):
    """
    Callable suitable for Airflow PythonOperator or @task decorator.
    Raises ValueError on mechanism breakdown to halt the DAG run.
    """
    guard = CausalOrchestrationGuard(cig_graph_path)
    passed, msg = guard.verify_invariance(df, time_col=time_col, alpha=alpha)
    if not passed:
        raise ValueError(f"[Airflow Causal Gate Failure] {msg}")
    return msg
