"""
CIG (Causal Inference Graph) CLI Interface.
Provides command-line commands for scanning codebase lineages, compiling SCM DAGs,
running do-calculus causal estimations, mediation decompositions, and confounder sensitivity analysis.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cig import PyCausalEngine
from cig.artifacts.markdown_exporter import SCMMarkdownExporter
from cig.artifacts.graphml_exporter import GraphMLExporter
from cig.artifacts.visualizer import HTMLDAGVisualizer
from cig.passes.pass1_lineage import (
    DBTManifestParser,
    PythonLineageParser,
    SQLLineageParser,
)
from cig.passes.pass2_semantic import (
    DBTSchemaDocExtractor,
    SemanticCausalExtractor,
)
from cig.passes.pass3_unify import SCMGraphUnification
from cig.passes.pass4_engine import CausalEstimationEngine

console = Console()


def resolve_node_name(query: str, all_nodes: list[str]) -> str:
    """Fuzzy resolves a variable name from the node list if not exactly matched."""
    if query in all_nodes:
        return query

    query_lower = query.lower()
    for node in all_nodes:
        if node.lower() == query_lower:
            return node

    # Match column suffix if qualified (e.g., 'table.column' matching 'column')
    for node in all_nodes:
        if node.split(".")[-1].lower() == query_lower:
            return node

    return query


@click.group()
@click.version_option(version="0.1.0", prog_name="cig")
def main():
    """⚡ cig — Continuous Causal Inference Graph engine for modern data & ML stacks."""
    pass


@main.command(name="init")
def init_cmd():
    """Initialize .cig configuration directory and causal metadata scaffolding."""
    dot_cig = Path(".cig")
    dot_cig.mkdir(exist_ok=True)

    config_file = dot_cig / "config.json"
    if not config_file.exists():
        default_config = {
            "version": "0.1.0",
            "strict_mode": False,
            "scan_paths": [".", "models", "dbt", "notebooks"],
            "god_confounder_in_degree_threshold": 5,
        }
        config_file.write_text(json.dumps(default_config, indent=2), encoding="utf-8")

    console.print(
        "[bold green]✔ Initialized .cig directory and configuration.[/bold green]"
    )


@main.command(name="scan")
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
def scan_cmd(path: str):
    """Scan directory for SQL, Python, dbt models, and documentation to build the SCM."""
    base_path = Path(path)
    console.print(
        Panel(
            f"[bold cyan]Scanning repository for causal lineages and domain hypotheses:[/bold cyan] {base_path.resolve()}"
        )
    )

    unifier = SCMGraphUnification()
    sql_parser = SQLLineageParser()
    py_parser = PythonLineageParser()
    dbt_manifest_parser = DBTManifestParser(sql_parser=sql_parser)
    dbt_doc_extractor = DBTSchemaDocExtractor()
    semantic_extractor = SemanticCausalExtractor()

    # Discover candidate files
    sql_files = [
        p
        for p in base_path.rglob("*.sql")
        if ".venv" not in p.parts and ".cig" not in p.parts
    ]
    py_files = [
        p
        for p in base_path.rglob("*.py")
        if ".venv" not in p.parts and ".cig" not in p.parts and "test" not in p.parts
    ]
    md_files = [
        p
        for p in base_path.rglob("*.md")
        if ".venv" not in p.parts and ".cig" not in p.parts
    ]
    yml_files = [
        p
        for p in base_path.rglob("*.yml")
        if ".venv" not in p.parts and ".cig" not in p.parts
    ]
    manifest_files = [
        p
        for p in base_path.rglob("manifest.json")
        if ".venv" not in p.parts and ".cig" not in p.parts
    ]

    # Pass 1: dbt Manifests
    for mf in manifest_files:
        try:
            edges = dbt_manifest_parser.parse_file(mf)
            if edges:
                unifier.add_extracted_edges(edges)
                console.print(
                    f"[bold green]✔ Ingested dbt manifest:[/bold green] {mf} ({len(edges)} lineages)"
                )
        except Exception as e:
            console.print(
                f"[dim yellow]⚠ Skipped dbt manifest {mf.name}: {e}[/dim yellow]"
            )

    # Pass 1: Standalone SQL
    for sf in sql_files:
        try:
            edges = sql_parser.parse_file(sf)
            if edges:
                unifier.add_extracted_edges(edges)
                console.print(
                    f"[bold green]✔ Parsed SQL lineage:[/bold green] {sf} ({len(edges)} lineages)"
                )
        except Exception as e:
            console.print(f"[dim yellow]⚠ Skipped SQL {sf.name}: {e}[/dim yellow]")

    # Pass 1: Python DataFrames AST
    for pf in py_files:
        try:
            edges = py_parser.parse_file(pf)
            if edges:
                unifier.add_extracted_edges(edges)
                console.print(
                    f"[bold green]✔ Parsed Python AST:[/bold green] {pf} ({len(edges)} lineages)"
                )
        except Exception as e:
            console.print(f"[dim yellow]⚠ Skipped Python {pf.name}: {e}[/dim yellow]")

    # Pass 2: dbt schema.yml & docs blocks
    for yf in yml_files:
        try:
            edges = dbt_doc_extractor.parse_yaml_file(yf)
            if edges:
                unifier.add_inferred_edges(edges)
                console.print(
                    f"[bold purple]✔ Ingested dbt schema docs:[/bold purple] {yf} ({len(edges)} causal hypotheses)"
                )
        except Exception as e:
            console.print(f"[dim yellow]⚠ Skipped YAML {yf.name}: {e}[/dim yellow]")

    # Pass 2: Markdown & Literature Documentation
    for mf in md_files:
        try:
            edges = semantic_extractor.parse_file(mf)
            if edges:
                unifier.add_inferred_edges(edges)
                console.print(
                    f"[bold purple]✔ Extracted domain hypotheses:[/bold purple] {mf} ({len(edges)} hypotheses)"
                )
        except Exception as e:
            console.print(f"[dim yellow]⚠ Skipped Markdown {mf.name}: {e}[/dim yellow]")

    # Pass 3: Unification & God Confounder Detection
    report = unifier.unify()

    # Export Artifacts
    dot_cig_dir = base_path / ".cig"
    dot_cig_dir.mkdir(exist_ok=True)

    json_path = dot_cig_dir / "cig_graph.json"
    html_path = dot_cig_dir / "cig.html"
    md_path = dot_cig_dir / "SCM.md"
    graphml_path = dot_cig_dir / "cig_graph.graphml"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    HTMLDAGVisualizer.generate_html(report, out_path=html_path)
    SCMMarkdownExporter.export_to_file(report, out_path=md_path)
    GraphMLExporter.export_to_file(report, out_path=graphml_path)

    table = Table(
        title="SCM Unification Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")
    table.add_row("Total Variables (Nodes)", str(report.total_nodes))
    table.add_row("Total Causal Edges", str(report.total_edges))
    table.add_row("Deterministic Lineages (EXTRACTED)", str(report.extracted_edges))
    table.add_row("Domain Hypotheses (INFERRED)", str(report.inferred_edges))
    table.add_row("Ambiguous / Conflicted Edges", str(report.ambiguous_edges))
    table.add_row("God Confounders (Hub Nodes)", str(len(report.god_confounders)))

    console.print(table)
    console.print(f"[bold green]✔ SCM Graph JSON:[/bold green] {json_path}")
    console.print(
        f"[bold green]✔ SCM GraphML (Gephi/Cytoscape):[/bold green] {graphml_path}"
    )
    console.print(f"[bold green]✔ Interactive visualizer:[/bold green] {html_path}")
    console.print(
        f"[bold green]✔ SCM Markdown & Mermaid documentation:[/bold green] {md_path}"
    )


@main.command(name="do")
@click.option(
    "--treatment", "-t", required=True, help="Treatment / Intervention variable name"
)
@click.option("--outcome", "-y", required=True, help="Outcome variable name")
@click.option(
    "--mediator",
    "-med",
    default=None,
    help="Optional mediator variable for NDE/NIE decomposition",
)
@click.option(
    "--sensitivity",
    "-s",
    is_flag=True,
    help="Run Cinelli-Hazlett & E-value sensitivity analysis for unobserved confounding",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, file_okay=True),
    help="Path to CSV dataset for causal effect estimation",
)
@click.option(
    "--method",
    "-m",
    default="backdoor.linear_regression",
    type=click.Choice(
        [
            "backdoor.linear_regression",
            "backdoor.econml.dml.LinearDML",
            "backdoor.econml.dml.CausalForestDML",
            "iv.instrumental_variable",
        ],
        case_sensitive=False,
    ),
    help="Estimation algorithm",
)
def do_cmd(
    treatment: str,
    outcome: str,
    mediator: str | None,
    sensitivity: bool,
    data: str | None,
    method: str,
):
    """Identify adjustment sets and estimate causal effect, mediation, or sensitivity bounds."""
    graph_json_path = Path(".cig/cig_graph.json")
    if not graph_json_path.exists():
        console.print(
            "[bold red]Error:[/bold red] No SCM found. Run [bold cyan]cig scan[/bold cyan] first."
        )
        sys.exit(1)

    # Load SCM into engine
    engine = PyCausalEngine()
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    for edge in graph_data.get("edges", []):
        if edge["edge_type"] == "EXTRACTED":
            engine.add_extracted_edge(edge["source"], edge["target"], None, None)
        elif edge["edge_type"] == "INFERRED":
            engine.add_inferred_edge(
                edge["source"], edge["target"], edge["confidence"], None, None
            )

    all_nodes = [node["name"] for node in json.loads(engine.export_json())["nodes"]]
    t_resolved = resolve_node_name(treatment, all_nodes)
    y_resolved = resolve_node_name(outcome, all_nodes)

    estimator = CausalEstimationEngine(engine)

    if not data:
        # DAG Identification Query Only
        backdoors = engine.find_backdoor_adjustment_sets(t_resolved, y_resolved)
        frontdoors = engine.find_frontdoor_adjustment_sets(t_resolved, y_resolved)
        instruments = engine.find_instrumental_variables(t_resolved, y_resolved)

        console.print(
            Panel(
                f"[bold]Treatment:[/bold] {t_resolved}\n"
                f"[bold]Outcome:[/bold] {y_resolved}\n\n"
                f"[bold green]Backdoor Adjustment Sets:[/bold green] {backdoors or 'None'}\n"
                f"[bold cyan]Frontdoor Adjustment Sets:[/bold cyan] {frontdoors or 'None'}\n"
                f"[bold yellow]Instrumental Variables:[/bold yellow] {instruments or 'None'}",
                title="🔍 Causal Identification Strategy",
                border_style="blue",
            )
        )
        return

    # Data Provided: Run Estimation / Mediation / Sensitivity
    df = pd.read_csv(data)
    if y_resolved not in df.columns and outcome in df.columns:
        df[y_resolved] = df[outcome]
    if t_resolved not in df.columns and treatment in df.columns:
        df[t_resolved] = df[treatment]

    if mediator:
        med_resolved = resolve_node_name(mediator, all_nodes)
        if med_resolved not in df.columns and mediator in df.columns:
            df[med_resolved] = df[mediator]

        med_summary = estimator.decompose_mediation(
            df,
            treatment=t_resolved,
            mediator=med_resolved,
            outcome=y_resolved,
        )

        console.print(
            Panel(
                f"[bold]Total Effect (TE):[/bold] [bold green]{med_summary.total_effect:.4f}[/bold green]\n"
                f"[bold]Natural Direct Effect (NDE):[/bold] {med_summary.natural_direct_effect:.4f}\n"
                f"[bold]Natural Indirect Effect (NIE):[/bold] {med_summary.natural_indirect_effect:.4f}\n"
                f"[bold]Proportion Mediated:[/bold] [bold cyan]{med_summary.proportion_mediated * 100:.1f}%[/bold cyan]\n"
                f"[bold]Confounders Adjusted:[/bold] {med_summary.confounders or 'None'}",
                title="🧬 Pearl's Mediation Decomposition",
                border_style="cyan",
            )
        )
        return

    if sensitivity:
        sens = estimator.compute_sensitivity_analysis(
            df, treatment=t_resolved, outcome=y_resolved
        )
        console.print(
            Panel(
                f"[bold]Cinelli-Hazlett Robustness Value (RV_q=1.0):[/bold] [bold cyan]{sens.rv_q * 100:.1f}%[/bold cyan]\n"
                f"[bold]Robustness Value for Significance (RV_α=0.05):[/bold] [bold cyan]{sens.rv_alpha * 100:.1f}%[/bold cyan]\n"
                f"[bold]VanderWeele-Ding E-Value:[/bold] [bold green]{sens.e_value:.2f}[/bold green]\n"
                f"[bold]Point Estimate / SE:[/bold] {sens.point_estimate:.4f} (SE: {sens.standard_error:.4f}, t: {sens.t_statistic:.2f})\n\n"
                f"[dim]{sens.interpretation}[/dim]",
                title="🛡️ Confounder Sensitivity Analysis",
                border_style="yellow",
            )
        )
        return

    # Standard ATE Estimation
    summary = estimator.estimate_effect(
        df, treatment=t_resolved, outcome=y_resolved, estimator_method=method
    )
    ci_str = (
        f"[{summary.confidence_intervals[0]:.4f}, {summary.confidence_intervals[1]:.4f}]"
        if summary.confidence_intervals
        else "N/A"
    )

    console.print(
        Panel(
            f"[bold]Estimated Average Treatment Effect (ATE):[/bold] [bold green]{summary.value:.4f}[/bold green]\n"
            f"[bold]Confidence Interval (95%):[/bold] {ci_str}\n"
            f"[bold]Identification Strategy:[/bold] {summary.identification_method}\n"
            f"[bold]Conditioned Covariates:[/bold] {summary.adjustment_set or 'None'}\n"
            f"[bold]Estimator Algorithm:[/bold] {summary.estimator_method}",
            title="📊 Causal Estimation Result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
