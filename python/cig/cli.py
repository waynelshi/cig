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
    ExtractedEdge,
    PythonLineageParser,
    SQLLineageParser,
    SQLMeshManifestParser,
)
from cig.passes.pass2_semantic import (
    DBTSchemaDocExtractor,
    SemanticCausalExtractor,
)
from cig.passes.pass2_discovery import CausalDiscoveryEngine
from cig.passes.pass3_unify import SCMGraphUnification, SCMUnificationReport
from cig.passes.pass4_engine import CausalEstimationEngine
from cig.passes.pass5_counterfactual import StructuralCounterfactualEngine
from cig.passes.pass6_active import ActiveInferenceAgent
from cig.passes.pass4b_cate import PolicyTreeEngine
from cig.passes.pass7_monitor import CausalInvarianceMonitor
from cig.artifacts.openlineage_exporter import OpenLineageExporter

console = Console()


def find_graph_path(
    custom_path: str | None = None, data_path: str | None = None
) -> Path:
    """Finds cig_graph.json in current dir, via custom arg, or relative to the data path."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    # 1. Check current working directory
    cwd_path = Path(".cig/cig_graph.json")
    if cwd_path.exists():
        return cwd_path

    # 2. Check parent folder of the provided dataset
    if data_path:
        data_parent = Path(data_path).resolve().parent
        for candidate in [
            data_parent / ".cig" / "cig_graph.json",
            data_parent.parent / ".cig" / "cig_graph.json",
        ]:
            if candidate.exists():
                return candidate

    return cwd_path


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
    openlineage_path = dot_cig_dir / "openlineage_event.json"
    OpenLineageExporter.export_to_file(report, out_path=openlineage_path)

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
    console.print(
        f"[bold green]✔ OpenLineage Event JSON:[/bold green] {openlineage_path}"
    )


@main.command(name="discover")
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=True),
    help="CSV dataset for causal discovery",
)
@click.option(
    "--alpha",
    "-a",
    default=0.05,
    type=float,
    help="Significance threshold for conditional independence",
)
@click.option(
    "--max-cond", "-m", default=2, type=int, help="Maximum conditioning set size"
)
def discover_cmd(data: str, alpha: float, max_cond: int):
    """Discovers causal skeleton and orientations directly from observational distributions."""
    df = pd.read_csv(data)
    engine = CausalDiscoveryEngine(significance_level=alpha, max_cond_size=max_cond)

    console.print(
        f"[bold cyan]🔍 Running Continuous Causal Discovery on {data}...[/bold cyan]"
    )
    discovered_edges = engine.discover_edges(df)

    if not discovered_edges:
        console.print(
            "[yellow]No statistically significant causal edges detected at this alpha threshold.[/yellow]"
        )
        return

    table = Table(
        title=f"Data-Discovered Causal Edges (α = {alpha})",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Learned Edge", style="dim")
    table.add_column("Confidence (1 - p)", justify="right")
    table.add_column("Partial Correlation", justify="right")
    table.add_column("p-value", justify="right")

    for e in discovered_edges:
        table.add_row(
            f"{e.source} -> {e.target}",
            f"{e.confidence:.4f}",
            f"{e.partial_corr:+.4f}",
            f"{e.p_value:.6f}",
        )

    console.print(table)


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


@main.command(name="counterfactual")
@click.option("--treatment", "-t", required=True, help="Treatment variable name")
@click.option(
    "--value",
    "-v",
    required=True,
    type=float,
    help="Counterfactual intervention target value",
)
@click.option("--outcome", "-y", required=True, help="Outcome variable to evaluate")
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=True),
    help="CSV dataset containing observational records",
)
def counterfactual_cmd(treatment: str, value: float, outcome: str, data: str):
    """Pearl's Level 3 Counterfactual inference: Inverts background noise (Abduction) and predicts Y_{T=v}."""
    graph_json_path = Path(".cig/cig_graph.json")
    if not graph_json_path.exists():
        console.print(
            "[bold red]Error:[/bold red] No SCM found. Run [bold cyan]cig scan[/bold cyan] first."
        )
        sys.exit(1)

    engine = PyCausalEngine()
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    for edge in graph_data.get("edges", []):
        if edge["edge_type"] == "EXTRACTED":
            engine.add_extracted_edge(edge["source"], edge["target"], None, None)
        else:
            engine.add_inferred_edge(
                edge["source"], edge["target"], edge["confidence"], None, None
            )

    all_nodes = [node["name"] for node in json.loads(engine.export_json())["nodes"]]
    t_resolved = resolve_node_name(treatment, all_nodes)
    y_resolved = resolve_node_name(outcome, all_nodes)

    df = pd.read_csv(data)
    if y_resolved not in df.columns and outcome in df.columns:
        df[y_resolved] = df[outcome]
    if t_resolved not in df.columns and treatment in df.columns:
        df[t_resolved] = df[treatment]

    cf_engine = StructuralCounterfactualEngine(engine)
    summary = cf_engine.fit_and_compute_counterfactual(
        df=df,
        treatment=t_resolved,
        intervention_value=value,
        outcome=y_resolved,
    )

    console.print(
        Panel(
            f"[bold]Target Outcome:[/bold] {summary.outcome}\n"
            f"[bold]Intervention Policy:[/bold] do({summary.treatment} = {summary.intervention_value})\n"
            f"[bold]Sample Size Analyzed:[/bold] N = {summary.sample_size}\n\n"
            f"[bold cyan]Observed Mean Outcome:[/bold cyan] {summary.mean_factual_outcome:.4f}\n"
            f"[bold green]Counterfactual Mean Outcome:[/bold green] {summary.mean_counterfactual_outcome:.4f}\n"
            f"[bold yellow]Net Policy Delta (Effect on Treated):[/bold yellow] [bold]{summary.average_treatment_effect_on_treated:+.4f}[/bold]",
            title="🔮 Pearl Level 3 Counterfactual Simulator",
            border_style="magenta",
        )
    )

    table = Table(
        title="Sample Unit-Level Counterfactual Simulations",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Unit ID", justify="center")
    table.add_column("Factual Outcome (Y)", justify="right")
    table.add_column(
        f"Counterfactual Y_{{{summary.treatment}={summary.intervention_value}}}",
        justify="right",
    )
    table.add_column("Individual Causal Gain (Δ)", justify="right")

    for idx, unit in enumerate(summary.unit_level_samples, 1):
        diff_color = "green" if unit.individual_causal_effect >= 0 else "red"
        table.add_row(
            f"#{idx}",
            f"{unit.predicted_factual_outcome:.2f}",
            f"{unit.counterfactual_outcome:.2f}",
            f"[{diff_color}]{unit.individual_causal_effect:+.2f}[/{diff_color}]",
        )

    console.print(table)


@main.command(name="active")
@click.option(
    "--outcome", "-y", required=True, help="Target business variable or goal to steer"
)
@click.option(
    "--target-val",
    "-v",
    required=True,
    type=float,
    help="Desired setpoint value for outcome",
)
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=True),
    help="CSV observational data",
)
@click.option("--graph", "-g", default=None, help="Path to .cig/cig_graph.json")
def active_inference_cmd(outcome: str, target_val: float, data: str, graph: str | None):
    graph_json_path = find_graph_path(graph, data)
    if not graph_json_path.exists():
        console.print(
            "[bold red]Error:[/bold red] No SCM found. Run [bold cyan]cig scan[/bold cyan] first."
        )
        sys.exit(1)

    with open(graph_json_path, "r", encoding="utf-8") as f:
        report = SCMUnificationReport.model_validate_json(f.read())

    engine = PyCausalEngine()
    for edge in report.edges:
        if edge.edge_type == "EXTRACTED":
            engine.add_extracted_edge(edge.source, edge.target, None, None)
        else:
            engine.add_inferred_edge(
                edge.source, edge.target, edge.confidence, None, None
            )

    df = pd.read_csv(data)
    agent = ActiveInferenceAgent(engine=engine, report=report)

    # Automatically derive candidate policies from upstream actionable parents
    all_nodes = [node["name"] for node in json.loads(engine.export_json())["nodes"]]
    y_resolved = resolve_node_name(outcome, all_nodes)

    candidate_policies = {
        "conservative_baseline": {"discount_pct": 0.05},
        "aggressive_growth": {"discount_pct": 0.20},
        "epistemic_intervention": {"marketing_tier": 1.0},
    }

    plan = agent.plan(
        candidate_interventions=candidate_policies,
        target_outcome=y_resolved,
        desired_value=target_val,
        df=df,
    )

    console.print(
        Panel(
            f"[bold]Target Outcome:[/bold] {y_resolved} (Goal: {target_val})\n"
            f"[bold]Identified Epistemic Gaps (Uncertainty):[/bold] {len(plan.epistemic_targets)} edge(s)\n"
            f"[bold cyan]Optimal Policy Selected:[/bold cyan] {plan.optimal_policy.policy_name if plan.optimal_policy else 'None'}\n\n"
            f"{plan.agent_rationale}",
            title="🧠 Active Inference Agent (Free Energy Minimization)",
            border_style="cyan",
        )
    )

    if plan.epistemic_targets:
        ep_table = Table(
            title="Epistemic Uncertainty Gaps in SCM DAG",
            show_header=True,
            header_style="bold yellow",
        )
        ep_table.add_column("Edge", style="dim")
        ep_table.add_column("Type")
        ep_table.add_column("Entropy Score", justify="right")
        ep_table.add_column("Recommended Diagnostic Experiment")

        for t in plan.epistemic_targets:
            ep_table.add_row(
                f"{t.source} -> {t.target}",
                t.edge_type,
                f"{t.uncertainty_score:.3f}",
                t.suggested_experiment,
            )
        console.print(ep_table)

    pol_table = Table(
        title="Policy Arbitration via Expected Free Energy (EFE)",
        show_header=True,
        header_style="bold green",
    )
    pol_table.add_column("Policy")
    pol_table.add_column("Intervention")
    pol_table.add_column("Expected Outcome", justify="right")
    pol_table.add_column("Pragmatic Value", justify="right")
    pol_table.add_column("Epistemic Value", justify="right")
    pol_table.add_column("Expected Free Energy (G)", justify="right")

    for pol in plan.evaluated_policies:
        pol_table.add_row(
            pol.policy_name,
            str(pol.intervention),
            f"{pol.expected_outcome:.2f}",
            f"{pol.pragmatic_value:.3f}",
            f"{pol.epistemic_value:.3f}",
            f"[bold]{pol.expected_free_energy:.3f}[/bold]",
        )
    console.print(pol_table)


@main.command(name="run")
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True, file_okay=True),
    default=None,
    help="Path to dbt manifest.json or SQLMesh state JSON",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, file_okay=True),
    default=None,
    help="Optional CSV dataset for continuous discovery, ATE estimation, and Active Inference",
)
@click.option(
    "--treatment",
    "-t",
    default=None,
    help="Primary treatment variable to evaluate",
)
@click.option(
    "--outcome",
    "-y",
    default=None,
    help="Primary target outcome variable to steer",
)
@click.option(
    "--goal",
    "-g",
    default=None,
    type=float,
    help="Target setpoint goal for Active Inference (EFE minimization)",
)
def run_pipeline_cmd(
    path: str,
    manifest: str | None,
    data: str | None,
    treatment: str | None,
    outcome: str | None,
    goal: float | None,
):
    """🚀 Run end-to-end multi-pass CIG causal pipeline: Scan -> Discover -> Unify -> Estimate -> Active Inference."""
    base_path = Path(path)
    console.print(
        Panel(
            f"[bold cyan]⚡ Running End-to-End CIG Causal Pipeline on:[/bold cyan] {base_path.resolve()}",
            border_style="cyan",
        )
    )

    unifier = SCMGraphUnification()
    sql_parser = SQLLineageParser()
    py_parser = PythonLineageParser()
    dbt_manifest_parser = DBTManifestParser(sql_parser=sql_parser)
    dbt_doc_extractor = DBTSchemaDocExtractor()
    semantic_extractor = SemanticCausalExtractor()

    if manifest:
        m_path = Path(manifest)
        try:
            m_data = json.loads(m_path.read_text(encoding="utf-8"))
            if "models" in m_data and "nodes" not in m_data:
                m_edges = SQLMeshManifestParser().parse_manifest_dict(
                    m_data, provenance=str(m_path)
                )
            else:
                m_edges = DBTManifestParser().parse_manifest_dict(
                    m_data, provenance=str(m_path)
                )
            unifier.add_extracted_edges(m_edges)
            console.print(
                f"[bold green]✔ Ingested {len(m_edges)} lineage edges from manifest:[/bold green] {m_path.name}"
            )
        except Exception as ex:
            console.print(f"[yellow]⚠ Failed parsing manifest {m_path}: {ex}[/yellow]")

    # Pass 1: Lineage Extraction
    for mf in base_path.rglob("manifest.json"):
        if ".venv" not in mf.parts and ".cig" not in mf.parts:
            try:
                edges = dbt_manifest_parser.parse_file(mf)
                if edges:
                    unifier.add_extracted_edges(edges)
            except Exception:
                pass

    for sf in base_path.rglob("*.sql"):
        if ".venv" not in sf.parts and ".cig" not in sf.parts:
            try:
                edges = sql_parser.parse_file(sf)
                if edges:
                    unifier.add_extracted_edges(edges)
            except Exception:
                pass

    for pf in base_path.rglob("*.py"):
        if (
            ".venv" not in pf.parts
            and ".cig" not in pf.parts
            and "test" not in pf.parts
        ):
            try:
                edges = py_parser.parse_file(pf)
                if edges:
                    unifier.add_extracted_edges(edges)
            except Exception:
                pass

    # Pass 2: Semantic Domain Extraction
    for yf in base_path.rglob("*.yml"):
        if ".venv" not in yf.parts and ".cig" not in yf.parts:
            try:
                edges = dbt_doc_extractor.parse_yaml_file(yf)
                if edges:
                    unifier.add_inferred_edges(edges)
            except Exception:
                pass

    for mf in base_path.rglob("*.md"):
        if ".venv" not in mf.parts and ".cig" not in mf.parts:
            try:
                edges = semantic_extractor.parse_file(mf)
                if edges:
                    unifier.add_inferred_edges(edges)
            except Exception:
                pass

    # Pass 2B: Tabular Causal Discovery (if data is provided or auto-detected)
    df = None
    data_file = None
    if data:
        data_file = Path(data)
    else:
        # Auto-discover CSV if present in path/data
        csv_candidates = [
            p
            for p in base_path.rglob("*.csv")
            if ".venv" not in p.parts and ".cig" not in p.parts
        ]
        if csv_candidates:
            data_file = csv_candidates[0]

    if data_file and data_file.exists():
        try:
            df = pd.read_csv(data_file)
            disc_engine = CausalDiscoveryEngine(
                significance_level=0.05, max_cond_size=2
            )
            discovered_edges = disc_engine.discover_edges(df)
            if discovered_edges:
                unifier.add_discovered_edges(discovered_edges)
                console.print(
                    f"[bold purple]✔ Pass 2B Continuous Discovery:[/bold purple] Learned {len(discovered_edges)} edges from {data_file.name}"
                )
        except Exception as e:
            console.print(f"[dim yellow]⚠ Data discovery skipped: {e}[/dim yellow]")

    # Pass 3: SCM Unification with Namespace Reconciliation
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
    openlineage_path = dot_cig_dir / "openlineage_event.json"
    OpenLineageExporter.export_to_file(report, out_path=openlineage_path)

    table = Table(
        title="SCM Pipeline Compilation Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Pipeline Stage", style="dim")
    table.add_column("Result", justify="right")
    table.add_row("Total Variables (Nodes)", str(report.total_nodes))
    table.add_row("Total Causal Edges", str(report.total_edges))
    table.add_row("Deterministic Lineages (Pass 1)", str(report.extracted_edges))
    table.add_row("Domain Hypotheses (Pass 2)", str(report.inferred_edges))
    table.add_row("Data-Driven Edges (Pass 2B)", str(report.discovered_edges))
    table.add_row("Confounder Hubs (God Nodes)", str(len(report.god_confounders)))
    console.print(table)

    # Ingest into Rust PyCausalEngine
    engine = PyCausalEngine()
    unifier.populate_rust_engine(engine)

    all_nodes = [node["name"] for node in json.loads(engine.export_json())["nodes"]]

    # Pass 4 & Pass 6: Estimation & Active Inference
    target_y = resolve_node_name(outcome, all_nodes) if outcome else None
    target_t = resolve_node_name(treatment, all_nodes) if treatment else None

    # Fallback auto-detection if none specified
    if not target_y:
        for candidate in ["revenue", "profit", "conversion_rate", "churn"]:
            resolved = resolve_node_name(candidate, all_nodes)
            if resolved in all_nodes:
                target_y = resolved
                break

    if not target_t:
        for candidate in ["discount_pct", "treatment_x", "price", "marketing_tier"]:
            resolved = resolve_node_name(candidate, all_nodes)
            if resolved in all_nodes:
                target_t = resolved
                break

    if (
        df is not None
        and target_t
        and target_y
        and target_t in df.columns
        and target_y in df.columns
    ):
        estimator = CausalEstimationEngine(engine)
        try:
            summary = estimator.estimate_effect(
                df, treatment=target_t, outcome=target_y, run_refutations=False
            )
            sens = estimator.compute_sensitivity_analysis(
                df, treatment=target_t, outcome=target_y
            )

            console.print(
                Panel(
                    f"[bold]Treatment:[/bold] {target_t} ➔ [bold]Outcome:[/bold] {target_y}\n"
                    f"[bold green]ATE Estimate:[/bold green] {summary.value:.4f} ({summary.identification_method})\n"
                    f"[bold]Cinelli-Hazlett Robustness (RV_q=1.0):[/bold] [bold cyan]{sens.rv_q * 100:.1f}%[/bold cyan]\n"
                    f"[bold]VanderWeele-Ding E-Value:[/bold] [bold yellow]{sens.e_value:.2f}[/bold yellow]",
                    title="📊 Pass 4: Automated Econometric & Sensitivity Audit",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(
                f"[dim yellow]⚠ Pass 4 Econometrics skipped: {e}[/dim yellow]"
            )

    if df is not None and target_y:
        agent = ActiveInferenceAgent(engine=engine, report=report)
        target_val = goal if goal is not None else float(df[target_y].mean() * 1.05)

        # Default action policies
        candidate_policies = {}
        if target_t:
            baseline_val = (
                float(df[target_t].mean()) if target_t in df.columns else 0.05
            )
            candidate_policies["conservative_baseline"] = {
                target_t: round(baseline_val, 3)
            }
            candidate_policies["aggressive_growth"] = {
                target_t: round(baseline_val * 1.5, 3)
            }

        # Add epistemic policy if unverified edges exist
        for edge in report.edges:
            if edge.edge_type != "EXTRACTED":
                candidate_policies["epistemic_probe"] = {edge.source: 1.0}
                break

        if candidate_policies:
            try:
                plan = agent.plan(
                    candidate_interventions=candidate_policies,
                    target_outcome=target_y,
                    desired_value=target_val,
                    df=df,
                )
                min_efe = (
                    plan.optimal_policy.expected_free_energy
                    if plan.optimal_policy
                    else 0.0
                )
                opt_name = (
                    plan.optimal_policy.policy_name if plan.optimal_policy else "None"
                )

                console.print(
                    Panel(
                        f"[bold]Target Setpoint:[/bold] {target_y} = {target_val:.2f}\n"
                        f"[bold cyan]Optimal Policy Selected:[/bold cyan] {opt_name}\n"
                        f"[bold]Min Expected Free Energy (G):[/bold] {min_efe:.3f}\n"
                        f"{plan.agent_rationale}",
                        title="🧠 Pass 6: Active Inference EFE Policy Optimization",
                        border_style="cyan",
                    )
                )
            except Exception as e:
                console.print(
                    f"[dim yellow]⚠ Pass 6 Active Inference skipped: {e}[/dim yellow]"
                )

    console.print(
        "[bold green]✔ CIG pipeline execution complete. Artifacts ready.[/bold green]"
    )


@main.command(name="policy")
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=True),
    help="CSV dataset with covariates",
)
@click.option(
    "--treatment",
    "-t",
    required=True,
    help="Treatment policy lever (e.g., discount_pct)",
)
@click.option("--outcome", "-y", required=True, help="Outcome metric (e.g., revenue)")
@click.option(
    "--max-depth",
    default=2,
    type=int,
    help="Maximum depth for interpretable policy tree",
)
def policy_cmd(data: str, treatment: str, outcome: str, max_depth: int):
    """🌲 Pass 4B: Distill Heterogeneous Treatment Effects (CATE) into interpretable policy trees."""
    df = pd.read_csv(data)

    unifier = SCMGraphUnification()
    # Populate baseline graph from directory if present
    base_path = Path(".")
    for sf in base_path.rglob("*.sql"):
        if ".venv" not in sf.parts and ".cig" not in sf.parts:
            try:
                edges = SQLLineageParser().parse_file(sf)
                if edges:
                    unifier.add_extracted_edges(edges)
            except Exception:
                pass

    unifier.unify()
    engine = PyCausalEngine()
    unifier.populate_rust_engine(engine)

    all_nodes = [node["name"] for node in json.loads(engine.export_json())["nodes"]]
    t_res = resolve_node_name(treatment, all_nodes)
    y_res = resolve_node_name(outcome, all_nodes)

    if t_res not in df.columns:
        df[t_res] = df[treatment]
    if y_res not in df.columns:
        df[y_res] = df[outcome]

    console.print(
        f"[bold cyan]🌲 Fitting Causal Forest DML & Distilling Policy Tree for {t_res} ➔ {y_res}...[/bold cyan]"
    )

    tree_engine = PolicyTreeEngine(engine)
    report = tree_engine.fit_policy_tree(
        df=df,
        treatment=t_res,
        outcome=y_res,
        max_depth=max_depth,
    )

    console.print(
        Panel(
            f"[bold]Population Average Treatment Effect (ATE):[/bold] [bold green]{report.population_ate:+.4f}[/bold green]\n"
            f"[bold]Heterogeneous Modifiers Analyzed:[/bold] {', '.join(report.heterogeneous_features)}\n\n"
            f"[bold cyan]Decision Tree Segmentation Logic:[/bold cyan]\n[dim]{report.rules_text}[/dim]",
            title="🎯 Pass 4B: Heterogeneous CATE & Policy Rules",
            border_style="magenta",
        )
    )

    table = Table(
        title="Distilled Policy Cohorts & Recommended Allocations",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Cohort ID", style="dim")
    table.add_column("Sample Size", justify="right")
    table.add_column("Mean CATE (τ)", justify="right")
    table.add_column("Recommended Action", justify="left")

    for seg in report.segments:
        eff_color = "green" if seg.mean_treatment_effect >= 0 else "red"
        table.add_row(
            seg.segment_id,
            str(seg.sample_size),
            f"[{eff_color}]{seg.mean_treatment_effect:+.2f}[/{eff_color}]",
            seg.recommended_policy,
        )

    console.print(table)


@main.command(name="check")
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True, file_okay=True),
    default=None,
    help="Path to dbt manifest.json or SQLMesh state JSON",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, file_okay=True),
    default=None,
    help="Observational CSV dataset to audit",
)
@click.option(
    "--treatment",
    "-t",
    required=True,
    help="Core treatment variable to protect",
)
@click.option(
    "--outcome",
    "-y",
    required=True,
    help="Core KPI / target outcome to protect",
)
@click.option(
    "--min-rv",
    default=0.10,
    type=float,
    help="Minimum allowable Cinelli-Hazlett Robustness Value (0.0 to 1.0)",
)
@click.option(
    "--min-e-value",
    default=1.5,
    type=float,
    help="Minimum allowable VanderWeele-Ding E-Value",
)
def check_cmd(
    path: str,
    manifest: str | None,
    data: str | None,
    treatment: str,
    outcome: str,
    min_rv: float,
    min_e_value: float,
):
    """🛡️ CI/CD Gatekeeper: Verify causal DAG integrity and audit robustness thresholds."""
    base_path = Path(path)
    console.print(
        Panel(
            f"[bold cyan]🛡️ CIG CI/CD Causal Integrity Check:[/bold cyan] {base_path.resolve()}",
            border_style="cyan",
        )
    )

    unifier = SCMGraphUnification()
    sql_parser = SQLLineageParser()
    py_parser = PythonLineageParser()

    if manifest:
        m_path = Path(manifest)
        try:
            m_data = json.loads(m_path.read_text(encoding="utf-8"))
            if "models" in m_data and "nodes" not in m_data:
                m_edges = SQLMeshManifestParser().parse_manifest_dict(
                    m_data, provenance=str(m_path)
                )
            else:
                m_edges = DBTManifestParser().parse_manifest_dict(
                    m_data, provenance=str(m_path)
                )
            unifier.add_extracted_edges(m_edges)
            console.print(
                f"[bold green]✔ Ingested {len(m_edges)} lineage edges from manifest:[/bold green] {m_path.name}"
            )
        except Exception as ex:
            console.print(f"[yellow]⚠ Failed parsing manifest {m_path}: {ex}[/yellow]")

    for sf in base_path.rglob("*.sql"):
        if ".venv" not in sf.parts and ".cig" not in sf.parts:
            try:
                edges = sql_parser.parse_file(sf)
                if edges:
                    unifier.add_extracted_edges(edges)
            except Exception:
                pass

    for pf in base_path.rglob("*.py"):
        if (
            ".venv" not in pf.parts
            and ".cig" not in pf.parts
            and "test" not in pf.parts
        ):
            try:
                edges = py_parser.parse_file(pf)
                if edges:
                    unifier.add_extracted_edges(edges)
            except Exception:
                pass

    report = unifier.unify()

    # Rule 1: Acyclicity & Ambiguity Audit
    if report.ambiguous_edges > 0:
        console.print(
            f"[bold red]❌ CI Failure:[/bold red] Found {report.ambiguous_edges} ambiguous or cyclic edges in code lineage."
        )
        for e in report.edges:
            if e.edge_type == "AMBIGUOUS":
                console.print(
                    f"  [red]● {e.source} ⟷ {e.target}: {e.conflict_notes}[/red]"
                )
        sys.exit(1)

    engine = PyCausalEngine()
    unifier.populate_rust_engine(engine)

    all_nodes = [node["name"] for node in json.loads(engine.export_json())["nodes"]]
    t_res = resolve_node_name(treatment, all_nodes)
    y_res = resolve_node_name(outcome, all_nodes)

    if t_res not in all_nodes or y_res not in all_nodes:
        console.print(
            f"[bold red]❌ CI Failure:[/bold red] Target variables not found in compiled SCM ({treatment}, {outcome})."
        )
        sys.exit(1)

    # Rule 2: Backdoor / Identification Verification
    backdoors = engine.find_backdoor_adjustment_sets(t_res, y_res)
    frontdoors = engine.find_frontdoor_adjustment_sets(t_res, y_res)
    instruments = engine.find_instrumental_variables(t_res, y_res)

    has_identification = bool(backdoors or frontdoors or instruments)
    if not has_identification:
        console.print(
            f"[bold red]❌ CI Failure:[/bold red] Identification path broken between '{t_res}' and '{y_res}'. "
            "Unblocked backdoor paths exist with no valid adjustment set."
        )
        sys.exit(1)

    console.print(
        f"[bold green]✔ Causal Identification Strategy Validated:[/bold green] Found {len(backdoors)} backdoor adjustment set(s)."
    )

    # Rule 3: Empirical Sensitivity Thresholds (if data provided)
    if data and Path(data).exists():
        df = pd.read_csv(data)
        if t_res in df.columns and y_res in df.columns:
            estimator = CausalEstimationEngine(engine)
            sens = estimator.compute_sensitivity_analysis(
                df, treatment=t_res, outcome=y_res
            )

            console.print(
                f"[bold cyan]Cinelli-Hazlett RV_q:[/bold cyan] {sens.rv_q * 100:.1f}% (Required: ≥ {min_rv * 100:.1f}%)\n"
                f"[bold cyan]VanderWeele-Ding E-Value:[/bold cyan] {sens.e_value:.2f} (Required: ≥ {min_e_value:.2f})"
            )

            if sens.rv_q < min_rv:
                console.print(
                    f"[bold red]❌ CI Failure:[/bold red] Robustness value ({sens.rv_q * 100:.1f}%) is below minimum threshold ({min_rv * 100:.1f}%)."
                )
                sys.exit(1)

            if sens.e_value < min_e_value:
                console.print(
                    f"[bold red]❌ CI Failure:[/bold red] E-value ({sens.e_value:.2f}) is below minimum threshold ({min_e_value:.2f})."
                )
                sys.exit(1)

    console.print(
        "[bold green]✔ All causal regression checks passed successfully.[/bold green]"
    )


@main.command(name="monitor")
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=True),
    help="Time-series or partitioned CSV dataset to audit for mechanism drift",
)
@click.option(
    "--time-col",
    "-tc",
    default=None,
    help="Optional timestamp or ordering column (e.g. created_at, date). Defaults to row index.",
)
@click.option(
    "--alpha",
    "-a",
    default=0.05,
    type=float,
    help="Significance threshold for Kolmogorov-Smirnov invariance test",
)
@click.option(
    "--target",
    "-t",
    default=None,
    help="Optional specific target variable to test mechanism invariance for",
)
@click.option("--graph", "-g", default=None, help="Path to .cig/cig_graph.json")
def monitor_cmd(
    data: str, time_col: str, alpha: float, target: str | None, graph: str | None
):
    """⏱️ Pass 7: Audit causal invariance and detect structural mechanism drift across time windows."""
    df = pd.read_csv(data)
    graph_json_path = find_graph_path(graph, data)

    if not graph_json_path.exists():
        console.print(
            "[bold red]Error:[/bold red] No compiled SCM found. Run [bold cyan]cig run[/bold cyan] or [bold cyan]cig scan[/bold cyan] first."
        )
        sys.exit(1)

    engine = PyCausalEngine()
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    for edge in graph_data.get("edges", []):
        if isinstance(edge, (list, tuple)):
            engine.add_extracted_edge(edge[0], edge[1], None, None)
        elif edge.get("edge_type") == "EXTRACTED":
            engine.add_extracted_edge(edge["source"], edge["target"], None, None)
        else:
            engine.add_inferred_edge(
                edge["source"],
                edge["target"],
                float(edge.get("confidence", 1.0)),
                None,
                None,
            )

    all_nodes = [node["name"] for node in json.loads(engine.export_json())["nodes"]]
    target_resolved = [resolve_node_name(target, all_nodes)] if target else None

    monitor = CausalInvarianceMonitor(engine, significance_level=alpha)
    report = monitor.audit_temporal_invariance(
        df=df,
        time_column=time_col,
        target_nodes=target_resolved,
    )

    console.print(
        Panel(
            f"[bold]Ordering Feature:[/bold] {report.time_column}\n"
            f"[bold]Mechanisms Audited:[/bold] {report.total_mechanisms_audited}\n"
            f"[bold green]Stable Structural Invariants:[/bold green] {report.stable_mechanisms}\n"
            f"[bold red]Broken Structural Mechanisms:[/bold red] {report.broken_mechanisms}",
            title="⏱️ Pass 7: Causal Invariance & Mechanism Drift Report",
            border_style="cyan",
        )
    )

    table = Table(
        title="Structural Equations P(Y | PA(Y)) Invariance Audit",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Target Variable (Y)", style="dim")
    table.add_column("Causal Parents PA(Y)")
    table.add_column("KS Statistic", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Status", justify="center")

    for r in report.results:
        color = "green" if r.is_invariant else "bold red"
        table.add_row(
            r.target,
            ", ".join(r.parents),
            f"{r.ks_statistic:.4f}",
            f"{r.p_value:.4e}",
            f"[{color}]{r.status}[/{color}]",
        )

    console.print(table)


if __name__ == "__main__":
    main()
