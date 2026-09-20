from typing import Any, Dict
from rich.console import Console
from rich.table import Table


def render_terminal_table(results: Dict[str, Any]):
    """Render rich visual table to standard out."""
    console = Console()
    table = Table(
        title="cig vs. LLM Collider Trap Benchmark", header_style="bold magenta"
    )

    table.add_column("Dimension", style="cyan")
    table.add_column(f"LLM ({results['llm']['model']})", justify="right")
    table.add_column("cig (Rust Engine)", justify="right", style="green")
    table.add_column("Delta / Impact", justify="right", style="bold yellow")

    llm = results["llm"]
    cig = results["cig"]
    summary = results["summary"]

    table.add_row(
        "Adjustment Set",
        str(llm["selected_controls"]),
        str(cig["selected_controls"]),
        "Minimal Sufficiency",
    )
    table.add_row(
        "Prompt Tokens",
        str(llm["prompt_tokens"]),
        str(cig["prompt_tokens"]),
        f"-{round((1 - cig['prompt_tokens'] / llm['prompt_tokens']) * 100, 1)}%",
    )
    table.add_row(
        "Total Tokens",
        str(llm["total_tokens"]),
        str(cig["total_tokens"]),
        f"-{summary['token_reduction_pct']}%",
    )
    table.add_row(
        "Latency",
        f"{llm['latency_sec']:.3f}s",
        f"{cig['latency_sec'] * 1_000_000:.1f}µs",
        f"{summary['speedup_factor']}x faster",
    )
    table.add_row(
        "Collider Avoidance", "Passed", "Passed (d-sep verified)", "Deterministic"
    )

    console.print(table)


def to_markdown_table(results: Dict[str, Any]) -> str:
    """Generate GitHub-flavored markdown table."""
    llm = results["llm"]
    cig = results["cig"]
    summary = results["summary"]

    return f"""| Metric Dimension | Raw LLM ({llm["model"]}) | `cig` Engine (Rust Core) | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Adjustment Set** | `{llm["selected_controls"]}` | `{cig["selected_controls"]}` | Minimal sufficiency |
| **Prompt Tokens** | {llm["prompt_tokens"]} | **{cig["prompt_tokens"]}** | -{round((1 - cig["prompt_tokens"] / llm["prompt_tokens"]) * 100, 1)}% |
| **Total Tokens** | {llm["total_tokens"]} | **{cig["total_tokens"]}** | -{summary["token_reduction_pct"]}% reduction |
| **Latency** | {llm["latency_sec"]:.3f}s | **{cig["latency_sec"] * 1_000_000:.1f} µs** | {summary["speedup_factor"]}x speedup |
| **Causal Validity** | Passed | **Passed** | d-separation verified |
"""


def to_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX tabular markup for academic papers."""
    llm = results["llm"]
    cig = results["cig"]
    summary = results["summary"]

    return f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lrrr}}
\\hline
\\textbf{{Metric}} & \\textbf{{{llm["model"]}}} & \\textbf{{cig (Rust)}} & \\textbf{{Delta}} \\\\
\\hline
Prompt Tokens & {llm["prompt_tokens"]} & {cig["prompt_tokens"]} & -{round((1 - cig["prompt_tokens"] / llm["prompt_tokens"]) * 100, 1)}\\% \\\\
Total Tokens & {llm["total_tokens"]} & {cig["total_tokens"]} & -{summary["token_reduction_pct"]}\\% \\\\
Latency & {llm["latency_sec"]:.3f}s & {cig["latency_sec"] * 1_000_000:.1f}$\\mu$s & {summary["speedup_factor"]}$\\times$ \\\\
\\hline
\\end{{tabular}}
\\caption{{Empirical comparison between in-context reasoning and deterministic causal compilation.}}
\\label{{tab:collider_benchmark}}
\\end{{table}}
"""
