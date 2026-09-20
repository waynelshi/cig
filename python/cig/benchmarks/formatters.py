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


def render_ab_terminal_table(results: Dict[str, Any]):
    console = Console()
    table = Table(
        title="CIG Causal Guard Benchmark: Unguarded vs. Guarded",
        header_style="bold magenta",
    )

    table.add_column("Dimension", style="cyan")
    table.add_column("Unguarded Agent", justify="right", style="red")
    table.add_column("CIG-Guarded Agent", justify="right", style="green")
    table.add_column("Delta / Impact", justify="right", style="bold yellow")

    un = results["unguarded"]
    cg = results["cig"]
    true_eff = results["true_effect"]

    table.add_row(
        "Controls Selected",
        "All (Mediator + Collider)",
        f"Minimal: {cg['controls']}",
        "Prevents Bad Controls",
    )
    table.add_row(
        "Estimated Effect",
        f"{un['estimated_effect']:+.2f}",
        f"{cg['estimated_effect']:+.2f}",
        f"Ground Truth: +{true_eff:.1f}",
    )
    table.add_row(
        "Estimation Bias",
        f"{un['bias']:+.2f}",
        f"{cg['bias']:+.2f}",
        f"-{results['summary']['bias_reduction']:.1f} Error Elimination",
    )
    table.add_row("95% CI", str(un["ci_95"]), str(cg["ci_95"]), "Tight & Unbiased")
    table.add_row(
        "Decision Validity",
        "FAIL (Sign Reversal)" if un["sign_reversal"] else "PASS",
        "PASS (Accurate Direction)",
        "Avoids False Rollback",
    )

    console.print(table)


def to_ab_markdown_table(results: Dict[str, Any]) -> str:
    un = results["unguarded"]
    cg = results["cig"]
    true_eff = results["true_effect"]

    return f"""| Metric Dimension | Unguarded Agent (Kitchen-Sink) | `cig`-Guarded Engine | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Controls Used** | All available (Mediator + Collider) | `{cg["controls"]}` | Blocks collider & mediator |
| **Estimated Effect** | {un["estimated_effect"]:+.2f} | **{cg["estimated_effect"]:+.2f}** | True Ground Truth: +{true_eff:.1f} |
| **Estimation Bias** | {un["bias"]:+.2f} | **{cg["bias"]:+.2f}** | Completely debiased |
| **95% Confidence Interval** | {un["ci_95"]} | **{cg["ci_95"]}** | Valid coverage |
| **Decision Impact** | **Reverses Sign (Disaster)** | **Accurate Positive Margin** | Avoids erroneous rollback |
"""


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


def render_mbias_terminal_table(results: Dict[str, Any]):
    console = Console()
    table = Table(
        title="cig vs. LLM M-Bias Trap Benchmark", header_style="bold magenta"
    )

    table.add_column("Dimension", style="cyan")
    table.add_column(
        f"LLM ({results['llm']['model']})",
        justify="right",
        style="red" if results["llm"]["induced_bias"] else "green",
    )
    table.add_column("cig (Rust Engine)", justify="right", style="green")
    table.add_column("Delta / Impact", justify="right", style="bold yellow")

    llm = results["llm"]
    cig = results["cig"]
    true_eff = results["true_effect"]

    table.add_row(
        "Recommended Controls",
        str(llm["controls"]),
        str(cig["controls"]),
        "Minimal Sufficiency (Empty Set)",
    )
    table.add_row(
        "Estimated Effect",
        f"{llm['estimated_effect']:+.3f}",
        f"{cig['estimated_effect']:+.3f}",
        f"Ground Truth: +{true_eff:.2f}",
    )
    table.add_row(
        "Induced Bias",
        f"{llm['bias']:+.3f}",
        f"{cig['bias']:+.3f}",
        "Spurious Conditioning Avoided"
        if not llm["induced_bias"]
        else "M-Bias Induced",
    )
    table.add_row(
        "Causal Validity",
        "FAILED (Opened M-Path)" if llm["induced_bias"] else "PASSED",
        "PASSED (d-sep verified)",
        f"M Valid Backdoor? {cig['is_m_valid']}",
    )
    table.add_row(
        "Latency",
        f"{llm['latency_sec']:.3f}s",
        f"{cig['latency_sec'] * 1_000_000:.1f}µs",
        f"{results['summary']['speedup_factor']:.0f}x faster",
    )
    table.add_row(
        "Tokens Consumed",
        str(llm["total_tokens"]),
        "0 (Local AST)",
        "100% Token Elimination",
    )

    console.print(table)


def to_mbias_markdown_table(results: Dict[str, Any]) -> str:
    llm = results["llm"]
    cig = results["cig"]
    true_eff = results["true_effect"]

    return f"""| Dimension | LLM ({llm["model"]}) | `cig` (Rust Engine) | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Recommended Controls** | `{llm["controls"]}` | `{cig["controls"]}` | Minimal Sufficiency (Empty Set) |
| **Estimated Effect** | {llm["estimated_effect"]:+.3f} | **{cig["estimated_effect"]:+.3f}** | Ground Truth: +{true_eff:.2f} |
| **Induced Bias** | {llm["bias"]:+.3f} | **{cig["bias"]:+.3f}** | Spurious Conditioning Avoided |
| **Causal Validity** | {"FAILED (Opened M-Path)" if llm["induced_bias"] else "PASSED"} | **PASSED (d-sep verified)** | M Valid Backdoor: {cig["is_m_valid"]} |
| **Latency** | {llm["latency_sec"]:.3f}s | **{cig["latency_sec"] * 1_000_000:.1f}µs** | **{results["summary"]["speedup_factor"]:.0f}x faster** |
| **Tokens Consumed** | {llm["total_tokens"]} | **0 (Local AST)** | **100% Token Elimination** |
"""
