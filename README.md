# ⚡ cig (Causal Inference Graph)

[![CI](https://github.com/waynelshi/cig/actions/workflows/ci.yml/badge.svg)](https://github.com/waynelshi/cig/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`cig`** is a high-performance continuous Causal Inference Graph engine designed for modern data stacks, analytics pipelines, and AI agent workflows. 

By unifying deterministic SQL/Python code lineages with domain literature hypotheses and compiled graph algorithms, `cig` automatically constructs Structural Causal Models (SCMs), detects confounder hazards (colliders, mediators, unblocked backdoor paths), and estimates treatment effects with rigorous sensitivity analysis.

---

## Key Features

- **4-Pass Pipeline Architecture**:
  - **Pass 1 (Deterministic Lineage)**: Column-level dependency tracking via `sqlglot` (PostgreSQL, BigQuery, Snowflake, DuckDB) and Python AST parsing (Tree-sitter) for DataFrame assignments and dbt manifests.
  - **Pass 2 (Semantic Extraction)**: Regex- and NLP-based causal hypothesis extraction from Markdown documentation, dbt `schema.yml`, and `docs` blocks.
  - **Pass 3 (Graph Unification & Conflict Resolution)**: Merges lineages into a unified DAG, detects 2-cycle contradictions, marks ambiguous edges, and flags "God Confounders" (high-degree hub nodes).
  - **Pass 4 (Causal Estimation & Refutation)**: Automated identification via Pearl's $do$-calculus (Backdoor, Frontdoor, Instrumental Variables) using `DoWhy`, `EconML` (LinearDML, CausalForestDML), mediation decomposition, and Cinelli-Hazlett sensitivity bounds.
- **PreToolUse IDE Interceptors**: Native hooks for **Cursor** (`.cursorrules`) and **Claude Code** (`settings.json`) to block AI agents from committing confounded regressions.
- **Standalone HTML Visualizer (`.cig/cig.html`)**:
  - Interactive force-directed DAG canvas with dynamic clustering.
  - Live Cinelli-Hazlett sensitivity contour simulator ($R^2_{Y \sim Z \mid X, D}$ vs $R^2_{D \sim Z \mid X}$).
  - In-browser ATE estimation powered by **Pyodide WebAssembly** (`numpy`, `pandas`, `statsmodels`).
  - 1-click SVG and PNG exports.
- **Automated Documentation**: Compiles human-readable summary tables and Mermaid flowchart diagrams (`.cig/SCM.md`).

---

## Architecture Overview
```text
Codebase (SQL, Python, dbt, Markdown)
│
├── [Pass 1] AST Lineage Extractor ────► Deterministic Edges (1.0)
│                                              │
├── [Pass 2] Semantic Doc Extractor ───► Inferred Hypotheses (0.75)
│                                              │
▼                                              ▼
[Pass 3] SCM Graph Unification & Conflict Resolver ──► Rust cig_core Engine
│
├── Export: .cig/cig_graph.json, .cig/SCM.md, .cig/cig.html
│
▼
[Pass 4] Estimation & Guardrails (DoWhy / EconML / cig-hook)
```
---

## Quick Start

### 1. Prerequisites

- [Rust Toolchain (1.75+)](https://rustup.rs/)
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) package manager

### 2. Installation & Build

Clone the repository and build the Rust PyO3 core extension:

```bash
git clone [https://github.com/your-username/cig.git](https://github.com/your-username/cig.git)
cd cig

# Create virtual environment and install dependencies
uv sync

# Compile the Rust core extension
uv run maturin develop
```

### CLI Usage
Initialize Configuration
Create the default .cig metadata scaffolding in your current repository:
```bash
uv run cig init
```
### Scan Repository & Compile SCM
Analyze SQL models, Python scripts, dbt manifests, and Markdown docs to build your DAG:
```bash
uv run cig scan .
```
This generates:
.cig/cig_graph.json — Machine-readable unified SCM representation.
.cig/cig.html — Interactive visualizer and WebAssembly estimation playground.
.cig/SCM.md — Markdown report with embedded Mermaid diagrams.

### Identify Causal Effects & Interventions
Query the DAG for identification strategies (Backdoor sets, Frontdoor sets, IVs):
```bash
uv run cig do -t discount_rate -y revenue
```
### Estimate Treatment Effects (ATE)
Estimate causal effects from observational data using double machine learning or linear models:
```bash
uv run cig do -t discount_rate -y revenue -d data/transactions.csv -m backdoor.econml.dml.LinearDML
```
### Natural Direct & Indirect Effect (Mediation)
Decompose total treatment effect through an intermediate mediator:
```bash
uv run cig do -t discount_rate -y revenue -med customer_satisfaction -d data/transactions.csv
```

## IDE & AI Agent Guardrails
To prevent LLM code generation tools from committing unadjusted or collider-biased regressions:
```bash
# Verify regression code directly via CLI
echo 'import statsmodels.formula.api as smf; model = smf.ols("revenue ~ discount_rate", data=df).fit()' | uv run cig-hook --strict
```
### Cursor Rules (`.cursorrules`)
Add .cursorrules to your project root to enforce causal verification before executing regression code:
```markdown
# Causal Inference Guardrail
Always verify candidate regressions against `.cig/cig_graph.json`.
Run `uv run cig-hook --file <target_file> --strict` before committing changes.
```
### Claude Code (`.claude/settings.json`)
Configure PreToolUse hooks to intercept Python edits automatically:
```json
{
  "hooks": {
    "preToolUse": [
      {
        "matcher": { "tool": "FileEdit|WriteFile", "params": { "path": ".*\\.py$" } },
        "command": "uv run cig-hook --strict"
      }
    ]
  }
}
```

### Development & Testing
Run all Rust and Python test suites:
```bash
# Run Rust unit tests
cargo test --workspace

# Run Python pytest suite
uv run pytest tests/ -v
```

### License
This project is licensed under the MIT License.
