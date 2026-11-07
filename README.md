# `cig`: Continuous Causal Governance Engine

[![CI](https://github.com/waynelshi/cig/actions/workflows/ci.yml/badge.svg)](https://github.com/waynelshi/cig/actions/workflows/ci.yml)
[![Causal Gatekeeper](https://github.com/waynelshi/cig/actions/workflows/cig_gatekeeper.yml/badge.svg)](https://github.com/waynelshi/cig/actions/workflows/cig_gatekeeper.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)

**`cig`** is a high-performance Continuous Causal Governance Engine designed for modern data stacks, analytics pipelines, and AI agent workflows.

By unifying deterministic SQL/Python code lineages with domain hypotheses, dbt/SQLMesh manifests, and observational data discovery, `cig` automatically compiles Structural Causal Models (SCMs), detects confounder hazards (colliders, mediators, unblocked backdoor paths), bounds unobserved sensitivity, and enforces causal invariance in CI/CD workflows.

---

## Key Features

- **7-Pass Unified Pipeline**:
  - **Pass 1 (Deterministic Lineage)**: Column-level dependency tracking via `sqlglot` (PostgreSQL, BigQuery, Snowflake, DuckDB), Python AST parsing (Tree-sitter), and native dbt/SQLMesh manifest parsing.
  - **Pass 2 & 2B (Semantic Extraction & Causal Discovery)**: Regex/NLP domain hypothesis extraction from Markdown docs and dbt schema YAML, combined with continuous observational causal discovery.
  - **Pass 3 (Graph Unification & Conflict Resolution)**: Merges multi-source lineages into a unified DAG, detects 2-cycle contradictions, flags "God Confounders" (high-degree hub nodes), and syncs with the native Rust `PyCausalEngine` core.
  - **Pass 4 & 4B (Identification, Sensitivity & Policy Trees)**: Automated backdoor/frontdoor/IV identification, EconML/DoWhy ATE estimation, Cinelli-Hazlett ($RV_q$) sensitivity bounds, VanderWeele-Ding E-values, and CATE distillation into interpretable policy trees.
  - **Pass 5 & 6 (Counterfactual Simulation & Active Inference)**: Pearl Level 3 noise abduction ($Y_{T=v}$) and Active Inference Expected Free Energy (EFE) minimization to arbitrate business interventions vs. epistemic exploration.
  - **Pass 7 (Temporal Causal Invariance Monitoring)**: Invariant Causal Prediction (ICP) mechanism drift detection to audit $P(Y \mid PA(Y))$ across temporal partitions and identify broken structural equations with `--strict` gating.
- **CI/CD Gatekeeper (`cig check`)**: Automated PR gate that validates acyclicity, verifies causal identification paths, checks sensitivity thresholds, and posts summary tables to GitHub PR comments or `$GITHUB_STEP_SUMMARY`.
- **Enterprise Ecosystem Connectors**: Standard OpenLineage event JSON exporter and drop-in Airflow/Dagster runtime invariant checks.
- **PreToolUse IDE Interceptors (`cig-hook`)**: Native hooks for **Cursor** (`.cursorrules`) and **Claude Code** (`settings.json`) to block AI agents from committing confounded regressions.
- **Structured Output (`--as-json`)**: Machine-readable JSON emission for `cig do` and `cig check` to enable autonomous agent tool loops.
- **Standalone HTML Visualizer (`.cig/cig.html`)**: Interactive force-directed DAG canvas with Pyodide WebAssembly in-browser estimation and sensitivity contours.

---

## Architecture Overview

```text
Codebase (SQL, Python, dbt, SQLMesh, Markdown, CSV)  
│  
├── [Pass 1] AST & Manifest Parser ────► Deterministic Edges (1.0)  
├── [Pass 2] Semantic Doc Extractor ───► Inferred Hypotheses (0.75)  
├── [Pass 2B] Continuous Discovery ────► Learned Conditional Edges  
│  
▼  
[Pass 3] SCM Graph Unification & Conflict Resolver ──► Rust PyCausalEngine  
│  
├── Export: .cig/cig_graph.json, .cig/SCM.md, .cig/cig.html, .cig/openlineage_event.json  
│  
├── [Pass 4] Identification, Refutation & Cinelli-Hazlett Sensitivity (--as-json)  
├── [Pass 4B] CATE Distillation into Interpretable Policy Trees  
├── [Pass 5] Pearl Level 3 Counterfactual Inversion (Abduction)  
├── [Pass 6] Active Inference EFE Policy Arbitration  
├── [Pass 7] Invariant Causal Prediction & Drift Monitor (--strict)  
│  
▼  
CI/CD Gatekeeper (`cig check`) & Orchestrator Hooks (Airflow / Dagster)
```

---

## Quick Start

### 1. Universal Installer (Precompiled Wheel)

Install precompiled binary wheels directly via the standalone installer:

```bash
curl -sSf https://cig.run | sh
```

### 2. Local Build from Source

**Prerequisites**

* [Rust Toolchain (1.75+)](https://rustup.rs/)
* Python 3.10+
* [uv](https://docs.astral.sh/uv/) package manager

```bash
git clone https://github.com/waynelshi/cig.git
cd cig

# Create virtual environment and sync dependencies
uv sync --all-groups

# Compile the Rust PyCausalEngine extension
uv run maturin develop
```

---

## CLI Usage

### Initialize Configuration

Create `.cig` directory and default causal configuration scaffolding:

```bash
uv run cig init
```

### Scan Repository & Compile SCM

Scan SQL models, Python scripts, dbt/SQLMesh manifests, and markdown docs to compile your DAG:

```bash
uv run cig scan .
```

This generates:

* `.cig/cig_graph.json`: Machine-readable unified SCM representation.
* `.cig/cig.html`: Interactive DAG visualizer and Pyodide WebAssembly sandbox.
* `.cig/SCM.md`: Markdown documentation with embedded Mermaid diagrams.
* `.cig/cig_graph.graphml`: GraphML export for Gephi and Cytoscape.
* `.cig/openlineage_event.json`: OpenLineage standard causal lineage event.

### End-to-End Execution (`cig run`)

Run the full multi-pass pipeline from scanning through Active Inference policy selection:

```bash
uv run cig run . \
  --manifest target/manifest.json \
  --data data/transactions.csv \
  --treatment discount_pct \
  --outcome revenue \
  --goal 1200.0
```

### Causal Estimation, Mediation & Sensitivity (`cig do`)

Verify identification strategies, compute ATE, evaluate mediation, or test unobserved confounding sensitivity:

```bash
# Query identification strategy only (Backdoor, Frontdoor, IV)
uv run cig do -t discount_pct -y revenue

# Estimate ATE via LinearDML / Double Machine Learning
uv run cig do -t discount_pct -y revenue -d data/transactions.csv -m backdoor.econml.dml.LinearDML

# Natural Direct & Indirect Effect (Mediation)
uv run cig do -t discount_pct -y revenue -med customer_satisfaction -d data/transactions.csv

# Cinelli-Hazlett (RV_q) & VanderWeele-Ding E-Value Sensitivity Analysis
uv run cig do -t discount_pct -y revenue -d data/transactions.csv --sensitivity
```

### Policy Tree Distillation (`cig policy`)

Distill Heterogeneous Treatment Effects (CATE) from CausalForestDML into interpretable segmentation rules:

```bash
uv run cig policy -d data/transactions.csv -t discount_pct -y revenue --max-depth 2
```

### Counterfactual Simulation (`cig counterfactual`)

Perform Pearl Level 3 counterfactual abduction to simulate individual and population interventions:

```bash
uv run cig counterfactual -d data/transactions.csv -t discount_pct -v 0.20 -y revenue
```

### Active Inference Planning (`cig active`)

Arbitrate pragmatic utility and epistemic value via Expected Free Energy (EFE) minimization:

```bash
uv run cig active -y revenue -v 1500.0 -d data/transactions.csv
```

### Temporal Mechanism Invariance Monitoring (`cig monitor`)

Audit structural equations across time slices to detect unobserved interventions or regime changes:

```bash
# Audit invariance and fail CI if structural mechanisms break
uv run cig monitor --data data/transactions.csv --time-col order_date --alpha 0.05 --strict
```

---

## CI/CD Gatekeeping (`cig check`)

Integrate `cig check` in pull requests to guard against DAG cycles, broken backdoor identification, and low sensitivity tolerance:

```bash
uv run cig check . \
  --manifest target/manifest.json \
  --data data/transactions.csv \
  --treatment discount_pct \
  --outcome revenue \
  --min-rv 0.10 \
  --min-e-value 1.5 \
  --summary-file pr_summary.md
```

### GitHub Actions Workflow

Add `.github/workflows/cig_gatekeeper.yml`:

```yaml
name: CIG Causal Gatekeeper

on:
  pull_request:
    paths:
      - "models/**"
      - "dbt/**"
      - "**/*.sql"
      - "**/*.py"
      - "**/manifest.json"
      - "data/**"
  workflow_dispatch:

env:
  UV_PYTHON: "3.11"
  PYO3_USE_ABI3_FORWARD_COMPATIBILITY: "1"

jobs:
  causal-integrity-audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - name: Checkout Code Repository
        uses: actions/checkout@v4

      - name: Install uv & Python
        uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"
          enable-cache: true

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Install CIG Engine Dependencies
        run: |
          uv sync --frozen --python 3.11 || uv sync --python 3.11

      - name: Run CIG CI/CD Gatekeeper Check
        id: cig_check
        run: |
          uv run --python 3.11 cig check . \
            --treatment "${{ vars.CIG_TREATMENT || 'discount_pct' }}" \
            --outcome "${{ vars.CIG_OUTCOME || 'revenue' }}" \
            --min-rv 0.10 \
            --min-e-value 1.5 \
            --summary-file "$GITHUB_STEP_SUMMARY"
        continue-on-error: true

      - name: Post CIG Audit Summary Comment on Pull Request
        uses: actions/github-script@v7
        if: github.event_name == 'pull_request'
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const fs = require('fs');
            const summaryPath = process.env.GITHUB_STEP_SUMMARY;
            if (fs.existsSync(summaryPath)) {
              const body = fs.readFileSync(summaryPath, 'utf8');
              const { data: comments } = await github.rest.issues.listComments({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
              });
              const botComment = comments.find(c => c.body.includes('🛡️ CIG Causal Gatekeeper Audit Report'));
              if (botComment) {
                await github.rest.issues.updateComment({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  comment_id: botComment.id,
                  body: body
                });
              } else {
                await github.rest.issues.createComment({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  issue_number: context.issue.number,
                  body: body
                });
              }
            }

      - name: Enforce Audit Exit Status
        if: steps.cig_check.outcome != 'success'
        run: |
          echo "::error::CIG Gatekeeper detected breaking causal violations or collider bias."
          exit 1
```

---

## Orchestration Hooks (Airflow & Dagster)

Halt data pipelines when structural invariance fails on production partitions:

### Dagster Asset Check

```python
from cig.integrations.orchestration import make_dagster_asset_check

orders_causal_check = make_dagster_asset_check(
    asset_name="fct_orders",
    cig_graph_path=".cig/cig_graph.json",
    time_col="order_timestamp",
    alpha=0.05,
)
```

### Airflow Python Operator / TaskFlow

```python
from airflow.decorators import task
from cig.integrations.orchestration import airflow_causal_invariance_operator

@task
def audit_orders_invariance(df):
    airflow_causal_invariance_operator(
        df,
        cig_graph_path=".cig/cig_graph.json",
        time_col="order_timestamp",
        alpha=0.01,
    )
```

---

## IDE & AI Agent Guardrails

Prevent LLM assistants from generating unadjusted or collider-biased regressions:

```bash
# Verify regression code directly via CLI
echo 'import statsmodels.formula.api as smf; model = smf.ols("revenue ~ discount_rate", data=df).fit()' | uv run cig-hook --strict
```

### Cursor Rules (`.cursorrules`)

```markdown
# Causal Inference Guardrail
Always verify candidate regressions against `.cig/cig_graph.json`.
Run `uv run cig-hook --file <target_file> --strict` before committing changes.
```

### Claude Code (`.claude/settings.json`)

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

---

## Development & Testing

Execute the comprehensive Rust and Python test suites:

```bash
# Run Rust core tests
cargo test --workspace --no-default-features

# Run Python pipeline, CLI, and integration tests
uv run pytest tests/ -v
```

---

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
