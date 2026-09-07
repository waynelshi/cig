from __future__ import annotations

from pathlib import Path
from click.testing import CliRunner
import numpy as np
import pandas as pd
from cig.cli import main


def test_cig_run_pipeline_end_to_end(tmp_path: Path):
    runner = CliRunner()

    # 1. Setup minimal repository structure
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # SQL lineage
    sql_file = models_dir / "pricing.sql"
    sql_file.write_text(
        "SELECT discount_pct, price * (1 - discount_pct) AS revenue FROM orders;"
    )

    # Python AST
    py_file = models_dir / "features.py"
    py_file.write_text("df['quantity'] = df['marketing_tier'] * 2")

    # CSV data
    n = 150
    np.random.seed(42)
    disc = np.random.uniform(0.01, 0.20, n)
    rev = 500.0 - 200.0 * disc + np.random.normal(0, 5, n)
    df = pd.DataFrame(
        {"discount_pct": disc, "revenue": rev, "marketing_tier": np.ones(n)}
    )
    csv_file = data_dir / "transactions.csv"
    df.to_csv(csv_file, index=False)

    # 2. Invoke `cig run`
    result = runner.invoke(
        main,
        [
            "run",
            str(tmp_path),
            "--data",
            str(csv_file),
            "--treatment",
            "discount_pct",
            "--outcome",
            "revenue",
            "--goal",
            "480.0",
        ],
    )

    assert result.exit_code == 0, f"CLI failed with: {result.output}"
    assert "SCM Pipeline Compilation Summary" in result.output
    assert "Pass 4: Automated Econometric & Sensitivity Audit" in result.output
    assert "Pass 6: Active Inference EFE Policy Optimization" in result.output

    # 3. Verify emitted artifacts
    dot_cig = tmp_path / ".cig"
    assert (dot_cig / "cig_graph.json").exists()
    assert (dot_cig / "cig.html").exists()
    assert (dot_cig / "SCM.md").exists()
    assert (dot_cig / "cig_graph.graphml").exists()
