from __future__ import annotations

from pathlib import Path
from click.testing import CliRunner
import numpy as np
import pandas as pd
from cig.cli import main
import click.testing
from cig.cli import main


def test_cig_check_passes_on_valid_lineage(tmp_path: Path):
    runner = CliRunner()
    models = tmp_path / "models"
    models.mkdir()

    # Clean lineage: price -> revenue, discount_pct -> revenue
    (models / "pricing.sql").write_text(
        "SELECT discount_pct, price * (1 - discount_pct) AS revenue FROM raw;"
    )

    n = 100
    np.random.seed(42)
    disc = np.random.uniform(0.01, 0.20, n)
    rev = 500.0 - 200.0 * disc + np.random.normal(0, 5, n)
    data_file = tmp_path / "data.csv"
    pd.DataFrame(
        {"discount_pct": disc, "revenue": rev, "price": np.full(n, 50.0)}
    ).to_csv(data_file, index=False)

    res = runner.invoke(
        main,
        [
            "check",
            str(tmp_path),
            "--data",
            str(data_file),
            "--treatment",
            "discount_pct",
            "--outcome",
            "revenue",
            "--min-rv",
            "0.10",
        ],
    )

    assert res.exit_code == 0
    assert "All causal regression checks passed successfully" in res.output


def test_cig_check_fails_on_broken_sensitivity(tmp_path: Path):
    runner = CliRunner()
    models = tmp_path / "models"
    models.mkdir()

    # Provide valid SQL lineage: discount_pct -> revenue
    (models / "pricing.sql").write_text(
        "SELECT discount_pct, (1 - discount_pct) * 100 AS revenue FROM raw;"
    )

    # Weak/noisy observational relationship where RV will be very low
    n = 100
    np.random.seed(42)
    disc = np.random.uniform(0.01, 0.20, n)
    rev = 500.0 + np.random.normal(0, 100, n)  # No real effect in data
    data_file = tmp_path / "data.csv"
    pd.DataFrame({"discount_pct": disc, "revenue": rev}).to_csv(data_file, index=False)

    res = runner.invoke(
        main,
        [
            "check",
            str(tmp_path),
            "--data",
            str(data_file),
            "--treatment",
            "discount_pct",
            "--outcome",
            "revenue",
            "--min-rv",
            "0.90",  # Demanding impossible robustness
        ],
    )

    assert res.exit_code == 1
    assert "is below minimum threshold" in res.output


def test_gatekeeper_summary_file_output(tmp_path):
    summary_out = tmp_path / "summary.md"
    runner = click.testing.CliRunner()

    result = runner.invoke(
        main,
        [
            "check",
            "demo_cig",
            "--data",
            "demo_cig/data/transactions.csv",
            "--treatment",
            "discount_pct",
            "--outcome",
            "revenue",
            "--summary-file",
            str(summary_out),
        ],
    )

    assert summary_out.exists()
    content = summary_out.read_text(encoding="utf-8")
    assert "🛡️ CIG Causal Gatekeeper Audit Report" in content
    assert "Cinelli-Hazlett RV_q Bound" in content
    assert "✅ Passed" in content
