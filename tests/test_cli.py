import pandas as pd
from click.testing import CliRunner
from cig.cli import main


def test_cli_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # 1. cig init
    res_init = runner.invoke(main, ["init"])
    assert res_init.exit_code == 0
    assert (tmp_path / ".cig").exists()

    # 2. Create mock SQL and data files
    sql_file = tmp_path / "marketing.sql"
    sql_file.write_text(
        """
        CREATE TABLE mart AS
        SELECT 
            ad_spend,
            conversions,
            ad_spend * 2.0 AS revenue
        FROM raw_data;
        """
    )

    data_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "ad_spend": [10, 20, 30, 40, 50] * 20,
            "conversions": [1, 2, 3, 4, 5] * 20,
            "revenue": [20, 40, 60, 80, 100] * 20,
        }
    )
    df.to_csv(data_file, index=False)

    # 3. cig scan
    res_scan = runner.invoke(main, ["scan", "."])
    assert res_scan.exit_code == 0
    assert (tmp_path / ".cig" / "cig_graph.json").exists()
    assert (tmp_path / ".cig" / "cig.html").exists()

    # 4. cig do
    res_do = runner.invoke(
        main,
        ["do", "-t", "ad_spend", "-y", "revenue", "-d", str(data_file)],
    )
    assert res_do.exit_code == 0
    assert "Estimated Average Treatment Effect" in res_do.output
