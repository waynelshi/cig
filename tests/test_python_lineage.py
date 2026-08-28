import pytest
from cig.passes.pass1_lineage import PythonLineageParser


@pytest.fixture
def py_parser():
    return PythonLineageParser()


def test_pandas_subscript_assignment(py_parser):
    code = """
import pandas as pd
import numpy as np

df['unit_margin'] = df['sale_price'] - df['unit_cost']
df['log_spend'] = np.log(df['ad_spend'] + 1.0)
    """
    edges = py_parser.parse_code(code, provenance="feature_engineering.py")

    assert len(edges) == 3
    edge_map = {(e.source, e.target) for e in edges}

    assert ("sale_price", "unit_margin") in edge_map
    assert ("unit_cost", "unit_margin") in edge_map
    assert ("ad_spend", "log_spend") in edge_map


def test_pandas_assign_method(py_parser):
    code = """
df = df.assign(
    total_cost=df['fixed_cost'] + df['variable_cost'],
    roi=df['net_revenue'] / df['total_investment']
)
    """
    edges = py_parser.parse_code(code, provenance="pipeline.py")

    assert len(edges) == 4
    edge_map = {(e.source, e.target) for e in edges}

    assert ("fixed_cost", "total_cost") in edge_map
    assert ("variable_cost", "total_cost") in edge_map
    assert ("net_revenue", "roi") in edge_map
    assert ("total_investment", "roi") in edge_map
