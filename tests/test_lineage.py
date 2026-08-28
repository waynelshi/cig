import pytest
from cig.passes.pass1_lineage import SQLLineageParser


@pytest.fixture
def parser():
    return SQLLineageParser(dialect="postgres")


def test_basic_arithmetic_lineage(parser):
    sql = """
    CREATE TABLE revenue_mart AS
    SELECT 
        unit_price * quantity * (1.0 - discount_rate) AS final_revenue
    FROM raw_orders;
    """
    edges = parser.parse_query(sql, provenance="revenue_mart.sql")

    assert len(edges) == 3
    sources = {e.source for e in edges}
    assert sources == {"unit_price", "quantity", "discount_rate"}

    for edge in edges:
        assert edge.target == "revenue_mart.final_revenue"
        assert edge.confidence == 1.0
        assert edge.edge_type == "EXTRACTED"


def test_case_when_lineage(parser):
    sql = """
    SELECT 
        CASE 
            WHEN age >= 65 THEN 'senior'
            WHEN age >= 18 THEN 'adult'
            ELSE 'minor'
        END AS age_group,
        income / household_size AS income_per_capita
    FROM census_data;
    """
    edges = parser.parse_query(sql, provenance="demographics.sql")

    # age -> age_group
    # income -> income_per_capita
    # household_size -> income_per_capita
    assert len(edges) == 3

    edge_map = {(e.source, e.target) for e in edges}
    assert ("age", "age_group") in edge_map
    assert ("income", "income_per_capita") in edge_map
    assert ("household_size", "income_per_capita") in edge_map


def test_subquery_cte_lineage(parser):
    sql = """
    WITH filtered_orders AS (
        SELECT order_id, order_value, tax_amount
        FROM orders
        WHERE status = 'completed'
    )
    SELECT 
        order_value + tax_amount AS total_charged
    FROM filtered_orders;
    """
    edges = parser.parse_query(sql, provenance="orders_cte.sql")

    sources = {e.source for e in edges if e.target == "total_charged"}
    assert sources == {"order_value", "tax_amount"}
