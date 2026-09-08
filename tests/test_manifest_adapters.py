from __future__ import annotations

import json
from pathlib import Path
from cig.passes.pass1_lineage import DBTManifestParser, SQLMeshManifestParser


def test_dbt_semantic_metrics_lineage(tmp_path: Path):
    manifest_data = {
        "nodes": {
            "model.ecommerce.orders": {
                "resource_type": "model",
                "name": "orders",
                "compiled_code": "SELECT order_id, order_total FROM raw_orders",
            }
        },
        "sources": {},
        "parent_map": {"model.ecommerce.orders": []},
        "metrics": {
            "metric.ecommerce.total_revenue": {
                "name": "total_revenue",
                "type": "simple",
                "type_params": {"measure": "order_total"},
            },
            "metric.ecommerce.revenue_per_order": {
                "name": "revenue_per_order",
                "type": "derived",
                "type_params": {"metrics": [{"name": "total_revenue"}, "order_count"]},
            },
        },
    }

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))

    parser = DBTManifestParser()
    edges = parser.parse_file(manifest_file)
    pairs = {(e.source, e.target) for e in edges}

    assert ("order_total", "total_revenue") in pairs
    assert ("total_revenue", "revenue_per_order") in pairs
    assert ("order_count", "revenue_per_order") in pairs


def test_sqlmesh_manifest_parser(tmp_path: Path):
    sqlmesh_data = {
        "models": {
            "db.marts.revenue": {
                "name": "db.marts.revenue",
                "depends_on": ["db.staging.stg_orders"],
                "query": "SELECT order_id, price * (1 - discount_pct) AS net_revenue FROM db.staging.stg_orders",
            }
        }
    }

    manifest_file = tmp_path / "sqlmesh_state.json"
    manifest_file.write_text(json.dumps(sqlmesh_data))

    parser = SQLMeshManifestParser()
    edges = parser.parse_file(manifest_file)
    pairs = {(e.source, e.target) for e in edges}

    # Table-level
    assert ("db.staging.stg_orders", "db.marts.revenue") in pairs
    # Column-level
    assert ("discount_pct", "db.marts.revenue.net_revenue") in pairs
    assert ("price", "db.marts.revenue.net_revenue") in pairs
