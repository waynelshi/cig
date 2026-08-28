import json
from cig.passes.pass1_lineage import DBTManifestParser


def test_dbt_manifest_parser(tmp_path):
    mock_manifest = {
        "nodes": {
            "model.jaffle_shop.stg_orders": {
                "resource_type": "model",
                "name": "stg_orders",
                "alias": "stg_orders",
                "original_file_path": "models/staging/stg_orders.sql",
                "compiled_code": """
                    SELECT
                        order_id,
                        customer_id,
                        order_total - tax_amount AS net_amount
                    FROM raw_orders
                """,
            },
            "model.jaffle_shop.customers": {
                "resource_type": "model",
                "name": "customers",
                "alias": "customers",
                "original_file_path": "models/marts/customers.sql",
                "compiled_code": """
                    SELECT
                        customer_id,
                        SUM(net_amount) AS lifetime_value
                    FROM stg_orders
                """,
            },
        },
        "sources": {
            "source.jaffle_shop.raw_orders": {
                "resource_type": "source",
                "name": "raw_orders",
                "alias": "raw_orders",
            }
        },
        "parent_map": {
            "model.jaffle_shop.stg_orders": ["source.jaffle_shop.raw_orders"],
            "model.jaffle_shop.customers": ["model.jaffle_shop.stg_orders"],
        },
    }

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(mock_manifest))

    parser = DBTManifestParser()
    edges = parser.parse_file(manifest_file)

    assert len(edges) >= 4

    edge_pairs = {(e.source, e.target) for e in edges}

    # Model/Table Level
    assert ("raw_orders", "stg_orders") in edge_pairs
    assert ("stg_orders", "customers") in edge_pairs

    # Column Level
    assert ("order_total", "stg_orders.net_amount") in edge_pairs
    assert ("tax_amount", "stg_orders.net_amount") in edge_pairs
    assert ("net_amount", "customers.lifetime_value") in edge_pairs
