from cig.passes.pass2_semantic import DBTSchemaDocExtractor


def test_dbt_schema_yml_column_extraction(tmp_path):
    schema_content = """
    version: 2

    models:
      - name: customers
        description: "Customer aggregate table. user_tenure drives customer_loyalty."
        columns:
          - name: discount_rate
            description: "Higher discount_rate leads to higher conversion_rate."
          - name: churn_risk
            description: "churn_risk is driven by support_tickets."

      - name: orders
        columns:
          - name: delivery_delay
            description: "delivery_delay directly impacts refund_requests."
    """

    schema_file = tmp_path / "schema.yml"
    schema_file.write_text(schema_content)

    extractor = DBTSchemaDocExtractor()
    edges = extractor.parse_yaml_file(schema_file)

    assert len(edges) >= 4
    pairs = {(e.source, e.target) for e in edges}

    assert ("user_tenure", "customer_loyalty") in pairs
    assert ("discount_rate", "conversion_rate") in pairs
    assert ("support_tickets", "churn_risk") in pairs
    assert ("delivery_delay", "refund_requests") in pairs

    for edge in edges:
        assert edge.edge_type == "INFERRED"
        assert edge.confidence == 0.75
        assert "schema.yml" in edge.provenance


def test_dbt_doc_block_extraction(tmp_path):
    doc_block_content = """
    {% docs churn_explanation %}
    pricing_tier affects churn_rate across all enterprise accounts.
    customer_satisfaction influences renewal_probability.
    {% enddocs %}
    """

    doc_file = tmp_path / "docs.md"
    doc_file.write_text(doc_block_content)

    extractor = DBTSchemaDocExtractor()
    edges = extractor.parse_doc_blocks(doc_block_content, provenance=str(doc_file))

    assert len(edges) == 2
    pairs = {(e.source, e.target) for e in edges}

    assert ("pricing_tier", "churn_rate") in pairs
    assert ("customer_satisfaction", "renewal_probability") in pairs
