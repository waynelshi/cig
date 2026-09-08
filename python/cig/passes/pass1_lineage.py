"""
Pass 1: Deterministic Lineage Extractor using SQL and Python Tree-sitter AST parsing.
Extracts column-level causal dependencies with 1.0 confidence (EXTRACTED).
"""

from __future__ import annotations

from pathlib import Path
import json
from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser


class ExtractedEdge(BaseModel):
    source: str = Field(description="Upstream cause (source column/feature)")
    target: str = Field(description="Downstream effect (target column/feature)")
    edge_type: str = "EXTRACTED"
    confidence: float = 1.0
    provenance: str = Field(
        description="File, function, or script where relationship was found"
    )
    expression: str = Field(
        description="The underlying deterministic formula or code snippet"
    )


class SQLLineageParser:
    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def parse_query(
        self, sql_code: str, provenance: str = "raw_query"
    ) -> list[ExtractedEdge]:
        edges: list[ExtractedEdge] = []
        try:
            parsed_statements = sqlglot.parse(sql_code, read=self.dialect)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL from {provenance}: {e}")

        for statement in parsed_statements:
            if not statement:
                continue

            target_table = None
            if isinstance(statement, (exp.Create, exp.Insert)):
                target_table = statement.this.name

            selects = statement.find_all(exp.Select)
            for select in selects:
                for projection in select.expressions:
                    target_col = projection.alias_or_name
                    if not target_col:
                        continue

                    target_node = (
                        f"{target_table}.{target_col}" if target_table else target_col
                    )

                    source_cols = set()
                    for col in projection.find_all(exp.Column):
                        col_name = col.name
                        if col_name != target_col:
                            source_cols.add(col_name)

                    for src in sorted(source_cols):
                        edges.append(
                            ExtractedEdge(
                                source=src,
                                target=target_node,
                                edge_type="EXTRACTED",
                                confidence=1.0,
                                provenance=provenance,
                                expression=projection.sql(dialect=self.dialect),
                            )
                        )
        return edges

    def parse_file(self, file_path: str | Path) -> list[ExtractedEdge]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SQL file not found: {path}")
        return self.parse_query(path.read_text(encoding="utf-8"), provenance=str(path))


class PythonLineageParser:
    def __init__(self):
        self.py_language = Language(tspython.language())
        self.parser = Parser(self.py_language)

    def parse_code(
        self, source_code: str, provenance: str = "script.py"
    ) -> list[ExtractedEdge]:
        """Parses Python source code and extracts column-level assignments from Pandas/DataFrame operations."""
        edges: list[ExtractedEdge] = []
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        root_node = tree.root_node
        source_bytes = source_code.encode("utf-8")

        self._walk_tree(root_node, source_bytes, provenance, edges)
        return edges

    def _walk_tree(
        self,
        node: Node,
        source_bytes: bytes,
        provenance: str,
        edges: list[ExtractedEdge],
    ):
        # Match assignments: df['target'] = ...
        if node.type == "assignment":
            left_node = node.child_by_field_name("left")
            right_node = node.child_by_field_name("right")

            if left_node and right_node:
                target_col = self._extract_column_from_subscript(
                    left_node, source_bytes
                )
                if target_col:
                    source_cols = self._extract_all_subscript_columns(
                        right_node, source_bytes
                    )
                    expr_text = (
                        source_bytes[node.start_byte : node.end_byte]
                        .decode("utf-8")
                        .strip()
                    )

                    for src in sorted(source_cols):
                        if src != target_col:
                            edges.append(
                                ExtractedEdge(
                                    source=src,
                                    target=target_col,
                                    edge_type="EXTRACTED",
                                    confidence=1.0,
                                    provenance=provenance,
                                    expression=expr_text,
                                )
                            )

        # Match .assign() calls: df.assign(target=df['a'] + df['b'])
        if node.type == "call":
            function_node = node.child_by_field_name("function")
            if function_node and function_node.type == "attribute":
                attr_name = self._node_text(
                    function_node.child_by_field_name("attribute"), source_bytes
                )
                if attr_name == "assign":
                    args_node = node.child_by_field_name("arguments")
                    if args_node:
                        self._extract_assign_keyword_args(
                            args_node, source_bytes, provenance, edges
                        )

        # Recursively visit children
        for child in node.children:
            self._walk_tree(child, source_bytes, provenance, edges)

    def _extract_assign_keyword_args(
        self,
        args_node: Node,
        source_bytes: bytes,
        provenance: str,
        edges: list[ExtractedEdge],
    ):
        for child in args_node.children:
            if child.type == "keyword_argument":
                name_node = child.child_by_field_name("name")
                val_node = child.child_by_field_name("value")
                if name_node and val_node:
                    target_col = self._node_text(name_node, source_bytes)
                    source_cols = self._extract_all_subscript_columns(
                        val_node, source_bytes
                    )
                    expr_text = (
                        source_bytes[child.start_byte : child.end_byte]
                        .decode("utf-8")
                        .strip()
                    )

                    for src in sorted(source_cols):
                        if src != target_col:
                            edges.append(
                                ExtractedEdge(
                                    source=src,
                                    target=target_col,
                                    edge_type="EXTRACTED",
                                    confidence=1.0,
                                    provenance=provenance,
                                    expression=expr_text,
                                )
                            )

    def _extract_column_from_subscript(
        self, node: Node, source_bytes: bytes
    ) -> str | None:
        """Detects df['col'] subscript patterns and returns the column name."""
        if node.type == "subscript":
            subscript_node = node.child_by_field_name("subscript")
            if subscript_node and subscript_node.type == "string":
                raw_str = self._node_text(subscript_node, source_bytes)
                return raw_str.strip("'\"")
        return None

    def _extract_all_subscript_columns(
        self, node: Node, source_bytes: bytes
    ) -> set[str]:
        """Recursively finds all df['column'] occurrences inside an expression."""
        cols = set()
        col = self._extract_column_from_subscript(node, source_bytes)
        if col:
            cols.add(col)

        for child in node.children:
            cols.update(self._extract_all_subscript_columns(child, source_bytes))
        return cols

    def _node_text(self, node: Node | None, source_bytes: bytes) -> str:
        if not node:
            return ""
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def parse_file(self, file_path: str | Path) -> list[ExtractedEdge]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Python file not found: {path}")
        return self.parse_code(path.read_text(encoding="utf-8"), provenance=str(path))


class DBTManifestParser:
    """Parses dbt target/manifest.json to extract table, column, and metric lineages."""

    def __init__(self, sql_parser: SQLLineageParser | None = None):
        self.sql_parser = sql_parser or SQLLineageParser()

    def parse_manifest_dict(
        self, manifest: dict, provenance: str = "target/manifest.json"
    ) -> list[ExtractedEdge]:
        edges: list[ExtractedEdge] = []
        nodes = manifest.get("nodes", {})
        parent_map = manifest.get("parent_map", {})
        sources = manifest.get("sources", {})
        metrics = manifest.get("metrics", {})
        semantic_models = manifest.get("semantic_models", {})

        # 1. Model and Column Lineages
        for node_id, node_info in nodes.items():
            resource_type = node_info.get("resource_type")
            if resource_type not in ("model", "snapshot", "seed"):
                continue

            target_model_name = node_info.get("alias") or node_info.get("name")
            if not target_model_name:
                continue

            # Table-level dependencies
            parents = parent_map.get(node_id, [])
            for parent_id in parents:
                parent_info = nodes.get(parent_id) or sources.get(parent_id, {})
                parent_name = (
                    parent_info.get("alias")
                    or parent_info.get("name")
                    or parent_id.split(".")[-1]
                )
                edges.append(
                    ExtractedEdge(
                        source=parent_name,
                        target=target_model_name,
                        edge_type="EXTRACTED",
                        confidence=1.0,
                        provenance=f"{provenance} ({node_id})",
                        expression=f"dbt ref('{parent_name}')",
                    )
                )

            # Column-level lineages via compiled SQL
            compiled_code = (
                node_info.get("compiled_code")
                or node_info.get("compiled_sql")
                or node_info.get("raw_code")
            )
            if compiled_code:
                try:
                    col_edges = self.sql_parser.parse_query(
                        compiled_code,
                        provenance=f"{node_info.get('original_file_path', provenance)}",
                    )
                    for ce in col_edges:
                        target_col = ce.target
                        if "." not in target_col:
                            target_col = f"{target_model_name}.{target_col}"
                        edges.append(
                            ExtractedEdge(
                                source=ce.source,
                                target=target_col,
                                edge_type="EXTRACTED",
                                confidence=1.0,
                                provenance=ce.provenance,
                                expression=ce.expression,
                            )
                        )
                except Exception:
                    pass

        # 2. Semantic Models & Metrics Layer
        for metric_id, m_info in metrics.items():
            metric_name = m_info.get("name", metric_id.split(".")[-1])
            # Direct calculation formulas (e.g. type_params.measure or calculation_method)
            type_params = m_info.get("type_params", {})
            measure = type_params.get("measure")
            if isinstance(measure, str):
                edges.append(
                    ExtractedEdge(
                        source=measure,
                        target=metric_name,
                        edge_type="EXTRACTED",
                        confidence=1.0,
                        provenance=f"{provenance} (metric: {metric_name})",
                        expression=f"metric: {m_info.get('type', 'derived')}",
                    )
                )
            elif isinstance(measure, dict):
                m_name = measure.get("name")
                if m_name:
                    edges.append(
                        ExtractedEdge(
                            source=m_name,
                            target=metric_name,
                            edge_type="EXTRACTED",
                            confidence=1.0,
                            provenance=f"{provenance} (metric: {metric_name})",
                            expression=f"metric: {m_info.get('type', 'derived')}",
                        )
                    )

            # Input metrics for derived metrics
            input_metrics = type_params.get("metrics", [])
            for im in input_metrics:
                im_name = im.get("name") if isinstance(im, dict) else str(im)
                if im_name:
                    edges.append(
                        ExtractedEdge(
                            source=im_name,
                            target=metric_name,
                            edge_type="EXTRACTED",
                            confidence=1.0,
                            provenance=f"{provenance} (derived_metric: {metric_name})",
                            expression=f"derived from {im_name}",
                        )
                    )

        return edges

    def parse_file(self, file_path: str | Path) -> list[ExtractedEdge]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"dbt manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        return self.parse_manifest_dict(manifest_data, provenance=str(path))


class SQLMeshManifestParser:
    """Parses SQLMesh models or context state to extract deterministic DAG lineages."""

    def __init__(self, sql_parser: SQLLineageParser | None = None):
        self.sql_parser = sql_parser or SQLLineageParser()

    def parse_manifest_dict(
        self, state_dict: dict, provenance: str = "sqlmesh_manifest.json"
    ) -> list[ExtractedEdge]:
        edges: list[ExtractedEdge] = []
        models = state_dict.get("models", {})

        for model_key, model_meta in models.items():
            model_name = model_meta.get("name", model_key)
            query = model_meta.get("query") or model_meta.get("definition")
            depends_on = model_meta.get("depends_on", [])

            # Table-level edges
            for dep in depends_on:
                edges.append(
                    ExtractedEdge(
                        source=dep,
                        target=model_name,
                        edge_type="EXTRACTED",
                        confidence=1.0,
                        provenance=provenance,
                        expression="sqlmesh depends_on",
                    )
                )

            # Column-level edges from query if present
            if query and isinstance(query, str):
                try:
                    col_edges = self.sql_parser.parse_query(
                        query, provenance=f"{provenance} ({model_name})"
                    )
                    for ce in col_edges:
                        tgt = ce.target
                        if "." not in tgt:
                            tgt = f"{model_name}.{tgt}"
                        edges.append(
                            ExtractedEdge(
                                source=ce.source,
                                target=tgt,
                                edge_type="EXTRACTED",
                                confidence=1.0,
                                provenance=ce.provenance,
                                expression=ce.expression,
                            )
                        )
                except Exception:
                    pass

        return edges

    def parse_file(self, file_path: str | Path) -> list[ExtractedEdge]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SQLMesh manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.parse_manifest_dict(data, provenance=str(path))
