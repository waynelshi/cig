"""
Pass 2: Semantic & Domain Hypothesis Extractor.
Extracts hypothesized causal mechanisms from documentation, literature notes,
dbt schema.yml column descriptions, and dbt doc blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import yaml
from pydantic import BaseModel, Field


class InferredEdge(BaseModel):
    source: str = Field(description="Hypothesized cause variable")
    target: str = Field(description="Hypothesized effect variable")
    edge_type: str = "INFERRED"
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    provenance: str = Field(description="Document, section, or schema.yml field")
    rationale: str = Field(description="Domain explanation or semantic context")


class SemanticCausalExtractor:
    """Extracts causal statements from Markdown files and notes."""

    STOPWORDS: set[str] = {
        "in",
        "our",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "on",
        "with",
        "by",
        "at",
        "from",
        "as",
        "into",
        "through",
        "across",
        "furthermore",
        "moreover",
        "however",
        "empirical",
        "study",
        "market",
        "overall",
        "general",
        "changes",
        "change",
        "all",
        "each",
        "every",
        "some",
        "higher",
        "lower",
        "more",
        "less",
        "increased",
        "decreased",
        "increasing",
        "decreasing",
        "reduced",
        "is",
        "are",
        "was",
        "were",
        "been",
        "table",
        "model",
        "aggregate",
        "column",
        "dataset",
    }

    PREP_CUTOFF = re.compile(
        r"\s+(?:across|for|in|during|within|throughout|among)\s+.*$", re.IGNORECASE
    )

    # Active patterns: matches "X directly impacted Y" or "changes in X shifted Y"
    ACTIVE_PATTERN = re.compile(
        r"(?:(?:changes|variations|shifts|increases|reductions)\s+in\s+)?(?P<source>[\w\s\.\_\-]+?)\s+"
        r"(?:directly\s+|significantly\s+|strongly\s+)?"
        r"(?:drives?|drove|causes?|caused|causing|"
        r"impacts?|impacted|impacting|"
        r"affects?|affected|affecting|"
        r"influences?|influenced|influencing|"
        r"shifts?|shifted|shifting|"
        r"alters?|altered|altering|"
        r"triggers?|triggered|triggering|"
        r"boosts?|boosted|boosting|"
        r"leads?\s+to|led\s+to|results?\s+in|resulted\s+in)\s+"
        r"(?P<target>[\w\s\.\_\-]+)",
        re.IGNORECASE,
    )

    # Passive patterns: matches "Y is driven by X"
    PASSIVE_PATTERN = re.compile(
        r"(?P<target>[\w\s\.\_\-]+?)\s+is\s+(?:directly\s+|significantly\s+|strongly\s+)?"
        r"(?:driven|caused|affected|influenced|impacted|shifted)\s+by\s+"
        r"(?:(?:changes|variations|shifts|increases|reductions)\s+in\s+)?(?P<source>[\w\s\.\_\-]+)",
        re.IGNORECASE,
    )

    LEAD_IN_PHRASES = re.compile(
        r"^(?:empirical\s+analysis\s+(?:indicates|shows|suggests)\s+(?:that\s+)?|"
        r"customer\s+feedback\s+(?:shows|indicates|reveals)\s+(?:that\s+)?|"
        r"data\s+(?:shows|demonstrates|indicates|reveals)\s+(?:that\s+)?|"
        r"our\s+(?:study|research|analysis)\s+(?:shows|finds)\s+(?:that\s+)?|"
        r"it\s+is\s+(?:observed|noted|known)\s+that\s+)",
        re.IGNORECASE,
    )

    def _clean_entity(self, text: str) -> str:
        """Strips punctuation, introductory clauses, prepositional tails, and stopwords."""
        # 1. Strip introductory clauses
        text = self.LEAD_IN_PHRASES.sub("", text.strip())

        # 2. Strip prepositional tails
        text = self.PREP_CUTOFF.sub("", text)
        tokens = [t.strip("`*_,.:;\"'()[]{}") for t in text.split()]

        # 3. Filter stopwords iteratively from head and tail
        while tokens and tokens[0].lower() in self.STOPWORDS:
            tokens.pop(0)
        while tokens and tokens[-1].lower() in self.STOPWORDS:
            tokens.pop()

        cleaned = " ".join(tokens).strip()
        if " " in cleaned:
            cleaned = "_".join(cleaned.split())
        return cleaned.lower()

    def parse_text(self, text: str, provenance: str = "doc.md") -> list[InferredEdge]:
        edges: list[InferredEdge] = []
        # Split on sentence boundaries, newlines, and clause commas
        clauses = re.split(r"[\.\;\n\r\,]+", text)

        for clause in clauses:
            clause_clean = clause.strip()
            if not clause_clean or len(clause_clean) < 5:
                continue

            # 1. Passive pattern ("Y is driven by X")
            matched = False
            for match in self.PASSIVE_PATTERN.finditer(clause_clean):
                src = self._clean_entity(match.group("source"))
                dst = self._clean_entity(match.group("target"))
                if src and dst and src != dst and len(src) > 1 and len(dst) > 1:
                    edges.append(
                        InferredEdge(
                            source=src,
                            target=dst,
                            confidence=0.75,
                            provenance=provenance,
                            rationale=clause_clean,
                        )
                    )
                    matched = True

            if matched:
                continue

            # 2. Active pattern ("X drives Y")
            for match in self.ACTIVE_PATTERN.finditer(clause_clean):
                src = self._clean_entity(match.group("source"))
                dst = self._clean_entity(match.group("target"))
                if src and dst and src != dst and len(src) > 1 and len(dst) > 1:
                    edges.append(
                        InferredEdge(
                            source=src,
                            target=dst,
                            confidence=0.75,
                            provenance=provenance,
                            rationale=clause_clean,
                        )
                    )

        return edges

    def parse_file(self, file_path: str | Path) -> list[InferredEdge]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Documentation file not found: {path}")
        return self.parse_text(path.read_text(encoding="utf-8"), provenance=str(path))


class DBTSchemaDocExtractor:
    """
    Parses dbt schema.yml / models.yml files and doc blocks ({% docs ... %})
    to extract domain causal hypotheses from column descriptions and model metadata.
    """

    def __init__(self, semantic_extractor: SemanticCausalExtractor | None = None):
        self.semantic_extractor = semantic_extractor or SemanticCausalExtractor()

    def parse_doc_blocks(
        self, text: str, provenance: str = "docs.md"
    ) -> list[InferredEdge]:
        """Extracts causal hypotheses from dbt {% docs doc_name %} blocks."""
        edges: list[InferredEdge] = []
        doc_block_regex = re.compile(
            r"{%\s*docs\s+([\w_]+)\s*%}(.*?){%\s*enddocs\s*%}",
            re.DOTALL | re.IGNORECASE,
        )

        for match in doc_block_regex.finditer(text):
            doc_name = match.group(1).strip()
            block_content = match.group(2).strip()
            parsed = self.semantic_extractor.parse_text(
                block_content,
                provenance=f"{provenance}#docs({doc_name})",
            )
            edges.extend(parsed)
        return edges

    def parse_schema_dict(
        self, data: dict[str, Any], provenance: str = "schema.yml"
    ) -> list[InferredEdge]:
        """Traverses dbt YAML structures (models, sources, semantic_models) to extract column descriptions."""
        edges: list[InferredEdge] = []
        if not isinstance(data, dict):
            return edges

        # 1. Models & Seeds
        models = data.get("models", []) or []
        for model in models:
            if not isinstance(model, dict):
                continue
            model_name = model.get("name", "unknown_model")

            model_desc = model.get("description", "")
            if model_desc:
                edges.extend(
                    self.semantic_extractor.parse_text(
                        model_desc,
                        provenance=f"{provenance}:{model_name}",
                    )
                )

            for col in model.get("columns", []) or []:
                if not isinstance(col, dict):
                    continue
                col_name = col.get("name", "")
                col_desc = col.get("description", "")
                target_qualified = (
                    f"{model_name}.{col_name}" if col_name else model_name
                )

                if col_desc:
                    extracted = self.semantic_extractor.parse_text(
                        col_desc,
                        provenance=f"{provenance}:{target_qualified}",
                    )
                    edges.extend(extracted)

        # 2. Semantic Models / Metrics (dbt MetricFlow)
        semantic_models = data.get("semantic_models", []) or []
        for sm in semantic_models:
            if not isinstance(sm, dict):
                continue
            sm_name = sm.get("name", "")
            for dim in sm.get("dimensions", []) or []:
                desc = dim.get("description", "")
                if desc:
                    edges.extend(
                        self.semantic_extractor.parse_text(
                            desc,
                            provenance=f"{provenance}:{sm_name}.{dim.get('name', '')}",
                        )
                    )

        return edges

    def parse_yaml_file(self, file_path: str | Path) -> list[InferredEdge]:
        """Parses a dbt schema.yml, sources.yml, or models.yml file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"dbt schema file not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        edges: list[InferredEdge] = []

        edges.extend(self.parse_doc_blocks(raw_text, provenance=str(path)))

        try:
            parsed_yaml = yaml.safe_load(raw_text)
            if isinstance(parsed_yaml, dict):
                edges.extend(self.parse_schema_dict(parsed_yaml, provenance=str(path)))
        except Exception:
            edges.extend(
                self.semantic_extractor.parse_text(raw_text, provenance=str(path))
            )

        return edges
