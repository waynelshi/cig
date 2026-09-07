"""
SCM Namespace Reconciliation Engine.
Normalizes table-qualified, alias-prefixed, and colloquial variable tokens
into canonical node identifiers across SCM passes.
"""

from __future__ import annotations

import re
from typing import Sequence


class NamespaceResolver:
    """
    Reconciles variant node names (e.g. 'pricing_mart.revenue' vs 'revenue',
    'df["churn_rate"]', 'user_tenure_months') into canonical node identifiers.
    """

    STRIP_CHARS = "\"`'{}()"

    def __init__(self, canonical_names: Sequence[str] | None = None):
        self.canonical_names: set[str] = set(canonical_names or [])
        self._name_map: dict[str, str] = {c: c for c in self.canonical_names}

    def register_canonical(self, names: Sequence[str]) -> None:
        """Adds verified target variable names to canonical lookup set."""
        for n in names:
            clean = self.sanitize(n)
            self.canonical_names.add(clean)
            self._name_map[clean] = clean

    @staticmethod
    def sanitize(raw_name: str) -> str:
        """Strips surrounding indexing, brackets, backticks, quotes, and whitespace."""
        s = raw_name.strip()

        # Handle indexing expressions like df['col'] or data["col"]
        match = re.search(r"\[['\"]?([^'\"\]]+)['\"]?\]", s)
        if match:
            s = match.group(1)

        for ch in NamespaceResolver.STRIP_CHARS:
            s = s.replace(ch, "")
        return s.strip().lower()

    @staticmethod
    def extract_terminal_column(name: str) -> str:
        """Extracts terminal column from table.column or schema.table.column."""
        clean = NamespaceResolver.sanitize(name)
        if "." in clean:
            return clean.split(".")[-1]
        return clean

    def resolve(self, raw_name: str) -> str:
        """
        Resolves any raw token to its canonical SCM node identifier.
        """
        clean = self.sanitize(raw_name)

        # 1. Exact canonical match
        if clean in self._name_map:
            return self._name_map[clean]

        # 2. Extract naked column if input is table-qualified ('pricing_mart.revenue' -> 'revenue')
        terminal = self.extract_terminal_column(clean)
        if terminal in self.canonical_names:
            self._name_map[clean] = terminal
            return terminal

        # 3. If a table-qualified canonical exists for a naked column input ('revenue' -> 'pricing_mart.revenue')
        qualified_matches = [c for c in self.canonical_names if c.endswith(f".{clean}")]
        if len(qualified_matches) == 1:
            self._name_map[clean] = qualified_matches[0]
            return qualified_matches[0]

        # 4. Strip common prefixes: df_..., data_..., features_..., mart_...
        df_stripped = re.sub(r"^(?:df|data|features|mart|dim|fct)_+", "", clean)
        if df_stripped in self.canonical_names:
            self._name_map[clean] = df_stripped
            return df_stripped

        # 5. Default fallback: keep cleaned naked terminal column
        self._name_map[clean] = terminal
        return terminal
