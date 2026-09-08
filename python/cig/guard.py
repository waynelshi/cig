"""
Runtime Causal Guard & Collider Bias Interceptor.
Prevents feature engineering pipelines and model datasets from conditioning on
colliders, collider descendants, or mediator leakages.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence
import json
import warnings
import pandas as pd

from cig import PyCausalEngine


class CausalViolationError(ValueError):
    """Raised when a feature set introduces collider bias or causal violations."""

    pass


class CausalGuard:
    """Audits feature sets against a compiled CIG causal DAG."""

    def __init__(
        self, engine: PyCausalEngine | None = None, cig_dir: str | Path = ".cig"
    ):
        if engine is not None:
            self.engine = engine
        else:
            self.engine = self._load_from_artifact(cig_dir)

        # Parse graph topology for fast ancestral and collider queries
        self._graph_data = json.loads(self.engine.export_json())
        self._nodes = {n["name"] for n in self._graph_data["nodes"]}
        self._parents: dict[str, set[str]] = {n: set() for n in self._nodes}
        self._children: dict[str, set[str]] = {n: set() for n in self._nodes}

        for edge in self._graph_data["edges"]:
            if isinstance(edge, (list, tuple)):
                src, dst = edge[0], edge[1]
            elif isinstance(edge, dict):
                src, dst = edge["source"], edge["target"]
            else:
                continue

            if src in self._nodes and dst in self._nodes:
                self._children[src].add(dst)
                self._parents[dst].add(src)

    def _load_from_artifact(self, cig_dir: str | Path) -> PyCausalEngine:
        path = Path(cig_dir) / "cig_graph.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find SCM artifact at '{path}'. Run 'cig run' or pass a PyCausalEngine instance."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        engine = PyCausalEngine()
        for e in data.get("edges", []):
            if isinstance(e, (list, tuple)):
                src, dst = e[0], e[1]
                edge_type = "EXTRACTED"
                conf = 1.0
            else:
                src = e.get("source")
                dst = e.get("target")
                edge_type = e.get("edge_type", "EXTRACTED")
                conf = float(e.get("confidence", 1.0))

            if edge_type == "EXTRACTED":
                engine.add_extracted_edge(src, dst, None, None)
            elif edge_type in ("INFERRED", "DISCOVERED"):
                engine.add_inferred_edge(src, dst, conf, None, None)
        return engine

    def get_descendants(self, node: str) -> set[str]:
        """Returns all downstream descendants of a given node."""
        desc = set()
        queue = [node]
        while queue:
            curr = queue.pop(0)
            for child in self._children.get(curr, set()):
                if child not in desc:
                    desc.add(child)
                    queue.append(child)
        return desc

    def get_ancestors(self, node: str) -> set[str]:
        """Returns all upstream ancestors of a given node."""
        anc = set()
        queue = [node]
        while queue:
            curr = queue.pop(0)
            for parent in self._parents.get(curr, set()):
                if parent not in anc:
                    anc.add(parent)
                    queue.append(parent)
        return anc

    def identify_colliders(self, treatment: str, outcome: str) -> set[str]:
        """
        Identifies all colliders (and descendants of colliders) between treatment and outcome.
        Any node C where Treatment -> ... -> C <- ... <- Outcome (or common descendants)
        causes selection bias / collider stratification when conditioned upon.
        """
        desc_t = self.get_descendants(treatment)
        desc_y = self.get_descendants(outcome)

        # 1. Direct mutual descendants: T -> ... -> C and Y -> ... -> C
        mutual_descendants = desc_t.intersection(desc_y)

        # 2. In-path structural colliders: nodes with >= 2 parents where parents trace back to T and Y/Confounders
        structural_colliders = set()
        for node in self._nodes:
            if node in (treatment, outcome):
                continue
            parents = self._parents.get(node, set())
            if len(parents) >= 2:
                # Check if this node receives directed influence from T and Y
                anc = self.get_ancestors(node)
                if treatment in anc and outcome in anc:
                    structural_colliders.add(node)
                    structural_colliders.update(self.get_descendants(node))

        return mutual_descendants.union(structural_colliders)

    def identify_mediators(self, treatment: str, outcome: str) -> set[str]:
        """Identifies intermediate variables along the directed causal path T -> M -> Y."""
        desc_t = self.get_descendants(treatment)
        anc_y = self.get_ancestors(outcome)
        return (desc_t.intersection(anc_y)) - {treatment, outcome}

    def audit_feature_set(
        self,
        features: Sequence[str],
        treatment: str,
        outcome: str,
        forbid_colliders: bool = True,
        forbid_mediators: bool = False,
    ) -> list[str]:
        """Returns violation messages if the feature set violates causal constraints."""
        violations = []
        feature_set = set(features)

        if forbid_colliders:
            colliders = self.identify_colliders(treatment, outcome)
            conditioned_colliders = feature_set.intersection(colliders)
            if conditioned_colliders:
                violations.append(
                    f"Collider Bias Detected: Features {sorted(conditioned_colliders)} are colliders "
                    f"or collider descendants between '{treatment}' and '{outcome}'. "
                    f"Conditioning on them opens spurious non-causal associations."
                )

        if forbid_mediators:
            mediators = self.identify_mediators(treatment, outcome)
            conditioned_mediators = feature_set.intersection(mediators)
            if conditioned_mediators:
                violations.append(
                    f"Mediator Blockade Detected: Features {sorted(conditioned_mediators)} are on the "
                    f"direct causal path between '{treatment}' and '{outcome}'. "
                    f"Conditioning on them absorbs the causal mechanism."
                )

        return violations


def causal_guard(
    treatment: str,
    outcome: str,
    engine: PyCausalEngine | None = None,
    cig_dir: str | Path = ".cig",
    forbid_colliders: bool = True,
    forbid_mediators: bool = False,
    action: str = "raise",  # "raise", "warn", or "prune"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that intercepts pipeline outputs (DataFrame or dict of features)
    and checks for collider stratification bias or mediator leaks.

    Actions:
      - 'raise': Throws CausalViolationError on detection.
      - 'warn': Emits a UserWarning but returns the data.
      - 'prune': Automatically drops violating columns from the returned DataFrame.
    """
    guard = CausalGuard(engine=engine, cig_dir=cig_dir)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)

            # Determine candidate feature names from the returned object or arguments
            if isinstance(result, pd.DataFrame):
                features = list(result.columns)
            elif (
                isinstance(result, (list, tuple))
                and result
                and isinstance(result[0], str)
            ):
                features = list(result)
            elif isinstance(result, dict):
                features = list(result.keys())
            else:
                return result

            violations = guard.audit_feature_set(
                features=features,
                treatment=treatment,
                outcome=outcome,
                forbid_colliders=forbid_colliders,
                forbid_mediators=forbid_mediators,
            )

            if violations:
                msg = "\n".join(violations)
                if action == "raise":
                    raise CausalViolationError(f"[CIG @causal_guard Violation]\n{msg}")
                elif action == "warn":
                    warnings.warn(f"[CIG @causal_guard Warning]\n{msg}", stacklevel=2)
                elif action == "prune" and isinstance(result, pd.DataFrame):
                    colliders = (
                        guard.identify_colliders(treatment, outcome)
                        if forbid_colliders
                        else set()
                    )
                    mediators = (
                        guard.identify_mediators(treatment, outcome)
                        if forbid_mediators
                        else set()
                    )
                    drop_cols = [
                        col
                        for col in result.columns
                        if col in colliders or col in mediators
                    ]
                    warnings.warn(
                        f"[CIG @causal_guard Prune] Dropped violating columns from pipeline: {drop_cols}",
                        stacklevel=2,
                    )
                    return result.drop(columns=drop_cols)

            return result

        return wrapper

    return decorator
