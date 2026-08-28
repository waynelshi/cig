"""
CIG PreToolUse & Pre-Commit Hook Interceptor.
Validates statistical code, econometrics scripts, and AI tool edits against the unified SCM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
import sys
from pydantic import BaseModel, Field

from cig import PyCausalEngine


class CausalViolation(BaseModel):
    violation_type: str = Field(
        description="'OMITTED_CONFOUNDER', 'CONDITIONED_COLLIDER', or 'CONDITIONED_MEDIATOR'"
    )
    treatment: str
    outcome: str
    covariates_used: list[str]
    problematic_variables: list[str]
    suggested_adjustment_set: list[str]
    message: str


class RegressionSpec(BaseModel):
    treatment: str
    outcome: str
    covariates: list[str]
    provenance_snippet: str


class IDEInterceptor:
    """Inspects code payloads for causal validity against the compiled SCM DAG."""

    # Regex patterns for statsmodels formulas (e.g., 'y ~ t + w1 + w2')
    FORMULA_PATTERN = re.compile(
        r"""(?:smf?\.ols|ols|glm|logit|probit)\s*\(\s*(?:formula\s*=\s*)?["'](?P<formula>[^"']+)["']""",
        re.IGNORECASE,
    )

    # Regex patterns for Python regressions: LinearRegression().fit(df[['t', 'w']], df['y'])
    SKLEARN_FIT_PATTERN = re.compile(
        r"""\.fit\s*\(\s*(?:X\s*=\s*)?(?P<features>\[[^\]]+\]|df\[[^\]]+\]|\w+)\s*,\s*(?:y\s*=\s*)?(?P<target>df\[[^\]]+\]|\w+)""",
        re.IGNORECASE,
    )

    def __init__(
        self,
        engine: PyCausalEngine | None = None,
        graph_json_path: str | Path = ".cig/cig_graph.json",
    ):
        self.engine = engine or self._load_engine_from_json(graph_json_path)

    def _load_engine_from_json(self, graph_json_path: str | Path) -> PyCausalEngine:
        path = Path(graph_json_path)
        engine = PyCausalEngine()
        if not path.exists():
            return engine

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for edge in data.get("edges", []):
            if edge["edge_type"] == "EXTRACTED":
                engine.add_extracted_edge(edge["source"], edge["target"], None, None)
            else:
                engine.add_inferred_edge(
                    edge["source"], edge["target"], edge["confidence"], None, None
                )
        return engine

    def parse_formula(self, formula_str: str) -> RegressionSpec | None:
        """Parses R/Statsmodels formula: 'outcome ~ treatment + cov1 + cov2'"""
        if "~" not in formula_str:
            return None

        lhs, rhs = formula_str.split("~", 1)
        outcome = lhs.strip().strip("`").lower()
        rhs_tokens = [t.strip().strip("`").lower() for t in rhs.split("+") if t.strip()]

        if not rhs_tokens:
            return None

        treatment = rhs_tokens[0]
        covariates = rhs_tokens[1:] if len(rhs_tokens) > 1 else []
        return RegressionSpec(
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            provenance_snippet=formula_str,
        )

    def extract_regressions_from_code(self, code_str: str) -> list[RegressionSpec]:
        """Extracts candidate regression specifications from source code."""
        specs: list[RegressionSpec] = []

        for match in self.FORMULA_PATTERN.finditer(code_str):
            formula = match.group("formula")
            parsed = self.parse_formula(formula)
            if parsed:
                specs.append(parsed)

        return specs

    def inspect_regression(self, spec: RegressionSpec) -> CausalViolation | None:
        """Checks a single regression specification against the SCM d-separation criteria."""
        t = spec.treatment
        y = spec.outcome
        z_set = spec.covariates

        all_nodes = {
            node["name"] for node in json.loads(self.engine.export_json())["nodes"]
        }
        if t not in all_nodes or y not in all_nodes:
            return None

        valid_backdoors = self.engine.find_backdoor_adjustment_sets(t, y)

        if not valid_backdoors:
            return None

        minimal_valid = valid_backdoors[0]
        minimal_set = set(minimal_valid)
        used_set = set(z_set)

        # Case 1: Omitted Confounders
        missing_confounders = sorted(list(minimal_set - used_set))
        if missing_confounders:
            return CausalViolation(
                violation_type="OMITTED_CONFOUNDER",
                treatment=t,
                outcome=y,
                covariates_used=z_set,
                problematic_variables=missing_confounders,
                suggested_adjustment_set=minimal_valid,
                message=(
                    f"⚠️ Causal Hazard [Omitted Confounders]: Regression of '{y}' on '{t}' "
                    f"omits essential confounder(s): {missing_confounders}. "
                    f"Recommended adjustment set: {minimal_valid}."
                ),
            )

        # Case 2: Conditioned Colliders / Superfluous Covariates
        extra_covariates = list(used_set - minimal_set)
        if extra_covariates:
            if not self.engine.is_d_separated([t], [y], list(used_set)):
                return CausalViolation(
                    violation_type="CONDITIONED_COLLIDER",
                    treatment=t,
                    outcome=y,
                    covariates_used=z_set,
                    problematic_variables=extra_covariates,
                    suggested_adjustment_set=minimal_valid,
                    message=(
                        f"⚠️ Causal Hazard [Conditioned Collider/Mediator]: Including {extra_covariates} "
                        f"violates causal identification (opens spurious non-causal paths). "
                        f"Use minimal adjustment set instead: {minimal_valid}."
                    ),
                )

        return None

    def intercept_code(self, code_str: str) -> list[CausalViolation]:
        """Scans code text and returns all identified causal regression violations."""
        violations: list[CausalViolation] = []
        regressions = self.extract_regressions_from_code(code_str)
        for reg in regressions:
            v = self.inspect_regression(reg)
            if v:
                violations.append(v)
        return violations


def cli_hook_entrypoint():
    """Command-line hook entry point for Cursor Agent, Claude Code, and pre-commit."""
    parser = argparse.ArgumentParser(
        prog="cig-hook",
        description="Pre-commit and AI Agent Interceptor for causal regression verification.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Positional file paths provided by pre-commit or CLI",
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Explicit path to Python file to check",
    )
    parser.add_argument(
        "--code",
        "-c",
        help="Raw Python code string to check directly",
    )
    parser.add_argument(
        "--graph",
        "-g",
        default=".cig/cig_graph.json",
        help="Path to unified SCM graph JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce non-zero exit code if unadjusted backdoor confounding is detected",
    )

    args = parser.parse_args()

    target_files: list[Path] = []
    if args.file:
        target_files.append(Path(args.file))
    for f_str in args.files:
        p = Path(f_str)
        if p.is_file() and p.suffix == ".py":
            target_files.append(p)

    if args.code:
        code_to_check = args.code
    elif not target_files and not sys.stdin.isatty():
        code_to_check = sys.stdin.read()
    else:
        code_to_check = None

    interceptor = IDEInterceptor(graph_json_path=args.graph)
    has_violations = False

    if code_to_check:
        violations = interceptor.intercept_code(code_to_check)
        if violations:
            has_violations = True
            sys.stderr.write("\n🚨 [CIG CAUSAL INTERCEPTOR WARNING]\n")
            for idx, v in enumerate(violations, 1):
                sys.stderr.write(f"[{idx}] {v.message}\n")
                sys.stderr.write(
                    f"    Suggested valid adjustment set: {v.suggested_adjustment_set}\n\n"
                )

    for py_file in target_files:
        try:
            content = py_file.read_text(encoding="utf-8")
            violations = interceptor.intercept_code(content)
            if violations:
                has_violations = True
                sys.stderr.write(f"\n🚨 [CIG CAUSAL INTERCEPTOR] {py_file}:\n")
                for idx, v in enumerate(violations, 1):
                    sys.stderr.write(f"[{idx}] {v.message}\n")
                    sys.stderr.write(
                        f"    Suggested valid adjustment set: {v.suggested_adjustment_set}\n\n"
                    )
        except Exception:
            pass

    if has_violations and args.strict:
        sys.exit(1)

    sys.exit(0)


def main():
    cli_hook_entrypoint()


if __name__ == "__main__":
    main()
