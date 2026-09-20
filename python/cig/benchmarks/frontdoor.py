"""
cig/benchmarks/frontdoor.py
Frontdoor Criterion Benchmark:
Evaluating unobserved confounding identification via mediator frontdoor formulas.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from cig import PyCausalEngine


def generate_frontdoor_dataset(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    u = np.random.normal(0, 1.5, n)
    treatment = 1.2 * u + np.random.normal(0, 1.0, n)
    mediator = 2.5 * treatment + np.random.normal(0, 0.8, n)
    outcome = 1.6 * mediator + 2.0 * u + np.random.normal(0, 1.0, n)

    return pd.DataFrame(
        {
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
        }
    )


def evaluate_frontdoor_llm(model_name: str = "gemini-3.6-flash") -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "model": model_name,
            "is_identifiable": False,
            "criterion": "unknown",
            "mediator": None,
            "reasoning": "GEMINI_API_KEY not configured.",
            "prompt_tokens": 0,
            "total_tokens": 0,
            "latency_sec": 0.0,
        }

    client = genai.Client(api_key=api_key)

    prompt = """
You are a causal inference econometrician analyzing observational data.
Graph structure:
- Unobserved confounder `U` directly causes both `treatment` and `outcome`.
- `treatment` directly causes `mediator`.
- `mediator` directly causes `outcome`.
- There is no direct edge from `treatment` to `outcome` (all effect flows through `mediator`).
- There is no edge between `U` and `mediator`.

We want to estimate the causal effect of `treatment` on `outcome`.

Questions:
1. Is the causal effect identifiable despite the unobserved confounder U?
2. What adjustment / identification strategy should be applied?
3. Should we run standard OLS of `outcome ~ treatment + mediator`?

Respond strictly in JSON format:
{
  "is_identifiable": true or false,
  "criterion": "frontdoor" or "backdoor" or "unidentifiable",
  "mediator_variable": "mediator" or null,
  "naive_controls": ["mediator"] or [],
  "justification": "short 1-2 sentence explanation"
}
"""
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    latency = time.perf_counter() - t0

    usage = resp.usage_metadata
    prompt_tokens = usage.prompt_token_count if usage else 0
    total_tokens = usage.total_token_count if usage else 0
    parsed = json.loads(resp.text)

    return {
        "model": model_name,
        "is_identifiable": parsed.get("is_identifiable", False),
        "criterion": parsed.get("criterion", "unknown"),
        "mediator": parsed.get("mediator_variable"),
        "reasoning": parsed.get("justification", ""),
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "latency_sec": latency,
    }


def execute_frontdoor_benchmark(model_name: str = "gemini-3.6-flash") -> Dict[str, Any]:
    df = generate_frontdoor_dataset(n=10000)
    true_effect = 4.00

    # 1. Confounded Naive OLS (outcome ~ treatment)
    fit_confounded = smf.ols("outcome ~ treatment", data=df).fit()
    est_confounded = float(fit_confounded.params["treatment"])

    # 2. Blocked Mediator OLS (outcome ~ treatment + mediator)
    fit_blocked = smf.ols("outcome ~ treatment + mediator", data=df).fit()
    est_blocked = float(fit_blocked.params["treatment"])

    # 3. Native Rust SCM graph verification
    engine = PyCausalEngine()
    engine.add_extracted_edge("u", "treatment")
    engine.add_extracted_edge("u", "outcome")
    engine.add_extracted_edge("treatment", "mediator")
    engine.add_extracted_edge("mediator", "outcome")

    t0_graph = time.perf_counter()
    cond2_no_backdoor_xm = engine.is_d_separated(["treatment"], ["mediator"], [])
    cond3_backdoor_my_blocked = engine.is_d_separated(
        ["mediator"], ["outcome"], ["treatment"]
    )
    is_frontdoor_valid = cond2_no_backdoor_xm and cond3_backdoor_my_blocked
    graph_latency = time.perf_counter() - t0_graph

    # 4. Numerical estimation via two-stage OLS
    t0_numeric = time.perf_counter()
    model_m = smf.ols("mediator ~ treatment", data=df).fit()
    beta_xm = float(model_m.params["treatment"])

    model_y = smf.ols("outcome ~ mediator + treatment", data=df).fit()
    beta_my = float(model_y.params["mediator"])

    frontdoor_effect = beta_xm * beta_my
    total_latency = graph_latency + (time.perf_counter() - t0_numeric)

    # 5. LLM Evaluation
    llm_res = evaluate_frontdoor_llm(model_name=model_name)
    is_frontdoor = llm_res["criterion"].lower() == "frontdoor"
    llm_est = frontdoor_effect if is_frontdoor else est_confounded

    speedup = llm_res["latency_sec"] / max(1e-7, graph_latency)

    return {
        "true_effect": true_effect,
        "est_confounded": round(est_confounded, 3),
        "est_blocked": round(est_blocked, 3),
        "llm": {
            **llm_res,
            "estimated_effect": round(llm_est, 3),
            "bias": round(llm_est - true_effect, 3),
        },
        "cig": {
            "is_valid": is_frontdoor_valid,
            "strategy": "FRONTDOOR (Pearl Formula)",
            "estimated_effect": round(frontdoor_effect, 3),
            "bias": round(frontdoor_effect - true_effect, 3),
            "graph_latency_sec": graph_latency,
            "total_latency_sec": total_latency,
        },
        "summary": {
            "speedup_factor": round(speedup, 1),
        },
    }
