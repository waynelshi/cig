"""
cig/benchmarks/mbias.py
M-Bias Benchmark: Testing spurious collider conditioning against frontier LLMs and Rust core.
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


def generate_mbias_dataset(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    u1 = np.random.normal(0, 1, n)
    u2 = np.random.normal(0, 1, n)

    # Treatment depends on latent U1
    treatment = 1.5 * u1 + np.random.normal(0, 1, n)

    # M is a collider between U1 and U2
    m_covariate = 1.2 * u1 + 1.4 * u2 + np.random.normal(0, 0.5, n)

    # Outcome depends on latent U2 and has a true causal effect of +2.0 from treatment
    true_beta = 2.0
    outcome = true_beta * treatment + 1.8 * u2 + np.random.normal(0, 1, n)

    return pd.DataFrame(
        {
            "treatment": treatment,
            "m_covariate": m_covariate,
            "outcome": outcome,
        }
    )


def evaluate_mbias_llm(model_name: str = "gemini-3.6-flash") -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "model": model_name,
            "controls": [],
            "prompt_tokens": 0,
            "total_tokens": 0,
            "latency_sec": 0.0,
            "should_control": False,
            "reasoning": "GEMINI_API_KEY not configured.",
        }

    client = genai.Client(api_key=api_key)
    prompt = """
You are an expert causal inference statistician and econometrician.
We want to estimate the causal effect of `treatment` on `outcome` in an observational dataset.
We observe a third variable: `m_covariate`.

Here is the known background structural information:
- Latent factor U1 causes both `treatment` and `m_covariate`.
- Latent factor U2 causes both `outcome` and `m_covariate`.
- There is no direct causal edge between U1 and U2.
- There are no other unobserved confounders between `treatment` and `outcome`.

Task:
Determine whether we should condition on / control for `m_covariate` in an OLS regression.
Identify the valid minimal adjustment set.

Respond strictly in valid JSON format:
{
  "control_variables": ["m_covariate"] or [],
  "should_control_for_m": true or false,
  "reasoning_summary": "one sentence explanation"
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
        "controls": parsed.get("control_variables", []),
        "should_control": parsed.get("should_control_for_m", False),
        "reasoning": parsed.get("reasoning_summary", ""),
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "latency_sec": latency,
    }


def execute_mbias_benchmark(model_name: str = "gemini-3.6-flash") -> Dict[str, Any]:
    df = generate_mbias_dataset(n=10000)
    true_effect = 2.0

    # 1. Biased OLS (Controlling for M collider)
    fit_biased = smf.ols("outcome ~ treatment + m_covariate", data=df).fit()
    est_biased = float(fit_biased.params["treatment"])

    # 2. Unbiased OLS (Empty adjustment set)
    fit_unbiased = smf.ols("outcome ~ treatment", data=df).fit()
    est_unbiased = float(fit_unbiased.params["treatment"])

    # 3. Rust SCM Traversal
    engine = PyCausalEngine()
    engine.add_extracted_edge("u1", "treatment")
    engine.add_extracted_edge("u1", "m_covariate")
    engine.add_extracted_edge("u2", "outcome")
    engine.add_extracted_edge("u2", "m_covariate")
    engine.add_extracted_edge("treatment", "outcome")

    t0_cig = time.perf_counter()
    valid_sets = engine.find_backdoor_adjustment_sets("treatment", "outcome")
    cig_latency = time.perf_counter() - t0_cig

    minimal_set = min(valid_sets, key=len) if valid_sets else []
    is_m_valid = engine.is_valid_backdoor_set("treatment", "outcome", ["m_covariate"])

    # 4. LLM Evaluation
    llm_res = evaluate_mbias_llm(model_name=model_name)
    llm_used_m = "m_covariate" in llm_res["controls"] or llm_res["should_control"]
    llm_est = est_biased if llm_used_m else est_unbiased

    speedup = llm_res["latency_sec"] / max(1e-7, cig_latency)

    return {
        "true_effect": true_effect,
        "llm": {
            **llm_res,
            "estimated_effect": round(llm_est, 3),
            "bias": round(llm_est - true_effect, 3),
            "induced_bias": llm_used_m,
        },
        "cig": {
            "controls": minimal_set,
            "estimated_effect": round(est_unbiased, 3),
            "bias": round(est_unbiased - true_effect, 3),
            "latency_sec": cig_latency,
            "is_m_valid": is_m_valid,
        },
        "summary": {
            "speedup_factor": round(speedup, 1),
        },
    }
