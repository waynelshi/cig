"""
benchmarks/benchmark_ab_test.py
Automated A/B evaluation: Naive (Unguarded) vs. CIG-Guarded Causal Estimation.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def generate_campaign_dataset(n: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    # Confounder: Customer Loyalty Score
    loyalty = np.random.normal(50, 15, n)

    # Treatment: Discount Percentage
    discount = 0.003 * loyalty + np.random.normal(0.10, 0.02, n)

    # Mediator: Units Purchased
    # Structural equation: units = 5 + 30 * discount
    units = 5 + 30 * discount + np.random.normal(0, 1, n)

    # Outcome: Net Margin ($)
    # Direct effect of discount is -50. Indirect effect through units is 30 * 12 = +360.
    # Total True Causal Effect: 360 - 50 = +310 per 1.0 (or +3.10 per 0.01 discount).
    margin = (
        200 + 2.5 * loyalty - 50 * discount + 12 * units + np.random.normal(0, 5, n)
    )

    # Collider: VIP Status (conditioned on high discount OR high margin)
    vip_prob = 1 / (1 + np.exp(-(0.1 * margin + 20 * discount - 35)))
    vip_status = np.random.binomial(1, vip_prob, n)

    return pd.DataFrame(
        {
            "customer_id": range(n),
            "loyalty_score": loyalty,
            "discount_pct": discount,
            "units_purchased": units,
            "vip_status": vip_status,
            "net_margin": margin,
        }
    )


def run_unguarded_agent_analysis(df: pd.DataFrame) -> dict:
    """Simulates an unguarded agent: kitchen-sink regression controlling for all available columns."""
    start_time = time.perf_counter()

    # Unguarded agent conditions on mediator (units_purchased) and collider (vip_status)
    formula = "net_margin ~ discount_pct + loyalty_score + units_purchased + vip_status"
    model = smf.ols(formula, data=df).fit()

    estimated_effect = model.params["discount_pct"]
    ci_lower, ci_upper = model.conf_int().loc["discount_pct"]
    elapsed = time.perf_counter() - start_time

    # Context estimate: Full table schema introspection + raw data sample
    simulated_prompt_tokens = 4250

    return {
        "strategy": "Unguarded Agent (Kitchen-Sink)",
        "controls": [
            "loyalty_score",
            "units_purchased (Mediator)",
            "vip_status (Collider)",
        ],
        "estimated_effect": estimated_effect,
        "ci_95": (ci_lower, ci_upper),
        "bias": estimated_effect - 310.0,
        "simulated_tokens": simulated_prompt_tokens,
        "elapsed_sec": elapsed,
    }


def run_cig_guarded_analysis(df: pd.DataFrame) -> dict:
    """Simulates a CIG-guarded agent: minimal sufficient backdoor adjustment set."""
    start_time = time.perf_counter()

    # CIG identifies that {loyalty_score} blocks the backdoor path,
    # while excluding units_purchased (mediator) and vip_status (collider).
    formula = "net_margin ~ discount_pct + loyalty_score"
    model = smf.ols(formula, data=df).fit()

    estimated_effect = model.params["discount_pct"]
    ci_lower, ci_upper = model.conf_int().loc["discount_pct"]
    elapsed = time.perf_counter() - start_time

    # Context estimate: Pre-compiled .cig/cig_graph.json slice + query
    simulated_prompt_tokens = 1150

    return {
        "strategy": "CIG-Guarded Agent",
        "controls": ["loyalty_score (Valid Backdoor Set)"],
        "estimated_effect": estimated_effect,
        "ci_95": (ci_lower, ci_upper),
        "bias": estimated_effect - 310.0,
        "simulated_tokens": simulated_prompt_tokens,
        "elapsed_sec": elapsed,
    }


def main():
    print("======================================================================")
    print("⚡ CIG A/B BENCHMARK: UNGUARDED VS. CIG-GUARDED AI ESTIMATION")
    print("======================================================================\n")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "campaign_data.csv"

    print("[*] Generating synthetic marketing dataset (N=10,000)...")
    df = generate_campaign_dataset()
    df.to_csv(csv_path, index=False)
    print(f"    Saved to: {csv_path}")

    true_effect = 310.0
    print(
        f"[*] True Causal Total Effect d(net_margin)/d(discount_pct) = {true_effect:+.2f}\n"
    )

    res_unguarded = run_unguarded_agent_analysis(df)
    res_cig = run_cig_guarded_analysis(df)

    print("-" * 70)
    print(f"{'Metric':<25} | {'Unguarded Agent':<20} | {'CIG-Guarded Agent':<20}")
    print("-" * 70)
    print(
        f"{'Controls Used':<25} | {'All available':<20} | {'Minimal Backdoor Only':<20}"
    )
    print(
        f"{'Estimated Effect':<25} | {res_unguarded['estimated_effect']:<20.2f} | {res_cig['estimated_effect']:<20.2f}"
    )
    print(
        f"{'Estimation Error / Bias':<25} | {res_unguarded['bias']:<20.2f} | {res_cig['bias']:<20.2f}"
    )
    print(
        f"{'95% Confidence Interval':<25} | [{res_unguarded['ci_95'][0]:.1f}, {res_unguarded['ci_95'][1]:.1f}]       | [{res_cig['ci_95'][0]:.1f}, {res_cig['ci_95'][1]:.1f}]"
    )
    print(
        f"{'Simulated Token Cost':<25} | {res_unguarded['simulated_tokens']:<20} | {res_cig['simulated_tokens']:<20}"
    )

    token_savings_pct = (
        1 - (res_cig["simulated_tokens"] / res_unguarded["simulated_tokens"])
    ) * 100
    print(
        f"{'Token Savings':<25} | {'0% (Baseline)':<20} | {f'{token_savings_pct:.1f}% reduction':<20}"
    )
    print("-" * 70)

    print("\nBusiness Implication:")
    if res_unguarded["estimated_effect"] < 0 and res_cig["estimated_effect"] > 0:
        print(
            "❌ Unguarded agent conclusion: 'Discounts destroy margin. Cancel campaign.'"
        )
        print(
            "✅ CIG-Guarded agent conclusion: 'Discounts expand unit volume sufficiently to grow net margin. Scale campaign.'"
        )
    print("======================================================================\n")


if __name__ == "__main__":
    main()
