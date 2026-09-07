from __future__ import annotations

import pandas as pd
from cig import PyCausalEngine
from cig.passes.pass3_unify import SCMUnificationReport, UnifiedEdge
from cig.passes.pass6_active import ActiveInferenceAgent


def test_active_inference_agent_efe_minimization():
    engine = PyCausalEngine()
    engine.add_extracted_edge("discount_pct", "revenue", None, None)
    engine.add_inferred_edge("marketing_tier", "quantity", 0.60, None, None)

    edges = [
        UnifiedEdge(
            source="discount_pct",
            target="revenue",
            edge_type="EXTRACTED",
            confidence=1.0,
            provenance=["mart.sql"],
        ),
        UnifiedEdge(
            source="marketing_tier",
            target="quantity",
            edge_type="INFERRED",
            confidence=0.60,
            provenance=["rules.md"],
        ),
        UnifiedEdge(
            source="ad_spend",
            target="brand_awareness",
            edge_type="AMBIGUOUS",
            confidence=0.50,
            conflict_notes="Contradictory studies found",
        ),
    ]

    report = SCMUnificationReport(
        total_nodes=5,
        total_edges=3,
        extracted_edges=1,
        inferred_edges=1,
        ambiguous_edges=1,
        god_confounders=[],
        edges=edges,
    )

    df = pd.DataFrame(
        {
            "discount_pct": [0.05, 0.10, 0.15, 0.20],
            "revenue": [500.0, 480.0, 460.0, 440.0],
            "marketing_tier": [0, 1, 0, 1],
        }
    )

    agent = ActiveInferenceAgent(engine=engine, report=report)

    # 1. Epistemic targets identified
    targets = agent.score_epistemic_uncertainty()
    assert len(targets) >= 2
    # AMBIGUOUS edge should have highest entropy (1.0)
    assert targets[0].edge_type == "AMBIGUOUS"
    assert targets[0].uncertainty_score == 1.0

    # 2. Plan arbitration
    candidate_policies = {
        "status_quo": {"discount_pct": 0.05},
        "epistemic_probe": {"marketing_tier": 1.0},
    }

    plan = agent.plan(
        candidate_interventions=candidate_policies,
        target_outcome="revenue",
        desired_value=500.0,
        df=df,
    )

    assert len(plan.evaluated_policies) == 2
    assert plan.optimal_policy is not None
    # Ensure policy with minimum EFE is selected
    assert (
        plan.optimal_policy.expected_free_energy
        <= plan.evaluated_policies[1].expected_free_energy
    )
