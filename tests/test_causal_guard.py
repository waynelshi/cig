import pandas as pd
import pytest
from cig import PyCausalEngine
from cig.guard import CausalViolationError, causal_guard


@pytest.fixture
def sample_dag():
    """
    Builds classic M-bias and collider structure:
      treatment -> collider
      outcome   -> collider
      treatment -> mediator -> outcome
    """
    engine = PyCausalEngine()
    engine.add_extracted_edge("treatment", "collider", "etl.sql", None)
    engine.add_extracted_edge("outcome", "collider", "etl.sql", None)
    engine.add_extracted_edge("treatment", "mediator", "etl.sql", None)
    engine.add_extracted_edge("mediator", "outcome", "etl.sql", None)
    return engine


def test_causal_guard_blocks_collider_conditioning(sample_dag):
    @causal_guard(
        treatment="treatment",
        outcome="outcome",
        engine=sample_dag,
        forbid_colliders=True,
    )
    def prepare_features():
        return pd.DataFrame(
            {
                "treatment": [1, 0, 1],
                "collider": [1, 1, 0],
                "feature_valid": [10.5, 20.1, 15.3],
            }
        )

    with pytest.raises(CausalViolationError, match="Collider Bias Detected"):
        prepare_features()


def test_causal_guard_passes_when_clean(sample_dag):
    @causal_guard(
        treatment="treatment",
        outcome="outcome",
        engine=sample_dag,
        forbid_colliders=True,
    )
    def prepare_features():
        return pd.DataFrame(
            {
                "treatment": [1, 0, 1],
                "feature_valid": [10.5, 20.1, 15.3],
            }
        )

    df = prepare_features()
    assert "feature_valid" in df.columns
    assert "collider" not in df.columns


def test_causal_guard_prunes_violating_columns(sample_dag):
    @causal_guard(
        treatment="treatment",
        outcome="outcome",
        engine=sample_dag,
        forbid_colliders=True,
        action="prune",
    )
    def prepare_features():
        return pd.DataFrame(
            {
                "treatment": [1, 0],
                "collider": [0, 1],
                "feature_valid": [5.0, 9.0],
            }
        )

    with pytest.warns(UserWarning, match=r"\[CIG @causal_guard Prune\]"):
        cleaned_df = prepare_features()

    assert "collider" not in cleaned_df.columns
    assert "feature_valid" in cleaned_df.columns


def test_causal_guard_detects_mediator_leakage(sample_dag):
    @causal_guard(
        treatment="treatment",
        outcome="outcome",
        engine=sample_dag,
        forbid_mediators=True,
        action="raise",
    )
    def prepare_features():
        return pd.DataFrame(
            {
                "treatment": [1, 0],
                "mediator": [2.5, 3.1],
            }
        )

    with pytest.raises(CausalViolationError, match="Mediator Blockade Detected"):
        prepare_features()
