"""
Tests for src/llm/risk_adapter.py

Verify that the Phase 7 integration adapter:
- Reads the validated Phase 5 output without recalculating anything
- Carries risk_score through UNCHANGED (numerical identity)
- Applies presentation-only renames (predicted_delay_days ->
  delay_prediction_days, computed_critical -> is_critical)
- Derives risk_level with the SAME thresholds as the Phase 7 analyzer
  (High >= 0.70, Medium >= 0.40, Low otherwise)
- Preserves every (project_id, activity_id) key (no invented/dropped rows)
- Rejects missing required columns and duplicate keys
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Match the project's existing test convention (see tests/test_risk_scoring.py):
# insert src/ so top-level package imports (llm.*) work regardless of run order.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.risk_adapter import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    risk_level_from_score,
    to_phase7_risk_frame,
)


def make_phase5_scores() -> pd.DataFrame:
    """Sample DataFrame mirroring Phase 5 risk_scores.csv schema."""
    return pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "activity_id": "P00001_A0001",
                "predicted_delay_days": 12.5,
                "probability_of_event_delay": 0.91,
                "total_float": 0.0,
                "computed_critical": 1,
                "criticality_weight": 1.0,
                "float_impact": 1.0,
                "risk_score": 0.91,
                "ES": 0.0,
                "EF": 10.0,
                "LS": 0.0,
                "LF": 10.0,
                "risk_rank": 1,
                "risk_percentile": 100.0,
            },
            {
                "project_id": "P00001",
                "activity_id": "P00001_A0002",
                "predicted_delay_days": 4.0,
                "probability_of_event_delay": 0.55,
                "total_float": 6.0,
                "computed_critical": 0,
                "criticality_weight": 0.7,
                "float_impact": 0.666667,
                "risk_score": 0.45,
                "ES": 10.0,
                "EF": 20.0,
                "LS": 16.0,
                "LF": 26.0,
                "risk_rank": 2,
                "risk_percentile": 50.0,
            },
            {
                "project_id": "P00002",
                "activity_id": "P00002_A0001",
                "predicted_delay_days": 0.0,
                "probability_of_event_delay": 0.10,
                "total_float": 30.0,
                "computed_critical": 0,
                "criticality_weight": 0.4,
                "float_impact": 0.0,
                "risk_score": 0.0,
                "ES": 0.0,
                "EF": 5.0,
                "LS": 30.0,
                "LF": 35.0,
                "risk_rank": 3,
                "risk_percentile": 16.7,
            },
        ]
    )


def test_risk_score_carried_through_unchanged() -> None:
    """The Phase 5 risk_score must survive the adapter bit-for-bit."""
    phase5 = make_phase5_scores()

    adapted = to_phase7_risk_frame(phase5)

    assert adapted["risk_score"].tolist() == phase5["risk_score"].tolist()
    # No re-scoring: the probability component is untouched too.
    assert (
        adapted["probability_of_event_delay"].tolist()
        == phase5["probability_of_event_delay"].tolist()
    )


def test_presentation_renames_applied() -> None:
    """predicted_delay_days -> delay_prediction_days with values preserved;
    computed_critical kept for the analyzer plus an identical is_critical
    alias."""
    phase5 = make_phase5_scores()

    adapted = to_phase7_risk_frame(phase5)

    assert "delay_prediction_days" in adapted.columns
    assert "predicted_delay_days" not in adapted.columns
    assert "computed_critical" in adapted.columns  # analyzer-required input
    assert "is_critical" in adapted.columns  # alias for downstream consumers
    assert (
        adapted["delay_prediction_days"].tolist()
        == phase5["predicted_delay_days"].tolist()
    )
    assert (
        adapted["is_critical"].tolist() == phase5["computed_critical"].tolist()
    )
    assert (
        adapted["computed_critical"].tolist()
        == phase5["computed_critical"].tolist()
    )


def test_risk_level_uses_analyzer_thresholds() -> None:
    """risk_level must match the analyzer thresholds exactly."""
    assert risk_level_from_score(0.70) == "High"
    assert risk_level_from_score(0.85) == "High"
    assert risk_level_from_score(0.40) == "Medium"
    assert risk_level_from_score(0.55) == "Medium"
    assert risk_level_from_score(0.39) == "Low"
    assert risk_level_from_score(0.0) == "Low"
    assert HIGH_RISK_THRESHOLD == 0.70
    assert MEDIUM_RISK_THRESHOLD == 0.40


def test_adapted_risk_level_matches_analyzer_classification() -> None:
    """Adapter risk_level must agree with the analyzer's own _risk_level."""
    from llm.analyzer import ConstructionRiskAnalyzer

    phase5 = make_phase5_scores()
    adapted = to_phase7_risk_frame(phase5)

    analyzer = ConstructionRiskAnalyzer(adapted)
    insights = analyzer.analyze()

    # The analyzer reuses the adapter-provided level; both must equal the
    # threshold classification of the SAME Phase 5 score.
    for _, row in insights.iterrows():
        score = float(row["risk_score"])
        expected = risk_level_from_score(score)
        assert row["risk_level"] == expected


def test_all_keys_preserved() -> None:
    """Every Phase 5 activity must appear in the adapted frame, once."""
    phase5 = make_phase5_scores()

    adapted = to_phase7_risk_frame(phase5)

    assert len(adapted) == len(phase5)
    original_keys = set(
        zip(phase5["project_id"], phase5["activity_id"], strict=True)
    )
    adapted_keys = set(
        zip(adapted["project_id"], adapted["activity_id"], strict=True)
    )
    assert original_keys == adapted_keys
    assert not adapted.duplicated(["project_id", "activity_id"]).any()


def test_phase5_columns_preserved() -> None:
    """Non-renamed Phase 5 columns survive untouched (traceability)."""
    phase5 = make_phase5_scores()

    adapted = to_phase7_risk_frame(phase5)

    for col in [
        "probability_of_event_delay",
        "criticality_weight",
        "float_impact",
        "total_float",
        "ES",
        "EF",
        "LS",
        "LF",
        "risk_rank",
        "risk_percentile",
    ]:
        assert col in adapted.columns
        assert adapted[col].tolist() == phase5[col].tolist()


def test_missing_required_column_rejected() -> None:
    """Missing Phase 5 columns must fail loudly."""
    phase5 = make_phase5_scores().drop(columns=["total_float"])

    with pytest.raises(ValueError, match="missing required columns"):
        to_phase7_risk_frame(phase5)


def test_duplicate_keys_rejected() -> None:
    """Duplicate (project_id, activity_id) keys must fail loudly."""
    phase5 = make_phase5_scores()
    phase5 = pd.concat([phase5, phase5.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        to_phase7_risk_frame(phase5)


def test_deterministic_output() -> None:
    """Repeated adapter runs must produce identical frames."""
    phase5 = make_phase5_scores()

    first = to_phase7_risk_frame(phase5)
    second = to_phase7_risk_frame(phase5)

    pd.testing.assert_frame_equal(first, second)
