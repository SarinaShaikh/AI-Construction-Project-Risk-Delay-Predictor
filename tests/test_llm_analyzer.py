from __future__ import annotations

import pandas as pd
import pytest

from src.llm.analyzer import ConstructionRiskAnalyzer


def make_risk_data() -> pd.DataFrame:
    """Create sample Phase 5 risk results for testing."""
    return pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "activity_id": "A001",
                "risk_score": 0.85,
                "delay_prediction_days": 12.0,
                "total_float": 0.0,
                "computed_critical": True,
            },
            {
                "project_id": "P00001",
                "activity_id": "A002",
                "risk_score": 0.55,
                "delay_prediction_days": 6.0,
                "total_float": 3.0,
                "computed_critical": False,
            },
            {
                "project_id": "P00001",
                "activity_id": "A003",
                "risk_score": 0.20,
                "delay_prediction_days": 1.0,
                "total_float": 20.0,
                "computed_critical": False,
            },
        ]
    )


def test_analyzer_initializes() -> None:
    """Analyzer should accept valid Phase 5 results."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    assert len(analyzer.results) == 3


def test_analyzer_rejects_missing_columns() -> None:
    """Analyzer should reject incomplete risk results."""
    data = pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "activity_id": "A001",
                "risk_score": 0.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        ConstructionRiskAnalyzer(data)


def test_risk_level_high() -> None:
    """Scores >= 0.70 should be High risk."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    insight = analyzer.analyze_activity(data.iloc[0])

    assert insight.risk_level == "High"


def test_risk_level_medium() -> None:
    """Scores >= 0.40 and < 0.70 should be Medium risk."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    insight = analyzer.analyze_activity(data.iloc[1])

    assert insight.risk_level == "Medium"


def test_risk_level_low() -> None:
    """Scores below 0.40 should be Low risk."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    insight = analyzer.analyze_activity(data.iloc[2])

    assert insight.risk_level == "Low"


def test_critical_activity_explanation() -> None:
    """Critical activities should mention critical-path risk."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    insight = analyzer.analyze_activity(data.iloc[0])

    assert "critical path" in insight.explanation


def test_zero_float_explanation() -> None:
    """Zero-float activities should mention their lack of float."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    insight = analyzer.analyze_activity(data.iloc[0])

    assert "zero or negative float" in insight.explanation


def test_recommendation_generated() -> None:
    """Every analyzed activity should receive a recommendation."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    results = analyzer.analyze()

    assert len(results) == 3
    assert results["recommendation"].notna().all()
    assert (results["recommendation"].str.len() > 0).all()


def test_analyze_returns_all_activities() -> None:
    """Analyzer should generate one insight per activity."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    results = analyzer.analyze()

    assert len(results) == len(data)
    assert set(results["activity_id"]) == {"A001", "A002", "A003"}


def test_top_risks_returns_highest_scores() -> None:
    """top_risks should return activities ordered by risk score."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    results = analyzer.top_risks(2)

    assert len(results) == 2
    assert results.iloc[0]["activity_id"] == "A001"
    assert results.iloc[1]["activity_id"] == "A002"


def test_top_risks_rejects_invalid_n() -> None:
    """top_risks should reject zero or negative values."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    with pytest.raises(ValueError, match="greater than zero"):
        analyzer.top_risks(0)


def test_to_dict_contains_expected_fields() -> None:
    """RiskInsight should convert to a serializable dictionary."""
    data = make_risk_data()

    analyzer = ConstructionRiskAnalyzer(data)

    insight = analyzer.analyze_activity(data.iloc[0])
    result = insight.to_dict()

    expected_fields = {
        "project_id",
        "activity_id",
        "risk_score",
        "risk_level",
        "delay_prediction_days",
        "total_float",
        "is_critical",
        "explanation",
        "recommendation",
    }

    assert set(result.keys()) == expected_fields