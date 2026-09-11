from __future__ import annotations

import pandas as pd
import pytest

from src.llm.project_summary import ProjectRiskSummarizer


def make_insight_data() -> pd.DataFrame:
    """Create sample activity-level risk insights."""
    return pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "activity_id": "A001",
                "risk_score": 0.85,
                "risk_level": "High",
                "delay_prediction_days": 12.0,
                "is_critical": True,
            },
            {
                "project_id": "P00001",
                "activity_id": "A002",
                "risk_score": 0.55,
                "risk_level": "Medium",
                "delay_prediction_days": 6.0,
                "is_critical": False,
            },
            {
                "project_id": "P00001",
                "activity_id": "A003",
                "risk_score": 0.20,
                "risk_level": "Low",
                "delay_prediction_days": 1.0,
                "is_critical": False,
            },
            {
                "project_id": "P00002",
                "activity_id": "A004",
                "risk_score": 0.30,
                "risk_level": "Low",
                "delay_prediction_days": 2.0,
                "is_critical": False,
            },
        ]
    )


def test_summarizer_initializes() -> None:
    """Summarizer should accept valid activity insights."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    assert len(summarizer.insights) == 4


def test_summarizer_rejects_missing_columns() -> None:
    """Summarizer should reject incomplete input."""
    data = pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "risk_score": 0.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        ProjectRiskSummarizer(data)


def test_project_summary_counts_risk_levels() -> None:
    """Project summary should count each risk level correctly."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00001")

    assert summary.total_activities == 3
    assert summary.high_risk_activities == 1
    assert summary.medium_risk_activities == 1
    assert summary.low_risk_activities == 1


def test_project_summary_counts_critical_activities() -> None:
    """Project summary should count critical activities."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00001")

    assert summary.critical_risk_activities == 1


def test_project_summary_calculates_average_risk() -> None:
    """Average risk score should be calculated correctly."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00001")

    assert summary.average_risk_score == pytest.approx(
        (0.85 + 0.55 + 0.20) / 3
    )


def test_project_summary_calculates_maximum_risk() -> None:
    """Maximum risk score should be identified."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00001")

    assert summary.maximum_risk_score == 0.85


def test_project_summary_calculates_delays() -> None:
    """Average and maximum predicted delays should be calculated."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00001")

    assert summary.average_predicted_delay_days == pytest.approx(
        (12.0 + 6.0 + 1.0) / 3
    )
    assert summary.maximum_predicted_delay_days == 12.0


def test_project_summary_overall_risk_high() -> None:
    """A project with any high-risk activity should be High risk."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00001")

    assert summary.overall_risk_level == "High"


def test_project_summary_overall_risk_low() -> None:
    """A project with only low-risk activities should be Low risk."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00002")

    assert summary.overall_risk_level == "Low"


def test_project_not_found() -> None:
    """Unknown projects should raise a clear error."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    with pytest.raises(ValueError, match="No risk insights found"):
        summarizer.summarize_project("P99999")


def test_summarize_all_projects() -> None:
    """summarize_all should create one row per project."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    results = summarizer.summarize_all()

    assert len(results) == 2
    assert set(results["project_id"]) == {"P00001", "P00002"}


def test_to_dict_contains_expected_fields() -> None:
    """ProjectRiskSummary should convert to a serializable dictionary."""
    data = make_insight_data()

    summarizer = ProjectRiskSummarizer(data)

    summary = summarizer.summarize_project("P00001")
    result = summary.to_dict()

    expected_fields = {
        "project_id",
        "total_activities",
        "high_risk_activities",
        "medium_risk_activities",
        "low_risk_activities",
        "critical_risk_activities",
        "average_risk_score",
        "maximum_risk_score",
        "average_predicted_delay_days",
        "maximum_predicted_delay_days",
        "overall_risk_level",
    }

    assert set(result.keys()) == expected_fields