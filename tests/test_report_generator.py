from __future__ import annotations

import pandas as pd
import pytest

from src.llm.report_generator import ConstructionRiskReportGenerator


def make_summary_data() -> pd.DataFrame:
    """Create sample project summary data."""
    return pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "overall_risk_level": "High",
                "total_activities": 3,
                "high_risk_activities": 1,
                "medium_risk_activities": 1,
                "low_risk_activities": 1,
                "critical_risk_activities": 1,
                "average_risk_score": 0.53,
                "maximum_risk_score": 0.85,
                "average_predicted_delay_days": 6.33,
                "maximum_predicted_delay_days": 12.0,
            }
        ]
    )


def make_insight_data() -> pd.DataFrame:
    """Create sample activity risk insights."""
    return pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "activity_id": "A001",
                "risk_score": 0.85,
                "risk_level": "High",
                "delay_prediction_days": 12.0,
                "is_critical": True,
                "explanation": "Critical activity with high predicted delay.",
                "recommendation": "Prioritize immediate mitigation.",
            },
            {
                "project_id": "P00001",
                "activity_id": "A002",
                "risk_score": 0.55,
                "risk_level": "Medium",
                "delay_prediction_days": 6.0,
                "is_critical": False,
                "explanation": "Limited float with predicted delay.",
                "recommendation": "Monitor the activity closely.",
            },
            {
                "project_id": "P00001",
                "activity_id": "A003",
                "risk_score": 0.20,
                "risk_level": "Low",
                "delay_prediction_days": 1.0,
                "is_critical": False,
                "explanation": "Low schedule risk.",
                "recommendation": "Continue normal monitoring.",
            },
        ]
    )


def test_report_generator_initializes() -> None:
    """Report generator should accept valid inputs."""
    generator = ConstructionRiskReportGenerator(
        make_summary_data(),
        make_insight_data(),
    )

    assert len(generator.project_summaries) == 1
    assert len(generator.activity_insights) == 3


def test_missing_summary_columns_rejected() -> None:
    """Missing project summary columns should raise an error."""
    summary = pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "overall_risk_level": "High",
            }
        ]
    )

    with pytest.raises(ValueError, match="Project summaries"):
        ConstructionRiskReportGenerator(
            summary,
            make_insight_data(),
        )


def test_missing_insight_columns_rejected() -> None:
    """Missing activity insight columns should raise an error."""
    insights = pd.DataFrame(
        [
            {
                "project_id": "P00001",
                "activity_id": "A001",
                "risk_score": 0.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="Activity insights"):
        ConstructionRiskReportGenerator(
            make_summary_data(),
            insights,
        )


def test_report_generation() -> None:
    """A complete project risk report should be generated."""
    generator = ConstructionRiskReportGenerator(
        make_summary_data(),
        make_insight_data(),
    )

    report = generator.generate_report("P00001")

    assert report.project_id == "P00001"
    assert report.overall_risk_level == "High"
    assert report.total_activities == 3
    assert report.high_risk_activities == 1
    assert report.medium_risk_activities == 1
    assert report.low_risk_activities == 1


def test_report_contains_top_risks() -> None:
    """Report should contain the highest-risk activities."""
    generator = ConstructionRiskReportGenerator(
        make_summary_data(),
        make_insight_data(),
    )

    report = generator.generate_report("P00001", top_n=2)

    assert len(report.top_risks) == 2
    assert report.top_risks[0]["activity_id"] == "A001"
    assert report.top_risks[1]["activity_id"] == "A002"


def test_report_top_risk_fields() -> None:
    """Each top risk should contain required information."""
    generator = ConstructionRiskReportGenerator(
        make_summary_data(),
        make_insight_data(),
    )

    report = generator.generate_report("P00001")

    risk = report.top_risks[0]

    expected_fields = {
        "activity_id",
        "risk_score",
        "risk_level",
        "delay_prediction_days",
        "is_critical",
        "explanation",
        "recommendation",
    }

    assert set(risk.keys()) == expected_fields


def test_report_rejects_invalid_top_n() -> None:
    """top_n must be greater than zero."""
    generator = ConstructionRiskReportGenerator(
        make_summary_data(),
        make_insight_data(),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        generator.generate_report("P00001", top_n=0)


def test_project_not_found() -> None:
    """Unknown project IDs should raise an error."""
    generator = ConstructionRiskReportGenerator(
        make_summary_data(),
        make_insight_data(),
    )

    with pytest.raises(ValueError, match="No project summary found"):
        generator.generate_report("P99999")


def test_activity_insights_not_found() -> None:
    """A project without activity insights should raise an error."""
    summary = pd.DataFrame(
        [
            {
                "project_id": "P00002",
                "overall_risk_level": "Low",
                "total_activities": 1,
                "high_risk_activities": 0,
                "medium_risk_activities": 0,
                "low_risk_activities": 1,
                "critical_risk_activities": 0,
                "average_risk_score": 0.2,
                "maximum_risk_score": 0.2,
                "average_predicted_delay_days": 1.0,
                "maximum_predicted_delay_days": 1.0,
            }
        ]
    )

    generator = ConstructionRiskReportGenerator(
        summary,
        make_insight_data(),
    )

    with pytest.raises(ValueError, match="No activity insights found"):
        generator.generate_report("P00002")


def test_report_to_dict() -> None:
    """Report should convert to a serializable dictionary."""
    generator = ConstructionRiskReportGenerator(
        make_summary_data(),
        make_insight_data(),
    )

    report = generator.generate_report("P00001")
    result = report.to_dict()

    expected_fields = {
        "project_id",
        "overall_risk_level",
        "total_activities",
        "high_risk_activities",
        "medium_risk_activities",
        "low_risk_activities",
        "critical_risk_activities",
        "average_risk_score",
        "maximum_risk_score",
        "average_predicted_delay_days",
        "maximum_predicted_delay_days",
        "top_risks",
    }

    assert set(result.keys()) == expected_fields