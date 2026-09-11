"""
Phase 7: Construction Risk Report Generator

Builds a structured project risk report from the deterministic
risk analysis and project-level risk summary.

This module does not call an external LLM.
"""

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class ConstructionRiskReport:
    """Structured construction risk report."""

    project_id: str
    overall_risk_level: str
    total_activities: int
    high_risk_activities: int
    medium_risk_activities: int
    low_risk_activities: int
    critical_risk_activities: int
    average_risk_score: float
    maximum_risk_score: float
    average_predicted_delay_days: float
    maximum_predicted_delay_days: float
    top_risks: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Convert the report into a dictionary."""
        return {
            "project_id": self.project_id,
            "overall_risk_level": self.overall_risk_level,
            "total_activities": self.total_activities,
            "high_risk_activities": self.high_risk_activities,
            "medium_risk_activities": self.medium_risk_activities,
            "low_risk_activities": self.low_risk_activities,
            "critical_risk_activities": self.critical_risk_activities,
            "average_risk_score": self.average_risk_score,
            "maximum_risk_score": self.maximum_risk_score,
            "average_predicted_delay_days": (
                self.average_predicted_delay_days
            ),
            "maximum_predicted_delay_days": (
                self.maximum_predicted_delay_days
            ),
            "top_risks": self.top_risks,
        }


class ConstructionRiskReportGenerator:
    """Generate project-level construction risk reports."""

    REQUIRED_SUMMARY_COLUMNS: ClassVar[set[str]] = {
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
    }

    REQUIRED_INSIGHT_COLUMNS: ClassVar[set[str]] = {
        "project_id",
        "activity_id",
        "risk_score",
        "risk_level",
        "delay_prediction_days",
        "is_critical",
        "explanation",
        "recommendation",
    }

    def __init__(
        self,
        project_summaries: pd.DataFrame,
        activity_insights: pd.DataFrame,
    ) -> None:
        """Initialize the report generator."""
        self.project_summaries = project_summaries.copy()
        self.activity_insights = activity_insights.copy()

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate summary and activity insight inputs."""
        missing_summary = (
            self.REQUIRED_SUMMARY_COLUMNS
            - set(self.project_summaries.columns)
        )

        if missing_summary:
            raise ValueError(
                "Project summaries are missing required columns: "
                + ", ".join(sorted(missing_summary))
            )

        missing_insights = (
            self.REQUIRED_INSIGHT_COLUMNS
            - set(self.activity_insights.columns)
        )

        if missing_insights:
            raise ValueError(
                "Activity insights are missing required columns: "
                + ", ".join(sorted(missing_insights))
            )

    def generate_report(
        self,
        project_id: str,
        top_n: int = 5,
    ) -> ConstructionRiskReport:
        """Generate a report for one project."""
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero.")

        project_rows = self.project_summaries[
            self.project_summaries["project_id"].astype(str)
            == str(project_id)
        ]

        if project_rows.empty:
            raise ValueError(
                f"No project summary found for project '{project_id}'."
            )

        summary = project_rows.iloc[0]

        project_insights = self.activity_insights[
            self.activity_insights["project_id"].astype(str)
            == str(project_id)
        ]

        if project_insights.empty:
            raise ValueError(
                f"No activity insights found for project '{project_id}'."
            )

        top_risk_rows = (
            project_insights.sort_values(
                by="risk_score",
                ascending=False,
            )
            .head(top_n)
        )

        top_risks = [
            {
                "activity_id": str(row["activity_id"]),
                "risk_score": float(row["risk_score"]),
                "risk_level": str(row["risk_level"]),
                "delay_prediction_days": float(
                    row["delay_prediction_days"]
                ),
                "is_critical": bool(row["is_critical"]),
                "explanation": str(row["explanation"]),
                "recommendation": str(row["recommendation"]),
            }
            for _, row in top_risk_rows.iterrows()
        ]

        return ConstructionRiskReport(
            project_id=str(summary["project_id"]),
            overall_risk_level=str(summary["overall_risk_level"]),
            total_activities=int(summary["total_activities"]),
            high_risk_activities=int(summary["high_risk_activities"]),
            medium_risk_activities=int(summary["medium_risk_activities"]),
            low_risk_activities=int(summary["low_risk_activities"]),
            critical_risk_activities=int(
                summary["critical_risk_activities"]
            ),
            average_risk_score=float(summary["average_risk_score"]),
            maximum_risk_score=float(summary["maximum_risk_score"]),
            average_predicted_delay_days=float(
                summary["average_predicted_delay_days"]
            ),
            maximum_predicted_delay_days=float(
                summary["maximum_predicted_delay_days"]
            ),
            top_risks=top_risks,
        )