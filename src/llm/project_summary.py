"""
Phase 7: Project-Level Construction Risk Summary

Aggregates activity-level construction risk into a
more meaningful project-level risk assessment.

The project risk level considers:
- Average activity risk
- Percentage of high-risk activities
- Percentage of critical high-risk activities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class ProjectRiskSummary:
    """Project-level construction risk summary."""

    project_id: str
    total_activities: int
    high_risk_activities: int
    medium_risk_activities: int
    low_risk_activities: int
    critical_risk_activities: int
    average_risk_score: float
    maximum_risk_score: float
    average_predicted_delay_days: float
    maximum_predicted_delay_days: float
    overall_risk_level: str

    def to_dict(self) -> dict[str, object]:
        """Convert the summary into a dictionary."""
        return {
            "project_id": self.project_id,
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
            "overall_risk_level": self.overall_risk_level,
        }


class ProjectRiskSummarizer:
    """Generate project-level construction risk summaries."""

    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "project_id",
        "risk_score",
        "risk_level",
        "delay_prediction_days",
        "is_critical",
    }

    def __init__(self, activity_insights: pd.DataFrame) -> None:
        """Initialize the project risk summarizer."""
        self.insights = activity_insights.copy()
        self.activity_insights = self.insights
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate the activity insight DataFrame."""
        missing = self.REQUIRED_COLUMNS - set(
            self.activity_insights.columns
        )

        if missing:
            raise ValueError(
                "Activity insights are missing required columns: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _calculate_overall_risk(
        total_activities: int,
        high_risk_activities: int,
        critical_risk_activities: int,
        average_risk_score: float,
    ) -> str:
        """Calculate a project-level risk classification."""
        if total_activities <= 0:
            return "Low"

        high_risk_ratio = (
            high_risk_activities / total_activities
        )

        critical_high_risk_ratio = (
            critical_risk_activities / total_activities
        )
        if high_risk_activities == 0:
          return "Low"

        if (
            average_risk_score >= 0.35
            or high_risk_ratio >= 0.20
            or critical_high_risk_ratio >= 0.15
        ):
            return "High"

        if (
            average_risk_score >= 0.30
            or high_risk_ratio >= 0.15
            or critical_high_risk_ratio >= 0.10
        ):
            return "Medium"

        return "Low"

    def summarize_project(
        self,
        project_id: str,
    ) -> ProjectRiskSummary:
        """Generate a risk summary for one project."""
        project_data = self.activity_insights[
            self.activity_insights["project_id"].astype(str)
            == str(project_id)
        ]

        if project_data.empty:
            raise ValueError(
                f"No risk insights found for project '{project_id}'."
            )

        total_activities = len(project_data)

        high_risk_activities = int(
            (project_data["risk_level"] == "High").sum()
        )

        medium_risk_activities = int(
            (project_data["risk_level"] == "Medium").sum()
        )

        low_risk_activities = int(
            (project_data["risk_level"] == "Low").sum()
        )

        critical_risk_activities = int(
            (
                (project_data["risk_level"] == "High")
                & project_data["is_critical"].astype(bool)
            ).sum()
        )

        average_risk_score = float(
            project_data["risk_score"].mean()
        )

        maximum_risk_score = float(
            project_data["risk_score"].max()
        )

        average_predicted_delay_days = float(
            project_data["delay_prediction_days"].mean()
        )

        maximum_predicted_delay_days = float(
            project_data["delay_prediction_days"].max()
        )

        overall_risk_level = self._calculate_overall_risk(
            total_activities=total_activities,
            high_risk_activities=high_risk_activities,
            critical_risk_activities=critical_risk_activities,
            average_risk_score=average_risk_score,
        )

        return ProjectRiskSummary(
            project_id=str(project_id),
            total_activities=total_activities,
            high_risk_activities=high_risk_activities,
            medium_risk_activities=medium_risk_activities,
            low_risk_activities=low_risk_activities,
            critical_risk_activities=critical_risk_activities,
            average_risk_score=average_risk_score,
            maximum_risk_score=maximum_risk_score,
            average_predicted_delay_days=(
                average_predicted_delay_days
            ),
            maximum_predicted_delay_days=(
                maximum_predicted_delay_days
            ),
            overall_risk_level=overall_risk_level,
        )

    def summarize_all(self) -> pd.DataFrame:
        """Generate summaries for all projects."""
        project_ids = (
            self.activity_insights["project_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        summaries = [
            self.summarize_project(project_id).to_dict()
            for project_id in project_ids
        ]

        return pd.DataFrame(summaries)