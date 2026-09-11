"""Tools for retrieving construction project risk information."""

from pathlib import Path
from typing import ClassVar

import pandas as pd


class ProjectRiskTool:
    """Retrieve project-level risk information from Phase 7 outputs."""

    REQUIRED_COLUMNS: ClassVar[set[str]] = {
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

    def __init__(
        self,
        risk_summary_path: str | Path = (
            "data/processed/risk/project_risk_summaries.csv"
        ),
    ) -> None:
        """Load project risk summaries."""
        self.risk_summary_path = Path(risk_summary_path)

        if not self.risk_summary_path.exists():
            raise FileNotFoundError(
                f"Risk summary file not found: {self.risk_summary_path}"
            )

        self.data = pd.read_csv(self.risk_summary_path)

        missing_columns = self.REQUIRED_COLUMNS - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

    def get_project_risk(self, project_id: str) -> dict:
        """Return risk information for a project."""
        if not project_id.strip():
            raise ValueError("Project ID cannot be empty.")

        project = self.data[
            self.data["project_id"].astype(str) == project_id
        ]

        if project.empty:
            raise ValueError(
                f"Project '{project_id}' was not found."
            )

        row = project.iloc[0]

        return {
            "project_id": str(row["project_id"]),
            "total_activities": int(row["total_activities"]),
            "high_risk_activities": int(row["high_risk_activities"]),
            "medium_risk_activities": int(row["medium_risk_activities"]),
            "low_risk_activities": int(row["low_risk_activities"]),
            "critical_risk_activities": int(row["critical_risk_activities"]),
            "average_risk_score": float(row["average_risk_score"]),
            "maximum_risk_score": float(row["maximum_risk_score"]),
            "average_predicted_delay_days": float(
                row["average_predicted_delay_days"]
            ),
            "maximum_predicted_delay_days": float(
                row["maximum_predicted_delay_days"]
            ),
            "overall_risk_level": str(row["overall_risk_level"]),
        }