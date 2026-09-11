"""
Phase 7: Construction Risk Analysis

Converts Phase 5 activity-level risk scores into structured,
human-readable construction risk insights.

This module is deterministic.

Important:
    It does not invent ML predictions or schedule values.
    All numerical values come directly from the Phase 5 risk output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class RiskInsight:
    """Structured explanation of an activity risk."""

    project_id: str
    activity_id: str
    risk_score: float
    risk_level: str
    delay_prediction_days: float
    total_float: float
    is_critical: bool
    explanation: str
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """Convert the insight into a dictionary."""
        return {
            "project_id": self.project_id,
            "activity_id": self.activity_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "delay_prediction_days": self.delay_prediction_days,
            "total_float": self.total_float,
            "is_critical": self.is_critical,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
        }


class ConstructionRiskAnalyzer:
    """Generate structured explanations from Phase 5 risk results."""

    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "project_id",
        "activity_id",
        "risk_score",
        "delay_prediction_days",
        "total_float",
        "computed_critical",
    }

    def __init__(
        self,
        risk_results: pd.DataFrame | str | Path,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            risk_results:
                DataFrame or path to the Phase 5 risk_scores.csv file.
        """
        if isinstance(risk_results, (str, Path)):
            self.results = pd.read_csv(risk_results)
        else:
            self.results = risk_results.copy()

        self._validate_input()

    def _validate_input(self) -> None:
        """Validate required Phase 5 columns."""
        missing = self.REQUIRED_COLUMNS - set(self.results.columns)

        if missing:
            raise ValueError(
                "Risk results are missing required columns: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _risk_level(risk_score: float) -> str:
        """Convert the Phase 5 risk score into a risk level."""
        if risk_score >= 0.70:
            return "High"

        if risk_score >= 0.40:
            return "Medium"

        return "Low"

    @staticmethod
    def _build_explanation(
        delay_days: float,
        total_float: float,
        is_critical: bool,
    ) -> str:
        """Build a deterministic explanation from risk factors."""
        factors: list[str] = []

        if delay_days > 0:
            factors.append(
                f"the predicted delay is {delay_days:.2f} days"
            )

        if total_float <= 0:
            factors.append("the activity has zero or negative float")
        elif total_float < 5:
            factors.append(
                f"the activity has only {total_float:.2f} days of float"
            )

        if is_critical:
            factors.append("the activity is on the critical path")

        if not factors:
            return (
                "The activity has limited schedule risk based on the "
                "available ML prediction and CPM indicators."
            )

        if len(factors) == 1:
            return f"Risk is driven by {factors[0]}."

        return "Risk is driven by " + ", ".join(factors[:-1]) + (
            f", and {factors[-1]}."
        )

    @staticmethod
    def _build_recommendation(
        delay_days: float,
        total_float: float,
        is_critical: bool,
        risk_level: str,
    ) -> str:
        """Build a deterministic mitigation recommendation."""
        if risk_level == "High":
            if is_critical or total_float <= 0:
                return (
                    "Prioritize immediate mitigation, monitor the activity "
                    "daily, and protect resources and dependencies on the "
                    "critical path."
                )

            return (
                "Prioritize immediate mitigation and closely monitor "
                "schedule progress and available float."
            )

        if risk_level == "Medium":
            if total_float < delay_days:
                return (
                    "Review the activity schedule, protect available float, "
                    "and prepare a mitigation plan before the predicted "
                    "delay consumes the remaining float."
                )

            return (
                "Monitor the activity closely and prepare contingency "
                "actions if the predicted delay increases."
            )

        return (
            "Continue normal monitoring and review the activity if its "
            "predicted delay or schedule conditions worsen."
        )

    def analyze_activity(
        self,
        row: pd.Series,
    ) -> RiskInsight:
        """Generate a structured risk insight for one activity."""
        project_id = str(row["project_id"])
        activity_id = str(row["activity_id"])

        risk_score = float(row["risk_score"])
        delay_days = float(row["delay_prediction_days"])
        total_float = float(row["total_float"])

        is_critical = bool(row["computed_critical"])

        risk_level = self._risk_level(risk_score)

        explanation = self._build_explanation(
            delay_days=delay_days,
            total_float=total_float,
            is_critical=is_critical,
        )

        recommendation = self._build_recommendation(
            delay_days=delay_days,
            total_float=total_float,
            is_critical=is_critical,
            risk_level=risk_level,
        )

        return RiskInsight(
            project_id=project_id,
            activity_id=activity_id,
            risk_score=risk_score,
            risk_level=risk_level,
            delay_prediction_days=delay_days,
            total_float=total_float,
            is_critical=is_critical,
            explanation=explanation,
            recommendation=recommendation,
        )

    def analyze(self) -> pd.DataFrame:
        """Generate insights for all activities."""
        insights = [
            self.analyze_activity(row)
            for _, row in self.results.iterrows()
        ]

        return pd.DataFrame(
            [insight.to_dict() for insight in insights]
        )

    def top_risks(self, n: int = 10) -> pd.DataFrame:
        """Return the highest-risk activities."""
        if n <= 0:
            raise ValueError("n must be greater than zero.")

        analyzed = self.analyze()

        return analyzed.sort_values(
            by="risk_score",
            ascending=False,
        ).head(n).reset_index(drop=True)