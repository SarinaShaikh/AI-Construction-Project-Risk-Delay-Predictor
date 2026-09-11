"""Generate and save AI-powered construction risk reports."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.llm.llm_risk_analyzer import LLMRiskAnalyzer


class AIReportGenerator:
    """Generate and save AI risk analysis reports."""

    def __init__(
        self,
        analyzer: LLMRiskAnalyzer | None = None,
        output_dir: str | Path = "data/processed/reports/ai",
    ) -> None:
        """Initialize the report generator."""
        self.analyzer = analyzer or LLMRiskAnalyzer()
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _clean_analysis(text: str) -> str:
        """Normalize common encoding artifacts in LLM responses."""

        replacements = {
            "â€“": "-",
            "â€”": "-",
            "â€‘": "-",
            "â€¯": " ",
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "â€¦": "...",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def generate_report(self, project_id: str) -> dict[str, Any]:
        """Generate an AI risk report for a project."""

        if not project_id.strip():
            raise ValueError("Project ID cannot be empty.")

        result = self.analyzer.analyze_project(project_id)

        analysis = self._clean_analysis(
            result["analysis"]
        )

        report = {
            "project_id": result["project_id"],
            "overall_risk_level": result["overall_risk_level"],
            "risk_score": result["risk_score"],
            "average_predicted_delay_days": (
                result["average_predicted_delay_days"]
            ),
            "maximum_predicted_delay_days": (
                result["maximum_predicted_delay_days"]
            ),
            "analysis": analysis,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        return report

    def save_report(
        self,
        project_id: str,
    ) -> Path:
        """Generate and save an AI risk report."""

        report = self.generate_report(project_id)

        output_path = (
            self.output_dir / f"{project_id}_ai_risk_report.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
            )

        return output_path