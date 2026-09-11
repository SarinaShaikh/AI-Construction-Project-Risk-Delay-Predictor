"""Tests for Phase 8 LLM and AI report components."""

import json
from pathlib import Path

import pytest

from src.llm.ai_report_generator import AIReportGenerator
from src.llm.llm_risk_analyzer import LLMRiskAnalyzer
from src.llm.prompt_builder import RiskPromptBuilder
from src.llm.risk_tools import ProjectRiskTool


def test_project_risk_tool_returns_project_data() -> None:
    """Project risk tool should return data for a valid project."""
    tool = ProjectRiskTool()

    result = tool.get_project_risk("P00001")

    assert result["project_id"] == "P00001"
    assert result["total_activities"] > 0
    assert result["overall_risk_level"] in {"Low", "Medium", "High"}


def test_project_risk_tool_rejects_unknown_project() -> None:
    """Project risk tool should reject unknown projects."""
    tool = ProjectRiskTool()

    with pytest.raises(ValueError, match="was not found"):
        tool.get_project_risk("P99999")


def test_project_risk_tool_rejects_empty_project_id() -> None:
    """Project risk tool should reject empty project IDs."""
    tool = ProjectRiskTool()

    with pytest.raises(ValueError, match="Project ID cannot be empty"):
        tool.get_project_risk("")


def test_prompt_builder_creates_project_prompt() -> None:
    """Prompt builder should create a structured prompt."""
    builder = RiskPromptBuilder()

    project_data = {
        "project_id": "P00001",
        "total_activities": 100,
        "high_risk_activities": 10,
        "medium_risk_activities": 20,
        "low_risk_activities": 70,
        "critical_risk_activities": 5,
        "average_risk_score": 0.30,
        "maximum_risk_score": 0.90,
        "average_predicted_delay_days": 15.0,
        "maximum_predicted_delay_days": 35.0,
        "overall_risk_level": "Medium",
    }

    prompt = builder.build_project_prompt(project_data)

    assert "P00001" in prompt
    assert "Total Activities: 100" in prompt
    assert "High-risk activities: 10" in prompt
    assert "Medium" in prompt
    assert "Recommended Mitigation Actions" in prompt


def test_prompt_builder_rejects_missing_fields() -> None:
    """Prompt builder should reject incomplete project data."""
    builder = RiskPromptBuilder()

    with pytest.raises(ValueError, match="Missing required project fields"):
        builder.build_project_prompt({"project_id": "P00001"})


def test_ai_report_generator_clean_analysis() -> None:
    """Report generator should clean common encoding artifacts."""
    generator = AIReportGenerator.__new__(AIReportGenerator)

    text = "Risk â€“ analysis â€™ test â€œexampleâ€"

    cleaned = generator._clean_analysis(text)

    assert "â€“" not in cleaned
    assert "â€™" not in cleaned
    assert "â€œ" not in cleaned


def test_ai_report_json_structure() -> None:
    """Generated P00001 report should contain the expected fields."""
    path = Path(
        "data/processed/reports/ai/P00001_ai_risk_report.json"
    )

    assert path.exists()

    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    required_fields = {
        "project_id",
        "overall_risk_level",
        "risk_score",
        "average_predicted_delay_days",
        "maximum_predicted_delay_days",
        "analysis",
        "generated_at",
    }

    assert required_fields.issubset(report)
    assert report["project_id"] == "P00001"
    assert report["overall_risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(report["analysis"], str)
    assert report["analysis"].strip()


def test_all_ai_reports_exist() -> None:
    """All 100 project AI reports should exist."""
    report_dir = Path("data/processed/reports/ai")

    files = sorted(report_dir.glob("*_ai_risk_report.json"))

    assert len(files) == 100


def test_llm_risk_analyzer_can_be_constructed_with_dependencies() -> None:
    """LLM analyzer should support dependency injection."""

    class FakeRiskTool:
        def get_project_risk(self, project_id: str) -> dict:
            return {
                "project_id": project_id,
                "total_activities": 10,
                "high_risk_activities": 2,
                "medium_risk_activities": 3,
                "low_risk_activities": 5,
                "critical_risk_activities": 1,
                "average_risk_score": 0.4,
                "maximum_risk_score": 0.8,
                "average_predicted_delay_days": 10.0,
                "maximum_predicted_delay_days": 20.0,
                "overall_risk_level": "Medium",
            }

    class FakePromptBuilder:
        def build_project_prompt(self, project_data: dict) -> str:
            return "test prompt"

    class FakeGroqClient:
        def generate(self, prompt: str) -> str:
            return "AI analysis"

    analyzer = LLMRiskAnalyzer(
        risk_tool=FakeRiskTool(),
        prompt_builder=FakePromptBuilder(),
        groq_client=FakeGroqClient(),
    )

    result = analyzer.analyze_project("P00001")

    assert result["project_id"] == "P00001"
    assert result["overall_risk_level"] == "Medium"
    assert result["analysis"] == "AI analysis"