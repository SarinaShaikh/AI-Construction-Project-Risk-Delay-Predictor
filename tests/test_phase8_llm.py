"""Tests for Phase 8 LLM and AI report components.

Phase 8 integration notes (branch phase7-integration):

Adapted from the friend's implementation (commit d01fb48) so the whole
file is offline and deterministic:

    - The friend's tests read pre-generated report artifacts
      (data/processed/reports/ai/*_ai_risk_report.json, 100 files) that
      only exist on the friend's machine. Here, a session fixture
      generates a small project_risk_summaries.csv in a tmp directory
      and every artifact-based test writes/reads its own tmp_path
      outputs. No repo data files are required.
    - A stub client replaces the Groq API everywhere: tests must never
      make real network calls and must not require GROQ_API_KEY.
    - One extra test asserts GroqClient fails fast when GROQ_API_KEY is
      absent, so no secret handling can regress silently.

The LLM layer is only an explanation layer: these tests pin the
guarantee that all numerical values flow through unchanged from the
deterministic Phase 7 summaries.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.ai_report_generator import AIReportGenerator
from src.llm.groq_client import GroqClient
from src.llm.llm_risk_analyzer import LLMRiskAnalyzer
from src.llm.prompt_builder import RiskPromptBuilder
from src.llm.risk_tools import ProjectRiskTool

SUMMARY_COLUMNS = [
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
]


@pytest.fixture()
def summaries_csv(tmp_path: Path) -> Path:
    """Write a small deterministic Phase 7 summaries CSV for tests."""
    frame = pd.DataFrame(
        [
            {
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
            },
            {
                "project_id": "P00002",
                "total_activities": 80,
                "high_risk_activities": 4,
                "medium_risk_activities": 16,
                "low_risk_activities": 60,
                "critical_risk_activities": 2,
                "average_risk_score": 0.25,
                "maximum_risk_score": 0.70,
                "average_predicted_delay_days": 9.5,
                "maximum_predicted_delay_days": 22.0,
                "overall_risk_level": "Low",
            },
        ],
        columns=SUMMARY_COLUMNS,
    )
    path = tmp_path / "project_risk_summaries.csv"
    frame.to_csv(path, index=False)
    return path


class StubGroqClient:
    """Deterministic stand-in for GroqClient (no network, no API key)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "AI analysis"


def make_tool(summaries_csv: Path) -> ProjectRiskTool:
    return ProjectRiskTool(summaries_csv)


def test_project_risk_tool_returns_project_data(summaries_csv: Path) -> None:
    """Project risk tool should return data for a valid project."""
    tool = make_tool(summaries_csv)

    result = tool.get_project_risk("P00001")

    assert result["project_id"] == "P00001"
    assert result["total_activities"] == 100
    assert result["overall_risk_level"] == "Medium"


def test_project_risk_tool_rejects_unknown_project(
    summaries_csv: Path,
) -> None:
    """Project risk tool should reject unknown projects."""
    tool = make_tool(summaries_csv)

    with pytest.raises(ValueError, match="was not found"):
        tool.get_project_risk("P99999")


def test_project_risk_tool_rejects_empty_project_id(
    summaries_csv: Path,
) -> None:
    """Project risk tool should reject empty project IDs."""
    tool = make_tool(summaries_csv)

    with pytest.raises(ValueError, match="Project ID cannot be empty"):
        tool.get_project_risk("")


def test_project_risk_tool_rejects_missing_columns(tmp_path: Path) -> None:
    """Project risk tool should reject an incomplete summaries schema."""
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"project_id": "P00001"}]).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        ProjectRiskTool(bad)


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
    # Grounding guard: the prompt must forbid invented values.
    assert "Do not invent project information" in prompt


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


def test_llm_risk_analyzer_can_be_constructed_with_dependencies() -> None:
    """LLM analyzer should support dependency injection (friend's test)."""

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


def test_ai_report_json_structure(tmp_path: Path, summaries_csv: Path) -> None:
    """Generated report should contain the expected grounded fields."""
    analyzer = LLMRiskAnalyzer(
        risk_tool=make_tool(summaries_csv),
        prompt_builder=RiskPromptBuilder(),
        groq_client=StubGroqClient(),
    )
    generator = AIReportGenerator(
        analyzer=analyzer,
        output_dir=tmp_path / "ai",
    )

    path = generator.save_report("P00001")

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
    assert report["overall_risk_level"] == "Medium"
    # Grounding: report numbers equal the fixture Phase 7 values exactly.
    assert report["risk_score"] == 0.30
    assert report["average_predicted_delay_days"] == 15.0
    assert report["maximum_predicted_delay_days"] == 35.0
    assert isinstance(report["analysis"], str)
    assert report["analysis"].strip()


def test_ai_report_values_are_grounded_in_phase7(
    tmp_path: Path,
    summaries_csv: Path,
) -> None:
    """Report numbers must come from Phase 7, never from the LLM."""
    tool = make_tool(summaries_csv)
    phase7_row = tool.get_project_risk("P00002")

    analyzer = LLMRiskAnalyzer(
        risk_tool=tool,
        prompt_builder=RiskPromptBuilder(),
        groq_client=StubGroqClient(),
    )
    generator = AIReportGenerator(
        analyzer=analyzer,
        output_dir=tmp_path / "ai",
    )

    report = generator.generate_report("P00002")

    assert report["risk_score"] == phase7_row["average_risk_score"]
    assert (
        report["average_predicted_delay_days"]
        == phase7_row["average_predicted_delay_days"]
    )
    assert (
        report["maximum_predicted_delay_days"]
        == phase7_row["maximum_predicted_delay_days"]
    )
    assert report["overall_risk_level"] == phase7_row["overall_risk_level"]


def test_groq_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """GroqClient must fail fast without GROQ_API_KEY (no secret risk)."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        GroqClient()


def test_demo_runner_generates_offline_reports(
    tmp_path: Path,
    summaries_csv: Path,
) -> None:
    """The --demo runner path must work with no API key and no network."""
    from scripts.generate_ai_reports import OfflineGroqClient, make_generator

    analyzer = LLMRiskAnalyzer(
        risk_tool=make_tool(summaries_csv),
        prompt_builder=RiskPromptBuilder(),
        groq_client=OfflineGroqClient(
            make_tool(summaries_csv).get_project_risk("P00001")
        ),
    )
    generator = make_generator(
        "P00001",
        summaries_csv,
        tmp_path / "ai",
        demo=True,
    )

    path = generator.save_report("P00001")

    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    assert report["project_id"] == "P00001"
    assert report["risk_score"] == 0.30
    assert "[Offline demo analysis" in report["analysis"]
    # The stub never consulted an external client.
    assert isinstance(analyzer.groq_client, OfflineGroqClient)
