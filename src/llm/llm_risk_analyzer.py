"""LLM-powered construction project risk analyzer."""

from typing import Any

from src.llm.groq_client import GroqClient
from src.llm.prompt_builder import RiskPromptBuilder
from src.llm.risk_tools import ProjectRiskTool


class LLMRiskAnalyzer:
    """Generate AI-powered risk analysis for construction projects."""

    def __init__(
        self,
        risk_tool: ProjectRiskTool | None = None,
        prompt_builder: RiskPromptBuilder | None = None,
        groq_client: GroqClient | None = None,
    ) -> None:
        """Initialize the LLM risk analyzer."""
        self.risk_tool = risk_tool or ProjectRiskTool()
        self.prompt_builder = prompt_builder or RiskPromptBuilder()
        self.groq_client = groq_client or GroqClient()

    def analyze_project(self, project_id: str) -> dict[str, Any]:
        """Generate an AI risk analysis for a project."""

        if not project_id.strip():
            raise ValueError("Project ID cannot be empty.")

        project_data = self.risk_tool.get_project_risk(project_id)

        prompt = self.prompt_builder.build_project_prompt(project_data)

        analysis = self.groq_client.generate(prompt)

        return {
            "project_id": project_id,
            "overall_risk_level": project_data["overall_risk_level"],
            "risk_score": project_data["average_risk_score"],
            "average_predicted_delay_days": (
                project_data["average_predicted_delay_days"]
            ),
            "maximum_predicted_delay_days": (
                project_data["maximum_predicted_delay_days"]
            ),
            "analysis": analysis,
        }