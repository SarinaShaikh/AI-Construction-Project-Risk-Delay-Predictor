"""Prompt builder for construction project risk analysis."""

from typing import Any


class RiskPromptBuilder:
    """Build structured prompts for LLM-based construction risk analysis."""

    def build_project_prompt(self, project_data: dict[str, Any]) -> str:
        """Build an LLM prompt from project risk data."""

        required_fields = {
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

        missing_fields = required_fields - set(project_data)

        if missing_fields:
            raise ValueError(
                f"Missing required project fields: {sorted(missing_fields)}"
            )

        project_id = project_data["project_id"]
        total_activities = project_data["total_activities"]
        high_risk = project_data["high_risk_activities"]
        medium_risk = project_data["medium_risk_activities"]
        low_risk = project_data["low_risk_activities"]
        critical_risk = project_data["critical_risk_activities"]
        average_risk = project_data["average_risk_score"]
        maximum_risk = project_data["maximum_risk_score"]
        average_delay = project_data["average_predicted_delay_days"]
        maximum_delay = project_data["maximum_predicted_delay_days"]
        overall_risk = project_data["overall_risk_level"]

        return f"""
You are an AI construction project risk analyst.

Analyze the following construction project risk information.

PROJECT INFORMATION
-------------------
Project ID: {project_id}
Total Activities: {total_activities}

Risk Distribution:
- High-risk activities: {high_risk}
- Medium-risk activities: {medium_risk}
- Low-risk activities: {low_risk}

Critical Activities:
- Critical-risk activities: {critical_risk}

Risk Scores:
- Average risk score: {average_risk:.2f}
- Maximum risk score: {maximum_risk:.2f}

Predicted Delay:
- Average predicted delay: {average_delay:.2f} days
- Maximum predicted delay: {maximum_delay:.2f} days

Overall Risk Level:
{overall_risk}

TASK
----
Provide a practical construction risk analysis.

Your response should include:

1. Overall Risk Assessment
   - Explain the overall project risk level.

2. Key Risk Factors
   - Identify the most important risks from the provided data.
   - Pay special attention to high-risk and critical-risk activities.

3. Delay Analysis
   - Explain what the predicted delays indicate.
   - Discuss the possible impact on project completion.

4. Recommended Mitigation Actions
   - Give practical actions that a construction project manager can take.
   - Prioritize actions for critical and high-risk activities.

5. Priority
   - Classify recommended actions as High, Medium, or Low priority.

Do not invent project information that is not provided.
Clearly distinguish between the model predictions and your recommendations.
Keep the analysis concise, professional, and easy for a project manager to understand.
""".strip()