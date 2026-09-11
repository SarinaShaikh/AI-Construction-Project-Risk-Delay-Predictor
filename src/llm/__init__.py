"""Phase 7: Deterministic risk analysis and reporting.

Converts Phase 5 activity-level risk scores into structured,
human-readable construction risk insights, project-level summaries,
and structured reports.

This package is deterministic:
    - No external LLM/API is used in Phase 7.
    - No ML predictions are invented; all numerical values come from the
      Phase 5 risk output (data/processed/risk_scores.csv).
    - No CPM is recalculated; schedule indicators come from Phase 5's
      Phase 3-derived columns.

Risk score != actual schedule slippage. The Phase 4/5 target is
event-induced/disruption delay (target_event_delay_days), not actual
project schedule delay.

Phase 8 (LLM/AI reporting, branch phase7-integration) lives in this
package too, as a separate explanation layer on top of Phase 7:
    - risk_tools.ProjectRiskTool reads the deterministic Phase 7 output
      data/processed/risk/project_risk_summaries.csv (no recalculation).
    - prompt_builder.RiskPromptBuilder renders those numbers into a
      structured prompt; the LLM must not invent values.
    - groq_client.GroqClient is the ONLY component that talks to an
      external API, and only when GROQ_API_KEY is configured.
Phase 7 modules above remain fully deterministic and untouched.
"""

# Relative imports (branch phase7-integration): resolve correctly whether
# the package is imported as `llm` (src/ on sys.path) or `src.llm`
# (repo root on sys.path, e.g. via pytest pythonpath = ["."] and the
# from src.llm.analyzer import ... style).
# Phase 8 (LLM/AI reporting) re-exports. These add new names only; the
# deterministic Phase 7 exports above are unchanged.
from .ai_report_generator import AIReportGenerator
from .analyzer import ConstructionRiskAnalyzer, RiskInsight
from .groq_client import GroqClient
from .llm_risk_analyzer import LLMRiskAnalyzer
from .project_summary import ProjectRiskSummarizer, ProjectRiskSummary
from .prompt_builder import RiskPromptBuilder
from .report_generator import (
    ConstructionRiskReport,
    ConstructionRiskReportGenerator,
)
from .risk_adapter import (
    load_phase5_risk_scores,
    load_phase7_risk_frame,
    risk_level_from_score,
    to_phase7_risk_frame,
)
from .risk_tools import ProjectRiskTool

__all__ = [
    "AIReportGenerator",
    "ConstructionRiskAnalyzer",
    "ConstructionRiskReport",
    "ConstructionRiskReportGenerator",
    "GroqClient",
    "LLMRiskAnalyzer",
    "ProjectRiskSummarizer",
    "ProjectRiskSummary",
    "ProjectRiskTool",
    "RiskInsight",
    "RiskPromptBuilder",
    "load_phase5_risk_scores",
    "load_phase7_risk_frame",
    "risk_level_from_score",
    "to_phase7_risk_frame",
]
