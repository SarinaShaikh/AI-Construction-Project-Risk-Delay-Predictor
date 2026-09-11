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
"""

# Relative imports (branch phase7-integration): resolve correctly whether
# the package is imported as `llm` (src/ on sys.path) or `src.llm`
# (repo root on sys.path, e.g. via pytest pythonpath = ["."] and the
# from src.llm.analyzer import ... style).
from .analyzer import ConstructionRiskAnalyzer, RiskInsight
from .project_summary import ProjectRiskSummarizer, ProjectRiskSummary
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

__all__ = [
    "ConstructionRiskAnalyzer",
    "ConstructionRiskReport",
    "ConstructionRiskReportGenerator",
    "ProjectRiskSummarizer",
    "ProjectRiskSummary",
    "RiskInsight",
    "load_phase5_risk_scores",
    "load_phase7_risk_frame",
    "risk_level_from_score",
    "to_phase7_risk_frame",
]
