"""
Phase 7: Adapter from the validated Phase 5 risk output to Phase 7 inputs.

Integration layer only. This module exists because the friend's Phase 7
components (src/llm/analyzer.py) expect a risk DataFrame with the columns

    project_id, activity_id, risk_score, delay_prediction_days,
    total_float, computed_critical, is_critical, risk_level

while the validated Phase 5 engine (src/risk/scoring.py via
scripts/run_phase5_risk.py) writes data/processed/risk_scores.csv with

    project_id, activity_id, predicted_delay_days,
    probability_of_event_delay, total_float, computed_critical,
    criticality_weight, float_impact, risk_score, ES, EF, LS, LF,
    risk_rank, risk_percentile

This adapter therefore:
    1. Reads the EXISTING Phase 5 output (data/processed/risk_scores.csv).
       Phase 5 remains the sole owner of risk scoring; no score is
       recalculated here.
    2. Renames ONLY one presentation column:
           predicted_delay_days -> delay_prediction_days
       and keeps computed_critical (the column the Phase 7 analyzer
       requires) while ALSO adding is_critical as an identical alias for
       downstream/report-facing consumers.
    3. Carries risk_score through UNCHANGED (numerical identity is
       asserted by tests/test_risk_adapter.py and validated by
       scripts/validate_risk_analysis.py).
    4. Derives the deterministic risk_level using the SAME thresholds the
       Phase 7 analyzer itself applies (High >= 0.70, Medium >= 0.40,
       Low otherwise), so downstream classification is consistent by
       construction and remains traceable to Phase 5 scores.
    5. Verifies the (project_id, activity_id) key sets match Phase 5
       exactly, so no activity is invented or silently dropped.

Deterministic risk LEVELS, explanations, recommendations, and reports are
Phase 7 responsibilities. Phase 7 does NOT:
    - invent ML predictions
    - retrain models
    - recalculate CPM
    - recalculate or alter Phase 5 risk scores
    - use an external LLM/API
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Phase 5 remains the owner of risk scoring; this is the file produced by
# scripts/run_phase5_risk.py (do NOT change Phase 5's output path).
DEFAULT_PHASE5_RISK_SCORES = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "risk_scores.csv"
)

# Risk-level thresholds. These MUST mirror ConstructionRiskAnalyzer
# thresholds in src/llm/analyzer.py (High >= 0.70, Medium >= 0.40).
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

# Deterministic, score-level guidance notes (Phase 7 responsibility).
# Phase 5 produces no textual notes; these are derived ONLY from the
# risk_level that is itself derived from the Phase 5 risk_score.
_LEVEL_NOTES = {
    "High": "Immediate mitigation recommended.",
    "Medium": "Mitigation plan recommended.",
    "Low": "Routine monitoring.",
}

# Columns Phase 7 expects; anything extra in Phase 5's file is preserved.
REQUIRED_PHASE5_COLUMNS = {
    "project_id",
    "activity_id",
    "risk_score",
    "predicted_delay_days",
    "total_float",
    "computed_critical",
}


def risk_level_from_score(risk_score: float) -> str:
    """Map a Phase 5 risk_score to a deterministic Phase 7 risk level.

    Thresholds mirror src/llm/analyzer.py::ConstructionRiskAnalyzer
    (_risk_level): High >= 0.70, Medium >= 0.40, otherwise Low.
    """
    if risk_score >= HIGH_RISK_THRESHOLD:
        return "High"
    if risk_score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def load_phase5_risk_scores(path: Path | str | None = None) -> pd.DataFrame:
    """Load the validated Phase 5 risk output (data/processed/risk_scores.csv).

    Phase 5 owns risk scoring; this only reads its output.
    """
    file_path = Path(path) if path is not None else DEFAULT_PHASE5_RISK_SCORES
    if not file_path.exists():
        raise FileNotFoundError(
            f"Phase 5 risk scores not found: {file_path}. "
            "Run scripts/run_phase5_risk.py first (Phase 5 owns risk scoring)."
        )

    df = pd.read_csv(file_path)

    missing = REQUIRED_PHASE5_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Phase 5 risk_scores.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["project_id"] = df["project_id"].astype(str)
    df["activity_id"] = df["activity_id"].astype(str)

    if df.duplicated(["project_id", "activity_id"]).any():
        raise ValueError(
            "Phase 5 risk_scores.csv contains duplicate "
            "(project_id, activity_id) keys."
        )

    return df


def to_phase7_risk_frame(phase5_scores: pd.DataFrame) -> pd.DataFrame:
    """Adapt a Phase 5 risk-score DataFrame to the Phase 7 input schema.

    Column mapping (presentation only, no recalculation):
        predicted_delay_days -> delay_prediction_days (renamed)
        computed_critical    -> kept (analyzer-required) plus an
                                identical is_critical alias column
        risk_score           -> risk_score (carried through unchanged)

    Adds:
        risk_level: deterministic classification of the Phase 5 score
            using the same thresholds as the Phase 7 analyzer.
        notes: deterministic score-level guidance derived from risk_level.

    All other Phase 5 columns (probability_of_event_delay,
    criticality_weight, float_impact, ES/EF/LS/LF, risk_rank,
    risk_percentile, ...) are preserved untouched.
    """
    df = phase5_scores.copy()

    missing = REQUIRED_PHASE5_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Phase 5 risk scores are missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["project_id"] = df["project_id"].astype(str)
    df["activity_id"] = df["activity_id"].astype(str)

    if df.duplicated(["project_id", "activity_id"]).any():
        raise ValueError(
            "Phase 5 risk scores contain duplicate (project_id, activity_id) keys."
        )

    # Presentation rename only. The VALUES are Phase 5's, unchanged.
    df = df.rename(columns={"predicted_delay_days": "delay_prediction_days"})
    # computed_critical is kept (the Phase 7 analyzer's required input
    # column); is_critical is added as an identical alias for
    # downstream/report-facing consumers.
    df["is_critical"] = df["computed_critical"]

    # Deterministic classification of the Phase 5 score (no re-scoring).
    df["risk_level"] = df["risk_score"].map(risk_level_from_score)
    df["notes"] = df["risk_level"].map(_LEVEL_NOTES)

    return df


def load_phase7_risk_frame(path: Path | str | None = None) -> pd.DataFrame:
    """One-step integration: Phase 5 risk_scores.csv -> Phase 7 input frame."""
    return to_phase7_risk_frame(load_phase5_risk_scores(path))


__all__ = [
    "DEFAULT_PHASE5_RISK_SCORES",
    "HIGH_RISK_THRESHOLD",
    "MEDIUM_RISK_THRESHOLD",
    "REQUIRED_PHASE5_COLUMNS",
    "load_phase5_risk_scores",
    "load_phase7_risk_frame",
    "risk_level_from_score",
    "to_phase7_risk_frame",
]
