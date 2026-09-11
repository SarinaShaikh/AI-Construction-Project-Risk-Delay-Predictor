"""
Phase 7: Risk Analysis Validation.

Validates the generated activity-level and project-level
construction risk analysis results.

This script does not modify the risk-scoring logic.
It only reads the generated CSV/JSON files and reports
their distributions and key statistics.

Branch phase7-integration additions:
    - activity_risk_scores.csv is the ADAPTER OUTPUT of the validated
      Phase 5 file (data/processed/risk_scores.csv). Phase 5 remains the
      owner of risk scoring; this script verifies — numerically, per
      activity — that Phase 7 consumed the Phase 5 scores unchanged.
    - Section 4 asserts the numerical identity of risk_score and the
      schedule/ML columns between Phase 5 output and Phase 7 input.
    - project_risk_reports.json lives at <repo>/reports/ on this branch.

Determinism/no-leakage checks:
    - risk_level counts must match the analyzer thresholds applied to
      the Phase 5 scores (traceable classification, no re-scoring).
    - Reports JSON must agree with the summary CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RISK_DIR = PROCESSED_DIR / "risk"

PHASE5_RISK_SCORES_FILE = PROCESSED_DIR / "risk_scores.csv"
ACTIVITY_RISK_FILE = RISK_DIR / "activity_risk_scores.csv"
ACTIVITY_INSIGHTS_FILE = RISK_DIR / "activity_risk_insights.csv"
PROJECT_SUMMARIES_FILE = RISK_DIR / "project_risk_summaries.csv"
PROJECT_REPORTS_FILE = ROOT_DIR / "reports" / "project_risk_reports.json"


def print_section(title: str) -> None:
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_files() -> None:
    """Check that all required output files exist."""
    required_files = [
        ACTIVITY_RISK_FILE,
        ACTIVITY_INSIGHTS_FILE,
        PROJECT_SUMMARIES_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required validation file not found: {file_path}"
            )


def validate_activity_risks() -> pd.DataFrame:
    """Validate activity-level risk scores."""
    print_section("1. ACTIVITY RISK SCORE VALIDATION")

    activity_risks = pd.read_csv(ACTIVITY_RISK_FILE)

    print(f"Activities: {len(activity_risks):,}")

    print("\nRisk score statistics:")
    print(
        activity_risks["risk_score"]
        .describe()
        .round(4)
        .to_string()
    )

    print("\nPredicted delay statistics:")
    print(
        activity_risks["delay_prediction_days"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nRisk score distribution:")

    risk_bins = pd.cut(
        activity_risks["risk_score"],
        bins=[-0.001, 0.40, 0.70, 1.0],
        labels=["Low", "Medium", "High"],
    )

    risk_distribution = risk_bins.value_counts().reindex(
        ["High", "Medium", "Low"],
        fill_value=0,
    )

    for level, count in risk_distribution.items():
        percentage = (count / len(activity_risks)) * 100
        print(
            f"  {level}: {int(count):,} "
            f"({percentage:.2f}%)"
        )

    critical_count = int(
        activity_risks["computed_critical"].astype(bool).sum()
    )

    high_risk_critical = int(
        (
            (activity_risks["risk_score"] >= 0.70)
            & activity_risks["computed_critical"].astype(bool)
        ).sum()
    )

    print(f"\nCritical activities: {critical_count:,}")
    print(
        "High-risk critical activities: "
        f"{high_risk_critical:,}"
    )

    print("\nTop 10 highest-risk activities:")

    columns = [
        "activity_id",
        "project_id",
        "risk_score",
        "delay_prediction_days",
        "total_float",
    ]

    available_columns = [
        column
        for column in columns
        if column in activity_risks.columns
    ]

    print(
        activity_risks.nlargest(
            10,
            "risk_score",
        )[available_columns].to_string(index=False)
    )

    return activity_risks


def validate_phase5_consistency(activity_risks: pd.DataFrame) -> None:
    """Verify Phase 7 inputs are numerically identical to Phase 5 output.

    Phase 5 owns risk scoring. This check proves the adapter neither
    altered scores nor invented/dropped activities.
    """
    print_section("4. PHASE 5 NUMERICAL CONSISTENCY (risk_score identity)")

    if not PHASE5_RISK_SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Phase 5 risk scores not found: {PHASE5_RISK_SCORES_FILE}. "
            "Run scripts/run_phase5_risk.py first."
        )

    phase5 = pd.read_csv(PHASE5_RISK_SCORES_FILE)
    phase5["project_id"] = phase5["project_id"].astype(str)
    phase5["activity_id"] = phase5["activity_id"].astype(str)

    adapted = activity_risks.copy()
    adapted["project_id"] = adapted["project_id"].astype(str)
    adapted["activity_id"] = adapted["activity_id"].astype(str)

    key_cols = ["project_id", "activity_id"]

    p5_keys = set(map(tuple, phase5[key_cols].itertuples(index=False)))
    p7_keys = set(map(tuple, adapted[key_cols].itertuples(index=False)))

    if p5_keys != p7_keys:
        only_p5 = sorted(p5_keys - p7_keys)[:5]
        only_p7 = sorted(p7_keys - p5_keys)[:5]
        raise AssertionError(
            "Phase 7 input keys differ from Phase 5 output. "
            f"only_in_phase5={only_p5}... only_in_phase7={only_p7}..."
        )

    print(f"  Key sets match exactly: {len(p5_keys):,} activities")

    phase5_indexed = phase5.set_index(key_cols).sort_index()
    adapted_indexed = adapted.set_index(key_cols).sort_index()

    # Identity checks: the underlying score and schedule/ML values must
    # be EXACTLY the Phase 5 values (Phase 7 must not re-score).
    checks = {
        "risk_score": "risk_score",
        # Adapter rename: Phase 5 predicted_delay_days was carried
        # through as delay_prediction_days.
        "predicted_delay_days": "delay_prediction_days",
        "total_float": "total_float",
    }

    for phase5_col, phase7_col in checks.items():
        left = phase5_indexed[phase5_col].astype(float)
        right = adapted_indexed[phase7_col].astype(float)
        max_abs_diff = float((left - right).abs().max())
        if not left.equals(right) and max_abs_diff != 0.0:
            raise AssertionError(
                f"Phase 7 '{phase7_col}' differs from Phase 5 "
                f"'{phase5_col}' (max |diff| = {max_abs_diff}). "
                "Phase 7 must consume Phase 5 values unchanged."
            )
        print(
            f"  {phase5_col} == {phase7_col}: EXACT match "
            f"(max |diff| = {max_abs_diff})"
        )

    # is_critical must be the Phase 5 computed_critical flag.
    crit_left = phase5_indexed["computed_critical"].astype(bool)
    crit_right = adapted_indexed["is_critical"].astype(bool)
    mismatches = int((crit_left != crit_right).sum())
    if mismatches:
        raise AssertionError(
            f"Phase 7 is_critical differs from Phase 5 computed_critical "
            f"for {mismatches} activities."
        )
    print("  computed_critical == is_critical: EXACT match")

    # risk_level must be the deterministic classification of the SAME
    # Phase 5 score (High >= 0.70, Medium >= 0.40), traceable to Phase 5.
    def expected_level(score: float) -> str:
        if score >= 0.70:
            return "High"
        if score >= 0.40:
            return "Medium"
        return "Low"

    expected = adapted_indexed["risk_score"].astype(float).map(expected_level)
    actual = adapted_indexed["risk_level"].astype(str)
    level_mismatches = int((expected != actual).sum())
    if level_mismatches:
        raise AssertionError(
            f"Phase 7 risk_level not traceable to Phase 5 scores for "
            f"{level_mismatches} activities."
        )
    print("  risk_level classification: traceable to Phase 5 scores")

    print("\n  PHASE 5 CONSISTENCY: PASSED")


def validate_activity_insights() -> pd.DataFrame:
    """Validate generated activity-level insights."""
    print_section("2. ACTIVITY INSIGHT VALIDATION")

    insights = pd.read_csv(ACTIVITY_INSIGHTS_FILE)

    print(f"Activity insights: {len(insights):,}")

    print("\nInsight risk-level distribution:")

    distribution = insights["risk_level"].value_counts()

    for level in ["High", "Medium", "Low"]:
        count = int(distribution.get(level, 0))
        percentage = (count / len(insights)) * 100
        print(
            f"  {level}: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nRecommendation coverage:")

    non_empty_recommendations = (
        insights["recommendation"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    coverage = (
        non_empty_recommendations
        / len(insights)
        * 100
    )

    print(
        f"  Recommendations available: "
        f"{int(non_empty_recommendations):,}"
    )
    print(f"  Coverage: {coverage:.2f}%")

    print("\nExplanation coverage:")

    non_empty_explanations = (
        insights["explanation"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    explanation_coverage = (
        non_empty_explanations
        / len(insights)
        * 100
    )

    print(
        f"  Explanations available: "
        f"{int(non_empty_explanations):,}"
    )
    print(f"  Coverage: {explanation_coverage:.2f}%")

    return insights


def validate_project_summaries() -> pd.DataFrame:
    """Validate project-level risk summaries."""
    print_section("3. PROJECT RISK SUMMARY VALIDATION")

    summaries = pd.read_csv(PROJECT_SUMMARIES_FILE)

    print(f"Projects: {len(summaries):,}")

    print("\nProject risk distribution:")

    distribution = summaries["overall_risk_level"].value_counts()

    for level in ["High", "Medium", "Low"]:
        count = int(distribution.get(level, 0))
        percentage = (count / len(summaries)) * 100
        print(
            f"  {level}: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nProject risk score statistics:")

    print(
        summaries["average_risk_score"]
        .describe()
        .round(4)
        .to_string()
    )

    print("\nHighest-risk projects:")

    columns = [
        "project_id",
        "overall_risk_level",
        "total_activities",
        "high_risk_activities",
        "critical_risk_activities",
        "average_risk_score",
        "maximum_risk_score",
        "maximum_predicted_delay_days",
    ]

    available_columns = [
        column
        for column in columns
        if column in summaries.columns
    ]

    print(
        summaries.sort_values(
            by="average_risk_score",
            ascending=False,
        )
        .head(10)[available_columns]
        .to_string(index=False)
    )

    return summaries


def validate_reports_json(summaries: pd.DataFrame) -> None:
    """Verify reports/project_risk_reports.json agrees with the summaries."""
    print_section("5. REPORTS JSON VALIDATION")

    if not PROJECT_REPORTS_FILE.exists():
        raise FileNotFoundError(
            f"Reports JSON not found: {PROJECT_REPORTS_FILE}"
        )

    with PROJECT_REPORTS_FILE.open("r", encoding="utf-8") as file:
        reports = json.load(file)

    print(f"  Reports in JSON: {len(reports):,}")

    if len(reports) != len(summaries):
        raise AssertionError(
            f"Reports JSON has {len(reports)} entries but summaries CSV "
            f"has {len(summaries)} projects."
        )

    for report in reports:
        project_id = str(report["project_id"])
        row = summaries[summaries["project_id"].astype(str) == project_id]
        if row.empty:
            raise AssertionError(
                f"Report project {project_id!r} missing from summaries CSV."
            )
        summary = row.iloc[0]
        if str(report["overall_risk_level"]) != str(
            summary["overall_risk_level"]
        ):
            raise AssertionError(
                f"overall_risk_level mismatch for project {project_id}."
            )
        if int(report["total_activities"]) != int(summary["total_activities"]):
            raise AssertionError(
                f"total_activities mismatch for project {project_id}."
            )
        top_ids = [str(t["activity_id"]) for t in report["top_risks"]]
        if len(top_ids) != len(set(top_ids)):
            raise AssertionError(
                f"Duplicate activities in top_risks for {project_id}."
            )

    print("  All report entries agree with project_risk_summaries.csv")
    print("  REPORTS JSON: PASSED")


def main() -> None:
    """Run Phase 7 validation."""
    print("=" * 70)
    print("PHASE 7 - RISK ANALYSIS VALIDATION")
    print("=" * 70)

    validate_files()

    activity_risks = validate_activity_risks()
    insights = validate_activity_insights()
    summaries = validate_project_summaries()

    validate_phase5_consistency(activity_risks)
    validate_reports_json(summaries)

    print_section("6. VALIDATION SUMMARY")

    print(
        f"Activity risk records: {len(activity_risks):,}"
    )
    print(
        f"Activity insight records: {len(insights):,}"
    )
    print(
        f"Project summaries: {len(summaries):,}"
    )

    print("\nValidation completed successfully.")

    print("=" * 70)
    print("PHASE 7 VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
