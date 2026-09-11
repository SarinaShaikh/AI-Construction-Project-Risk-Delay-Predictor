"""
Phase 7: Risk Analysis Validation

Validates the generated activity-level and project-level
construction risk analysis results.

This script does not modify the risk-scoring logic.
It only reads the generated CSV files and reports
their distributions and key statistics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

RISK_DIR = ROOT_DIR / "data" / "processed" / "risk"

ACTIVITY_RISK_FILE = RISK_DIR / "activity_risk_scores.csv"
ACTIVITY_INSIGHTS_FILE = RISK_DIR / "activity_risk_insights.csv"
PROJECT_SUMMARIES_FILE = RISK_DIR / "project_risk_summaries.csv"


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


def main() -> None:
    """Run Phase 7 validation."""
    print("=" * 70)
    print("PHASE 7 - RISK ANALYSIS VALIDATION")
    print("=" * 70)

    validate_files()

    activity_risks = validate_activity_risks()
    insights = validate_activity_insights()
    summaries = validate_project_summaries()

    print_section("4. VALIDATION SUMMARY")

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