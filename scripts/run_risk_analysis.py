"""
Phase 7: End-to-End Construction Risk Analysis Pipeline

Runs the complete deterministic risk-analysis workflow:

1. Generate ML delay predictions
2. Calculate activity-level risk scores
3. Generate activity-level explanations and recommendations
4. Generate project-level risk summaries
5. Generate structured project risk reports

No external LLM/API is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.analyzer import ConstructionRiskAnalyzer
from src.llm.project_summary import ProjectRiskSummarizer
from src.llm.report_generator import ConstructionRiskReportGenerator
from src.risk.scoring import RiskScorer

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RISK_DIR = PROCESSED_DIR / "risk"
REPORT_DIR = PROCESSED_DIR / "reports"

RISK_SCORES_FILE = RISK_DIR / "activity_risk_scores.csv"
ACTIVITY_INSIGHTS_FILE = RISK_DIR / "activity_risk_insights.csv"
PROJECT_SUMMARIES_FILE = RISK_DIR / "project_risk_summaries.csv"
PROJECT_REPORTS_FILE = REPORT_DIR / "project_risk_reports.json"


def save_json(data: list[dict[str, object]], path: Path) -> None:
    """Save a list of dictionaries as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    """Run the complete Phase 7 risk-analysis pipeline."""
    print("=" * 70)
    print("PHASE 7 - CONSTRUCTION RISK ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate ML predictions and deterministic risk scores
    # ------------------------------------------------------------------
    print("\n[1/5] Calculating activity risk scores...")

    scorer = RiskScorer()
    
    risk_scores = scorer.calculate_risk_scores()

    RISK_DIR.mkdir(parents=True, exist_ok=True)
    risk_scores.to_csv(RISK_SCORES_FILE, index=False)

    print(f"  Activities analyzed: {len(risk_scores):,}")
    print(f"  Saved: {RISK_SCORES_FILE}")

    # ------------------------------------------------------------------
    # Step 2: Generate activity-level explanations/recommendations
    # ------------------------------------------------------------------
    print("\n[2/5] Generating activity-level risk insights...")

    analyzer = ConstructionRiskAnalyzer(risk_scores)

    activity_insights = analyzer.analyze()

    activity_insights.to_csv(
        ACTIVITY_INSIGHTS_FILE,
        index=False,
    )

    print(f"  Insights generated: {len(activity_insights):,}")
    print(f"  Saved: {ACTIVITY_INSIGHTS_FILE}")

    # ------------------------------------------------------------------
    # Step 3: Generate project-level summaries
    # ------------------------------------------------------------------
    print("\n[3/5] Generating project risk summaries...")

    summarizer = ProjectRiskSummarizer(activity_insights)

    project_summaries = summarizer.summarize_all()

    project_summaries.to_csv(
        PROJECT_SUMMARIES_FILE,
        index=False,
    )

    print(f"  Projects summarized: {len(project_summaries):,}")
    print(f"  Saved: {PROJECT_SUMMARIES_FILE}")

    # ------------------------------------------------------------------
    # Step 4: Generate structured project reports
    # ------------------------------------------------------------------
    print("\n[4/5] Generating project risk reports...")

    report_generator = ConstructionRiskReportGenerator(
        project_summaries,
        activity_insights,
    )

    reports: list[dict[str, object]] = []

    for project_id in project_summaries["project_id"]:
        report = report_generator.generate_report(
            str(project_id),
            top_n=5,
        )

        reports.append(report.to_dict())

    save_json(reports, PROJECT_REPORTS_FILE)

    print(f"  Reports generated: {len(reports):,}")
    print(f"  Saved: {PROJECT_REPORTS_FILE}")

    # ------------------------------------------------------------------
    # Step 5: Display overall results
    # ------------------------------------------------------------------
    print("\n[5/5] Risk analysis completed.")

    risk_distribution = project_summaries[
        "overall_risk_level"
    ].value_counts()

    print("\nProject Risk Distribution:")

    for level in ["High", "Medium", "Low"]:
        count = int(risk_distribution.get(level, 0))
        print(f"  {level}: {count}")

    print("\nSample Project Reports:")

    for report in reports[:3]:
        print(
            f"  {report['project_id']}: "
            f"{report['overall_risk_level']} risk | "
            f"{report['total_activities']} activities"
        )

    print("\n" + "=" * 70)
    print("PHASE 7 RISK ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()