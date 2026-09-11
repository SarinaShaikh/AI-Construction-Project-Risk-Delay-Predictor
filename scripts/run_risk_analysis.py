"""
Phase 7: End-to-End Deterministic Risk Analysis Pipeline.

Runs the complete deterministic risk-analysis workflow:

1. Load the validated Phase 5 risk scores (data/processed/risk_scores.csv)
   produced by scripts/run_phase5_risk.py. Phase 5 remains the sole owner
   of risk scoring; scores are consumed as-is.
2. Adapt the Phase 5 output to the Phase 7 input schema (presentation
   column renames only; risk_score carried through unchanged).
3. Generate activity-level explanations and recommendations.
4. Generate project-level risk summaries.
5. Generate structured project risk reports.

Phase 7 is fully deterministic:
    - No external LLM/API is used.
    - No ML predictions are invented or recomputed.
    - No CPM is recalculated.
    - No Phase 5 risk score is recalculated or altered.

Risk score != actual schedule slippage: the underlying prediction target is
event-induced/disruption delay (target_event_delay_days), not actual
project schedule delay.

Integration note (branch phase7-integration):
    The upstream Phase 7 prototype ran RiskScorer().calculate_risk_scores()
    itself and wrote data/processed/risk/activity_risk_scores.csv. On this
    branch, Phase 5's validated output path and API are preserved:
    Phase 5 output -> src/llm/risk_adapter.py -> Phase 7 analysis. The
    adapted scores are still written to
    data/processed/risk/activity_risk_scores.csv so that
    scripts/validate_risk_analysis.py and Phase 8 can consume the
    documented Phase 7 layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.analyzer import ConstructionRiskAnalyzer
from llm.project_summary import ProjectRiskSummarizer
from llm.report_generator import ConstructionRiskReportGenerator
from llm.risk_adapter import load_phase5_risk_scores, to_phase7_risk_frame

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RISK_DIR = PROCESSED_DIR / "risk"
REPORT_DIR = ROOT_DIR / "reports"

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
    # Step 1: Load the validated Phase 5 risk scores
    # ------------------------------------------------------------------
    print("\n[1/5] Loading Phase 5 risk scores (no recalculation)...")

    phase5_scores = load_phase5_risk_scores()
    risk_scores = to_phase7_risk_frame(phase5_scores)

    RISK_DIR.mkdir(parents=True, exist_ok=True)
    risk_scores.to_csv(RISK_SCORES_FILE, index=False)

    print(f"  Phase 5 activities: {len(phase5_scores):,}")
    print(f"  Adapted activities: {len(risk_scores):,}")
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
