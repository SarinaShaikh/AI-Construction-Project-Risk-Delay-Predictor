"""Generate AI-powered risk reports for all construction projects.

Phase 8 integration notes (branch phase7-integration):

Adapted from the friend's implementation (commit d01fb48) with three
additive, documented changes; the generation logic itself is unchanged:

1.  Optional CLI overrides --summaries and --output-dir. Defaults keep
    the friend's paths exactly:
        data/processed/risk/project_risk_summaries.csv   (our Phase 7 output)
        data/processed/reports/ai/                       (gitignored via data/processed/)
2.  Optional --demo flag: runs the full pipeline with a deterministic
    OfflineGroqClient stub instead of the Groq API, so the workflow can
    be demonstrated offline with no GROQ_API_KEY. Demo analyses restate
    only the numbers provided by Phase 7 — nothing is invented.
3.  The LLM layer only explains the deterministic Phase 7 numbers:
    ProjectRiskTool reads project_risk_summaries.csv produced by
    scripts/run_risk_analysis.py (Phase 7). No CPM, ML, or risk score
    is calculated here.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.ai_report_generator import AIReportGenerator
from src.llm.llm_risk_analyzer import LLMRiskAnalyzer
from src.llm.prompt_builder import RiskPromptBuilder
from src.llm.risk_tools import ProjectRiskTool

RISK_SUMMARY_PATH = (
    ROOT_DIR / "data" / "processed" / "risk" / "project_risk_summaries.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "reports" / "ai"


class OfflineGroqClient:
    """Deterministic stub for GroqClient used by --demo mode.

    Implements the same generate(prompt) interface as GroqClient but
    never contacts an external API. The produced analysis restates only
    the grounding numbers that came from the Phase 7 summaries file.
    """

    def __init__(self, project_data: dict) -> None:
        """Hold the deterministic Phase 7 data for one project."""
        self.project_data = project_data

    def generate(self, prompt: str) -> str:
        """Return a grounded, offline analysis for one project."""
        data = self.project_data
        return (
            "[Offline demo analysis — no external LLM was called.]\n"
            "This analysis restates only the deterministic Phase 7 "
            "numbers for project "
            f"{data['project_id']}.\n\n"
            "1. Overall Risk Assessment\n"
            f"- The project carries {data['total_activities']} activities "
            f"with an overall risk level of {data['overall_risk_level']}.\n"
            f"- Average risk score: {data['average_risk_score']:.2f} "
            f"(maximum: {data['maximum_risk_score']:.2f}).\n\n"
            "2. Key Risk Factors\n"
            f"- {data['high_risk_activities']} high-risk and "
            f"{data['critical_risk_activities']} critical-risk activities "
            "require priority attention.\n"
            f"- {data['medium_risk_activities']} medium-risk and "
            f"{data['low_risk_activities']} low-risk activities make up "
            "the remainder.\n\n"
            "3. Delay Analysis\n"
            f"- Average predicted delay: "
            f"{data['average_predicted_delay_days']:.2f} days "
            f"(maximum: {data['maximum_predicted_delay_days']:.2f} days).\n"
            "- These are event-induced/disruption delay predictions, not "
            "actual project schedule slippage.\n\n"
            "4. Recommended Mitigation Actions\n"
            "- Focus mitigation planning on the critical and high-risk "
            "activities listed above.\n"
            "- Use the Phase 7 deterministic insights "
            "(activity_risk_insights.csv) for per-activity actions.\n\n"
            "5. Priority\n"
            "- High priority: critical-risk activities.\n"
            "- Medium priority: high-risk activities.\n"
            "- Low priority: remaining activities."
        )


def parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Generate AI risk reports from Phase 7 summaries."
    )
    parser.add_argument(
        "--summaries",
        type=Path,
        default=RISK_SUMMARY_PATH,
        help="Path to project_risk_summaries.csv (Phase 7 output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the generated AI report JSON files.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run offline with a deterministic stub LLM client "
            "(no Groq API call, no API key needed)."
        ),
    )
    return parser.parse_args()


def make_generator(
    project_id: str,
    summaries_path: Path,
    output_dir: Path,
    demo: bool,
) -> AIReportGenerator:
    """Build the report generator for one project."""
    if demo:
        risk_tool = ProjectRiskTool(summaries_path)
        return AIReportGenerator(
            analyzer=LLMRiskAnalyzer(
                risk_tool=risk_tool,
                prompt_builder=RiskPromptBuilder(),
                groq_client=OfflineGroqClient(
                    risk_tool.get_project_risk(project_id)
                ),
            ),
            output_dir=output_dir,
        )

    return AIReportGenerator(output_dir=output_dir)


def main() -> None:
    """Generate AI reports for all projects."""
    args = parse_args()

    risk_summary_path = args.summaries
    output_dir = args.output_dir

    if not risk_summary_path.exists():
        raise FileNotFoundError(
            f"Risk summary file not found: {risk_summary_path}"
        )

    df = pd.read_csv(risk_summary_path)

    project_ids = sorted(
        df["project_id"].astype(str).unique()
    )

    print(f"Total projects found: {len(project_ids)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    skipped = 0
    failed = 0

    for index, project_id in enumerate(project_ids, start=1):
        output_path = output_dir / f"{project_id}_ai_risk_report.json"

        print(
            f"\n[{index}/{len(project_ids)}] "
            f"Processing {project_id}..."
        )

        if output_path.exists():
            print("  SKIPPED: Report already exists.")
            skipped += 1
            continue

        try:
            generator = make_generator(
                project_id,
                risk_summary_path,
                output_dir,
                args.demo,
            )

            report = generator.generate_report(project_id)

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    report,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            print(f"  SUCCESS: Saved {output_path.name}")
            successful += 1

        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(
                f"  FAILED: {project_id} - "
                f"{type(exc).__name__}: {exc}"
            )
            failed += 1

    print("\n" + "=" * 60)
    print("AI REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total projects : {len(project_ids)}")
    print(f"Successful     : {successful}")
    print(f"Skipped        : {skipped}")
    print(f"Failed         : {failed}")
    print(f"Output folder  : {output_dir}")


if __name__ == "__main__":
    main()
