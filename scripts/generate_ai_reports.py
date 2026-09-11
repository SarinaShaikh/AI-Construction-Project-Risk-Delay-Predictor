"""Generate AI-powered risk reports for all construction projects."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.ai_report_generator import AIReportGenerator

RISK_SUMMARY_PATH = (
    ROOT_DIR / "data" / "processed" / "risk" / "project_risk_summaries.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "reports" / "ai"


def main() -> None:
    """Generate AI reports for all projects."""
    if not RISK_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Risk summary file not found: {RISK_SUMMARY_PATH}"
        )

    df = pd.read_csv(RISK_SUMMARY_PATH)

    project_ids = sorted(
        df["project_id"].astype(str).unique()
    )

    print(f"Total projects found: {len(project_ids)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = AIReportGenerator()

    successful = 0
    skipped = 0
    failed = 0

    for index, project_id in enumerate(project_ids, start=1):
        output_path = OUTPUT_DIR / f"{project_id}_ai_risk_report.json"

        print(
            f"\n[{index}/{len(project_ids)}] "
            f"Processing {project_id}..."
        )

        if output_path.exists():
            print("  SKIPPED: Report already exists.")
            skipped += 1
            continue

        try:
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
    print(f"Output folder  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()