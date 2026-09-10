
"""
Phase 6 — What-if Simulation Runner

Loads the raw construction schedule data and runs an
activity-delay scenario using the Phase 6 simulator.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scenarios.simulator import WhatIfSimulator

ACTIVITIES_FILE = PROJECT_ROOT / "data" / "raw" / "activities.csv"
DEPENDENCIES_FILE = PROJECT_ROOT / "data" / "raw" / "dependencies.csv"


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dictionaries."""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    """Run a sample activity-delay what-if scenario."""

    print("=" * 70)
    print("PHASE 6 — WHAT-IF SIMULATION")
    print("=" * 70)
    print()

    if not ACTIVITIES_FILE.exists():
        raise FileNotFoundError(
            f"Activities file not found: {ACTIVITIES_FILE}"
        )

    if not DEPENDENCIES_FILE.exists():
        raise FileNotFoundError(
            f"Dependencies file not found: {DEPENDENCIES_FILE}"
        )

    print("Loading activity data...")
    activity_rows = load_csv(ACTIVITIES_FILE)

    print("Loading dependency data...")
    dependency_rows = load_csv(DEPENDENCIES_FILE)

    print(f"Activities loaded: {len(activity_rows):,}")
    print(f"Dependencies loaded: {len(dependency_rows):,}")
    print()

    # Sample scenario:
    # Use the first activity of the first project.
    project_id = activity_rows[0]["project_id"]

    project_activities = [
        row
        for row in activity_rows
        if row["project_id"] == project_id
    ]

    activity_id = project_activities[0]["activity_id"]

    # Delay the selected activity by 7 days.
    delay_days = 7

    print("Scenario configuration")
    print("-" * 70)
    print(f"Project:       {project_id}")
    print(f"Activity:      {activity_id}")
    print(f"Delay:         {delay_days} days")
    print()

    simulator = WhatIfSimulator(
        activity_rows=activity_rows,
        dependency_rows=dependency_rows,
    )

    result = simulator.simulate_activity_delay(
        project_id=project_id,
        activity_id=activity_id,
        delay_days=delay_days,
    )

    print("--- WHAT-IF RESULT ---")
    print(f"Project: {result.project_id}")
    print(f"Scenario: {result.scenario_description}")

    print(f"Baseline duration: {result.baseline_project_duration}")

    print(f"Scenario duration: {result.scenario_project_duration}")

    print(f"Duration change: {result.project_duration_delta}")

    print(f"Critical path changed: {result.critical_path_changed}")

    print()
    print("Baseline critical activities:")
    print(result.baseline_critical_activity_ids)

    print()
    print("Scenario critical activities:")
    print(result.scenario_critical_activity_ids)

    print()
    print("Baseline critical paths:")
    for path in result.baseline_critical_paths:
        print(path)

    print()
    print("Scenario critical paths:")
    for path in result.scenario_critical_paths:
        print(path)

    print()
    print(f"Baseline cycles: {result.baseline_has_cycles}")
    print(f"Scenario cycles: {result.scenario_has_cycles}")

    print()
    print(
        "Baseline dependency violations: "
        f"{len(result.baseline_dependency_violations)}"
    )

    print(
        "Scenario dependency violations: "
        f"{len(result.scenario_dependency_violations)}"
    )


if __name__ == "__main__":
    main()

