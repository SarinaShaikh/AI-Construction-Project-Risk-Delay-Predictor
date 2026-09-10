
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scenarios.simulator import ScenarioResult, WhatIfSimulator

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

ACTIVITIES_FILE = (
    REPO_ROOT / "data" / "raw" / "activities.csv"
)

DEPENDENCIES_FILE = (
    REPO_ROOT / "data" / "raw" / "dependencies.csv"
)

OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "scenarios"
)

JSON_OUTPUT_FILE = (
    OUTPUT_DIR / "scenario_results.json"
)

CSV_OUTPUT_FILE = (
    OUTPUT_DIR / "scenario_results.csv"
)


# ----------------------------------------------------------------------
# CSV LOADING
# ----------------------------------------------------------------------

def load_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CSV file into a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


# ----------------------------------------------------------------------
# DATA HELPERS
# ----------------------------------------------------------------------

def get_project_ids(
    activities: list[dict[str, Any]],
) -> list[str]:
    """Return sorted unique project IDs."""

    project_ids = {
        str(row["project_id"])
        for row in activities
        if row.get("project_id")
    }

    return sorted(project_ids)


def get_project_activities(
    activities: list[dict[str, Any]],
    project_id: str,
) -> list[dict[str, Any]]:
    """Return activities belonging to one project."""

    return [
        row
        for row in activities
        if str(row.get("project_id")) == project_id
    ]


def choose_project(
    activities: list[dict[str, Any]],
) -> str:
    """
    Choose a project for the demonstration.

    The project with the largest number of activities is selected.
    This provides a reasonably representative schedule for testing.
    """

    project_counts: dict[str, int] = {}

    for row in activities:
        project_id = str(row.get("project_id", ""))

        if not project_id:
            continue

        project_counts[project_id] = (
            project_counts.get(project_id, 0) + 1
        )

    if not project_counts:
        raise ValueError(
            "No valid project_id values found in activities.csv"
        )

    return max(
        project_counts,
        key=project_counts.get,
    )


def choose_activities(
    activities: list[dict[str, Any]],
    dependencies: list[dict[str, Any]],
    project_id: str,
) -> tuple[str, str, str]:
    """
    Select representative activities for the three scenarios.

    Selection strategy:
    - Activity delay: activity with the largest duration.
    - Resource reduction: activity with the largest duration
      that is different from the delay activity when possible.
    - Weather delay: another activity with a dependency relationship
      when possible.

    This keeps the script deterministic and avoids hard-coding
    activity IDs that may differ between datasets.
    """

    project_activities = get_project_activities(
        activities,
        project_id,
    )

    if not project_activities:
        raise ValueError(
            f"No activities found for project: {project_id}"
        )

    sorted_activities = sorted(
        project_activities,
        key=lambda row: (
            -float(
                row.get(
                    "planned_duration_days",
                    0,
                )
            ),
            str(row.get("activity_id", "")),
        ),
    )

    activity_ids = [
        str(row["activity_id"])
        for row in sorted_activities
    ]

    activity_delay_id = activity_ids[0]

    resource_reduction_id = (
        activity_ids[1]
        if len(activity_ids) > 1
        else activity_ids[0]
    )

    # Prefer an activity that participates in a dependency.
    dependency_activity_ids: list[str] = []

    for row in dependencies:
        if str(row.get("project_id")) != project_id:
            continue

        predecessor = str(
            row.get("predecessor_id", "")
        )

        successor = str(
            row.get("successor_id", "")
        )

        if predecessor:
            dependency_activity_ids.append(
                predecessor
            )

        if successor:
            dependency_activity_ids.append(
                successor
            )

    dependency_activity_ids = list(
        dict.fromkeys(dependency_activity_ids)
    )

    weather_candidates = [
        activity_id
        for activity_id in dependency_activity_ids
        if activity_id in activity_ids
    ]

    if weather_candidates:
        weather_id = weather_candidates[0]
    else:
        weather_id = (
            activity_ids[2]
            if len(activity_ids) > 2
            else activity_ids[0]
        )

    return (
        activity_delay_id,
        resource_reduction_id,
        weather_id,
    )


# ----------------------------------------------------------------------
# RESULT SERIALIZATION
# ----------------------------------------------------------------------

def result_to_dict(
    result: ScenarioResult,
) -> dict[str, Any]:
    """Convert ScenarioResult to a JSON-compatible dictionary."""

    return result.to_dict()


def write_json(
    results: list[ScenarioResult],
    path: Path,
) -> None:
    """Write scenario results to JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = [
        result_to_dict(result)
        for result in results
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def write_csv(
    results: list[ScenarioResult],
    path: Path,
) -> None:
    """Write a compact scenario summary to CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    for result in results:
        rows.append(
            {
                "project_id": result.project_id,
                "scenario_type": result.scenario_type,
                "scenario_description": (
                    result.scenario_description
                ),
                "activity_id": result.activity_id,
                "delay_days": result.delay_days,
                "baseline_project_duration": (
                    result.baseline_project_duration
                ),
                "scenario_project_duration": (
                    result.scenario_project_duration
                ),
                "project_duration_delta": (
                    result.project_duration_delta
                ),
                "baseline_float_days": (
                    result.baseline_float_days
                ),
                "scenario_float_days": (
                    result.scenario_float_days
                ),
                "float_consumed_days": (
                    result.float_consumed_days
                ),
                "downstream_activity_count": len(
                    result.downstream_activity_ids
                ),
                "newly_critical_activity_count": len(
                    result.newly_critical_activity_ids
                ),
                "no_longer_critical_activity_count": len(
                    result.no_longer_critical_activity_ids
                ),
                "critical_path_changed": (
                    result.critical_path_changed
                ),
                "risk_score": result.risk_score,
                "risk_level": result.risk_level,
                "baseline_has_cycles": (
                    result.baseline_has_cycles
                ),
                "scenario_has_cycles": (
                    result.scenario_has_cycles
                ),
            }
        )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# DISPLAY
# ----------------------------------------------------------------------

def print_result(
    result: ScenarioResult,
) -> None:
    """Print a readable scenario result."""

    print()
    print("-" * 70)
    print(
        f"Scenario: {result.scenario_type}"
    )
    print("-" * 70)

    print(
        f"Project ID:              {result.project_id}"
    )

    print(
        f"Activity ID:             {result.activity_id}"
    )

    print(
        f"Scenario:                "
        f"{result.scenario_description}"
    )

    print(
        f"Effective delay:         "
        f"{result.delay_days:.2f} days"
    )

    print(
        f"Baseline duration:       "
        f"{result.baseline_project_duration:.2f} days"
    )

    print(
        f"Scenario duration:       "
        f"{result.scenario_project_duration:.2f} days"
    )

    print(
        f"Project duration impact: "
        f"{result.project_duration_delta:.2f} days"
    )

    print(
        f"Baseline float:          "
        f"{result.baseline_float_days:.2f} days"
    )

    print(
        f"Scenario float:          "
        f"{result.scenario_float_days:.2f} days"
    )

    print(
        f"Float consumed:          "
        f"{result.float_consumed_days:.2f} days"
    )

    print(
        f"Downstream activities:   "
        f"{len(result.downstream_activity_ids)}"
    )

    print(
        f"Newly critical:          "
        f"{len(result.newly_critical_activity_ids)}"
    )

    print(
        f"No longer critical:     "
        f"{len(result.no_longer_critical_activity_ids)}"
    )

    print(
        f"Critical path changed:   "
        f"{result.critical_path_changed}"
    )

    print(
        f"Risk score:              "
        f"{result.risk_score:.2f}/100"
    )

    print(
        f"Risk level:              "
        f"{result.risk_level}"
    )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:
    """Run the Phase 6 what-if scenarios."""

    print("=" * 70)
    print("PHASE 6 — WHAT-IF SCENARIO ANALYSIS")
    print("=" * 70)
    print()

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    print("Loading construction schedule data...")

    activities = load_csv(
        ACTIVITIES_FILE
    )

    dependencies = load_csv(
        DEPENDENCIES_FILE
    )

    print(
        f"Activities loaded:   {len(activities):,}"
    )

    print(
        f"Dependencies loaded: {len(dependencies):,}"
    )

    # --------------------------------------------------------------
    # Select project
    # --------------------------------------------------------------

    project_ids = get_project_ids(
        activities
    )

    print(
        f"Projects available:   {len(project_ids):,}"
    )

    project_id = choose_project(
        activities
    )

    project_activities = get_project_activities(
        activities,
        project_id,
    )

    print()
    print(
        f"Selected project:      {project_id}"
    )

    print(
        f"Project activities:    "
        f"{len(project_activities):,}"
    )

    # --------------------------------------------------------------
    # Select scenario activities
    # --------------------------------------------------------------

    (
        activity_delay_id,
        resource_reduction_id,
        weather_id,
    ) = choose_activities(
        activities,
        dependencies,
        project_id,
    )

    print()
    print("Scenario activities:")
    print(
        f"  Activity delay:       "
        f"{activity_delay_id}"
    )
    print(
        f"  Resource reduction:   "
        f"{resource_reduction_id}"
    )
    print(
        f"  Weather delay:        "
        f"{weather_id}"
    )

    # --------------------------------------------------------------
    # Create simulator
    # --------------------------------------------------------------

    simulator = WhatIfSimulator(
        activity_rows=activities,
        dependency_rows=dependencies,
    )

    # --------------------------------------------------------------
    # Run scenarios
    # --------------------------------------------------------------

    results: list[ScenarioResult] = []

    print()
    print("Running scenarios...")

    # Scenario 1:
    # Activity delayed by 7 days.
    activity_delay_result = (
        simulator.simulate_activity_delay(
            project_id=project_id,
            activity_id=activity_delay_id,
            delay_days=7,
        )
    )

    results.append(
        activity_delay_result
    )

    # Scenario 2:
    # Resource allocation reduced by 20%.
    resource_result = (
        simulator.simulate_resource_reduction(
            project_id=project_id,
            activity_id=resource_reduction_id,
            reduction_percent=20,
        )
    )

    results.append(
        resource_result
    )

    # Scenario 3:
    # Weather event causes 5 days of lost working time.
    weather_result = (
        simulator.simulate_weather_delay(
            project_id=project_id,
            activity_id=weather_id,
            delay_days=5,
        )
    )

    results.append(
        weather_result
    )

    # --------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------

    for result in results:
        print_result(result)

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    write_json(
        results,
        JSON_OUTPUT_FILE,
    )

    write_csv(
        results,
        CSV_OUTPUT_FILE,
    )

    # --------------------------------------------------------------
    # Final message
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("SCENARIO ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"JSON output: {JSON_OUTPUT_FILE}"
    )

    print(
        f"CSV output:  {CSV_OUTPUT_FILE}"
    )

    print()
    print(
        f"Scenarios executed: {len(results)}"
    )


if __name__ == "__main__":
    main()
