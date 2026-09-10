
"""
src/scenarios/simulator.py

Phase 6 — What-if Simulation Engine

This module wraps the existing Phase 3 CPM engine.

It performs project-level what-if simulations by:
1. Filtering activities and dependencies for one project.
2. Running the existing CPM calculation for the baseline.
3. Copying the activity data.
4. Applying a delay to one activity.
5. Running the existing CPM calculation again.
6. Comparing the two ProjectCpmSummary objects.

IMPORTANT:
- CPM mathematics is NOT duplicated here.
- src/cpm/calculation.py is NOT modified.
- This module uses only the public ProjectCpmSummary fields.
- ProjectCpmSummary does NOT expose ActivityCpmResult objects.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.cpm.calculation import calculate_project_cpm


@dataclass
class ScenarioResult:
    """Result of one activity-delay what-if scenario."""

    project_id: str
    scenario_type: str
    scenario_description: str

    activity_id: str
    delay_days: float

    baseline_project_duration: float
    scenario_project_duration: float
    project_duration_delta: float

    baseline_critical_activity_ids: list[str]
    scenario_critical_activity_ids: list[str]

    baseline_critical_paths: list[list[str]]
    scenario_critical_paths: list[list[str]]

    critical_path_changed: bool

    baseline_has_cycles: bool
    scenario_has_cycles: bool

    baseline_dependency_violations: list[dict[str, Any]]
    scenario_dependency_violations: list[dict[str, Any]]


class WhatIfSimulator:
    """
    Phase 6 What-if simulation engine.

    The simulator uses the existing Phase 3 CPM engine and does not
    reimplement CPM calculations.
    """

    def __init__(
        self,
        activity_rows: Iterable[dict[str, Any]],
        dependency_rows: Iterable[dict[str, Any]],
    ) -> None:
        """
        Store the raw activity and dependency rows.

        Expected activity fields:
            project_id
            activity_id
            planned_duration_days

        Expected dependency fields:
            project_id
            predecessor_id
            successor_id
            relationship
            lag_days
        """

        self._activity_rows = list(activity_rows)
        self._dependency_rows = list(dependency_rows)

    def simulate_activity_delay(
        self,
        project_id: str,
        activity_id: str,
        delay_days: float,
    ) -> ScenarioResult:
        """
        Simulate a delay to one activity.

        Example:

            simulator = WhatIfSimulator(
                activity_rows,
                dependency_rows,
            )

            result = simulator.simulate_activity_delay(
                project_id="P001",
                activity_id="A005",
                delay_days=7,
            )

        The method runs CPM twice:

            1. Baseline schedule
            2. Schedule after applying the activity delay
        """

        # --------------------------------------------------------------
        # 1. Validate delay
        # --------------------------------------------------------------

        if delay_days < 0:
            raise ValueError(
                f"delay_days must be >= 0, got {delay_days}."
            )

        # --------------------------------------------------------------
        # 2. Filter activities for the requested project
        # --------------------------------------------------------------

        project_activity_rows = [
            row
            for row in self._activity_rows
            if row.get("project_id") == project_id
        ]

        if not project_activity_rows:
            raise ValueError(
                f"No activities found for project_id={project_id!r}."
            )

        # --------------------------------------------------------------
        # 3. Validate activity exists in this project
        # --------------------------------------------------------------

        target_row = None

        for row in project_activity_rows:
            if row.get("activity_id") == activity_id:
                target_row = row
                break

        if target_row is None:
            raise ValueError(
                f"Activity {activity_id!r} was not found "
                f"in project {project_id!r}."
            )

        # --------------------------------------------------------------
        # 4. Filter dependencies for the requested project
        # --------------------------------------------------------------

        project_activity_ids = {
            row["activity_id"]
            for row in project_activity_rows
        }

        project_dependency_rows = [
            dependency
            for dependency in self._dependency_rows
            if (
                dependency.get("project_id") == project_id
                and dependency.get("predecessor_id")
                in project_activity_ids
                and dependency.get("successor_id")
                in project_activity_ids
            )
        ]

        # --------------------------------------------------------------
        # 5. BASELINE CPM
        # --------------------------------------------------------------

        baseline_summary = calculate_project_cpm(
            project_id=project_id,
            activity_rows=project_activity_rows,
            dependency_rows=project_dependency_rows,
            activity_id_col="activity_id",
            duration_col="planned_duration_days",
            pred_col="predecessor_id",
            succ_col="successor_id",
            rel_col="relationship",
            lag_col="lag_days",
            include_virtual_end=True,
        )

        # --------------------------------------------------------------
        # 6. Create scenario copy
        # --------------------------------------------------------------

        scenario_activity_rows = copy.deepcopy(
            project_activity_rows
        )

        # --------------------------------------------------------------
        # 7. Apply delay to selected activity
        # --------------------------------------------------------------

        scenario_target_row = None

        for row in scenario_activity_rows:
            if row.get("activity_id") == activity_id:
                scenario_target_row = row
                break

        if scenario_target_row is None:
            # This should never happen because the activity was already
            # validated above.
            raise RuntimeError(
                f"Could not find activity {activity_id!r} "
                "in the scenario copy."
            )

        original_duration = scenario_target_row[
            "planned_duration_days"
        ]

        scenario_target_row["planned_duration_days"] = (
            original_duration + delay_days
        )

        # --------------------------------------------------------------
        # 8. SCENARIO CPM
        # --------------------------------------------------------------

        scenario_summary = calculate_project_cpm(
            project_id=project_id,
            activity_rows=scenario_activity_rows,
            dependency_rows=project_dependency_rows,
            activity_id_col="activity_id",
            duration_col="planned_duration_days",
            pred_col="predecessor_id",
            succ_col="successor_id",
            rel_col="relationship",
            lag_col="lag_days",
            include_virtual_end=True,
        )

        # --------------------------------------------------------------
        # 9. Compare project-level CPM results
        # --------------------------------------------------------------

        baseline_duration = baseline_summary.project_duration
        scenario_duration = scenario_summary.project_duration

        duration_delta = (
            scenario_duration - baseline_duration
        )

        baseline_critical_ids = list(
            baseline_summary.critical_activity_ids
        )

        scenario_critical_ids = list(
            scenario_summary.critical_activity_ids
        )

        baseline_paths = [
            list(path)
            for path in baseline_summary.critical_paths
        ]

        scenario_paths = [
            list(path)
            for path in scenario_summary.critical_paths
        ]

        critical_path_changed = (
            baseline_paths != scenario_paths
            or baseline_critical_ids != scenario_critical_ids
        )

        # --------------------------------------------------------------
        # 10. Return scenario result
        # --------------------------------------------------------------

        return ScenarioResult(
            project_id=project_id,
            scenario_type="activity_delay",
            scenario_description=(
                f"Activity {activity_id} delayed by "
                f"{delay_days:g} day(s)"
            ),
            activity_id=activity_id,
            delay_days=delay_days,
            baseline_project_duration=baseline_duration,
            scenario_project_duration=scenario_duration,
            project_duration_delta=duration_delta,
            baseline_critical_activity_ids=baseline_critical_ids,
            scenario_critical_activity_ids=scenario_critical_ids,
            baseline_critical_paths=baseline_paths,
            scenario_critical_paths=scenario_paths,
            critical_path_changed=critical_path_changed,
            baseline_has_cycles=baseline_summary.has_cycles,
            scenario_has_cycles=scenario_summary.has_cycles,
            baseline_dependency_violations=list(
                baseline_summary.dependency_violations
            ),
            scenario_dependency_violations=list(
                scenario_summary.dependency_violations
            ),
        )


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 6 — WHAT-IF SIMULATION ENGINE")
    print("=" * 70)
    print()
    print("WhatIfSimulator is ready.")
    print()
    print("The simulator uses the Phase 3 CPM engine and supports:")
    print("- Activity delay scenarios")
    print("- Project duration comparison")
    print("- Critical activity comparison")
    print("- Critical path comparison")
    print("- Cycle detection comparison")
    print("- Dependency violation comparison")
    print()
    print("Use scripts/run_whatif.py to run a scenario with project data.")

