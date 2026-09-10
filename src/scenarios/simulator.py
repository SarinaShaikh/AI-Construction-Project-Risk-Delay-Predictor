from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from cpm.calculation import (
    ActivityCpmResult,
    ActivityRef,
    ProjectCpmSummary,
    calculate_project_cpm,
    calculate_project_cpm_details,
)


@dataclass
class ScenarioResult:
    """Result of a deterministic construction schedule what-if scenario."""

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

    downstream_activity_ids: list[str]

    baseline_float_days: float
    scenario_float_days: float
    float_consumed_days: float

    newly_critical_activity_ids: list[str]
    no_longer_critical_activity_ids: list[str]

    risk_score: float
    risk_level: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the scenario result to a JSON-compatible dictionary."""

        return asdict(self)


class WhatIfSimulator:
    """
    Deterministic what-if simulator built on the project's CPM engine.

    Supported scenarios:
    1. Activity delay
    2. Resource allocation reduction
    3. Weather delay

    The simulator modifies the affected activity and recalculates CPM.
    It never directly adds days to the project completion date.
    """

    def __init__(
        self,
        activity_rows: Iterable[dict[str, Any]],
        dependency_rows: Iterable[dict[str, Any]],
    ) -> None:
        self.activity_rows = list(activity_rows)
        self.dependency_rows = list(dependency_rows)

    # ------------------------------------------------------------------
    # DATA HELPERS
    # ------------------------------------------------------------------

    def _get_project_data(
        self,
        project_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return activities and dependencies belonging to one project."""

        activities = [
            row
            for row in self.activity_rows
            if str(row.get("project_id")) == project_id
        ]

        dependencies = [
            row
            for row in self.dependency_rows
            if str(row.get("project_id")) == project_id
        ]

        if not activities:
            raise ValueError(f"Project not found: {project_id}")

        return activities, dependencies

    def _find_activity(
        self,
        activities: list[dict[str, Any]],
        activity_id: str,
    ) -> dict[str, Any]:
        """Find one activity inside a project."""

        target = next(
            (
                row
                for row in activities
                if str(row.get("activity_id")) == activity_id
            ),
            None,
        )

        if target is None:
            raise ValueError(
                f"Activity not found in project: {activity_id}"
            )

        return target

    # ------------------------------------------------------------------
    # CPM CALCULATIONS
    # ------------------------------------------------------------------

    def _calculate_summary(
        self,
        project_id: str,
        activities: list[dict[str, Any]],
        dependencies: list[dict[str, Any]],
    ) -> ProjectCpmSummary:
        """Calculate project-level CPM summary."""

        return calculate_project_cpm(
            project_id=project_id,
            activity_rows=activities,
            dependency_rows=dependencies,
            activity_id_col="activity_id",
            duration_col="planned_duration_days",
            pred_col="predecessor_id",
            succ_col="successor_id",
            rel_col="relationship",
            lag_col="lag_days",
            include_virtual_end=True,
        )

    def _calculate_details(
        self,
        project_id: str,
        activities: list[dict[str, Any]],
        dependencies: list[dict[str, Any]],
    ) -> dict[ActivityRef, ActivityCpmResult]:
        """Calculate activity-level CPM details."""

        return calculate_project_cpm_details(
            project_id=project_id,
            activity_rows=activities,
            dependency_rows=dependencies,
            activity_id_col="activity_id",
            duration_col="planned_duration_days",
            pred_col="predecessor_id",
            succ_col="successor_id",
            rel_col="relationship",
            lag_col="lag_days",
            include_virtual_end=True,
        )

    # ------------------------------------------------------------------
    # DOWNSTREAM ANALYSIS
    # ------------------------------------------------------------------

    def _get_downstream_activities(
        self,
        project_id: str,
        activity_id: str,
        activity_details: dict[ActivityRef, ActivityCpmResult],
    ) -> list[str]:
        """
        Return all downstream activities affected by an activity.

        A breadth-first traversal follows the CPM successor relationships.
        """

        start_ref = ActivityRef(
            project_id=project_id,
            activity_id=activity_id,
        )

        if start_ref not in activity_details:
            raise ValueError(
                f"Activity not found in CPM results: {activity_id}"
            )

        visited: set[ActivityRef] = set()
        queue: list[ActivityRef] = [
            start_ref,
        ]

        downstream: list[str] = []

        while queue:
            current = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)

            current_result = activity_details.get(current)

            if current_result is None:
                continue

            for successor in current_result.successors:
                if successor in visited:
                    continue

                if successor.activity_id not in downstream:
                    downstream.append(successor.activity_id)

                queue.append(successor)

        return downstream

    # ------------------------------------------------------------------
    # FLOAT ANALYSIS
    # ------------------------------------------------------------------

    def _get_activity_float(
        self,
        project_id: str,
        activity_id: str,
        activity_details: dict[ActivityRef, ActivityCpmResult],
    ) -> float:
        """Return total float for one activity."""

        ref = ActivityRef(
            project_id=project_id,
            activity_id=activity_id,
        )

        result = activity_details.get(ref)

        if result is None:
            raise ValueError(
                f"Activity not found in CPM results: {activity_id}"
            )

        return float(result.total_float)

    def _calculate_float_consumption(
        self,
        project_id: str,
        activity_id: str,
        baseline_details: dict[ActivityRef, ActivityCpmResult],
        scenario_details: dict[ActivityRef, ActivityCpmResult],
    ) -> tuple[float, float, float]:
        """
        Compare baseline and scenario float.

        Returns:
            baseline float,
            scenario float,
            consumed float.
        """

        baseline_float = self._get_activity_float(
            project_id,
            activity_id,
            baseline_details,
        )

        scenario_float = self._get_activity_float(
            project_id,
            activity_id,
            scenario_details,
        )

        consumed = max(
            0.0,
            baseline_float - scenario_float,
        )

        return (
            baseline_float,
            scenario_float,
            consumed,
        )

    # ------------------------------------------------------------------
    # CRITICALITY ANALYSIS
    # ------------------------------------------------------------------

    def _critical_activity_ids(
        self,
        activity_details: dict[ActivityRef, ActivityCpmResult],
    ) -> set[str]:
        """Return activity IDs that are critical."""

        return {
            ref.activity_id
            for ref, result in activity_details.items()
            if result.is_critical and not result.is_virtual
        }

    def _criticality_changes(
        self,
        baseline_details: dict[ActivityRef, ActivityCpmResult],
        scenario_details: dict[ActivityRef, ActivityCpmResult],
    ) -> tuple[list[str], list[str]]:
        """Find newly critical and no-longer-critical activities."""

        baseline_critical = self._critical_activity_ids(
            baseline_details
        )

        scenario_critical = self._critical_activity_ids(
            scenario_details
        )

        newly_critical = sorted(
            scenario_critical - baseline_critical
        )

        no_longer_critical = sorted(
            baseline_critical - scenario_critical
        )

        return (
            newly_critical,
            no_longer_critical,
        )

    # ------------------------------------------------------------------
    # RISK ANALYSIS
    # ------------------------------------------------------------------

    def _calculate_risk_score(
        self,
        project_duration_delta: float,
        float_consumed_days: float,
        newly_critical_count: int,
        critical_path_changed: bool,
    ) -> tuple[float, str]:
        """
        Calculate a deterministic scenario risk score.

        This is a schedule-impact risk indicator, not an ML probability.
        It does not invent a probability of delay.

        Score components:
        - project duration impact
        - float consumption
        - newly critical activities
        - critical-path change
        """

        score = 0.0

        # Project completion impact.
        score += min(
            max(project_duration_delta, 0.0) * 2.0,
            40.0,
        )

        # Float consumption.
        score += min(
            max(float_consumed_days, 0.0) * 1.5,
            30.0,
        )

        # Newly critical activities.
        score += min(
            newly_critical_count * 5.0,
            20.0,
        )

        # Critical path changed.
        if critical_path_changed:
            score += 10.0

        score = min(max(score, 0.0), 100.0)

        if score >= 70:
            level = "High"
        elif score >= 40:
            level = "Medium"
        else:
            level = "Low"

        return (
            round(score, 2),
            level,
        )

    # ------------------------------------------------------------------
    # RESULT BUILDER
    # ------------------------------------------------------------------

    def _build_result(
        self,
        project_id: str,
        scenario_type: str,
        scenario_description: str,
        activity_id: str,
        delay_days: float,
        baseline: ProjectCpmSummary,
        scenario: ProjectCpmSummary,
        baseline_details: dict[ActivityRef, ActivityCpmResult],
        scenario_details: dict[ActivityRef, ActivityCpmResult],
    ) -> ScenarioResult:
        """Build the complete scenario result."""

        baseline_paths = [
            list(path)
            for path in baseline.critical_paths
        ]

        scenario_paths = [
            list(path)
            for path in scenario.critical_paths
        ]

        critical_path_changed = (
            baseline_paths != scenario_paths
            or baseline.critical_activity_ids
            != scenario.critical_activity_ids
        )

        (
            baseline_float,
            scenario_float,
            float_consumed,
        ) = self._calculate_float_consumption(
            project_id,
            activity_id,
            baseline_details,
            scenario_details,
        )

        (
            newly_critical,
            no_longer_critical,
        ) = self._criticality_changes(
            baseline_details,
            scenario_details,
        )

        project_duration_delta = (
            scenario.project_duration
            - baseline.project_duration
        )

        risk_score, risk_level = self._calculate_risk_score(
            project_duration_delta=project_duration_delta,
            float_consumed_days=float_consumed,
            newly_critical_count=len(newly_critical),
            critical_path_changed=critical_path_changed,
        )

        downstream = self._get_downstream_activities(
            project_id,
            activity_id,
            baseline_details,
        )

        return ScenarioResult(
            project_id=project_id,
            scenario_type=scenario_type,
            scenario_description=scenario_description,
            activity_id=activity_id,
            delay_days=delay_days,
            baseline_project_duration=baseline.project_duration,
            scenario_project_duration=scenario.project_duration,
            project_duration_delta=project_duration_delta,
            baseline_critical_activity_ids=list(
                baseline.critical_activity_ids
            ),
            scenario_critical_activity_ids=list(
                scenario.critical_activity_ids
            ),
            baseline_critical_paths=baseline_paths,
            scenario_critical_paths=scenario_paths,
            critical_path_changed=critical_path_changed,
            baseline_has_cycles=baseline.has_cycles,
            scenario_has_cycles=scenario.has_cycles,
            baseline_dependency_violations=list(
                baseline.dependency_violations
            ),
            scenario_dependency_violations=list(
                scenario.dependency_violations
            ),
            downstream_activity_ids=downstream,
            baseline_float_days=baseline_float,
            scenario_float_days=scenario_float,
            float_consumed_days=float_consumed,
            newly_critical_activity_ids=newly_critical,
            no_longer_critical_activity_ids=no_longer_critical,
            risk_score=risk_score,
            risk_level=risk_level,
        )

    # ------------------------------------------------------------------
    # ACTIVITY MODIFICATION
    # ------------------------------------------------------------------

    def _modify_activity_duration(
        self,
        activities: list[dict[str, Any]],
        activity_id: str,
        additional_days: float,
    ) -> None:
        """Increase one activity's planned duration."""

        target_row = self._find_activity(
            activities,
            activity_id,
        )

        original_duration = float(
            target_row["planned_duration_days"]
        )

        target_row["planned_duration_days"] = (
            original_duration + additional_days
        )

    # ------------------------------------------------------------------
    # ACTIVITY DELAY
    # ------------------------------------------------------------------

    def simulate_activity_delay(
        self,
        project_id: str,
        activity_id: str,
        delay_days: float,
    ) -> ScenarioResult:
        """
        Simulate a direct delay to an activity.

        Example:
            Activity A delayed by 7 days.
        """

        if delay_days < 0:
            raise ValueError(
                "delay_days cannot be negative"
            )

        activities, dependencies = self._get_project_data(
            project_id
        )

        baseline = self._calculate_summary(
            project_id,
            activities,
            dependencies,
        )

        baseline_details = self._calculate_details(
            project_id,
            activities,
            dependencies,
        )

        scenario_activities = copy.deepcopy(
            activities
        )

        self._modify_activity_duration(
            scenario_activities,
            activity_id,
            delay_days,
        )

        scenario = self._calculate_summary(
            project_id,
            scenario_activities,
            dependencies,
        )

        scenario_details = self._calculate_details(
            project_id,
            scenario_activities,
            dependencies,
        )

        description = (
            f"Activity {activity_id} delayed by "
            f"{delay_days:g} day(s)"
        )

        return self._build_result(
            project_id=project_id,
            scenario_type="activity_delay",
            scenario_description=description,
            activity_id=activity_id,
            delay_days=delay_days,
            baseline=baseline,
            scenario=scenario,
            baseline_details=baseline_details,
            scenario_details=scenario_details,
        )

    # ------------------------------------------------------------------
    # RESOURCE REDUCTION
    # ------------------------------------------------------------------

    def simulate_resource_reduction(
        self,
        project_id: str,
        activity_id: str,
        reduction_percent: float,
    ) -> ScenarioResult:
        """
        Simulate reduced resource allocation.

        The current dataset does not contain a reliable resource
        allocation field, so the resource reduction is represented
        deterministically by increasing duration:

            new_duration =
                original_duration / (1 - reduction / 100)

        Example:
            20% resource reduction causes a proportional increase
            in required activity duration.
        """

        if reduction_percent < 0:
            raise ValueError(
                "reduction_percent cannot be negative"
            )

        if reduction_percent >= 100:
            raise ValueError(
                "reduction_percent must be less than 100"
            )

        activities, dependencies = self._get_project_data(
            project_id
        )

        baseline = self._calculate_summary(
            project_id,
            activities,
            dependencies,
        )

        baseline_details = self._calculate_details(
            project_id,
            activities,
            dependencies,
        )

        scenario_activities = copy.deepcopy(
            activities
        )

        target_row = self._find_activity(
            scenario_activities,
            activity_id,
        )

        original_duration = float(
            target_row["planned_duration_days"]
        )

        resource_factor = (
            1 - reduction_percent / 100
        )

        new_duration = (
            original_duration / resource_factor
        )

        target_row["planned_duration_days"] = new_duration

        effective_delay = (
            new_duration - original_duration
        )

        scenario = self._calculate_summary(
            project_id,
            scenario_activities,
            dependencies,
        )

        scenario_details = self._calculate_details(
            project_id,
            scenario_activities,
            dependencies,
        )

        description = (
            f"Activity {activity_id} resource allocation "
            f"reduced by {reduction_percent:g}%"
        )

        return self._build_result(
            project_id=project_id,
            scenario_type="resource_reduction",
            scenario_description=description,
            activity_id=activity_id,
            delay_days=effective_delay,
            baseline=baseline,
            scenario=scenario,
            baseline_details=baseline_details,
            scenario_details=scenario_details,
        )

    # ------------------------------------------------------------------
    # WEATHER DELAY
    # ------------------------------------------------------------------

    def simulate_weather_delay(
        self,
        project_id: str,
        activity_id: str,
        delay_days: float,
    ) -> ScenarioResult:
        """
        Simulate weather-related lost working time.

        The weather delay is represented as additional activity
        duration. CPM is recalculated to determine the actual
        project completion impact.
        """

        if delay_days < 0:
            raise ValueError(
                "delay_days cannot be negative"
            )

        activities, dependencies = self._get_project_data(
            project_id
        )

        baseline = self._calculate_summary(
            project_id,
            activities,
            dependencies,
        )

        baseline_details = self._calculate_details(
            project_id,
            activities,
            dependencies,
        )

        scenario_activities = copy.deepcopy(
            activities
        )

        self._modify_activity_duration(
            scenario_activities,
            activity_id,
            delay_days,
        )

        scenario = self._calculate_summary(
            project_id,
            scenario_activities,
            dependencies,
        )

        scenario_details = self._calculate_details(
            project_id,
            scenario_activities,
            dependencies,
        )

        description = (
            f"Weather event delays Activity {activity_id} "
            f"by {delay_days:g} day(s)"
        )

        return self._build_result(
            project_id=project_id,
            scenario_type="weather_delay",
            scenario_description=description,
            activity_id=activity_id,
            delay_days=delay_days,
            baseline=baseline,
            scenario=scenario,
            baseline_details=baseline_details,
            scenario_details=scenario_details,
        )


# ----------------------------------------------------------------------
# SIMPLE MANUAL TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import csv

    print("=" * 70)
    print("PHASE 6 — WHAT-IF SIMULATION")
    print("=" * 70)

    with open(
        "data/raw/activities.csv",
        encoding="utf-8",
        newline="",
    ) as file:
        activities = list(
            csv.DictReader(file)
        )

    with open(
        "data/raw/dependencies.csv",
        encoding="utf-8",
        newline="",
    ) as file:
        dependencies = list(
            csv.DictReader(file)
        )

    print(
        f"Activities loaded: {len(activities):,}"
    )
    print(
        f"Dependencies loaded: {len(dependencies):,}"
    )
    print()

    print("Available scenarios:")
    print("1. Activity delay")
    print("2. Resource reduction")
    print("3. Weather delay")