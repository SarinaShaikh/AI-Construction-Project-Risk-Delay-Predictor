"""Phase 6: What-if scenario simulation engine.

Deterministic schedule impact analysis on top of the validated Phase 3 CPM
engine (src/cpm/calculation.py). Given an activity and a scenario modification,
the simulator:

1. Computes the baseline CPM for the project (using the existing engine).
2. Deep-copies the activity data.
3. Modifies ONLY the selected activity's planned duration.
4. Recomputes CPM on the modified copy.
5. Compares baseline vs scenario.

Scenario types:
    - activity_delay:     extend an activity's planned duration by N days.
    - weather_delay:      identical mechanics to activity_delay; documented as a
                          separate scenario for reporting/traceability.
    - resource_reduction: deterministic scenario assumption (NOT a learned
                          construction law):
                              new_duration = original_duration / (1 - reduction% / 100)
                          Rejected for reduction <= 0 or reduction >= 100.

This module does NOT create any risk score or probability. Phase 5
(src/risk/scoring.py) owns risk scoring. Phase 6 reports schedule impact only.

No event/rework/decision/outcome data is used. Everything is computed from the
planned activities + dependencies via the deterministic CPM engine.
"""

from __future__ import annotations

import copy
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make src/ importable regardless of entry point.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cpm.calculation import (
    ActivityRef,
    ProjectCpmSummary,
    _make_results,
    _topological_order,
    build_activity_index,
    build_dependencies,
    calculate_project_cpm,
)

# ============================================================================
# Scenario result
# ============================================================================


@dataclass
class ScenarioResult:
    """Structured result of a single what-if scenario comparison."""

    project_id: str
    scenario_type: str
    scenario_description: str
    activity_id: str

    # Scenario parameters
    delay_days: float = 0.0
    reduction_percent: float | None = None

    # Project-level impact
    baseline_project_duration: float = 0.0
    scenario_project_duration: float = 0.0
    project_duration_delta: float = 0.0

    # Criticality
    baseline_critical_activity_ids: list[str] = field(default_factory=list)
    scenario_critical_activity_ids: list[str] = field(default_factory=list)
    baseline_critical_paths: list[list[str]] = field(default_factory=list)
    scenario_critical_paths: list[list[str]] = field(default_factory=list)
    critical_path_changed: bool = False

    # Downstream / criticality shifts
    downstream_activity_ids: list[str] = field(default_factory=list)
    newly_critical_activity_ids: list[str] = field(default_factory=list)
    no_longer_critical_activity_ids: list[str] = field(default_factory=list)

    # Float (for the modified activity)
    baseline_float_days: float | None = None
    scenario_float_days: float | None = None
    float_consumed_days: float = 0.0

    # Integrity
    baseline_has_cycles: bool = False
    scenario_has_cycles: bool = False
    baseline_dependency_violations: list[dict[str, Any]] = field(default_factory=list)
    scenario_dependency_violations: list[dict[str, Any]] = field(default_factory=list)

    # Optional: full per-activity comparison (not part of the minimum contract)
    activity_comparisons: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict form (safe for JSON/CSV writers)."""
        return asdict(self)


# ============================================================================
# Internal helpers
# ============================================================================


def _compute_cpm_details(
    project_id: str,
    activity_rows: list[dict[str, Any]],
    dependency_rows: list[dict[str, Any]],
) -> tuple[ProjectCpmSummary, dict[ActivityRef, Any]]:
    """Run CPM and return (summary, per-activity results).

    Phase 6 intentionally reuses the same CPM building blocks already used by
    the validated Phase 3 runner (scripts/run_cpm.py):

        - build_activity_index
        - build_dependencies
        - _topological_order
        - _make_results

    This reuse is necessary because the public ``calculate_project_cpm()``
    function returns a ``ProjectCpmSummary``, but does not expose the
    per-activity CPM details required by Phase 6, including:

        - total float
        - successor links
        - per-activity CPM results

    ``src/cpm/calculation.py`` is NOT modified. Phase 6 intentionally uses the
    same validated CPM calculation/result-building components and does not
    introduce a separate independent CPM engine.
    """
    summary = calculate_project_cpm(
        project_id,
        activity_rows,
        dependency_rows,
        activity_id_col="activity_id",
        duration_col="planned_duration_days",
        pred_col="predecessor_id",
        succ_col="successor_id",
        rel_col="relationship",
        lag_col="lag_days",
    )

    index = build_activity_index(activity_rows, project_id)
    edges = build_dependencies(dependency_rows, project_id)
    nodes = set(index.keys())

    preds_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    succs_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    edge_map: dict[tuple[ActivityRef, ActivityRef], list[Any]] = {}
    for e in edges:
        preds_by[e.successor].append(e.predecessor)
        succs_by[e.predecessor].append(e.successor)
        edge_map.setdefault((e.predecessor, e.successor), []).append(e)

    if not edges:
        project_duration = float(max(index.values()))
        es = {n: 0.0 for n in nodes}
        ef = {n: float(index[n]) for n in nodes}
        float_map = {n: project_duration - ef[n] for n in nodes}
        ls = {n: project_duration - float(index[n]) for n in nodes}
        lf = {n: float(project_duration) for n in nodes}
        results = _make_results(
            index,
            es,
            ef,
            ls,
            lf,
            float_map,
            preds_by=preds_by,
            succs_by=succs_by,
            project_duration=project_duration,
        )
        return summary, results

    order = _topological_order(nodes, edges)

    # Forward pass (mirrors the engine's documented math).
    es: dict[ActivityRef, float] = {n: 0.0 for n in nodes}
    ef: dict[ActivityRef, float] = {}
    for n in order:
        dur = float(index[n])
        bound = 0.0
        for pred in preds_by[n]:
            pred_dur = float(index[pred])
            for e in edge_map.get((pred, n), []):
                if e.relationship == "FS":
                    bound = max(bound, es[pred] + pred_dur + float(e.lag_days))
                elif e.relationship == "SS":
                    bound = max(bound, es[pred] + float(e.lag_days))
                elif e.relationship == "FF":
                    bound = max(bound, es[pred] + pred_dur + float(e.lag_days) - dur)
        es[n] = bound
        ef[n] = es[n] + dur

    terminal = [n for n in nodes if not succs_by[n]]
    if not terminal:
        terminal = [n for n in nodes if es[n] == max(es.values())]
    completion = (
        float(max(ef[n] for n in terminal)) if terminal else float(max(ef.values()))
    )

    lf: dict[ActivityRef, float] = {}
    ls: dict[ActivityRef, float] = {}
    for n in nodes:
        lf[n] = max(completion, ef[n])
        ls[n] = lf[n] - float(index[n])
        if ls[n] < es[n]:
            ls[n] = es[n]
            lf[n] = ls[n] + float(index[n])

    rev_order = list(reversed(order))
    changed = True
    while changed:
        changed = False
        for n in rev_order:
            cur_lf = lf[n]
            bound = float(completion)
            for succ in succs_by[n]:
                for e in edge_map.get((n, succ), []):
                    if e.relationship == "FS":
                        bound = min(bound, ls[succ] - float(e.lag_days))
                    elif e.relationship == "SS":
                        bound = min(
                            bound, ls[succ] - float(index[n]) - float(e.lag_days)
                        )
                    elif e.relationship == "FF":
                        bound = min(bound, lf[succ] - float(e.lag_days))
            new_lf = max(bound, ef[n])
            if new_lf < cur_lf - 1e-9:
                lf[n] = new_lf
                ls[n] = new_lf - float(index[n])
                if ls[n] < es[n]:
                    ls[n] = es[n]
                    lf[n] = ls[n] + float(index[n])
                changed = True

    float_map = {n: (ls[n] - es[n]) for n in nodes}
    results = _make_results(
        index,
        es,
        ef,
        ls,
        lf,
        float_map,
        preds_by=preds_by,
        succs_by=succs_by,
        project_duration=completion,
    )
    return summary, results


def _downstream_ids(
    results: dict[ActivityRef, Any],
    target: ActivityRef,
) -> list[str]:
    """All transitive successors of target, excluding target itself.

    Uses the CPM engine's own successor links; deterministic order by id.
    """
    seen: set[ActivityRef] = set()
    queue: deque[ActivityRef] = deque([target])
    while queue:
        cur = queue.popleft()
        for succ in results[cur].successors:
            if succ in seen:
                continue
            seen.add(succ)
            queue.append(succ)
    seen.discard(target)
    return sorted(a.activity_id for a in seen)


# ============================================================================
# Simulator
# ============================================================================


class WhatIfSimulator:
    """Deterministic what-if schedule simulator on the Phase 3 CPM engine.

    The simulator holds the full activity/dependency row lists (all projects)
    and scopes to a single project per scenario call. Input data is never
    mutated: each scenario deep-copies the rows it modifies.
    """

    def __init__(
        self,
        activity_rows: list[dict[str, Any]],
        dependency_rows: list[dict[str, Any]],
    ) -> None:
        if not isinstance(activity_rows, list) or not activity_rows:
            raise TypeError("activity_rows must be a non-empty list of row dicts")
        if not isinstance(dependency_rows, list):
            raise TypeError("dependency_rows must be a list of row dicts")

        self._activity_rows = copy.deepcopy(activity_rows)
        self._dependency_rows = copy.deepcopy(dependency_rows)

        self._projects: dict[str, list[dict[str, Any]]] = {}
        for row in self._activity_rows:
            pid = str(row.get("project_id", ""))
            if not pid:
                raise ValueError("activity row missing project_id")
            self._projects.setdefault(pid, []).append(row)

        self._deps_by_project: dict[str, list[dict[str, Any]]] = {}
        for row in self._dependency_rows:
            pid = str(row.get("project_id", ""))
            self._deps_by_project.setdefault(pid, []).append(row)

        self._project_ids = sorted(self._projects.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def project_ids(self) -> list[str]:
        return list(self._project_ids)

    def simulate_activity_delay(
        self,
        project_id: str,
        activity_id: str,
        delay_days: float,
    ) -> ScenarioResult:
        """Extend one activity's planned duration by delay_days (>= 0)."""
        if delay_days < 0:
            raise ValueError(f"delay_days must be >= 0, got {delay_days}")
        description = (
            f"Activity {activity_id} delayed by {delay_days} days "
            f"(planned duration extended)."
        )
        return self._run_scenario(
            project_id=project_id,
            activity_id=activity_id,
            scenario_type="activity_delay",
            scenario_description=description,
            delay_days=delay_days,
        )

    def simulate_weather_delay(
        self,
        project_id: str,
        activity_id: str,
        delay_days: float,
    ) -> ScenarioResult:
        """Weather delay represented as additional activity duration (>= 0)."""
        if delay_days < 0:
            raise ValueError(f"delay_days must be >= 0, got {delay_days}")
        description = (
            f"Weather delay: {delay_days} additional days added to "
            f"activity {activity_id}'s planned duration."
        )
        return self._run_scenario(
            project_id=project_id,
            activity_id=activity_id,
            scenario_type="weather_delay",
            scenario_description=description,
            delay_days=delay_days,
        )

    def simulate_resource_reduction(
        self,
        project_id: str,
        activity_id: str,
        reduction_percent: float,
    ) -> ScenarioResult:
        """Reduce resources -> longer duration (deterministic scenario assumption).

        new_duration = original_duration / (1 - reduction_percent / 100)

        This is a documented scenario assumption, not a learned construction law.
        """
        if reduction_percent <= 0:
            raise ValueError(f"reduction_percent must be > 0, got {reduction_percent}")
        if reduction_percent >= 100:
            raise ValueError(
                f"reduction_percent must be < 100, got {reduction_percent}"
            )

        description = (
            f"Resource reduction of {reduction_percent}% on activity {activity_id}: "
            f"new_duration = original_duration / (1 - {reduction_percent}/100) "
            f"(deterministic scenario assumption, not a learned construction law)."
        )

        delay_days = self._resource_reduction_delay(
            project_id, activity_id, reduction_percent
        )
        return self._run_scenario(
            project_id=project_id,
            activity_id=activity_id,
            scenario_type="resource_reduction",
            scenario_description=description,
            delay_days=delay_days,
            reduction_percent=reduction_percent,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resource_reduction_delay(
        self,
        project_id: str,
        activity_id: str,
        reduction_percent: float,
    ) -> float:
        rows = self._get_activity_rows(project_id)
        original = self._find_activity(rows, activity_id)["planned_duration_days"]
        original = float(original)
        if original <= 0:
            raise ValueError(
                f"Activity {activity_id!r} has non-positive planned duration "
                f"({original}); resource reduction scenario is undefined."
            )
        factor = 1.0 - reduction_percent / 100.0
        new_duration = original / factor
        return max(0.0, new_duration - original)

    def _get_activity_rows(self, project_id: str) -> list[dict[str, Any]]:
        try:
            return self._projects[project_id]
        except KeyError:
            raise KeyError(
                f"Unknown project_id {project_id!r}. "
                f"Known projects: {self._project_ids[:5]}{'...' if len(self._project_ids) > 5 else ''}"
            ) from None

    def _get_dependency_rows(self, project_id: str) -> list[dict[str, Any]]:
        return self._deps_by_project.get(project_id, [])

    @staticmethod
    def _find_activity(
        rows: list[dict[str, Any]],
        activity_id: str,
    ) -> dict[str, Any]:
        for row in rows:
            if str(row.get("activity_id")) == str(activity_id):
                return row
        raise KeyError(f"Activity {activity_id!r} not found in project")

    def _run_scenario(
        self,
        *,
        project_id: str,
        activity_id: str,
        scenario_type: str,
        scenario_description: str,
        delay_days: float,
        reduction_percent: float | None = None,
    ) -> ScenarioResult:
        base_rows = self._get_activity_rows(project_id)
        dep_rows = self._get_dependency_rows(project_id)
        self._find_activity(base_rows, activity_id)  # raises KeyError if missing

        target_ref = ActivityRef(project_id, str(activity_id))

        # ---- Baseline (on the unmodified originals) ----
        baseline_summary, baseline_results = _compute_cpm_details(
            project_id, base_rows, dep_rows
        )
        baseline_float = float(baseline_results[target_ref].total_float)
        baseline_downstream = _downstream_ids(baseline_results, target_ref)

        # ---- Scenario (deep copy, modify ONLY the one activity) ----
        scenario_rows = copy.deepcopy(base_rows)
        target_row = self._find_activity(scenario_rows, activity_id)
        original_duration = float(target_row["planned_duration_days"])
        # Durations stay on the integer-day grid the CPM engine is validated for;
        # fractional resource-reduction outcomes round to the nearest whole day.
        target_row["planned_duration_days"] = round(original_duration + delay_days)

        scenario_summary, scenario_results = _compute_cpm_details(
            project_id, scenario_rows, dep_rows
        )
        scenario_float = float(scenario_results[target_ref].total_float)

        # ---- Comparison ----
        baseline_crit = sorted(baseline_summary.critical_activity_ids)
        scenario_crit = sorted(scenario_summary.critical_activity_ids)
        newly_critical = sorted(set(scenario_crit) - set(baseline_crit))
        no_longer_critical = sorted(set(baseline_crit) - set(scenario_crit))

        cp_changed = (
            baseline_summary.critical_paths != scenario_summary.critical_paths
            or baseline_crit != scenario_crit
        )

        result = ScenarioResult(
            project_id=project_id,
            scenario_type=scenario_type,
            scenario_description=scenario_description,
            activity_id=str(activity_id),
            delay_days=float(delay_days),
            reduction_percent=reduction_percent,
            baseline_project_duration=float(baseline_summary.project_duration),
            scenario_project_duration=float(scenario_summary.project_duration),
            project_duration_delta=float(
                scenario_summary.project_duration - baseline_summary.project_duration
            ),
            baseline_critical_activity_ids=baseline_crit,
            scenario_critical_activity_ids=scenario_crit,
            baseline_critical_paths=[list(p) for p in baseline_summary.critical_paths],
            scenario_critical_paths=[list(p) for p in scenario_summary.critical_paths],
            critical_path_changed=cp_changed,
            downstream_activity_ids=baseline_downstream,
            baseline_float_days=baseline_float,
            scenario_float_days=scenario_float,
            float_consumed_days=max(0.0, baseline_float - scenario_float),
            newly_critical_activity_ids=newly_critical,
            no_longer_critical_activity_ids=no_longer_critical,
            baseline_has_cycles=bool(baseline_summary.has_cycles),
            scenario_has_cycles=bool(scenario_summary.has_cycles),
            baseline_dependency_violations=list(baseline_summary.dependency_violations),
            scenario_dependency_violations=list(scenario_summary.dependency_violations),
        )

        # Per-activity comparison detail (optional extra, useful for reports).
        comparisons: list[dict[str, Any]] = []
        for ref in sorted(baseline_results.keys(), key=lambda r: r.activity_id):
            b = baseline_results[ref]
            s = scenario_results[ref]
            comparisons.append(
                {
                    "activity_id": ref.activity_id,
                    "baseline_duration": b.duration,
                    "scenario_duration": s.duration,
                    "baseline_es": b.es,
                    "scenario_es": s.es,
                    "baseline_ef": b.ef,
                    "scenario_ef": s.ef,
                    "baseline_float": b.total_float,
                    "scenario_float": s.total_float,
                    "baseline_is_critical": b.is_critical,
                    "scenario_is_critical": s.is_critical,
                }
            )
        result.activity_comparisons = comparisons
        return result


# ============================================================================
# Loader for SCOPE processed data
# ============================================================================


def load_scope_rows(
    processed_dir: Path | str,
    project_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load one project's activity and dependency rows from data/processed."""
    import csv

    processed = Path(processed_dir)
    acts: list[dict[str, Any]] = []
    deps: list[dict[str, Any]] = []

    with open(processed / "activities.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["project_id"] == project_id:
                acts.append(row)

    with open(processed / "dependencies.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["project_id"] == project_id:
                deps.append(row)

    if not acts:
        raise KeyError(
            f"Project {project_id!r} not found in {processed / 'activities.csv'}"
        )
    return acts, deps
