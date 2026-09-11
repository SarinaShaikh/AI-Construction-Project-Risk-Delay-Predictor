"""
Phase 6 — what-if scenario simulator unit tests (src/scenarios/simulator.py).

Covers the 15 required cases:
 1.  activity delay on a simple A -> B -> C schedule
 2.  delay of a critical activity increases project duration
 3.  zero delay
 4.  negative delay rejected
 5.  resource reduction
 6.  invalid resource reduction rejected
 7.  weather delay
 8.  missing project rejected
 9.  missing activity rejected
 10. downstream activity detection
 11. float consumption
 12. newly critical activity detection
 13. critical path change detection
 14. scenario does not mutate original input data
 15. CPM engine integration (real engine values match summary)

Run with:  .venv/Scripts/python.exe -m pytest tests/test_simulator.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scenarios.simulator import (
    ScenarioResult,
    WhatIfSimulator,
    load_scope_rows,
)

# ---------------------------------------------------------------------------
# Helpers: tiny deterministic fixture graphs
# ---------------------------------------------------------------------------


def act(project_id: str, activity_id: str, duration: int) -> dict[str, object]:
    return {
        "project_id": project_id,
        "activity_id": activity_id,
        "planned_duration_days": duration,
    }


def dep(
    project_id: str,
    predecessor: str,
    successor: str,
    relationship: str = "FS",
    lag_days: int = 0,
) -> dict[str, object]:
    return {
        "project_id": project_id,
        "predecessor_id": predecessor,
        "successor_id": successor,
        "relationship": relationship,
        "lag_days": lag_days,
    }


@pytest.fixture()
def serial_abc() -> WhatIfSimulator:
    """A(2) -> B(3) -> C(4): serial chain, all critical, duration 9."""
    return WhatIfSimulator(
        activity_rows=[act("P", "A", 2), act("P", "B", 3), act("P", "C", 4)],
        dependency_rows=[dep("P", "A", "B"), dep("P", "B", "C")],
    )


@pytest.fixture()
def parallel_with_float() -> WhatIfSimulator:
    """A(2) -> B(3) and A(2) -> C(5): C critical, B has float 2, duration 7."""
    return WhatIfSimulator(
        activity_rows=[act("P", "A", 2), act("P", "B", 3), act("P", "C", 5)],
        dependency_rows=[dep("P", "A", "B"), dep("P", "A", "C")],
    )


# ---------------------------------------------------------------------------
# 1. Activity delay on a simple A -> B -> C schedule
# ---------------------------------------------------------------------------


def test_activity_delay_simple_chain(serial_abc: WhatIfSimulator) -> None:
    res = serial_abc.simulate_activity_delay("P", "B", 5)
    assert isinstance(res, ScenarioResult)
    assert res.project_id == "P"
    assert res.scenario_type == "activity_delay"
    assert res.activity_id == "B"
    assert res.delay_days == 5.0
    assert res.baseline_project_duration == pytest.approx(9.0)
    assert res.scenario_project_duration == pytest.approx(14.0)
    assert res.project_duration_delta == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 2. Delay of a critical activity increases project duration
# ---------------------------------------------------------------------------


def test_delay_critical_activity_increases_duration(
    serial_abc: WhatIfSimulator,
) -> None:
    res = serial_abc.simulate_activity_delay("P", "A", 3)
    assert res.project_duration_delta == pytest.approx(3.0)
    assert res.scenario_project_duration > res.baseline_project_duration
    # A stays critical in the scenario
    assert "A" in res.scenario_critical_activity_ids


def test_delay_noncritical_within_float_no_duration_change(
    parallel_with_float: WhatIfSimulator,
) -> None:
    # B has float 2; a 1-day delay must NOT move project duration.
    res = parallel_with_float.simulate_activity_delay("P", "B", 1)
    assert res.project_duration_delta == pytest.approx(0.0)
    assert res.float_consumed_days == pytest.approx(1.0)
    assert res.baseline_float_days == pytest.approx(2.0)
    assert res.scenario_float_days == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Zero delay
# ---------------------------------------------------------------------------


def test_zero_delay_allowed(serial_abc: WhatIfSimulator) -> None:
    res = serial_abc.simulate_activity_delay("P", "B", 0)
    assert res.delay_days == 0.0
    assert res.project_duration_delta == pytest.approx(0.0)
    assert res.float_consumed_days == pytest.approx(0.0)
    assert res.critical_path_changed is False
    assert res.baseline_project_duration == res.scenario_project_duration


# ---------------------------------------------------------------------------
# 4. Negative delay rejected
# ---------------------------------------------------------------------------


def test_negative_delay_rejected(serial_abc: WhatIfSimulator) -> None:
    with pytest.raises(ValueError, match="delay_days must be >= 0"):
        serial_abc.simulate_activity_delay("P", "B", -1)


def test_negative_weather_delay_rejected(serial_abc: WhatIfSimulator) -> None:
    with pytest.raises(ValueError, match="delay_days must be >= 0"):
        serial_abc.simulate_weather_delay("P", "B", -2)


# ---------------------------------------------------------------------------
# 5. Resource reduction
# ---------------------------------------------------------------------------


def test_resource_reduction(serial_abc: WhatIfSimulator) -> None:
    # B(3) with 50% reduction -> new duration = 3 / 0.5 = 6 -> delay 3
    res = serial_abc.simulate_resource_reduction("P", "B", 50)
    assert res.scenario_type == "resource_reduction"
    assert res.reduction_percent == 50.0
    assert res.delay_days == pytest.approx(3.0)
    assert res.scenario_project_duration == pytest.approx(12.0)
    # The scenario duration for B must reflect the assumption: 6 days
    b_row = next(c for c in res.activity_comparisons if c["activity_id"] == "B")
    assert b_row["scenario_duration"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 6. Invalid resource reduction rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -5, 100, 150])
def test_invalid_resource_reduction_rejected(
    serial_abc: WhatIfSimulator, bad: float
) -> None:
    with pytest.raises(ValueError, match="reduction_percent"):
        serial_abc.simulate_resource_reduction("P", "B", bad)


# ---------------------------------------------------------------------------
# 7. Weather delay
# ---------------------------------------------------------------------------


def test_weather_delay(parallel_with_float: WhatIfSimulator) -> None:
    res = parallel_with_float.simulate_weather_delay("P", "C", 4)
    assert res.scenario_type == "weather_delay"
    assert res.project_duration_delta == pytest.approx(4.0)
    assert res.scenario_project_duration == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# 8. Missing project rejected
# ---------------------------------------------------------------------------


def test_missing_project_rejected(serial_abc: WhatIfSimulator) -> None:
    with pytest.raises(KeyError, match="Unknown project_id"):
        serial_abc.simulate_activity_delay("NOPE", "A", 1)


def test_missing_project_resource_reduction_rejected(
    serial_abc: WhatIfSimulator,
) -> None:
    with pytest.raises(KeyError, match="Unknown project_id"):
        serial_abc.simulate_resource_reduction("NOPE", "A", 10)


# ---------------------------------------------------------------------------
# 9. Missing activity rejected
# ---------------------------------------------------------------------------


def test_missing_activity_rejected(serial_abc: WhatIfSimulator) -> None:
    with pytest.raises(KeyError, match="not found"):
        serial_abc.simulate_activity_delay("P", "Z", 1)


def test_missing_activity_resource_reduction_rejected(
    serial_abc: WhatIfSimulator,
) -> None:
    with pytest.raises(KeyError, match="not found"):
        serial_abc.simulate_resource_reduction("P", "Z", 10)


# ---------------------------------------------------------------------------
# 10. Downstream activity detection
# ---------------------------------------------------------------------------


def test_downstream_detection(serial_abc: WhatIfSimulator) -> None:
    res = serial_abc.simulate_activity_delay("P", "A", 1)
    # A's downstream = B and C (transitive), excluding A itself.
    assert res.downstream_activity_ids == ["B", "C"]


def test_downstream_terminal_empty(serial_abc: WhatIfSimulator) -> None:
    res = serial_abc.simulate_activity_delay("P", "C", 1)
    assert res.downstream_activity_ids == []


def test_downstream_transitive_parallel(parallel_with_float: WhatIfSimulator) -> None:
    res = parallel_with_float.simulate_activity_delay("P", "A", 1)
    assert res.downstream_activity_ids == ["B", "C"]


# ---------------------------------------------------------------------------
# 11. Float consumption
# ---------------------------------------------------------------------------


def test_float_consumption(parallel_with_float: WhatIfSimulator) -> None:
    res = parallel_with_float.simulate_activity_delay("P", "B", 2)
    assert res.baseline_float_days == pytest.approx(2.0)
    assert res.scenario_float_days == pytest.approx(0.0)
    assert res.float_consumed_days == pytest.approx(2.0)


def test_float_consumed_never_negative(parallel_with_float: WhatIfSimulator) -> None:
    # A delay of 5 on B exceeds its float: float consumed clamps at 0 diff,
    # scenario float cannot go below 0.
    res = parallel_with_float.simulate_activity_delay("P", "B", 5)
    assert res.scenario_float_days == pytest.approx(0.0)
    assert res.float_consumed_days == pytest.approx(2.0)
    # Project duration now moves: B becomes part of the critical chain.
    assert res.project_duration_delta == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 12. Newly critical activity detection
# ---------------------------------------------------------------------------


def test_newly_critical_detection(parallel_with_float: WhatIfSimulator) -> None:
    # Delaying B(3) by 3 days makes B critical (was float 2, path ties at 7... )
    # A(2) -> B(6) => EF_B = 8 > EF_C = 7, so B becomes critical and duration 8.
    res = parallel_with_float.simulate_activity_delay("P", "B", 3)
    assert "B" in res.newly_critical_activity_ids
    assert "B" in res.scenario_critical_activity_ids
    assert "B" not in res.baseline_critical_activity_ids
    assert res.project_duration_delta == pytest.approx(1.0)


def test_no_longer_critical_detection(parallel_with_float: WhatIfSimulator) -> None:
    # Delay B by 5: B path = 2+8 = 10 dominates; C(5) no longer critical.
    res = parallel_with_float.simulate_activity_delay("P", "B", 5)
    assert "C" in res.no_longer_critical_activity_ids
    assert "B" in res.newly_critical_activity_ids
    assert res.project_duration_delta == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 13. Critical path change detection
# ---------------------------------------------------------------------------


def test_critical_path_change_detected(parallel_with_float: WhatIfSimulator) -> None:
    res = parallel_with_float.simulate_activity_delay("P", "B", 5)
    assert res.critical_path_changed is True
    assert res.baseline_critical_paths == [["A", "C"]]
    assert res.scenario_critical_paths == [["A", "B"]]


def test_critical_path_unchanged_for_zero_delay(serial_abc: WhatIfSimulator) -> None:
    res = serial_abc.simulate_activity_delay("P", "A", 0)
    assert res.critical_path_changed is False
    assert res.baseline_critical_paths == [["A", "B", "C"]]
    assert res.scenario_critical_paths == [["A", "B", "C"]]


# ---------------------------------------------------------------------------
# 14. Scenario does not mutate original input data
# ---------------------------------------------------------------------------


def test_no_input_mutation(serial_abc: WhatIfSimulator) -> None:
    original_rows = [act("P", "A", 2), act("P", "B", 3), act("P", "C", 4)]
    original_deps = [dep("P", "A", "B"), dep("P", "B", "C")]

    sim = WhatIfSimulator(activity_rows=original_rows, dependency_rows=original_deps)
    sim.simulate_activity_delay("P", "B", 10)

    assert original_rows == [act("P", "A", 2), act("P", "B", 3), act("P", "C", 4)]
    assert original_deps == [dep("P", "A", "B"), dep("P", "B", "C")]
    # Re-running the same scenario gives the same answer (stateless per call).
    res2 = sim.simulate_activity_delay("P", "B", 10)
    assert res2.scenario_project_duration == pytest.approx(19.0)


def test_constructor_deep_copies_inputs() -> None:
    rows = [act("P", "A", 2)]
    deps: list[dict[str, object]] = []
    sim = WhatIfSimulator(activity_rows=rows, dependency_rows=deps)
    sim.simulate_activity_delay("P", "A", 7)
    assert rows[0]["planned_duration_days"] == 2


# ---------------------------------------------------------------------------
# 15. CPM engine integration
# ---------------------------------------------------------------------------


def test_uses_real_cpm_engine_values(serial_abc: WhatIfSimulator) -> None:
    """Cross-check the simulator against calculate_project_cpm directly."""
    from cpm.calculation import calculate_project_cpm

    rows = [act("P", "A", 2), act("P", "B", 3), act("P", "C", 4)]
    deps = [dep("P", "A", "B"), dep("P", "B", "C")]
    summary = calculate_project_cpm("P", rows, deps)

    res = serial_abc.simulate_activity_delay("P", "B", 0)
    assert res.baseline_project_duration == pytest.approx(summary.project_duration)
    assert res.baseline_critical_activity_ids == sorted(summary.critical_activity_ids)


def test_multi_project_isolation() -> None:
    sim = WhatIfSimulator(
        activity_rows=[act("P1", "A", 2), act("P2", "A", 2)],
        dependency_rows=[],
    )
    assert sim.project_ids == ["P1", "P2"]
    res = sim.simulate_activity_delay("P1", "A", 3)
    assert res.baseline_project_duration == pytest.approx(2.0)
    assert res.scenario_project_duration == pytest.approx(5.0)
    # P2 untouched
    res2 = sim.simulate_activity_delay("P2", "A", 0)
    assert res2.baseline_project_duration == pytest.approx(2.0)


def test_no_dependencies_scenario() -> None:
    sim = WhatIfSimulator(
        activity_rows=[act("P", "A", 2), act("P", "B", 3)], dependency_rows=[]
    )
    res = sim.simulate_activity_delay("P", "B", 4)
    # No deps: duration = max(2, 7) = 7; baseline = 3.
    assert res.baseline_project_duration == pytest.approx(3.0)
    assert res.scenario_project_duration == pytest.approx(7.0)
    assert res.project_duration_delta == pytest.approx(4.0)


def test_result_to_dict_serializable(serial_abc: WhatIfSimulator) -> None:
    res = serial_abc.simulate_activity_delay("P", "B", 2)
    d = res.to_dict()
    assert d["project_id"] == "P"
    assert d["scenario_type"] == "activity_delay"
    assert d["project_duration_delta"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# load_scope_rows loader (used by scripts; graceful on missing files)
# ---------------------------------------------------------------------------


def test_load_scope_rows_missing_project(tmp_path: Path) -> None:
    import csv

    with open(tmp_path / "activities.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["project_id", "activity_id", "planned_duration_days"])
        w.writerow(["P1", "A", 2])
    with open(tmp_path / "dependencies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["project_id", "predecessor_id", "successor_id", "relationship", "lag_days"]
        )

    with pytest.raises(KeyError, match="not found"):
        load_scope_rows(tmp_path, "MISSING")
