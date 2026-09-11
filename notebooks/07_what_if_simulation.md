# Phase 6 — What-If Scenario Simulation

## 1. Purpose

Phase 6 adds a **deterministic what-if schedule simulation engine** on top of the
validated Phase 3 CPM engine (`src/cpm/calculation.py`).

Given an activity and a hypothetical modification (extra delay days or a
resource-driven duration change), the simulator:

1. Computes the **baseline CPM** for the project with the existing engine.
2. **Deep-copies** the project's activity rows.
3. Modifies **only** the selected activity's planned duration.
4. **Recomputes CPM** on the modified copy (real engine, same code path).
5. Compares baseline vs scenario and reports structured schedule impact.

**What this is NOT:**

- It does **NOT** predict actual future schedule slippage. It answers
  "what would the CPM schedule look like if this activity took N days longer?"
- It does **NOT** create a new ML probability or risk score. Phase 5
  (`src/risk/scoring.py`) owns risk scoring; Phase 6 contains **no ML and no
  probabilities**. No `risk_score` / `risk_level` fields are produced.
- It uses **no post-outcome information**: no events, no rework, no decision
  outcomes, no actual procurement, no environmental observations, no observed
  activity states, no `critical_path` / `critical_path_position` labels.
  All inputs are planned durations and dependency relationships.

## 2. Deterministic Scenario Methodology

Every scenario follows the identical deterministic pipeline:

```
load project rows -> baseline CPM (existing engine)
                  -> deep-copy activity rows
                  -> modify ONLY the target activity's planned_duration_days
                  -> scenario CPM (existing engine)
                  -> structured comparison (ScenarioResult)
```

- Same input -> same output, always. No randomness, no system-dependent ordering.
- The Phase 3 CPM engine is used **exactly as it exists** — `calculate_project_cpm`
  for the summary and the engine's own building blocks (`build_activity_index`,
  `build_dependencies`, `_topological_order`, `_make_results`) for per-activity
  float/successor detail, the same pattern the validated `scripts/run_cpm.py`
  uses. The engine file itself is untouched.

## 3. Scenario Types

### 3.1 Activity delay — `simulate_activity_delay(project_id, activity_id, delay_days)`

Extends the activity's planned duration by `delay_days`. Negative delays are
rejected; zero delay is allowed (useful as a no-op sanity check).

### 3.2 Resource reduction — `simulate_resource_reduction(project_id, activity_id, reduction_percent)`

Models "fewer resources -> longer duration" with the deterministic assumption:

```
new_duration = original_duration / (1 - reduction_percent / 100)
```

**This is a scenario assumption, NOT a learned construction law.** It is a
simple, transparent productivity heuristic used only to translate a resource
change into a duration change so the CPM engine can compute its real schedule
consequences. Values with `reduction_percent <= 0` or `>= 100` are rejected
(a 100% reduction would imply an infinitely long activity).

### 3.3 Weather delay — `simulate_weather_delay(project_id, activity_id, delay_days)`

Weather delay is represented as **additional activity duration** — mechanically
identical to `activity_delay`, kept as a separate scenario type for reporting
and traceability (a stakeholder reading the report should see *why* days were
added). Negative delays rejected, zero allowed.

## 4. Baseline vs Scenario CPM Comparison

For every scenario the engine is run twice — once on the original rows
(baseline) and once on the deep-copied, modified rows (scenario). Because the
modification touches only one activity's duration, any difference in the two
CPM results is attributable to that single change propagating through the real
dependency graph.

## 5. Float Consumption

For the modified activity:

- `baseline_float_days` — total float before the scenario.
- `scenario_float_days` — total float after the scenario.
- `float_consumed_days = max(0, baseline_float - scenario_float)`.

Float consumption shows how much of an activity's schedule slack a scenario
eats before the project completion date is affected. When the consumed float
exceeds the available float, `project_duration_delta` becomes positive.

## 6. Critical-Path Changes

Critical-path change detection compares:

- baseline vs scenario critical **activity ID sets** (via
  `critical_activity_ids` from the engine summary — criticality is never
  recomputed independently by Phase 6), and
- baseline vs scenario `critical_paths` sequences.

`critical_path_changed` is `True` if either differs.

## 7. Newly / No-Longer Critical Activities

- `newly_critical_activity_ids` — activities critical in the scenario but not
  in the baseline (e.g. a parallel branch whose float is exhausted and that
  then ties or overtakes the old critical path).
- `no_longer_critical_activity_ids` — activities critical in the baseline that
  lose criticality in the scenario (the old critical path is overtaken).

## 8. ScenarioResult Contract

At minimum each result carries: `project_id`, `scenario_type`,
`scenario_description`, `activity_id`, `delay_days`, `reduction_percent`,
`baseline_project_duration`, `scenario_project_duration`,
`project_duration_delta`, `baseline_critical_activity_ids`,
`scenario_critical_activity_ids`, `baseline_critical_paths`,
`scenario_critical_paths`, `critical_path_changed`,
`downstream_activity_ids`, `baseline_float_days`, `scenario_float_days`,
`float_consumed_days`, `newly_critical_activity_ids`,
`no_longer_critical_activity_ids`, `baseline_has_cycles`,
`scenario_has_cycles`, `baseline_dependency_violations`,
`scenario_dependency_violations`. A per-activity ES/EF/float comparison
(`activity_comparisons`) is included as an optional detail field.

Downstream analysis uses the CPM engine's own successor links and returns all
**transitive** successors excluding the modified activity itself.

## 9. Usage

```bash
# Deterministic demo (first project, first activity, +5 d / weather +3 d / -25% resources)
python scripts/run_scenarios.py

# Explicit scenario
python scripts/run_scenarios.py --project P00001 --activity P00001_A0002 --delay 5
python scripts/run_scenarios.py --scenario weather_delay --project P00001 --activity P00001_A0002 --delay 3
python scripts/run_scenarios.py --scenario resource_reduction --project P00001 --activity P00001_A0003 --reduction 25

# Batch mode
python scripts/run_scenarios.py --batch scenarios.json   # list of scenario specs
```

Outputs: `data/processed/scenarios/scenario_results.csv` and
`data/processed/scenarios/scenario_report.md`. `scripts/run_whatif.py` is a
thin alias of `run_scenarios.py`.

## 10. Tests

`tests/test_simulator.py` — 32 focused tests covering: simple-chain delay,
critical-activity delay, zero delay, negative-delay rejection, resource
reduction math, invalid reduction rejection, weather delay, missing
project/activity rejection, downstream (incl. transitive and terminal) checks,
float consumption (incl. clamping), newly/no-longer critical detection,
critical-path change detection, input non-mutation, real-engine value
cross-check, multi-project isolation, no-dependency graphs, serialization,
and the loader.

## 11. Limitations

- Durations are modified at **planned-duration** granularity; resource
  reduction is a **scenario assumption** (Section 3.2), not a calibrated model.
- Weather delay is represented as **additional activity duration** only — there
  is no weather probability model.
- The engine supports FS/SS/FF relationships with integer lags (Phase 3
  validated scope). No Start-to-Finish (SF) relationships.
- Results are schedule mathematics, not forecasts: a +5 day scenario says
  what the CPM schedule *would be*, not what *will* happen.
- Risk prioritization of *which* activity to stress-test should come from
  Phase 5 risk scores; Phase 6 deliberately consumes no risk scores and
  produces none.
