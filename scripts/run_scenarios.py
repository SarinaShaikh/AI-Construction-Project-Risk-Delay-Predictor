#!/usr/bin/env python3
"""
Phase 6 — what-if scenario simulation runner.

Deterministic schedule-impact simulation on top of the validated Phase 3 CPM
engine. Runs one or more scenarios against real SCOPE project data from
data/processed/ and writes results to data/processed/scenarios/.

Usage:
  # Deterministic demonstration default (first project, first activity):
  python scripts/run_scenarios.py

  # Explicit scenario (recommended for real use):
  python scripts/run_scenarios.py --project P00001 --activity P00001_A0002 --delay 5

  # Scenario type / extra options:
  python scripts/run_scenarios.py --scenario weather_delay --project P00001 --activity P00001_A0002 --delay 3
  python scripts/run_scenarios.py --scenario resource_reduction --project P00001 --activity P00001_A0003 --reduction 25

  # Batch: every scenario in a JSON spec file
  python scripts/run_scenarios.py --batch scenarios.json

The runner never mutates input data and never uses the Phase 4 test split;
this is pure deterministic schedule mathematics (no ML, no probabilities).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scenarios.simulator import (
    ScenarioResult,
    WhatIfSimulator,
    load_scope_rows,
)

PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED / "scenarios"


def _print_result(res: ScenarioResult) -> None:
    print(f"\n  scenario          : {res.scenario_type}")
    print(f"  description       : {res.scenario_description}")
    print(f"  baseline duration : {res.baseline_project_duration:g} days")
    print(f"  scenario duration : {res.scenario_project_duration:g} days")
    print(f"  duration delta    : {res.project_duration_delta:+g} days")
    print(f"  baseline float    : {res.baseline_float_days:g} days")
    print(f"  scenario float    : {res.scenario_float_days:g} days")
    print(f"  float consumed    : {res.float_consumed_days:g} days")
    print(f"  critical path changed : {res.critical_path_changed}")
    print(
        f"  downstream ({len(res.downstream_activity_ids)})      : {', '.join(res.downstream_activity_ids[:8]) or '-'}"
    )
    print(
        f"  newly critical ({len(res.newly_critical_activity_ids)})   : {', '.join(res.newly_critical_activity_ids[:8]) or '-'}"
    )
    print(
        f"  no longer critical ({len(res.no_longer_critical_activity_ids)}) : {', '.join(res.no_longer_critical_activity_ids[:8]) or '-'}"
    )
    print(
        f"  cycles baseline/scenario : {res.baseline_has_cycles}/{res.scenario_has_cycles}"
    )
    print(
        f"  dep violations baseline/scenario : {len(res.baseline_dependency_violations)}/{len(res.scenario_dependency_violations)}"
    )


def _result_row(res: ScenarioResult) -> dict[str, Any]:
    d = res.to_dict()
    # activity_comparisons is a nested list; keep the top-level row flat.
    d.pop("activity_comparisons", None)
    d["baseline_critical_activity_ids"] = ";".join(res.baseline_critical_activity_ids)
    d["scenario_critical_activity_ids"] = ";".join(res.scenario_critical_activity_ids)
    d["downstream_activity_ids"] = ";".join(res.downstream_activity_ids)
    d["newly_critical_activity_ids"] = ";".join(res.newly_critical_activity_ids)
    d["no_longer_critical_activity_ids"] = ";".join(res.no_longer_critical_activity_ids)
    d["baseline_critical_paths"] = json.dumps(res.baseline_critical_paths)
    d["scenario_critical_paths"] = json.dumps(res.scenario_critical_paths)
    return d


def run_scenarios(
    specs: list[dict[str, Any]], sim: WhatIfSimulator
) -> list[ScenarioResult]:
    """Execute a list of scenario specs: dicts with scenario_type + params."""
    results: list[ScenarioResult] = []
    for spec in specs:
        stype = spec["scenario_type"]
        pid, aid = spec["project_id"], spec["activity_id"]
        print(f"\n[{stype}] {pid} / {aid}")
        if stype == "activity_delay":
            res = sim.simulate_activity_delay(pid, aid, float(spec["delay_days"]))
        elif stype == "weather_delay":
            res = sim.simulate_weather_delay(pid, aid, float(spec["delay_days"]))
        elif stype == "resource_reduction":
            res = sim.simulate_resource_reduction(
                pid, aid, float(spec["reduction_percent"])
            )
        else:
            raise ValueError(f"Unknown scenario_type {stype!r}")
        _print_result(res)
        results.append(res)
    return results


def demo_default(sim: WhatIfSimulator) -> list[ScenarioResult]:
    """Deterministic demonstration: first project, first activity, 5-day delay."""
    pid = sim.project_ids[0]
    rows, _ = load_scope_rows(PROCESSED, pid)
    aid = str(rows[0]["activity_id"])
    print("=== Phase 6 deterministic demonstration (activity_delay +5 days) ===")
    print(f"project={pid} activity={aid}")
    return [
        sim.simulate_activity_delay(pid, aid, 5),
        sim.simulate_weather_delay(pid, aid, 3),
        sim.simulate_resource_reduction(pid, aid, 25),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 what-if scenario runner")
    parser.add_argument("--project", help="Project ID (e.g. P00001)")
    parser.add_argument("--activity", help="Activity ID (e.g. P00001_A0002)")
    parser.add_argument("--delay", type=float, default=None, help="Delay in days")
    parser.add_argument(
        "--reduction", type=float, default=None, help="Resource reduction %%"
    )
    parser.add_argument(
        "--scenario",
        choices=["activity_delay", "weather_delay", "resource_reduction"],
        default="activity_delay",
    )
    parser.add_argument(
        "--batch", help="Path to JSON file with a list of scenario specs"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run the deterministic demo default"
    )
    args = parser.parse_args()

    import csv

    # Discover all project IDs from activities.csv.
    with open(PROCESSED / "activities.csv", newline="", encoding="utf-8") as f:
        project_ids = sorted({row["project_id"] for row in csv.DictReader(f)})

    # The simulator holds ALL projects' rows so scenarios can target any project.
    with open(PROCESSED / "activities.csv", newline="", encoding="utf-8") as f:
        all_activity_rows = list(csv.DictReader(f))
    with open(PROCESSED / "dependencies.csv", newline="", encoding="utf-8") as f:
        all_dependency_rows = list(csv.DictReader(f))

    sim = WhatIfSimulator(
        activity_rows=all_activity_rows, dependency_rows=all_dependency_rows
    )

    if args.batch:
        with open(args.batch, encoding="utf-8") as f:
            specs = json.load(f)
        results = run_scenarios(specs, sim)
    elif args.demo or (args.project is None and args.activity is None):
        results = demo_default(sim)
    else:
        if args.project not in project_ids:
            print(
                f"ERROR: unknown project {args.project!r}; known: {project_ids[:5]}...",
                file=sys.stderr,
            )
            return 2
        if args.scenario == "resource_reduction":
            if args.reduction is None:
                print(
                    "ERROR: --reduction required for resource_reduction",
                    file=sys.stderr,
                )
                return 2
            specs = [
                {
                    "scenario_type": args.scenario,
                    "project_id": args.project,
                    "activity_id": args.activity,
                    "reduction_percent": args.reduction,
                }
            ]
        else:
            if args.delay is None:
                print(f"ERROR: --delay required for {args.scenario}", file=sys.stderr)
                return 2
            specs = [
                {
                    "scenario_type": args.scenario,
                    "project_id": args.project,
                    "activity_id": args.activity,
                    "delay_days": args.delay,
                }
            ]
        results = run_scenarios(specs, sim)

    # ---- Persist results ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "scenario_results.csv"
    fieldnames = [
        "project_id",
        "scenario_type",
        "scenario_description",
        "activity_id",
        "delay_days",
        "reduction_percent",
        "baseline_project_duration",
        "scenario_project_duration",
        "project_duration_delta",
        "critical_path_changed",
        "baseline_float_days",
        "scenario_float_days",
        "float_consumed_days",
        "baseline_critical_activity_ids",
        "scenario_critical_activity_ids",
        "baseline_critical_paths",
        "scenario_critical_paths",
        "downstream_activity_ids",
        "newly_critical_activity_ids",
        "no_longer_critical_activity_ids",
        "baseline_has_cycles",
        "scenario_has_cycles",
        "baseline_dependency_violations",
        "scenario_dependency_violations",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for res in results:
            w.writerow(_result_row(res))

    # Human-readable report
    out_md = OUTPUT_DIR / "scenario_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Phase 6 — What-If Scenario Run Report\n\n")
        f.write(f"**Generated:** {datetime.now(tz=UTC).isoformat()}\n\n")
        f.write("Deterministic CPM-based schedule simulation. No ML probabilities, ")
        f.write("no risk scores, no post-outcome information.\n\n")
        for res in results:
            f.write(f"## {res.scenario_type}: {res.project_id} / {res.activity_id}\n\n")
            f.write(f"- {res.scenario_description}\n")
            f.write(
                f"- Baseline project duration: **{res.baseline_project_duration:g} d**\n"
            )
            f.write(
                f"- Scenario project duration: **{res.scenario_project_duration:g} d**\n"
            )
            f.write(
                f"- Project duration delta: **{res.project_duration_delta:+g} d**\n"
            )
            f.write(f"- Float consumed on activity: **{res.float_consumed_days:g} d** ")
            f.write(f"({res.baseline_float_days:g} -> {res.scenario_float_days:g})\n")
            f.write(f"- Critical path changed: **{res.critical_path_changed}**\n")
            f.write(f"- Baseline critical path: {res.baseline_critical_paths}\n")
            f.write(f"- Scenario critical path: {res.scenario_critical_paths}\n")
            f.write(f"- Downstream activities ({len(res.downstream_activity_ids)}): ")
            f.write(f"{', '.join(res.downstream_activity_ids) or '-'}\n")
            f.write(
                f"- Newly critical: {', '.join(res.newly_critical_activity_ids) or '-'}\n"
            )
            f.write(
                f"- No longer critical: {', '.join(res.no_longer_critical_activity_ids) or '-'}\n\n"
            )

    print(f"\nPHASE 6 SCENARIO RUN COMPLETE — {len(results)} scenario(s)")
    print(f"  results CSV : {out_csv}")
    print(f"  report      : {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
