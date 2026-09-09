#!/usr/bin/env python3
"""
Run the deterministic CPM engine against all 100 SCOPE v0.2 projects.

Reads cleaned data from data/processed/ and writes CPM results to
data/processed/cpm/ (git-ignored).

Outputs:
  - data/processed/cpm/activity_cpm_results.csv  (activity-level CPM)
  - data/processed/cpm/project_cpm_summary.csv   (project-level summary)
  - data/processed/cpm/cpm_validation_report.md  (full validation report)
  - data/processed/cpm/critical_path_comparison.csv (dataset vs computed)
"""

from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cpm.calculation import (
    CyclicGraphError,
    calculate_project_cpm,
    detect_cycles,
)

PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED / "cpm"


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dicts (all values are strings)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run():
    print(f"[{datetime.now(tz=UTC):%H:%M:%S}] Loading cleaned data from {PROCESSED}")
    
    # Load all processed data
    activities_raw = load_csv(PROCESSED / "activities.csv")
    dependencies_raw = load_csv(PROCESSED / "dependencies.csv")
    projects_raw = load_csv(PROCESSED / "projects.csv")
    
    print(f"  activities: {len(activities_raw)} rows")
    print(f"  dependencies: {len(dependencies_raw)} rows")
    print(f"  projects: {len(projects_raw)} rows")
    
    # Get unique project IDs
    project_ids = sorted({row["project_id"] for row in projects_raw})
    print(f"  unique projects: {len(project_ids)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each project
    activity_results: list[dict] = []
    project_summaries: list[dict] = []
    total_violations = 0
    total_cycles = 0
    total_activities = 0
    total_edges = 0
    float_inconsistencies = 0
    
    # For critical path comparison
    dataset_critical = {}  # (project_id, activity_id) -> int
    computed_critical = {}  # (project_id, activity_id) -> bool
    
    for i, pid in enumerate(project_ids):
        print(f"  [{i+1:3d}/100] Processing {pid}...", end=" ", flush=True)
        
        # Filter rows for this project
        proj_activities = [r for r in activities_raw if r["project_id"] == pid]
        proj_deps = [r for r in dependencies_raw if r["project_id"] == pid]
        
        num_activities = len(proj_activities)
        num_edges = len(proj_deps)
        total_activities += num_activities
        total_edges += num_edges
        
        # Check for cycles first
        from cpm.calculation import ActivityRef, Dependency
        nodes = set()
        edges = []
        for act in proj_activities:
            nodes.add(ActivityRef(pid, act["activity_id"]))
        for dep in proj_deps:
            pred = ActivityRef(pid, dep["predecessor_id"])
            succ = ActivityRef(pid, dep["successor_id"])
            edges.append(Dependency(pred, succ, dep["relationship"].upper(), int(dep["lag_days"])))
        
        cycle_nodes = detect_cycles(nodes, edges)
        
        if cycle_nodes:
            total_cycles += 1
            print(f"CYCLE DETECTED ({len(cycle_nodes)} nodes)")
            continue
        
        # Run CPM
        try:
            summary = calculate_project_cpm(
                pid,
                proj_activities,
                proj_deps,
                activity_id_col="activity_id",
                duration_col="planned_duration_days",
                pred_col="predecessor_id",
                succ_col="successor_id",
                rel_col="relationship",
                lag_col="lag_days",
            )
        except CyclicGraphError as e:
            print(f"CYCLE: {e}")
            total_cycles += 1
            continue
        except (ValueError, KeyError, TypeError) as e:
            print(f"ERROR: invalid data: {e}")
            continue
        
        # Collect activity-level results
        # We need to re-run to get per-activity results
        # The summary doesn't include per-activity details, so we need to compute them
        
        # Build activity index and run CPM to get results
        import collections

        from cpm.calculation import (
            _make_results,
            _topological_order,
            build_activity_index,
            build_dependencies,
        )
        
        index = build_activity_index(proj_activities, pid)
        edges_out = build_dependencies(proj_deps, pid)
        
        nodes = set(index.keys())
        
        if not edges_out:
            # No dependencies case
            project_duration = float(max(index.values()))
            es = {n: 0.0 for n in nodes}
            ef = {n: float(index[n]) for n in nodes}
            float_map = {n: project_duration - ef[n] for n in nodes}
            ls = {n: project_duration - float(index[n]) for n in nodes}
            lf = {n: project_duration for n in nodes}
            preds_by = {n: [] for n in nodes}
            succs_by = {n: [] for n in nodes}
            
            results = _make_results(index, es, ef, ls, lf, float_map, 
                                    preds_by=preds_by, succs_by=succs_by,
                                    project_duration=project_duration)
            
            violations = []
        else:
            # Normal case with edges
            order = _topological_order(nodes, edges_out)
            
            preds_by = {n: [] for n in nodes}
            succs_by = {n: [] for n in nodes}
            edge_map = collections.defaultdict(list)
            for e in edges_out:
                preds_by[e.successor].append(e.predecessor)
                succs_by[e.predecessor].append(e.successor)
                edge_map[(e.predecessor, e.successor)].append(e)
            
            # Forward pass
            es = {n: 0.0 for n in nodes}
            ef = {}
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
                ef[n] = es[n] + dur            # Backward pass
            terminal_activities = [n for n in nodes if not succs_by[n]]
            if not terminal_activities:
                terminal_activities = [n for n in nodes if es[n] == max(es.values())]

            project_completion = float(max(ef[n] for n in terminal_activities)) if terminal_activities else float(max(ef.values()))

            # Initialize LF/LS from project completion, but never below EF/ES
            lf = {}
            ls = {}
            for n in nodes:
                # LF cannot be less than EF (otherwise float would be negative)
                lf[n] = max(project_completion, ef[n])
                ls[n] = lf[n] - float(index[n])
                # LS cannot be less than ES
                if ls[n] < es[n]:
                    ls[n] = es[n]
                    lf[n] = ls[n] + float(index[n])

            # Iterate backward to tighten, but never below EF/ES
            rev_order = list(reversed(order))
            changed = True
            while changed:
                changed = False
                for n in rev_order:
                    cur_lf = lf[n]
                    bound = float(project_completion)
                    for succ in succs_by[n]:
                        for e in edge_map.get((n, succ), []):
                            if e.relationship == "FS":
                                bound = min(bound, ls[succ] - float(e.lag_days))
                            elif e.relationship == "SS":
                                bound = min(bound, ls[succ] - float(index[n]) - float(e.lag_days))
                            elif e.relationship == "FF":
                                bound = min(bound, lf[succ] - float(e.lag_days))
                    new_lf = max(bound, ef[n])  # LF cannot be less than EF
                    if new_lf < cur_lf - 1e-9:
                        lf[n] = new_lf
                        ls[n] = new_lf - float(index[n])
                        if ls[n] < es[n]:  # LS cannot be less than ES
                            ls[n] = es[n]
                            lf[n] = ls[n] + float(index[n])
                        changed = True
            
            float_map = {n: (ls[n] - es[n]) for n in nodes}
            
            results = _make_results(index, es, ef, ls, lf, float_map,
                                    preds_by=preds_by, succs_by=succs_by,
                                    project_duration=project_completion)
            
            # Validate constraints
            from cpm.calculation import validate_dependency_constraints
            violations = validate_dependency_constraints(edges_out, es, ef, ls, lf, index)
        
        total_violations += len(violations)
        
        # Check float consistency
        for n, r in results.items():
            if r.is_virtual:
                continue
            ls_es = r.ls - r.es
            lf_ef = r.lf - r.ef
            if abs(ls_es - lf_ef) > 1e-6:
                float_inconsistencies += 1
        
        # Collect activity results
        for n, r in results.items():
            if r.is_virtual:
                continue
            
            activity_results.append({
                "project_id": r.ref.project_id,
                "activity_id": r.ref.activity_id,
                "duration": r.duration,
                "ES": round(r.es, 6),
                "EF": round(r.ef, 6),
                "LS": round(r.ls, 6),
                "LF": round(r.lf, 6),
                "total_float": round(r.total_float, 6),
                "computed_critical": 1 if r.is_critical else 0,
            })
            
            # Track for comparison
            key = (r.ref.project_id, r.ref.activity_id)
            computed_critical[key] = r.is_critical
            
            # Get dataset's critical_path value
            for act in proj_activities:
                if act["activity_id"] == r.ref.activity_id:
                    dataset_critical[key] = int(act.get("critical_path", 0))
                    break
        
        # Project summary
        project_summaries.append({
            "project_id": pid,
            "project_duration": round(summary.project_duration, 2),
            "num_activities": summary.num_activities,
            "num_edges": summary.num_edges,
            "num_terminal_activities": summary.num_terminal_activities,
            "num_critical_activities": summary.num_critical_activities,
            "critical_path": ";".join(summary.critical_paths[0]) if summary.critical_paths else "",
            "critical_activity_ids": ";".join(summary.critical_activity_ids),
        })
        
        print(f"OK (duration={summary.project_duration:.1f}d, critical={summary.num_critical_activities}/{summary.num_activities})")
    
    # Write activity-level results
    print("\nWriting activity-level CPM results...")
    with open(OUTPUT_DIR / "activity_cpm_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "project_id", "activity_id", "duration", "ES", "EF", "LS", "LF",
            "total_float", "computed_critical"
        ])
        writer.writeheader()
        writer.writerows(activity_results)
    print(f"  Written {len(activity_results)} activity records")
    
    # Write project-level summary
    print("Writing project-level CPM summary...")
    with open(OUTPUT_DIR / "project_cpm_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "project_id", "project_duration", "num_activities", "num_edges",
            "num_terminal_activities", "num_critical_activities",
            "critical_path", "critical_activity_ids"
        ])
        writer.writeheader()
        writer.writerows(project_summaries)
    print(f"  Written {len(project_summaries)} project records")
    
    # Write critical path comparison
    print("Writing critical path comparison...")
    comparison_results = []
    agreement_count = 0
    false_positives = 0
    false_negatives = 0
    
    for key, computed_val_bool in computed_critical.items():
        dataset_val = dataset_critical.get(key, 0)
        computed_val = 1 if computed_val_bool else 0
        
        if dataset_val == computed_val:
            agreement_count += 1
        elif computed_val == 1 and dataset_val == 0:
            false_positives += 1
        elif computed_val == 0 and dataset_val == 1:
            false_negatives += 1
        
        comparison_results.append({
            "project_id": key[0],
            "activity_id": key[1],
            "dataset_critical_path": dataset_val,
            "computed_critical": computed_val,
            "agree": 1 if dataset_val == computed_val else 0,
        })
    
    with open(OUTPUT_DIR / "critical_path_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "project_id", "activity_id", "dataset_critical_path",
            "computed_critical", "agree"
        ])
        writer.writeheader()
        writer.writerows(comparison_results)
    print(f"  Written {len(comparison_results)} comparison records")
    
    # Write validation report
    print("Writing validation report...")
    
    total_computed_critical = sum(1 for v in computed_critical.values() if v)
    total_dataset_critical = sum(dataset_critical.values())
    
    report = f"""# CPM Validation Report

**Generated:** {datetime.now(tz=UTC):%Y-%m-%d %H:%M:%S}
**Engine:** Deterministic CPM (NetworkX-free, Kahn's algorithm)

## Summary

| Metric | Value |
|--------|-------|
| Projects processed | {len(project_ids)} |
| Projects with cycles | {total_cycles} |
| Projects successfully processed | {len(project_summaries)} |
| Total activities | {total_activities} |
| Total dependency edges | {total_edges} |
| Dependency constraint violations | {total_violations} |
| Float inconsistencies (LS-ES ≠ LF-EF) | {float_inconsistencies} |

## Critical Path Comparison

| Metric | Value |
|--------|-------|
| Activities marked critical by dataset | {total_dataset_critical} |
| Activities marked critical by computed CPM | {total_computed_critical} |
| Agreement count | {agreement_count} |
| Agreement percentage | {100*agreement_count/len(computed_critical):.2f}% |
| False positives (computed=1, dataset=0) | {false_positives} |
| False negatives (computed=0, dataset=1) | {false_negatives} |

## Project Duration Statistics

| Statistic | Value |
|-----------|-------|
| Min project duration | {min(s['project_duration'] for s in project_summaries):.1f} days |
| Max project duration | {max(s['project_duration'] for s in project_summaries):.1f} days |
| Mean project duration | {sum(s['project_duration'] for s in project_summaries)/len(project_summaries):.1f} days |

## Notes

- This CPM calculation is based on planned durations and dependency relationships from the SCOPE v0.2 dataset.
- The dataset's `critical_path` field may use different criteria or conventions than this CPM engine.
- Discrepancies do not necessarily indicate errors in either calculation.
- This is deterministic scheduling logic, NOT an ML prediction of actual delays.
"""
    
    with open(OUTPUT_DIR / "cpm_validation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nDone! Output written to {OUTPUT_DIR}")
    print("\nQuick stats:")
    print(f"  Projects: {len(project_ids)} total, {len(project_summaries)} processed, {total_cycles} with cycles")
    print(f"  Activities: {total_activities}")
    print(f"  Dependencies: {total_edges}")
    print(f"  Violations: {total_violations}")
    print(f"  Critical (computed): {total_computed_critical}")
    print(f"  Critical (dataset): {total_dataset_critical}")
    print(f"  Agreement: {agreement_count}/{len(computed_critical)} ({100*agreement_count/len(computed_critical):.1f}%)")


if __name__ == "__main__":
    run()
