#!/usr/bin/env python3
"""
SCOPE CPM logic feasibility check (read-only, no data modification).

For each project:
- Load activities + dependencies
- Build DAG adjacency (pred -> succ)
- Identify roots (no predecessors) and leaves (no successors)
- Attempt a simplified forward pass assuming each activity starts at ES=0,
  EF = ES + planned_duration, successors ES = max(EF + lag) of predecessors.
- Check: is there a SINGLE root / SINGLE leaf per project (typical of a
  single planned start/end)?
- Check: do critical_path flags appear consistent with float==0?
- Report any projects where the dependency structure suggests multiple
  entry/exit points (parallel start/end) that a CPM engine would need to
  handle explicitly.

NOTE: This is a structural sanity check, not a full CPM implementation.
"""
import csv, collections, os, sys

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

def read_csv(name):
    with open(os.path.join(ROOT, f"{name}.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        rows = [dict(zip(h, row)) for row in r]
    return h, rows

def main():
    print("SCOPE v0.2 — CPM graph structure feasibility check")
    print("="*72)

    _, activities = read_csv("activities")
    _, deps = read_csv("dependencies")

    # group by project
    proj_acts = collections.defaultdict(list)
    for a in activities:
        proj_acts[a["project_id"]].append(a)

    proj_deps = collections.defaultdict(list)
    for d in deps:
        proj_deps[d["project_id"]].append(d)

    # stats
    single_root_projects = 0
    multi_root_projects = 0
    single_leaf_projects = 0
    multi_leaf_projects = 0
    no_root_projects = 0
    no_leaf_projects = 0
    proj_results = []

    for pid in sorted(proj_acts.keys()):
        acts = proj_acts[pid]
        adict = {a["activity_id"]: a for a in acts}
        act_ids = set(adict.keys())

        # predecessors / successors per activity
        preds = collections.defaultdict(list)
        succs = collections.defaultdict(list)
        for d in proj_deps[pid]:
            preds[d["successor_id"]].append(d)
            succs[d["predecessor_id"]].append(d)

        roots = [aid for aid in act_ids if not preds[aid]]
        leaves = [aid for aid in act_ids if not succs[aid]]

        n_preds_missing = 0
        n_succ_missing = 0
        for aid in act_ids:
            for d in preds[aid]:
                if d["predecessor_id"] not in act_ids:
                    n_preds_missing += 1
            for d in succs[aid]:
                if d["successor_id"] not in act_ids:
                    n_succ_missing += 1

        # count critical activities according to dataset's critical_path flag
        n_crit = sum(1 for a in acts if a["critical_path"] == "1")
        n_total = len(acts)

        if not roots:
            no_root_projects += 1
        elif len(roots) == 1:
            single_root_projects += 1
        else:
            multi_root_projects += 1

        if not leaves:
            no_leaf_projects += 1
        elif len(leaves) == 1:
            single_leaf_projects += 1
        else:
            multi_leaf_projects += 1

        proj_results.append({
            "pid": pid,
            "n_acts": n_total,
            "n_deps": len(proj_deps[pid]),
            "n_crit": n_crit,
            "roots": len(roots),
            "leaves": len(leaves),
            "roots_sample": roots[:3],
            "leaves_sample": leaves[:3],
            "pred_missing_refs": n_preds_missing,
            "succ_missing_refs": n_succ_missing,
            "crit_pct": 100.0 * n_crit / n_total if n_total else 0,
        })

    print(f"Total projects: {len(proj_acts)}")
    print(f"  projects with a SINGLE root (no predecessors): {single_root_projects}")
    print(f"  projects with MULTIPLE roots: {multi_root_projects}")
    print(f"  projects with NO roots: {no_root_projects}")
    print(f"  projects with a SINGLE leaf (no successors): {single_leaf_projects}")
    print(f"  projects with MULTIPLE leaves: {multi_leaf_projects}")
    print(f"  projects with NO leaves: {no_leaf_projects}")
    print(f"  projects with missing dependency references: "
          f"{sum(1 for p in proj_results if p['pred_missing_refs'] or p['succ_missing_refs'])}")

    # show a few multi-root / multi-leaf examples
    print()
    print("Sample MULTI-ROOT projects (first 5):")
    for p in [p for p in proj_results if p["roots"] > 1][:5]:
        print(f"  {p['pid']}: {p['n_acts']} activities, {p['n_deps']} deps, "
              f"{p['roots']} roots: {p['roots_sample']}, "
              f"{p['leaves']} leaves: {p['leaves_sample']}, "
              f"crit={p['n_crit']}/{p['n_acts']} ({p['crit_pct']:.1f}%)")

    print()
    print("Sample MULTI-LEAF projects (first 5):")
    for p in [p for p in proj_results if p["leaves"] > 1][:5]:
        print(f"  {p['pid']}: {p['n_acts']} activities, {p['n_deps']} deps, "
              f"{p['roots']} roots: {p['roots_sample']}, "
              f"{p['leaves']} leaves: {p['leaves_sample']}, "
              f"crit={p['n_crit']}/{p['n_acts']} ({p['crit_pct']:.1f}%)")

    # critical_path_position sanity: for activities flagged critical, what's the
    # distribution of critical_path_position? (should be >=1 for critical ones)
    print()
    print("Critical-path-position sanity for CRITICAL activities (critical_path==1):")
    crit_positions = collections.Counter()
    for a in activities:
        if a["critical_path"] == "1":
            crit_positions[int(a["critical_path_position"])] += 1
    print(f"  distinct critical_path_position values observed: {len(crit_positions)}")
    print(f"  total critical activities: {sum(crit_positions.values())}")
    if -1 in crit_positions:
        print(f"  WARNING: {crit_positions[-1]} critical activities have critical_path_position == -1")
    print(f"  top positions: {crit_positions.most_common(10)}")

    # critical_path_position for non-critical activities
    noncrit_positions = collections.Counter()
    for a in activities:
        if a["critical_path"] == "0":
            noncrit_positions[int(a["critical_path_position"])] += 1
    print()
    print("Critical-path-position for NON-CRITICAL activities (critical_path==0):")
    print(f"  distinct values: {len(noncrit_positions)}")
    print(f"  total non-critical activities: {sum(noncrit_positions.values())}")
    print(f"  top values: {noncrit_positions.most_common(10)}")

    # Distribution of project_network_duration values across activities (should be
    # constant per project → same for all activities within a project)
    print()
    print("project_network_duration consistency check (one value per project expected):")
    pnd_per_project = collections.defaultdict(set)
    for a in activities:
        pnd_per_project[a["project_id"]].add(int(a["project_network_duration"]))
    inconsistent = {p: v for p, v in pnd_per_project.items() if len(v) > 1}
    print(f"  projects with >1 distinct project_network_duration: {len(inconsistent)}")
    if inconsistent:
        for p, vals in list(inconsistent.items())[:5]:
            print(f"    {p}: {sorted(vals)}")

    # Check that predecessor_count / successor_count in activities matches actual counts
    print()
    print("Predecessor/successor COUNT consistency (activities summary fields vs actual deps):")
    actual_pred_counts = collections.defaultdict(int)
    actual_succ_counts = collections.defaultdict(int)
    for d in deps:
        actual_pred_counts[(d["project_id"], d["successor_id"])] += 1  # succ has this many preds
        actual_succ_counts[(d["project_id"], d["predecessor_id"])] += 1  # pred has this many succs
    mismatches = 0
    for a in activities:
        key = (a["project_id"], a["activity_id"])
        actual_pred = actual_pred_counts.get(key, 0)
        actual_succ = actual_succ_counts.get(key, 0)
        if actual_pred != int(a["predecessor_count"]):
            mismatches += 1
            if mismatches <= 5:
                print(f"  {a['activity_id']}: stated pred_count={a['predecessor_count']} actual={actual_pred}")
        if actual_succ != int(a["successor_count"]):
            mismatches += 1
            if mismatches <= 5:
                print(f"  {a['activity_id']}: stated succ_count={a['successor_count']} actual={actual_succ}")
    print(f"  total count mismatches: {mismatches}")

    # Lag values by relationship type
    print()
    print("Lag_days by relationship type:")
    lag_by_rel = collections.defaultdict(list)
    for d in deps:
        lag_by_rel[d["relationship"]].append(int(d["lag_days"]))
    for rel in ["FS", "FF", "SS"]:
        vals = lag_by_rel.get(rel, [])
        if vals:
            print(f"  {rel}: n={len(vals)} min={min(vals)} max={max(vals)} "
                  f"mean={sum(vals)/len(vals):.2f} negatives={sum(1 for v in vals if v<0)}")
        else:
            print(f"  {rel}: no records")

    print()
    print("CPM GRAPH FEASIBILITY SUMMARY:")
    print(f"  - All dependency references are intra-project: {sum(1 for p in proj_results if not p['pred_missing_refs'] and not p['succ_missing_refs']) == len(proj_results)}")
    print(f"  - Projects with single entry + single exit (typical textbook CPM): "
          f"{sum(1 for p in proj_results if p['roots']==1 and p['leaves']==1)}")
    print(f"  - Projects with multiple roots/leaves that need explicit handling: "
          f"{sum(1 for p in proj_results if p['roots']>1 or p['leaves']>1)}")
    print("DONE")

if __name__ == "__main__":
    main()
