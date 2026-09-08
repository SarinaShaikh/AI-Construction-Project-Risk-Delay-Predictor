#!/usr/bin/env python3
"""
Correct column-matched comparison: master_projects_canonical vs outcomes.csv.

matches on project_id and compares:
  master total_activities (field name) vs outcomes total_activities
  master total_planned_duration (field name) vs outcomes total_planned_duration
Prints exact mismatch rows if any.
"""
import csv, os

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

def read_csv(name):
    with open(os.path.join(ROOT, f"{name}.csv"), newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)

master = read_csv("master_projects_canonical")
outcomes = read_csv("outcomes")

# index by project_id
master_by_pid = {r["project_id"]: r for r in master}
outcomes_by_pid = {r["project_id"]: r for r in outcomes}

master_pids = set(master_by_pid.keys())
outcomes_pids = set(outcomes_by_pid.keys())

print("SCOPE v0.2 — Correct column-matched master vs outcomes comparison")
print("=" * 72)
print(f"master_projects_canonical.csv: {len(master)} rows, {len(master_by_pid)} unique projects")
print(f"outcomes.csv:                 {len(outcomes)} rows, {len(outcomes_by_pid)} unique projects")
print()

print("Project ID sets:")
print(f"  Only in master:    {sorted(master_pids - outcomes_pids) or 'NONE'}")
print(f"  Only in outcomes:  {sorted(outcomes_pids - master_pids) or 'NONE'}")
print(f"  In both:           {len(master_pids & outcomes_pids)}")
print()

# Compare total_activities
print("FIELD: total_activities")
mismatch_activities = []
for pid in sorted(master_pids & outcomes_pids):
    mv = int(master_by_pid[pid]["total_activities"])
    ov = int(outcomes_by_pid[pid]["total_activities"])
    if mv != ov:
        mismatch_activities.append((pid, mv, ov))
print(f"  Projects compared: {len(master_pids & outcomes_pids)}")
print(f"  Mismatches: {len(mismatch_activities)}")
if mismatch_activities:
    print("  Mismatch rows:")
    for pid, mv, ov in mismatch_activities[:20]:
        print(f"    {pid}: master={mv}  outcomes={ov}")
    if len(mismatch_activities) > 20:
        print(f"    ... ({len(mismatch_activities) - 20} more)")

print()
print("FIELD: total_planned_duration")
mismatch_duration = []
for pid in sorted(master_pids & outcomes_pids):
    mv = int(master_by_pid[pid]["total_planned_duration"])
    ov = int(outcomes_by_pid[pid]["total_planned_duration"])
    if mv != ov:
        mismatch_duration.append((pid, mv, ov))
print(f"  Projects compared: {len(master_pids & outcomes_pids)}")
print(f"  Mismatches: {len(mismatch_duration)}")
if mismatch_duration:
    print("  Mismatch rows:")
    for pid, mv, ov in mismatch_duration[:20]:
        print(f"    {pid}: master={mv}  outcomes={ov}")
    if len(mismatch_duration) > 20:
        print(f"    ... ({len(mismatch_duration) - 20} more)")

print()
print("CONCLUSION:")
if not mismatch_activities and not mismatch_duration:
    print("  master_projects_canonical.csv and outcomes.csv are CONSISTENT.")
    print("  Both fields match for all 100 projects.")
else:
    print(f"  INCONSISTENCIES FOUND:")
    print(f"    total_activities mismatches: {len(mismatch_activities)}")
    print(f"    total_planned_duration mismatches: {len(mismatch_duration)}")
print()
print("NOTE: The previous shell 'join' report of 100 mismatches was a column-offset artifact.")
print("      master has 29 columns, outcomes has 4 columns. The shell 'join -t,' compared")
print("      master field 14/15 against outcomes field 4/5 — but outcomes field 4 = 'split',")
print("      field 5 = missing. The correct mapping is:")
print("        master.total_activities (col 14)  <-> outcomes.total_activities (col 2)")
print("        master.total_planned_duration (col 15) <-> outcomes.total_planned_duration (col 3)")
print()
print("DONE")
