#!/usr/bin/env python3
"""
SCOPE v0.2 project split verification.

Verifies:
- 70/15/15 project split as declared
- That the split assignment is consistent across all tables
- Lists a sample of project IDs in each split
- Checks for any project that appears in more than one split (should be impossible)
"""
import csv, collections, os

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

def read_csv(name):
    with open(os.path.join(ROOT, f"{name}.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        return h, [dict(zip(h, row)) for row in r]

def main():
    print("SCOPE v0.2 — Project split verification")
    print("="*72)

    proj_h, projects = read_csv("projects")
    prj_split = {p["project_id"]: p["split"] for p in projects}

    # Expected
    expected = {"train": 70, "validation": 15, "test": 15}
    actual = collections.Counter(prj_split.values())
    print(f"Expected split: {expected}")
    print(f"Actual split:   {dict(actual)}")
    print(f"Match: {actual == collections.Counter(expected)}")
    print()

    # Per-table split membership (should match projects table exactly)
    print("Per-table split membership vs projects table:")
    tables = [
        "activities", "dependencies", "resources", "resource_allocation",
        "environment", "activity_states", "procurement", "decisions",
        "events", "rework", "friction_canonical", "intervention_options",
        "counterfactuals", "construction_memory", "outcomes",
        "friction_summary_canonical", "master_projects_canonical",
    ]
    consistent = True
    for t in tables:
        _, rows = read_csv(t)
        splits = collections.Counter(r.get("split", "MISSING") for r in rows)
        if t in ("projects", "outcomes", "friction_summary_canonical", "master_projects_canonical"):
            # these have exactly one row per project
            match = (splits == actual)
            print(f"  {t:32s} {dict(splits)} {'OK' if match else 'MISMATCH'}")
            if not match:
                consistent = False
        else:
            # these have multiple rows per project but same split value
            project_splits_in_table = set()
            for r in rows:
                key = r.get("project_id")
                if key:
                    project_splits_in_table.add((key, r.get("split")))
            # check all projects present in this table have correct split
            bad = [(pid, tbl_split, prj_split.get(pid)) for pid, tbl_split in project_splits_in_table
                   if prj_split.get(pid) != tbl_split]
            print(f"  {t:32s} projects_in_table={len(project_splits_in_table)} "
                  f"inconsistent={len(bad)} {'OK' if not bad else 'MISMATCH'}")
            if bad:
                consistent = False
                for pid, tbl_s, prj_s in bad[:5]:
                    print(f"      example: {pid} table_split={tbl_s} project_split={prj_s}")

    print()
    print(f"Overall split consistency across all tables: {consistent}")
    print()

    # List member project IDs per split
    print("Project IDs per split (first 10 each):")
    for split in ["train", "validation", "test"]:
        members = sorted(p for p, s in prj_split.items() if s == split)
        print(f"  {split} ({len(members)} projects): {members[:10]}{'...' if len(members) > 10 else ''}")

    # Verify outcomes and master_projects_canonical have exactly the same 100 projects
    _, outcomes = read_csv("outcomes")
    _, master = read_csv("master_projects_canonical")
    out_ids = set(r["project_id"] for r in outcomes)
    mstr_ids = set(r["project_id"] for r in master)
    prj_ids = set(prj_split.keys())
    print()
    print(f"projects.csv projects: {len(prj_ids)}")
    print(f"outcomes.csv projects: {len(out_ids)}")
    print(f"master_projects_canonical.csv projects: {len(mstr_ids)}")
    print(f"All three agree on project set: {prj_ids == out_ids == mstr_ids}")
    print(f"Any project in outcomes not in projects.csv: {out_ids - prj_ids}")
    print(f"Any project in master not in projects.csv: {mstr_ids - prj_ids}")

    print()
    print("DONE")

if __name__ == "__main__":
    main()
