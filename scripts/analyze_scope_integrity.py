#!/usr/bin/env python
"""SCOPE v0.2 structural integrity analysis — Phase 1 exploration.

Reads every CSV in data/raw/SCOPE_v02_Public/ and reports:
- row/column counts, duplicate keys, split distribution, null columns
- foreign-key integrity across tables
- dependency orphan refs, relationship/lag values
- enum values for status/phase/event_type/severity/etc.
- numeric validity checks (negative durations, etc.)
- construction_memory missingness by column
- master-vs-outcomes consistency
- circular dependency detection (per project, recursive DFS)
"""

import csv
import collections
import os
import sys
from collections import defaultdict

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    return header, rows


def idx(h, name):
    return h.index(name)


def build_set(name, cols):
    h, rows = read(os.path.join(ROOT, f"{name}.csv"))
    return set(tuple(row[idx(h, c)] for c in cols) for row in rows)


def counter_enum(name, col):
    h, rows = read(os.path.join(ROOT, f"{name}.csv"))
    return collections.Counter(row[idx(h, col)] for row in rows)


def main():
    anomalies = []

    # ---------- 1. Table inventory ----------
    print("=" * 80)
    print("TABLE INVENTORY: rows, cols, duplicate keys, split distribution, null columns")
    print("=" * 80)

    key_spec = [
        ("projects", ["project_id"]),
        ("activities", ["project_id", "activity_id"]),
        ("dependencies", ["project_id", "predecessor_id", "successor_id"]),
        ("resources", ["resource_id"]),
        ("resource_allocation", ["project_id", "activity_id", "resource_id"]),
        ("environment", ["project_id", "day_index"]),
        ("activity_states", ["project_id", "activity_id", "day_index"]),
        ("procurement", ["procurement_id"]),
        ("decisions", ["decision_id"]),
        ("events", ["event_id"]),
        ("rework", ["rework_id"]),
        ("friction_canonical", ["project_id", "activity_id", "friction_type"]),
        ("intervention_options", ["intervention_id"]),
        ("counterfactuals", ["intervention_id"]),
        ("construction_memory", ["memory_id"]),
        ("outcomes", ["project_id"]),
        ("friction_summary_canonical", ["project_id"]),
        ("master_projects_canonical", ["project_id"]),
    ]

    for name, keycols in key_spec:
        h, rows = read(os.path.join(ROOT, f"{name}.csv"))
        seen = collections.Counter(
            tuple(row[idx(h, c)] for c in keycols) for row in rows
        )
        dups = sum(1 for v in seen.values() if v > 1)
        splits = collections.Counter(
            row[idx(h, "split")] for row in rows if "split" in h
        )
        nulls = collections.Counter()
        for row in rows:
            for i, c in enumerate(h):
                if row[i] == "" or row[i] is None:
                    nulls[c] += 1
        null_cols = [c for c, n in nulls.items() if n > 0]
        print(
            f"{name:32s} rows={len(rows):8d} cols={len(h):3d} "
            f"dup={dups:5d} splits={dict(splits)} null_cols={null_cols}"
        )
        if dups:
            anomalies.append((name, "duplicate keys", dups))

    # ---------- 2. Split alignment ----------
    print()
    print("=" * 80)
    print("PROJECT SPLIT CONSISTENCY (expected 70 train / 15 val / 15 test)")
    print("=" * 80)

    project_tables = {
        "projects",
        "outcomes",
        "friction_summary_canonical",
        "master_projects_canonical",
    }
    for name in [
        "projects",
        "activities",
        "dependencies",
        "resources",
        "resource_allocation",
        "environment",
        "activity_states",
        "procurement",
        "decisions",
        "events",
        "rework",
        "friction_canonical",
        "intervention_options",
        "counterfactuals",
        "construction_memory",
        "outcomes",
        "friction_summary_canonical",
        "master_projects_canonical",
    ]:
        h, rows = read(os.path.join(ROOT, f"{name}.csv"))
        sp = collections.Counter(
            row[idx(h, "split")] for row in rows if "split" in h
        )
        if name in project_tables:
            ok = (
                sp.get("train", 0) == 70
                and sp.get("validation", 0) == 15
                and sp.get("test", 0) == 15
            )
        else:
            ok = sp.get("train", 0) > 0
        flag = "" if ok else "  <-- MISMATCH"
        print(f"{name:32s} {dict(sp)}{flag}")
        if not ok:
            anomalies.append((name, "split mismatch", dict(sp)))

    # ---------- 3. Foreign-key integrity ----------
    print()
    print("=" * 80)
    print("FOREIGN-KEY CROSS-REFERENCE INTEGRITY")
    print("=" * 80)

    proj_ids = build_set("projects", ["project_id"])
    act_ids = build_set("activities", ["project_id", "activity_id"])
    res_ids = build_set("resources", ["resource_id"])
    proc_ids = build_set("procurement", ["procurement_id"])
    ev_ids = build_set("events", ["event_id"])
    int_ids = build_set("intervention_options", ["intervention_id"])

    checks = [
        ("dependencies.predecessor_id -> activities", "dependencies", ["project_id", "predecessor_id"]),
        ("dependencies.successor_id -> activities", "dependencies", ["project_id", "successor_id"]),
        ("resource_allocation.activity_id -> activities", "resource_allocation", ["project_id", "activity_id"]),
        ("resource_allocation.resource_id -> resources", "resource_allocation", ["resource_id"]),
        ("environment.project_id -> projects", "environment", ["project_id"]),
        ("activity_states.(proj,activity) -> activities", "activity_states", ["project_id", "activity_id"]),
        ("procurement.project_id -> projects", "procurement", ["project_id"]),
        ("decisions.(proj,activity) -> activities", "decisions", ["project_id", "activity_id"]),
        ("events.(proj,activity) -> activities", "events", ["project_id", "activity_id"]),
        ("rework.event_id -> events", "rework", ["event_id"]),
        ("rework.(proj,activity) -> activities", "rework", ["project_id", "activity_id"]),
        ("friction_canonical.(proj,activity) -> activities", "friction_canonical", ["project_id", "activity_id"]),
        ("intervention_options.event_id -> events", "intervention_options", ["event_id"]),
        ("intervention_options.(proj,activity) -> activities", "intervention_options", ["project_id", "activity_id"]),
        ("counterfactuals.intervention_id -> intervention_options", "counterfactuals", ["intervention_id"]),
        ("counterfactuals.event_id -> events", "counterfactuals", ["event_id"]),
        ("construction_memory.(proj,activity) -> activities", "construction_memory", ["project_id", "activity_id"]),
        ("outcomes.project_id -> projects", "outcomes", ["project_id"]),
        ("friction_summary_canonical.project_id -> projects", "friction_summary_canonical", ["project_id"]),
        ("master_projects_canonical.project_id -> projects", "master_projects_canonical", ["project_id"]),
    ]

    for label, src, cols in checks:
        src_set = build_set(src, cols)
        if "project_id" in cols and len(cols) == 1:
            missing = src_set - proj_ids
        elif len(cols) == 2 and cols[0] == "project_id" and cols[1] == "activity_id":
            missing = src_set - act_ids
        elif "resource_id" in cols:
            missing = src_set - res_ids
        elif "procurement" in src:
            missing = src_set - proc_ids
        elif "event_id" in cols and len(cols) == 1:
            missing = src_set - ev_ids
        elif "intervention_id" in cols and len(cols) == 1 and src == "counterfactuals":
            missing = src_set - int_ids
        else:
            missing = set()
        if missing:
            anomalies.append((label, "broken FK", len(missing)))
            print(f"{label:58s} BROKEN: {len(missing)} missing references")
        else:
            print(f"{label:58s} OK ({len(src_set)} referenced)")

    # ---------- 4. Dependency orphan refs + relationship/lag ----------
    print()
    print("=" * 80)
    print("DEPENDENCY CONSISTENCY (predecessor/successor must exist in activities)")
    print("=" * 80)

    dh, drows = read(os.path.join(ROOT, "dependencies.csv"))
    i_proj = idx(dh, "project_id")
    i_pred = idx(dh, "predecessor_id")
    i_succ = idx(dh, "successor_id")
    i_rel = idx(dh, "relationship")
    i_lag = idx(dh, "lag_days")

    act_set = build_set("activities", ["project_id", "activity_id"])
    bad_pred = collections.Counter()
    bad_succ = collections.Counter()
    for row in drows:
        if (row[i_proj], row[i_pred]) not in act_set:
            bad_pred[(row[i_proj], row[i_pred])] += 1
        if (row[i_proj], row[i_succ]) not in act_set:
            bad_succ[(row[i_proj], row[i_succ])] += 1
    print(f"Orphan predecessor refs: {sum(bad_pred.values())} (unique pairs: {len(bad_pred)})")
    print(f"Orphan successor refs:   {sum(bad_succ.values())} (unique pairs: {len(bad_succ)})")
    if bad_pred:
        anomalies.append(("dependencies.orphan_pred", dict(bad_pred)))
    if bad_succ:
        anomalies.append(("dependencies.orphan_succ", dict(bad_succ)))

    print()
    print("DEPENDENCY RELATIONSHIP VALUES")
    rels = collections.Counter(row[i_rel] for row in drows)
    print(f"  relationships: {dict(rels)}")

    print("DEPENDENCY lag_days DISTRIBUTION")
    lag_counter = collections.Counter()
    neg_lags = 0
    for row in drows:
        r = int(row[i_lag])
        lag_counter[r] += 1
        if r < 0:
            neg_lags += 1
    print(f"  min={min(lag_counter)} max={max(lag_counter)} negatives={neg_lags}")
    print(f"  top lags: {lag_counter.most_common(10)}")

    # ---------- 5. Activities enums + numeric ----------
    print()
    print("=" * 80)
    print("ACTIVITIES ENUM & NUMERIC VALIDITY")
    print("=" * 80)

    ah, arows = read(os.path.join(ROOT, "activities.csv"))
    print(f"  status: {dict(counter_enum('activities', 'status'))}")
    print(f"  phase: {dict(counter_enum('activities', 'phase'))}")
    print(f"  resource_type: {dict(counter_enum('activities', 'resource_type'))}")
    print(f"  critical_path (0/1): {dict(counter_enum('activities', 'critical_path'))}")
    cpos = [int(row[idx(ah, "critical_path_position")]) for row in arows]
    pcount = [int(row[idx(ah, "predecessor_count")]) for row in arows]
    scount = [int(row[idx(ah, "successor_count")]) for row in arows]
    print(f"  critical_path_position range: {min(cpos)} .. {max(cpos)}")
    print(f"  predecessor_count range: {min(pcount)} .. {max(pcount)}")
    print(f"  successor_count range: {min(scount)} .. {max(scount)}")
    print(f"  project_network_duration unique values: {len(set(row[idx(ah,'project_network_duration')] for row in arows))}")

    neg_dur = sum(1 for row in arows if int(row[idx(ah, "planned_duration_days")]) < 0)
    zero_dur = sum(1 for row in arows if int(row[idx(ah, "planned_duration_days")]) == 0)
    qty_neg = sum(1 for row in arows if float(row[idx(ah, "quantity")]) < 0)
    cost_neg = sum(1 for row in arows if float(row[idx(ah, "planned_cost")]) < 0)
    print(f"  planned_duration_days <0: {neg_dur}; ==0: {zero_dur}")
    print(f"  quantity <0: {qty_neg}")
    print(f"  planned_cost <0: {cost_neg}")

    # ---------- 6. Decisions ----------
    print()
    print("=" * 80)
    print("DECISIONS")
    print("=" * 80)
    print(f"  decision_status: {dict(counter_enum('decisions', 'decision_status'))}")
    print(f"  decision_debt_level: {dict(counter_enum('decisions', 'decision_debt_level'))}")
    print(f"  decision_type: {dict(counter_enum('decisions', 'decision_type'))}")
    ddelay_neg = sum(1 for row in read(os.path.join(ROOT, "decisions.csv"))[1] if int(row[idx(read(os.path.join(ROOT,"decisions.csv"))[0], "decision_delay_days")]) < 0)
    # simpler:
    dh2, drows2 = read(os.path.join(ROOT, "decisions.csv"))
    ddelay_neg = sum(1 for row in drows2 if int(row[idx(dh2, "decision_delay_days")]) < 0)
    print(f"  decision_delay_days <0: {ddelay_neg}")

    # ---------- 7. Events ----------
    print()
    print("=" * 80)
    print("EVENTS")
    print("=" * 80)
    eh, erows = read(os.path.join(ROOT, "events.csv"))
    print(f"  event_type: {dict(counter_enum('events', 'event_type'))}")
    print(f"  severity: {dict(counter_enum('events', 'severity'))}")
    ev_dur_neg = sum(1 for row in erows if int(row[idx(eh, "duration_days")]) < 0)
    impact_neg = sum(1 for row in erows if float(row[idx(eh, "impact_factor")]) < 0)
    print(f"  duration_days <0: {ev_dur_neg}")
    print(f"  impact_factor <0: {impact_neg}")

    # ---------- 8. Environment ----------
    print()
    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)
    print(f"  extreme_weather: {dict(counter_enum('environment', 'extreme_weather'))}")
    print(f"  weather_risk: {dict(counter_enum('environment', 'weather_risk'))}")
    _, erows2 = read(os.path.join(ROOT, "environment.csv"))
    dates = sorted(row[idx(read(os.path.join(ROOT, "environment.csv"))[0], "date")] for row in erows2)
    print(f"  date range: {dates[0]} .. {dates[-1]}")
    print(f"  unique projects: {len(set(row[idx(read(os.path.join(ROOT, 'environment.csv'))[0], 'project_id')] for row in erows2))}")

    # ---------- 9. Procurement ----------
    print()
    print("=" * 80)
    print("PROCUREMENT")
    print("=" * 80)
    print(f"  material types: {dict(counter_enum('procurement', 'material'))}")
    ph, prows = read(os.path.join(ROOT, "procurement.csv"))
    delays = [int(row[idx(ph, "procurement_delay_days")]) for row in prows]
    print(f"  procurement_delay_days: min={min(delays)} max={max(delays)} negatives={sum(1 for d in delays if d<0)}")

    # ---------- 10. Intervention / Counterfactuals ----------
    print()
    print("=" * 80)
    print("INTERVENTION_OPTIONS & COUNTERFACTUALS")
    print("=" * 80)
    ih, irows = read(os.path.join(ROOT, "intervention_options.csv"))
    print(f"  category: {dict(counter_enum('intervention_options', 'category'))}")
    print(f"  intervention_type: {dict(collections.Counter(row[idx(ih,'intervention_type')] for row in irows))}")
    effs = [float(row[idx(ih, "true_effectiveness")]) for row in irows]
    print(f"  true_effectiveness: min={min(effs)} max={max(effs)}")
    ch, crows = read(os.path.join(ROOT, "counterfactuals.csv"))
    bline = [float(row[idx(ch, "baseline_delay_days")]) for row in crows]
    cline = [float(row[idx(ch, "counterfactual_delay_days")]) for row in crows]
    avoid = [float(row[idx(ch, "avoided_delay_days")]) for row in crows]
    print(f"  baseline_delay_days: min={min(bline)} max={max(bline)}")
    print(f"  counterfactual_delay_days: min={min(cline)} max={max(cline)}")
    print(f"  avoided_delay_days: min={min(avoid)} max={max(avoid)} negatives={sum(1 for a in avoid if a<0)}")

    # ---------- 11. Construction memory missingness ----------
    print()
    print("=" * 80)
    print("CONSTRUCTION_MEMORY MISSINGNESS BY COLUMN")
    print("=" * 80)
    mh, mrows = read(os.path.join(ROOT, "construction_memory.csv"))
    for c in mh:
        miss = sum(1 for r in mrows if r[idx(mh, c)] == "")
        if miss > 0:
            print(f"  {c:25s} missing {miss:7d} ({100*miss/len(mrows):.2f}%)")

    # ---------- 12. Master vs outcomes vs friction_summary consistency ----------
    print()
    print("=" * 80)
    print("PROJECT-LEVEL CONSISTENCY: master vs outcomes vs friction_summary")
    print("=" * 80)
    moh, mrows_o = read(os.path.join(ROOT, "outcomes.csv"))
    mfh, frows_o = read(os.path.join(ROOT, "friction_summary_canonical.csv"))
    mmh, mrows_m = read(os.path.join(ROOT, "master_projects_canonical.csv"))
    om = {r[idx(moh, "project_id")]: r for r in mrows_o}
    fm = {r[idx(mfh, "project_id")]: r for r in mrows_o}
    mism = 0
    for r in mrows_m:
        pid = r[idx(mmh, "project_id")]
        if pid not in om:
            mism += 1
            continue
        if int(r[idx(mmh, "total_activities")]) != int(om[pid][idx(moh, "total_activities")]):
            mism += 1
            if mism <= 8:
                print(f"  {pid}: activities master={r[idx(mmh,'total_activities')]} vs outcomes={om[pid][idx(moh,'total_activities')]}")
        if int(r[idx(mmh, "total_planned_duration")]) != int(om[pid][idx(moh, "total_planned_duration")]):
            mism += 1
            if mism <= 8:
                print(f"  {pid}: duration master={r[idx(mmh,'total_planned_duration')]} vs outcomes={om[pid][idx(moh,'total_planned_duration')]}")
    print(f"Total master-vs-outcomes mismatches: {mism}")

    # ---------- 13. Circular dependency detection ----------
    print()
    print("=" * 80)
    print("CIRCULAR DEPENDENCY DETECTION (per project DFS)")
    print("=" * 80)

    adj = defaultdict(list)
    proj_acts = defaultdict(set)
    for row in drows:
        proj = row[i_proj]
        pred = row[i_pred]
        succ = row[i_succ]
        adj[(proj, pred)].append((proj, succ))
        proj_acts[proj].add(pred)
        proj_acts[proj].add(succ)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    cycles_found = []
    total_edges = len(drows)

    def dfs(node, path):
        color[node] = GRAY
        path.append(node)
        for nxt in adj.get(node, []):
            if color.get(nxt, WHITE) == GRAY:
                # cycle: locate in path
                start_idx = path.index(nxt)
                cycle = path[start_idx:] + [nxt]
                cycles_found.append(cycle)
            elif color.get(nxt, WHITE) == WHITE:
                dfs(nxt, path)
        path.pop()
        color[node] = BLACK

    for proj in sorted(proj_acts.keys()):
        for act in sorted(proj_acts[proj]):
            node = (proj, act)
            if color.get(node, WHITE) == WHITE:
                dfs(node, [])

    print(f"Total dependency edges: {total_edges}")
    print(f"Cycles detected: {len(cycles_found)}")
    if cycles_found:
        for c in cycles_found[:20]:
            print("  CYCLE:", " -> ".join(f"{p}:{a}" for p, a in c))
        if len(cycles_found) > 20:
            print(f"  ... ({len(cycles_found) - 20} more)")
        anomalies.append(("circular_dependencies", len(cycles_found)))
    else:
        print("  No circular dependencies found.")

    # ---------- 14. Activity-to-activity dependency within same project ----------
    print()
    print("=" * 80)
    print("DEPENDENCY GRAPH SUMMARY PER PROJECT")
    print("=" * 80)
    proj_edge_count = collections.Counter()
    proj_node_count = collections.Counter()
    for row in drows:
        proj_edge_count[row[i_proj]] += 1
    for (proj, _) in proj_acts:
        proj_node_count[proj] += 1
    multi = sum(1 for p in proj_edge_count if proj_edge_count[p] > 0)
    print(f"Projects with >=1 dependency: {multi}")
    print(f"Dependency edge count range: {min(proj_edge_count.values())} .. {max(proj_edge_count.values())}")
    print(f"Activity node count range: {min(proj_node_count.values())} .. {max(proj_node_count.values())}")

    # ---------- 15. Summary ----------
    print()
    print("=" * 80)
    print("ANOMALIES SUMMARY")
    print("=" * 80)
    if anomalies:
        for a in anomalies:
            print(" -", a)
    else:
        print("  None detected.")

    print()
    print("COMPLETE.")


if __name__ == "__main__":
    main()
