#!/usr/bin/env python
"""Lightweight SCOPE v0.2 inventory: no deep DFS cycle detection.

Counts rows/cols, duplicate keys, split distribution, null columns,
enumerated values, FK integrity (fast set-based), numeric validity.
"""
import csv, collections, os

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        h = next(r)
        return h, list(r)

def idx(h, name): return h.index(name)
def enum(name, col):
    h, rows = read(os.path.join(ROOT, f"{name}.csv"))
    return collections.Counter(row[idx(h, col)] for row in rows)

print("SCOPE v0.2 — dataset inventory & integrity check")
print("=" * 72)

# ---- 1. Row/col/duplicates/splits per table ----
spec = [
    ("projects", ["project_id"]),
    ("activities", ["project_id","activity_id"]),
    ("dependencies", ["project_id","predecessor_id","successor_id"]),
    ("resources", ["resource_id"]),
    ("resource_allocation", ["project_id","activity_id","resource_id"]),
    ("environment", ["project_id","day_index"]),
    ("activity_states", ["project_id","activity_id","day_index"]),
    ("procurement", ["procurement_id"]),
    ("decisions", ["decision_id"]),
    ("events", ["event_id"]),
    ("rework", ["rework_id"]),
    ("friction_canonical", ["project_id","activity_id","friction_type"]),
    ("intervention_options", ["intervention_id"]),
    ("counterfactuals", ["intervention_id"]),
    ("construction_memory", ["memory_id"]),
    ("outcomes", ["project_id"]),
    ("friction_summary_canonical", ["project_id"]),
    ("master_projects_canonical", ["project_id"]),
]

for name, keycols in spec:
    h, rows = read(os.path.join(ROOT, f"{name}.csv"))
    seen = collections.Counter(tuple(row[idx(h,c)] for c in keycols) for row in rows)
    dups = sum(1 for v in seen.values() if v > 1)
    sp = collections.Counter(row[idx(h,"split")] for row in rows if "split" in h)
    nulls = [c for c in h if any(row[idx(h,c)]=="" for row in rows)]
    print(f"{name:30s} rows={len(rows):7d} cols={len(h):2d} dups={dups:3d} splits={dict(sp)} nulls={nulls}")

# ---- 2. FK integrity (fast) ----
print("\nFOREIGN-KEY INTEGRITY")
act = set(tuple(r) for r in read(os.path.join(ROOT,"activities.csv"))[1])
act_set = {(r[0], r[1]) for r in read(os.path.join(ROOT,"activities.csv"))[1]}  # (proj, act)
proj_set = set(r[0] for r in act)  # all project_ids from activities = all projects
res_set = set(r[0] for r in read(os.path.join(ROOT,"resources.csv"))[1])  # resource_id
ev_set = set(r[0] for r in read(os.path.join(ROOT,"events.csv"))[1])
int_set = set(r[0] for r in read(os.path.join(ROOT,"intervention_options.csv"))[1])

def check(label, pairs):
    missing = [p for p in pairs if p not in act_set and p not in proj_set]
    if missing:
        print(f"  {label}: BROKEN {len(missing)} refs")
    else:
        print(f"  {label}: OK")

dh, drows = read(os.path.join(ROOT,"dependencies.csv"))
dep_pairs = [(r[0], r[1]) for r in drows]  # (proj, pred)
check("dependencies.predecessor_id -> activities", dep_pairs)
dep_pairs2 = [(r[0], r[2]) for r in drows]
check("dependencies.successor_id -> activities", dep_pairs2)

rah = read(os.path.join(ROOT,"resource_allocation.csv"))
ra_set = {(r[0], r[1], r[2]) for r in rah[1]}
check("resource_allocation.resource_id -> resources", set(r[2] for r in rah[1]))
check("resource_allocation.(proj,act) -> activities", set((r[0],r[1]) for r in rah[1]))

check("environment.project_id -> projects", set(r[0] for r in read(os.path.join(ROOT,"environment.csv"))[1]))
check("activity_states.(proj,act) -> activities", set((r[0],r[1]) for r in read(os.path.join(ROOT,"activity_states.csv"))[1]))
check("procurement.project_id -> projects", set(r[1] for r in read(os.path.join(ROOT,"procurement.csv"))[1]))
check("decisions.(proj,act) -> activities", set((r[1],r[2]) for r in read(os.path.join(ROOT,"decisions.csv"))[1]))
check("events.(proj,act) -> activities", set((r[1],r[3]) for r in read(os.path.join(ROOT,"events.csv"))[1]))
check("rework.event_id -> events", set(r[2] for r in read(os.path.join(ROOT,"rework.csv"))[1]))
check("rework.(proj,act) -> activities", set((r[1],r[3]) for r in read(os.path.join(ROOT,"rework.csv"))[1]))
check("friction_canonical.(proj,act) -> activities", set((r[0],r[1]) for r in read(os.path.join(ROOT,"friction_canonical.csv"))[1]))
check("intervention_options.event_id -> events", set(r[1] for r in read(os.path.join(ROOT,"intervention_options.csv"))[1]))
check("counterfactuals.intervention_id -> intervention_options", set(r[0] for r in read(os.path.join(ROOT,"counterfactuals.csv"))[1]))
check("construction_memory.(proj,act) -> activities", set((r[1],r[2]) for r in read(os.path.join(ROOT,"construction_memory.csv"))[1]))

# ---- 3. Dependency orphans + relationships + lags ----
print("\nDEPENDENCY ORPHANS (predecessor/successor not in activities)")
or_pred = sum(1 for r in drows if (r[0],r[1]) not in act_set)
or_succ = sum(1 for r in drows if (r[0],r[2]) not in act_set)
print(f"  orphan predecessors: {or_pred}")
print(f"  orphan successors:   {or_succ}")
rels = collections.Counter(r[3] for r in drows)
lags = collections.Counter(int(r[4]) for r in drows)
print(f"  relationships: {dict(rels)}")
print(f"  lag_days range: {min(lags)}..{max(lags)} negatives={sum(1 for v in lags if v<0)}")

# ---- 4. Activities enums/numeric ----
ah, arows = read(os.path.join(ROOT,"activities.csv"))
print("\nACTIVITIES")
print(f"  status: {dict(collections.Counter(r[12] for r in arows))}")
print(f"  phase: {dict(collections.Counter(r[2] for r in arows))}")
print(f"  resource_type: {dict(collections.Counter(r[6] for r in arows))}")
print(f"  critical_path: {dict(collections.Counter(r[13] for r in arows))}")
print(f"  critical_path_position range: {min(int(r[14]) for r in arows)}..{max(int(r[14]) for r in arows)}")
print(f"  predecessor_count range: {min(int(r[16]) for r in arows)}..{max(int(r[16]) for r in arows)}")
print(f"  successor_count range: {min(int(r[17]) for r in arows)}..{max(int(r[17]) for r in arows)}")
print(f"  planned_duration_days <0: {sum(1 for r in arows if int(r[7])<0)}, ==0: {sum(1 for r in arows if int(r[7])==0)}")
print(f"  quantity<0: {sum(1 for r in arows if float(r[8])<0)}; planned_cost<0: {sum(1 for r in arows if float(r[10])<0)}")

# ---- 5. Decisions ----
dh2, drows2 = read(os.path.join(ROOT,"decisions.csv"))
print("\nDECISIONS")
print(f"  decision_status: {dict(collections.Counter(r[8] for r in drows2))}")
print(f"  decision_debt_level: {dict(collections.Counter(r[13] for r in drows2))}")
print(f"  decision_delay_days <0: {sum(1 for r in drows2 if int(r[6])<0)}")

# ---- 6. Events ----
eh, erows = read(os.path.join(ROOT,"events.csv"))
print("\nEVENTS")
print(f"  event_type: {dict(collections.Counter(r[4] for r in erows))}")
print(f"  severity: {dict(collections.Counter(r[5] for r in erows))}")
print(f"  duration_days <0: {sum(1 for r in erows if int(r[6])<0)}; impact_factor<0: {sum(1 for r in erows if float(r[8])<0)}")

# ---- 7. Environment ----
eh2, erows2 = read(os.path.join(ROOT,"environment.csv"))
print("\nENVIRONMENT")
print(f"  extreme_weather: {dict(collections.Counter(r[7] for r in erows2))}")
print(f"  weather_risk: {dict(collections.Counter(r[8] for r in erows2))}")
dates = sorted(r[1] for r in erows2)
print(f"  date range: {dates[0]} .. {dates[-1]}")
print(f"  unique projects: {len(set(r[0] for r in erows2))}")

# ---- 8. Procurement ----
ph, prows = read(os.path.join(ROOT,"procurement.csv"))
print("\nPROCUREMENT")
print(f"  material: {dict(collections.Counter(r[2] for r in prows))}")
print(f"  procurement_delay_days: min={min(int(r[5]) for r in prows)} max={max(int(r[5]) for r in prows)}")

# ---- 9. Intervention / Counterfactuals ----
ih, irows = read(os.path.join(ROOT,"intervention_options.csv"))
print("\nINTERVENTION_OPTIONS")
print(f"  category: {dict(collections.Counter(r[6] for r in irows))}")
print(f"  intervention_type: {dict(collections.Counter(r[5] for r in irows))}")
effs = [float(r[9]) for r in irows]
print(f"  true_effectiveness: min={min(effs)} max={max(effs)}")

ch, crows = read(os.path.join(ROOT,"counterfactuals.csv"))
bl = [float(r[4]) for r in crows]
cl = [float(r[5]) for r in crows]
av = [float(r[6]) for r in crows]
print("\nCOUNTERFACTUALS")
print(f"  baseline_delay_days: min={min(bl)} max={max(bl)}")
print(f"  counterfactual_delay_days: min={min(cl)} max={max(cl)}")
print(f"  avoided_delay_days: min={min(av)} max={max(av)} negatives={sum(1 for a in av if a<0)}")

# ---- 10. Construction memory missingness ----
mh, mrows = read(os.path.join(ROOT,"construction_memory.csv"))
print("\nCONSTRUCTION_MEMORY MISSINGNESS (top 12)")
miss = sorted(((c, sum(1 for r in mrows if r[i]=="")) for i,c in enumerate(mh)), key=lambda x:-x[1])
for c,n in miss[:12]:
    if n>0:
        print(f"  {c:25s} {n:6d} ({100*n/len(mrows):.1f}%)")

# ---- 11. Master vs outcomes consistency ----
print("\nMASTER_VS_OUTCOMES CONSISTENCY")
mh2, mrows_m = read(os.path.join(ROOT,"master_projects_canonical.csv"))
oh, orows_o = read(os.path.join(ROOT,"outcomes.csv"))
om = {r[0]: r for r in orows_o}
mm = 0
for r in mrows_m:
    pid = r[0]
    if pid in om:
        if int(r[13]) != int(om[pid][1]): mm+=1
        if int(r[14]) != int(om[pid][2]): mm+=1
print(f"  total_activities mismatches: {mm} (expected 0)")

# ---- 12. Activity dependency within same project: any cross-project refs? ----
print("\nCROSS-PROJECT DEPENDENCY CHECK")
cp = sum(1 for r in drows if r[0] != r[1].split('_')[0])
print(f"  rows where project_id != predecessor project prefix: {cp} (should be 0)")

print("\nDONE.")
