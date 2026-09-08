#!/usr/bin/env python3
"""
Check construction_memory ↔ activities relationship.

Tests:
- Is each activity represented exactly once in construction_memory?
- Or many:many?
- Does observed_delay_days vary across memory records for the same activity?
- What memory_types exist per activity?
"""
import csv, collections, os

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

def read_csv(name):
    with open(os.path.join(ROOT, f"{name}.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        return h, [dict(zip(h, row)) for row in r]

def main():
    print("Construction memory - activities mapping analysis")
    print("="*72)
    _, acts = read_csv("activities")
    _, cm = read_csv("construction_memory")

    act_keys = {(a["project_id"], a["activity_id"]) for a in acts}
    print(f"Unique activities: {len(act_keys)}")
    print(f"Construction memory records: {len(cm)}")

    mem_keys = collections.Counter((r["project_id"], r["activity_id"]) for r in cm)
    mem_per_act = collections.Counter()
    for k, c in mem_keys.items():
        mem_per_act[c] += 1

    print(f"Activities with exactly 1 memory record: {mem_per_act.get(1, 0)}")
    print(f"Activities with >1 memory records: {sum(v for k,v in mem_per_act.items() if k>1)}")
    print(f"Activities with 0 memory records: {len(act_keys) - len(mem_keys)}")
    print()
    print("Memory records per activity distribution:")
    for n in sorted(mem_per_act):
        print(f"  {n} record(s): {mem_per_act[n]} activities")

    # Which activities have multiple memory records?
    multi = [(k, c) for k, c in mem_keys.items() if c > 1]
    print()
    print(f"Sample multi-memory activities (first 10):")
    for (pid, aid), cnt in multi[:10]:
        recs = [r for r in cm if r["project_id"]==pid and r["activity_id"]==aid]
        types = collections.Counter(r["memory_type"] for r in recs)
        delays = [float(r["observed_delay_days"]) for r in recs]
        print(f"  {aid}: {cnt} records, types={dict(types)}, "
              f"observed_delay_days={delays}, all_same={len(set(delays))==1}")

    # Is observed_delay_days constant for same activity?
    print()
    print("Does observed_delay_days vary for same activity across memory types?")
    vary_count = 0
    sample_varies = []
    for (pid, aid), cnt in multi[:100]:
        recs = [r for r in cm if r["project_id"]==pid and r["activity_id"]==aid]
        delays = set(float(r["observed_delay_days"]) for r in recs)
        if len(delays) > 1:
            vary_count += 1
            if len(sample_varies) < 5:
                sample_varies.append((aid, delays))
    print(f"  Activities with >1 distinct observed_delay_days: {vary_count} (of {len(multi)} multi-memory activities)")
    for aid, delays in sample_varies:
        print(f"    {aid}: delays={sorted(delays)}")

    # memory_type breakdown
    print()
    print("Memory type counts:")
    print(f"  {dict(collections.Counter(r['memory_type'] for r in cm))}")

    # activity_pattern records: do they match activities 1:1?
    ap = [r for r in cm if r["memory_type"] == "activity_pattern"]
    ap_keys = {(r["project_id"], r["activity_id"]) for r in ap}
    print()
    print(f"activity_pattern records: {len(ap)}")
    print(f"Unique (proj,act) in activity_pattern: {len(ap_keys)}")
    print(f"Activities NOT in activity_pattern: {len(act_keys - ap_keys)}")
    print(f"activity_pattern keys NOT in activities: {len(ap_keys - act_keys)}")

    # risk_pattern records: how do they map?
    rp = [r for r in cm if r["memory_type"] == "risk_pattern"]
    print()
    print(f"risk_pattern records: {len(rp)}")
    # risk_pattern should map to events (event_id) not activities — but it has activity_id column
    rp_act_keys = collections.Counter((r["project_id"], r["activity_id"]) for r in rp)
    print(f"Unique (proj,act) in risk_pattern: {len(rp_act_keys)}")
    print(f"Risk patterns per activity (top 10): {rp_act_keys.most_common(10)}")

    # decision_pattern
    dp = [r for r in cm if r["memory_type"] == "decision_pattern"]
    print()
    print(f"decision_pattern records: {len(dp)}")
    print(f"Unique (proj,act): {len({(r['project_id'], r['activity_id']) for r in dp})}")
    d_status = collections.Counter(r.get('decision_status','?') for r in dp)
    print(f"  decision_status present in memory: {dict(d_status)}")

    # Is observed_delay_days in activity_pattern actually 0 for all?
    ap_delays = [float(r["observed_delay_days"]) for r in ap]
    print()
    print(f"activity_pattern observed_delay_days: min={min(ap_delays)} max={max(ap_delays)} "
          f"nonzero={sum(1 for d in ap_delays if d>0)}")

    # friction_pattern
    fp = [r for r in cm if r["memory_type"] == "friction_pattern"]
    print()
    print(f"friction_pattern records: {len(fp)}")
    print(f"Unique (proj,act): {len({(r['project_id'], r['activity_id']) for r in fp})}")

    # rework_pattern
    rwp = [r for r in cm if r["memory_type"] == "rework_pattern"]
    print()
    print(f"rework_pattern records: {len(rwp)}")
    print(f"Unique (proj,act): {len({(r['project_id'], r['activity_id']) for r in rwp})}")
    rwp_delays = [float(r["observed_delay_days"]) for r in rwp]
    print(f"  observed_delay_days: min={min(rwp_delays)} max={max(rwp_delays)} nonzero={sum(1 for d in rwp_delays if d>0)}")

    print()
    print("DONE")

if __name__ == "__main__":
    main()
