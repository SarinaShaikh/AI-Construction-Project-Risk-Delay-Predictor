#!/usr/bin/env python3
"""
Events and decisions per activity summary.

Determines:
- How many events and decisions are linked to each activity (distribution)
- Whether we can define an activity-level delay target from event durations
  or decision delays
- Summary of event delay per activity (sum of event duration_days per activity)
"""
import csv, collections, os

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

def read_csv(name):
    with open(os.path.join(ROOT, f"{name}.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        return h, [dict(zip(h, row)) for row in r]

def main():
    print("Events and decisions per activity summary")
    print("="*72)
    _, acts = read_csv("activities")
    _, events = read_csv("events")
    _, decisions = read_csv("decisions")
    _, cm = read_csv("construction_memory")

    act_keys = {(a["project_id"], a["activity_id"]) for a in acts}
    print(f"Total activities: {len(act_keys)}")
    print()

    # Events per activity
    ev_per_act = collections.Counter((e["project_id"], e["activity_id"]) for e in events)
    ev_count_dist = collections.Counter(ev_per_act.values())
    print(f"Total events: {len(events)}")
    print(f"Activities with >=1 event: {len(ev_per_act)}")
    print(f"Activities with 0 events: {len(act_keys) - len(ev_per_act)}")
    print()
    print("Events per activity distribution:")
    for n in sorted(ev_count_dist):
        print(f"  {n} event(s): {ev_count_dist[n]} activities")
    print()
    print(f"Events per activity: max={max(ev_per_act.values())} "
          f"(activity {max(ev_per_act, key=ev_per_act.get)})")

    # Decisions per activity
    dec_per_act = collections.Counter((d["project_id"], d["activity_id"]) for d in decisions)
    dec_dist = collections.Counter(dec_per_act.values())
    print()
    print(f"Total decisions: {len(decisions)}")
    print(f"Activities with >=1 decision: {len(dec_per_act)}")
    print(f"Activities with 0 decisions: {len(act_keys) - len(dec_per_act)}")
    print()
    print("Decisions per activity distribution:")
    for n in sorted(dec_dist):
        print(f"  {n} decision(s): {dec_dist[n]} activities")
    print()
    print(f"Decisions per activity: max={max(dec_per_act.values())} "
          f"(activity {max(dec_per_act, key=dec_per_act.get)})")

    # Event delay per activity (sum of event duration_days per activity)
    ev_delay_per_act = collections.defaultdict(float)
    for e in events:
        key = (e["project_id"], e["activity_id"])
        ev_delay_per_act[key] += int(e["duration_days"])
    delay_vals = list(ev_delay_per_act.values())
    print()
    print("Event delay (sum of duration_days) per affected activity:")
    print(f"  n_activities_with_events={len(ev_delay_per_act)}")
    print(f"  total_event_delay_days: min={min(delay_vals)} max={max(delay_vals)} "
          f"mean={sum(delay_vals)/len(delay_vals):.2f}")
    print(f"  total_event_delay_days > 0: {sum(1 for d in delay_vals if d > 0)}")
    print(f"  total_event_delay_days == 0: {sum(1 for d in delay_vals if d == 0)}")

    # Decision delay per activity (sum of decision_delay_days per activity)
    dec_delay_per_act = collections.defaultdict(int)
    for d in decisions:
        key = (d["project_id"], d["activity_id"])
        dec_delay_per_act[key] += int(d["decision_delay_days"])
    dec_delay_vals = list(dec_delay_per_act.values())
    print()
    print("Decision delay (sum of decision_delay_days) per affected activity:")
    print(f"  n_activities_with_decisions={len(dec_delay_per_act)}")
    print(f"  total_decision_delay_days: min={min(dec_delay_vals)} max={max(dec_delay_vals)} "
          f"mean={sum(dec_delay_vals)/len(dec_delay_vals):.2f}")
    print(f"  total_decision_delay_days > 0: {sum(1 for d in dec_delay_vals if d > 0)}")
    print(f"  total_decision_delay_days == 0: {sum(1 for d in dec_delay_vals if d == 0)}")

    # construction_memory observed_delay_days per activity (sum across all memory types)
    mem_delay_per_act = collections.defaultdict(list)
    for r in cm:
        key = (r["project_id"], r["activity_id"])
        mem_delay_per_act[key].append(float(r["observed_delay_days"]))
    mem_totals = {k: sum(v) for k, v in mem_delay_per_act.items()}
    mem_totals_vals = list(mem_totals.values())
    print()
    print("construction_memory observed_delay_days (sum across all memory types) per activity:")
    print(f"  n_activities_with_memory={len(mem_totals)}")
    print(f"  total_observed_delay_days: min={min(mem_totals_vals)} max={max(mem_totals_vals)} "
          f"mean={sum(mem_totals_vals)/len(mem_totals_vals):.2f}")
    print(f"  total_observed_delay_days > 0: {sum(1 for d in mem_totals_vals if d > 0)}")
    print(f"  total_observed_delay_days == 0: {sum(1 for d in mem_totals_vals if d == 0)}")

    # How many activities have BOTH event and decision and memory?
    both = act_keys & set(ev_per_act.keys()) & set(dec_per_act.keys())
    print()
    print(f"Activities with BOTH event AND decision: {len(both)}")
    all_three = both & set(mem_delay_per_act.keys())
    print(f"Activities with event AND decision AND memory: {len(all_three)}")

    # Sample activities with high event delay
    print()
    print("Top 10 activities by total event delay (duration_days sum):")
    top_ev = sorted(ev_delay_per_act.items(), key=lambda x: -x[1])[:10]
    for (pid, aid), d in top_ev:
        print(f"  {aid}: event_delay={d} days, n_events={ev_per_act[(pid,aid)]}")

    print()
    print("Top 10 activities by total decision delay:")
    top_dec = sorted(dec_delay_per_act.items(), key=lambda x: -x[1])[:10]
    for (pid, aid), d in top_dec:
        print(f"  {aid}: decision_delay={d} days, n_decisions={dec_per_act[(pid,aid)]}")

    print()
    print("Top 10 activities by total observed_delay_days (construction_memory):")
    top_mem = sorted(mem_totals.items(), key=lambda x: -x[1])[:10]
    for (pid, aid), d in top_mem:
        print(f"  {aid}: observed_delay={d:.2f} days")

    print()
    print("DONE")

if __name__ == "__main__":
    main()
