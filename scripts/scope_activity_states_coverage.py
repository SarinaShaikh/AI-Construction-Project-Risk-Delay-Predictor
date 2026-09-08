#!/usr/bin/env python3
"""
Activity states coverage vs planned duration analysis.

Determines whether activity_states records can be used to infer actual
activity duration, by checking:
- How many state records exist per activity?
- Does the count correlate with planned_duration_days?
- Is there a start/stop pattern visible in the daily states?
- Do completed vs not-yet-started activities behave differently?
"""
import csv, collections, os

ROOT = "AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

def read_csv(name):
    with open(os.path.join(ROOT, f"{name}.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        return h, [dict(zip(h, row)) for row in r]

def main():
    print("Activity states coverage analysis")
    print("="*72)
    _, acts = read_csv("activities")
    _, states = read_csv("activity_states")

    act_planned = {a["activity_id"]: int(a["planned_duration_days"]) for a in acts}
    act_proj = {a["activity_id"]: a["project_id"] for a in acts}

    # state count per activity
    states_per_act = collections.Counter((s["project_id"], s["activity_id"]) for s in states)
    print(f"Total unique (proj,act) in activity_states: {len(states_per_act)}")
    print(f"Total activities: {len(act_planned)}")
    print(f"Activities with 0 state records: {sum(1 for k in act_planned if k not in {a for a in states_per_act})}")
    print()

    # distribution of state count per activity
    counts = list(states_per_act.values())
    print(f"State records per activity: min={min(counts)} max={max(counts)} "
          f"mean={sum(counts)/len(counts):.1f} median={sorted(counts)[len(counts)//2]}")
    print()
    print("State-count distribution buckets:")
    buckets = collections.Counter()
    for c in counts:
        if c <= 10: buckets["1-10"] += 1
        elif c <= 30: buckets["11-30"] += 1
        elif c <= 60: buckets["31-60"] += 1
        elif c <= 100: buckets["61-100"] += 1
        elif c <= 200: buckets["101-200"] += 1
        else: buckets["200+"] += 1
    for b in ["1-10","11-30","31-60","61-100","101-200","200+"]:
        print(f"  {b}: {buckets[b]}")

    print()
    print("Correlation check: state_count vs planned_duration_days")
    act_with_states = [(aid, states_per_act[(ap, aid)], act_planned[aid])
                       for (ap, aid), sc in states_per_act.items() if aid in act_planned]
    n = len(act_with_states)
    if n > 0:
        # simple Pearson-ish scatter stats
        x = [a[1] for a in act_with_states]  # state count
        y = [a[2] for a in act_with_states]  # planned duration
        mean_x = sum(x)/n
        mean_y = sum(y)/n
        cov = sum((xi-mean_x)*(yi-mean_y) for xi,yi in zip(x,y))
        var_x = sum((xi-mean_x)**2 for xi in x)
        var_y = sum((yi-mean_y)**2 for yi in y)
        if var_x > 0 and var_y > 0:
            corr = cov / (var_x**0.5 * var_y**0.5)
        else:
            corr = 0
        print(f"  n={n} Pearson r(state_count, planned_duration) = {corr:.3f}")
        print(f"  state_count: mean={mean_x:.1f}  planned_duration: mean={mean_y:.1f}")
        # ratio
        ratios = [s/p for (_, s, p) in act_with_states if p > 0]
        print(f"  ratio state_count/planned_duration: min={min(ratios):.2f} max={max(ratios):.2f} "
              f"mean={sum(ratios)/len(ratios):.2f} median={sorted(ratios)[len(ratios)//2]:.2f}")

    print()
    print("Temporal coverage per activity (sample = first 15 activities in P00001):")
    sample_acts = [a["activity_id"] for a in acts if a["project_id"]=="P00001"][:15]
    for aid in sample_acts:
        recs = [s for s in states if s["activity_id"]==aid]
        if not recs:
            print(f"  {aid}: NO STATE RECORDS")
            continue
        dates = sorted(s["date"] for s in recs)
        days = sorted(int(s["day_index"]) for s in recs)
        prods = [float(s["productivity_index"]) for s in recs]
        print(f"  {aid}: planned={act_planned.get(aid,'?')} days, "
              f"state_records={len(recs)}, date_range={dates[0]}..{dates[-1]}, "
              f"day_range={days[0]}..{days[-1]}, "
              f"productivity_range=[{min(prods):.2f}, {max(prods):.2f}], "
              f"mean_prod={sum(prods)/len(prods):.2f}")

    print()
    print("Does productivity_index drop to ~0 after planned end?")
    print("(Checking P00001_A0001 as a deep example)")
    aid = "P00001_A0001"
    recs = sorted([s for s in states if s["activity_id"]==aid], key=lambda s: int(s["day_index"]))
    planned = act_planned.get(aid)
    if recs and planned:
        print(f"  {aid}: planned_duration={planned} days")
        print(f"  first 10 days:")
        for s in recs[:10]:
            print(f"    day={s['day_index']} date={s['date']} prod={s['productivity_index']} "
                  f"weather_risk={s['weather_risk']} site_access={s['site_access_index']} "
                  f"event_pressure={s['event_pressure']}")
        print(f"  around planned end (day {planned-2} to {planned+5}):")
        for s in recs:
            d = int(s["day_index"])
            if planned-2 <= d <= planned+5:
                print(f"    day={d} date={s['date']} prod={s['productivity_index']} "
                      f"weather_risk={s['weather_risk']} site_access={s['site_access_index']} "
                      f"event_pressure={s['event_pressure']}")
        print(f"  last 5 records:")
        for s in recs[-5:]:
            print(f"    day={s['day_index']} date={s['date']} prod={s['productivity_index']} "
                  f"weather_risk={s['weather_risk']} site_access={s['site_access_index']} "
                  f"event_pressure={s['event_pressure']}")

    print()
    print("DONE")

if __name__ == "__main__":
    main()
