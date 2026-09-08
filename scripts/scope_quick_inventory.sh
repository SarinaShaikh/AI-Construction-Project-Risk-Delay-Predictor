#!/usr/bin/env bash
set -euo pipefail
ROOT="AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

echo "=== FILE INVENTORY (rows / cols / sample) ==="
for f in projects activities dependencies resources resource_allocation environment activity_states procurement decisions events rework friction_canonical intervention_options counterfactuals construction_memory outcomes friction_summary_canonical master_projects_canonical; do
  fn="$ROOT/$f.csv"
  lines=$(tail -n +2 "$fn" | wc -l)
  cols=$(head -1 "$fn" | awk -F',' '{print NF}')
  echo "--- $f.csv  rows=$lines  cols=$cols"
  head -2 "$fn" | tail -1
  echo
done

echo "=== MISSING CELLS PER TABLE ==="
for f in projects activities dependencies resources resource_allocation environment activity_states procurement decisions events rework friction_canonical intervention_options counterfactuals construction_memory outcomes friction_summary_canonical master_projects_canonical; do
  fn="$ROOT/$f.csv"
  miss=$(tail -n +2 "$fn" | awk -F',' '{for(i=1;i<=NF;i++) if($i=="") c++} END{print c+0}')
  echo "$f.csv  empty_cells=$miss"
done

echo "=== DEPENDENCIES: relationship & lag distribution ==="
echo "-- relationships --"
tail -n +2 "$ROOT/dependencies.csv" | cut -d',' -f4 | sort | uniq -c
echo "-- lag_days (top 15) --"
tail -n +2 "$ROOT/dependencies.csv" | cut -d',' -f5 | sort -n | uniq -c | sort -rn | head -15

echo "=== ACTIVITIES enums ==="
echo "-- status --"; tail -n +2 "$ROOT/activities.csv" | cut -d',' -f13 | sort | uniq -c
echo "-- phase --"; tail -n +2 "$ROOT/activities.csv" | cut -d',' -f3 | sort | uniq -c
echo "-- resource_type --"; tail -n +2 "$ROOT/activities.csv" | cut -d',' -f7 | sort | uniq -c
echo "-- critical_path --"; tail -n +2 "$ROOT/activities.csv" | cut -d',' -f14 | sort | uniq -c
echo "-- critical_path_position summary (min/max) --"
tail -n +2 "$ROOT/activities.csv" | cut -d',' -f15 | sort -n | sed -n '1p;$p'
echo "-- predecessor_count summary (min/max) --"
tail -n +2 "$ROOT/activities.csv" | cut -d',' -f17 | sort -n | sed -n '1p;$p'
echo "-- successor_count summary (min/max) --"
tail -n +2 "$ROOT/activities.csv" | cut -d',' -f18 | sort -n | sed -n '1p;$p'

echo "=== numeric validity: negative/zero durations in activities ==="
echo "-- planned_duration_days <0 --"
tail -n +2 "$ROOT/activities.csv" | awk -F',' '$8<0{c++} END{print c+0}'
echo "-- planned_duration_days ==0 --"
tail -n +2 "$ROOT/activities.csv" | awk -F',' '$8==0{c++} END{print c+0}'

echo "=== DECISIONS enums ==="
echo "-- decision_status --"; tail -n +2 "$ROOT/decisions.csv" | cut -d',' -f9 | sort | uniq -c
echo "-- decision_debt_level --"; tail -n +2 "$ROOT/decisions.csv" | cut -d',' -f14 | sort | uniq -c

echo "=== EVENTS enums ==="
echo "-- event_type --"; tail -n +2 "$ROOT/events.csv" | cut -d',' -f5 | sort | uniq -c
echo "-- severity --"; tail -n +2 "$ROOT/events.csv" | cut -d',' -f6 | sort | uniq -c

echo "=== ENVIRONMENT enum + date range ==="
echo "-- extreme_weather --"; tail -n +2 "$ROOT/environment.csv" | cut -d',' -f8 | sort | uniq -c
echo "-- weather_risk --"; tail -n +2 "$ROOT/environment.csv" | cut -d',' -f9 | sort | uniq -c
echo "-- date range --"
tail -n +2 "$ROOT/environment.csv" | cut -d',' -f2 | sort | sed -n '1p;$p'

echo "=== PROCUREMENT ==="
echo "-- material --"; tail -n +2 "$ROOT/procurement.csv" | cut -d',' -f3 | sort | uniq -c
echo "-- procurement_delay_days (min/max) --"
tail -n +2 "$ROOT/procurement.csv" | cut -d',' -f6 | sort -n | sed -n '1p;$p'

echo "=== INTERVENTION_OPTIONS ==="
echo "-- category --"; tail -n +2 "$ROOT/intervention_options.csv" | cut -d',' -f7 | sort | uniq -c
echo "-- intervention_type --"; tail -n +2 "$ROOT/intervention_options.csv" | cut -d',' -f6 | sort | uniq -c
echo "-- true_effectiveness (min/max) --"
tail -n +2 "$ROOT/intervention_options.csv" | cut -d',' -f10 | sort -n | sed -n '1p;$p'

echo "=== COUNTERFACTUALS ==="
echo "-- baseline_delay_days (min/max) --"
tail -n +2 "$ROOT/counterfactuals.csv" | cut -d',' -f5 | sort -n | sed -n '1p;$p'
echo "-- counterfactual_delay_days (min/max) --"
tail -n +2 "$ROOT/counterfactuals.csv" | cut -d',' -f6 | sort -n | sed -n '1p;$p'
echo "-- avoided_delay_days (min/max/negatives) --"
tail -n +2 "$ROOT/counterfactuals.csv" | cut -d',' -f7 | sort -n | sed -n '1p;$p'
neg=$(tail -n +2 "$ROOT/counterfactuals.csv" | awk -F',' '$7<0{c++} END{print c+0}')
echo "   negatives: $neg"

echo "=== CONSTRUCTION_MEMORY missing columns (top 12) ==="
for col in $(head -1 "$ROOT/construction_memory.csv" | awk -F',' '{for(i=1;i<=NF;i++) print i":"$i}'); do
  i=${col%%:*}; name=${col##*:}
  miss=$(tail -n +2 "$ROOT/construction_memory.csv" | awk -F',' -v c=$i '{if($c=="") n++} END{print n+0}')
  if [ "$miss" -gt 0 ]; then
    echo "  $name  missing=$miss"
  fi
done | sort -t= -k2 -rn | head -12

echo "=== MASTER_VS_OUTCOMES: total_activities & total_planned_duration ==="
join -t',' -1 1 -2 1 <(tail -n +2 "$ROOT/master_projects_canonical.csv" | sort -t',' -k1,1) <(tail -n +2 "$ROOT/outcomes.csv" | sort -t',' -k1,1) | awk -F',' '{if($14!=$4 || $15!=$5){print $1,"MISMATCH master=(" $14 "," $15 ") outcomes=(" $4 "," $5 ")"; m++}} END{print "mismatches=" m+0}'

echo "=== DEPENDENCY CROSS-PROJECT CHECK ==="
# A dependency's predecessor/successor id contains the project id as prefix P####_
# Check that the project_id field matches that prefix for each row
awk -F',' 'NR>1{
  split($2,a,"_"); proj=a[1];
  if(proj!=$1) cp++;
  split($3,b,"_"); proj2=b[1];
  if(proj2!=$1) cp2++;
} END{print "cross-project refs (pred): " cp+0; print "cross-project refs (succ): " cp2+0}' "$ROOT/dependencies.csv"

echo "DONE"
