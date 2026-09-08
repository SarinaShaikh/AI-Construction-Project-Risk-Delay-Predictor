#!/usr/bin/env bash
set -euo pipefail
ROOT="AI-Construction-Project-Risk-Delay-Predictor/data/raw/SCOPE_v02_Public"

echo "SCOPE v0.2 — Project split verification (shell-based, fast)"
echo "============================================================"

echo "--- projects.csv split distribution ---"
tail -n +2 "$ROOT/projects.csv" | cut -d',' -f14 | sort | uniq -c

echo "--- Expected vs actual ---"
train=$(tail -n +2 "$ROOT/projects.csv" | awk -F',' '$14=="train"{c++} END{print c+0}')
val=$(tail -n +2 "$ROOT/projects.csv" | awk -F',' '$14=="validation"{c++} END{print c+0}')
test=$(tail -n +2 "$ROOT/projects.csv" | awk -F',' '$14=="test"{c++} END{print c+0}')
echo "train=$train  validation=$val  test=$test"
if [ "$train" -eq 70 ] && [ "$val" -eq 15 ] && [ "$test" -eq 15 ]; then
  echo "MATCH: 70/15/15"
else
  echo "MISMATCH"
fi

echo ""
echo "--- Split consistency across all tables (project_id, split values) ---"
for f in activities dependencies resources resource_allocation environment activity_states procurement decisions events rework friction_canonical intervention_options counterfactuals construction_memory outcomes friction_summary_canonical master_projects_canonical; do
  fn="$ROOT/$f.csv"
  # get unique (project_id, split) pairs
  n_unique=$(tail -n +2 "$fn" | awk -F',' '{print $1","$NF}' | sort -u | wc -l)
  # check for any project_id with mixed split
  mixed=$(tail -n +2 "$fn" | awk -F',' '{print $1","$NF}' | sort | uniq -c | awk '$1>1{print $2}' | wc -l)
  echo "$f.csv: unique(project_id,split)=$n_unique  mixed_split_projects=$mixed"
done

echo ""
echo "--- Project IDs per split (first 10) ---"
for s in train validation test; do
  echo "  $s:"
  tail -n +2 "$ROOT/projects.csv" | awk -F',' -v s="$s" '$14==s{print $1}' | head -10
  echo ""
done

echo "--- Verify outcomes/master have same project set as projects.csv ---"
proj_set=$(tail -n +2 "$ROOT/projects.csv" | cut -d',' -f1 | sort)
out_set=$(tail -n +2 "$ROOT/outcomes.csv" | cut -d',' -f1 | sort)
mst_set=$(tail -n +2 "$ROOT/master_projects_canonical.csv" | cut -d',' -f1 | sort)
extra_out=$(comm -13 <(echo "$proj_set") <(echo "$out_set") | wc -l)
missing_out=$(comm -23 <(echo "$proj_set") <(echo "$out_set") | wc -l)
extra_mst=$(comm -13 <(echo "$proj_set") <(echo "$mst_set") | wc -l)
missing_mst=$(comm -23 <(echo "$proj_set") <(echo "$mst_set") | wc -l)
echo "outcomes:  extra=$extra_out  missing=$missing_out"
echo "master:    extra=$extra_mst  missing=$missing_mst"

echo ""
echo "DONE"
