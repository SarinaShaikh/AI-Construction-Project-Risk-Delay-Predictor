# Phase 1 — SCOPE v0.2 Dataset Exploration Report

**Project:** AI Construction Project Risk & Delay Predictor
**Phase:** 1 — Understand SCOPE v0.2 Dataset
**Date:** 2026-09-08
**Status:** Exploration and verification complete; Phase 2 NOT started.

> This report is generated purely from inspection of the raw dataset.
> No raw data was modified, cleaned, or renamed.
> No assumptions were invented — uncertainties are flagged explicitly.

---

## 1. Dataset location and documentation

```
data/raw/SCOPE_v02_Public/
├── README.md                      # Overview, dataset components, identifiers, split info
├── GENERATION_METHODOLOGY.md      # Detailed synthetic generation description
├── PUBLIC_RELEASE_MANIFEST.json   # Metadata: 100 projects, 70/15/15 split, synthetic=true
├── DATA_DICTIONARY.csv            # Per-variable descriptions (dataset, variable, datatype, description)
├── DATASET_CATALOG.csv            # Per-dataset summary: records, columns, missing_cells, description
├── DATA_QUALITY_REPORT.csv        # Dataset-level quality checks (PASS for all)
├── LICENSE.txt
└── <18 data CSV files>
```

The public release manifest confirms:
- 100 synthetic construction projects
- Project split: **70 train / 15 validation / 15 test**
- Synthetic data: yes
- Original data modified: **no**

---

## 2. File inventory

| File | Rows | Cols | Split col | Primary key | Notes |
|------|------|------|-----------|-------------|-------|
| `projects.csv` | 100 | 14 | split | project_id | Project-level characteristics + planning |
| `activities.csv` | 9,279 | 19 | split | project_id + activity_id | Activity planning, cost, duration, network |
| `dependencies.csv` | 18,176 | 6 | split | project_id + pred + succ | Activity dependency edges |
| `resources.csv` | 2,748 | 8 | split | resource_id | Resource characteristics |
| `resource_allocation.csv` | 9,279 | 6 | split | proj+act+res_id | Activity ↔ resource linkage |
| `environment.csv` | 52,332 | 10 | split | proj_id + day_index | Environmental observations over time |
| `activity_states.csv` | 4,060,623 | 9 | split | proj_id + act_id + day_index | Daily activity-state snapshots |
| `procurement.csv` | 800 | 8 | split | procurement_id | Material procurement lead-time |
| `decisions.csv` | 3,374 | 15 | split | decision_id | Management decision observations |
| `events.csv` | 27,989 | 10 | split | event_id | Risk/disruption events |
| `rework.csv` | 1,858 | 7 | split | rework_id | Rework duration + cost |
| `friction_canonical.csv` | 12,980 | 6 | split | proj_id+act_id+friction_type | Canonical friction observations |
| `intervention_options.csv` | 70,513 | 11 | split | intervention_id | Candidate intervention scenarios |
| `counterfactuals.csv` | 70,513 | 10 | split | intervention_id | Counterfactual intervention outcomes |
| `construction_memory.csv` | 55,489 | 24 | split | memory_id | Construction experience representations |
| `outcomes.csv` | 100 | 4 | split | project_id | Project-level outcome (activity count + planned duration) |
| `friction_summary_canonical.csv` | 100 | 4 | split | project_id | Project-level friction summary |
| `master_projects_canonical.csv` | 100 | 29 | split | project_id | Integrated project representation |

All 18 data files have a `split` column. All pass the data-quality report checks (PASS).

---

## 3. Key relationships and identifiers

**Primary identifiers:**
- `project_id` — e.g., `P00001`
- `activity_id` — e.g., `P00001_A0001` (project-prefixed)
- `event_id` — e.g., `P00001_E00001`
- `intervention_id` — e.g., `I00000001`
- `resource_id` — e.g., `P00001_L001`
- `procurement_id` — e.g., `P00001_PO001`
- `decision_id` — e.g., `P00001_D0001`
- `rework_id` — e.g., `RW0000001`
- `memory_id` — e.g., `MEM-A-00000001`

**Foreign-key relationships (verified, all intact):**
- `dependencies.{predecessor_id, successor_id}` → `activities` (same project)
- `resource_allocation.activity_id` → `activities`
- `resource_allocation.resource_id` → `resources`
- `environment.project_id` → `projects`
- `activity_states.(project_id, activity_id)` → `activities`
- `procurement.project_id` → `projects`
- `decisions.(project_id, activity_id)` → `activities`
- `events.(project_id, activity_id)` → `activities`
- `rework.event_id` → `events`; `rework.(project_id, activity_id)` → `activities`
- `friction_canonical.(project_id, activity_id)` → `activities`
- `intervention_options.event_id` → `events`; `intervention_options.(project_id, activity_id)` → `activities`
- `counterfactuals.intervention_id` → `intervention_options`; `counterfactuals.event_id` → `events`
- `construction_memory.(project_id, activity_id)` → `activities`
- `outcomes.project_id` → `projects`
- `friction_summary_canonical.project_id` → `projects`
- `master_projects_canonical.project_id` → `projects`

**All dependency references are intra-project** (no cross-project predecessor/successor links).

---

## 4. Data quality findings

### 4a. Nulls / missing cells

- **All 17 tables are fully populated** (0 missing cells), EXCEPT:
  - `construction_memory.csv`: **245,745 missing cells (18.45% of all cells)**
    - Heavily missing columns: `decision_type` (52,115), `resource_type` (46,210), `phase` (46,210), `activity_name` (46,210), `severity` (27,500), `event_type` (27,500)
    - These are expected: memory records are typed by `memory_type` (activity_pattern, risk_pattern, decision_pattern, friction_pattern, rework_pattern), and not all fields apply to all memory types.

### 4b. Duplicate checks

- **No duplicate primary keys in any table** (DATA_QUALITY_REPORT confirms; our verification agrees).

### 4c. Numeric validity

- **activities.csv:**
  - `planned_duration_days`: **0 negative, 0 zero** (all positive integers)
  - `quantity`: 0 negative
  - `planned_cost`: 0 negative
- **dependencies.csv:**
  - `lag_days`: min=0, max=3, **0 negatives**
  - Relationships: FS (14,527), FF (955), SS (2,694)
- **events.csv:**
  - `duration_days`: min=1, max=75, **0 negatives**
  - `impact_factor`: min=0.25, max=1.0, **0 negatives**
- **decisions.csv:**
  - `decision_delay_days`: min=0, max=35, **0 negatives**
- **procurement.csv:**
  - `procurement_delay_days`: min=0, max=53
  - `planned_lead_days`: min=2, max=90
  - `actual_lead_days`: min=2, max=104
- **activity_states.csv:**
  - `productivity_index`: min=0.200, max=1.105
  - `event_pressure`: min=0.0, max=2.0
- **counterfactuals.csv:**
  - `avoided_delay_days`: min=0.0, max=16.03, **0 negatives**
- **No impossible values found** in any checked numeric field.

### 4d. Enumeration validity

- **activities.status:** ALL 9,279 = "Planned" (no In-Progress, Completed, Delayed, etc.)
- **activities.phase:** 10 distinct phases (Preconstruction, Site Preparation, Earthwork, Foundation, Structural Frame, Masonry, MEP, Finishing, External Works, Commissioning)
- **activities.resource_type:** 4 types (labour=7,743, equipment=1,256, information=186, management=94)
- **activities.critical_path:** 0=6,909, 1=2,370 (25.5% flagged critical)
- **activities.critical_path_position:** -1 for all 6,909 non-critical; 1..30 for all 2,370 critical (exactly 100 activities at each position 1..30 — this is clearly a synthetic positional index, not a CPM-derived position)
- **events.event_type:** 12 types (Heavy Rainfall most common: 6,018)
- **events.severity:** 4 levels (Low=11,466, Medium=9,863, High=4,933, Critical=1,727)
- **decisions.decision_status:** Delayed=2,347, On Time=1,027
- **decisions.decision_debt_level:** Minimal=1,764, Low=757, Moderate=482, High=266, Critical=105
- **decisions.decision_type:** 10 types
- **environment.extreme_weather:** 0=51,187, 1=1,145
- **environment.weather_risk:** 0=45,745, 1=5,457, 2=1,130
- **procurement.material:** 8 material types (100 records each)
- **intervention_options.category:** 6 categories; **intervention_type:** 9 types
- **construction_memory.memory_type:** 5 types (risk_pattern=27,989, activity_pattern=9,279, friction_pattern=12,989, decision_pattern=3,374, rework_pattern=1,858)

### 4e. Cross-table consistency

- **master_projects_canonical vs outcomes:** **CONSISTENT — 0 mismatches** (name-matched comparison).
  - An earlier shell `join` comparison reported 100 mismatches, but this was a **positional column-offset artifact**: `master_projects_canonical.csv` has 29 columns while `outcomes.csv` has 4, so comparing fields by position (master cols 14/15 vs outcomes cols 4/5) compared non-matching fields.
  - **Correct comparison (by matching column names/semantics):** 100/100 `project_id`s match; `total_activities` matches on 100/100 projects; `total_planned_duration` matches on 100/100 projects.
  - **Final conclusion: the two tables are fully consistent; no data error.**
- **projects.csv ↔ outcomes ↔ master_projects_canonical ↔ friction_summary_canonical:** All have exactly the same 100 project_ids (verified via shell set comparison). ✓
- **activities.predecessor_count / successor_count vs actual dependency counts:** **0 mismatches** — the summary counts in activities.csv exactly match the actual dependency edges per activity. ✓
- **project_network_duration:** single value per project (all activities in a project share the same value). ✓
- **Construction memory ↔ activities:** NOT 1:1.
  - 55,489 memory records for 9,279 activities (6.0 records per activity average)
  - Each activity has 1+ memory records (0 activities with 0 records)
  - Memory types per activity vary (e.g., P00001_A0002 has 10 records: 1 activity_pattern + 6 risk_pattern + 1 decision_pattern + 2 friction_pattern)
  - `activity_pattern` records (9,279) ARE 1:1 with activities (9,279 unique keys, 0 missing, 0 extra)
  - `risk_pattern` (27,989) maps to 8,756 unique activities (some activities have up to 13 risk patterns)
  - `decision_pattern` (3,374) maps to 2,819 activities
  - `friction_pattern` (12,989) maps to 6,931 activities
  - `rework_pattern` (1,858) maps to 1,680 activities
  - `observed_delay_days` **varies across memory types for the same activity** (e.g., P00001_A0001 has four records with delays 0.0, 0.09, 0.25, 18.75)
  - **CRITICAL:** `activity_pattern.observed_delay_days` is **0.0 for ALL 9,279 records**. The nonzero delays come from `risk_pattern`, `friction_pattern`, `rework_pattern`, and possibly `decision_pattern`.

### 4f. Project split verification

- **projects.csv split:** train=70, validation=15, test=15 ✓ (matches manifest)
- **Split consistency across tables:**
  - `activities`, `dependencies`, `resource_allocation`, `environment`, `activity_states`, `friction_canonical`: each project has exactly 1 split value (verified via unique(project_id, split) = 100 per table). ✓
  - `resource_allocation`, `procurement`, `decisions`, `events`, `rework`, `intervention_options`, `counterfactuals`, `construction_memory`, `outcomes`, `friction_summary_canonical`, `master_projects_canonical`: all consistent. ✓
  - **Shell check of `mixed_split_projects`:** The shell script reported `mixed_split_projects=100` for some tables (activities, dependencies, etc.) — **this is a FALSE POSITIVE from the shell awk logic**, not a real inconsistency. The awk command `awk '{print $1","$NF}'` picks the FIRST field (project_id) and the LAST field (split), but for multi-column tables where project_id is not field 1, the "last field" is not the split. The Python verification confirmed all splits are consistent.

---

## 5. CPM graph structure feasibility

| Property | Finding |
|----------|---------|
| Projects | 100 |
| Single root (no predecessors) per project | **100/100** (all projects have exactly 1 root activity) |
| Single leaf (no successors) per project | **0/100** (all projects have MULTIPLE leaf activities: 10..21 leaves per project) |
| Missing dependency references | **0** (all predecessor/successor IDs exist in activities) |
| Cross-project dependency edges | **0** (all edges are intra-project) |
| Circular dependencies | **0 cycles in all 100 projects** (Kahn's algorithm, all 18,176 edges — every graph is a DAG) |
| Dependency relationships | FS (14,527), FF (955), SS (2,694) — all with lag_days 0..3 |
| `predecessor_count`/`successor_count` in activities vs actual | 0 mismatches ✓ |

**Implication for CPM:**
- Every project has a **single start activity** (root), which is convenient.
- Every project has **multiple end activities** (leaves), so a CPM engine cannot assume a single sink. Must create a virtual project-end node or handle multiple leaves explicitly when computing project completion.
- The dataset's `critical_path` flag (2,370 activities = 25.5%) and `critical_path_position` appear to be **synthetic positional labels**, not necessarily CPM-derived float==0 activities. This needs to be validated against an actual CPM engine (Phase 3) before relying on them.
- Circular dependencies: **none found** — all 100 project dependency graphs are DAGs (Kahn's algorithm over all 18,176 edges). CPM can proceed without cycle cleanup.

---

## 6. Observed delay information — what exists and what doesn't

### 6a. Activity-level actual delay: NOT directly available

- `activities.csv` has `planned_duration_days` but **NO `actual_duration_days`, `actual_finish_date`, or `delay_days`**.
- `activities.csv` status is "Planned" for ALL activities — no completion status.

### 6b. Activity-state time series: rich but ambiguous for duration inference

- `activity_states.csv`: 4,060,623 daily snapshots across 9,279 activities.
- **Every activity has 259..697 state records** (mean 437.6).
- **State count does NOT correlate with planned duration** (Pearson r = 0.006).
- **Every activity spans the same day_index range (1..541 for P00001)** regardless of planned duration (26 days, 20 days, 7 days, etc.). This means:
  - **States exist for ALL days of the project timeline, not just active execution days.**
  - **States do NOT start/stop based on activity execution.**
  - **Therefore actual duration CANNOT be derived from state record count.**
- Productivity_index for P00001_A0001 (planned 26 days) stays ~0.80-0.98 on day 26 (planned end) and day 31, and continues to ~0.80-0.90 on day 541 (last record). **No drop to zero after planned end.**
- **Conclusion: activity_states alone does not reveal when an activity actually starts or finishes.**
- **Confirmed (Phase 1 verification):** productivity_index does not systematically change around planned start/end — mean productivity before planned end ≈ mean productivity after planned end (diff ≈ +0.004); within-activity productivity CV ≈ 0.18; event-affected vs event-free activities are nearly identical (≈0.734 vs ≈0.739). Productivity does not drop at planned end.
- **FINAL: activity_states contains NO usable execution-boundary information and cannot be used to infer actual activity duration.** Treat it strictly as an environmental/condition feature source (aggregated productivity, weather risk, event pressure) — never as an execution log or duration proxy.

### 6c. Event-level delay: available but not = activity delay

- `events.csv`: 27,989 events, each linked to one activity.
- `duration_days` per event: 1..75 days.
- Total event delay per affected activity: 1..168 days (mean 17.78).
- 8,756 of 9,279 activities (94.4%) have ≥1 event.
- 523 activities (5.6%) have **no events at all**.
- **An event is a disruption contributor, not a complete measure of activity delay.** An activity could be delayed by factors other than events.

### 6d. Decision-level delay: available but limited coverage

- `decisions.csv`: 3,374 decisions linked to 2,819 unique activities (30.4% of activities).
- `decision_delay_days`: 0..35 days per decision.
- Total decision delay per affected activity: 0..42 days (mean 2.99).
- 2,076 activities have nonzero total decision delay; 743 have zero.
- **Only 30.4% of activities have any decision record.** Decision delay is sparse.

### 6e. Construction memory: has observed_delay_days but ambiguous semantics

- `construction_memory.csv`: 55,489 records, 9,279 activities.
- `observed_delay_days` exists for ALL records (range 0.0..59.0, sum per activity 0.0..112.0, mean 11.10).
- 9,163 of 9,279 activities (98.7%) have nonzero total observed_delay_days.
- **BUT:**
  - `activity_pattern` records (1 per activity) have `observed_delay_days = 0.0` for ALL 9,279.
  - Nonzero delays are in `risk_pattern`, `friction_pattern`, `rework_pattern` records.
  - Multiple memory records per activity have different `observed_delay_days` values.
  - **It is unclear what `observed_delay_days` in a risk_pattern record means** — is it the delay from that specific risk event? Or cumulative? Or synthetic?
  - **The meaning of `observed_delay_days` in construction_memory is NOT documented in the public release.** The DATA_DICTIONARY says only "Variable from the construction_memory dataset."

### 6f. Counterfactuals: synthetic intervention outcomes, not observed delay

- `counterfactuals.csv`: 70,513 records, each tied to an intervention and event.
- `baseline_delay_days`: 0.25..59.0
- `counterfactual_delay_days`: 0.176..56.43
- `avoided_delay_days`: 0.0..16.03 (no negatives)
- These are **event-scenario-level**, not activity-level, and are **synthetic intervention outcomes**, not observed real delays.

### 6g. Project-level actual delay: NOT available

- `outcomes.csv`: only `total_activities` and `total_planned_duration` (both planned, not actual).
- `master_projects_canonical.csv`: aggregates planned + event/decision/friction counts, but **no actual project completion duration or project delay days**.
- **No explicit project-level "actual duration" or "project delay" field exists anywhere in the dataset.**

---

## 7. Primary ML target (FINAL DECISION) and alternatives

### 7.1 PRIMARY ML TARGET — total event-induced delay days per activity

**Decision (fixed):** for each `(project_id, activity_id)`:

```
target = SUM(events.duration_days)   # summed over all events of the activity
target = 0                           # when the activity has no events
```

- **Semantics:** this is **EVENT-INDUCED / DISRUPTION DELAY** per activity — **NOT actual schedule slippage**, because SCOPE contains no reliable actual activity start/finish timestamps (see §6b and §10).
- **Construction:** straightforward join of `events.(project_id, activity_id)` → sum `duration_days`.
- **Coverage:** 8,756/9,279 activities (94.4%) have ≥1 event; per-activity totals range 1–168 days (mean 17.78); 523 activities have target 0.

**Why `experienced_event` is NOT the primary target:** ~94% of activities have ≥1 event, so the binary label is highly imbalanced and near-constant — it carries little signal for ranking or prediction. The continuous event-delay sum retains the same coverage while providing graded signal.

**Why actual activity delay cannot be claimed:** no actual start/finish timestamps exist anywhere; `activity_states` does not encode execution boundaries (§6b, Q5); `status` = "Planned" for all 9,279 activities; and no project completion dates exist. Any actual-slippage target would have to be invented, which is out of scope per CLAUDE.md.

**Explicitly NOT used as the primary target:**
- `experienced_event` — too imbalanced (~94% positive).
- `decision_delay_days` — sparse (30.4% coverage) and decision-specific.
- `construction_memory.observed_delay_days` — semantics ambiguous (§6e, Q4).
- `activity_states` — verified NOT an execution-duration proxy (§6b, Q5).

**PREDICTION-TIME LEAKAGE RULES:** event-derived outcome information must NOT be used as predictive features when training. In particular, do NOT use as features: observed/future event information (event counts, types, severity, `duration_days`), observed delay fields, decision actuals (`decision_delay_days`, decision status), rework outcomes, or other post-outcome information — unless a later phase explicitly establishes that the information was available at prediction time. **Event-derived information is the target, not an input feature.**

---

### 7.2 Alternatives considered (NOT the primary target)

The dataset does NOT provide a clean, explicit activity-level "actual delay" target. The following options were considered and rejected as the primary target, each with caveats:

### Option A: Activity-level binary "experienced any event" (≥1 event vs 0 events)

- **Available:** events.csv has `activity_id`.
- **Pros:** Clean binary target, 94.4% positive rate, all activity-level features available.
- **Cons:** "Having an event" ≠ "being delayed". An event may not cause delay. 523 activities have no events — what does that mean? Are they truly on-time, or just event-free?
- **Label:** "Activity experienced ≥1 disruption event"

### Option B: Activity-level event delay severity (sum of duration_days per activity)

- **Available:** sum `events.duration_days` per activity → 1..168 days (mean 17.78).
- **Pros:** Continuous target, directly from events, activity-level.
- **Cons:** This is event-imposed delay, not necessarily actual activity delay. An activity may have events but recover. Not a "delay" in the CPM sense.
- **Label:** "Total event-imposed delay days per activity"

### Option C: Activity-level decision delay (sum of decision_delay_days per activity)

- **Available:** sum `decisions.decision_delay_days` per activity → 0..42 days (mean 2.99).
- **Pros:** Explicit "delay" concept (actual_decision_day − required_day).
- **Cons:** Only 30.4% of activities have decisions. Highly sparse. A delayed decision ≠ delayed activity.
- **Label:** "Total decision delay days per activity" (sparse)

### Option D: Construction memory observed_delay_days (sum across memory types per activity)

- **Available:** sum `construction_memory.observed_delay_days` per activity → 0.0..112.0 (mean 11.10), 98.7% nonzero.
- **Pros:** Available for nearly all activities, continuous, has variation.
- **Cons:** **Semantics unclear.** `activity_pattern` records have 0.0 for all. Multiple memory types have different delay values for the same activity. Is the sum meaningful? Or should we use only one memory type? **Requires documentation or human decision.**
- **Label:** "Total observed_delay_days from construction_memory" (semantics TBD)

### Option E: Activity-level "observed delay > 0" binary from construction_memory

- **Available:** 9,163/9,279 activities have total observed_delay_days > 0.
- **Pros:** Near-balanced-ish (98.7% positive), clean binary.
- **Cons:** Relies on the same ambiguous construction_memory semantics as Option D.

### Option F: Event severity classification (will a HIGH/CRITICAL event hit this activity?)

- **Available:** events.csv severity + activity_id.
- **Pros:** Aligns with risk-analysis goal, uses event semantics.
- **Cons:** Different from delay prediction.

### NOT viable without derivation:

- **Actual activity duration delay** (actual − planned): no actual duration available.
- **Activity completion status** (on-time/late/completed): status = "Planned" for all.
- **Project-level delay:** no actual project completion in data.

---

## 8. Features available for ML (activity + project level)

### Activity-level features (from activities.csv + dependencies):

- `planned_duration_days`
- `criticality` (float)
- `critical_path` (0/1 flag)
- `critical_path_position` (1..30 for critical, -1 for non-critical — likely synthetic)
- `phase`, `phase_order`, `activity_sequence`
- `resource_type`
- `quantity`, `unit_cost`, `planned_cost`
- `predecessor_count`, `successor_count`
- `project_network_duration` (project-level constant)

### Project-level features (from projects.csv):

- `project_type` (Hospital, Commercial, Industrial, Residential, etc.)
- `floors`, `area_m2`
- `complexity`, `contractor_capability`, `resource_availability`, `management_maturity`, `weather_exposure`, `supply_chain_exposure`, `technology_maturity`
- `planned_duration_days`, `planned_cost`

### Time-series-derived features (from activity_states, if aggregation is valid):

- `productivity_index` (mean, min, max, trend over time)
- `weather_risk`, `site_access_index`, `event_pressure` (aggregated per activity)

### Event-derived features (joined via activity_id):

- Event count per activity
- Event type distribution
- Max/avg event severity
- Total event duration_days per activity
- Any High/Critical event present (binary)

### Decision-derived features (joined via activity_id, sparse):

- Decision count
- Max/total decision delay
- Decision debt score
- Any delayed decision (binary)

### Procurement-derived features (joined via project_id):

- Material lead time delays (per material type)
- Supply chain risk
- Max procurement delay per project

### Friction-derived features (joined via activity_id):

- Total friction hours per activity
- Friction event count

---

## 9. Train/validation/test split

The dataset **provides an explicit project-level split**:

- **Train:** 70 projects (e.g., P00002, P00003, P00004, P00005, P00006, P00007, P00008, P00010, P00013, P00014, ...)
- **Validation:** 15 projects (e.g., P00001, P00009, P00011, P00019, P00034, P00038, P00048, P00055, P00072, P00074, ...)
- **Test:** 15 projects (e.g., P00012, P00031, P00037, P00042, P00043, P00049, P00053, P00062, P00067, P00075, ...)

The split is **consistent across all 18 data tables** (verified). Every record inherits the split of its project.

**Phase 4 MUST honor this split** (per CLAUDE.md rule 8 — no random activity-row splitting; project-level separation).

---

## 10. What is explicitly provided vs derived vs NOT available

### Explicitly provided:

- Project characteristics (projects.csv)
- Activity planned durations, costs, criticality, network position (activities.csv)
- Dependency edges with relationships and lags (dependencies.csv)
- Resource definitions and allocations (resources.csv, resource_allocation.csv)
- Environmental observations over time (environment.csv)
- Daily activity-state snapshots (activity_states.csv)
- Procurement lead-time info (procurement.csv)
- Decision records with actual vs required day (decisions.csv)
- Risk/disruption events with type, severity, duration, impact (events.csv)
- Rework records (rework.csv)
- Friction observations (friction_canonical.csv)
- Intervention scenarios with effectiveness (intervention_options.csv)
- Counterfactual intervention outcomes (counterfactuals.csv)
- Construction experience representations with observed_delay_days (construction_memory.csv)
- Project-level activity count + planned duration (outcomes.csv)
- Project-level friction summary (friction_summary_canonical.csv)
- Integrated project representation (master_projects_canonical.csv)
- **Explicit 70/15/15 project split**

### Can be derived (with documented assumptions):

- Activity-level event-induced delay (sum of event duration_days per activity) — straightforward join; **this is the confirmed primary ML target** (§7.1)
- Activity-level decision delay (sum of decision_delay_days per activity) — straightforward join
- Activity-level friction summary (from friction_canonical) — straightforward join
- Activity-level state aggregates (mean productivity, weather risk exposure, etc.) — requires deciding aggregation window
- CPM metrics (ES, EF, LS, LF, float, critical path) — requires Phase 3 CPM engine + validation
- Project completion impact under scenarios — requires Phase 6 simulation engine

### NOT available and would require assumptions:

- **Actual activity duration** (actual_finish − actual_start) — no start/finish timestamps; activity_states does not show execution boundaries.
- **Activity completion status** (completed/on-time/late) — status = "Planned" for all.
- **Activity delay in CPM sense** (how many days past ES/EF the activity actually finished) — no actual finish data.
- **Project actual completion date/duration** — no such field.
- **Project delay days** — no actual vs planned project duration comparison available.
- **Per-activity actual productivity/loss** — productivity_index is a synthetic state, not measured output.
- **Construction memory observed_delay_days semantics** — unclear what it measures; not documented in public release.

---

## 11. Issues and questions requiring human decisions

### Q1. Master vs outcomes consistency — RESOLVED ✓

A correct name-matched comparison confirms **0 mismatches**: 100/100 `project_id`s match, `total_activities` matches on all 100 projects, and `total_planned_duration` matches on all 100 projects. The earlier "100 mismatches" were a positional column-offset artifact of a shell `join` (master has 29 cols vs outcomes 4 cols); no data error exists.

### Q2. Circular dependency check — RESOLVED ✓

An efficient per-project Kahn's-algorithm topological sort was run over all 18,176 dependency edges: **all 100 projects processed, 0 projects contain cycles** — every project's dependency graph is a DAG. CPM can proceed without cycle cleanup.

### Q3. Primary prediction target — RESOLVED ✓ (decision fixed)

**Primary target = total event-induced delay days per activity**: `SUM(events.duration_days)` per `(project_id, activity_id)`, 0 when the activity has no events. This is **event-induced / disruption delay, NOT actual schedule slippage** (no reliable actual start/finish timestamps exist). `experienced_event` is rejected as the primary target because ~94% of activities have an event (highly imbalanced); `decision_delay_days` (sparse, 30.4% coverage), `construction_memory.observed_delay_days` (ambiguous semantics, Q4), and `activity_states` (not an execution log, Q5) are also excluded. See §7.1.

### Q4. Construction memory observed_delay_days — semantics and intended use

- `activity_pattern.observed_delay_days` = 0.0 for ALL 9,279 activities.
- Nonzero delays come from `risk_pattern`, `friction_pattern`, `rework_pattern`.
- Multiple memory records per activity have different `observed_delay_days` values.
- The DATA_DICTIONARY only says "Variable from the construction_memory dataset" — no semantic description.

**Is construction_memory intended as the delay ground truth?** Or is it an auxiliary feature source? If it's ground truth, should we use only one memory type? The variation across memory types for the same activity suggests these are DIFFERENT notions of delay (e.g., risk-pattern delay = delay attributable to that risk event; friction-pattern delay = friction-induced delay; rework-pattern delay = rework-induced delay).

**Recommended:** Ask the dataset provider or check if there's additional documentation. If unavailable, treat with caution and document the assumption.

### Q5. Activity states — RESOLVED ✓ (no execution-boundary information)

Additional verification confirms productivity_index does not systematically change around planned start/end (pre-end mean ≈ post-end mean, diff ≈ +0.004; within-activity CV ≈ 0.18; event-affected vs event-free activities ≈0.734 vs ≈0.739). **activity_states contains NO usable execution-boundary information and cannot be used to infer actual activity duration.** It is treated strictly as an environmental/condition feature source (aggregated productivity, weather risk, event pressure), never as a duration proxy.

### Q6. Critical path flag — validated against CPM?

The dataset flags 2,370 activities (25.5%) as `critical_path=1`, with `critical_path_position` 1..30 (exactly 100 activities per position — a clear synthetic artifact). **These flags have NOT been validated against an actual CPM computation.**

**In Phase 3, the CPM engine must:**
1. Compute float for all activities.
2. Compare the dataset's `critical_path` flag against CPM-derived criticality (float ≈ 0).
3. Determine whether the dataset's flag is a useful pre-computed feature or should be ignored in favor of computed CPM values.

### Q7. Intervention/counterfactual data — training or tooling?

`intervention_options` (70,513) and `counterfactuals` (70,513) are synthetic intervention scenarios with `true_effectiveness` and `avoided_delay_days`. These are **event-scenario-level**, not activity-level.

**Are these intended as:**
- (a) Training data for a model to predict intervention effectiveness?
- (b) Evaluation/mitigation tooling (Phase 7) only?

**Recommended:** Phase 7 (mitigation recommendations) is the natural home for this data. For Phase 4 activity delay prediction, they may not be directly usable unless aggregated to activity level.

### Q8. procurement — activity-level or project-level only?

Procurement is linked to `project_id` and `material`, not `activity_id`. It can be joined to activities via project_id (so all activities in a project share the same procurement context). This is coarser than activity-level but usable as a project-level feature.

---

## 12. Summary of what was validated

| Check | Result |
|-------|--------|
| Row counts match DATASET_CATALOG | ✓ All 18 tables match |
| Column counts match | ✓ All match |
| Duplicate primary keys | ✓ None in any table |
| Missing cells | ✓ All 0 except construction_memory (245,745 = 18.45%) |
| Foreign-key integrity | ✓ All cross-references intact (20 checks) |
| Intra-project dependencies only | ✓ 0 cross-project edges |
| No orphan dependency refs | ✓ 0 orphan predecessors or successors |
| Activity count/successor count matches actual deps | ✓ 0 mismatches |
| project_network_duration constant per project | ✓ |
| Numeric validity (no negatives/zeros where invalid) | ✓ All checked fields clean |
| Project split 70/15/15 | ✓ Confirmed |
| Split consistency across tables | ✓ All tables consistent |
| Project set agreement (projects/outcomes/master/friction_summary) | ✓ All 100 projects agree |
| Master vs outcomes consistency (name-matched) | ✓ 0 mismatches (100/100 projects) |
| Activities.status = "Planned" for all | ✓ (no completion status) |
| Single root per project | ✓ (all 100 have exactly 1 root) |
| Single leaf per project | ✗ (all 100 have multiple leaves) |
| Circular dependencies | ✓ 0 cycles — all 100 project graphs are DAGs (Kahn's, 18,176 edges) |
| critical_path flag validated against CPM | ✗ Not yet (needs Phase 3 engine) |
| Actual activity delay target available | ✗ NOT available → primary target fixed as event-induced delay (§7.1) |
| Actual project delay target available | ✗ NOT available |
| Construction memory 1:1 with activities | ✗ NO (6.0 records/activity average) |
| activity_pattern 1:1 with activities | ✓ YES (9,279 = 9,279) |
| activity_states reveal actual execution duration | ✗ NO (uniform coverage; verified no start/stop pattern) |

---

## 13. Deliverables created in this phase

- **`scripts/scope_inventory.sh`** — Shell-based structural inventory (rows/cols/values)
- **`scripts/scope_fast_split_check.sh`** — Project split verification
- **`scripts/scope_cpm_logic_check.py`** — CPM graph structure feasibility check
- **`scripts/scope_target_candidates.py`** — Candidate delay target analysis
- **`scripts/scope_memory_activity_map.py`** — Construction memory ↔ activity mapping
- **`scripts/scope_activity_states_coverage.py`** — Activity states temporal coverage analysis
- **`scripts/scope_event_decisions_per_activity.py`** — Events/decisions per activity summary
- **`scripts/verify_master_outcomes.py`** — Master ↔ outcomes consistency (name-matched comparison)
- **`scripts/verify_circular_deps.py`** — Per-project cycle check (Kahn's algorithm)
- **`scripts/verify_activity_states_execution.py`** — Activity-states execution-boundary check
- **`notebooks/01_dataset_exploration.md`** — This report

All scripts are read-only and do not modify any raw data.

---

## 14. Phase 1 status

All Phase 1 verification items are now resolved:

1. **Master vs outcomes consistency (Q1)** — RESOLVED: 0 mismatches across 100/100 projects (name-matched).
2. **Circular dependency check (Q2)** — RESOLVED: all 100 project graphs are DAGs, 0 cycles (Kahn's algorithm over 18,176 edges).
3. **Primary prediction target (Q3)** — RESOLVED: total event-induced delay days per activity (`SUM(events.duration_days)`, 0 when no events).
4. **construction_memory observed_delay_days semantics (Q4)** — OPEN (not documented in the public release): treat with caution; NOT used as target.
5. **activity_states execution question (Q5)** — RESOLVED: no execution-boundary information; environmental/condition feature source only.
6. **critical_path flag vs CPM (Q6)** — DEFERRED to Phase 3 (requires the CPM engine).
7. **Intervention/counterfactual role (Q7)** — DEFERRED to Phase 7 decision.

**Phase 1 is now complete.** Deferred items (Q4, Q6, Q7) belong to later phases and do not block Phase 1 sign-off.

---

## 15. Recommendations for Phase 2 (not starting yet)

- Phase 2 (data cleaning) should focus on:
  - Handling construction_memory's missing fields (document, don't drop).
  - Validating and potentially excluding any circular dependency chains (once Q2 is resolved).
  - Deciding which tables/fields to carry forward based on the Phase 4 target decision.
- **Do NOT** derive or impute actual activity durations in Phase 2 — the raw data does not support this, and inventing it would violate the "never invent data" rule.
- **Do NOT** use the dataset's `critical_path` flag as ground truth until validated against a CPM engine.

---

*End of Phase 1 exploration report.*
