# Phase 4 — Feature Engineering

**Date:** 2026-09-09
**Status:** Feature extraction implemented; no models trained yet.

This document describes the deterministic leakage-safe feature matrix built for Phase 4 ML,
the feature groups included, and the leakage tests that guard against contaminated features.

---

## 1. Sources

The feature matrix is built from these processed Phase 2 tables only:

- `activities.csv`
- `dependencies.csv`
- `projects.csv`
- `resources.csv`
- `resource_allocation.csv`
- `procurement.csv`
- `targets_event_delay_days.csv` (target only)

No other SCOPE tables are used as features in this step.

---

## 2. Prediction-time safety

Every feature traces back to a prediction-time-available source per
`data/processed/prediction_time_availability.csv`.

The following are explicitly **excluded** from X:

- `target_event_delay_days` — attached only as the target column
- `critical_path`, `critical_path_position` — undocumented methodology
- `actual_lead_days`, `procurement_delay_days` — realized outcomes
- `actual_decision_day`, `decision_delay_days`, `decision_status` — decision actuals
- `rework_days`, `rework_cost` — rework outcomes
- `duration_days`, `event_type`, `severity`, `impact_factor` — event-derived
- `productivity_index`, `event_pressure`, `weather_risk`, `site_access_index` — environment/activity-state fields
- `rainfall_mm`, `temperature_c`, `humidity_pct`, `extreme_weather` — future environment observations
- `observed_delay_days` — observed delay fields
- any field containing the word `outcome`
- any UNKNOWN field not established as prediction-time available

---

## 3. Feature groups

### 3.1 Activity features (10)

| Feature | Source | Notes |
|---------|--------|-------|
| `activity_planned_duration_days` | activities | raw |
| `activity_quantity` | activities | raw |
| `activity_unit_cost` | activities | raw |
| `activity_planned_cost` | activities | raw |
| `activity_phase_order` | activities | raw |
| `activity_sequence` | activities | raw |
| `activity_predecessor_count` | activities | raw |
| `activity_successor_count` | activities | raw |
| `activity_phase` | activities | categorical |
| `activity_resource_type` | activities | categorical |

### 3.2 Network features (14)

Derived from `dependencies.csv` per activity within each project:

| Feature | Meaning |
|---------|---------|
| `net_incoming_fs_count` | incoming FS dependencies |
| `net_incoming_ss_count` | incoming SS dependencies |
| `net_incoming_ff_count` | incoming FF dependencies |
| `net_outgoing_fs_count` | outgoing FS dependencies |
| `net_outgoing_ss_count` | outgoing SS dependencies |
| `net_outgoing_ff_count` | outgoing FF dependencies |
| `net_incoming_total_lag` | sum of incoming lags |
| `net_outgoing_total_lag` | sum of outgoing lags |
| `net_incoming_max_lag` | max incoming lag |
| `net_incoming_min_lag` | min incoming lag |
| `net_has_incoming_ss` | 1 if any incoming SS |
| `net_has_incoming_ff` | 1 if any incoming FF |
| `net_has_outgoing_ss` | 1 if any outgoing SS |
| `net_has_outgoing_ff` | 1 if any outgoing FF |

### 3.3 Project features (20)

| Feature | Source | Notes |
|---------|--------|-------|
| `project_floors` | projects | raw |
| `project_area_m2` | projects | raw |
| `project_complexity` | projects | raw |
| `project_contractor_capability` | projects | raw |
| `project_resource_availability` | projects | raw |
| `project_management_maturity` | projects | raw |
| `project_weather_exposure` | projects | raw |
| `project_supply_chain_exposure` | projects | raw |
| `project_technology_maturity` | projects | raw |
| `project_planned_duration_days` | projects | raw |
| `project_planned_cost` | projects | raw |
| `project_type` | projects | categorical |
| `project_activity_count` | derived | activities per project |
| `project_dependency_count` | derived | dependencies per project |
| `project_mean_planned_activity_duration` | derived | mean planned duration |
| `project_median_planned_activity_duration` | derived | median planned duration |
| `project_fs_count` | derived | FS edges per project |
| `project_ss_count` | derived | SS edges per project |
| `project_ff_count` | derived | FF edges per project |
| `procurement_supply_chain_risk` | procurement | max per project |

### 3.4 Resource features (7)

Derived from `resource_allocation.csv` + `resources.csv`:

| Feature | Meaning |
|---------|---------|
| `resource_allocated_count` | resources allocated to activity |
| `resource_labour_count` | labour resources |
| `resource_equipment_count` | equipment resources |
| `resource_material_count` | material resources |
| `resource_total_allocation_fraction` | sum of allocation fractions |
| `resource_mean_availability` | mean resource availability |
| `resource_mean_capacity` | mean resource capacity |

### 3.5 Procurement features (5)

Derived from `procurement.csv` (planned side only):

| Feature | Meaning |
|---------|---------|
| `procurement_planned_lead_days` | planned lead days |
| `procurement_record_count` | procurement records for project |
| `procurement_mean_planned_lead_days` | mean planned lead days |
| `procurement_max_planned_lead_days` | max planned lead days |
| `procurement_supply_chain_risk` | max supply chain risk |

### 3.6 Total

- Total engineered features: 55
- Categorical features: 3 (`activity_phase`, `activity_resource_type`, `project_type`)
- Numeric features: 53

---

## 4. Target

The target `target_event_delay_days` is attached **only** as `y`, never as an input feature.

---

## 5. Split integrity

The matrix preserves the existing SCOPE project-level split:

| Split | Projects | Activities |
|-------|----------|------------|
| train | 70 | (derived from data) |
| validation | 15 | (derived from data) |
| test | 15 | (derived from data) |

Verified by tests:

- No project appears in multiple splits
- Every activity belongs to exactly one split
- Activity counts are derived from the data, not hard-coded

---

## 6. Leakage tests

`tests/test_feature_engineering.py` includes leakage tests that verify:

- `target_event_delay_days` is not in X
- All `EXCLUDED_INPUT_COLUMNS` are absent from X
- `critical_path` / `critical_path_position` are not in X
- No event-derived terms appear in feature names
- No rework terms appear in feature names
- No procurement actual terms appear in feature names
- No observed delay terms appear in feature names
- No outcome terms appear in feature names
- Train/validation/test projects are disjoint
- Target values match the processed target file exactly

---

## 7. Determinism

The feature matrix is fully deterministic:

- Activities are sorted by `(project_id, activity_id)`
- Feature columns are in a fixed registered order
- Network features are computed by a single pass over dependencies
- Project aggregates use sorted planned durations for median computation

Running `build_feature_matrix()` twice produces identical results.

---

## 8. Missing values

The processed SCOPE v0.2 tables are fully populated. The feature matrix has **0 missing values** across all features.

---

## 9. Files

- `src/ml/feature_engineering.py` — feature extraction module
- `tests/test_feature_engineering.py` — leakage and correctness tests
- `notebooks/04_feature_engineering.md` — this document

---

## 10. Next steps

After this step:

- Encode categorical features
- Scale numeric features
- Train baseline models
- Evaluate on the project-level split
