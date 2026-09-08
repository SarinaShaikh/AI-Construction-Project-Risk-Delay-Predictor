# Phase 2 — Data Cleaning & Preparation Report

**Project:** AI Construction Project Risk & Delay Predictor
**Phase:** 2 — Data Cleaning & Preparation
**Date:** 2026-09-09
**Status:** COMPLETE — all validations PASS (see §15)

> This report documents every transformation applied to SCOPE v0.2 during Phase 2.
> The raw dataset (`data/raw/SCOPE_v02_Public/`) was **never modified** — the
> pipeline reads it read-only and writes exclusively to `data/processed/`.
> Phase 1 decisions (primary target, leakage rules, project split) are unchanged.

---

## 1. Objective

Produce a clean, validated, reproducible copy of the SCOPE v0.2 core tables for
downstream phases (CPM in Phase 3, ML in Phase 4), and construct the fixed
activity-level prediction target:

```
target_event_delay_days = SUM(events.duration_days)   per (project_id, activity_id)
target_event_delay_days = 0                            when the activity has no events
```

The target is **event-induced / disruption delay**, NOT actual schedule slippage
(the dataset contains no reliable actual start/finish timestamps — see §16).

Phase 2 explicitly does **not**:
- modify raw data,
- train or evaluate any model,
- impute missing values,
- fabricate dates or durations,
- convert negative values to positive,
- create a new random split,
- implement CPM or ML.

## 2. Input data

| Item | Value |
|------|-------|
| Location | `data/raw/SCOPE_v02_Public/` (read-only) |
| Version | SCOPE v0.2 Public Research Release (synthetic) |
| Projects | 100 (70 train / 15 validation / 15 test) |
| Raw files used | the 12 core tables listed in §3 |

`data/raw/` is git-ignored (see §"Git" note in §17) so the dataset is never
accidentally committed or modified through version control.

## 3. Tables processed

Only the 12 core downstream tables are processed (per Phase 2 scope). The other
SCOPE tables (`construction_memory`, `intervention_options`, `counterfactuals`,
`friction_canonical`, `friction_summary_canonical`, `master_projects_canonical`)
are **not** processed in Phase 2 — no demonstrated reason from the Phase 1
findings; they are revisited in later phases (Q4/Q7 decisions in Phase 1).

| Table | Primary key | Rows | Cols |
|-------|-------------|------|------|
| `projects.csv` | `project_id` | 100 | 14 |
| `activities.csv` | `project_id` + `activity_id` | 9,279 | 19 |
| `dependencies.csv` | `project_id` + `predecessor_id` + `successor_id` | 18,176 | 6 |
| `resources.csv` | `resource_id` | 2,748 | 8 |
| `resource_allocation.csv` | `project_id` + `activity_id` + `resource_id` | 9,279 | 6 |
| `environment.csv` | `project_id` + `day_index` | 52,332 | 10 |
| `activity_states.csv` | `project_id` + `activity_id` + `day_index` | 4,060,623 | 9 |
| `procurement.csv` | `procurement_id` | 800 | 8 |
| `events.csv` | `event_id` | 27,989 | 10 |
| `decisions.csv` | `decision_id` | 3,374 | 15 |
| `rework.csv` | `rework_id` | 1,858 | 7 |
| `outcomes.csv` | `project_id` | 100 | 4 |

The pipeline verifies the actual raw schema (column names/order per table) before
processing and fails loudly on any schema drift. No columns are dropped — even
columns not needed for Phase 4 are preserved.

## 4. Cleaning rules

Rules applied, following Phase 2 spec (STEP 4 of the phase plan):

- **A. Identifiers** — original IDs preserved exactly (never regenerated);
  required IDs checked for missing values; primary-key uniqueness enforced.
- **B. Duplicates** — exact duplicate rows detected and removed (none found);
  conflicting records sharing a primary key are *reported and never silently
  resolved* (none found).
- **C. Dates/timestamps** — parsed and validated consistently (`YYYY-MM-DD`);
  invalid values reported (none found). No dates fabricated; no actual
  activity start/finish dates inferred.
- **D. Durations** — validated for negatives and impossible zeros; negative
  values are reported and investigated, **never converted to positive** (none found).
- **E. Dependencies** — predecessor/successor references, project boundaries,
  relationship types, and lags validated; efficient per-project cycle check
  (Kahn's algorithm). `FS`, `FF`, `SS` preserved.
- **F. Numeric values** — converted to appropriate types (`int64`/`float64`)
  where justified by the data dictionary; parse failures reported (none found).
  Statistical outliers are **not** removed — an unusual value is not invalid.
- **G. Categorical values** — only obvious formatting inconsistencies would be
  normalized; none found, so no category merging was performed.
- **H. Missing values** — per-column missing summaries produced; **no
  imputation**. Structural missingness (e.g., `construction_memory`, not
  processed in Phase 2) keeps its semantic meaning.

## 5. Transformations performed

| # | Transformation | Applies to | Notes |
|---|----------------|-----------|-------|
| T1 | Read raw CSVs read-only with `''` as the only missing marker | all tables | no BOM, no whitespace padding found |
| T2 | dtype normalization: int/float columns converted to `int64`/`float64` | all numeric columns | lossless; parse failures would be reported (0 found) |
| T3 | Float string normalization on round-trip | e.g., `decisions.dependency_exposure` | raw value `0.9382299999999999` → `0.93823`; numerically identical (`9.554267613486049` → `...047`; identical float64) |
| T4 | Date validation (`format="%Y-%m-%d"`) | `environment.date`, `events.date`, `activity_states.date` | 0 invalid; original string preserved (already ISO 8601) |
| T5 | Split-consistency check against `projects.csv` | all tables with `split` | 0 mismatches |
| T6 | Primary-key uniqueness + exact-duplicate detection | all tables | 0 duplicates; 0 rows removed |
| T7 | Duration validation (negatives / impossible zeros) | duration fields listed in §9 | 0 issues |
| T8 | Foreign-key validation (18 checks) | cross-table | all PASS (§10) |
| T9 | Dependency graph validation + Kahn cycle check | `dependencies` | 0 cycles (§10) |
| T10 | Cross-field consistency checks | decisions/procurement/activities | all PASS (§15) |
| T11 | Target construction | `events` + `activities` → `targets_event_delay_days.csv` | §12 |
| T12 | Prediction-time availability classification | all processed columns | §13, machine-readable CSV |
| T13 | Deterministic CSV output (LF line endings, original row order) | all outputs | byte-identical across runs (§15) |

No rows were dropped, no values were changed semantically, no categories were
merged, and no statistical outliers were removed.

## 6. Missing-value handling

All 12 processed tables are **fully populated**:

- Missing (empty) cells: **0 across all 12 tables** (4,060,623 activity-state
  rows included).
- No imputation was performed or needed.
- `construction_memory.csv` (the only SCOPE table with structural missingness,
  245,745 cells) is **not processed in Phase 2**; its missingness is semantic
  (typed memory records) and preserved in the raw data.

## 7. Duplicate handling

| Check | Result |
|-------|--------|
| Exact duplicate rows (all columns) | **0** in every table |
| Duplicate primary keys | **0** in every table |
| Conflicting records sharing a PK | **0** (nothing to resolve) |
| Rows removed | **0** |

The pipeline implements removal of exact duplicates (keep first) when they
appear, with every removal reported in `cleaning_manifest.json`. No removals
occurred, so no data was lost.

## 8. Date handling

- Date columns: `environment.date`, `events.date`, `activity_states.date`.
- Format verified: ISO 8601 `YYYY-MM-DD` for all values.
- Invalid/unparseable dates: **0** (verified with strict `format="%Y-%m-%d"` parsing).
- Global range: 2024-01-01 … 2026-05-21 (`activity_states` ends 2025-11-27).
- All `events.date` values fall within each project's `environment.date` range.
- Original string values are preserved verbatim in the cleaned output (already
  ISO 8601); no timezone conversion, no date fabrication.
- **No actual activity start/finish dates exist or are inferred** (see §16).

## 9. Duration validation

All duration-like fields were checked for negatives and impossible zeros:

| Table | Field | Range | Negatives | Zeros allowed? |
|-------|-------|-------|-----------|----------------|
| projects | `planned_duration_days` | 259–697 | 0 | no |
| activities | `planned_duration_days` | 3–40 | 0 | no |
| dependencies | `lag_days` | 0–3 | 0 | yes |
| procurement | `planned_lead_days` | 2–90 | 0 | no |
| procurement | `actual_lead_days` | 2–104 | 0 | no |
| procurement | `procurement_delay_days` | 0–53 | 0 | yes |
| events | `duration_days` | 1–75 | 0 | no |
| decisions | `decision_delay_days` | 0–35 | 0 | yes |
| rework | `rework_days` | 1–27 | 0 | no |

**Result:** no negative or impossible duration values exist. Nothing was
converted, clamped, or imputed.

## 10. Dependency validation

| Check | Result |
|-------|--------|
| Dependency edges | 18,176 |
| Projects with dependencies | 100 |
| Orphan predecessors (not in activities) | **0** |
| Orphan successors (not in activities) | **0** |
| Cross-project edges (project boundary violation) | **0** |
| Relationship types | FS=14,527, FF=955, SS=2,694 (all valid; preserved) |
| `lag_days` | 0–3, no negatives |
| Duplicate dependency edges | 0 (PK unique) |
| Cycles (Kahn's algorithm, per project) | **0 cycles — all 100 graphs are DAGs** |

Every project graph is a directed acyclic graph, so Phase 3 CPM can proceed
without cycle cleanup. Per Phase 1, each project has a single root activity and
multiple leaves — a CPM engine must handle multiple sinks.

## 11. Project split validation

| Split | Projects |
|-------|----------|
| train | **70** |
| validation | **15** |
| test | **15** |

- The split is **project-level** — never split individual activity rows.
- Every table's `split` column is consistent with `projects.csv` for 100/100
  projects (0 mismatches in any table).
- No projects were moved between splits; no new random split was created.

## 12. Target construction

`target_event_delay_days` per activity = `SUM(events.duration_days)`, `0` when
no events exist. Constructed in `scripts/clean_scope.py` (STEP 6), output:
`data/processed/targets_event_delay_days.csv` (`project_id`, `activity_id`,
`target_event_delay_days`).

**Guarantees verified:**
- exactly one target row per activity (9,279 = 9,279) — no activities lost;
- no duplicate `(project_id, activity_id)` rows;
- no missing target values;
- target recomputed **independently from the raw events file** and matched
  exactly (0 mismatches);
- every zero-target activity has no events in the raw data (523/523 verified).

**Statistics (all 9,279 activities):**

| Statistic | Value |
|-----------|-------|
| Activities | 9,279 |
| target = 0 | 523 (5.64%) |
| target > 0 | 8,756 (94.36%) |
| Minimum | 0 |
| Maximum | 168 |
| Mean | 16.7741 |
| Median | 13.0 |
| Std dev | 15.405 |
| P25 | 5.0 |
| P50 | 13.0 |
| P75 | 24.0 |
| P90 | 38.0 |
| P95 | 47.0 |
| P99 | 67.0 |

> Note: Phase 1 reported a mean of 17.78 for affected activities only; 16.7741
> is the mean over all 9,279 activities including the 523 zeros. Same data,
> different denominator — documented here to avoid confusion.

## 13. Leakage precautions (prediction-time data availability)

**Core rule (unchanged from Phase 1):** information generated after or because
of an outcome must not be used as a predictive feature. In particular, **no
event-derived delay information is used as an input feature** — events are the
target source, not features.

Each processed column is classified into one of three buckets
(machine-readable: `data/processed/prediction_time_availability.csv`):

1. **AVAILABLE_AT_PREDICTION_TIME** — a project manager could reasonably know
   this before the event-induced delay occurs (planning inputs: project
   characteristics, planned durations/costs, network structure, resource
   plans, planned lead times, planned aggregates).
2. **OUTCOME / POST-OUTCOME** — known only after disruption/delay occurs; must
   NOT become ML features. This includes **all of `events`** (the target
   source), all of `rework`, `decisions.actual_decision_day`,
   `decisions.decision_delay_days`, `decisions.decision_status`,
   `procurement.actual_lead_days`, `procurement.procurement_delay_days`, and
   `activity_states.event_pressure` (event-derived).
3. **UNKNOWN** — availability unclear; classified as UNKNOWN rather than
   assumed safe. This includes daily time-varying observations
   (`environment.*` weather/site fields, `activity_states` productivity /
   weather / site access) and the `decisions` execution-phase records whose
   existence/derivation depends on the prediction point
   (`decision_type`, `required_day`, `criticality`, `dependency_exposure`,
   `time_exposure`, `reversibility_penalty`, `decision_debt_score`,
   `decision_debt_level`, `decision_id`).

Summary counts by category (116 processed columns total):

| Category | Columns |
|----------|---------|
| AVAILABLE_AT_PREDICTION_TIME | 78 |
| UNKNOWN | 18 |
| OUTCOME_OR_POST_OUTCOME | 20 |

This is a **conservative working classification**; Phase 3.5/4 will finalize it
before any model training. No speculative assumptions were made — uncertain
fields are marked UNKNOWN, never assumed safe.

## 14. Before/after statistics

Raw vs processed row counts (identical for every table — **no rows lost**):

| Table | Raw rows | Processed rows | Raw cols | Processed cols | PK unique | Missing cells |
|-------|----------|----------------|----------|----------------|-----------|---------------|
| projects | 100 | 100 | 14 | 14 | yes | 0 |
| activities | 9,279 | 9,279 | 19 | 19 | yes | 0 |
| dependencies | 18,176 | 18,176 | 6 | 6 | yes | 0 |
| resources | 2,748 | 2,748 | 8 | 8 | yes | 0 |
| resource_allocation | 9,279 | 9,279 | 6 | 6 | yes | 0 |
| environment | 52,332 | 52,332 | 10 | 10 | yes | 0 |
| activity_states | 4,060,623 | 4,060,623 | 9 | 9 | yes | 0 |
| procurement | 800 | 800 | 8 | 8 | yes | 0 |
| events | 27,989 | 27,989 | 10 | 10 | yes | 0 |
| decisions | 3,374 | 3,374 | 15 | 15 | yes | 0 |
| rework | 1,858 | 1,858 | 7 | 7 | yes | 0 |
| outcomes | 100 | 100 | 4 | 4 | yes | 0 |

Invalid values found: **0** (numeric parse failures, invalid dates, negative
durations, missing IDs). Transformations: only the dtype/format normalizations
listed in §5 — every column and row is preserved.

## 15. Validation results

All checks **PASS** (machine-readable audit: `data/processed/cleaning_manifest.json`):

| Check | Result |
|-------|--------|
| Schema matches registry (12 tables) | PASS |
| Primary keys unique (12 tables) | PASS |
| No missing IDs / no empty required fields | PASS |
| Exact duplicates: none (0 removed) | PASS |
| All numeric values parse; int/float dtypes correct | PASS |
| All dates parse (ISO 8601) | PASS |
| Split consistent with `projects.csv` (12 tables) | PASS |
| Durations: no negatives / no impossible zeros | PASS |
| FK: dependencies → activities (pred & succ) | PASS |
| FK: dependencies intra-project only | PASS |
| FK: dependencies relationships {FS,FF,SS} + lag ≥ 0 | PASS |
| FK: dependencies acyclic (per project) | PASS |
| FK: resource_allocation → activities, resources (+ category match) | PASS |
| FK: environment → projects | PASS |
| FK: activity_states → activities | PASS |
| FK: procurement → projects | PASS |
| FK: decisions → activities | PASS |
| FK: events → activities | PASS |
| FK: rework → events & activities | PASS |
| FK: outcomes / resources → projects | PASS |
| Consistency: `decision_delay_days == actual_decision_day - required_day` (0 mismatches) | PASS |
| Consistency: `procurement_delay_days == actual_lead_days - planned_lead_days` (0 mismatches) | PASS |
| Consistency: `planned_cost ≈ quantity × unit_cost` (max rel diff 7.5e-05) | PASS |
| Consistency: activities `predecessor_count`/`successor_count` == actual edges | PASS |
| Project split 70/15/15 | PASS |
| Target: one row per activity, no dups, no missing | PASS |
| Target: matches independent recomputation from raw events | PASS |
| Determinism: fresh run byte-identical to `data/processed/` | PASS |
| Raw data untouched during pipeline run | PASS |

**Tests:** 47 pytest tests in `tests/test_clean_scope.py` — all pass
(`uv run pytest tests/ -v`; run locally with `.venv/Scripts/python -m pytest tests/ -v`
because `uv` is not installed on this machine).

## 16. Known limitations

1. **The target is event-induced delay, not schedule slippage.** SCOPE v0.2 has
   no reliable actual activity start/finish timestamps (`status = "Planned"` for
   all 9,279 activities). The target measures disruption duration from events
   only, per the fixed Phase 1 decision.
2. **`environment` timeline spans ~1.25× planned duration** (ratio 1.25–1.2528,
   mean 1.251 across 100 projects): environmental observations include a
   post-planned-completion buffer. `activity_states` spans exactly 1.0× planned
   duration (max `day_index` == `planned_duration_days` for all 100 projects).
   Both are structural properties of the synthetic dataset — preserved as-is and
   documented; relevant when aggregating time-series features in Phase 4.
3. **Float string normalization.** A handful of raw values stored with
   floating-point artifacts (e.g., `0.9382299999999999`) round-trip to their
   shortest representation (`0.93823`). Numerically identical; cosmetic only.
4. **`critical_path` / `critical_path_position` flags are synthetic** (Phase 1,
   Q6) and remain unvalidated against a real CPM computation — Phase 3 must
   compare them against CPM-derived criticality before they are trusted.
5. **`decisions` table semantics.** Decision-process attributes (e.g.,
   `decision_debt_score`) are classified UNKNOWN for prediction-time
   availability; only decision *actuals* are hard-classified as outcomes.
   Final decision on their use belongs to Phase 3.5/4.
6. **Synthetic data.** All results are methodological, within a controlled
   synthetic environment; external validation with real-world data is required
   before deployment (per SCOPE README).
7. **Environment note.** `uv` is not installed on this machine, so `uv.lock`
   was not regenerated after adding `pytest` to `pyproject.toml`; a local
   `.venv` (git-ignored) with `pytest` + `pandas` was used instead. Run
   `uv lock` once `uv` is available to refresh the lockfile.

## 17. Remaining issues

- **None blocking.** All Phase 2 criteria passed; no data issues remain open.
- Deferred (per Phase 1, out of Phase 2 scope): construction-memory semantics
  (Q4), `critical_path` validation (Q6, needs Phase 3 CPM), intervention/
  counterfactual role (Q7, Phase 7).
- **Git hygiene:** `data/raw/`, `data/processed/`, and `data/exports/` are
  git-ignored (`.gitignore`). Nothing was committed or pushed; the working tree
  keeps the user's pre-existing `CLAUDE.md`/`PROGRESS.md` modifications
  untouched.

---

## Deliverables created in Phase 2

| File | Purpose |
|------|---------|
| `scripts/clean_scope.py` | Deterministic cleaning pipeline (run: `python scripts/clean_scope.py`) |
| `tests/test_clean_scope.py` | 47 validation tests (run: `uv run pytest tests/ -v`) |
| `data/processed/*.csv` | 12 cleaned tables + `targets_event_delay_days.csv` + `prediction_time_availability.csv` |
| `data/processed/cleaning_manifest.json` | Full audit trail (transformations + all validation results) |
| `notebooks/02_data_cleaning.md` | This report |
| `.gitignore` (modified) | Added `data/raw/` |
| `pyproject.toml` (modified) | Added `pytest` to dev dependencies |

*End of Phase 2 report.*