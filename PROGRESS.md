# AI Construction Project Risk & Delay Predictor — Progress Tracker

## Current Status
**Phase:** 2 (Data Cleaning & Preparation) — ✅ COMPLETE  
**Last Updated:** 2026-09-09  
**Next Task:** Phase 3 — CPM Engine (not started; awaiting approval)

---

## Completed Phases

### Phase 0 — Project Setup ✅
- ✅ Initialize Git repository on GitHub
- ✅ Create folder structure (data/, src/, app/, tests/, scripts/, notebooks/)
- ✅ Create pyproject.toml with Python 3.12 + proper metadata
- ✅ Create .gitignore (protects .venv, __pycache__, .env, logs, generated outputs)
- ✅ Create PROGRESS.md
- ✅ Push to GitHub: https://github.com/SarinaShaikh/AI-Construction-Project-Risk-Delay-Predictor

**Repository Structure:**
construction-risk-predictor/
├── data/
│ ├── raw/ (SCOPE v0.2 dataset)
│ ├── processed/ (Cleaned data - ignored by Git)
│ └── exports/ (CSV exports - ignored by Git)
├── notebooks/ (Jupyter exploratory analysis)
├── src/
│ ├── ai_construction/ (Main package)
│ ├── cpm/ (Critical Path Method calculations)
│ ├── data/ (Data loading & cleaning)
│ ├── features/ (Feature engineering)
│ ├── models/ (ML models)
│ ├── risk/ (Risk analysis)
│ ├── scenarios/ (What-if simulation)
│ ├── recommendations/ (Mitigation recommendations)
│ └── llm/ (LLM/Agent integration)
├── app/ (Streamlit dashboard)
├── tests/ (Unit & integration tests)
├── scripts/ (Standalone utilities)
├── PROGRESS.md (This file)
├── README.md (Project documentation)
├── pyproject.toml (Python 3.12 config)
└── .gitignore (Git exclusions)

---


### Phase 1 — Understand SCOPE v0.2 Dataset ✅
**Objective:** Explore dataset structure, understand projects, activities, dependencies, and data quality.

**Completed:**
- ✅ Explore complete SCOPE v0.2 dataset contents (`data/raw/SCOPE_v02_Public/`: 18 data files + documentation)
- ✅ Build file inventory (row/column counts, keys, relationships) — matches DATASET_CATALOG
- ✅ Validate integrity: no duplicate primary keys, all foreign keys intact, no negative/invalid durations, no orphan dependency references
- ✅ Verify dependency relationships (FS/FF/SS, lags 0–3) and CPM graph feasibility (single root per project; multiple leaves per project)
- ✅ Verify project split: 70 train / 15 validation / 15 test, consistent across all 18 tables
- ✅ Verify no circular dependencies: all 100 project graphs are DAGs (Kahn's algorithm over 18,176 edges)
- ✅ Verify master_projects_canonical ↔ outcomes consistency: 0 mismatches across 100/100 projects
- ✅ Determine activity_states is NOT execution-boundary data (uniform timeline coverage; productivity does not stop at planned end)
- ✅ Establish primary ML target and prediction-time leakage rules
- ✅ Create `notebooks/01_dataset_exploration.md` exploration & verification report

**Key Decisions:**
- **Primary ML target (fixed):** event-induced / disruption delay days per activity = `SUM(events.duration_days)` per `(project_id, activity_id)`; `0` when the activity has no events.
- **Actual schedule slippage cannot be claimed** from this dataset: no reliable actual activity start/finish timestamps exist anywhere (`status` = "Planned" for all 9,279 activities; no project completion dates).
- **Not used as primary target:** `experienced_event` (~94% positive → highly imbalanced), `decision_delay_days` (sparse, 30.4% coverage), `construction_memory.observed_delay_days` (ambiguous semantics), `activity_states` (not an execution log).
- **Prediction-time leakage rules established:** observed/future event information, observed delay fields, decision actuals, rework outcomes, and other post-outcome information must NOT be used as predictive features unless a later phase establishes they are available at prediction time. Event-derived information is the target, not an input feature.

**Deliverables:**
- `notebooks/01_dataset_exploration.md` — Phase 1 exploration & verification report
- `notebooks/01_dataset_exploration.ipynb` — Phase 1 EDA notebook with visualizations (added on GitHub)
- `src/data/loader.py` — reusable data loading & validation module (added on GitHub)
- `reports/phase1_data_quality_report.csv`, `reports/phase1_project_statistics.csv` (added on GitHub)
- `scripts/verify_master_outcomes.py` — master ↔ outcomes consistency check
- `scripts/verify_circular_deps.py` — per-project cycle check (Kahn's algorithm)
- `scripts/verify_activity_states_execution.py` — activity_states execution-boundary check
- Supporting exploration scripts (`scripts/scope_*.py`, `scripts/scope_*.sh`)

---

### Phase 2 — Data Cleaning & Preparation ✅
**Objective:** Clean, validate, and prepare data for analysis — reproducibly, without touching raw data.

**Completed:**
- ✅ Protect raw data: `data/raw/` added to `.gitignore` (with `data/processed/`, `data/exports/`); raw files never modified
- ✅ Inspect actual raw schema of all 12 core tables (source of truth, not assumptions)
- ✅ Create deterministic cleaning pipeline `scripts/clean_scope.py` (reads only `data/raw/`, writes only `data/processed/`, exit 0 only if all validations pass)
- ✅ Clean 12 core tables (projects, activities, dependencies, resources, resource_allocation, environment, activity_states, procurement, events, decisions, rework, outcomes) — no rows lost, no columns dropped
- ✅ Missing values: 0 missing cells across all 12 tables; no imputation (construction_memory structural missingness preserved in raw, not processed)
- ✅ Duplicates: 0 exact duplicates, 0 duplicate primary keys, 0 conflicts — nothing removed
- ✅ Dates: all ISO 8601 `YYYY-MM-DD`, 0 invalid; no fabricated/inferred dates
- ✅ Durations: no negatives/impossible zeros in any duration field
- ✅ Dependencies: 18,176 edges, 0 orphans, 0 cross-project, FS/FF/SS preserved, 0 cycles (Kahn) — all 100 graphs are DAGs
- ✅ Project split: 70/15/15 verified project-level, consistent across all tables
- ✅ Construct target `target_event_delay_days` = SUM(events.duration_days) per activity (0 if none); 9,279 rows, validated against raw events; stats in `notebooks/02_data_cleaning.md` §12
- ✅ Leakage: per-field prediction-time availability classification (`data/processed/prediction_time_availability.csv`); events/rework/decision-actuals = outcome; uncertain fields = UNKNOWN
- ✅ 47 pytest tests in `tests/test_clean_scope.py` — all pass
- ✅ Create `notebooks/02_data_cleaning.md` — full Phase 2 report (17 sections)

**Key Findings (Phase 2):**
- All 12 core tables are fully populated, no duplicates, no invalid values — cleaning was validation-heavy, transformation-light (dtype/format normalization only)
- `environment` timeline spans ~1.25× planned duration per project (post-planned buffer); `activity_states` spans exactly 1.0× planned duration — documented structural properties
- Raw float artifacts normalized on round-trip (e.g., `0.9382299999999999` → `0.93823`), numerically identical
- Target mean = 16.7741 over all 9,279 activities (Phase 1's 17.78 was over the 8,756 affected activities only)

**Deliverables:**
- `scripts/clean_scope.py` — deterministic cleaning pipeline
- `tests/test_clean_scope.py` — 47 validation tests
- `data/processed/` — 12 cleaned tables + `targets_event_delay_days.csv` + `prediction_time_availability.csv` + `cleaning_manifest.json` (all git-ignored)
- `notebooks/02_data_cleaning.md` — Phase 2 report
- `.gitignore` (modified) — added `data/raw/`
- `pyproject.toml` (modified) — added `pytest` to dev dependencies (note: `uv.lock` not regenerated — `uv` not installed on this machine; run `uv lock` when available)

---

## In-Progress Phase

None — Phase 2 complete. Awaiting approval to begin Phase 3 (CPM Engine).

---

## Upcoming Phases

### Phase 3 — CPM Engine ✅ COMPLETE
**Objective:** Implement Critical Path Method calculations.

**Completed:**
- ✅ Build deterministic dependency graph (NetworkX-free, Kahn's algorithm)
- ✅ Calculate ES (Earliest Start), EF (Earliest Finish)
- ✅ Calculate LS (Latest Start), LF (Latest Finish)
- ✅ Calculate Total Float for each activity
- ✅ Identify Critical Path (activities with float ≈ 0, tolerance = 1e-6)
- ✅ Support FS, SS, FF dependency relationships with lags (0–3 days)
- ✅ Handle multiple terminal activities via virtual project-end node
- ✅ Cycle detection: raises CyclicGraphError on cyclic graphs
- ✅ Validate CPM against 36 unit tests (all pass)
- ✅ Run CPM across all 100 SCOPE v0.2 projects (0 violations, 0 cycles, 0 float inconsistencies)
- ✅ Compare computed critical path vs dataset critical_path field (82.1% agreement)
- ✅ Create `notebooks/03_cpm_validation.md` — full validation report

**Key Results:**
- All 100 projects processed successfully (0 cycles, 0 constraint violations)
- 9,279 activities processed, 18,176 dependency edges validated
- Project durations: 224–449 days (mean 340 days)
- 3,153 activities marked critical by computed CPM
- 82.1% agreement with dataset's critical_path field (differences due to methodology)
- Zero negative float values, zero EF < ES, zero LF < LS violations

**Deliverables:**
- `src/cpm/__init__.py` — CPM package
- `src/cpm/calculation.py` — Deterministic CPM engine (NetworkX-free)
- `tests/test_cpm.py` — 36 unit tests (all pass)
- `scripts/run_cpm.py` — Full-dataset CPM runner
- `notebooks/03_cpm_validation.md` — Phase 3 validation report
- `data/processed/cpm/` — CPM output artifacts (git-ignored)

---

### Phase 4 — ML Delay Prediction
**Objective:** Train models to predict activity-level delay probability & duration.

- [ ] Feature engineering (resource availability, procurement time, activity conditions, weather)
- [ ] Handle data leakage (project-level train/val/test split)
- [ ] Train baseline models: Logistic Regression, Random Forest, Gradient Boosting
- [ ] Evaluate metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Calibration
- [ ] Hyperparameter tuning
- [ ] Select best model based on ROC-AUC & calibration
- [ ] Serialize model (pickle/joblib)

**Deliverables:**
- `src/models/delay_classifier.py` — Trained model wrapper
- `src/features/engineering.py` — Feature transformations
- `notebooks/04_model_training.ipynb` — Model development & evaluation

---

### Phase 5 — Risk Scoring
**Objective:** Rank activities by risk (delay probability × impact on critical path).

- [ ] Define risk score formula: Delay Probability × CPM Float Impact
- [ ] Calculate criticality score (activities near critical path weighted higher)
- [ ] Rank all activities by risk
- [ ] Identify top N at-risk activities

**Deliverables:**
- `src/risk/scoring.py` — Risk ranking logic
- Risk report with activity rankings

---

### Phase 6 — What-if Simulation
**Objective:** Model scenario impacts (delay, resource reduction, etc.).

**Scenarios to support:**
- "Material delivery for Activity A delayed by X days"
- "Resource allocation to Activity B reduced by Y%"
- "Weather event delays activity start by Z days"

**For each scenario, calculate:**
- Direct activity impact
- Successor impact (downstream propagation)
- Float consumption
- Critical path changes
- Project completion date impact
- Updated risk ranking

**Deliverables:**
- `src/scenarios/simulator.py` — Scenario engine
- Scenario result analysis (JSON/CSV)

---

### Phase 7 — Mitigation Recommendations
**Objective:** Generate context-aware mitigation actions.

- [ ] Define mitigation strategies (fast-track, crash, resource reallocation, etc.)
- [ ] Match strategies to project data & ML predictions
- [ ] Rank recommendations by impact & feasibility
- [ ] Generate manager-friendly explanations

**Deliverables:**
- `src/recommendations/engine.py` — Recommendation logic
- Manager-facing recommendation report

---

### Phase 8 — LLM / Agentic AI
**Objective:** Use Groq LLM to investigate risks & explain results.

**LLM responsibilities:**
- Investigate project risks (call analysis tools)
- Explain CPM results
- Summarize delay predictions
- Explain what-if scenarios
- Generate manager-friendly insights

**LLM constraints:**
- ❌ Must NOT invent numerical predictions
- ✅ Must call analysis tools for numerical results
- ✅ Can synthesize insights from tool results

**Deliverables:**
- `src/llm/agent.py` — Groq agent with tool integration
- Tool definitions (CPM analysis, risk analysis, scenario simulation)

---

### Phase 9 — Streamlit Dashboard
**Objective:** Interactive UI for project managers.

**Dashboard features:**
- Project overview (timeline, # activities, # dependencies)
- Critical path visualization
- Risk heatmap (activity rankings)
- ML delay predictions by activity
- What-if scenario builder
- Mitigation recommendations panel
- LLM chat interface for risk investigation
- Export reports (PDF, CSV)

**Deliverables:**
- `app/dashboard.py` — Main Streamlit app
- Supporting components

---

### Phase 10 — Deployment
**Objective:** Deploy to production (Render free tier).

- [ ] Create requirements for Render
- [ ] Deploy Streamlit app to Render
- [ ] Test in production
- [ ] Monitor logs & performance
- [ ] Optimize cold start time

**Deliverables:**
- Deployed live dashboard
- Deployment documentation

---

## Data Model (SCOPE v0.2)

### Core Tables
- **projects.csv** (100 projects)
  - Columns: project_id, project_type, floors, area_m2, complexity, contractor_capability, resource_availability, management_maturity, weather_exposure, supply_chain_exposure, technology_maturity, planned_duration_days, planned_cost
  
- **activities.csv** (9,279 activities)
  - Columns: project_id, activity_id, phase, activity_name, resource_type, planned_duration_days, quantity, unit_cost, planned_cost, criticality, status, critical_path, predecessor_count, successor_count
  
- **dependencies.csv** (18,176 dependencies)
  - Columns: project_id, predecessor_id, successor_id, relationship, lag_days
  - **KEY TABLE for CPM graph construction**

### Supporting Tables
- **activity_states.csv** (4M+ records) — Daily productivity, weather risk, site access
- **construction_memory.csv** — Observed delays, rework, risk scores (**ML training labels**)
- **events.csv**, **environment.csv**, **decisions.csv** — Context & features for ML
- Plus 14 more tables for procurement, resources, friction, counterfactuals, etc.

### Data Loader
- **`src/data/loader.py`** — Reusable module for loading & validating all 25 CSV files
  - Call: `data = load_raw_dataset()` → returns dict of DataFrames
  - Validates referential integrity, row counts, column names, nulls, duplicates
  - Getter functions: `get_projects(data)`, `get_activities(data)`, `get_dependencies(data)`, etc.

---

## Known Gotchas

### Data Leakage (Phase 4+)
**Issue:** ML models trained on future data will predict unrealistically well.
**Prevention:** Always split by project, not by row. Train on past projects, validate/test on future projects.

### CPM Tolerance
**Issue:** Which activities are "critical"? Float = 0.0 exactly? Or < 0.5 days?
**Prevention:** Define and document tolerance threshold (e.g., float < 0.1 days = critical). Consistency matters.

### What-if Scenario Logic
**Issue:** Simply adding delay to project end date is wrong.
**Fix:** Recalculate CPM for each scenario; track float consumption, path changes, and successor impacts.

### LLM Hallucination
**Issue:** LLM might invent numbers ("50% chance of delay") without evidence.
**Prevention:** LLM must call analysis tools; only use numerical results from CPM/ML models.

### Circular Dependencies
**Issue:** Data might contain circular dependency chains (Activity A → B → C → A).
**Prevention:** Validate during Phase 2; flag and exclude circular chains before CPM (Phase 3).

### Large Model Files
**Issue:** Trained `.pkl` or `.joblib` files can be 100MB+; don't commit to Git.
**Prevention:** `.gitignore` excludes `models/*.pkl` and `models/*.joblib`. Regenerate on deployment.

---

## Development Workflow

### Before Every Commit
```powershell
uv run ruff check --fix
git add .
git commit -m "Descriptive message"
git push
```

### Running Tests
```powershell
uv run pytest tests/ -v
```

### Running Jupyter Notebooks
```powershell
uv run jupyter notebook
```

### Linting & Formatting
```powershell
uv run ruff check --fix
```

---

## Notes & Decisions

- **Python 3.12** — Modern, fast, stable. All dependencies support it via `uv`.
- **`uv` over `pip`** — Faster, more reliable, better dependency resolution.
- **Free tier stack** — Groq (LLM), Open-Metoe (weather), Render (deployment), Neon/Supabase (optional DB).
- **Project-level train/test split** — Prevents data leakage in Phase 4.
- **NetworkX for CPM** — Mature graph library; well-documented for scheduling problems.
- **Streamlit for dashboard** — Rapid prototyping, no frontend expertise needed.
- **Single data loader module** — `src/data/loader.py` is reused across all phases (exploratory, CPM, ML, dashboard).

---

## Team / Contact
- **Project Owner:** Samiya Shaikh
- **GitHub:** https://github.com/SarinaShaikh/AI-Construction-Project-Risk-Delay-Predictor
- **Email:** samiyaazgar@gmail.com

---

**Last Updated:** Phase 2 Complete — Ready for Phase 3 (CPM Engine)
