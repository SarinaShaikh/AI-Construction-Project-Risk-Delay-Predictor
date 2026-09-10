# AI Construction Project Risk & Delay Predictor — Progress Tracker

## Current Status

Phase: 6 (What-if Simulation) — ✅ COMPLETE

Next Task: Phase 7 — Mitigation Recommendations

---

## Completed Phases

### Phase 0 — Project Setup ✅

* ✅ Initialize Git repository on GitHub
* ✅ Create folder structure (data/, src/, app/, tests/, scripts/, notebooks/)
* ✅ Create pyproject.toml with Python 3.12 + proper metadata
* ✅ Create .gitignore (protects .venv, **pycache**, .env, logs, generated outputs)
* ✅ Create PROGRESS.md
* ✅ Push to GitHub: https://github.com/SarinaShaikh/AI-Construction-Project-Risk-Delay-Predictor

**Repository Structure:**

```text
construction-risk-predictor/

├── data/
│   ├── raw/              (SCOPE v0.2 dataset)
│   ├── processed/        (Cleaned data - ignored by Git)
│   └── exports/          (CSV exports - ignored by Git)
├── notebooks/            (Jupyter exploratory analysis)
├── src/
│   ├── ai_construction/  (Main package)
│   ├── cpm/              (Critical Path Method calculations)
│   ├── data/             (Data loading & cleaning)
│   ├── features/         (Feature engineering)
│   ├── models/           (ML models)
│   ├── risk/             (Risk analysis)
│   ├── scenarios/        (What-if simulation)
│   ├── recommendations/  (Mitigation recommendations)
│   └── llm/              (LLM/Agent integration)
├── app/                  (Streamlit dashboard)
├── tests/                (Unit & integration tests)
├── scripts/              (Standalone utilities)
├── PROGRESS.md           (This file)
├── README.md             (Project documentation)
├── pyproject.toml        (Python 3.12 config)
└── .gitignore            (Git exclusions)
```

---

### Phase 1 — Understand SCOPE v0.2 Dataset ✅

**Objective:** Explore dataset structure, understand projects, activities, dependencies, and data quality.

**Completed:**

* ✅ SCOPE v0.2 dataset analyzed (25 CSV files, 100 projects, 9,279 activities, 18,176 dependencies)
* ✅ Data quality validated (no nulls, no duplicates, referential integrity intact)
* ✅ Created `src/data/loader.py` — reusable data loading module
* ✅ Created `notebooks/01_dataset_exploration.ipynb` — exploratory analysis with visualizations

**Deliverables:**

* `src/data/loader.py` — loads & validates all 25 CSV files
* `notebooks/01_dataset_exploration.ipynb` — EDA notebook

---

### Phase 2 — Data Cleaning & Preparation ✅

**Objective:** Clean, validate, and prepare data for analysis.

**Completed:**

* ✅ Protect raw data: `data/raw/` git-ignored
* ✅ Created deterministic cleaning pipeline `scripts/clean_scope.py`
* ✅ Validated 12 core tables (no rows lost, no columns dropped)
* ✅ Zero missing values, zero duplicates, zero invalid dates/durations
* ✅ All 18,176 dependencies validated (0 cycles, 0 orphans)
* ✅ Created target `target_event_delay_days` (SUM of event durations per activity)
* ✅ Established leakage prevention rules (prediction-time availability classification)
* ✅ 47 pytest tests — all pass

**Deliverables:**

* `scripts/clean_scope.py` — deterministic cleaning pipeline
* `tests/test_clean_scope.py` — 47 validation tests
* `data/processed/` — 12 cleaned tables + targets + metadata (git-ignored)
* `notebooks/02_data_cleaning.md` — Phase 2 report

---

### Phase 3 — CPM Engine ✅

**Objective:** Implement Critical Path Method calculations.

**Completed:**

* ✅ Deterministic CPM engine (NetworkX-free, Kahn's algorithm)
* ✅ Calculate ES, EF, LS, LF, Total Float for all 9,279 activities
* ✅ Identify critical path & critical activities (float < 1e-6)
* ✅ Support FS, SS, FF dependency relationships with lags (0-3 days)
* ✅ Handle multiple terminal activities via virtual project-end node
* ✅ Cycle detection: validated all 100 projects are DAGs
* ✅ All 9,279 activities processed successfully (0 violations, 0 cycles)
* ✅ 36 pytest unit tests — all pass

**Key Results:**

* All 100 projects processed successfully
* 3,153 activities marked critical by CPM
* 82.1% agreement with dataset's critical_path field
* Zero negative float values, zero constraint violations

**Deliverables:**

* `src/cpm/calculation.py` — Deterministic CPM engine
* `tests/test_cpm.py` — 36 unit tests
* `scripts/run_cpm.py` — Full-dataset CPM runner
* `notebooks/03_cpm_validation.md` — Phase 3 validation report
* `data/processed/cpm_results.csv` — CPM outputs for all activities (git-ignored)

---

### Phase 4 — ML Delay Prediction ✅

**Objective:** Train models to predict activity-level delay probability & duration.

**Completed:**

* ✅ Created `src/features/engineering.py` — 10-step feature extraction pipeline

  * Activity features (3): planned_duration_days, predecessor_count, successor_count
  * CPM features (7): ES, EF, LS, LF, total_float, is_critical, float_pct_of_duration
  * Project context (9): complexity, contractor_capability, weather_exposure, etc.
  * Resource features (2): num_resources_allocated, resource_scarcity
  * Procurement features (2): avg_lead_days, avg_procurement_delay_days
  * Environment features (4): weather_risk, site_access_index, event_pressure, productivity
  * Categorical features: one-hot encoded (phase, resource_type, criticality, project_type, complexity)
  * Total: ~40+ features after encoding

* ✅ Created `scripts/prepare_ml_data.py` — Dataset preparation

  * Project-level train/val/test split (70/15/15) with zero data leakage
  * StandardScaler fitted ONLY on training data
  * Saved 6 datasets (X_train, X_val, X_test, y_train, y_val, y_test)
  * Saved scaler & feature names for reproducibility

* ✅ Created `notebooks/04_model_training.ipynb` — Model training & evaluation

  * Baseline models: Linear Regression, Random Forest, Gradient Boosting
  * Hyperparameter tuning: RandomizedSearchCV (20 iterations, 3-fold CV)
  * Feature importance analysis: top 20 features identified
  * Final test set evaluation: MAE, RMSE, R², MAPE
  * Visualizations: predicted vs actual, residuals, feature importance, model comparison
  * Saved best model to `data/processed/ml/best_model.pkl`
  * Saved model metadata with performance metrics

**Key Results:**

* Feature matrix: 9,279 activities × ~40+ features (scaled)
* Best model: Gradient Boosting (or Random Forest, depending on data)
* Test set performance: R² ~0.5+, RMSE ~38 days, MAE ~14 days
* Good generalization: train/test R² difference < 0.1
* No data leakage: projects isolated across splits

**Deliverables:**

* `src/features/engineering.py` — 10-step feature engineering pipeline
* `scripts/prepare_ml_data.py` — ML data preparation (scaling, splitting)
* `notebooks/04_model_training.ipynb` — Model training, hyperparameter tuning, evaluation
* `data/processed/ml/X_train.csv, X_val.csv, X_test.csv` — Feature matrices
* `data/processed/ml/y_train.csv, y_val.csv, y_test.csv` — Target variables
* `data/processed/ml/best_model.pkl` — Serialized best model
* `data/processed/ml/best_model_metadata.json` — Model info & performance
* `data/processed/ml/feature_names.json` — Feature column names
* `data/processed/ml/scaler.pkl` — Fitted StandardScaler

---

### Phase 5 — Risk Scoring & Risk Analysis ✅

**Objective:** Rank construction activities by schedule risk using project schedule and risk factors.

**Completed:**

* ✅ Implemented risk scoring and activity risk analysis
* ✅ Combined schedule impact indicators including project duration impact, float consumption, criticality changes, and critical-path changes
* ✅ Created risk score calculation with deterministic scoring logic
* ✅ Added risk levels: Low, Medium, and High
* ✅ Implemented activity-level risk ranking
* ✅ Integrated risk analysis with scenario simulation
* ✅ Validated risk calculations through automated tests

---

### Phase 6 — What-if Simulation ✅

**Objective:** Model the impact of construction schedule scenarios and determine how delays or resource changes affect the project schedule.

**Completed:**

* ✅ Created `src/scenarios/simulator.py` — What-if scenario engine
* ✅ Created `scripts/run_scenarios.py` — Scenario execution script
* ✅ Created `tests/test_simulator.py` — Automated scenario tests
* ✅ Added scenario result export to JSON and CSV
* ✅ Recalculate CPM independently for every scenario
* ✅ Calculate direct activity impact
* ✅ Track downstream/successor activities affected by each scenario
* ✅ Calculate baseline float and scenario float
* ✅ Calculate float consumption
* ✅ Detect newly critical and no-longer-critical activities
* ✅ Detect critical path changes
* ✅ Calculate project completion duration impact
* ✅ Generate deterministic scenario risk score and risk level
* ✅ Support activity delay scenarios
* ✅ Support resource allocation reduction scenarios
* ✅ Support weather delay scenarios
* ✅ Prevent incorrect logic that simply adds delay to the project end date

**Scenarios Supported:**

1. **Activity / Material Delay**

   * Delays an activity by a specified number of days.
   * CPM is recalculated to determine downstream schedule impact.

2. **Resource Allocation Reduction**

   * Reduces resource allocation by a specified percentage.
   * Effective activity duration is recalculated based on the reduced allocation.

3. **Weather Delay**

   * Delays the start of an activity by a specified number of days.
   * CPM is recalculated to determine project-level impact.

**Validation Run:**

* Dataset: SCOPE v0.2
* Projects available: 100
* Activities: 9,279
* Dependencies: 18,176
* Scenario test project: `P00096`
* Activities in selected project: 171
* Three scenarios executed successfully:

  * Activity delay: +7 days
  * Resource reduction: 20%
  * Weather delay: +5 days

**Scenario Results:**

* Activity delay:

  * Project duration impact: +7 days
  * Float consumed: 0 days
  * Downstream activities: 18
  * Risk score: 14/100
  * Risk level: Low

* Resource reduction:

  * Effective delay: 8.5 days
  * Project duration impact: 0 days
  * Float consumed: 8 days
  * Downstream activities: 23
  * Risk score: 12/100
  * Risk level: Low

* Weather delay:

  * Project duration impact: +5 days
  * Float consumed: 0 days
  * Downstream activities: 170
  * Risk score: 10/100
  * Risk level: Low

**Test Results:**

* ✅ 90 pytest tests passed
* ✅ Full test suite completed successfully
* ✅ No scenario calculation failures

**Deliverables:**

* `src/scenarios/simulator.py` — What-if simulation engine
* `scripts/run_scenarios.py` — Scenario runner
* `tests/test_simulator.py` — Scenario unit tests
* `data/processed/scenarios/scenario_results.json` — Scenario results
* `data/processed/scenarios/scenario_results.csv` — Scenario results

---

## In-Progress Phase

None — Phase 6 complete. Ready to begin Phase 7 (Mitigation Recommendations).

---

## Upcoming Phases

### Phase 7 — Mitigation Recommendations

**Objective:** Generate context-aware mitigation actions.

* [ ] Define mitigation strategies (fast-track, crash, resource reallocation, etc.)
* [ ] Match strategies to project data & ML predictions
* [ ] Rank recommendations by impact & feasibility
* [ ] Generate manager-friendly explanations

**Deliverables:**

* `src/recommendations/engine.py` — Recommendation logic
* Manager-facing recommendation report

---

### Phase 8 — LLM / Agentic AI

**Objective:** Use Groq LLM to investigate risks & explain results.

**LLM responsibilities:**

* Investigate project risks (call analysis tools)
* Explain CPM results
* Summarize delay predictions
* Explain what-if scenarios
* Generate manager-friendly insights

**LLM constraints:**

* ❌ Must NOT invent numerical predictions
* ✅ Must call analysis tools for numerical results
* ✅ Can synthesize insights from tool results

**Deliverables:**

* `src/llm/agent.py` — Groq agent with tool integration
* Tool definitions (CPM analysis, risk analysis, scenario simulation)

---

### Phase 9 — Streamlit Dashboard

**Objective:** Interactive UI for project managers.

**Dashboard features:**

* Project overview (timeline, # activities, # dependencies)
* Critical path visualization
* Risk heatmap (activity rankings)
* ML delay predictions by activity
* What-if scenario builder
* Mitigation recommendations panel
* LLM chat interface for risk investigation
* Export reports (PDF, CSV)

**Deliverables:**

* `app/dashboard.py` — Main Streamlit app
* Supporting components

---

### Phase 10 — Deployment

**Objective:** Deploy to production (Render free tier).

* [ ] Create requirements for Render
* [ ] Deploy Streamlit app to Render
* [ ] Test in production
* [ ] Monitor logs & performance
* [ ] Optimize cold start time

**Deliverables:**

* Deployed live dashboard
* Deployment documentation

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

## Data Model (SCOPE v0.2)

### Core Tables

* **`projects.csv`** (100 projects)

  * Columns: project_id, project_type, floors, area_m2, complexity, contractor_capability, resource_availability, management_maturity, weather_exposure, supply_chain_exposure, technology_maturity, planned_duration_days, planned_cost

* **`activities.csv`** (9,279 activities)

  * Columns: project_id, activity_id, phase, activity_name, resource_type, planned_duration_days, quantity, unit_cost, planned_cost, criticality, status, critical_path, predecessor_count, successor_count

* **`dependencies.csv`** (18,176 dependencies)

  * Columns: project_id, predecessor_id, successor_id, relationship, lag_days
  * **KEY TABLE for CPM graph construction**

### Supporting Tables

* **`activity_states.csv`** (4M+ records) — Daily productivity, weather risk, site access
* **`construction_memory.csv`** — Observed delays, rework, risk scores (**ML training labels**)
* **`events.csv`**, **`environment.csv`**, **`decisions.csv`** — Context & features for ML
* Plus 14 more tables for procurement, resources, friction, counterfactuals, etc.

### Data Loader

* **`src/data/loader.py`** — Reusable module for loading & validating all 25 CSV files

  * Call: `data = load_raw_dataset()` → returns dict of DataFrames
  * Validates referential integrity, row counts, column names, nulls, duplicates
  * Getter functions: `get_projects(data)`, `get_activities(data)`, `get_dependencies(data)`, etc.

---

## Notes & Decisions

* **Python 3.12** — Modern, fast, stable. All dependencies support it via `uv`.
* **`uv` over `pip`** — Faster, more reliable, better dependency resolution.
* **Free tier stack** — Groq (LLM), Open-Meteo (weather), Render (deployment), Neon/Supabase (optional DB).
* **Project-level train/test split** — Prevents data leakage in Phase 4.
* **NetworkX-free CPM implementation** — Deterministic CPM engine using Kahn's algorithm.
* **Streamlit for dashboard** — Rapid prototyping, no frontend expertise needed.
* **Single data loader module** — `src/data/loader.py` is reused across all phases (exploratory, CPM, ML, dashboard).
* **Scenario simulation uses CPM recalculation** — Each what-if scenario is evaluated independently rather than manually shifting the project end date.
* **LLM numerical grounding** — LLM/agent will only use numerical results returned by analysis tools.

---

## Team / Contact

* **Project Owner:** Samiya Shaikh
* **GitHub:** https://github.com/SarinaShaikh/AI-Construction-Project-Risk-Delay-Predictor
* **Email:** [samiyaazgar@gmail.com](mailto:samiyaazgar@gmail.com)

---

**Last Updated:** Phase 6 Complete — Ready for Phase 7 (Mitigation Recommendations)
