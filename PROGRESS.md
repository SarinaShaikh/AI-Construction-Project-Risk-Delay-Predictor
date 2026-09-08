# AI Construction Project Risk & Delay Predictor — Progress Tracker

## Current Status
**Phase:** 1 (Understand SCOPE v0.2 Dataset)  
**Last Updated:** 2026-01-XX  
**Next Task:** Download & explore SCOPE v0.2 dataset structure

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

## In-Progress Phase


---

## In-Progress Phase

### Phase 1 — Understand SCOPE v0.2 Dataset
**Objective:** Explore dataset structure, understand projects, activities, dependencies, and data quality.

**Dataset Stats:**
- 100 projects
- 9,279 activities
- 18,176 dependencies (stored in dependencies.csv)
- 4M+ activity-state records
- 25 total CSV files

**Completed Tasks:**
- ✅ SCOPE v0.2 dataset accessed (25 CSV files in data/raw/)
- ✅ Created `src/data/loader.py` — Data loading & validation module
  - Loads all 25 CSV files with pandas
  - Validates schema (row counts, columns, nulls, duplicates, referential integrity)
  - Provides convenience getter functions (get_projects, get_activities, etc.)
  - Tested and working ✓

**Remaining Tasks:**
- [ ] Create `notebooks/01_dataset_exploration.ipynb` — Exploratory data analysis notebook
  - Load data using loader.py
  - Generate summary statistics (#projects, #activities, #dependencies, avg activities/project)
  - Analyze delay distributions (observed_delay_days)
  - Check data quality (missing values, outliers, duplicates)
  - Resource type analysis
  - Criticality breakdown
  - Create visualizations (histograms, distributions, correlations)
  - Document findings

**Deliverables:**
- `notebooks/01_dataset_exploration.ipynb` — Jupyter notebook with EDA & visualizations
- Summary statistics (projects, activities, dependencies, delays)
- Data quality assessment report

---

## Upcoming Phases

### Phase 2 — Data Cleaning & Preparation
**Objective:** Clean, validate, and prepare data for analysis.

- [ ] Handle missing values (dates, durations, resource info)
- [ ] Standardize date formats (ISO 8601)
- [ ] Validate durations (no negative values)
- [ ] Remove/flag duplicate activities
- [ ] Validate dependency relationships (no circular deps)
- [ ] Create cleaned dataset in `data/processed/`
- [ ] Document all transformations

---

### Phase 3 — CPM Engine
**Objective:** Implement Critical Path Method calculations.

- [ ] Build dependency graph with NetworkX
- [ ] Calculate ES (Earliest Start), EF (Earliest Finish)
- [ ] Calculate LS (Latest Start), LF (Latest Finish)
- [ ] Calculate Total Float for each activity
- [ ] Identify Critical Path (activities with float ≈ 0)
- [ ] Handle near-zero float tolerance (e.g., float < 0.1 days = critical)
- [ ] Return results: ES, EF, LS, LF, Total Float, Critical Path
- [ ] Validate CPM against known examples

**Deliverables:**
- `src/cpm/calculation.py` — CPM algorithm
- `tests/test_cpm.py` — CPM unit tests
- `notebooks/03_cpm_validation.ipynb` — Validation against sample projects

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

**Last Updated:** Phase 1 In Progress — loader.py complete, next: exploratory notebook