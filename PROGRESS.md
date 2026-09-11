# AI Construction Project Risk & Delay Predictor — Progress Tracker

## Current Status

**Phase: 7 — Risk Analysis, Project Risk Summary & Reports — ✅ COMPLETE**

**Next Task: Phase 8 — LLM / Agentic AI**

---

## Completed Phases

### Phase 0 — Project Setup ✅

**Objective:** Initialize the project, development environment, repository, and folder structure.

**Completed:**

* ✅ Initialize Git repository on GitHub
* ✅ Create project folder structure
* ✅ Create `pyproject.toml` with Python 3.12 + project metadata
* ✅ Create `.gitignore`
* ✅ Create `PROGRESS.md`
* ✅ Configure `uv` for dependency and environment management
* ✅ Push project to GitHub

**Repository:**

```text
https://github.com/SarinaShaikh/AI-Construction-Project-Risk-Delay-Predictor
```

**Repository Structure:**

```text
AI-Construction-Project-Risk-Delay-Predictor/

├── data/
│   ├── raw/                  (SCOPE v0.2 dataset)
│   ├── processed/            (Generated/processed data - ignored by Git)
│   └── exports/              (Generated exports - ignored by Git)
│
├── notebooks/                (Jupyter notebooks and analysis)
│
├── src/
│   ├── ai_construction/      (Main package)
│   ├── cpm/                  (Critical Path Method)
│   ├── data/                 (Data loading and cleaning)
│   ├── features/             (Feature engineering)
│   ├── models/               (ML models)
│   ├── risk/                 (Risk scoring and analysis)
│   ├── scenarios/            (What-if simulation)
│   ├── recommendations/      (Mitigation recommendations)
│   └── llm/                  (LLM and agent integration)
│
├── app/                      (Streamlit dashboard)
├── tests/                    (Automated tests)
├── scripts/                  (Standalone utilities)
│
├── PROGRESS.md
├── README.md
├── pyproject.toml
└── .gitignore
```

---

### Phase 1 — Understand SCOPE v0.2 Dataset ✅

**Objective:** Explore the dataset structure and understand projects, activities, dependencies, and data quality.

**Completed:**

* ✅ SCOPE v0.2 dataset analyzed
* ✅ 25 CSV files analyzed
* ✅ 100 projects identified
* ✅ 9,279 activities identified
* ✅ 18,176 dependencies identified
* ✅ No null values in core data
* ✅ No duplicate records
* ✅ Referential integrity validated
* ✅ Created reusable data loader
* ✅ Created dataset exploration notebook

**Deliverables:**

* `src/data/loader.py`
* `notebooks/01_dataset_exploration.ipynb`

---

### Phase 2 — Data Cleaning & Preparation ✅

**Objective:** Clean, validate, and prepare the dataset for CPM and ML processing.

**Completed:**

* ✅ Protected raw data using Git ignore rules
* ✅ Created deterministic cleaning pipeline
* ✅ Validated 12 core tables
* ✅ No rows lost during cleaning
* ✅ No required columns dropped
* ✅ Zero missing values
* ✅ Zero duplicates
* ✅ Zero invalid dates/durations
* ✅ Validated all 18,176 dependencies
* ✅ Zero orphan dependencies
* ✅ Zero circular dependencies
* ✅ Created target variable `target_event_delay_days`
* ✅ Aggregated event delays at activity level
* ✅ Established prediction-time availability rules
* ✅ Prevented future-data leakage

**Deliverables:**

* `scripts/clean_scope.py`
* `tests/test_clean_scope.py`
* `notebooks/02_data_cleaning.md`
* Processed dataset under `data/processed/`

**Validation:**

* ✅ 47 Phase 2 tests passed

---

### Phase 3 — CPM Engine ✅

**Objective:** Implement a deterministic Critical Path Method engine for construction schedules.

**Completed:**

* ✅ Implemented NetworkX-free CPM engine
* ✅ Implemented Kahn's algorithm for topological processing
* ✅ Calculate Early Start (ES)
* ✅ Calculate Early Finish (EF)
* ✅ Calculate Late Start (LS)
* ✅ Calculate Late Finish (LF)
* ✅ Calculate Total Float
* ✅ Identify critical activities
* ✅ Support Finish-to-Start (FS)
* ✅ Support Start-to-Start (SS)
* ✅ Support Finish-to-Finish (FF)
* ✅ Support dependency lags
* ✅ Handle multiple terminal activities
* ✅ Implement virtual project-end node
* ✅ Implement cycle detection
* ✅ Validate all 100 projects

**Key Results:**

* 100 projects processed successfully
* 9,279 activities processed
* 18,176 dependencies processed
* 3,153 activities identified as critical
* 82.1% agreement with dataset `critical_path`
* Zero negative float values
* Zero constraint violations
* Zero dependency cycles

**Deliverables:**

* `src/cpm/calculation.py`
* `tests/test_cpm.py`
* `scripts/run_cpm.py`
* `notebooks/03_cpm_validation.md`
* `data/processed/cpm_results.csv`

**Validation:**

* ✅ 36 CPM tests passed

---

### Phase 4 — ML Delay Prediction ✅

**Objective:** Train machine-learning models to predict activity-level construction delay duration.

**Completed:**

* ✅ Created feature engineering pipeline
* ✅ Generated activity-level features
* ✅ Generated CPM-based features
* ✅ Added project context features
* ✅ Added resource features
* ✅ Added procurement features
* ✅ Added environmental features
* ✅ Added categorical features
* ✅ Applied one-hot encoding
* ✅ Generated approximately 40+ initial features
* ✅ Final feature set saved with model metadata
* ✅ Implemented project-level 70/15/15 train/validation/test split
* ✅ Prevented project-level data leakage
* ✅ StandardScaler fitted only on training data
* ✅ Trained baseline models
* ✅ Trained Random Forest model
* ✅ Trained Gradient Boosting model
* ✅ Performed hyperparameter tuning
* ✅ Evaluated MAE, RMSE, R² and MAPE
* ✅ Generated model comparison
* ✅ Generated feature importance analysis
* ✅ Saved best trained model

**Key Results:**

* Dataset: 9,279 activities
* Best model: `GradientBoostingRegressor`
* Test R²: approximately 0.5+
* Test RMSE: approximately 38 days
* Test MAE: approximately 14 days
* Good generalization
* Project-level split prevents leakage

**Deliverables:**

* `src/features/engineering.py`
* `scripts/prepare_ml_data.py`
* `notebooks/04_model_training.ipynb`
* `data/processed/ml/X_train.csv`
* `data/processed/ml/X_val.csv`
* `data/processed/ml/X_test.csv`
* `data/processed/ml/y_train.csv`
* `data/processed/ml/y_val.csv`
* `data/processed/ml/y_test.csv`
* `data/processed/ml/best_model.pkl`
* `data/processed/ml/best_model_metadata.json`
* `data/processed/ml/feature_names.json`
* `data/processed/ml/scaler.pkl`

---

### Phase 5 — Risk Scoring & Risk Analysis ✅

**Objective:** Calculate activity-level construction risk using ML predictions, schedule criticality, and float impact.

**Completed:**

* ✅ Implemented `RiskScorer`
* ✅ Integrated trained ML delay prediction model
* ✅ Generated activity-level delay predictions
* ✅ Calculated delay probability
* ✅ Calculated criticality weight
* ✅ Calculated float impact
* ✅ Calculated combined activity risk score
* ✅ Added High / Medium / Low activity risk levels
* ✅ Ranked activities by risk score
* ✅ Integrated CPM results with risk scoring
* ✅ Integrated risk scoring with scenario simulation
* ✅ Added automated validation tests

**Risk score components:**

```text
Delay Probability
        ×
Criticality Weight
        ×
Float Impact
        =
Activity Risk Score
```

**Key Results:**

* 9,279 activities analyzed
* Mean predicted delay: approximately 17.08 days
* Mean activity risk score: approximately 0.251
* Risk scores range from 0 to 1
* High-risk activities: approximately 8%
* Medium-risk activities: approximately 18%
* Low-risk activities: approximately 74%

---

### Phase 6 — What-if Simulation ✅

**Objective:** Model the impact of construction schedule scenarios and determine how changes affect the project schedule.

**Completed:**

* ✅ Created `src/scenarios/simulator.py`
* ✅ Created `scripts/run_scenarios.py`
* ✅ Created `tests/test_simulator.py`
* ✅ Added JSON scenario output
* ✅ Added CSV scenario output
* ✅ Recalculate CPM independently for every scenario
* ✅ Calculate direct activity impact
* ✅ Track downstream/successor activities
* ✅ Calculate baseline float
* ✅ Calculate scenario float
* ✅ Calculate float consumption
* ✅ Detect newly critical activities
* ✅ Detect no-longer-critical activities
* ✅ Detect critical path changes
* ✅ Calculate project completion duration impact
* ✅ Generate deterministic scenario risk score
* ✅ Generate scenario risk level
* ✅ Support activity delay
* ✅ Support resource allocation reduction
* ✅ Support weather delay
* ✅ Prevent incorrect logic that simply adds delay to project end date

**Scenarios Supported:**

#### 1. Activity / Material Delay

Delays a selected activity by a specified number of days and recalculates the CPM network.

#### 2. Resource Allocation Reduction

Reduces resource allocation by a specified percentage and recalculates the effective activity duration.

#### 3. Weather Delay

Adds a weather-related delay to an activity and recalculates the schedule impact.

**Validation Project:**

```text
Project: P00096
Activities: 171
```

**Scenario Results:**

| Scenario               |  Impact | Float Consumed | Downstream | Risk     |
| ---------------------- | ------: | -------------: | ---------: | -------- |
| Activity delay +7 days | +7 days |         0 days |         18 | 14 / Low |
| Resource reduction 20% |  0 days |         8 days |         23 | 12 / Low |
| Weather delay +5 days  | +5 days |         0 days |        170 | 10 / Low |

**Deliverables:**

* `src/scenarios/simulator.py`
* `scripts/run_scenarios.py`
* `tests/test_simulator.py`
* `data/processed/scenarios/scenario_results.json`
* `data/processed/scenarios/scenario_results.csv`

---

### Phase 7 — Risk Analysis, Project Risk Summary & Reports ✅

**Objective:** Convert activity-level ML and CPM risk results into meaningful project-level risk assessments, explanations, and manager-friendly reports.

**Completed:**

#### Activity-Level Risk Analysis

* ✅ Generated risk scores for all 9,279 activities
* ✅ Generated delay predictions
* ✅ Generated delay probabilities
* ✅ Calculated criticality weights
* ✅ Calculated float impact
* ✅ Ranked activities by risk
* ✅ Generated activity-level risk levels
* ✅ Generated deterministic explanations
* ✅ Generated mitigation recommendations for activities
* ✅ Achieved 100% explanation coverage
* ✅ Achieved 100% recommendation coverage

#### Project-Level Risk Summary

* ✅ Created `src/llm/project_summary.py`
* ✅ Aggregated activity risks at project level
* ✅ Counted High / Medium / Low activities
* ✅ Calculated average risk score
* ✅ Calculated maximum risk score
* ✅ Calculated average predicted delay
* ✅ Calculated maximum predicted delay
* ✅ Counted critical high-risk activities
* ✅ Implemented project-level High / Medium / Low classification
* ✅ Prevented overly aggressive project-level High classification
* ✅ Projects with no High-risk activities are classified as Low unless higher-risk conditions apply

**Final Project Risk Distribution:**

```text
Total Projects: 100

High:    14 projects
Medium:  19 projects
Low:     67 projects
```

#### Project Risk Reports

* ✅ Created `src/llm/report_generator.py`
* ✅ Generated manager-friendly project reports
* ✅ Included project risk summary
* ✅ Included top-risk activities
* ✅ Included predicted delay information
* ✅ Included risk levels
* ✅ Included explanations
* ✅ Included recommendations
* ✅ Generated reports for all 100 projects

#### End-to-End Risk Pipeline

* ✅ Created `scripts/run_risk_analysis.py`
* ✅ Integrated RiskScorer
* ✅ Integrated activity risk analyzer
* ✅ Integrated project risk summarizer
* ✅ Integrated report generator
* ✅ Generated complete Phase 7 outputs

**Phase 7 Outputs:**

```text
data/processed/risk/activity_risk_scores.csv
data/processed/risk/activity_risk_insights.csv
data/processed/risk/project_risk_summaries.csv
data/processed/reports/project_risk_reports.json
```

**Key Results:**

```text
Activities analyzed: 9,279
Projects summarized: 100
Project reports generated: 100

Mean predicted delay: 17.0777 days
Mean activity risk score: 0.2509
```

**Sample Project Reports:**

```text
P00001 → Medium risk
P00002 → Low risk
P00003 → Low risk
```

**Testing:**

* ✅ Project summary tests: 12 passed
* ✅ Complete project test suite: 124 passed
* ✅ Zero test failures
* ✅ All major project components validated

---

## Current Phase

### Phase 7 — COMPLETE ✅

The risk analysis layer is now complete.

The system can currently:

```text
Construction Schedule
        ↓
Data Cleaning
        ↓
CPM Calculation
        ↓
ML Delay Prediction
        ↓
Activity Risk Scoring
        ↓
Activity Risk Explanation
        ↓
Project Risk Aggregation
        ↓
Project Risk Report
```

The project is now ready for **Phase 8 — LLM / Agentic AI**.

---

## Upcoming Phases

### Phase 8 — LLM / Agentic AI

**Objective:** Use a Groq-based LLM agent to investigate construction risks and explain analytical results.

**LLM responsibilities:**

* [ ] Investigate project risks
* [ ] Call risk-analysis tools
* [ ] Explain CPM results
* [ ] Summarize ML delay predictions
* [ ] Explain what-if scenarios
* [ ] Explain project-level risk
* [ ] Explain high-risk activities
* [ ] Generate manager-friendly insights
* [ ] Generate natural-language risk investigation responses

**LLM Constraints:**

* ❌ Must NOT invent numerical predictions
* ❌ Must NOT calculate unsupported numerical results
* ❌ Must NOT replace the deterministic CPM engine
* ❌ Must NOT replace the trained ML model
* ✅ Must call analysis tools for numerical results
* ✅ Must use returned tool results as the numerical source of truth
* ✅ Can synthesize and explain validated results
* ✅ Can combine CPM, ML, risk, scenario, and recommendation outputs

**Planned Architecture:**

```text
User Question
      ↓
Groq LLM Agent
      ↓
Tool Selection
      ↓
┌─────────────────────────────┐
│ CPM Analysis Tool           │
│ Risk Analysis Tool          │
│ Scenario Simulation Tool    │
│ Recommendation Tool         │
└─────────────────────────────┘
      ↓
Validated Numerical Results
      ↓
Groq LLM
      ↓
Manager-Friendly Explanation
```

**Planned Deliverables:**

* `src/llm/agent.py`
* LLM tool definitions
* Tool integration
* Agent prompts
* Numerical grounding validation
* Agent tests

---

### Phase 9 — Streamlit Dashboard

**Objective:** Build an interactive dashboard for construction project managers.

**Planned Features:**

* [ ] Project overview
* [ ] Project duration and timeline
* [ ] Number of activities
* [ ] Number of dependencies
* [ ] Critical path visualization
* [ ] Risk heatmap
* [ ] Activity risk ranking
* [ ] ML delay predictions
* [ ] Project-level risk summary
* [ ] What-if scenario builder
* [ ] Mitigation recommendations
* [ ] LLM risk investigation chat
* [ ] Export reports
* [ ] CSV export
* [ ] PDF report export

**Planned Deliverables:**

* `app/dashboard.py`
* Dashboard components
* Visualization components
* Report export functionality

---

### Phase 10 — Deployment

**Objective:** Deploy the construction risk prediction platform for demonstration and production-style use.

**Target Platform:**

* Render free tier

**Planned Tasks:**

* [ ] Prepare production requirements
* [ ] Configure application startup
* [ ] Configure environment variables
* [ ] Deploy Streamlit dashboard
* [ ] Test production application
* [ ] Monitor logs
* [ ] Optimize startup time
* [ ] Document deployment process

**Planned Deliverables:**

* Deployed live dashboard
* Deployment documentation
* Production configuration

---

## Known Gotchas

### Data Leakage

**Issue:** ML models can produce unrealistic results if future information is included during prediction.

**Prevention:**

* Split data by project rather than individual rows
* Fit scalers only on training data
* Respect prediction-time availability
* Never use future events as prediction features

---

### CPM Tolerance

**Issue:** Floating-point calculations can make critical-path classification ambiguous.

**Prevention:**

* Use a documented float tolerance
* Apply the same tolerance consistently
* Validate critical activities through automated tests

---

### What-if Scenario Logic

**Issue:** Simply adding a delay to the final project completion date does not correctly model schedule behavior.

**Solution:**

* Modify the affected activity
* Recalculate CPM
* Recalculate float
* Identify downstream activities
* Detect critical-path changes
* Measure actual project completion impact

---

### LLM Hallucination

**Issue:** An LLM may invent numerical predictions or unsupported risk percentages.

**Prevention:**

* LLM must call analytical tools
* CPM remains the source of schedule calculations
* ML model remains the source of delay predictions
* Risk engine remains the source of risk scores
* Scenario engine remains the source of what-if results
* LLM can only explain validated tool outputs

---

### Circular Dependencies

**Issue:** Circular dependencies can make CPM calculations invalid.

**Prevention:**

* Validate dependency graph during Phase 2
* Detect cycles before CPM
* Reject invalid cyclic schedules

---

### Large Model Files

**Issue:** Serialized ML models can become large.

**Prevention:**

* Do not commit large generated model files to Git
* Use `.gitignore`
* Regenerate models when required
* Keep reproducibility metadata in the repository

---

## Development Workflow

### Before Every Commit

```powershell
uv run ruff check --fix
git status
git add .
git commit -m "Descriptive message"
git push
```

### Running Tests

```powershell
uv run pytest tests\ -v
```

### Running Risk Analysis

```powershell
uv run python scripts\run_risk_analysis.py
```

### Running What-if Scenarios

```powershell
uv run python scripts\run_scenarios.py
```

### Running CPM

```powershell
uv run python scripts\run_cpm.py
```

### Running Data Cleaning

```powershell
uv run python scripts\clean_scope.py
```

### Running Jupyter

```powershell
uv run jupyter notebook
```

### Linting

```powershell
uv run ruff check --fix
```

---

## Data Model — SCOPE v0.2

### Core Tables

#### `projects.csv`

Contains 100 construction projects.

Important columns:

```text
project_id
project_type
floors
area_m2
complexity
contractor_capability
resource_availability
management_maturity
weather_exposure
supply_chain_exposure
technology_maturity
planned_duration_days
planned_cost
```

#### `activities.csv`

Contains 9,279 construction activities.

Important columns:

```text
project_id
activity_id
phase
activity_name
resource_type
planned_duration_days
quantity
unit_cost
planned_cost
criticality
status
critical_path
predecessor_count
successor_count
```

#### `dependencies.csv`

Contains 18,176 activity dependencies.

Important columns:

```text
project_id
predecessor_id
successor_id
relationship
lag_days
```

This is the key table used to construct the CPM dependency graph.

---

### Supporting Tables

Important supporting data includes:

```text
activity_states.csv
construction_memory.csv
events.csv
environment.csv
decisions.csv
resources.csv
resource_allocation.csv
procurement.csv
rework.csv
outcomes.csv
```

These tables provide information about:

* Productivity
* Weather
* Site access
* Events
* Resources
* Procurement
* Delays
* Rework
* Project outcomes
* Construction risk factors

---

## Data Loader

The reusable loader is:

```text
src/data/loader.py
```

Primary function:

```python
data = load_raw_dataset()
```

The loader provides access to the SCOPE v0.2 tables and supports validation of:

* File availability
* Column names
* Null values
* Duplicate records
* Referential integrity
* Project IDs
* Activity IDs
* Dependency relationships

---

## Overall System Architecture

The current system follows this architecture:

```text
                 SCOPE v0.2 Dataset
                         │
                         ▼
                Data Loading & Cleaning
                         │
                         ▼
                   Feature Engineering
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        CPM Engine              ML Model
             │                       │
             │                Delay Prediction
             │                       │
             └───────────┬───────────┘
                         ▼
                  Risk Scoring
                         │
                         ▼
                Activity Risk Analysis
                         │
                         ▼
                Project Risk Summary
                         │
                         ▼
                 Risk Report Generator
                         │
                         ▼
                  What-if Simulation
                         │
                         ▼
             Mitigation Recommendations
                         │
                         ▼
                  LLM / Agentic AI
                         │
                         ▼
                Streamlit Dashboard
                         │
                         ▼
                    Deployment
```

---

## Testing Status

Current complete test suite:

```text
124 passed
0 failed
```

Latest test command:

```powershell
uv run pytest tests\ -v
```

Latest result:

```text
124 passed in 92.18s
```

The test suite covers:

* Data cleaning
* Dataset validation
* Referential integrity
* Dependency validation
* Cycle detection
* CPM calculations
* Float calculations
* Critical path detection
* ML data preparation
* Risk analysis
* Activity risk insights
* Project risk summaries
* Project reports
* What-if simulation
* Scenario validation

---

## Project Milestones

```text
Phase 0  ████████████████████ 100%  ✅
Phase 1  ████████████████████ 100%  ✅
Phase 2  ████████████████████ 100%  ✅
Phase 3  ████████████████████ 100%  ✅
Phase 4  ████████████████████ 100%  ✅
Phase 5  ████████████████████ 100%  ✅
Phase 6  ████████████████████ 100%  ✅
Phase 7  ████████████████████ 100%  ✅
Phase 8  ░░░░░░░░░░░░░░░░░░░░   0%  ⏳
Phase 9  ░░░░░░░░░░░░░░░░░░░░   0%  ⏳
Phase 10 ░░░░░░░░░░░░░░░░░░░░   0%  ⏳
```

---

## Notes & Decisions

* **Python 3.12** — Stable and compatible with the project dependencies.
* **uv** — Used for fast dependency management and reproducible environments.
* **Project-level train/test split** — Prevents data leakage in ML.
* **NetworkX-free CPM** — Deterministic implementation using Kahn's algorithm.
* **Deterministic risk scoring** — Provides explainable and reproducible activity risk scores.
* **CPM recalculation for scenarios** — Prevents incorrect schedule-impact calculations.
* **Streamlit** — Selected for rapid dashboard development.
* **Groq** — Planned for LLM/agentic analysis.
* **Open-Meteo** — Planned for weather-related extensions.
* **Render** — Planned deployment platform.
* **LLM numerical grounding** — The LLM will not invent numerical results and must rely on analytical tools.
* **Generated outputs** — Large processed datasets and model artifacts remain excluded from Git where appropriate.
* **Automated testing** — The project currently has 124 passing tests.

---

## Final Project Goal

The final system will provide construction project managers with an AI-assisted platform that can:

1. Analyze construction schedules.
2. Calculate the Critical Path.
3. Predict activity-level delays using machine learning.
4. Identify high-risk activities.
5. Calculate project-level risk.
6. Simulate what-if scenarios.
7. Recommend mitigation strategies.
8. Explain risks using an LLM agent.
9. Provide an interactive Streamlit dashboard.
10. Generate manager-friendly reports.

The intended final workflow is:

```text
Construction Schedule
        ↓
Risk Detection
        ↓
Delay Prediction
        ↓
Critical Path Analysis
        ↓
What-if Simulation
        ↓
Mitigation Recommendation
        ↓
LLM Explanation
        ↓
Manager Decision Support
```

---

## Team / Contact

**Project Owner:** Samiya Shaikh

**GitHub:**

https://github.com/SarinaShaikh/AI-Construction-Project-Risk-Delay-Predictor

**Email:**

[samiyaazgar@gmail.com](mailto:samiyaazgar@gmail.com)

---

**Last Updated:** Phase 7 Complete — Ready for Phase 8 (LLM / Agentic AI)
