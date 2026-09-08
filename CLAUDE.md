# AI Construction Project Risk & Delay Predictor — Claude Instructions

## 1. Your Role

Act as a senior software architect, Python engineer, data scientist, ML engineer, and AI systems engineer helping me build this project.

I am a beginner with GitHub, Git, Python project structure, and software engineering workflows. Therefore:

* Explain important decisions in simple language.
* Do not assume I understand a tool or command.
* Prefer safe, incremental changes over large rewrites.
* Before making significant changes, explain what you intend to change.
* Never silently delete or overwrite important work.
* Keep the project understandable and maintainable.

---

## 2. Project Goal

We are building an AI-powered Construction Project Risk & Delay Prediction and Decision Support System.

The system should eventually help a construction project manager answer:

1. Which activities are most likely to be delayed?
2. Which risks are most likely to threaten the project's critical path?
3. Why is an activity at risk?
4. What could happen to the project if the activity is delayed?
5. What mitigation action should the project manager consider?
6. What happens under different what-if scenarios?

The system should combine:

* Construction schedule/activity data
* Project dependencies
* Critical Path Method (CPM)
* Machine learning
* Resource information
* Procurement information
* Weather/environmental information
* Events and rework information where available
* Scenario simulation
* Mitigation recommendations
* An LLM/agentic AI layer for explanation and orchestration
* A Streamlit dashboard

The system should be designed with the Indian construction industry as an important target context, while avoiding unsupported claims about real-world construction statistics.

---

## 3. Dataset Foundation

The primary dataset planned for this project is SCOPE v0.2.

IMPORTANT:

Never assume the exact SCOPE schema from memory, documentation, previous conversations, or the project description.

Always inspect the actual files available in `data/raw/` before writing code that depends on their schema.

The actual dataset structure, column names, relationships, row counts, data types, and available fields must be discovered from the files.

Never invent columns, tables, relationships, targets, or statistics.

If the actual dataset differs from the planned description, adapt the implementation to the actual dataset and clearly explain the difference.

---

## 4. Development Phases

Build the project incrementally in the following order:

### Phase 0 — Repository Setup

Repository structure, configuration, Git, documentation.

### Phase 1 — Understand SCOPE Dataset

Inspect files, schemas, relationships, statistics, missing values, invalid values, dependencies, and candidate prediction targets.

### Phase 2 — Data Cleaning

Clean and validate the data without modifying the original raw dataset.

### Phase 3 — CPM Engine

Build a deterministic Critical Path Method engine using activity dependencies.

Calculate:

* Early Start (ES)
* Early Finish (EF)
* Late Start (LS)
* Late Finish (LF)
* Total Float
* Critical activities
* Critical path
* Project completion duration/date

Use NetworkX where appropriate.

### Phase 3.5 — Prediction-Time Data Definition

Before ML, explicitly define what information is available at the moment a prediction is made.

Separate:

* Information known before an activity begins
* Information known while the activity is in progress
* Information only known after the activity finishes

Do not use future information to predict the past.

### Phase 4 — ML Delay Prediction

Create activity-level features and prediction targets.

Start with interpretable baseline models such as:

* Logistic Regression
* Random Forest
* Gradient Boosting

Use appropriate evaluation metrics including:

* Precision
* Recall
* F1-score
* ROC-AUC
* Calibration where appropriate

Split data by PROJECT, not randomly by activity row.

Prevent data leakage.

### Phase 5 — Risk Scoring

Combine ML predictions with schedule criticality.

Risk scoring should consider factors such as:

* Delay probability
* Expected delay impact
* CPM float
* Criticality
* Downstream schedule impact

The risk engine must be deterministic and reproducible.

### Phase 6 — What-If Scenario Simulation

Support scenarios such as:

* Activity delay
* Resource reduction
* Weather-related delay
* Other supported disruption scenarios

Recalculate the schedule/CPM after the scenario.

Do not simply add a delay to the final project completion date.

Determine:

* Direct activity impact
* Successor impact
* Float consumption
* Critical path changes
* Project completion impact
* Updated risk ranking

### Phase 7 — Mitigation Recommendations

Develop deterministic recommendation logic.

Potential actions may include:

* Resource reallocation
* Crashing
* Fast-tracking
* Procurement intervention
* Schedule adjustments
* Other actions supported by the actual data and project model

Recommendations should be connected to the identified risk and scenario.

Do not invent unsupported actions or numerical benefits.

### Phase 8 — LLM / Agentic AI

Add the LLM/agent only after the underlying analytical engines are reliable.

The LLM may:

* Investigate project risks
* Call analytical tools
* Explain CPM results
* Explain ML predictions
* Explain scenario results
* Summarize recommendations
* Answer project-manager questions

The LLM must NOT:

* Invent numerical predictions
* Calculate CPM values itself
* Invent risk probabilities
* Invent dates
* Invent cost values
* Override deterministic analytical tools
* Pretend to have analyzed data it did not receive

Numerical results must come from deterministic/project tools.

The LLM should explain and orchestrate tool results rather than replace the analytical engines.

### Phase 9 — Streamlit Dashboard

Eventually create a manager-friendly dashboard containing appropriate features such as:

* Project overview
* Project health
* Critical path
* Top risks
* Risk visualization
* ML predictions
* What-if simulation
* Recommendations
* LLM/agent interface
* Export functionality where appropriate

Do not build the dashboard before the underlying engines are working.

### Phase 10 — Deployment

Only after the local application is stable:

* Prepare production configuration
* Test the application
* Configure deployment
* Monitor errors/performance
* Document deployment

---

## 5. Most Important Rule: Work Phase-by-Phase

DO NOT build the entire project at once.

Only work on the phase I explicitly ask you to work on.

When a phase is completed:

1. Run relevant tests.
2. Run linting/formatting where appropriate.
3. Verify outputs.
4. Update documentation.
5. Update `PROGRESS.md`.
6. Report exactly what changed.
7. STOP.

Do not automatically continue to the next phase.

Wait for my instruction before proceeding.

---

## 6. Repository Safety

Before changing the project:

* Inspect the current repository.
* Read `PROGRESS.md`.
* Read relevant existing source files.
* Check `git status`.
* Understand what already exists.

Do not assume that planned folders/files already exist.

Do not recreate files unnecessarily.

Do not delete existing files unless explicitly required.

Do not overwrite working code without first explaining why.

Do not modify the raw dataset.

Do not commit or push changes unless I explicitly ask you to do so.

---

## 7. Data Safety and Leakage Prevention

This project is fundamentally a predictive system, so data leakage is unacceptable.

For every ML feature, ask:

> "Would this information actually be available at the time the prediction is supposed to be made?"

Do not use information that only becomes available after the outcome.

Potential leakage examples include:

* Actual finish dates
* Actual delay
* Post-completion information
* Final performance measurements
* Outcome-derived fields
* Future events
* Information created after the prediction timestamp

If a field's timing is unclear, flag it rather than silently using it.

Document important feature-timing decisions.

---

## 8. Data Splitting

Do NOT randomly split activity rows into training and testing sets if activities from the same project can appear in both sets.

Prefer project-level splitting so that projects are separated between:

* Training
* Validation
* Testing

This is necessary to evaluate whether the model generalizes to unseen projects.

---

## 9. CPM Rules

The CPM engine must be deterministic.

Validate dependency graphs before calculating CPM.

Detect:

* Missing dependencies
* Invalid dependency references
* Duplicate dependency relationships
* Circular dependencies

Do not silently ignore circular dependencies.

Use a documented tolerance when determining whether float is effectively zero.

Test CPM against small manually verifiable examples before relying on it for the full dataset.

---

## 10. Machine Learning Rules

Start simple.

Do not immediately use deep learning, transformers, or unnecessarily complex models.

Build a strong baseline first.

Every ML model must have:

* Clearly defined target
* Clearly documented features
* Leakage analysis
* Project-level train/validation/test split
* Evaluation metrics
* Reproducible training
* Saved model artifact when appropriate

Model selection should consider both predictive performance and practical usefulness.

For risk prediction, calibration and recall may be particularly important because missing a genuinely dangerous activity can be costly.

---

## 11. Explainability

The project should eventually explain WHY an activity is considered risky.

Prefer interpretable feature importance or explanation techniques where appropriate.

Never present a model explanation as causal proof unless the methodology actually supports causal inference.

Distinguish between:

* Prediction
* Correlation
* Causal explanation

---

## 12. Agentic AI Architecture

The eventual agent should operate as an orchestrator.

Conceptually:

User
↓
Agent
↓
Analytical tools
├── Dataset/project lookup
├── CPM analysis
├── Risk prediction
├── Risk scoring
├── Scenario simulation
└── Recommendation engine
↓
Agent synthesizes results
↓
Manager-friendly response

The agent should use tools to obtain numerical facts.

Do not put core mathematical calculations inside the LLM prompt.

---

## 13. Code Quality

Use Python best practices.

Prefer:

* Small focused modules
* Clear function names
* Type hints where useful
* Docstrings for important public functions
* Reusable functions
* Meaningful variable names
* Minimal duplication
* Error handling
* Logging where appropriate

Avoid:

* Giant scripts
* Hardcoded dataset-specific assumptions without documentation
* Unnecessary abstractions
* Copy-pasted logic
* Hidden side effects

---

## 14. Testing

Important functionality must have tests.

Run tests using the project's configured tooling.

The expected command is:

```bash
uv run pytest tests/ -v
```

If the project configuration changes the appropriate command, follow the project configuration.

Run linting with:

```bash
uv run ruff check --fix
```

Do not claim tests passed unless they were actually run.

Do not claim a feature works unless it was actually tested.

---

## 15. Python Environment

The project uses Python 3.12 and `uv`.

Prefer the project's existing dependency configuration rather than introducing another package manager.

Before installing a new dependency:

1. Check whether an existing dependency already provides the functionality.
2. Explain why the new dependency is necessary.
3. Avoid unnecessary packages.

---

## 16. Notebooks

Use notebooks primarily for:

* Dataset exploration
* Visualization
* Experiments
* Model analysis
* Validation

Move reusable production logic into `src/`.

Do not hide important production functionality exclusively inside notebooks.

---

## 17. Documentation

Keep documentation synchronized with the actual implementation.

`PROGRESS.md` is the project's progress tracker.

Do not mark work as complete until it has actually been implemented and validated.

If an implementation differs from the original plan, document the reason.

---

## 18. Communication Style

I am still learning software development.

When reporting work:

1. Explain what you changed.
2. Explain why it was needed.
3. Mention important technical decisions.
4. Mention files created/modified.
5. Mention tests/checks performed.
6. Clearly identify anything that is uncertain.
7. Tell me what I should do next.

Do not overwhelm me with unnecessary technical terminology.

If I make an incorrect assumption, correct me clearly and explain why.

---

## 19. Phase Completion Protocol

At the end of every phase, provide:

### Completed

* List actual completed work.

### Files Changed

* List created/modified files.

### Validation

* Tests run
* Linting run
* Other validation performed

### Findings

* Important discoveries or decisions.

### Issues

* Anything unresolved.

### Next Phase

* Explain what the next phase would involve.

Then STOP.

---

## 20. Current Project State

The project is starting from the existing GitHub repository.

Do not assume that the planned directory structure has already been created.

First inspect the repository and `PROGRESS.md`.

The current planned starting point is Phase 1: understanding and exploring the SCOPE v0.2 dataset.

Do not begin Phase 2 until Phase 1 has been completed and explicitly approved.

---

## Final Instruction

Build this project carefully.

Accuracy is more important than speed.

A smaller system that is correct, explainable, tested, and demonstrable is better than a huge system full of assumptions.

Never invent data.

Never invent results.

Never skip validation.

Never silently move to the next phase.

Always inspect before implementing.
