# Phase 4 — ML Problem Design and Feature Audit

**Date:** 2026-09-09
**Status:** Design-only — no models trained, no ML pipeline created.

> This document defines the Phase 4 prediction problem, audits prediction-time feature
> availability, and proposes a leakage-safe ML formulation. It does **not** train models,
> create a pipeline, or implement any feature engineering. Those belong to later sub-steps
> of Phase 4, after this design is approved.

---

## 1. Prediction Problem

**Prediction unit:** one activity.

For each `(project_id, activity_id)` we define a single prediction row.

**Target:** `target_event_delay_days`

```
target_event_delay_days =
    SUM(events.duration_days)       per (project_id, activity_id)
    0                               when the activity has no events
```

This is the same target fixed in Phase 1 and constructed in Phase 2.

**Semantics:** event-induced / disruption delay days per activity.

**Important:** this is **NOT** actual schedule slippage. The SCOPE v0.2 dataset does not
contain reliable actual activity start/finish timestamps (`activities.status = "Planned"`
for all 9,279 activities; no actual finish data exists). Therefore this model predicts the
delay associated with disruption events, **not** measured schedule lateness.

**Prediction-time interpretation:**

- The model answers: given only information that would plausibly be known *before* the
  event-induced delay is realized, what is the expected event-induced delay for this
  activity?
- Anything that is only known after the disruption occurs must not be used as a feature.

---

## 2. Target Distribution

All statistics below are taken from the Phase 2 target artifact
`data/processed/targets_event_delay_days.csv`, independently verified in this session.

| Metric | Value |
|--------|-------|
| Activities | 9,279 |
| target = 0 | 523 (5.64%) |
| target > 0 | 8,756 (94.36%) |
| Minimum | 0 |
| Maximum | 168 |
| Mean | 16.7741 |
| Median | 13 |
| Std dev | 15.4042 |
| P25 | 5 |
| P50 | 13 |
| P75 | 24 |
| P90 | 38 |
| P95 | 47 |
| P99 | 67 |

**Implications for modeling:**

- The target is highly right-skewed.
- The zero class is very small (5.64%).
- A binary "delay > 0" classification would be extremely imbalanced in the *negative* direction
  (only 5.64% negative class).
- A plain regression on the raw target would be dominated by the large mass of positive values
  and would have to handle the zeros explicitly.

---

## 3. Prediction-Time Availability

Prediction-time availability was classified in Phase 2 and stored in
`data/processed/prediction_time_availability.csv`. The authoritative counts are:

| Classification | Columns |
|----------------|---------|
| AVAILABLE_AT_PREDICTION_TIME | 75 |
| UNKNOWN | 18 |
| OUTCOME_OR_POST_OUTCOME | 23 |

These counts are **not** changed in this document.

### 3.1 AVAILABLE — eligible candidate features

These columns are judged defensible as prediction-time information. They describe planned
or structural attributes that a project manager could reasonably know before event-induced
delay occurs.

Safe groups include:

- Project-level planning attributes
- Activity-level planned attributes
- Dependency/network structure
- Resource definitions and planned allocations
- Planned procurement lead times (not actuals)

### 3.2 UNKNOWN — excluded for now

These columns are not assumed safe. They are **excluded** from the Phase 4 feature set
unless a later, explicit audit establishes prediction-time availability.

This includes many daily time-varying observations and decision-process attributes whose
availability depends on the exact prediction point.

### 3.3 OUTCOME / POST-OUTCOME — leakage, must not be used

These columns are generated after or because of the outcome. They **must not** become
predictive features.

---

## 4. Safe Feature Sources

The following processed tables contain defensible prediction-time candidate features.

### 4.1 Projects

Source: `data/processed/projects.csv`

| Column | Meaning | Availability |
|--------|---------|--------------|
| project_type | project sector/type | AVAILABLE |
| floors | number of floors | AVAILABLE |
| area_m2 | project area | AVAILABLE |
| complexity | project complexity rating | AVAILABLE |
| contractor_capability | contractor capability rating | AVAILABLE |
| resource_availability | resource availability rating | AVAILABLE |
| management_maturity | management maturity rating | AVAILABLE |
| weather_exposure | weather exposure rating | AVAILABLE |
| supply_chain_exposure | supply chain exposure rating | AVAILABLE |
| technology_maturity | technology maturity rating | AVAILABLE |
| planned_duration_days | planned project duration | AVAILABLE |
| planned_cost | planned project cost | AVAILABLE |

These are project-level, planned, and available before activity execution.

### 4.2 Activities

Source: `data/processed/activities.csv`

| Column | Meaning | Availability |
|--------|---------|--------------|
| phase | activity phase | AVAILABLE |
| phase_order | phase ordering | AVAILABLE |
| activity_sequence | activity sequence within phase | AVAILABLE |
| resource_type | planned resource type | AVAILABLE |
| planned_duration_days | planned activity duration | AVAILABLE |
| quantity | activity quantity | AVAILABLE |
| unit_cost | unit cost | AVAILABLE |
| planned_cost | planned cost | AVAILABLE |
| criticality | criticality rating | AVAILABLE |
| predecessor_count | number of predecessors | AVAILABLE |
| successor_count | number of successors | AVAILABLE |
| project_network_duration | project network duration | AVAILABLE |

Status, `critical_path`, and `critical_path_position` are treated carefully below.

### 4.3 Dependencies

Source: `data/processed/dependencies.csv`

| Column | Meaning | Availability |
|--------|---------|--------------|
| relationship | FS / SS / FF | AVAILABLE |
| lag_days | integer lag 0–3 | AVAILABLE |

These describe the planned schedule network, so they are available at prediction time.

### 4.4 Resources

Source: `data/processed/resources.csv`

| Column | Meaning | Availability |
|--------|---------|--------------|
| resource_category | Labour / Equipment / Material | AVAILABLE |
| resource_type | resource type | AVAILABLE |
| capacity | resource capacity | AVAILABLE |
| availability | resource availability | AVAILABLE |
| daily_cost | daily cost | AVAILABLE |

### 4.5 Resource Allocation

Source: `data/processed/resource_allocation.csv`

| Column | Meaning | Availability |
|--------|---------|--------------|
| resource_id | linked resource | AVAILABLE |
| allocation_fraction | planned allocation fraction | AVAILABLE |
| resource_category | allocation category | AVAILABLE |

This is a planned allocation table, so it is a defensible prediction-time source.

### 4.6 Environment and Activity States — restricted use only

The Phase 2 classification marks the daily environment and activity-state fields as UNKNOWN
for prediction-time availability.

Therefore:

- They are **not** included in the default Phase 4 feature set.
- If later used, only aggregates over a defensible prediction window could be considered, and
  only if the window can be justified as known at prediction time.
- The current conservative stance is to exclude them.

---

## 5. Leakage Exclusions

The following are **excluded** as predictive features.

### 5.1 Events and event-derived fields

Source tables: `events.csv` and any field derived from it.

Excluded because:

- `events` is the source of the target (`SUM(duration_days)`).
- Event type, severity, impact factor, duration_days, date, cause node, and any per-activity
  event count/aggregate are outcome-derived.
- Using them would leak the target into the features.

### 5.2 Target-derived fields

Excluded because they are direct or indirect functions of the target.

### 5.3 Rework outcomes

Source table: `rework.csv`

Excluded because:

- Rework is a post-event outcome.
- `rework_days`, `rework_cost`, `event_id`, `activity_id`, `project_id`, `rework_id`, and
  `split` for rework are outcome/post-outcome.

### 5.4 Post-outcome decisions

Source table: `decisions.csv`

Excluded because:

- `actual_decision_day`, `decision_delay_days`, `decision_status` are decision actuals known
  only after the decision is made.
- Many decision-process fields are UNKNOWN for prediction-time availability and are excluded
  by default.

### 5.5 Actual procurement outcomes

Source table: `procurement.csv`

Excluded:

- `actual_lead_days` and `procurement_delay_days` are realized outcomes.
- `planned_lead_days` is the safer planned-side field and may be kept where appropriate.

### 5.6 Future environment observations

Source table: `environment.csv`

Excluded by default:

- Daily observations (`rainfall_mm`, `temperature_c`, `humidity_pct`, `site_access_index`,
  `weather_risk`, `extreme_weather`) are future time-series values relative to a
  prediction-time point.
- Their availability at a specific prediction moment is not established, so they are UNKNOWN
  and excluded for now.

### 5.7 Future activity-state observations

Source table: `activity_states.csv`

Excluded by default:

- `productivity_index`, `weather_risk`, `site_access_index`, `event_pressure` are
  execution-time or event-derived daily observations.
- Phase 2 already classified `event_pressure` as OUTCOME and the other state fields as
  UNKNOWN.
- They are not used as features in this design.

### 5.8 Observed delay fields and outcome fields

Excluded:

- Any field that represents observed delay, realized delay, or outcome status.
- This includes construction-memory observed delay fields and any field that would only be
  known after disruption.

---

## 6. Feature Engineering Proposal

The following feature groups are proposed for later implementation. All are designed to use
only prediction-time information.

### 6.1 Activity-level planned features

- `planned_duration_days` — raw
- `quantity` — raw
- `planned_cost` — raw
- `unit_cost` — raw
- `criticality` — raw
- `resource_type` — categorical encoding
- `phase` — categorical encoding

### 6.2 Network structure features

From `dependencies.csv` and `activities.csv`:

- `predecessor_count` — raw
- `successor_count` — raw
- `out_degree` = successor_count — derived
- `in_degree` = predecessor_count — derived
- count of FS dependencies for the activity — derived
- count of SS dependencies for the activity — derived
- count of FF dependencies for the activity — derived
- total lag on incoming FS edges — derived
- total lag on outgoing FS edges — derived
- whether the activity has any FF dependency — derived binary
- whether the activity has any SS dependency — derived binary
- max lag across incoming dependencies — derived
- min lag across incoming dependencies — derived

These describe the planned schedule network only.

### 6.3 Project-level aggregates

From `projects.csv`, joined to each activity in the project:

- `project_type` — categorical, shared by all activities in the project
- `floors` — shared
- `area_m2` — shared
- `complexity` — shared
- `contractor_capability` — shared
- `resource_availability` — shared
- `management_maturity` — shared
- `weather_exposure` — shared
- `supply_chain_exposure` — shared
- `technology_maturity` — shared
- `planned_duration_days` — shared
- `planned_cost` — shared
- project-level mean planned activity duration — derived
- project-level median planned activity duration — derived
- project-level activity count — derived
- project-level dependency count — derived
- project-level FS/SS/FF counts — derived

These are safe because they are project-level planning attributes.

### 6.4 Resource features

From `resources.csv` and `resource_allocation.csv`:

- number of resources allocated to the activity — derived
- number of labour resources allocated — derived
- number of equipment resources allocated — derived
- number of material resources allocated — derived
- total planned allocation fraction — derived
- mean resource availability across allocated resources — derived
- mean resource capacity across allocated resources — derived
- resource category mix flags — derived categoricals

### 6.5 Procurement features — planned side only

From `procurement.csv`, joined at project level:

- `planned_lead_days` — raw, project-level
- number of procurement records for the project — derived
- planned lead-day statistics per project — derived

`actual_lead_days` and `procurement_delay_days` are excluded.

### 6.6 Categorical encoding plan

Proposed encodings:

- `project_type` — one-hot or target-agnostic encoding
- `phase` — one-hot or ordinal if a defensible order exists
- `resource_type` — one-hot
- `resource_category` — one-hot
- boolean flags for relationship types and resource mixes

### 6.7 Scaling and transformations

Proposed:

- numeric features: standard scaling or robust scaling as appropriate
- skewed numeric features: consider log or clip transformations during modeling,
  not during this design
- categorical features: encode before scaling

All transformations must be fit on train only and applied to validation/test.

---

## 7. Formulation Comparison

Three candidate formulations were considered.

### 7.1 Regression

Model the target directly:

```
y = target_event_delay_days   (0 .. 168)
```

Pros:

- Uses the full continuous target.
- Matches the natural target definition.
- Produces an expected delay estimate.

Cons:

- Highly skewed distribution.
- Small zero mass (5.64%) is diluted in regression loss.
- Regression may under-emphasize the zero vs non-zero distinction.

### 7.2 Binary classification

Model:

```
y_binary = 1 if target_event_delay_days > 0 else 0
```

Pros:

- Simple and interpretable.
- Useful if the business question is "will this activity experience any event-induced delay?"

Cons:

- Extremely imbalanced: 94.36% positive, 5.64% negative.
- Majority-class baseline would be 94.36% accurate, so accuracy is useless.
- The more interesting question is not "any delay?" but "how much delay?", since nearly all
  activities have some event-induced delay in this dataset.

### 7.3 Two-stage / hurdle model

Stage 1 — classification:

```
P(delay > 0 | features)
```

Stage 2 — regression on positive target only:

```
E[target | target > 0, features]
```

Then combine for an expected delay estimate.

Pros:

- Separates the zero vs non-zero question from severity.
- Can use classification metrics for the first stage and regression metrics for the second.
- Matches the structure of the target better than pure regression.

Cons:

- More complex.
- Stage 2 training set is smaller (8,756 vs 9,279).
- The zero class is small, so Stage 1 may still be dominated by the positive class.

### 7.4 Recommendation

**Recommended formulation: two-stage/hurdle model.**

Reasons:

1. The target is continuous and highly skewed, so pure regression on the raw target is
   defensible but does not explicitly handle the zero mass.
2. The binary "any delay?" question is nearly trivial in this dataset (94.36% positive), so a
   standalone classifier is less informative.
3. A two-stage model lets us:
   - explicitly model the small zero-delay class in Stage 1, and
   - model delay severity in Stage 2.

If a single-model approach is preferred for simplicity, **regression on the raw target** is the
next best choice, with the understanding that the zeros are a minority and must be represented
in training and evaluation.

**Do not use** standalone binary classification as the primary formulation just because
Logistic Regression was named in the original Phase 4 plan. The target distribution makes the
two-stage approach more defensible.

---

## 8. Dataset Split

Use the existing SCOPE project-level split:

| Split | Projects | Activities (approx) |
|-------|----------|---------------------|
| Train | 70 | 6,556 |
| Validation | 15 | 1,339 |
| Test | 15 | 1,384 |

These activity counts come from the CPM/cleaned data coverage by split and are consistent
with the project-level split.

### 8.1 Why project-level splitting is required

Activities within the same project share:

- project characteristics
- planned network structure
- resource environment
- procurement context
- dependencies

A random activity-level split would place activities from the same project into both training
and test. That leaks project-specific structure and produces unrealistically optimistic
evaluation.

Therefore:

- Train on 70 projects.
- Validate on 15 different projects.
- Test on 15 different projects.
- Never mix activities from the same project across splits.

### 8.2 Split usage

- **Train:** model fitting and internal model selection.
- **Validation:** hyperparameter tuning, early stopping, model comparison, calibration checks.
- **Test:** final evaluation only, used once at the end.

---

## 9. Baselines

Baselines must be evaluated before any ML model.

### 9.1 Regression baselines

- **Mean predictor:** predict the training-set mean `target_event_delay_days` for every
  activity.
- **Median predictor:** predict the training-set median.
- **Zero predictor:** predict 0 for every activity. Useful as a lower bound reference
  although it will be poor here because 94.36% of activities have positive target.
- **Plan-duration baseline:** predict a simple function of `planned_duration_days`, for
  example a fitted mean ratio or a naive scaling. This is a weak but interpretable baseline.

### 9.2 Classification baselines (if classification is used)

For Stage 1 (delay > 0):

- **Majority-class baseline:** predict positive for every activity. This gives 94.36% accuracy
  but 0% recall for the negative class, demonstrating why accuracy is not a useful metric here.

### 9.3 What baselines establish

- The ML model must beat the mean/median baseline on regression metrics.
- The ML model must beat the majority baseline on classification metrics in a meaningful way,
  especially on the minority class.
- Baselines provide a sanity check that the features carry more signal than naive constants.

---

## 10. Model Plan

No models are trained in this step. The following are proposed for later implementation.

### 10.1 Logistic Regression

Use:

- Stage 1 classification (delay > 0), if two-stage formulation is used.
- Optionally as a calibrated baseline classifier.

Considerations:

- Requires feature scaling.
- Requires categorical encoding before fitting.
- Linear decision boundary; may underfit complex interactions.
- Good interpretability baseline.

### 10.2 Random Forest

Use:

- Classification (Stage 1) and/or regression (Stage 2).
- Strong baseline for tabular data.

Considerations:

- Handles non-linear interactions.
- Less feature scaling sensitive.
- Can capture interactions between planned duration, network structure, and project context.
- Use fixed random seed for reproducibility.
- Prefer out-of-bag estimates or validation-based tuning.

### 10.3 Gradient Boosting

Use:

- Classification and/or regression.
- Often strongest tabular performer.

Considerations:

- Use early stopping on the validation split.
- Use a fixed random seed for reproducibility.
- Needs careful handling of class imbalance in the classification stage, for example class
  weights or appropriate sampling.
- Calibration may be needed if probability estimates are used downstream.

### 10.4 Recommended modeling strategy

1. Start with Logistic Regression and Random Forest as baselines.
2. Add Gradient Boosting as the primary candidate.
3. For the two-stage formulation:
   - Stage 1: classify delay > 0.
   - Stage 2: regress target on the positive subset.
4. Compare against the direct regression formulation as an alternative.

---

## 11. Evaluation Plan

Evaluation is defined per formulation.

### 11.1 Regression metrics

- **MAE** — primary regression metric if the goal is expected delay in days.
- **RMSE** — secondary, penalizes large errors more heavily.
- **Median absolute error** — robust to skew.
- **R-squared** — descriptive, not primary.

If regression is used on the raw target, MAE is the most interpretable primary metric because
it is in the same units as the target.

### 11.2 Classification metrics

For Stage 1 (delay > 0):

- **ROC-AUC** — primary ranking metric for binary classification.
- **PR-AUC** — important because the positive class dominates; precision/recall behavior
  matters more than ROC in imbalanced settings.
- **Precision** — useful for understanding false-positive cost.
- **Recall** — useful for understanding how many delayed activities are caught.
- **F1** — composite measure when a single operating point is needed.
- **Calibration** — important if predicted probabilities feed Phase 5 risk scoring.

Because 94.36% of activities are positive, accuracy is not a meaningful primary metric.

### 11.3 Two-stage evaluation

- Evaluate Stage 1 as classification.
- Evaluate Stage 2 as regression on the positive subset.
- Also evaluate the combined expected-delay estimate on the full test set using regression-style
  metrics where appropriate.

### 11.4 Primary metrics by formulation

- If two-stage: primary metrics are **PR-AUC** (Stage 1) and **MAE** (Stage 2).
- If direct regression: primary metric is **MAE**, with RMSE and calibration-style checks as
  secondary.

In all cases, the test-set evaluation must be project-separated.

---

## 12. CPM Integration

This document does **not** combine ML and CPM.

### 12.1 What ML provides

- `P(event-induced delay > 0 | prediction-time features)`
- and/or `E[target_event_delay_days | prediction-time features]`

These are predictive estimates based on planned and structural information only.

### 12.2 What CPM provides

- ES, EF, LS, LF
- Total float
- Computed critical activities and critical path
- Project completion duration

CPM is deterministic schedule structure from planned durations and dependencies. It does not
predict future delays.

### 12.3 Phase 5 relationship

Phase 5 will later combine:

- ML: predicted probability and/or expected event-induced delay
- CPM: float, criticality, downstream schedule impact

into a risk ranking.

This Phase 4 design deliberately stops before that combination.

---

## 13. Limitations and Assumptions

1. **Target semantics:** the target is event-induced/disruption delay days, **not** observed
   schedule slippage. The dataset has no reliable actual activity start/finish timestamps.

2. **Prediction-time availability:** only the 75 AVAILABLE columns are eligible by default.
   The 18 UNKNOWN columns are excluded until their availability is established. The 23
   OUTCOME columns are leakage and must not be used.

3. **No feature implementation yet:** this document proposes features but does not implement
   them. Any implemented feature must be traceable to an AVAILABLE source.

4. **Environment and activity-state data:** excluded by default because their exact
   prediction-time availability is not established.

5. **Procurement actuals:** excluded; only planned procurement fields are considered safe.

6. **Synthetic data:** SCOPE v0.2 is a synthetic public release. Results are methodological
   and must be validated with real data before any deployment claim.

7. **Not causal:** model outputs are predictive associations, not causal effects. They should
   not be interpreted as the causal impact of any planned attribute.

8. **Class imbalance:** the target zero class is small; evaluation metrics must reflect that.

9. **Split integrity:** all evaluation must honor the project-level split. No activity-level
   random split is acceptable.

---

## 14. Files Created or Modified

### Created

- `notebooks/04_ml_problem_design.md` — this design/audit document

### Not modified

- No Phase 1, Phase 2, or Phase 3 implementation files were modified.
- No raw data was modified.
- No ML pipeline, model, or feature matrix was created.
- `PROGRESS.md` was not modified in this step.

---

## 15. Validation Performed

- Existing repository test suite re-run.
- No new Python files were created, so no Ruff run on new Python code was needed.
- Git status and diff reviewed after document creation.

---

## 16. Current Status

Phase 4 Step 1 is a design-only artifact. It is ready for review.

**Next actions, if approved:**

- Implement prediction-time feature extraction from the AVAILABLE sources.
- Build the feature matrix using the project-level split.
- Train and compare the proposed baselines and models.
- Evaluate on the validation split and report final results on the test split.
- Decide whether the two-stage or direct regression formulation performs better in practice.
