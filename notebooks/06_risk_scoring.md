# Phase 5 — Risk Scoring

**Generated:** 2026-09-11

## 1. Objective

This document describes the Phase 5 risk scoring engine, which combines:

1. **Phase 4 ML delay predictions** — LogisticRegression probability and LinearRegression magnitude
2. **Phase 3 CPM results** — total float, computed criticality, ES/EF/LS/LF
3. **Heuristic components** — criticality weight and float impact

to produce activity-level risk scores.

## 2. Risk Components

The risk score is a product of three components:

```
risk_score = probability_of_event_delay * criticality_weight * float_impact
```

### 2.1 Probability of Event Delay (TRUE probability)

**Source:** LogisticRegression `predict_proba(X)[:, 1]`

This is the **TRUE probability** that an activity will experience event-induced delay.
It comes directly from the trained LogisticRegression classification model.

- Range: [0, 1]
- Interpretation: P(event-induced delay > 0 | activity features)

This is NOT a normalized score or heuristic. It is the actual model-predicted probability.

### 2.2 Predicted Delay Days (predicted magnitude)

**Source:** LinearRegression `predict(X)`

This is the predicted magnitude of event-induced delay in days.

- Range: [0, ∞) (negative predictions clipped to 0)
- Interpretation: Expected event-induced delay days for this activity

### 2.3 Criticality Weight (HEURISTIC)

**Source:** Binned total float from CPM results

This is a **HEURISTIC**, not a learned or statistical quantity.

| Total Float | Criticality Weight |
|-------------|-------------------|
| ≤ 0 days    | 1.0               |
| < 5 days    | 0.9               |
| < 20 days   | 0.7               |
| < 50 days   | 0.4               |
| ≥ 50 days   | 0.1               |

The intuition: activities with less float are more critical to the schedule.

### 2.4 Float Impact (HEURISTIC)

**Source:** Predicted delay days / available float

This is a **HEURISTIC**, not a learned or statistical quantity.

Formula:
```
float_impact = min(predicted_delay / max(total_float, 1), 1)
```

For zero/near-zero float:
- Positive delay → 1.0
- Zero delay → 0.0

The intuition: an activity whose predicted delay exceeds its available float has
high float impact.

### 2.5 Risk Score

```
risk_score = probability_of_event_delay * criticality_weight * float_impact
```

Range: [0, 1]

Higher scores indicate higher risk (more likely to be delayed AND more critical AND
more likely to consume float).

## 3. Selected Models

- **Classification (probability):** LogisticRegression
- **Regression (magnitude):** LinearRegression

These are the models selected in Phase 4 Step 4 based on validation-only evaluation.

## 4. Data Sources

- **CPM results:** `data/processed/cpm/activity_cpm_results.csv`
- **ML predictions:** Generated via Phase 4 modeling pipeline (re-fit on training data)
- **Activities metadata:** `data/raw/activities.csv` (for activity names, optional)

## 5. Validation

The risk engine performs the following validation:

1. **Duplicate (project_id, activity_id) keys** in CPM results → error
2. **Duplicate predictions** → error
3. **Missing prediction columns** → error
4. **Probability outside [0, 1]** → error
5. **Negative predicted delays** → error
6. **Missing ML predictions for CPM activities** → error
7. **Required CPM columns missing** → error

All matching uses `(project_id, activity_id)` with `validate="one_to_one"`.

## 6. Important Notes

### 6.1 Probability is a TRUE probability

`probability_of_event_delay` comes from `LogisticRegression.predict_proba()`.
It is NOT a normalized score or heuristic. It represents P(event-induced delay > 0).

### 6.2 Criticality weight and float impact are HEURISTICS

These components are **documented as heuristics**. They are reasonable engineering
judgments, but they are NOT learned from data and NOT statistical quantities.

### 6.3 Target definition

The target is `target_event_delay_days` = **event-induced/disruption delay days**.
This is NOT actual schedule slippage. The dataset does not provide reliable actual
activity start/finish timestamps.

### 6.4 Test set

The test set remains locked. No test-set information is used for risk score design,
tuning, or model selection.

## 7. Limitations

- The heuristics (criticality weight, float impact) are engineering judgments, not
  empirically validated formulas.
- The target is event-induced delay days, not actual schedule impact.
- No causal claims are made from these risk scores.
- The risk score is a relative ranking tool, not an absolute probability of schedule failure.

## 8. Output

The risk scores are saved to `data/processed/risk_scores.csv` with columns:

| Column | Description |
|--------|-------------|
| project_id | Project identifier |
| activity_id | Activity identifier |
| activity_name | Activity name (if available) |
| activity_type | Activity type (if available) |
| phase | Phase (if available) |
| resource_type | Resource type (if available) |
| predicted_delay_days | LinearRegression prediction |
| probability_of_event_delay | LogisticRegression probability |
| total_float | CPM total float |
| computed_critical | CPM computed criticality |
| criticality_weight | Heuristic (binned total float) |
| float_impact | Heuristic (predicted delay / float) |
| risk_score | Final risk score |
| ES, EF, LS, LF | CPM schedule values |
| risk_rank | Rank by risk score (1 = highest) |
| risk_percentile | Percentile (0-100) |

## 9. Interpretation Guide

- **High risk_score (>0.5):** Activities with high probability of delay, high criticality,
  and high float impact. These are the most concerning activities.
- **Medium risk_score (0.1-0.5):** Activities with moderate risk across one or more dimensions.
- **Low risk_score (<0.1):** Activities with low probability, low criticality, or low float impact.

**Remember:** These are relative rankings, not absolute probabilities of schedule failure.
