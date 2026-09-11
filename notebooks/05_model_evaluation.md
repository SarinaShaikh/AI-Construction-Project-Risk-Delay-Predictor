# Phase 4 Step 4 — Validation-Only Model Evaluation and Selection

**Generated:** 2026-09-11T06:39:39.741300

## 1. Objective

This report evaluates the Phase 4 Step 3 baseline models on the **validation** split using **train-only** preprocessing and fitting. Model selection is performed using validation metrics only. The test set remains locked and is not used for model selection, hyperparameter tuning, threshold selection, or calibration fitting.

Target: `target_event_delay_days` = event-induced/disruption delay days.

This is NOT actual schedule slippage. The dataset does not provide reliable actual activity start/finish timestamps.

## 2. Dataset and Project-Level Split

- Activities: 9279
- Train projects/activities: 70 / 6556
- Validation projects/activities: 15 / 1339
- Test projects/activities: 15 / 1384
- Features: 71

## 3. Test-Set Lock Policy

The test set is locked for this step. No test-set information influenced preprocessing fitting, model fitting, model selection, threshold selection, or calibration fitting.

## 4. Regression Baseline Comparison (Validation)

Primary metric: MAE. Selected model: LinearRegression

| Model | MAE | RMSE | MedianAE | R² | Train MAE | Train R² |
|-------|-----|------|----------|----|-----------|----------|
| DummyRegressor(median) | 11.2061 | 15.5641 | 9.0000 | -0.0454 | 11.4582 | -0.0702 |
| LinearRegression | 11.3157 | 14.8412 | 9.6268 | 0.0494 | 11.6774 | 0.0377 |
| Ridge | 11.3307 | 14.8428 | 9.7119 | 0.0492 | 11.6771 | 0.0376 |
| GradientBoostingRegressor | 11.7460 | 15.1827 | 10.3797 | 0.0052 | 11.1547 | 0.1460 |
| DummyRegressor(mean) | 11.9133 | 15.2488 | 11.1464 | -0.0035 | 11.9029 | 0.0000 |
| RandomForestRegressor | 12.1461 | 15.3813 | 10.9900 | -0.0210 | 4.5531 | 0.8555 |

### Selected regression model

- Selected: LinearRegression
- Validation MAE: 11.3157
- Validation RMSE: 14.8412
- Validation MedianAE: 9.6268
- Validation R²: 0.0494
- Train MAE: 11.6774
- Train R²: 0.0377
- Dummy mean MAE: 11.9133

### Regression interpretation

- LinearRegression is the best non-dummy trained direct regression model by validation MAE on this target.
- However, DummyRegressor(median) achieves a lower validation MAE (11.2061) than LinearRegression (11.3157) — and also lower than every other trained regression model here.
- Therefore the available prediction-time features show limited predictive signal for event-induced delay magnitude at the activity level. Do not claim strong regression predictive performance from these results.

## 5. Classification Baseline Comparison (Validation)

Primary metric: PR-AUC. Selected model: LogisticRegression

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |
|-------|--------|---------|-----------|--------|-----|-------|
| RandomForestClassifier | 0.9485 | 0.5179 | 0.9440 | 1.0000 | 0.9712 | 0.0553 |
| GradientBoostingClassifier | 0.9538 | 0.5398 | 0.9439 | 0.9976 | 0.9700 | 0.0564 |
| LogisticRegression | 0.9612 | 0.6171 | 0.9440 | 1.0000 | 0.9712 | 0.0529 |

### Selected classification model

- Selected: LogisticRegression
- Validation PR-AUC: 0.9612
- Validation ROC-AUC: 0.6171
- Validation Precision: 0.9440
- Validation Recall: 1.0000
- Validation F1: 0.9712
- Validation Brier: 0.0529

### Classification interpretation

- The target is highly imbalanced: 8,756 of 9,279 activities (94.36%) have positive event-induced delay.
- High PR-AUC, F1, and recall must therefore be interpreted in the context of that near-ubiquitous positive class — most activities already have event-induced delay, so predicting the majority class is easy.
- ROC-AUC of 0.6171 indicates only modest discrimination beyond the majority-class baseline. The classification results do not imply strong predictive discrimination of which activities will be delayed.

## 6. Hurdle Model Comparison (Validation)

Selected model: hurdle_LogisticRegression_stage2_Ridge

| Hurdle Model | Reg MAE | Reg RMSE | Reg MedianAE | Reg R² | Stage1 PR-AUC |
|--------------|--------|---------|-------------|--------|---------------|
| hurdle_LogisticRegression_stage2_Ridge | 11.6281 | 14.8951 | 10.4203 | 0.0425 | 0.9612 |
| hurdle_RandomForestClassifier_stage2_Ridge | 11.6281 | 14.8951 | 10.4203 | 0.0425 | 0.9485 |
| hurdle_LogisticRegression_stage2_RandomForestRegressor | 12.4775 | 15.5207 | 11.5100 | -0.0396 | 0.9612 |
| hurdle_RandomForestClassifier_stage2_RandomForestRegressor | 12.4775 | 15.5207 | 11.5100 | -0.0396 | 0.9485 |

### Selected hurdle model

- Selected: hurdle_LogisticRegression_stage2_Ridge
- Validation combined MAE: 11.6281
- Validation combined RMSE: 14.8951
- Validation combined MedianAE: 10.4203
- Validation combined R²: 0.0425
- Stage 1 model: LogisticRegression(max_iter=1000, random_state=42)
- Stage 2 model: Ridge(random_state=42)

### Hurdle interpretation

- The selected hurdle model (LogisticRegression + Ridge) has combined validation MAE = 11.6281.
- This beats the DummyRegressor(mean) baseline (MAE = 11.9133), but it does **not** beat the DummyRegressor(median) baseline (MAE = 11.2061) and it does **not** beat the best trained direct regression model (LinearRegression, MAE = 11.3157).
- The hurdle approach is therefore selected only as the best two-stage combination among the alternatives tested; it is not a strong model and does not dominate the direct regression result.

## 7. Calibration Analysis

Calibration is assessed using cross-validation on the **training set only**, with validation used only for evaluation reporting. No test data is used.

Model: LogisticRegression (calibrated with sigmoid method, 5-fold CV on train).

**Training set**
- Observed positive-delay frequency: 0.9449
- Mean predicted positive probability: 0.9392
- Brier score: 0.0519
- PR-AUC: 0.9571
- ROC-AUC: 0.6089

**Validation set**
- Observed positive-delay frequency: 0.9440
- Mean predicted positive probability: 0.9375
- Brier score: 0.0527
- PR-AUC: 0.9580
- ROC-AUC: 0.6106

## 8. Selection Reasoning

- **metric_priority_regression**: validation MAE, then RMSE/MedianAE, then R2 and train/val gap
- **metric_priority_classification**: validation PR-AUC, then ROC-AUC, Brier, recall/precision
- **regression_selected_name**: LinearRegression
- **classification_selected_name**: LogisticRegression
- **hurdle_selected_name**: hurdle_LogisticRegression_stage2_Ridge
- **hurdle_vs_regression_note**: Hurdle combined MAE is compared against direct regression MAE to assess whether separating occurrence from magnitude helps on this target distribution.
- **calibration_note**: Calibration is assessed on train/validation only. Probabilities may be used as risk signals later; calibration quality is therefore documented here.

## 9. Limitations

- Results are validation-only. Test-set performance is not reported here because the test set is reserved for final unbiased evaluation after model selection.
- The target is event-induced/disruption delay days, not actual schedule slippage.
- No causal claims are made from these predictive results.
