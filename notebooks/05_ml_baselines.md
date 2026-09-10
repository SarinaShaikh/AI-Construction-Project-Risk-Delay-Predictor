# Phase 4 Step 3 — Modeling Pipeline and Baselines

**Date:** 2026-09-09
**Status:** Modeling pipeline implemented; broad hyperparameter tuning not started.

This document describes the leakage-safe modeling pipeline built in Step 3, the
preprocessing architecture, baseline models prepared, and evaluation structure.

---

## 1. Purpose

Step 3 converts the Step 2 feature matrix into train/validation/test matrices
suitable for ML modeling, with strict leakage prevention:

- Preprocessing is fit on TRAIN ONLY
- Validation and test are transformed using train-fitted preprocessing
- No validation/test information influences encoding or scaling

---

## 2. Pipeline Architecture

```
Step 2 feature matrix (55 features, 9279 activities)
    |
    v
load_step2_as_dataframes()
    |
    +--> SplitData train (6556 activities, 70 projects)
    +--> SplitData val (1339 activities, 15 projects)
    +--> SplitData test (1384 activities, 15 projects)
    |
    v
build_preprocessor()
    |--> ColumnTransformer
    |       +--> numeric pipeline: SimpleImputer(median) -> StandardScaler
    |       +--> categorical pipeline: OneHotEncoder(handle_unknown="ignore")
    |
    v
fit_preprocessor(preprocessor, X_train)  # FIT ON TRAIN ONLY
    |
    v
transform_splits()  # transform train/val/test using train-fitted preprocessor
    |
    +--> preprocessed_split.X_train
    +--> preprocessed_split.X_val
    +--> preprocessed_split.X_test
```

---

## 3. Preprocessing Details

### 3.1 Numeric features (52)

- Imputed with median (fit on train only)
- Scaled with StandardScaler (fit on train only)

### 3.2 Categorical features (3)

- `activity_phase` (10 categories)
- `activity_resource_type` (4 categories)
- `project_type` (5 categories)

Encoded with OneHotEncoder(handle_unknown="ignore").

**Critical:** the encoder is fit on training data only. Unseen categories in
validation or test are silently ignored (encoded as all zeros for that category).

### 3.3 Column consistency

All three splits (train/val/test) have identical transformed feature columns
after preprocessing, in deterministic order from `get_feature_names_out()`.

---

## 4. Target

The target is `target_event_delay_days`.

- Raw continuous target for regression
- Binary target `target_event_delay_days > 0` for classification
- Both targets are available through `SplitData.target_binary_train/val/test`

Target is **never** included in feature matrices.

---

## 5. Split Integrity

| Split | Projects | Activities |
|-------|----------|------------|
| Train | 70 | 6,556 |
| Validation | 15 | 1,339 |
| Test | 15 | 1,384 |

No project appears in multiple splits. No activity-level random splitting.

---

## 6. Baseline Models Prepared

### 6.1 Regression baselines

- `DummyRegressor(strategy="mean")`
- `DummyRegressor(strategy="median")`
- `LinearRegression()`
- `Ridge(random_state=42)`

### 6.2 Classification baselines

- `LogisticRegression(max_iter=1000, random_state=42)`
- `RandomForestClassifier(random_state=42, n_estimators=100)`
- `GradientBoostingClassifier(random_state=42, n_estimators=100)`

### 6.3 Two-stage hurdle models

Stage 1 — classify delay > 0:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Stage 2 — regress positive delay amount:
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The hurdle model is not yet evaluated in this step.

---

## 7. Evaluation Metrics

### 7.1 Regression

- MAE (primary)
- RMSE
- Median Absolute Error
- R²

### 7.2 Classification

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Brier score

### 7.3 Hurdle

- Regression metrics on final predicted delay days
- Classification metrics on stage 1 (delay occurrence)

---

## 8. Leakage Tests

Step 3 includes tests that verify:

- Preprocessor is fit on train only
- Scaler means match training data means
- Scaler parameters do NOT depend on validation data
- Encoder categories are subsets of training categories
- Unseen categories do not crash validation/test transformation
- Preprocessing is NOT fit on the full dataset
- Target is not in transformed features
- Critical_path / critical_path_position remain absent
- Event/rework/decision actual/procurement actual/outcome fields absent
- Project-level split remains disjoint
- Transformed columns identical across splits
- No missing or infinite values in transformed data
- Preprocessing is deterministic

---

## 9. Files

- `src/ml/modeling.py` — modeling pipeline, preprocessing, baselines, evaluation
- `tests/test_modeling.py` — leakage-safe modeling tests
- `notebooks/05_ml_baselines.md` — this document
- `notebooks/04_feature_engineering.md` — corrected (55 features)

---

## 10. Next Steps

- Run full model comparison
- Hyperparameter tuning for best candidates
- Final evaluation on test set
- Consideration of two-stage hurdle vs direct regression
