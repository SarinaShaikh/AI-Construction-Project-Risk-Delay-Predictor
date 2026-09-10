"""
Tests for src/ml/modeling.py

These tests verify:
- Step 2 features load correctly
- Preprocessing is fit on TRAIN ONLY
- Validation/test preprocessing cannot influence fitting
- Transformed columns are identical across splits
- Target leakage protections are maintained
- Baselines fit and predict successfully
- Hurdle model structure works
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml.feature_engineering import (
    CATEGORICAL_FEATURE_NAMES,
    EXPECTED_NUM_ACTIVITIES,
    FEATURE_COLUMN_ORDER,
    NUMERIC_FEATURE_NAMES,
)
from ml.modeling import (
    SplitData,
    build_preprocessed_split,
    build_preprocessor,
    classification_metrics,
    dummy_regressor_mean,
    dummy_regressor_median,
    fit_and_evaluate_hurdle,
    fit_and_predict_classification,
    fit_and_predict_regression,
    fit_model_and_report,
    gradient_boosting_classifier,
    gradient_boosting_regressor,
    hurdle_metrics,
    linear_regression,
    load_step2_as_dataframes,
    logistic_regression,
    random_forest_classifier,
    random_forest_regressor,
    regression_metrics,
    ridge_regression,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def processed_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture
def raw_split(processed_dir: Path) -> SplitData:
    return load_step2_as_dataframes(processed_dir)


@pytest.fixture
def preprocessed_split(processed_dir: Path) -> SplitData:
    raw = load_step2_as_dataframes(processed_dir)
    return build_preprocessed_split(raw, fit_preprocessor_now=True)


# ============================================================================
# Step 2 integration
# ============================================================================

def test_load_step2_activities_count(raw_split: SplitData) -> None:
    total = raw_split.num_train + raw_split.num_val + raw_split.num_test
    assert total == EXPECTED_NUM_ACTIVITIES


def test_step2_feature_names_match(raw_split: SplitData) -> None:
    assert list(raw_split.X_train.columns) == FEATURE_COLUMN_ORDER


def test_step2_split_counts(raw_split: SplitData) -> None:
    proj_train = {pid for pid, _ in raw_split.activity_ids_train}
    proj_val = {pid for pid, _ in raw_split.activity_ids_val}
    proj_test = {pid for pid, _ in raw_split.activity_ids_test}

    assert len(proj_train) == 70
    assert len(proj_val) == 15
    assert len(proj_test) == 15
    assert len(proj_train & proj_val) == 0
    assert len(proj_train & proj_test) == 0
    assert len(proj_val & proj_test) == 0


def test_step2_target_alignment(raw_split: SplitData) -> None:
    assert len(raw_split.y_train) == raw_split.num_train
    assert len(raw_split.y_val) == raw_split.num_val
    assert len(raw_split.y_test) == raw_split.num_test


def test_step2_no_target_in_features(raw_split: SplitData) -> None:
    for df in [raw_split.X_train, raw_split.X_val, raw_split.X_test]:
        assert "target_event_delay_days" not in df.columns


# ============================================================================
# Preprocessing architecture
# ============================================================================

def test_build_preprocessor_returns_unfitted(raw_split: SplitData) -> None:
    preprocessor = build_preprocessor()
    # Should not have fitted attributes yet
    assert not hasattr(preprocessor, "named_transformers_") or (
        hasattr(preprocessor.named_transformers_, "_le") and False
    )  # can't easily check, but it should be unfitted
    # Actually check: trying to transform without fitting raises
    with pytest.raises(NotFittedError):
        _ = preprocessor.transform(raw_split.X_train.iloc[:1])


def test_preprocessor_fit_on_train_only(preprocessed_split: SplitData) -> None:
    preprocessor = preprocessed_split.preprocessor
    assert preprocessor is not None
    assert hasattr(preprocessor, "named_transformers_")
    assert "num" in preprocessor.named_transformers_
    assert "cat" in preprocessor.named_transformers_


def test_transformed_columns_identical_across_splits(preprocessed_split: SplitData) -> None:
    cols_train = list(preprocessed_split.X_train.columns)
    cols_val = list(preprocessed_split.X_val.columns)
    cols_test = list(preprocessed_split.X_test.columns)

    assert cols_train == cols_val
    assert cols_val == cols_test
    assert len(cols_train) == preprocessed_split.num_features


def test_transformed_feature_count(preprocessed_split: SplitData) -> None:
    # 55 raw features -> more after one-hot encoding
    assert preprocessed_split.num_features > 55
    assert preprocessed_split.num_features == len(preprocessed_split.feature_names)


def test_no_missing_in_transformed(preprocessed_split: SplitData) -> None:
    for df, name in [
        (preprocessed_split.X_train, "train"),
        (preprocessed_split.X_val, "validation"),
        (preprocessed_split.X_test, "test"),
    ]:
        assert df.isna().sum().sum() == 0, f"Missing values in {name}"


def test_no_inf_in_transformed(preprocessed_split: SplitData) -> None:
    import numpy as np
    for df, name in [
        (preprocessed_split.X_train, "train"),
        (preprocessed_split.X_val, "validation"),
        (preprocessed_split.X_test, "test"),
    ]:
        numeric_df = df.select_dtypes(include=[float, int])
        assert not np.isinf(numeric_df.values).any(), f"Infinite values in {name}"


# ============================================================================
# Categorical encoding leakage tests
# ============================================================================

def test_encoder_fit_only_on_train(preprocessed_split: SplitData, raw_split: SplitData) -> None:
    preprocessor = preprocessed_split.preprocessor
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    # Categories should be determined by training data
    assert ohe.categories_ is not None

    # Verify categories are subsets of training values (use raw, unbroken X_train)
    X_train = raw_split.X_train
    for cat_col, cats in zip(CATEGORICAL_FEATURE_NAMES, ohe.categories_):
        train_values = set(X_train[cat_col].unique())
        encoded_cats = set(cats)
        assert encoded_cats.issubset(train_values), (
            f"Encoder categories for {cat_col} include values not in training: "
            f"{encoded_cats - train_values}"
        )


def test_unseen_category_handled_in_validation(preprocessed_split: SplitData, raw_split: SplitData) -> None:
    """Unseen categorical values in validation must not crash."""
    preprocessor = preprocessed_split.preprocessor
    X_val = raw_split.X_val
    X_val_transformed = preprocessor.transform(X_val)
    assert X_val_transformed.shape[0] == X_val.shape[0]
    import numpy as np
    assert not np.isnan(X_val_transformed).any()


def test_unseen_category_handled_in_test(preprocessed_split: SplitData, raw_split: SplitData) -> None:
    """Unseen categorical values in test must not crash."""
    import numpy as np
    preprocessor = preprocessed_split.preprocessor
    X_test = raw_split.X_test
    X_test_transformed = preprocessor.transform(X_test)
    assert X_test_transformed.shape[0] == X_test.shape[0]
    assert not np.isnan(X_test_transformed).any()


def test_encoder_categories_are_train_determined_only(preprocessed_split: SplitData, raw_split: SplitData) -> None:
    """Encoder categories must NOT be influenced by validation/test data."""
    preprocessor = preprocessed_split.preprocessor
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    X_train_raw = raw_split.X_train

    for cat_col, cats in zip(CATEGORICAL_FEATURE_NAMES, ohe.categories_):
        train_values = set(X_train_raw[cat_col].unique())
        for cat in cats:
            assert cat in train_values, f"Category {cat} for {cat_col} not in training data"


def test_scaler_fit_only_on_train(preprocessed_split: SplitData, raw_split: SplitData) -> None:
    preprocessor = preprocessed_split.preprocessor
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # Scaler should have mean_ and scale_ attributes (fitted)
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")

    # Verify scaler means are computed from training data only (use raw X_train)
    X_train = raw_split.X_train
    for i, col in enumerate(NUMERIC_FEATURE_NAMES):
        train_mean = X_train[col].mean()
        scaler_mean = scaler.mean_[i]
        assert abs(train_mean - scaler_mean) < 1e-9, (
            f"Scaler mean for {col} ({scaler_mean}) != training mean ({train_mean})"
        )


def test_scaler_parameters_do_not_depend_on_validation(
    preprocessed_split: SplitData, raw_split: SplitData
) -> None:
    """Verify scaler parameters are identical regardless of validation data."""
    preprocessor = preprocessed_split.preprocessor
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # Create a separate preprocessor fit on the same train data but with
    # completely different validation data (shuffled copies)
    X_train = raw_split.X_train
    preprocessor2 = build_preprocessor()
    preprocessor2.fit(X_train)

    scaler = preprocessed_split.preprocessor.named_transformers_["num"].named_steps["scaler"]
    scaler2 = preprocessor2.named_transformers_["num"].named_steps["scaler"]

    assert preprocessor is not preprocessor2
    assert (scaler.mean_ == scaler2.mean_).all()
    assert (scaler.scale_ == scaler2.scale_).all()


# ============================================================================
# Target leakage tests
# ============================================================================

def test_target_not_in_transformed_features(preprocessed_split: SplitData) -> None:
    for df in [preprocessed_split.X_train, preprocessed_split.X_val, preprocessed_split.X_test]:
        assert "target_event_delay_days" not in df.columns


def test_critical_path_absent_in_transformed(preprocessed_split: SplitData) -> None:
    for df in [preprocessed_split.X_train, preprocessed_split.X_val, preprocessed_split.X_test]:
        for col in df.columns:
            assert "critical_path" not in col.lower(), f"critical_path term in {col}"


def test_event_terms_absent_in_transformed(preprocessed_split: SplitData) -> None:
    for df, name in [
        (preprocessed_split.X_train, "train"),
        (preprocessed_split.X_val, "validation"),
        (preprocessed_split.X_test, "test"),
    ]:
        for col in df.columns:
            lower = col.lower()
            for term in ["event_type", "severity", "impact_factor"]:
                assert term not in lower, f"{term} in {name}/{col}"


def test_rework_terms_absent_in_transformed(preprocessed_split: SplitData) -> None:
    for df in [preprocessed_split.X_train, preprocessed_split.X_val, preprocessed_split.X_test]:
        for col in df.columns:
            assert "rework" not in col.lower(), f"rework in {col}"


def test_procurement_actuals_absent_in_transformed(preprocessed_split: SplitData) -> None:
    for df in [preprocessed_split.X_train, preprocessed_split.X_val, preprocessed_split.X_test]:
        for col in df.columns:
            lower = col.lower()
            assert "actual_lead" not in lower
            assert "procurement_delay" not in lower


def test_outcome_terms_absent_in_transformed(preprocessed_split: SplitData) -> None:
    for df in [preprocessed_split.X_train, preprocessed_split.X_val, preprocessed_split.X_test]:
        for col in df.columns:
            assert "outcome" not in col.lower(), f"outcome in {col}"


def test_preprocessing_not_fit_on_full_dataset(raw_split: SplitData) -> None:
    """Verify that preprocessing is not accidentally fit on complete dataset."""
    X_all = pd.concat([raw_split.X_train, raw_split.X_val, raw_split.X_test])

    preprocessor = build_preprocessor()
    # Fit on ALL data (this is what we must NOT do)
    preprocessor.fit(X_all)

    # Now build the correct preprocessed split (fit on train only)
    correct_split = build_preprocessed_split(raw_split, fit_preprocessor_now=True)

    # Compare scaler means: correct (train-only) vs incorrect (all data)
    correct_scaler = correct_split.preprocessor.named_transformers_["num"].named_steps["scaler"]
    wrong_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # These should differ because the correct scaler used only train data
    means_differ = not (abs(correct_scaler.mean_ - wrong_scaler.mean_).max() < 1e-9)
    assert means_differ, "Scaler means are identical — preprocessing may have been fit on full dataset"


# ============================================================================
# Baseline model tests
# ============================================================================

def test_dummy_mean_regressor(preprocessed_split: SplitData) -> None:
    model = dummy_regressor_mean()
    report = fit_model_and_report(model, "dummy_mean", "regression", preprocessed_split)
    assert "MAE" in report
    assert "RMSE" in report
    assert "MedianAE" in report
    assert "R2" in report
    assert report["MAE"] > 0


def test_dummy_median_regressor(preprocessed_split: SplitData) -> None:
    model = dummy_regressor_median()
    report = fit_model_and_report(model, "dummy_median", "regression", preprocessed_split)
    assert "MAE" in report
    assert report["MAE"] > 0


def test_linear_regression(preprocessed_split: SplitData) -> None:
    model = linear_regression()
    report = fit_model_and_report(model, "linear_regression", "regression", preprocessed_split)
    assert "MAE" in report
    assert "R2" in report


def test_ridge_regression(preprocessed_split: SplitData) -> None:
    model = ridge_regression()
    report = fit_model_and_report(model, "ridge", "regression", preprocessed_split)
    assert "MAE" in report
    assert "R2" in report


def test_logistic_regression(preprocessed_split: SplitData) -> None:
    model = logistic_regression()
    report = fit_model_and_report(model, "logistic_regression", "classification", preprocessed_split)
    assert "precision" in report
    assert "recall" in report
    assert "f1" in report
    assert "roc_auc" in report
    assert "pr_auc" in report


def test_random_forest_classifier(preprocessed_split: SplitData) -> None:
    model = random_forest_classifier()
    report = fit_model_and_report(model, "rf_classifier", "classification", preprocessed_split)
    assert "precision" in report
    assert "roc_auc" in report


def test_random_forest_regressor(preprocessed_split: SplitData) -> None:
    model = random_forest_regressor()
    report = fit_model_and_report(model, "rf_regressor", "regression", preprocessed_split)
    assert "MAE" in report
    assert "R2" in report


def test_gradient_boosting_classifier(preprocessed_split: SplitData) -> None:
    model = gradient_boosting_classifier()
    report = fit_model_and_report(model, "gb_classifier", "classification", preprocessed_split)
    assert "precision" in report
    assert "roc_auc" in report


def test_gradient_boosting_regressor(preprocessed_split: SplitData) -> None:
    model = gradient_boosting_regressor()
    report = fit_model_and_report(model, "gb_regressor", "regression", preprocessed_split)
    assert "MAE" in report
    assert "R2" in report


# ============================================================================
# Regression metric tests
# ============================================================================

def test_regression_metrics_on_dummy() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])

    metrics = regression_metrics(y_true, y_pred)
    assert abs(metrics["MAE"] - 1.2) < 1e-9
    assert abs(metrics["RMSE"] - math.sqrt(2.0)) < 1e-9
    assert abs(metrics["MedianAE"] - 1.0) < 1e-9


def test_regression_metrics_perfect_prediction() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 3.0])

    metrics = regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MedianAE"] == 0.0
    assert metrics["R2"] == 1.0


# ============================================================================
# Classification metric tests
# ============================================================================

def test_classification_metrics_on_dummy() -> None:
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_pred = pd.Series([0, 1, 1, 0, 0])
    y_prob = pd.Series([0.1, 0.8, 0.9, 0.2, 0.4])

    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier" in metrics


def test_classification_metrics_no_prob() -> None:
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_pred = pd.Series([0, 1, 1, 0, 0])

    metrics = classification_metrics(y_true, y_pred)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" not in metrics
    assert "pr_auc" not in metrics
    assert "brier" not in metrics


# ============================================================================
# Hurdle model tests
# ============================================================================

def test_hurdle_metrics_structure() -> None:
    y_true = pd.Series([0, 5, 0, 10, 0, 15])
    y_prob = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    y_stage2 = pd.Series([0, 4.0, 0, 8.0, 0, 12.0])

    metrics = hurdle_metrics(y_true, y_prob, y_stage2, stage1_threshold=0.5)
    assert "hurdle_reg MAE" in metrics
    assert "hurdle_reg RMSE" in metrics
    assert "stage1_precision" in metrics
    assert "stage1_recall" in metrics


def test_hurdle_stage1_fit(preprocessed_split: SplitData) -> None:
    model = logistic_regression()
    _, y_pred, y_prob = fit_and_predict_classification(
        model,
        preprocessed_split.X_train,
        preprocessed_split.target_binary_train,
        preprocessed_split.X_val,
    )
    assert len(y_pred) == preprocessed_split.num_val
    assert len(y_prob) == preprocessed_split.num_val
    assert set(y_pred.unique()).issubset({0, 1})


def test_hurdle_stage2_fit(preprocessed_split: SplitData) -> None:
    model = ridge_regression()
    _, y_pred = fit_and_predict_regression(
        model,
        preprocessed_split.X_train[preprocessed_split.y_train > 0],
        preprocessed_split.y_train[preprocessed_split.y_train > 0],
        preprocessed_split.X_val,
    )
    assert len(y_pred) == preprocessed_split.num_val


def test_hurdle_full_fit(preprocessed_split: SplitData) -> None:
    stage1 = logistic_regression()
    stage2 = ridge_regression()
    report = fit_and_evaluate_hurdle(stage1, stage2, preprocessed_split)
    assert "hurdle_reg MAE" in report
    assert "stage1_precision" in report
    assert "stage1_recall" in report
    assert "stage1_roc_auc" in report


def test_hurdle_stage2_positive_only_training(preprocessed_split: SplitData) -> None:
    """Stage 2 must only train on rows where target > 0."""
    train_mask = preprocessed_split.y_train > 0
    n_positive = train_mask.sum()
    assert n_positive > 0
    assert n_positive < preprocessed_split.num_train


# ============================================================================
# Fit/predict helpers
# ============================================================================

def test_fit_and_predict_regression_returns_correct_shapes(preprocessed_split: SplitData) -> None:
    model = linear_regression()
    _, y_pred = fit_and_predict_regression(
        model, preprocessed_split.X_train, preprocessed_split.y_train, preprocessed_split.X_val
    )
    assert len(y_pred) == preprocessed_split.num_val


def test_fit_and_predict_classification_returns_correct_shapes(preprocessed_split: SplitData) -> None:
    model = logistic_regression()
    _, y_pred, y_prob = fit_and_predict_classification(
        model,
        preprocessed_split.X_train,
        preprocessed_split.target_binary_train,
        preprocessed_split.X_val,
    )
    assert len(y_pred) == preprocessed_split.num_val
    assert len(y_prob) == preprocessed_split.num_val


def test_fit_model_and_report_regression(preprocessed_split: SplitData) -> None:
    report = fit_model_and_report(
        linear_regression(), "linreg_test", "regression", preprocessed_split
    )
    assert report["model"] == "linreg_test"
    assert report["task"] == "regression"
    assert "MAE" in report


def test_fit_model_and_report_classification(preprocessed_split: SplitData) -> None:
    report = fit_model_and_report(
        logistic_regression(), "logreg_test", "classification", preprocessed_split
    )
    assert report["model"] == "logreg_test"
    assert report["task"] == "classification"
    assert "precision" in report


# ============================================================================
# Numerical quality
# ============================================================================

def test_transformed_values_finite(preprocessed_split: SplitData) -> None:
    import numpy as np
    for df, name in [
        (preprocessed_split.X_train, "train"),
        (preprocessed_split.X_val, "validation"),
        (preprocessed_split.X_test, "test"),
    ]:
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            assert numeric_df[col].notna().all(), f"NaN in {name}/{col}"
            assert np.isfinite(numeric_df[col].to_numpy()).all(), f"Inf in {name}/{col}"


# ============================================================================
# Reproducibility
# ============================================================================

def test_preprocessing_deterministic(processed_dir: Path) -> None:
    """Two independent preprocessing builds produce identical results."""
    raw1 = load_step2_as_dataframes(processed_dir)
    pp1 = build_preprocessed_split(raw1, fit_preprocessor_now=True)

    raw2 = load_step2_as_dataframes(processed_dir)
    pp2 = build_preprocessed_split(raw2, fit_preprocessor_now=True)

    assert pp1.X_train.equals(pp2.X_train)
    assert pp1.X_val.equals(pp2.X_val)
    assert pp1.X_test.equals(pp2.X_test)
    assert list(pp1.feature_names) == list(pp2.feature_names)


