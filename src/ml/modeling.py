"""
Phase 4 Step 3 — leakage-safe modeling pipeline.

Converts the Step 2 feature matrix into train/validation/test matrices suitable
for ML without allowing validation/test information to influence preprocessing.

Preprocessing:
- Categorical encoding fit on TRAIN ONLY
- Numeric scaling fit on TRAIN ONLY
- Validation and test transformed using train-fitted preprocessing only

Do NOT start Phase 5.
Do NOT train broad hyperparameter searches here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import build_feature_matrix

# ============================================================================
# Availability constants
# ============================================================================

NUMERIC_FEATURE_NAMES = [
    "activity_planned_duration_days",
    "activity_quantity",
    "activity_unit_cost",
    "activity_planned_cost",
    "activity_phase_order",
    "activity_sequence",
    "activity_predecessor_count",
    "activity_successor_count",
    "net_incoming_fs_count",
    "net_incoming_ss_count",
    "net_incoming_ff_count",
    "net_outgoing_fs_count",
    "net_outgoing_ss_count",
    "net_outgoing_ff_count",
    "net_incoming_total_lag",
    "net_outgoing_total_lag",
    "net_incoming_max_lag",
    "net_incoming_min_lag",
    "net_has_incoming_ss",
    "net_has_incoming_ff",
    "net_has_outgoing_ss",
    "net_has_outgoing_ff",
    "project_floors",
    "project_area_m2",
    "project_complexity",
    "project_contractor_capability",
    "project_resource_availability",
    "project_management_maturity",
    "project_weather_exposure",
    "project_supply_chain_exposure",
    "project_technology_maturity",
    "project_planned_duration_days",
    "project_planned_cost",
    "project_activity_count",
    "project_dependency_count",
    "project_mean_planned_activity_duration",
    "project_median_planned_activity_duration",
    "project_fs_count",
    "project_ss_count",
    "project_ff_count",
    "resource_allocated_count",
    "resource_labour_count",
    "resource_equipment_count",
    "resource_material_count",
    "resource_total_allocation_fraction",
    "resource_mean_availability",
    "resource_mean_capacity",
    "procurement_planned_lead_days",
    "procurement_record_count",
    "procurement_mean_planned_lead_days",
    "procurement_max_planned_lead_days",
    "procurement_supply_chain_risk",
]

CATEGORICAL_FEATURE_NAMES = [
    "activity_phase",
    "activity_resource_type",
    "project_type",
]

ALL_FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

TARGET_COLUMN = "target_event_delay_days"

# Expected counts (derived from data, not hard-coded for correctness)
EXPECTED_NUM_ACTIVITIES = 9279
EXPECTED_NUM_FEATURES = 55


# ============================================================================
# Data loading and split helpers
# ============================================================================

class SplitData:
    """Holds train/validation/test matrices with aligned metadata."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        activity_ids_train: list[tuple[str, str]],
        activity_ids_val: list[tuple[str, str]],
        activity_ids_test: list[tuple[str, str]],
        preprocessor: Any | None = None,
        target_binary_train: pd.Series | None = None,
        target_binary_val: pd.Series | None = None,
        target_binary_test: pd.Series | None = None,
    ) -> None:
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.activity_ids_train = activity_ids_train
        self.activity_ids_val = activity_ids_val
        self.activity_ids_test = activity_ids_test
        self.preprocessor = preprocessor
        self.target_binary_train = target_binary_train
        self.target_binary_val = target_binary_val
        self.target_binary_test = target_binary_test

    @property
    def num_train(self) -> int:
        return len(self.X_train)

    @property
    def num_val(self) -> int:
        return len(self.X_val)

    @property
    def num_test(self) -> int:
        return len(self.X_test)

    @property
    def num_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    def describe_splits(self) -> dict[str, Any]:
        return {
            "num_train": self.num_train,
            "num_val": self.num_val,
            "num_test": self.num_test,
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "activity_count_train": len(self.activity_ids_train),
            "activity_count_val": len(self.activity_ids_val),
            "activity_count_test": len(self.activity_ids_test),
        }


def load_step2_as_dataframes(processed_dir: Path | str) -> SplitData:
    """Load Step 2 feature matrix and convert to train/val/test DataFrames.

    This is a thin wrapper around build_feature_matrix that produces
    pandas objects suitable for sklearn pipelines.
    """
    processed_dir = Path(processed_dir)
    result = build_feature_matrix(processed_dir)

    # Build DataFrame from list-of-dicts
    X_all = pd.DataFrame(result["X"], columns=result["feature_names"])
    y_all = pd.Series(result["y"], name=TARGET_COLUMN)
    activity_ids_all = result["activity_ids"]
    split_assignments = result["split_assignments"]

    # Split by project-level assignment
    train_mask = []
    val_mask = []
    test_mask = []
    train_ids = []
    val_ids = []
    test_ids = []

    for i, (pid, _) in enumerate(activity_ids_all):
        split = split_assignments[(pid, _)]
        if split == "train":
            train_mask.append(True)
            val_mask.append(False)
            test_mask.append(False)
            train_ids.append(activity_ids_all[i])
        elif split == "validation":
            train_mask.append(False)
            val_mask.append(True)
            test_mask.append(False)
            val_ids.append(activity_ids_all[i])
        elif split == "test":
            train_mask.append(False)
            val_mask.append(False)
            test_mask.append(True)
            test_ids.append(activity_ids_all[i])
        else:
            raise ValueError(f"Unknown split: {split}")

    X_train = X_all[train_mask].reset_index(drop=True)
    X_val = X_all[val_mask].reset_index(drop=True)
    X_test = X_all[test_mask].reset_index(drop=True)
    y_train = y_all[train_mask].reset_index(drop=True)
    y_val = y_all[val_mask].reset_index(drop=True)
    y_test = y_all[test_mask].reset_index(drop=True)

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        activity_ids_train=train_ids,
        activity_ids_val=val_ids,
        activity_ids_test=test_ids,
    )


# ============================================================================
# Preprocessing: fit on TRAIN ONLY
# ============================================================================

def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that encodes categorical and scales numeric.

    IMPORTANT: this returns an UNFITTED transformer. Fitting must be done
    on training data only.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURE_NAMES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURE_NAMES),
        ],
        remainder="drop",
    )

    return preprocessor


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessor on TRAINING data only.

    Returns the fitted preprocessor.
    """
    preprocessor.fit(X_train)
    return preprocessor


def transform_splits(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transform train/val/test using a preprocessor that has already been fit.

    Returns DataFrames with deterministic column names.
    """
    X_train_t = pd.DataFrame(
        preprocessor.transform(X_train),
        columns=preprocessor.get_feature_names_out(),
    )
    X_val_t = pd.DataFrame(
        preprocessor.transform(X_val),
        columns=preprocessor.get_feature_names_out(),
    )
    X_test_t = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out(),
    )
    return X_train_t, X_val_t, X_test_t


def build_preprocessed_split(
    raw_split: SplitData,
    fit_preprocessor_now: bool = True,
) -> SplitData:
    """Build a SplitData with preprocessed matrices.

    Preprocessing is fit on train ONLY (unless fit_preprocessor_now is False,
    in which case the raw_split must already contain a fitted preprocessor).
    """
    if fit_preprocessor_now:
        preprocessor = build_preprocessor()
        preprocessor = fit_preprocessor(preprocessor, raw_split.X_train)
    else:
        preprocessor = raw_split.preprocessor
        if preprocessor is None:
            raise ValueError("No preprocessor provided and fit_preprocessor_now=False")

    X_train_t, X_val_t, X_test_t = transform_splits(
        preprocessor,
        raw_split.X_train,
        raw_split.X_val,
        raw_split.X_test,
    )

    # Binary target versions
    target_binary_train = (raw_split.y_train > 0).astype(int)
    target_binary_val = (raw_split.y_val > 0).astype(int)
    target_binary_test = (raw_split.y_test > 0).astype(int)

    return SplitData(
        X_train=X_train_t,
        X_val=X_val_t,
        X_test=X_test_t,
        y_train=raw_split.y_train,
        y_val=raw_split.y_val,
        y_test=raw_split.y_test,
        activity_ids_train=raw_split.activity_ids_train,
        activity_ids_val=raw_split.activity_ids_val,
        activity_ids_test=raw_split.activity_ids_test,
        preprocessor=preprocessor,
        target_binary_train=target_binary_train,
        target_binary_val=target_binary_val,
        target_binary_test=target_binary_test,
    )


# ============================================================================
# Evaluation utilities
# ============================================================================

def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute regression evaluation metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series | None = None,
) -> dict[str, float]:
    """Compute classification evaluation metrics.

    If y_prob is provided, also compute ROC-AUC, PR-AUC, and Brier score.
    """
    metrics: dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        try:
            from sklearn.metrics import average_precision_score
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ImportError:
            metrics["pr_auc"] = 0.0
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))

    return metrics


def hurdle_metrics(
    y_true: pd.Series,
    y_pred_stage1_prob: pd.Series,
    y_pred_stage2: pd.Series,
    stage1_threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a two-stage hurdle model.

    Stage 1: predict probability of delay > 0
    Stage 2: predict positive delay amount
    Final prediction: stage2_pred * (stage1_prob >= threshold)
    """
    stage1_pred = (y_pred_stage1_prob >= stage1_threshold).astype(int)
    final_pred = y_pred_stage2 * stage1_pred

    reg_metrics = regression_metrics(y_true, final_pred)

    cls_metrics = classification_metrics(
        (y_true > 0).astype(int),
        stage1_pred,
        y_prob=y_pred_stage1_prob,
    )

    return {
        "hurdle_reg MAE": reg_metrics["MAE"],
        "hurdle_reg RMSE": reg_metrics["RMSE"],
        "hurdle_reg MedianAE": reg_metrics["MedianAE"],
        "hurdle_reg R2": reg_metrics["R2"],
        "stage1_precision": cls_metrics["precision"],
        "stage1_recall": cls_metrics["recall"],
        "stage1_f1": cls_metrics["f1"],
        "stage1_roc_auc": cls_metrics.get("roc_auc", float("nan")),
        "stage1_pr_auc": cls_metrics.get("pr_auc", float("nan")),
        "stage1_brier": cls_metrics.get("brier", float("nan")),
    }


# ============================================================================
# Baseline model factories
# ============================================================================

def dummy_regressor_mean() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def dummy_regressor_median() -> DummyRegressor:
    return DummyRegressor(strategy="median")


def ridge_regression() -> Ridge:
    return Ridge(random_state=42)


def linear_regression() -> LinearRegression:
    return LinearRegression()


def logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=42)


def random_forest_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=42, n_estimators=100)


def random_forest_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=42, n_estimators=100)


def gradient_boosting_classifier() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=42, n_estimators=100)


def gradient_boosting_regressor() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=42, n_estimators=100)


# ============================================================================
# Fit-and-evaluate helpers
# ============================================================================

def fit_and_predict_regression(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
) -> tuple[Any, pd.Series]:
    """Fit a regression model on train and predict on validation.

    Returns (fitted_model, y_val_pred).
    """
    model.fit(X_train, y_train)
    y_val_pred = pd.Series(model.predict(X_val), name="y_val_pred")
    return model, y_val_pred


def fit_and_predict_classification(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
) -> tuple[Any, pd.Series, pd.Series]:
    """Fit a classification model on train and predict on validation.

    Returns (fitted_model, y_val_pred_class, y_val_pred_prob).
    """
    model.fit(X_train, y_train)
    y_val_pred = pd.Series(model.predict(X_val), name="y_val_pred")
    y_val_prob = pd.Series(model.predict_proba(X_val)[:, 1], name="y_val_prob")
    return model, y_val_pred, y_val_prob


def fit_model_and_report(
    model: Any,
    model_name: str,
    task: str,
    split: SplitData,
) -> dict[str, Any]:
    """Fit a model on train and evaluate on validation.

    task: "regression" or "classification"
    """
    report: dict[str, Any] = {"model": model_name, "task": task}

    if task == "regression":
        fitted, y_pred = fit_and_predict_regression(
            model, split.X_train, split.y_train, split.X_val
        )
        metrics = regression_metrics(split.y_val, y_pred)
        report.update(metrics)
        report["y_val_pred"] = y_pred
        report["fitted_model"] = fitted
    elif task == "classification":
        fitted, y_pred, y_prob = fit_and_predict_classification(
            model, split.X_train, split.target_binary_train, split.X_val
        )
        metrics = classification_metrics(split.target_binary_val, y_pred, y_prob)
        report.update(metrics)
        report["y_val_pred"] = y_pred
        report["y_val_prob"] = y_prob
        report["fitted_model"] = fitted
    else:
        raise ValueError(f"Unknown task: {task}")

    return report


# ============================================================================
# Hurdle model (two-stage)
# ============================================================================

def fit_hurdle_stage1(
    model: Any,
    split: SplitData,
) -> tuple[Any, pd.Series, pd.Series]:
    """Fit stage 1 classifier (delay > 0) and return predictions."""
    return fit_and_predict_classification(
        model, split.X_train, split.target_binary_train, split.X_val
    )


def fit_hurdle_stage2(
    model: Any,
    split: SplitData,
) -> tuple[Any, pd.Series]:
    """Fit stage 2 regressor on positive-target train rows only.

    Only rows where target > 0 are used in training.
    """
    train_mask = split.y_train > 0
    if train_mask.sum() == 0:
        raise ValueError("No positive-target rows for stage 2 training")
    model.fit(split.X_train[train_mask], split.y_train[train_mask])
    y_val_pred = pd.Series(model.predict(split.X_val), name="y_val_pred_stage2")
    return model, y_val_pred


def fit_and_evaluate_hurdle(
    stage1_model: Any,
    stage2_model: Any,
    split: SplitData,
    stage1_threshold: float = 0.5,
) -> dict[str, Any]:
    """Fit and evaluate a two-stage hurdle model."""
    _, y_stage1_pred, y_stage1_prob = fit_hurdle_stage1(
        stage1_model, split
    )
    _, y_stage2_pred = fit_hurdle_stage2(stage2_model, split)

    metrics = hurdle_metrics(
        split.y_val,
        y_stage1_prob,
        y_stage2_pred,
        stage1_threshold=stage1_threshold,
    )

    return {
        "model": "hurdle",
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "y_val_stage1_pred": y_stage1_pred,
        "y_val_stage1_prob": y_stage1_prob,
        "y_val_stage2_pred": y_stage2_pred,
        **metrics,
    }
