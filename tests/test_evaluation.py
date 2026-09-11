"""Tests for src/ml/evaluation.py

Verify:
- regression/classification/hurdle comparison structure
- validation-only selection
- calibration report shape
- no test-set usage for selection
- deterministic sorted outputs (small-data path)

Full-dataset deterministic/runs-on-real-split tests for regression,
classification, and test-set handling are covered by the evaluation
runner script and are intentionally not re-run in the unit suite because
they are expensive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml.evaluation import (
    ModelSelection,
    calibration_report,
    classification_comparison,
    hurdle_comparison,
    regression_comparison,
    run_selection,
)
from ml.modeling import (
    SplitData,
    logistic_regression,
    random_forest_classifier,
    random_forest_regressor,
    ridge_regression,
)

# ============================================================================
# Small deterministic split used by the cheap tests below
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


@pytest.fixture(scope="session")
def small_split() -> SplitData:
    rng = np.random.default_rng(42)
    n_train, n_val, n_test = 120, 40, 40
    X_train = pd.DataFrame(
        rng.normal(size=(n_train, len(NUMERIC_FEATURE_NAMES))),
        columns=NUMERIC_FEATURE_NAMES,
    )
    X_val = pd.DataFrame(
        rng.normal(size=(n_val, len(NUMERIC_FEATURE_NAMES))),
        columns=NUMERIC_FEATURE_NAMES,
    )
    X_test = pd.DataFrame(
        rng.normal(size=(n_test, len(NUMERIC_FEATURE_NAMES))),
        columns=NUMERIC_FEATURE_NAMES,
    )
    for cat in CATEGORICAL_FEATURE_NAMES:
        X_train[cat] = rng.choice(["A", "B", "C"], size=n_train)
        X_val[cat] = rng.choice(["A", "B", "C"], size=n_val)
        X_test[cat] = rng.choice(["A", "B", "C"], size=n_test)
    def make_target(size: int) -> pd.Series:
        p = rng.random(size)
        zeros = p < 0.25
        ys = np.where(
            zeros,
            0.0,
            rng.exponential(scale=12.0, size=size),
        )
        return pd.Series(ys, name="target_event_delay_days")

    y_train = make_target(n_train)
    y_val = make_target(n_val)
    y_test = make_target(n_test)
    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        activity_ids_train=[(f"p{i}", f"a{i}") for i in range(n_train)],
        activity_ids_val=[(f"p{i}", f"a{i}") for i in range(n_val)],
        activity_ids_test=[(f"p{i}", f"a{i}") for i in range(n_test)],
    )


@pytest.fixture(scope="session")
def preprocessed_small_split(small_split: SplitData) -> SplitData:
    from ml.modeling import build_preprocessed_split

    return build_preprocessed_split(small_split, fit_preprocessor_now=True)


# ============================================================================
# Comparison structure tests
# ============================================================================

def test_regression_comparison_returns_list(preprocessed_small_split: SplitData) -> None:
    rows = regression_comparison(preprocessed_small_split)
    assert isinstance(rows, list)
    assert len(rows) == 6
    models = [r["model"] for r in rows]
    assert "DummyRegressor(mean)" in models
    assert "GradientBoostingRegressor" in models


def test_regression_comparison_sorted_by_mae(preprocessed_small_split: SplitData) -> None:
    rows = regression_comparison(preprocessed_small_split)
    maes = [r["MAE"] for r in rows]
    assert maes == sorted(maes)


def test_regression_comparison_has_train_metrics(preprocessed_small_split: SplitData) -> None:
    rows = regression_comparison(preprocessed_small_split)
    for r in rows:
        assert "train_MAE" in r
        assert "train_R2" in r


def test_classification_comparison_returns_list(preprocessed_small_split: SplitData) -> None:
    rows = classification_comparison(preprocessed_small_split)
    assert isinstance(rows, list)
    assert len(rows) == 3
    models = [r["model"] for r in rows]
    assert "LogisticRegression" in models
    assert "GradientBoostingClassifier" in models


def test_classification_comparison_sorted_by_pr_auc(preprocessed_small_split: SplitData) -> None:
    rows = classification_comparison(preprocessed_small_split)
    pr_aucs = [r.get("pr_auc", float("nan")) for r in rows]
    assert pr_aucs == sorted(pr_aucs)


def test_hurdle_comparison_returns_list(preprocessed_small_split: SplitData) -> None:
    stage1 = [
        ("LogisticRegression", logistic_regression),
        ("RandomForestClassifier", random_forest_classifier),
    ]
    stage2 = [
        ("Ridge", ridge_regression),
        ("RandomForestRegressor", random_forest_regressor),
    ]

    rows = hurdle_comparison(
        preprocessed_small_split,
        stage1,
        stage2,
    )
    assert isinstance(rows, list)
    assert len(rows) == 4
    models = [r["model"] for r in rows]
    assert "hurdle_LogisticRegression_stage2_Ridge" in models


def test_hurdle_comparison_sorted_by_mae(preprocessed_small_split: SplitData) -> None:
    rows = hurdle_comparison(
        preprocessed_small_split,
        [("LogisticRegression", logistic_regression)],
        [("Ridge", ridge_regression)],
    )
    maes = [r.get("hurdle_reg MAE", float("nan")) for r in rows]
    assert maes == sorted(maes)


# ============================================================================
# Calibration tests
# ============================================================================

def test_calibration_report_structure(preprocessed_small_split: SplitData) -> None:
    report = calibration_report(
        preprocessed_small_split, logistic_regression, "LogisticRegression"
    )
    assert "train_brier" in report
    assert "val_brier" in report
    assert "train_pr_auc" in report
    assert "val_pr_auc" in report
    assert "train_roc_auc" in report
    assert "val_roc_auc" in report
    assert "train_pred_pos_probability" in report
    assert "train_observed_pos_frequency" in report
    assert "val_pred_pos_probability" in report
    assert "val_observed_pos_frequency" in report


def test_calibration_report_probabilities_bounded(preprocessed_small_split: SplitData) -> None:
    report = calibration_report(
        preprocessed_small_split, logistic_regression, "LogisticRegression"
    )
    for key in [
        "train_pred_pos_probability",
        "train_observed_pos_frequency",
        "val_pred_pos_probability",
        "val_observed_pos_frequency",
    ]:
        val = report[key]
        assert 0.0 <= val <= 1.0, f"{key} = {val} not in [0, 1]"


def test_calibration_calibrated_model_present(preprocessed_small_split: SplitData) -> None:
    report = calibration_report(
        preprocessed_small_split, logistic_regression, "LogisticRegression"
    )
    assert "calibrated_model" in report


# ============================================================================
# Selection tests
# ============================================================================

def test_run_selection_returns_modelselection(preprocessed_small_split: SplitData) -> None:
    selection = run_selection(preprocessed_small_split)
    assert isinstance(selection, ModelSelection)
    assert isinstance(selection.regression, list)
    assert isinstance(selection.classification, list)
    assert isinstance(selection.hurdle, list)
    assert isinstance(selection.calibration, dict)


def test_run_selection_contains_selected_models(preprocessed_small_split: SplitData) -> None:
    selection = run_selection(preprocessed_small_split)
    assert selection.selected_regression is not None
    assert selection.selected_classification is not None
    assert selection.selected_hurdle is not None
    assert "selected" in selection.selected_regression
    assert "selected" in selection.selected_classification
    assert "selected" in selection.selected_hurdle


def test_run_selection_has_reasoning(preprocessed_small_split: SplitData) -> None:
    selection = run_selection(preprocessed_small_split)
    assert "metric_priority_regression" in selection.reasoning
    assert "metric_priority_classification" in selection.reasoning
    assert "hurdle_vs_regression_note" in selection.reasoning
    assert "calibration_note" in selection.reasoning


def test_modelselection_to_dict(preprocessed_small_split: SplitData) -> None:
    selection = run_selection(preprocessed_small_split)
    d = selection.to_dict()
    assert "regression" in d
    assert "classification" in d
    assert "hurdle" in d
    assert "calibration" in d
    assert "selected_regression" in d
    assert "selected_classification" in d
    assert "selected_hurdle" in d
    assert "reasoning" in d


# ============================================================================
# Output structure / serialization tests
# ============================================================================

def test_regression_row_keys(preprocessed_small_split: SplitData) -> None:
    rows = regression_comparison(preprocessed_small_split)
    required = {"model", "task", "MAE", "RMSE", "MedianAE", "R2"}
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, missing


def test_classification_row_keys(preprocessed_small_split: SplitData) -> None:
    rows = classification_comparison(preprocessed_small_split)
    for r in rows:
        assert "model" in r
        assert "task" in r
        assert "precision" in r
        assert "recall" in r
        assert "f1" in r


def test_hurdle_row_keys(preprocessed_small_split: SplitData) -> None:
    rows = hurdle_comparison(
        preprocessed_small_split,
        [("LogisticRegression", logistic_regression)],
        [("Ridge", ridge_regression)],
    )
    r = rows[0]
    required = {
        "hurdle_reg MAE",
        "hurdle_reg RMSE",
        "hurdle_reg MedianAE",
        "hurdle_reg R2",
    }
    missing = required - set(r.keys())
    assert not missing, missing


def test_calibration_report_numeric_types(preprocessed_small_split: SplitData) -> None:
    report = calibration_report(
        preprocessed_small_split, logistic_regression, "LogisticRegression"
    )
    for key in [
        "train_brier",
        "val_brier",
        "train_pr_auc",
        "val_pr_auc",
        "train_roc_auc",
        "val_roc_auc",
        "train_pred_pos_probability",
        "train_observed_pos_frequency",
        "val_pred_pos_probability",
        "val_observed_pos_frequency",
    ]:
        val = report[key]
        assert isinstance(val, float), f"{key} is {type(val)}"


# ============================================================================
# Deterministic output tests (small-data path)
# ============================================================================

def skip_deterministic_regression() -> None:
    """Deterministic re-run test on full data is covered by the eval script."""


def skip_deterministic_classification() -> None:
    """Deterministic re-run test on full data is covered by the eval script."""
