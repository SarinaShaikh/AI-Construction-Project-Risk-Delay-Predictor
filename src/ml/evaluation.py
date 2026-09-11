"""
Phase 4 Step 4 — Model evaluation and selection.

Evaluates the baseline models built in Phase 4 Step 3 on the validation split,
selects model(s) using validation metrics only, and documents calibration.

Test set remains locked until final evaluation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .modeling import (
    SplitData,
    classification_metrics,
    dummy_regressor_mean,
    dummy_regressor_median,
    fit_and_evaluate_hurdle,
    fit_model_and_report,
    gradient_boosting_classifier,
    gradient_boosting_regressor,
    linear_regression,
    logistic_regression,
    random_forest_classifier,
    random_forest_regressor,
    regression_metrics,
    ridge_regression,
)

# ============================================================================
# Model lists
# ============================================================================

REGRESSION_MODELS = [
    ("DummyRegressor(mean)", dummy_regressor_mean),
    ("DummyRegressor(median)", dummy_regressor_median),
    ("LinearRegression", linear_regression),
    ("Ridge", ridge_regression),
    ("RandomForestRegressor", random_forest_regressor),
    ("GradientBoostingRegressor", gradient_boosting_regressor),
]

CLASSIFICATION_MODELS = [
    ("LogisticRegression", logistic_regression),
    ("RandomForestClassifier", random_forest_classifier),
    ("GradientBoostingClassifier", gradient_boosting_classifier),
]


# ============================================================================
# Comparison helpers
# ============================================================================

def regression_comparison(
    split: SplitData,
) -> list[dict[str, Any]]:
    """Evaluate every regression baseline on validation using train only.

    Returns a list of reports, one per model, sorted by validation MAE.
    """
    rows = []
    for name, factory in REGRESSION_MODELS:
        report = fit_model_and_report(factory(), name, "regression", split)
        train_pred = report["fitted_model"].predict(split.X_train)
        train_metrics = regression_metrics(split.y_train, pd.Series(train_pred))
        report["train_MAE"] = train_metrics["MAE"]
        report["train_R2"] = train_metrics["R2"]
        rows.append(report)
    rows.sort(key=lambda r: r["MAE"])
    return rows


def classification_comparison(
    split: SplitData,
) -> list[dict[str, Any]]:
    """Evaluate every classification baseline on validation using train only.

    Returns a list of reports, one per model, sorted by validation PR-AUC.
    """
    rows = []
    for name, factory in CLASSIFICATION_MODELS:
        report = fit_model_and_report(factory(), name, "classification", split)
        rows.append(report)
    rows.sort(key=lambda r: r.get("pr_auc", float("nan")))
    return rows


def hurdle_comparison(
    split: SplitData,
    stage1_models,
    stage2_models,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Evaluate a small set of two-stage hurdle combinations on validation.

    Uses train only.
    """
    rows = []
    for s1_name, s1_factory in stage1_models:
        for s2_name, s2_factory in stage2_models:
            combo_name = f"hurdle_{s1_name}_stage2_{s2_name}"
            report = fit_and_evaluate_hurdle(
                s1_factory(), s2_factory(), split, stage1_threshold=threshold
            )
            report["model"] = combo_name
            rows.append(report)
    rows.sort(key=lambda r: r.get("hurdle_reg MAE", float("nan")))
    return rows


def calibration_report(
    split: SplitData,
    model_factory,
    model_name: str,
    cv: int = 5,
) -> dict[str, Any]:
    """Assess probability calibration and discrimination using cross-validation
    on the TRAINING split only.

    Does not use validation or test.
    """
    model = model_factory()
    calibrated = CalibratedClassifierCV(
        model, method="sigmoid", cv=cv
    )
    calibrated.fit(split.X_train, split.target_binary_train)

    prob_train = pd.Series(
        calibrated.predict_proba(split.X_train)[:, 1],
        name="prob_train",
    )

    prob_val = pd.Series(
        calibrated.predict_proba(split.X_val)[:, 1],
        name="prob_val",
    )

    train_cls = classification_metrics(
        split.target_binary_train, (prob_train >= 0.5).astype(int), y_prob=prob_train
    )
    val_cls = classification_metrics(
        split.target_binary_val, (prob_val >= 0.5).astype(int), y_prob=prob_val
    )

    train_pred_pos = prob_train.mean()
    train_pos_freq = split.target_binary_train.mean()
    val_pred_pos = prob_val.mean()
    val_pos_freq = split.target_binary_val.mean()

    return {
        "model": model_name,
        "calibrated_model": calibrated,
        "train_brier": train_cls.get("brier", float("nan")),
        "val_brier": val_cls.get("brier", float("nan")),
        "train_pr_auc": train_cls.get("pr_auc", float("nan")),
        "val_pr_auc": val_cls.get("pr_auc", float("nan")),
        "train_roc_auc": train_cls.get("roc_auc", float("nan")),
        "val_roc_auc": val_cls.get("roc_auc", float("nan")),
        "train_pred_pos_probability": float(train_pred_pos),
        "train_observed_pos_frequency": float(train_pos_freq),
        "val_pred_pos_probability": float(val_pred_pos),
        "val_observed_pos_frequency": float(val_pos_freq),
    }


# ============================================================================
# Selection policy
# ============================================================================

class ModelSelection:
    """Hold final validation-only model selection results.

    This object is returned by run_selection() and serialized for reporting.
    It deliberately does not include any test-set evaluation.
    """

    def __init__(
        self,
        regression: list[dict[str, Any]],
        classification: list[dict[str, Any]],
        hurdle: list[dict[str, Any]],
        calibration: dict[str, Any],
        selected_regression: dict[str, Any] | None,
        selected_classification: dict[str, Any] | None,
        selected_hurdle: dict[str, Any] | None,
        reasoning: dict[str, Any],
    ) -> None:
        self.regression = regression
        self.classification = classification
        self.hurdle = hurdle
        self.calibration = calibration
        self.selected_regression = selected_regression
        self.selected_classification = selected_classification
        self.selected_hurdle = selected_hurdle
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, Any]:
        return {
            "regression": self.regression,
            "classification": self.classification,
            "hurdle": self.hurdle,
            "calibration": self.calibration,
            "selected_regression": self.selected_regression,
            "selected_classification": self.selected_classification,
            "selected_hurdle": self.selected_hurdle,
            "reasoning": self.reasoning,
        }


def run_selection(
    split: SplitData,
    *,
    hurdle_stage1_models=None,
    hurdle_stage2_models=None,
    calibration_model_factory=None,
    calibration_model_name: str = "LogisticRegression",
) -> ModelSelection:
    """Run validation-only comparison and selection.

    Parameters
    ----------
    split : SplitData
        Must contain preprocessed train/val/test matrices.
    hurdle_stage1_models : list of (name, factory), optional
        If None, uses a small default set.
    hurdle_stage2_models : list of (name, factory), optional
        If None, uses a small default set.
    calibration_model_factory : callable, optional
        If None, uses LogisticRegression.
    """
    if hurdle_stage1_models is None:
        hurdle_stage1_models = [
            ("LogisticRegression", logistic_regression),
            ("RandomForestClassifier", random_forest_classifier),
        ]
    if hurdle_stage2_models is None:
        hurdle_stage2_models = [
            ("Ridge", ridge_regression),
            ("RandomForestRegressor", random_forest_regressor),
        ]
    if calibration_model_factory is None:
        calibration_model_factory = logistic_regression

    regression = regression_comparison(split)
    classification = classification_comparison(split)
    hurdle = hurdle_comparison(split, hurdle_stage1_models, hurdle_stage2_models)

    calibration = calibration_report(
        split,
        calibration_model_factory,
        calibration_model_name,
    )

    # ---- Selection rules ----
    selected_regression = _select_regression(regression)
    selected_classification = _select_classification(classification)
    selected_hurdle = _select_hurdle(hurdle)

    reasoning = {
        "metric_priority_regression": "validation MAE, then RMSE/MedianAE, then R2 and train/val gap",
        "metric_priority_classification": "validation PR-AUC, then ROC-AUC, Brier, recall/precision",
        "regression_selected_name": selected_regression["selected"] if selected_regression else None,
        "classification_selected_name": selected_classification["selected"] if selected_classification else None,
        "hurdle_selected_name": selected_hurdle["selected"] if selected_hurdle else None,
        "hurdle_vs_regression_note": (
            "Hurdle combined MAE is compared against direct regression MAE to assess "
            "whether separating occurrence from magnitude helps on this target distribution."
        ),
        "calibration_note": (
            "Calibration is assessed on train/validation only. Probabilities may be used "
            "as risk signals later; calibration quality is therefore documented here."
        ),
    }

    return ModelSelection(
        regression=regression,
        classification=classification,
        hurdle=hurdle,
        calibration=calibration,
        selected_regression=selected_regression,
        selected_classification=selected_classification,
        selected_hurdle=selected_hurdle,
        reasoning=reasoning,
    )


def _select_regression(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    best = rows[0]
    dummy_mae = None
    non_dummy_best = None
    for r in rows:
        if r["model"].startswith("DummyRegressor"):
            dummy_mae = r["MAE"]
        elif non_dummy_best is None or r["MAE"] < non_dummy_best["MAE"]:
            non_dummy_best = r
    if non_dummy_best is None:
        non_dummy_best = best
    selected = {
        "best_overall": best["model"],
        "best_overall_mae": best["MAE"],
        "best_overall_rmse": best["RMSE"],
        "best_overall_median_ae": best["MedianAE"],
        "best_overall_r2": best["R2"],
        "best_non_dummy": non_dummy_best["model"],
        "best_non_dummy_mae": non_dummy_best["MAE"],
        "best_non_dummy_rmse": non_dummy_best["RMSE"],
        "best_non_dummy_median_ae": non_dummy_best["MedianAE"],
        "best_non_dummy_r2": non_dummy_best["R2"],
        "best_non_dummy_train_mae": non_dummy_best.get("train_MAE"),
        "best_non_dummy_train_r2": non_dummy_best.get("train_R2"),
        "dummy_mean_mae": dummy_mae,
        "selected": non_dummy_best["model"],
    }
    return selected


def _select_classification(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    best = rows[-1]  # rows sorted by PR-AUC ascending
    selected = {
        "best_overall": best["model"],
        "best_pr_auc": best.get("pr_auc"),
        "best_roc_auc": best.get("roc_auc"),
        "best_brier": best.get("brier"),
        "best_recall": best.get("recall"),
        "best_precision": best.get("precision"),
        "best_f1": best.get("f1"),
        "selected": best["model"],
    }
    return selected


def _select_hurdle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    best = rows[0]  # sorted ASCENDING by hurdle MAE, so index 0 is the best (lowest) validation MAE
    selected = {
        "best_hurdle_mae": best.get("hurdle_reg MAE"),
        "best_hurdle_rmse": best.get("hurdle_reg RMSE"),
        "best_hurdle_median_ae": best.get("hurdle_reg MedianAE"),
        "best_hurdle_r2": best.get("hurdle_reg R2"),
        "best_stage1_model": best.get("stage1_model"),
        "best_stage2_model": best.get("stage2_model"),
        "selected": best["model"],
    }
    return selected
