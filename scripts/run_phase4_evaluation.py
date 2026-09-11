#!/usr/bin/env python
"""Phase 4 Step 4 — Run actual validation-only model selection.

This script:
- loads the Step 2 feature matrix
- builds the preprocessed split (preprocessing fit on train only)
- runs regression / classification / hurdle comparison
- runs calibration on TRAIN only
- runs validation-only selection
- writes notebooks/05_model_evaluation.md

Test set is NOT used for selection, tuning, or calibration fitting.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from ml.evaluation import (
    ModelSelection,
    calibration_report,
    classification_comparison,
    hurdle_comparison,
    regression_comparison,
    run_selection,
)
from ml.modeling import (
    build_preprocessed_split,
    load_step2_as_dataframes,
    logistic_regression,
    random_forest_classifier,
    random_forest_regressor,
    ridge_regression,
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_PATH = PROJECT_DIR / "notebooks" / "05_model_evaluation.md"

REG_STAGE1 = [
    ("LogisticRegression", logistic_regression),
    ("RandomForestClassifier", random_forest_classifier),
]

REG_STAGE2 = [
    ("Ridge", ridge_regression),
    ("RandomForestRegressor", random_forest_regressor),
]


def fmt(x, digits=4):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def build_regression_table(rows):
    lines = ["| Model | MAE | RMSE | MedianAE | R² | Train MAE | Train R² |"]
    lines.append("|-------|-----|------|----------|----|-----------|----------|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {fmt(r['MAE'])} | {fmt(r['RMSE'])} | "
            f"{fmt(r['MedianAE'])} | {fmt(r['R2'])} | {fmt(r.get('train_MAE'))} | "
            f"{fmt(r.get('train_R2'))} |"
        )
    return "\n".join(lines)


def build_classification_table(rows):
    lines = ["| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |"]
    lines.append("|-------|--------|---------|-----------|--------|-----|-------|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {fmt(r.get('pr_auc'))} | {fmt(r.get('roc_auc'))} | "
            f"{fmt(r.get('precision'))} | {fmt(r.get('recall'))} | "
            f"{fmt(r.get('f1'))} | {fmt(r.get('brier'))} |"
        )
    return "\n".join(lines)


def build_hurdle_table(rows):
    lines = ["| Hurdle Model | Reg MAE | Reg RMSE | Reg MedianAE | Reg R² | Stage1 PR-AUC |"]
    lines.append("|--------------|--------|---------|-------------|--------|---------------|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {fmt(r.get('hurdle_reg MAE'))} | "
            f"{fmt(r.get('hurdle_reg RMSE'))} | {fmt(r.get('hurdle_reg MedianAE'))} | "
            f"{fmt(r.get('hurdle_reg R2'))} | {fmt(r.get('stage1_pr_auc'))} |"
        )
    return "\n".join(lines)


def build_calibration_block(c):
    out = []
    out.append("**Training set**")
    out.append(f"- Observed positive-delay frequency: {fmt(c['train_observed_pos_frequency'], 4)}")
    out.append(f"- Mean predicted positive probability: {fmt(c['train_pred_pos_probability'], 4)}")
    out.append(f"- Brier score: {fmt(c['train_brier'], 4)}")
    out.append(f"- PR-AUC: {fmt(c['train_pr_auc'], 4)}")
    out.append(f"- ROC-AUC: {fmt(c['train_roc_auc'], 4)}")
    out.append("")
    out.append("**Validation set**")
    out.append(f"- Observed positive-delay frequency: {fmt(c['val_observed_pos_frequency'], 4)}")
    out.append(f"- Mean predicted positive probability: {fmt(c['val_pred_pos_probability'], 4)}")
    out.append(f"- Brier score: {fmt(c['val_brier'], 4)}")
    out.append(f"- PR-AUC: {fmt(c['val_pr_auc'], 4)}")
    out.append(f"- ROC-AUC: {fmt(c['val_roc_auc'], 4)}")
    return "\n".join(out)


def main():
    print("loading Step 2 data", flush=True)
    raw_split = load_step2_as_dataframes(PROCESSED_DIR)
    print(
        f"split loaded: train={raw_split.num_train}, "
        f"val={raw_split.num_val}, test={raw_split.num_test}",
        flush=True,
    )

    print("building preprocessed split (preprocessing fit on train only)", flush=True)
    preprocessed = build_preprocessed_split(raw_split, fit_preprocessor_now=True)
    print(f"preprocessed features: {preprocessed.num_features}", flush=True)

    print("regression comparison", flush=True)
    regression_rows = regression_comparison(preprocessed)

    print("classification comparison", flush=True)
    classification_rows = classification_comparison(preprocessed)

    print("hurdle comparison", flush=True)
    hurdle_rows = hurdle_comparison(
        preprocessed,
        REG_STAGE1,
        REG_STAGE2,
    )

    print("calibration report (train-only fitting)", flush=True)
    calibration = calibration_report(
        preprocessed,
        logistic_regression,
        "LogisticRegression",
    )

    print("running validation-only selection", flush=True)
    selection = run_selection(
        preprocessed,
        hurdle_stage1_models=REG_STAGE1,
        hurdle_stage2_models=REG_STAGE2,
        calibration_model_factory=logistic_regression,
        calibration_model_name="LogisticRegression",
    )

    report = []
    report.append("# Phase 4 Step 4 — Validation-Only Model Evaluation and Selection")
    report.append("")
    report.append(f"**Generated:** {datetime.datetime.now().isoformat()}")
    report.append("")
    report.append("## 1. Objective")
    report.append("")
    report.append(
        "This report evaluates the Phase 4 Step 3 baseline models on the **validation** "
        "split using **train-only** preprocessing and fitting. Model selection is performed "
        "using validation metrics only. The test set remains locked and is not used for "
        "model selection, hyperparameter tuning, threshold selection, or calibration fitting."
    )
    report.append("")
    report.append("Target: `target_event_delay_days` = event-induced/disruption delay days.")
    report.append("")
    report.append(
        "This is NOT actual schedule slippage. The dataset does not provide reliable actual "
        "activity start/finish timestamps."
    )
    report.append("")
    report.append("## 2. Dataset and Project-Level Split")
    report.append("")
    report.append(f"- Activities: {raw_split.num_train + raw_split.num_val + raw_split.num_test}")
    report.append(f"- Train projects/activities: 70 / {raw_split.num_train}")
    report.append(f"- Validation projects/activities: 15 / {raw_split.num_val}")
    report.append(f"- Test projects/activities: 15 / {raw_split.num_test}")
    report.append(f"- Features: {preprocessed.num_features}")
    report.append("")
    report.append("## 3. Test-Set Lock Policy")
    report.append("")
    report.append(
        "The test set is locked for this step. No test-set information influenced "
        "preprocessing fitting, model fitting, model selection, threshold selection, "
        "or calibration fitting."
    )
    report.append("")
    report.append("## 4. Regression Baseline Comparison (Validation)")
    report.append("")
    report.append("Primary metric: MAE. Selected model: "
                     + selection.selected_regression["selected"])
    report.append("")
    report.append(build_regression_table(regression_rows))
    report.append("")
    report.append("### Selected regression model")
    report.append("")
    reg = selection.selected_regression
    report.append(f"- Selected: {reg['selected']}")
    report.append(f"- Validation MAE: {fmt(reg['best_non_dummy_mae'])}")
    report.append(f"- Validation RMSE: {fmt(reg['best_non_dummy_rmse'])}")
    report.append(f"- Validation MedianAE: {fmt(reg['best_non_dummy_median_ae'])}")
    report.append(f"- Validation R²: {fmt(reg['best_non_dummy_r2'])}")
    report.append(f"- Train MAE: {fmt(reg.get('best_non_dummy_train_mae'))}")
    report.append(f"- Train R²: {fmt(reg.get('best_non_dummy_train_r2'))}")
    report.append(f"- Dummy mean MAE: {fmt(reg.get('dummy_mean_mae'))}")
    report.append("")
    report.append("## 5. Classification Baseline Comparison (Validation)")
    report.append("")
    report.append("Primary metric: PR-AUC. Selected model: "
                     + selection.selected_classification["selected"])
    report.append("")
    report.append(build_classification_table(classification_rows))
    report.append("")
    report.append("### Selected classification model")
    report.append("")
    cls = selection.selected_classification
    report.append(f"- Selected: {cls['selected']}")
    report.append(f"- Validation PR-AUC: {fmt(cls.get('best_pr_auc'))}")
    report.append(f"- Validation ROC-AUC: {fmt(cls.get('best_roc_auc'))}")
    report.append(f"- Validation Precision: {fmt(cls.get('best_precision'))}")
    report.append(f"- Validation Recall: {fmt(cls.get('best_recall'))}")
    report.append(f"- Validation F1: {fmt(cls.get('best_f1'))}")
    report.append(f"- Validation Brier: {fmt(cls.get('best_brier'))}")
    report.append("")
    report.append("## 6. Hurdle Model Comparison (Validation)")
    report.append("")
    report.append("Selected model: " + selection.selected_hurdle["selected"])
    report.append("")
    report.append(build_hurdle_table(hurdle_rows))
    report.append("")
    report.append("### Selected hurdle model")
    report.append("")
    hur = selection.selected_hurdle
    report.append(f"- Selected: {hur['selected']}")
    report.append(f"- Validation combined MAE: {fmt(hur.get('best_hurdle_mae'))}")
    report.append(f"- Validation combined RMSE: {fmt(hur.get('best_hurdle_rmse'))}")
    report.append(f"- Validation combined MedianAE: {fmt(hur.get('best_hurdle_median_ae'))}")
    report.append(f"- Validation combined R²: {fmt(hur.get('best_hurdle_r2'))}")
    report.append(f"- Stage 1 model: {hur.get('best_stage1_model')}")
    report.append(f"- Stage 2 model: {hur.get('best_stage2_model')}")
    report.append("")
    report.append("## 7. Calibration Analysis")
    report.append("")
    report.append(
        "Calibration is assessed using cross-validation on the **training set only**, "
        "with validation used only for evaluation reporting. No test data is used."
    )
    report.append("")
    report.append("Model: LogisticRegression (calibrated with sigmoid method, 5-fold CV on train).")
    report.append("")
    report.append(build_calibration_block(calibration))
    report.append("")
    report.append("## 8. Selection Reasoning")
    report.append("")
    for k, v in selection.reasoning.items():
        report.append(f"- **{k}**: {v}")
    report.append("")
    report.append("## 9. Limitations")
    report.append("")
    report.append(
        "- Results are validation-only. Test-set performance is not reported here because "
        "the test set is reserved for final unbiased evaluation after model selection."
    )
    report.append(
        "- The target is event-induced/disruption delay days, not actual schedule slippage."
    )
    report.append(
        "- No causal claims are made from these predictive results."
    )
    report.append("")

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"report written: {REPORT_PATH}", flush=True)

    summary = {
        "train": raw_split.num_train,
        "val": raw_split.num_val,
        "test": raw_split.num_test,
        "features": preprocessed.num_features,
        "selected_regression": selection.selected_regression["selected"],
        "selected_classification": selection.selected_classification["selected"],
        "selected_hurdle": selection.selected_hurdle["selected"],
        "regression_mae": selection.selected_regression["best_non_dummy_mae"],
        "classification_pr_auc": selection.selected_classification["best_pr_auc"],
        "hurdle_mae": selection.selected_hurdle["best_hurdle_mae"],
    }
    print("SUMMARY", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
