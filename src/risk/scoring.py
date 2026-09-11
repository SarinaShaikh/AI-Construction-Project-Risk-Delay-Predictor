"""
Phase 5: Risk Scoring Engine.

Combines Phase 4 ML delay predictions with Phase 3 CPM results to produce
activity-level risk scores.

Risk components (all clearly separated):
    1. probability_of_event_delay  — LogisticRegression predict_proba output (TRUE probability)
    2. predicted_delay_days        — LinearRegression prediction (predicted magnitude)
    3. criticality_weight          — Heuristic based on total_float bins
    4. float_impact                — Heuristic: predicted_delay / available_float
    5. risk_score                  — probability * criticality_weight * float_impact

Heuristic components (criticality_weight, float_impact) are documented as
heuristics, NOT learned/statistical facts.

The production risk engine does NOT depend on validation-set predictions specifically.
It uses the trained models to generate predictions for all activities in the provided
split, then combines them with CPM results.

Target: target_event_delay_days = event-induced/disruption delay days.
This is NOT actual schedule slippage.

Test set remains locked. No test-set information is used for risk score design,
tuning, or model selection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.modeling import (
    SplitData,
    build_preprocessed_split,
    load_step2_as_dataframes,
    logistic_regression,
    LinearRegression,
    fit_and_predict_classification,
    fit_and_predict_regression,
)


# ============================================================================
# Criticality Weight Heuristic
# ============================================================================

def criticality_weight(total_float: pd.Series) -> pd.Series:
    """
    Convert CPM total float into a criticality weight using binned thresholds.

    This is a HEURISTIC, not a learned or statistical quantity.
    Rules:
        float <= 0 days       -> 1.0  (critical or near-critical)
        float < 5 days        -> 0.9
        float < 20 days       -> 0.7
        float < 50 days       -> 0.4
        otherwise             -> 0.1

    Parameters
    ----------
    total_float : pd.Series
        Total float values (in days) for each activity.

    Returns
    -------
    pd.Series
        Criticality weights in [0.1, 1.0].
    """
    values = pd.to_numeric(total_float, errors="coerce").fillna(0.0)

    weights = np.select(
        [
            values <= 1e-6,
            values < 5.0,
            values < 20.0,
            values < 50.0,
        ],
        [
            1.0,
            0.9,
            0.7,
            0.4,
        ],
        default=0.1,
    )

    return pd.Series(weights, index=total_float.index, dtype=float)


# ============================================================================
# Float Impact Heuristic
# ============================================================================

def float_impact(
    predicted_delay: pd.Series,
    total_float: pd.Series,
) -> pd.Series:
    """
    Measure how much predicted delay consumes available float.

    This is a HEURISTIC, not a learned or statistical quantity.

    Formula:
        min(predicted_delay / max(total_float, 1), 1)

    For zero/near-zero float:
        positive delay -> 1.0
        zero delay     -> 0.0

    Parameters
    ----------
    predicted_delay : pd.Series
        Predicted delay days for each activity.
    total_float : pd.Series
        Total float values (in days) for each activity.

    Returns
    -------
    pd.Series
        Float impact values in [0, 1].
    """
    delay = pd.to_numeric(predicted_delay, errors="coerce").fillna(0.0).clip(lower=0.0)
    free_float = pd.to_numeric(total_float, errors="coerce").fillna(0.0)

    impact = np.zeros(len(delay), dtype=float)

    zero_float = free_float <= 1e-6
    positive_float = ~zero_float

    impact[zero_float] = np.where(
        delay[zero_float].to_numpy() > 0,
        1.0,
        0.0,
    )

    impact[positive_float] = np.minimum(
        delay[positive_float].to_numpy()
        / np.maximum(
            free_float[positive_float].to_numpy(),
            1.0,
        ),
        1.0,
    )

    return pd.Series(impact, index=predicted_delay.index, dtype=float)


# ============================================================================
# Risk Score Calculator
# ============================================================================

class RiskScoreCalculator:
    """
    Calculate activity-level risk scores combining ML predictions and CPM.

    Heuristics used:
    - criticality_weight: binned total_float (documented as heuristic)
    - float_impact: predicted_delay / available_float (documented as heuristic)

    Probability comes from LogisticRegression predict_proba (true probability).
    Delay magnitude comes from LinearRegression prediction.

    The production engine does NOT depend on validation-set predictions.
    It uses the trained models to generate predictions for all activities
    in the provided split.
    """

    def __init__(
        self,
        repo_root: Path | str | None = None,
    ) -> None:
        """
        Initialize the risk score calculator.

        Parameters
        ----------
        repo_root : Path or str, optional
            Root directory of the repository. If None, uses the parent
            of this file's location.
        """
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )

        self.processed_dir = self.repo_root / "data" / "processed"
        self.cpm_path = self.processed_dir / "cpm" / "activity_cpm_results.csv"
        self.activities_path = self.repo_root / "data" / "raw" / "activities.csv"

        self.cpm_results: pd.DataFrame | None = None
        self.activities: pd.DataFrame | None = None

    def load_cpm_results(self, cpm_path: Path | str | None = None) -> pd.DataFrame:
        path = Path(cpm_path) if cpm_path is not None else self.cpm_path

        if not path.exists():
            raise FileNotFoundError(f"CPM results not found: {path}")

        df = pd.read_csv(path)

        required_cpm = {
            "project_id",
            "activity_id",
            "ES",
            "EF",
            "LS",
            "LF",
            "total_float",
            "computed_critical",
        }

        missing = required_cpm - set(df.columns)
        if missing:
            raise ValueError(
                f"CPM results missing required columns: {', '.join(sorted(missing))}"
            )

        for col in ["project_id", "activity_id"]:
            df[col] = df[col].astype(str)

        keys = df[["project_id", "activity_id"]].astype(str)
        if keys.duplicated().any():
            raise ValueError(
                "CPM results contain duplicate (project_id, activity_id) keys."
            )

        self.cpm_results = df
        return df

    def load_activities(self, activities_path: Path | str | None = None) -> pd.DataFrame:
        path = Path(activities_path) if activities_path is not None else self.activities_path

        if not path.exists():
            self.activities = pd.DataFrame()
            return self.activities

        df = pd.read_csv(path)

        required_activities = {"project_id", "activity_id"}
        missing = required_activities - set(df.columns)
        if missing:
            raise ValueError(
                f"Activities file missing required columns: {', '.join(sorted(missing))}"
            )

        for col in ["project_id", "activity_id"]:
            df[col] = df[col].astype(str)

        keys = df[["project_id", "activity_id"]].astype(str)
        if keys.duplicated().any():
            raise ValueError(
                "Activities file contains duplicate (project_id, activity_id) keys."
            )

        self.activities = df
        return df

    def _validate_predictions(self, predictions: pd.DataFrame) -> None:
        required_cols = {"project_id", "activity_id"}
        missing = required_cols - set(predictions.columns)
        if missing:
            raise ValueError(
                f"Predictions missing required columns: {', '.join(sorted(missing))}"
            )

        for col in ["project_id", "activity_id"]:
            predictions[col] = predictions[col].astype(str)

        if "project_id" in predictions.columns and "activity_id" in predictions.columns:
            if predictions.duplicated(["project_id", "activity_id"]).any():
                raise ValueError(
                    "Duplicate ML predictions for (project_id, activity_id)."
                )
        elif "activity_id" in predictions.columns:
            if predictions["activity_id"].duplicated().any():
                raise ValueError(
                    "Duplicate activity_id values in ML data. "
                    "project_id is required to disambiguate them."
                )

        pred_cols = ["predicted_delay_days", "probability_of_event_delay"]
        for col in pred_cols:
            if col in predictions.columns:
                values = pd.to_numeric(predictions[col], errors="coerce")
                if values.isna().any():
                    raise ValueError(f"Missing/non-numeric values in {col}.")
                if col == "probability_of_event_delay":
                    if (values < 0).any() or (values > 1).any():
                        raise ValueError(
                            f"probability_of_event_delay must be in [0, 1], got range [{values.min()}, {values.max()}]"
                        )
                if col == "predicted_delay_days":
                    if (values < 0).any():
                        raise ValueError(
                            f"predicted_delay_days must be non-negative, got min {values.min()}"
                        )

    def _merge_predictions(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        self._validate_predictions(predictions)

        cpm = self.cpm_results.copy()

        if "project_id" in predictions.columns and "activity_id" in predictions.columns:
            merged = cpm.merge(
                predictions[
                    [
                        "project_id",
                        "activity_id",
                        "predicted_delay_days",
                        "probability_of_event_delay",
                    ]
                ],
                on=["project_id", "activity_id"],
                how="left",
                validate="one_to_one",
            )
        elif "activity_id" in predictions.columns:
            merged = cpm.merge(
                predictions[
                    [
                        "activity_id",
                        "predicted_delay_days",
                        "probability_of_event_delay",
                    ]
                ],
                on=["activity_id"],
                how="left",
                validate="one_to_one",
            )
        else:
            raise ValueError("ML predictions have no usable activity identifier.")

        missing_predictions = merged["predicted_delay_days"].isna().sum()
        if missing_predictions:
            import logging
            logging.warning(
                "%d CPM activities have no ML prediction and will be excluded from risk scoring. "
                "This is expected when scoring is restricted to a subset of CPM activities "
                "(e.g. a validation split).",
                missing_predictions,
            )

        return merged

    def calculate(
        self,
        preprocessed_split: SplitData,
        *,
        fit_models_now: bool = True,
    ) -> pd.DataFrame:
        if fit_models_now:
            clf = logistic_regression()
            _, _y_pred_class, y_prob = fit_and_predict_classification(
                clf,
                preprocessed_split.X_train,
                preprocessed_split.target_binary_train,
                preprocessed_split.X_val,
            )

            reg = LinearRegression()
            _, y_pred_delay = fit_and_predict_regression(
                reg,
                preprocessed_split.X_train,
                preprocessed_split.y_train,
                preprocessed_split.X_val,
            )
        else:
            raise NotImplementedError(
                "Using pre-fitted models requires the models to be attached "
                "to the SplitData, which is not currently supported."
            )

        predictions = pd.DataFrame({
            "project_id": [pid for pid, _ in preprocessed_split.activity_ids_val],
            "activity_id": [aid for _, aid in preprocessed_split.activity_ids_val],
            "predicted_delay_days": np.maximum(y_pred_delay, 0.0),
            "probability_of_event_delay": y_prob,
        })

        predictions["probability_of_event_delay"] = predictions[
            "probability_of_event_delay"
        ].clip(0.0, 1.0)

        merged = self._merge_predictions(predictions)

        merged["criticality_weight"] = criticality_weight(merged["total_float"])
        merged["float_impact"] = float_impact(
            merged["predicted_delay_days"],
            merged["total_float"],
        )

        merged["risk_score"] = (
            merged["probability_of_event_delay"]
            * merged["criticality_weight"]
            * merged["float_impact"]
        )

        if self.activities is not None and len(self.activities) > 0:
            activity_meta = self.activities.copy()
            activity_meta["project_id"] = activity_meta["project_id"].astype(str)
            activity_meta["activity_id"] = activity_meta["activity_id"].astype(str)

            metadata_candidates = [
                "project_id",
                "activity_id",
                "activity_name",
                "activity_type",
                "phase",
                "resource_type",
            ]

            metadata_columns = [
                col for col in metadata_candidates if col in activity_meta.columns
            ]

            if metadata_columns:
                activity_meta = activity_meta[metadata_columns].drop_duplicates(
                    ["project_id", "activity_id"]
                )

                merged = merged.merge(
                    activity_meta,
                    on=["project_id", "activity_id"],
                    how="left",
                    suffixes=("", "_activity"),
                    validate="one_to_one",
                )

        preferred_columns = [
            "project_id",
            "activity_id",
            "activity_name",
            "activity_type",
            "phase",
            "resource_type",
            "predicted_delay_days",
            "probability_of_event_delay",
            "total_float",
            "computed_critical",
            "criticality_weight",
            "float_impact",
            "risk_score",
            "ES",
            "EF",
            "LS",
            "LF",
        ]

        final_columns = [col for col in preferred_columns if col in merged.columns]

        results = merged.dropna(subset=["predicted_delay_days"]).copy()
        if len(results) == 0:
            raise ValueError("Risk scoring produced no rows after dropping rows without predictions.")

        results = results[final_columns].copy()

        results = results.sort_values(
            "risk_score",
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)

        results["risk_rank"] = np.arange(1, len(results) + 1)

        if len(results) > 1:
            results["risk_percentile"] = (
                results["risk_score"].rank(method="average", pct=True) * 100.0
            )
        else:
            results["risk_percentile"] = 100.0

        for col in [
            "predicted_delay_days",
            "probability_of_event_delay",
            "total_float",
            "criticality_weight",
            "float_impact",
            "risk_score",
            "risk_percentile",
            "ES",
            "EF",
            "LS",
            "LF",
        ]:
            if col in results.columns:
                results[col] = results[col].round(6)

        return results

    def summary(self, results: pd.DataFrame) -> dict[str, Any]:
        scores = results["risk_score"]

        q75 = float(scores.quantile(0.75))
        q95 = float(scores.quantile(0.95))

        return {
            "total_activities": int(len(results)),
            "mean_risk": float(scores.mean()),
            "std_risk": float(scores.std()),
            "min_risk": float(scores.min()),
            "max_risk": float(scores.max()),
            "high_risk_threshold": q75,
            "critical_risk_threshold": q95,
            "high_risk_activities": int((scores > q75).sum()),
            "critical_risk_activities": int((scores > q95).sum()),
        }


def calculate_risk_scores(
    repo_root: Path | str | None = None,
    fit_models_now: bool = True,
) -> pd.DataFrame:
    calculator = RiskScoreCalculator(repo_root=repo_root)
    calculator.load_cpm_results()
    calculator.load_activities()

    processed_dir = calculator.processed_dir
    raw_split = load_step2_as_dataframes(processed_dir)
    preprocessed = build_preprocessed_split(raw_split, fit_preprocessor_now=True)

    results = calculator.calculate(preprocessed, fit_models_now=fit_models_now)
    return results
