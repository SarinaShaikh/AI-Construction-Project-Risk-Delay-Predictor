#!/usr/bin/env python3
"""
Phase 5: Risk Scoring Engine

Combines:
    1. Phase 4 ML delay predictions
    2. Phase 3 CPM total float / criticality
    3. Activity metadata

Risk formula:
    risk_score = delay_probability * criticality_weight * float_impact

Input files:
    data/raw/activities.csv
    data/processed/cpm/activity_cpm_results.csv
    data/processed/ml/best_model.pkl
    data/processed/ml/scaler.pkl
    data/processed/ml/feature_names.json
    data/processed/ml/X_train.csv
    data/processed/ml/X_val.csv
    data/processed/ml/X_test.csv

Output:
    data/processed/risk_scores.csv

Important:
    ML predictions are matched to activities using activity_id/project_id
    whenever those identifiers are present in the ML feature files. This
    avoids silently relying on row order.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class RiskScorer:
    """Calculate activity-level construction risk scores."""

    def __init__(
        self,
        repo_root: Path | str | None = None,
    ) -> None:
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )

        self.raw_dir = self.repo_root / "data" / "raw"
        self.processed_dir = self.repo_root / "data" / "processed"
        self.ml_dir = self.processed_dir / "ml"
        self.cpm_dir = self.processed_dir / "cpm"

        self.model_path = self.ml_dir / "best_model.pkl"
        self.scaler_path = self.ml_dir / "scaler.pkl"
        self.feature_names_path = self.ml_dir / "feature_names.json"

        self.cpm_path = self.cpm_dir / "activity_cpm_results.csv"
        self.activities_path = self.raw_dir / "activities.csv"

        self.X_paths = [
            self.ml_dir / "X_train.csv",
            self.ml_dir / "X_val.csv",
            self.ml_dir / "X_test.csv",
        ]

        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.cpm_results: pd.DataFrame | None = None
        self.activities: pd.DataFrame | None = None
        self.results: pd.DataFrame | None = None

        self._load_resources()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_resources(self) -> None:
        print("Initializing RiskScorer...")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ML model not found: {self.model_path}"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found: {self.scaler_path}"
            )

        if not self.feature_names_path.exists():
            raise FileNotFoundError(
                f"Feature names not found: {self.feature_names_path}"
            )

        if not self.cpm_path.exists():
            raise FileNotFoundError(
                f"CPM results not found: {self.cpm_path}"
            )

        if not self.activities_path.exists():
            raise FileNotFoundError(
                f"Activities file not found: {self.activities_path}"
            )

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(self.feature_names_path, "r", encoding="utf-8") as f:
            loaded_names = json.load(f)

        # Support either:
        #   ["feature1", "feature2", ...]
        # or:
        #   {"feature_names": ["feature1", ...]}
        if isinstance(loaded_names, dict):
            self.feature_names = loaded_names.get(
                "feature_names",
                loaded_names.get("features", []),
            )
        elif isinstance(loaded_names, list):
            self.feature_names = loaded_names
        else:
            raise ValueError(
                "feature_names.json must contain a list or a dictionary "
                "containing 'feature_names'."
            )

        if not self.feature_names:
            raise ValueError("No feature names found in feature_names.json.")

        self.cpm_results = pd.read_csv(self.cpm_path)
        self.activities = pd.read_csv(self.activities_path)

        print(
            f"  ML model loaded: "
            f"{type(self.model).__name__}"
        )
        print("  StandardScaler loaded")
        print(f"  Feature names loaded: {len(self.feature_names)} features")
        print(
            f"  CPM results loaded: "
            f"{len(self.cpm_results)} activities"
        )
        print(
            f"  Activities loaded: "
            f"{len(self.activities)} activities"
        )

        self._validate_inputs()

        print("  RiskScorer initialized successfully")

    def _validate_inputs(self) -> None:
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

        missing_cpm = required_cpm - set(self.cpm_results.columns)
        if missing_cpm:
            raise ValueError(
                "CPM results are missing required columns: "
                + ", ".join(sorted(missing_cpm))
            )

        if "project_id" not in self.activities.columns:
            raise ValueError(
                "activities.csv is missing required column: project_id"
            )

        if "activity_id" not in self.activities.columns:
            raise ValueError(
                "activities.csv is missing required column: activity_id"
            )

        cpm_keys = self.cpm_results[
            ["project_id", "activity_id"]
        ].astype(str)

        if cpm_keys.duplicated().any():
            raise ValueError(
                "CPM results contain duplicate (project_id, activity_id) keys."
            )

    # ------------------------------------------------------------------
    # ML feature loading
    # ------------------------------------------------------------------

    def _load_ml_rows(self) -> pd.DataFrame:
        """
        Load all ML feature rows.

        The preferred design is that each X file contains project_id and/or
        activity_id. If identifiers are absent, we cannot safely match
        predictions to CPM activities, so we fail instead of guessing.
        """
        frames: list[pd.DataFrame] = []

        for path in self.X_paths:
            if not path.exists():
                raise FileNotFoundError(f"ML feature file not found: {path}")

            df = pd.read_csv(path)
            df["_ml_source"] = path.name
            frames.append(df)

        ml = pd.concat(frames, ignore_index=True)

        id_columns = {
            "project_id",
            "activity_id",
        }

        available_ids = id_columns.intersection(ml.columns)

        if not available_ids:
            raise ValueError(
                "X_train.csv, X_val.csv, and X_test.csv do not contain "
                "project_id or activity_id. "
                "Risk scoring cannot safely match ML predictions to CPM "
                "activities without an identifier."
            )

        # Normalize identifiers for reliable merging.
        for col in available_ids:
            ml[col] = ml[col].astype(str)

        return ml

    def _prepare_features(
        self,
        ml_rows: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return:
            identifiers: project/activity IDs used to match predictions
            X: model feature matrix
        """
        missing_features = [
            f for f in self.feature_names
            if f not in ml_rows.columns
        ]

        if missing_features:
            raise ValueError(
                "The ML data is missing required model features: "
                + ", ".join(missing_features)
            )

        identifiers = pd.DataFrame(index=ml_rows.index)

        if "project_id" in ml_rows.columns:
            identifiers["project_id"] = ml_rows["project_id"].astype(str)

        if "activity_id" in ml_rows.columns:
            identifiers["activity_id"] = ml_rows["activity_id"].astype(str)

        X = ml_rows[self.feature_names].copy()

        # Convert to numeric and reject missing/infinite values.
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any():
            bad_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(
                "Missing/non-numeric values found in ML features: "
                + ", ".join(bad_cols)
            )

        X_values = X.to_numpy(dtype=float)

        if not np.isfinite(X_values).all():
            raise ValueError("ML feature matrix contains NaN or infinite values.")

        # X files produced by prepare_ml_data.py are already standardized.
        # Therefore we normally pass them directly to the saved model.
        #
        # The scaler is loaded/validated because it belongs to the same
        # Phase 4 pipeline, but it is NOT applied a second time here.
        return identifiers, X

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def generate_predictions(self) -> pd.DataFrame:
        """Generate one ML delay prediction per ML activity row."""
        print("\n1. Generating ML predictions...")

        ml_rows = self._load_ml_rows()
        identifiers, X = self._prepare_features(ml_rows)

        predictions = self.model.predict(X)
        predictions = np.asarray(predictions, dtype=float)

        # Delay cannot be negative for the risk engine.
        predictions = np.maximum(predictions, 0.0)

        pred_df = identifiers.copy()
        pred_df["delay_prediction_days"] = predictions

        # If both IDs exist, use both as the safest key.
        # If only activity_id exists, activity_id must be unique.
        if "project_id" in pred_df.columns and "activity_id" in pred_df.columns:
            if pred_df.duplicated(
                ["project_id", "activity_id"]
            ).any():
                raise ValueError(
                    "Duplicate ML predictions for "
                    "(project_id, activity_id)."
                )
        elif "activity_id" in pred_df.columns:
            if pred_df["activity_id"].duplicated().any():
                raise ValueError(
                    "Duplicate activity_id values in ML data. "
                    "project_id is required to disambiguate them."
                )

        print(
            f"   Predictions generated for {len(pred_df)} activities"
        )
        print(
            f"   Mean prediction: {predictions.mean():.4f} days"
        )
        print(
            f"   Std prediction: {predictions.std():.4f} days"
        )

        return pred_df

    # ------------------------------------------------------------------
    # Risk components
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_delay_probability(
        predictions: pd.Series,
    ) -> pd.Series:
        """
        Normalize predicted delay using the 95th percentile.

        Values above the 95th percentile are clipped to 1.
        """
        predictions = pd.to_numeric(
            predictions,
            errors="coerce",
        ).fillna(0.0)

        predictions = predictions.clip(lower=0.0)

        p95 = float(predictions.quantile(0.95))

        if p95 <= 0:
            return pd.Series(
                np.zeros(len(predictions)),
                index=predictions.index,
                dtype=float,
            )

        probability = predictions / p95

        return probability.clip(0.0, 1.0)

    @staticmethod
    def calculate_criticality_weight(
        total_float: pd.Series,
    ) -> pd.Series:
        """
        Convert CPM total float into a criticality weight.

        Rules:
            float <= 0 days       -> 1.0
            float < 5 days        -> 0.9
            float < 20 days       -> 0.7
            float < 50 days       -> 0.4
            otherwise             -> 0.1
        """
        values = pd.to_numeric(
            total_float,
            errors="coerce",
        ).fillna(0.0)

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

        return pd.Series(
            weights,
            index=total_float.index,
            dtype=float,
        )

    @staticmethod
    def calculate_float_impact(
        predictions: pd.Series,
        total_float: pd.Series,
    ) -> pd.Series:
        """
        Measure how much predicted delay consumes available float.

        Formula:
            min(predicted_delay / max(total_float, 1), 1)

        For zero/near-zero float:
            positive delay -> 1
            zero delay     -> 0
        """
        delay = pd.to_numeric(
            predictions,
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)

        free_float = pd.to_numeric(
            total_float,
            errors="coerce",
        ).fillna(0.0)

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

        return pd.Series(
            impact,
            index=predictions.index,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Matching predictions with CPM
    # ------------------------------------------------------------------

    def _merge_predictions(
        self,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Safely merge ML predictions with CPM results."""
        cpm = self.cpm_results.copy()

        for col in ["project_id", "activity_id"]:
            cpm[col] = cpm[col].astype(str)

        if "project_id" in predictions.columns and "activity_id" in predictions.columns:
            merged = cpm.merge(
                predictions[
                    [
                        "project_id",
                        "activity_id",
                        "delay_prediction_days",
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
                        "delay_prediction_days",
                    ]
                ],
                on=["activity_id"],
                how="left",
                validate="one_to_one",
            )
        else:
            raise ValueError(
                "ML predictions have no usable activity identifier."
            )

        missing_predictions = merged[
            "delay_prediction_days"
        ].isna().sum()

        if missing_predictions:
            raise ValueError(
                f"{missing_predictions} CPM activities have no ML prediction. "
                "Check that the Phase 4 ML split files cover all 9,279 "
                "activities and contain matching identifiers."
            )

        return merged

    # ------------------------------------------------------------------
    # Build risk results
    # ------------------------------------------------------------------

    def calculate_risk_scores(self) -> pd.DataFrame:
        """Calculate complete activity-level risk scores."""
        print("\n" + "=" * 70)
        print("CALCULATING RISK SCORES")
        print("=" * 70)

        predictions = self.generate_predictions()

        print("\n2. Calculating delay probability...")
        merged = self._merge_predictions(predictions)

        merged["delay_probability"] = self.calculate_delay_probability(
            merged["delay_prediction_days"]
        )

        print("   Delay probability calculated")
        print(
            f"   Mean: {merged['delay_probability'].mean():.4f}"
        )
        print(
            "   Range: "
            f"[{merged['delay_probability'].min():.4f}, "
            f"{merged['delay_probability'].max():.4f}]"
        )

        print("\n3. Calculating criticality weight...")
        merged["criticality_weight"] = self.calculate_criticality_weight(
            merged["total_float"]
        )

        print("   Criticality weight calculated")
        print(
            f"   Mean: {merged['criticality_weight'].mean():.4f}"
        )
        print(
            "   Critical activities (weight>=0.9): "
            f"{(merged['criticality_weight'] >= 0.9).sum()}"
        )

        print("\n4. Calculating float impact...")
        merged["float_impact"] = self.calculate_float_impact(
            merged["delay_prediction_days"],
            merged["total_float"],
        )

        print("   Float impact calculated")
        print(
            f"   Mean: {merged['float_impact'].mean():.4f}"
        )
        print(
            f"   Max: {merged['float_impact'].max():.4f}"
        )

        print("\n5. Calculating risk score...")

        merged["risk_score"] = (
            merged["delay_probability"]
            * merged["criticality_weight"]
            * merged["float_impact"]
        )

        print(
            "   Risk score calculated "
            "(delay_prob x criticality x float_impact)"
        )
        print(
            f"   Mean: {merged['risk_score'].mean():.4f}"
        )
        print(
            f"   Std: {merged['risk_score'].std():.4f}"
        )
        print(
            f"   Range: "
            f"[{merged['risk_score'].min():.4f}, "
            f"{merged['risk_score'].max():.4f}]"
        )

        # Add activity metadata.
        activity_meta = self.activities.copy()

        activity_meta["project_id"] = activity_meta[
            "project_id"
        ].astype(str)
        activity_meta["activity_id"] = activity_meta[
            "activity_id"
        ].astype(str)

        # Only add metadata columns that actually exist.
        metadata_candidates = [
            "project_id",
            "activity_id",
            "phase",
            "resource_type",
            "activity_name",
            "activity_type",
        ]

        metadata_columns = [
            col
            for col in metadata_candidates
            if col in activity_meta.columns
        ]

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

        # Build final output columns.
        preferred_columns = [
            "activity_id",
            "project_id",
            "phase",
            "resource_type",
            "activity_name",
            "activity_type",
            "delay_prediction_days",
            "delay_probability",
            "total_float",
            "criticality_weight",
            "float_impact",
            "risk_score",
            "computed_critical",
            "ES",
            "EF",
            "LS",
            "LF",
        ]

        final_columns = [
            col for col in preferred_columns
            if col in merged.columns
        ]

        results = merged[final_columns].copy()

        # Rank: highest risk = rank 1.
        results = results.sort_values(
            "risk_score",
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)

        results["risk_rank"] = np.arange(1, len(results) + 1)

        # Percentile expressed as 0-100.
        if len(results) > 1:
            results["risk_percentile"] = (
                results["risk_score"].rank(
                    method="average",
                    pct=True,
                )
                * 100.0
            )
        else:
            results["risk_percentile"] = 100.0

        # Round numeric fields for readable CSV output.
        for col in [
            "delay_prediction_days",
            "delay_probability",
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

        self.results = results

        print("\n6. Building results DataFrame...")
        print("   Ranking complete")

        print("\n   Top 10 highest-risk activities:")

        display_columns = [
            col
            for col in [
                "activity_id",
                "project_id",
                "risk_score",
                "delay_prediction_days",
                "total_float",
                "risk_rank",
            ]
            if col in results.columns
        ]

        print(
            results[display_columns]
            .head(10)
            .to_string(index=False)
        )

        print("\n7. Ranking activities...")
        print("=" * 70)
        print("RISK SCORE CALCULATION COMPLETE")
        print("=" * 70)

        return results

    # ------------------------------------------------------------------
    # Summary / save
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a compact risk-score summary."""
        if self.results is None:
            raise RuntimeError(
                "Run calculate_risk_scores() before summary()."
            )

        scores = self.results["risk_score"]

        q75 = float(scores.quantile(0.75))
        q95 = float(scores.quantile(0.95))

        return {
            "total_activities": int(len(self.results)),
            "mean_risk": float(scores.mean()),
            "std_risk": float(scores.std()),
            "min_risk": float(scores.min()),
            "max_risk": float(scores.max()),
            "high_risk_threshold": q75,
            "critical_risk_threshold": q95,
            "high_risk_activities": int((scores > q75).sum()),
            "critical_risk_activities": int((scores > q95).sum()),
        }

    def print_summary(self) -> None:
        """Print risk-score summary."""
        summary = self.summary()

        print("\nRisk Score Summary:")
        print(f"  Total activities: {summary['total_activities']}")
        print(f"  Mean risk: {summary['mean_risk']:.4f}")
        print(f"  Std risk: {summary['std_risk']:.4f}")
        print(f"  Min risk: {summary['min_risk']:.4f}")
        print(f"  Max risk: {summary['max_risk']:.4f}")
        print(
            "  High-risk activities (>75th percentile): "
            f"{summary['high_risk_activities']}"
        )
        print(
            "  Critical-risk activities (>95th percentile): "
            f"{summary['critical_risk_activities']}"
        )

        print("\nTop 10 Highest-Risk Activities:")

        display_columns = [
            col
            for col in [
                "activity_id",
                "project_id",
                "delay_prediction_days",
                "risk_score",
                "risk_rank",
            ]
            if col in self.results.columns
        ]

        print(
            self.results[display_columns]
            .head(10)
            .to_string(index=False)
        )

    def save(
        self,
        output_path: Path | str | None = None,
    ) -> Path:
        """Save risk scores to CSV."""
        if self.results is None:
            raise RuntimeError(
                "Run calculate_risk_scores() before save()."
            )

        if output_path is None:
            output_path = self.processed_dir / "risk_scores.csv"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(output_path, index=False)

        print(f"\nRisk scores saved to {output_path}")

        return output_path


def main() -> None:
    print("=" * 70)
    print("PHASE 5: RISK SCORING TEST")
    print("=" * 70)

    scorer = RiskScorer()

    scorer.calculate_risk_scores()
    scorer.print_summary()
    scorer.save()

    print("\n" + "=" * 70)
    print("PHASE 5 RISK SCORING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
