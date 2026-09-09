"""
Phase 4 - ML Data Preparation

Prepares engineered construction features for machine learning.

Pipeline:
1. Create feature matrix using FeatureEngineer
2. Split data using existing project-level split
3. Preserve project_id and activity_id as metadata
4. Fit StandardScaler ONLY on training features
5. Transform train/validation/test features
6. Save datasets, identifiers, scaler, feature names and metadata

IMPORTANT:
- project_id and activity_id are NOT ML features.
- They are saved only so Phase 5 can match predictions
  with CPM results.
"""

import sys
from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


from src.features.engineering import FeatureEngineer


# =========================================================
# CONFIGURATION
# =========================================================

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml"
)


# =========================================================
# ML DATA PREPARER
# =========================================================

class MLDataPreparer:
    """Prepare train, validation and test datasets."""

    def __init__(
        self,
        raw_data_dir=RAW_DATA_DIR,
        output_dir=OUTPUT_DIR,
    ):

        self.raw_data_dir = Path(raw_data_dir)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # CREATE FEATURE MATRIX
    # =====================================================

    def create_feature_matrix(self):
        """Create engineered ML feature matrix."""

        print("\n[1/7] Creating feature matrix...")

        engineer = FeatureEngineer(
            data_dir=str(self.raw_data_dir)
        )

        (
            X,
            y,
            project_split,
        ) = engineer.create_feature_matrix()

        feature_names = engineer.get_feature_names()

        print(
            f"Feature matrix created: {X.shape}"
        )

        print(
            f"Target vector: {y.shape}"
        )

        print(
            f"Number of features: {len(feature_names)}"
        )

        return (
            X,
            y,
            project_split,
            feature_names,
        )

    # =====================================================
    # PROJECT-LEVEL SPLIT
    # =====================================================

    def split_data(
        self,
        X,
        y,
        project_split,
    ):
        """
        Split data using the existing project-level split.

        Activities belonging to the same project remain
        in the same dataset.

        Also preserves:
            project_id
            activity_id

        as metadata for Phase 5.
        """

        print(
            "\n[2/7] Splitting into "
            "train/validation/test..."
        )

        split_info = project_split.copy()

        # -------------------------------------------------
        # Check project_id
        # -------------------------------------------------

        if "project_id" not in split_info.columns:

            raise ValueError(
                "project_split must contain project_id."
            )

        # -------------------------------------------------
        # Check split column
        # -------------------------------------------------

        if "split" not in split_info.columns:

            raise ValueError(
                "No 'split' column found. "
                "Expected project-level split information."
            )

        # -------------------------------------------------
        # Remove duplicate projects
        # -------------------------------------------------

        split_info = split_info.drop_duplicates(
            subset=["project_id"]
        )

        # -------------------------------------------------
        # Normalize split names
        # -------------------------------------------------

        split_info["split"] = (
            split_info["split"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # -------------------------------------------------
        # Valid split names
        # -------------------------------------------------

        valid_splits = {
            "train",
            "training",
            "val",
            "valid",
            "validation",
            "test",
            "testing",
        }

        invalid = (
            set(split_info["split"].unique())
            - valid_splits
        )

        if invalid:

            raise ValueError(
                f"Unknown split labels: {invalid}"
            )

        # -------------------------------------------------
        # Convert split names
        # -------------------------------------------------

        split_info["standard_split"] = (
            split_info["split"].replace(
                {
                    "training": "train",
                    "valid": "val",
                    "validation": "val",
                    "testing": "test",
                }
            )
        )

        # =================================================
        # LOAD ACTIVITY IDENTIFIERS
        # =================================================

        activities_path = (
            self.raw_data_dir
            / "activities.csv"
        )

        activities = pd.read_csv(
            activities_path,
            usecols=[
                "project_id",
                "activity_id",
            ],
        )

        # -------------------------------------------------
        # Verify alignment
        # -------------------------------------------------

        if len(activities) != len(X):

            raise ValueError(
                "Activities and feature matrix have "
                "different row counts."
            )

        print(
            f"  Activity identifiers loaded: "
            f"{len(activities)}"
        )

        # =================================================
        # MAP EACH ACTIVITY TO PROJECT SPLIT
        # =================================================

        activity_split = activities.merge(
            split_info[
                [
                    "project_id",
                    "standard_split",
                ]
            ],
            on="project_id",
            how="left",
            validate="many_to_one",
        )

        # -------------------------------------------------
        # Check missing projects
        # -------------------------------------------------

        if activity_split[
            "standard_split"
        ].isna().any():

            missing_projects = (
                activity_split.loc[
                    activity_split[
                        "standard_split"
                    ].isna(),
                    "project_id",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Some projects do not have a split: "
                f"{missing_projects[:10]}"
            )

        # =================================================
        # CREATE MASKS
        # =================================================

        train_mask = (
            activity_split[
                "standard_split"
            ]
            == "train"
        )

        val_mask = (
            activity_split[
                "standard_split"
            ]
            == "val"
        )

        test_mask = (
            activity_split[
                "standard_split"
            ]
            == "test"
        )

        # =================================================
        # SPLIT FEATURES
        # =================================================

        X_train = X.loc[
            train_mask
        ].copy()

        X_val = X.loc[
            val_mask
        ].copy()

        X_test = X.loc[
            test_mask
        ].copy()

        # =================================================
        # SPLIT TARGET
        # =================================================

        y_train = y.loc[
            train_mask
        ].copy()

        y_val = y.loc[
            val_mask
        ].copy()

        y_test = y.loc[
            test_mask
        ].copy()

        # =================================================
        # SPLIT IDENTIFIERS
        # =================================================

        ids_train = activities.loc[
            train_mask
        ].copy()

        ids_val = activities.loc[
            val_mask
        ].copy()

        ids_test = activities.loc[
            test_mask
        ].copy()

        # -------------------------------------------------
        # Reset indexes
        # -------------------------------------------------

        X_train.reset_index(
            drop=True,
            inplace=True,
        )

        X_val.reset_index(
            drop=True,
            inplace=True,
        )

        X_test.reset_index(
            drop=True,
            inplace=True,
        )

        y_train.reset_index(
            drop=True,
            inplace=True,
        )

        y_val.reset_index(
            drop=True,
            inplace=True,
        )

        y_test.reset_index(
            drop=True,
            inplace=True,
        )

        ids_train.reset_index(
            drop=True,
            inplace=True,
        )

        ids_val.reset_index(
            drop=True,
            inplace=True,
        )

        ids_test.reset_index(
            drop=True,
            inplace=True,
        )

        # =================================================
        # PROJECT SETS
        # =================================================

        train_projects = set(
            activity_split.loc[
                train_mask,
                "project_id",
            ]
        )

        val_projects = set(
            activity_split.loc[
                val_mask,
                "project_id",
            ]
        )

        test_projects = set(
            activity_split.loc[
                test_mask,
                "project_id",
            ]
        )

        # =================================================
        # LEAKAGE CHECK
        # =================================================

        train_val_overlap = (
            train_projects
            & val_projects
        )

        train_test_overlap = (
            train_projects
            & test_projects
        )

        val_test_overlap = (
            val_projects
            & test_projects
        )

        if (
            train_val_overlap
            or train_test_overlap
            or val_test_overlap
        ):

            raise ValueError(
                "DATA LEAKAGE DETECTED: "
                "projects appear in multiple splits."
            )

        # =================================================
        # PRINT SUMMARY
        # =================================================

        print(
            f"  Train: {len(X_train)} activities"
        )

        print(
            f"  Val:   {len(X_val)} activities"
        )

        print(
            f"  Test:  {len(X_test)} activities"
        )

        print(
            f"  Total: "
            f"{len(X_train) + len(X_val) + len(X_test)} "
            f"activities"
        )

        print(
            f"  Train projects: "
            f"{len(train_projects)}"
        )

        print(
            f"  Val projects:   "
            f"{len(val_projects)}"
        )

        print(
            f"  Test projects:  "
            f"{len(test_projects)}"
        )

        print(
            "  No data leakage: "
            "splits are project-isolated"
        )

        print(
            "  Activity IDs preserved "
            "for Phase 5 risk scoring"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            ids_train,
            ids_val,
            ids_test,
            train_projects,
            val_projects,
            test_projects,
        )

    # =====================================================
    # SCALE DATA
    # =====================================================

    def scale_data(
        self,
        X_train,
        X_val,
        X_test,
    ):
        """
        Fit StandardScaler only on training data.

        Validation and test data are transformed using
        the training-fitted scaler.
        """

        print(
            "\n[3/7] Fitting StandardScaler "
            "on training data..."
        )

        scaler = StandardScaler()

        # IMPORTANT:
        # Fit ONLY on training data.

        scaler.fit(X_train)

        print(
            f"  Fitting scaler on "
            f"{X_train.shape[1]} features"
        )

        print(
            f"  Training data size: "
            f"{len(X_train)} activities"
        )

        print(
            "  Scaler fitted successfully"
        )

        # =================================================
        # TRANSFORM DATA
        # =================================================

        print(
            "\n[4/7] Applying scaler "
            "to train/val/test..."
        )

        X_train_scaled = scaler.transform(
            X_train
        )

        X_val_scaled = scaler.transform(
            X_val
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # -------------------------------------------------
        # Convert back to DataFrames
        # -------------------------------------------------

        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
        )

        X_val_scaled = pd.DataFrame(
            X_val_scaled,
            columns=X_val.columns,
        )

        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
        )

        print(
            f"  TRAIN: "
            f"{X_train_scaled.shape} scaled"
        )

        print(
            f"  VAL:   "
            f"{X_val_scaled.shape} scaled"
        )

        print(
            f"  TEST:  "
            f"{X_test_scaled.shape} scaled"
        )

        return (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            scaler,
        )

    # =====================================================
    # ADD IDENTIFIERS
    # =====================================================

    def add_identifiers(
        self,
        X_train,
        X_val,
        X_test,
        ids_train,
        ids_val,
        ids_test,
    ):
        """
        Add project_id and activity_id as metadata.

        IMPORTANT:
        These columns are NOT ML features.

        They are saved only to identify predictions
        during Phase 5 risk scoring.
        """

        print(
            "\n[5/7] Adding activity identifiers..."
        )

        # -------------------------------------------------
        # Add IDs to each dataset
        # -------------------------------------------------

        X_train_output = pd.concat(
            [
                ids_train.reset_index(drop=True),
                X_train.reset_index(drop=True),
            ],
            axis=1,
        )

        X_val_output = pd.concat(
            [
                ids_val.reset_index(drop=True),
                X_val.reset_index(drop=True),
            ],
            axis=1,
        )

        X_test_output = pd.concat(
            [
                ids_test.reset_index(drop=True),
                X_test.reset_index(drop=True),
            ],
            axis=1,
        )

        # -------------------------------------------------
        # Verify identifier columns
        # -------------------------------------------------

        required_ids = [
            "project_id",
            "activity_id",
        ]

        for dataframe_name, dataframe in [
            ("X_train", X_train_output),
            ("X_val", X_val_output),
            ("X_test", X_test_output),
        ]:

            for column in required_ids:

                if column not in dataframe.columns:

                    raise ValueError(
                        f"{dataframe_name} is missing "
                        f"{column}"
                    )

        # -------------------------------------------------
        # Verify row counts
        # -------------------------------------------------

        if len(X_train_output) != len(X_train):

            raise ValueError(
                "X_train identifier alignment error."
            )

        if len(X_val_output) != len(X_val):

            raise ValueError(
                "X_val identifier alignment error."
            )

        if len(X_test_output) != len(X_test):

            raise ValueError(
                "X_test identifier alignment error."
            )

        print(
            "  project_id added"
        )

        print(
            "  activity_id added"
        )

        print(
            "  IDs are metadata only "
            "(not ML features)"
        )

        print(
            f"  X_train output: "
            f"{X_train_output.shape}"
        )

        print(
            f"  X_val output:   "
            f"{X_val_output.shape}"
        )

        print(
            f"  X_test output:  "
            f"{X_test_output.shape}"
        )

        return (
            X_train_output,
            X_val_output,
            X_test_output,
        )

    # =====================================================
    # SAVE DATASETS
    # =====================================================

    def save_datasets(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ):
        """Save processed datasets."""

        print(
            "\n[6/7] Saving datasets..."
        )

        # -------------------------------------------------
        # Save X datasets
        # -------------------------------------------------

        X_train.to_csv(
            self.output_dir / "X_train.csv",
            index=False,
        )

        X_val.to_csv(
            self.output_dir / "X_val.csv",
            index=False,
        )

        X_test.to_csv(
            self.output_dir / "X_test.csv",
            index=False,
        )

        # -------------------------------------------------
        # Save target datasets
        # -------------------------------------------------

        y_train.to_csv(
            self.output_dir / "y_train.csv",
            index=False,
            header=[
                "target_event_delay_days"
            ],
        )

        y_val.to_csv(
            self.output_dir / "y_val.csv",
            index=False,
            header=[
                "target_event_delay_days"
            ],
        )

        y_test.to_csv(
            self.output_dir / "y_test.csv",
            index=False,
            header=[
                "target_event_delay_days"
            ],
        )

        print(
            "  Saved X_train.csv"
        )

        print(
            "  Saved X_val.csv"
        )

        print(
            "  Saved X_test.csv"
        )

        print(
            "  Saved y_train.csv"
        )

        print(
            "  Saved y_val.csv"
        )

        print(
            "  Saved y_test.csv"
        )

        # -------------------------------------------------
        # Verify saved feature count
        # -------------------------------------------------

        expected_feature_count = 64

        actual_train_feature_count = (
            len(X_train.columns)
            - 2
        )

        actual_val_feature_count = (
            len(X_val.columns)
            - 2
        )

        actual_test_feature_count = (
            len(X_test.columns)
            - 2
        )

        if actual_train_feature_count != expected_feature_count:

            raise ValueError(
                "Unexpected X_train feature count: "
                f"{actual_train_feature_count}. "
                f"Expected {expected_feature_count}."
            )

        if actual_val_feature_count != expected_feature_count:

            raise ValueError(
                "Unexpected X_val feature count: "
                f"{actual_val_feature_count}. "
                f"Expected {expected_feature_count}."
            )

        if actual_test_feature_count != expected_feature_count:

            raise ValueError(
                "Unexpected X_test feature count: "
                f"{actual_test_feature_count}. "
                f"Expected {expected_feature_count}."
            )

        print(
            "  Verified: 64 ML features "
            "+ 2 metadata columns"
        )

    # =====================================================
    # SAVE SCALER
    # =====================================================

    def save_scaler(
        self,
        scaler,
    ):
        """Save fitted scaler."""

        scaler_path = (
            self.output_dir
            / "scaler.pkl"
        )

        with open(
            scaler_path,
            "wb",
        ) as file:

            pickle.dump(
                scaler,
                file,
            )

        print(
            "  Saved scaler.pkl"
        )

    # =====================================================
    # SAVE FEATURE NAMES
    # =====================================================

    def save_feature_names(
        self,
        feature_names,
    ):
        """Save ML feature names."""

        path = (
            self.output_dir
            / "feature_names.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                feature_names,
                file,
                indent=2,
            )

        print(
            "  Saved feature_names.json"
        )

    # =====================================================
    # SAVE METADATA
    # =====================================================

    def save_manifest(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
        train_projects,
        val_projects,
        test_projects,
    ):
        """Save dataset metadata."""

        print(
            "\n[7/7] Saving metadata..."
        )

        manifest = {

            "feature_count": len(
                feature_names
            ),

            "metadata_columns": [
                "project_id",
                "activity_id",
            ],

            "total_samples": (
                len(X_train)
                + len(X_val)
                + len(X_test)
            ),

            "train": {

                "X_shape_with_metadata": list(
                    X_train.shape
                ),

                "ML_feature_count": len(
                    feature_names
                ),

                "y_shape": list(
                    y_train.shape
                ),

                "project_count": len(
                    train_projects
                ),

                "target_mean": float(
                    y_train.mean()
                ),

                "target_std": float(
                    y_train.std()
                ),

                "target_min": float(
                    y_train.min()
                ),

                "target_max": float(
                    y_train.max()
                ),
            },

            "validation": {

                "X_shape_with_metadata": list(
                    X_val.shape
                ),

                "ML_feature_count": len(
                    feature_names
                ),

                "y_shape": list(
                    y_val.shape
                ),

                "project_count": len(
                    val_projects
                ),

                "target_mean": float(
                    y_val.mean()
                ),

                "target_std": float(
                    y_val.std()
                ),

                "target_min": float(
                    y_val.min()
                ),

                "target_max": float(
                    y_val.max()
                ),
            },

            "test": {

                "X_shape_with_metadata": list(
                    X_test.shape
                ),

                "ML_feature_count": len(
                    feature_names
                ),

                "y_shape": list(
                    y_test.shape
                ),

                "project_count": len(
                    test_projects
                ),

                "target_mean": float(
                    y_test.mean()
                ),

                "target_std": float(
                    y_test.std()
                ),

                "target_min": float(
                    y_test.min()
                ),

                "target_max": float(
                    y_test.max()
                ),
            },

            "preprocessing": {

                "scaler": "StandardScaler",

                "scaler_fitted_on":
                    "training_data_only",

                "outlier_removal": False,

                "feature_selection": False,
            },

            "data_leakage_check": {

                "train_validation_overlap": 0,

                "train_test_overlap": 0,

                "validation_test_overlap": 0,
            },

            "identifier_policy": {

                "project_id_used_as_ml_feature":
                    False,

                "activity_id_used_as_ml_feature":
                    False,

                "purpose":
                    "Metadata for matching ML predictions "
                    "with CPM results in Phase 5.",
            },

            "files": [

                "X_train.csv",

                "X_val.csv",

                "X_test.csv",

                "y_train.csv",

                "y_val.csv",

                "y_test.csv",

                "scaler.pkl",

                "feature_names.json",

                "ml_data_manifest.json",
            ],
        }

        path = (
            self.output_dir
            / "ml_data_manifest.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                indent=2,
            )

        print(
            "  Saved ml_data_manifest.json"
        )

    # =====================================================
    # RUN COMPLETE PIPELINE
    # =====================================================

    def run(self):
        """Run the complete ML preparation pipeline."""

        print("=" * 70)

        print(
            "PHASE 4: ML DATA PREPARATION"
        )

        print("=" * 70)

        # =================================================
        # STEP 1
        # =================================================

        (
            X,
            y,
            project_split,
            feature_names,
        ) = self.create_feature_matrix()

        # -------------------------------------------------
        # Feature count check
        # -------------------------------------------------

        if X.shape[1] != len(feature_names):

            raise ValueError(
                "Feature count mismatch."
            )

        # =================================================
        # STEP 2
        # =================================================

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            ids_train,
            ids_val,
            ids_test,
            train_projects,
            val_projects,
            test_projects,
        ) = self.split_data(
            X,
            y,
            project_split,
        )

        # =================================================
        # STEP 3 + 4
        # =================================================

        (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            scaler,
        ) = self.scale_data(
            X_train,
            X_val,
            X_test,
        )

        # =================================================
        # STEP 5
        # =================================================

        (
            X_train_output,
            X_val_output,
            X_test_output,
        ) = self.add_identifiers(
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            ids_train,
            ids_val,
            ids_test,
        )

        # =================================================
        # STEP 6
        # =================================================

        self.save_datasets(
            X_train_output,
            X_val_output,
            X_test_output,
            y_train,
            y_val,
            y_test,
        )

        self.save_scaler(
            scaler
        )

        self.save_feature_names(
            feature_names
        )

        # =================================================
        # STEP 7
        # =================================================

        self.save_manifest(
            X_train_output,
            X_val_output,
            X_test_output,
            y_train,
            y_val,
            y_test,
            feature_names,
            train_projects,
            val_projects,
            test_projects,
        )

        # =================================================
        # FINAL SUMMARY
        # =================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "DATASET SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"\nTotal ML features: "
            f"{len(feature_names)}"
        )

        print(
            "Metadata columns: "
            "project_id, activity_id"
        )

        print(
            f"Total activities: "
            f"{len(X)}"
        )

        print("\nTRAIN:")

        print(
            f"  ML features: "
            f"{X_train_scaled.shape}"
        )

        print(
            f"  Output with IDs: "
            f"{X_train_output.shape}"
        )

        print(
            f"  y: "
            f"{y_train.shape}"
        )

        print("\nVALIDATION:")

        print(
            f"  ML features: "
            f"{X_val_scaled.shape}"
        )

        print(
            f"  Output with IDs: "
            f"{X_val_output.shape}"
        )

        print(
            f"  y: "
            f"{y_val.shape}"
        )

        print("\nTEST:")

        print(
            f"  ML features: "
            f"{X_test_scaled.shape}"
        )

        print(
            f"  Output with IDs: "
            f"{X_test_output.shape}"
        )

        print(
            f"  y: "
            f"{y_test.shape}"
        )

        print("\nSCALING:")

        print(
            "  Method: StandardScaler"
        )

        print(
            "  Fitted on: training data only"
        )

        print("\nIDENTIFIERS:")

        print(
            "  project_id: preserved as metadata"
        )

        print(
            "  activity_id: preserved as metadata"
        )

        print(
            "  IDs used as ML features: NO"
        )

        print("\nOUTPUT DIRECTORY:")

        print(
            f"  {self.output_dir}"
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "ML DATA PREPARATION SUCCESSFUL!"
        )

        print(
            "=" * 70
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    preparer = MLDataPreparer()

    preparer.run()