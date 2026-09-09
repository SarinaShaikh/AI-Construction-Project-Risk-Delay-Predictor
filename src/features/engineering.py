"""
Feature engineering for the AI Construction Risk & Delay Predictor.

This module:
1. Loads the raw construction datasets.
2. Extracts activity-level features.
3. Adds project context.
4. Adds resource features.
5. Adds procurement features.
6. Adds environmental features.
7. Creates the final ML feature matrix.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create activity-level features for delay prediction."""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.feature_names = []

    # ---------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------

    def _load_csv(self, filename):
        """Load a CSV file from the data directory."""

        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        return pd.read_csv(path)

    def load_data(self):
        """Load all datasets required for feature engineering."""

        data = {
            "activities": self._load_csv("activities.csv"),
            "projects": self._load_csv("projects.csv"),
            "resources": self._load_csv("resources.csv"),
            "resource_allocation": self._load_csv(
                "resource_allocation.csv"
            ),
            "procurement": self._load_csv("procurement.csv"),
            "environment": self._load_csv("environment.csv"),
            "activity_states": self._load_csv(
                "activity_states.csv"
            ),
            "events": self._load_csv("events.csv"),
        }

        return data

    # ---------------------------------------------------------
    # ACTIVITY FEATURES
    # ---------------------------------------------------------

    def extract_activity_features(self, activities_df):
        """
        Extract activity-level features.

        Includes:
        - planned duration
        - planned cost
        - quantity
        - phase
        - resource type
        - criticality
        - CPM-related activity information
        """

        df = activities_df.copy()

        columns = [
            "project_id",
            "activity_id",
            "phase",
            "resource_type",
            "planned_duration_days",
            "quantity",
            "unit_cost",
            "planned_cost",
            "criticality",
            "critical_path",
            "critical_path_position",
            "project_network_duration",
            "predecessor_count",
            "successor_count",
        ]

        available = [
            column
            for column in columns
            if column in df.columns
        ]

        return df[available].copy()

    # ---------------------------------------------------------
    # PROJECT FEATURES
    # ---------------------------------------------------------

    def extract_project_context(self, projects_df):
        """
        Extract project-level context.

        Project-level values are later joined to activities
        using project_id.
        """

        columns = [
            "project_id",
            "project_type",
            "floors",
            "area_m2",
            "complexity",
            "contractor_capability",
            "resource_availability",
            "management_maturity",
            "weather_exposure",
            "supply_chain_exposure",
            "technology_maturity",
            "planned_duration_days",
            "planned_cost",
        ]

        available = [
            column
            for column in columns
            if column in projects_df.columns
        ]

        return projects_df[available].copy()

    # ---------------------------------------------------------
    # RESOURCE FEATURES
    # ---------------------------------------------------------

    def extract_resource_features(
        self,
        activities_df,
        resource_allocation_df,
        resources_df,
    ):
        """
        Calculate activity-level resource features.

        Features:
        - resource_count
        - allocation_mean
        - allocation_max
        - allocation_total
        - available_resource_count
        - resource_scarcity
        """

        allocation = resource_allocation_df.copy()
        resources = resources_df.copy()

        # -----------------------------------------------------
        # Handle empty resource allocation data
        # -----------------------------------------------------

        if allocation.empty:
            return pd.DataFrame(
                columns=[
                    "project_id",
                    "activity_id",
                    "resource_count",
                    "allocation_mean",
                    "allocation_max",
                    "allocation_total",
                    "available_resource_count",
                    "resource_scarcity",
                ]
            )

        # -----------------------------------------------------
        # Calculate resource usage for each activity
        # -----------------------------------------------------

        result = (
            allocation.groupby(
                ["project_id", "activity_id"]
            )
            .agg(
                resource_count=(
                    "resource_id",
                    "nunique",
                ),
                allocation_mean=(
                    "allocation_fraction",
                    "mean",
                ),
                allocation_max=(
                    "allocation_fraction",
                    "max",
                ),
                allocation_total=(
                    "allocation_fraction",
                    "sum",
                ),
            )
            .reset_index()
        )

        # -----------------------------------------------------
        # Count available resources in each project
        # -----------------------------------------------------

        if not resources.empty:
            resource_counts = (
                resources.groupby("project_id")[
                    "resource_id"
                ]
                .nunique()
                .reset_index(
                    name="available_resource_count"
                )
            )

            result = result.merge(
                resource_counts,
                on="project_id",
                how="left",
            )

        else:
            result["available_resource_count"] = 0

        # -----------------------------------------------------
        # Calculate resource scarcity
        # -----------------------------------------------------

        result["resource_scarcity"] = (
            result["resource_count"]
            / result["available_resource_count"].replace(
                0,
                np.nan,
            )
        )

        return result

    # ---------------------------------------------------------
    # PROCUREMENT FEATURES
    # ---------------------------------------------------------

    def extract_procurement_features(self, procurement_df):
        """
        Create project-level procurement features.

        Procurement is project-level in the supplied dataset,
        so the statistics are aggregated by project.
        """

        if procurement_df.empty:
            return pd.DataFrame(
                columns=["project_id"]
            )

        result = (
            procurement_df.groupby("project_id")
            .agg(
                procurement_count=(
                    "procurement_id",
                    "count",
                ),
                avg_planned_lead_days=(
                    "planned_lead_days",
                    "mean",
                ),
                avg_actual_lead_days=(
                    "actual_lead_days",
                    "mean",
                ),
                total_procurement_delay_days=(
                    "procurement_delay_days",
                    "sum",
                ),
                avg_procurement_delay_days=(
                    "procurement_delay_days",
                    "mean",
                ),
                avg_supply_chain_risk=(
                    "supply_chain_risk",
                    "mean",
                ),
            )
            .reset_index()
        )

        return result

    # ---------------------------------------------------------
    # ENVIRONMENT FEATURES
    # ---------------------------------------------------------

    def extract_environment_features(
        self,
        environment_df,
        activity_states_df,
    ):
        """
        Create environmental features.

        Activity states contain activity-level environmental
        observations, so they are preferred for activity-level
        features.
        """

        if activity_states_df.empty:
            return pd.DataFrame(
                columns=["project_id", "activity_id"]
            )

        result = (
            activity_states_df.groupby(
                ["project_id", "activity_id"]
            )
            .agg(
                avg_productivity_index=(
                    "productivity_index",
                    "mean",
                ),
                avg_weather_risk=(
                    "weather_risk",
                    "mean",
                ),
                avg_site_access_index=(
                    "site_access_index",
                    "mean",
                ),
                avg_event_pressure=(
                    "event_pressure",
                    "mean",
                ),
            )
            .reset_index()
        )

        # -----------------------------------------------------
        # Add project-level environmental information
        # -----------------------------------------------------

        if not environment_df.empty:

            env_project = (
                environment_df.groupby("project_id")
                .agg(
                    avg_rainfall_mm=(
                        "rainfall_mm",
                        "mean",
                    ),
                    avg_temperature_c=(
                        "temperature_c",
                        "mean",
                    ),
                    avg_humidity_pct=(
                        "humidity_pct",
                        "mean",
                    ),
                    avg_environment_weather_risk=(
                        "weather_risk",
                        "mean",
                    ),
                    extreme_weather_days=(
                        "extreme_weather",
                        "sum",
                    ),
                )
                .reset_index()
            )

            result = result.merge(
                env_project,
                on="project_id",
                how="left",
            )

        return result

    # ---------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------

    def create_target(self, activities_df):
        """
        Create the regression target.

        Target:
            target_event_delay_days

        This is calculated from events associated with each
        activity.
        """

        events = self._load_csv("events.csv")

        if events.empty:
            return pd.DataFrame(
                columns=[
                    "project_id",
                    "activity_id",
                    "target_event_delay_days",
                ]
            )

        target = (
            events.groupby(
                ["project_id", "activity_id"]
            )["duration_days"]
            .sum()
            .reset_index()
        )

        target = target.rename(
            columns={
                "duration_days":
                    "target_event_delay_days"
            }
        )

        return target

    # ---------------------------------------------------------
    # FEATURE MATRIX
    # ---------------------------------------------------------

    def create_feature_matrix(self):
        """
        Create the final ML feature matrix.

        Returns:
            X             -> feature DataFrame
            y             -> target Series
            project_split -> project split information
        """

        data = self.load_data()

        # -----------------------------------------------------
        # Extract individual feature groups
        # -----------------------------------------------------

        activities = self.extract_activity_features(
            data["activities"]
        )

        projects = self.extract_project_context(
            data["projects"]
        )

        resources = self.extract_resource_features(
            data["activities"],
            data["resource_allocation"],
            data["resources"],
        )

        procurement = self.extract_procurement_features(
            data["procurement"]
        )

        environment = self.extract_environment_features(
            data["environment"],
            data["activity_states"],
        )

        target = self.create_target(
            data["activities"]
        )

        # -----------------------------------------------------
        # Merge activity + project
        # -----------------------------------------------------

        df = activities.merge(
            projects,
            on="project_id",
            how="left",
            suffixes=("", "_project"),
        )

        # -----------------------------------------------------
        # Merge resources
        # -----------------------------------------------------

        df = df.merge(
            resources,
            on=["project_id", "activity_id"],
            how="left",
        )

        # -----------------------------------------------------
        # Merge procurement
        # -----------------------------------------------------

        df = df.merge(
            procurement,
            on="project_id",
            how="left",
        )

        # -----------------------------------------------------
        # Merge environment
        # -----------------------------------------------------

        df = df.merge(
            environment,
            on=["project_id", "activity_id"],
            how="left",
        )

        # -----------------------------------------------------
        # Merge target
        # -----------------------------------------------------

        df = df.merge(
            target,
            on=["project_id", "activity_id"],
            how="left",
        )

        # -----------------------------------------------------
        # Missing target = no recorded event delay
        # -----------------------------------------------------

        df["target_event_delay_days"] = (
            df["target_event_delay_days"]
            .fillna(0)
        )

        # -----------------------------------------------------
        # Keep project split separately
        # -----------------------------------------------------

        project_split = df[
            ["project_id"]
        ].copy()

        if "split" in data["activities"].columns:

            split_df = data["activities"][
                ["project_id", "split"]
            ].drop_duplicates(
                subset=["project_id"]
            )

            project_split = project_split.merge(
                split_df,
                on="project_id",
                how="left",
            )

        # -----------------------------------------------------
        # Create target
        # -----------------------------------------------------

        y = df[
            "target_event_delay_days"
        ].copy()

        # -----------------------------------------------------
        # Remove identifiers and target from X
        # -----------------------------------------------------

        X = df.drop(
            columns=[
                "target_event_delay_days",
                "project_id",
                "activity_id",
            ],
            errors="ignore",
        )

        # split is metadata, not a model feature

        X = X.drop(
            columns=["split"],
            errors="ignore",
        )

        # -----------------------------------------------------
        # Encode categorical features
        # -----------------------------------------------------

        categorical_columns = (
            X.select_dtypes(
                include=["object", "category"]
            )
            .columns
            .tolist()
        )

        if categorical_columns:

            X = pd.get_dummies(
                X,
                columns=categorical_columns,
                dummy_na=True,
            )

        # -----------------------------------------------------
        # Convert boolean columns to integers
        # -----------------------------------------------------

        bool_columns = X.select_dtypes(
            include=["bool"]
        ).columns

        for column in bool_columns:
            X[column] = X[column].astype(int)

        # -----------------------------------------------------
        # Convert remaining values to numeric
        # -----------------------------------------------------

        for column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        # -----------------------------------------------------
        # Handle infinite values
        # -----------------------------------------------------

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # -----------------------------------------------------
        # Handle missing values
        # -----------------------------------------------------

        X = X.fillna(0)

        # -----------------------------------------------------
        # Store feature names
        # -----------------------------------------------------

        self.feature_names = X.columns.tolist()

        return X, y, project_split

    # ---------------------------------------------------------
    # FEATURE NAMES
    # ---------------------------------------------------------

    def get_feature_names(self):
        """Return the final feature names."""

        return self.feature_names

    # ---------------------------------------------------------
    # FEATURE SUMMARY
    # ---------------------------------------------------------

    def get_feature_summary(self):
        """Return a summary of the engineered features."""

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "dtype": [
                    "numeric"
                    for _ in self.feature_names
                ],
            }
        )