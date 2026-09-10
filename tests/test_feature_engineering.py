"""
Tests for src/ml/feature_engineering.py

These tests verify:
- Deterministic feature matrix construction
- No leakage features in X
- Correct target attachment
- Project-level split integrity
- Feature counts and shapes
- Exclusion of critical_path / critical_path_position
- Presence of required feature groups
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.feature_engineering import (
    CATEGORICAL_FEATURE_NAMES,
    EXCLUDED_INPUT_COLUMNS,
    FEATURE_COLUMN_ORDER,
    NUMERIC_FEATURE_NAMES,
    BuildResult,
    build_feature_matrix,
    split_activity_indices,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def processed_dir() -> Path:
    return REPO_ROOT / "data" / "processed"


@pytest.fixture
def result(processed_dir: Path) -> BuildResult:
    return build_feature_matrix(processed_dir)


# ============================================================================
# Build sanity
# ============================================================================

def test_build_returns_expected_keys(result: BuildResult) -> None:
    assert "X" in result
    assert "y" in result
    assert "activity_ids" in result
    assert "split_assignments" in result
    assert "project_splits" in result
    assert "feature_names" in result
    assert "metadata" in result


def test_feature_matrix_shape(result: BuildResult) -> None:
    assert len(result["X"]) == len(result["y"])
    assert len(result["X"]) == len(result["activity_ids"])
    assert result["metadata"]["num_activities"] == len(result["X"])


def test_feature_column_order_is_deterministic(result: BuildResult) -> None:
    assert result["feature_names"] == FEATURE_COLUMN_ORDER


def test_categorical_and_numeric_lists(result: BuildResult) -> None:
    assert set(result["categorical_feature_names"]) == set(CATEGORICAL_FEATURE_NAMES)
    assert set(result["numeric_feature_names"]) == set(NUMERIC_FEATURE_NAMES)
    assert len(result["categorical_feature_names"]) + len(result["numeric_feature_names"]) == len(
        result["feature_names"]
    )


def test_metadata_has_project_counts(result: BuildResult) -> None:
    counts = result["metadata"]["project_counts"]
    assert counts["train"] == 70
    assert counts["validation"] == 15
    assert counts["test"] == 15


# ============================================================================
# Split integrity
# ============================================================================

def test_project_splits_match_processed_data(result: BuildResult) -> None:
    # Load project splits from processed data
    with open(result["metadata"]["processed_dir"] + "/projects.csv") as f:
        projects = list(csv.DictReader(f))
    expected = {r["project_id"]: r["split"] for r in projects}

    assert result["project_splits"] == expected


def test_activity_split_assignment(result: BuildResult) -> None:
    for (pid, aid), split in result["split_assignments"].items():
        assert split in {"train", "validation", "test"}
        assert result["project_splits"][pid] == split


def test_train_validation_test_disjoint(result: BuildResult) -> None:
    project_to_splits: dict[str, set[str]] = defaultdict(set)
    for (pid, _), split in result["split_assignments"].items():
        project_to_splits[pid].add(split)

    for pid, splits in project_to_splits.items():
        assert len(splits) == 1, f"Project {pid} appears in multiple splits: {splits}"


def test_split_indices_cover_all_activities(result: BuildResult) -> None:
    indices_by_split = split_activity_indices(result)
    all_indices = []
    for idx_list in indices_by_split.values():
        all_indices.extend(idx_list)

    assert sorted(all_indices) == list(range(len(result["X"])))


def test_split_counts_match_metadata(result: BuildResult) -> None:
    # Each project contributes all its activities to exactly one split
    split_activity_counts: dict[str, int] = Counter()
    for split in result["split_assignments"].values():
        split_activity_counts[split] += 1

    # Verify counts are plausible
    assert split_activity_counts["train"] > 0
    assert split_activity_counts["validation"] > 0
    assert split_activity_counts["test"] > 0
    assert split_activity_counts["train"] + split_activity_counts["validation"] + split_activity_counts["test"] == len(
        result["X"]
    )


# ============================================================================
# Leakage tests
# ============================================================================

def test_target_not_in_X(result: BuildResult) -> None:
    """CRITICAL: target_event_delay_days must not be in X."""
    all_columns = set()
    for row in result["X"]:
        all_columns.update(row.keys())

    assert "target_event_delay_days" not in all_columns


def test_excluded_columns_not_in_X(result: BuildResult) -> None:
    """None of the known excluded columns may appear in X."""
    all_columns = set()
    for row in result["X"]:
        all_columns.update(row.keys())

    for col in EXCLUDED_INPUT_COLUMNS:
        assert col not in all_columns, f"Excluded column {col} found in X"


def test_critical_path_not_in_X(result: BuildResult) -> None:
    """critical_path and critical_path_position must not be in X."""
    all_columns = set()
    for row in result["X"]:
        all_columns.update(row.keys())

    assert "critical_path" not in all_columns
    assert "critical_path_position" not in all_columns


def test_x_columns_match_feature_names(result: BuildResult) -> None:
    """Every X row should have exactly the registered feature columns."""
    expected = set(FEATURE_COLUMN_ORDER)
    for i, row in enumerate(result["X"]):
        actual = set(row.keys())
        assert actual == expected, f"Row {i} has columns {sorted(actual - expected)} missing {sorted(expected - actual)}"


def test_no_events_in_features(result: BuildResult) -> None:
    """Feature names must not reference event-derived concepts."""
    feature_names = result["feature_names"]
    event_terms = {"event_type", "event_id", "severity", "impact_factor"}
    for name in feature_names:
        lower = name.lower()
        for term in event_terms:
            assert term not in lower, f"Event-derived term '{term}' found in feature '{name}'"


def test_no_rework_in_features(result: BuildResult) -> None:
    feature_names = result["feature_names"]
    for name in feature_names:
        assert "rework" not in name.lower(), f"Rework term found in feature '{name}'"


def test_no_procurement_actuals_in_features(result: BuildResult) -> None:
    feature_names = result["feature_names"]
    for name in feature_names:
        assert "actual_lead" not in name.lower()
        assert "procurement_delay" not in name.lower()


def test_no_observed_delay_in_features(result: BuildResult) -> None:
    feature_names = result["feature_names"]
    for name in feature_names:
        assert "observed_delay" not in name.lower()


def test_no_outcome_terms_in_features(result: BuildResult) -> None:
    feature_names = result["feature_names"]
    outcome_terms = {"outcome"}
    for name in feature_names:
        lower = name.lower()
        for term in outcome_terms:
            assert term not in lower, f"Outcome term '{term}' found in feature '{name}'"


def _resolve_target_match(result: BuildResult) -> list[tuple[int, float, int]]:
    """Return list of (expected, actual, mismatch_magnitude) for each activity."""
    processed_dir = Path(result["metadata"]["processed_dir"])
    with open(processed_dir / "targets_event_delay_days.csv") as f:
        targets = list(csv.DictReader(f))
    target_by_key = {}
    for row in targets:
        key = (row["project_id"], row["activity_id"])
        target_by_key[key] = float(row["target_event_delay_days"])
    mismatches = []
    for i, (pid, aid) in enumerate(result["activity_ids"]):
        expected = target_by_key[(pid, aid)]
        actual = result["y"][i]
        if abs(expected - actual) > 1e-9:
            mismatches.append((i, expected, actual, abs(expected - actual)))
    return mismatches


def test_target_matches_processed_data(result: BuildResult) -> None:
    """Verify y values match the processed target file exactly."""
    mismatches = _resolve_target_match(result)
    assert not mismatches, f"{len(mismatches)} target mismatches found: {mismatches[:5]}"


@pytest.fixture
def _target_rows(result: BuildResult) -> list[dict[str, str]]:
    """Load target rows for tests that need them."""
    processed_dir = Path(result["metadata"]["processed_dir"])
    with open(processed_dir / "targets_event_delay_days.csv") as f:
        return list(csv.DictReader(f))


# ============================================================================
# Feature group presence tests
# ============================================================================

def test_activity_features_present(result: BuildResult) -> None:
    expected = {
        "activity_planned_duration_days",
        "activity_quantity",
        "activity_unit_cost",
        "activity_planned_cost",
        "activity_phase_order",
        "activity_sequence",
        "activity_predecessor_count",
        "activity_successor_count",
        "activity_phase",
        "activity_resource_type",
    }
    assert expected.issubset(set(result["feature_names"]))


def test_network_features_present(result: BuildResult) -> None:
    expected = {
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
    }
    assert expected.issubset(set(result["feature_names"]))


def test_project_features_present(result: BuildResult) -> None:
    expected = {
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
        "project_type",
        "project_activity_count",
        "project_dependency_count",
        "project_mean_planned_activity_duration",
        "project_median_planned_activity_duration",
        "project_fs_count",
        "project_ss_count",
        "project_ff_count",
    }
    assert expected.issubset(set(result["feature_names"]))


def test_resource_features_present(result: BuildResult) -> None:
    expected = {
        "resource_allocated_count",
        "resource_labour_count",
        "resource_equipment_count",
        "resource_material_count",
        "resource_total_allocation_fraction",
        "resource_mean_availability",
        "resource_mean_capacity",
    }
    assert expected.issubset(set(result["feature_names"]))


def test_procurement_features_present(result: BuildResult) -> None:
    expected = {
        "procurement_planned_lead_days",
        "procurement_record_count",
        "procurement_mean_planned_lead_days",
        "procurement_max_planned_lead_days",
        "procurement_supply_chain_risk",
    }
    assert expected.issubset(set(result["feature_names"]))


# ============================================================================
# Feature value sanity
# ============================================================================

def test_feature_values_are_finite(result: BuildResult) -> None:
    for i, row in enumerate(result["X"]):
        for name, val in row.items():
            if isinstance(val, (int, float)):
                assert math.isfinite(val), f"Row {i} feature {name} = {val} is not finite"


def test_no_nan_in_features(result: BuildResult) -> None:
    for i, row in enumerate(result["X"]):
        for name, val in row.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"Row {i} feature {name} is NaN"


def test_planned_duration_positive(result: BuildResult) -> None:
    for row in result["X"]:
        assert row["activity_planned_duration_days"] > 0
        assert row["project_planned_duration_days"] > 0


def test_network_counts_are_nonnegative(result: BuildResult) -> None:
    count_fields = [
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
        "resource_allocated_count",
        "resource_labour_count",
        "resource_equipment_count",
        "resource_material_count",
        "project_activity_count",
        "project_dependency_count",
        "project_fs_count",
        "project_ss_count",
        "project_ff_count",
        "procurement_record_count",
    ]
    for row in result["X"]:
        for name in count_fields:
            assert row[name] >= 0, f"Negative value in {name}: {row[name]}"


def test_allocation_fraction_bounded(result: BuildResult) -> None:
    for row in result["X"]:
        af = row["resource_total_allocation_fraction"]
        assert 0.0 <= af <= 999.0, f"Unreasonable allocation fraction: {af}"


# ============================================================================
# Categorical feature values
# ============================================================================

def test_phase_values_known(result: BuildResult) -> None:
    valid_phases = {
        "Commissioning",
        "Earthwork",
        "External Works",
        "Finishing",
        "Foundation",
        "MEP",
        "Masonry",
        "Preconstruction",
        "Site Preparation",
        "Structural Frame",
    }
    for row in result["X"]:
        assert row["activity_phase"] in valid_phases, f"Unknown phase: {row['activity_phase']}"


def test_resource_type_values_known(result: BuildResult) -> None:
    valid_types = {"labour", "equipment", "information", "management"}
    for row in result["X"]:
        assert row["activity_resource_type"] in valid_types, f"Unknown resource_type: {row['activity_resource_type']}"


def test_project_type_values_known(result: BuildResult) -> None:
    valid_types = {"Commercial", "Hospital", "Industrial", "Institutional", "Residential"}
    for row in result["X"]:
        assert row["project_type"] in valid_types, f"Unknown project_type: {row['project_type']}"


# ============================================================================
# Reproducibility
# ============================================================================

def test_build_is_deterministic(result: BuildResult) -> None:
    """Running build_feature_matrix twice produces identical results."""
    processed_dir = Path(result["metadata"]["processed_dir"])
    result2 = build_feature_matrix(processed_dir)

    assert result["X"] == result2["X"]
    assert result["y"] == result2["y"]
    assert result["activity_ids"] == result2["activity_ids"]
    assert result["split_assignments"] == result2["split_assignments"]
    assert result["project_splits"] == result2["project_splits"]
    assert result["feature_names"] == result2["feature_names"]
    assert result["metadata"]["num_activities"] == result2["metadata"]["num_activities"]


def test_activity_ids_sorted(result: BuildResult) -> None:
    """Activity IDs should be in sorted order for deterministic output."""
    for i in range(1, len(result["activity_ids"])):
        assert result["activity_ids"][i - 1] < result["activity_ids"][i]


# ============================================================================
# Missing values
# ============================================================================

def test_no_missing_values_in_features(result: BuildResult) -> None:
    """Every feature cell should be filled (no None, no missing)."""
    missing = 0
    for i, row in enumerate(result["X"]):
        for name in result["feature_names"]:
            val = row[name]
            if val is None:
                missing += 1
    assert missing == 0, f"Found {missing} missing feature values"


# ============================================================================
# Build metadata
# ============================================================================

def test_metadata_contains_required_fields(result: BuildResult) -> None:
    m = result["metadata"]
    required = [
        "processed_dir",
        "num_activities",
        "num_features",
        "feature_names",
        "categorical_feature_names",
        "numeric_feature_names",
        "project_counts",
        "excluded_input_columns",
        "used_source_tables",
    ]
    for key in required:
        assert key in m, f"metadata missing {key}"


def test_used_source_tables(result: BuildResult) -> None:
    used = set(result["metadata"]["used_source_tables"])
    expected = {"activities", "dependencies", "projects", "resources", "resource_allocation", "procurement", "targets_event_delay_days"}
    assert used == expected, f"Used tables {sorted(used)} != expected {sorted(expected)}"


def test_excluded_input_columns_listed(result: BuildResult) -> None:
    excluded = set(result["metadata"]["excluded_input_columns"])
    assert "target_event_delay_days" in excluded
    assert "critical_path" in excluded
    assert "critical_path_position" in excluded


# ============================================================================
# Activity count matches processed data
# ============================================================================

def test_activity_count_matches_activities_csv(processed_dir: Path, result: BuildResult) -> None:
    with open(processed_dir / "activities.csv") as f:
        activities = list(csv.DictReader(f))
    assert len(result["X"]) == len(activities)


def test_target_count_matches_targets_csv_and_activity_ids_match_keys(
    processed_dir: Path, result: BuildResult
) -> None:
    with open(processed_dir / "targets_event_delay_days.csv") as f:
        targets = list(csv.DictReader(f))
    assert len(result["y"]) == len(targets)
    assert len(result["activity_ids"]) == len(targets)

    # Verify that activity_ids keys match target keys exactly
    target_keys = {(r["project_id"], r["activity_id"]) for r in targets}
    activity_keys = set(result["activity_ids"])
    assert target_keys == activity_keys, (
        f"activity_ids keys != target keys. "
        f"Only in targets: {len(target_keys - activity_keys)}, "
        f"Only in activity_ids: {len(activity_keys - target_keys)}"
    )


# ============================================================================
# Dependency feature coverage
# ============================================================================

def test_activities_without_dependencies_have_zero_network_features(result: BuildResult) -> None:
    """Activities with no dependencies should have zero network features."""
    processed_dir = Path(result["metadata"]["processed_dir"])
    with open(processed_dir / "dependencies.csv") as f:
        deps = list(csv.DictReader(f))
    dep_activities: set[str] = {r["successor_id"] for r in deps} | {r["predecessor_id"] for r in deps}

    for i, (pid, aid) in enumerate(result["activity_ids"]):
        if aid not in dep_activities or result["split_assignments"][(pid, aid)] != result["project_splits"][pid]:
            # Check this specific activity
            row = result["X"][i]
            if row["net_incoming_fs_count"] == 0 and row["net_outgoing_fs_count"] == 0:
                assert row["net_incoming_ss_count"] == 0
                assert row["net_incoming_ff_count"] == 0
                assert row["net_outgoing_ss_count"] == 0
                assert row["net_outgoing_ff_count"] == 0


# ============================================================================
# Project feature consistency
# ============================================================================

def test_project_features_same_for_all_activities_in_project(result: BuildResult) -> None:
    """All activities in the same project should have the same project-level features."""
    project_features: dict[str, dict[str, object]] = {}

    project_level_fields = {
        "project_floors", "project_area_m2", "project_complexity",
        "project_contractor_capability", "project_resource_availability",
        "project_management_maturity", "project_weather_exposure",
        "project_supply_chain_exposure", "project_technology_maturity",
        "project_planned_duration_days", "project_planned_cost", "project_type",
        "project_activity_count", "project_dependency_count",
        "project_mean_planned_activity_duration", "project_median_planned_activity_duration",
        "project_fs_count", "project_ss_count", "project_ff_count",
        "procurement_planned_lead_days", "procurement_record_count",
        "procurement_mean_planned_lead_days", "procurement_max_planned_lead_days",
        "procurement_supply_chain_risk",
    }

    for i, (pid, _) in enumerate(result["activity_ids"]):
        row = {k: result["X"][i][k] for k in project_level_fields}
        if pid not in project_features:
            project_features[pid] = row
        else:
            assert project_features[pid] == row, f"Project {pid} has inconsistent project features"


# ============================================================================
# End-to-end: verify feature counts
# ============================================================================

def test_total_feature_count(result: BuildResult) -> None:
    assert len(result["feature_names"]) == len(FEATURE_COLUMN_ORDER)
