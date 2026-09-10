"""
Leakage-safe feature engineering for Phase 4 ML.

Builds a deterministic activity-level feature matrix from SCOPE v0.2 processed data.

Every feature is derived only from prediction-time-available sources:
- project planning attributes
- activity planned attributes
- dependency / network structure
- resource definitions
- planned resource allocation
- planned procurement information

Excluded by design:
- events and event-derived fields
- target_event_delay_days (attached only as the target column)
- rework outcomes
- decision actuals
- procurement actuals
- observed delay fields
- construction_memory outcome fields
- future environment / activity-state observations
- UNKNOWN fields
- OUTCOME / POST-OUTCOME fields
- activities.critical_path / critical_path_position (undocumented methodology)

The output preserves the existing project-level split:
- 70 train projects
- 15 validation projects
- 15 test projects
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ============================================================================
# Feature registry
# ============================================================================

# Ordered list of engineered feature column names.
# This ordering is deterministic and used for the final matrix column order.
FEATURE_COLUMN_ORDER = [
    # Activity-level planned features
    "activity_planned_duration_days",
    "activity_quantity",
    "activity_unit_cost",
    "activity_planned_cost",
    "activity_phase_order",
    "activity_sequence",
    "activity_predecessor_count",
    "activity_successor_count",
    # Activity categoricals (encoded later by the model pipeline)
    "activity_phase",
    "activity_resource_type",
    # Dependency / network features
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
    # Project-level planning attributes
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
    # Project-level structural aggregates
    "project_activity_count",
    "project_dependency_count",
    "project_mean_planned_activity_duration",
    "project_median_planned_activity_duration",
    "project_fs_count",
    "project_ss_count",
    "project_ff_count",
    # Resource features
    "resource_allocated_count",
    "resource_labour_count",
    "resource_equipment_count",
    "resource_material_count",
    "resource_total_allocation_fraction",
    "resource_mean_availability",
    "resource_mean_capacity",
    # Procurement features (planned side only)
    "procurement_planned_lead_days",
    "procurement_record_count",
    "procurement_mean_planned_lead_days",
    "procurement_max_planned_lead_days",
    "procurement_supply_chain_risk",
]

# Columns that are categorical and should be encoded before modeling.
CATEGORICAL_FEATURE_NAMES = [
    "activity_phase",
    "activity_resource_type",
    "project_type",
]

# Numeric feature names = all feature columns minus categoricals.
NUMERIC_FEATURE_NAMES = [c for c in FEATURE_COLUMN_ORDER if c not in CATEGORICAL_FEATURE_NAMES]

# Columns that must never appear in X, even if they exist in the source data.
EXCLUDED_INPUT_COLUMNS = {
    "target_event_delay_days",
    "critical_path",
    "critical_path_position",
    "actual_lead_days",
    "procurement_delay_days",
    "actual_decision_day",
    "decision_delay_days",
    "decision_status",
    "rework_days",
    "rework_cost",
    "duration_days",
    "event_type",
    "severity",
    "impact_factor",
    "productivity_index",
    "event_pressure",
    "weather_risk",
    "site_access_index",
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "extreme_weather",
    "observed_delay_days",
}

# Only these source tables are used.
USED_SOURCE_TABLES = {
    "activities",
    "dependencies",
    "projects",
    "resources",
    "resource_allocation",
    "procurement",
    "targets_event_delay_days",
}

EXPECTED_NUM_ACTIVITIES = 9279
EXPECTED_NUM_FEATURES = 55


# ============================================================================
# CSV loaders
# ============================================================================

def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV as list of string dicts (matches Phase 2 processed format)."""
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================================
# Deterministic helpers
# ============================================================================

def median_sorted(values: list[float]) -> float:
    """Median of a sorted list of numbers."""
    if not values:
        return 0.0
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0


# ============================================================================
# Feature builders
# ============================================================================

def compute_network_features(
    activity_ids: set[str],
    dependency_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Compute per-activity dependency/network features.

    Returns:
        {activity_id: {feature_name: value}}
    """
    # Initialize all activity structures
    features: dict[str, dict[str, Any]] = {}
    for aid in activity_ids:
        features[aid] = {
            "incoming_fs": 0,
            "incoming_ss": 0,
            "incoming_ff": 0,
            "outgoing_fs": 0,
            "outgoing_ss": 0,
            "outgoing_ff": 0,
            "incoming_lag_sum": 0,
            "outgoing_lag_sum": 0,
            "incoming_lag_max": -1,
            "incoming_lag_min": -1,
            "outgoing_lag_max": -1,
            "outgoing_lag_min": -1,
        }

    for row in dependency_rows:
        pred = row["predecessor_id"]
        succ = row["successor_id"]
        rel = row["relationship"]
        lag = int(row["lag_days"])

        if pred not in activity_ids or succ not in activity_ids:
            continue

        if rel == "FS":
            features[pred]["outgoing_fs"] += 1
            features[succ]["incoming_fs"] += 1
            features[pred]["outgoing_lag_sum"] += lag
            features[succ]["incoming_lag_sum"] += lag
            if features[pred]["outgoing_lag_max"] < 0 or lag > features[pred]["outgoing_lag_max"]:
                features[pred]["outgoing_lag_max"] = lag
            if features[succ]["incoming_lag_max"] < 0 or lag > features[succ]["incoming_lag_max"]:
                features[succ]["incoming_lag_max"] = lag
            if features[pred]["outgoing_lag_min"] < 0 or lag < features[pred]["outgoing_lag_min"]:
                features[pred]["outgoing_lag_min"] = lag
            if features[succ]["incoming_lag_min"] < 0 or lag < features[succ]["incoming_lag_min"]:
                features[succ]["incoming_lag_min"] = lag
        elif rel == "SS":
            features[pred]["outgoing_ss"] += 1
            features[succ]["incoming_ss"] += 1
            features[pred]["outgoing_lag_sum"] += lag
            features[succ]["incoming_lag_sum"] += lag
            if features[pred]["outgoing_lag_max"] < 0 or lag > features[pred]["outgoing_lag_max"]:
                features[pred]["outgoing_lag_max"] = lag
            if features[succ]["incoming_lag_max"] < 0 or lag > features[succ]["incoming_lag_max"]:
                features[succ]["incoming_lag_max"] = lag
            if features[pred]["outgoing_lag_min"] < 0 or lag < features[pred]["outgoing_lag_min"]:
                features[pred]["outgoing_lag_min"] = lag
            if features[succ]["incoming_lag_min"] < 0 or lag < features[succ]["incoming_lag_min"]:
                features[succ]["incoming_lag_min"] = lag
        elif rel == "FF":
            features[pred]["outgoing_ff"] += 1
            features[succ]["incoming_ff"] += 1
            features[pred]["outgoing_lag_sum"] += lag
            features[succ]["incoming_lag_sum"] += lag
            if features[pred]["outgoing_lag_max"] < 0 or lag > features[pred]["outgoing_lag_max"]:
                features[pred]["outgoing_lag_max"] = lag
            if features[succ]["incoming_lag_max"] < 0 or lag > features[succ]["incoming_lag_max"]:
                features[succ]["incoming_lag_max"] = lag
            if features[pred]["outgoing_lag_min"] < 0 or lag < features[pred]["outgoing_lag_min"]:
                features[pred]["outgoing_lag_min"] = lag
            if features[succ]["incoming_lag_min"] < 0 or lag < features[succ]["incoming_lag_min"]:
                features[succ]["incoming_lag_min"] = lag

    # Convert to final feature dict
    result: dict[str, dict[str, Any]] = {}
    for aid, f in features.items():
        result[aid] = {
            "net_incoming_fs_count": f["incoming_fs"],
            "net_incoming_ss_count": f["incoming_ss"],
            "net_incoming_ff_count": f["incoming_ff"],
            "net_outgoing_fs_count": f["outgoing_fs"],
            "net_outgoing_ss_count": f["outgoing_ss"],
            "net_outgoing_ff_count": f["outgoing_ff"],
            "net_incoming_total_lag": f["incoming_lag_sum"],
            "net_outgoing_total_lag": f["outgoing_lag_sum"],
            "net_incoming_max_lag": max(f["incoming_lag_max"], 0),
            "net_incoming_min_lag": max(f["incoming_lag_min"], 0),
            "net_has_incoming_ss": 1 if f["incoming_ss"] > 0 else 0,
            "net_has_incoming_ff": 1 if f["incoming_ff"] > 0 else 0,
            "net_has_outgoing_ss": 1 if f["outgoing_ss"] > 0 else 0,
            "net_has_outgoing_ff": 1 if f["outgoing_ff"] > 0 else 0,
        }
    return result


def compute_project_features(
    project_ids: set[str],
    project_rows: list[dict[str, str]],
    activity_rows: list[dict[str, str]],
    dependency_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Compute project-level planning + structural aggregate features.

    Aggregates are computed from planned data only.
    """
    # Project planning attributes
    proj_plan: dict[str, dict[str, Any]] = {}
    for row in project_rows:
        pid = row["project_id"]
        proj_plan[pid] = {
            "project_floors": int(row["floors"]),
            "project_area_m2": float(row["area_m2"]),
            "project_complexity": float(row["complexity"]),
            "project_contractor_capability": float(row["contractor_capability"]),
            "project_resource_availability": float(row["resource_availability"]),
            "project_management_maturity": float(row["management_maturity"]),
            "project_weather_exposure": float(row["weather_exposure"]),
            "project_supply_chain_exposure": float(row["supply_chain_exposure"]),
            "project_technology_maturity": float(row["technology_maturity"]),
            "project_planned_duration_days": int(row["planned_duration_days"]),
            "project_planned_cost": float(row["planned_cost"]),
            "project_type": row["project_type"],
        }

    # Activity counts and planned duration stats per project
    proj_activities: dict[str, list[int]] = defaultdict(list)
    for row in activity_rows:
        pid = row["project_id"]
        proj_activities[pid].append(int(row["planned_duration_days"]))

    # Dependency counts per project
    proj_dep_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"FS": 0, "SS": 0, "FF": 0})
    for row in dependency_rows:
        pid = row["project_id"]
        rel = row["relationship"]
        proj_dep_counts[pid][rel] += 1

    result: dict[str, dict[str, Any]] = {}
    for pid in project_ids:
        plan = proj_plan.get(pid, {})
        durations = sorted(proj_activities.get(pid, []))
        dep = proj_dep_counts.get(pid, {"FS": 0, "SS": 0, "FF": 0})

        result[pid] = {
            **plan,
            "project_activity_count": len(durations),
            "project_dependency_count": sum(dep.values()),
            "project_mean_planned_activity_duration": (
                float(sum(durations)) / len(durations) if durations else 0.0
            ),
            "project_median_planned_activity_duration": median_sorted(durations),
            "project_fs_count": dep["FS"],
            "project_ss_count": dep["SS"],
            "project_ff_count": dep["FF"],
        }
    return result


def compute_resource_features(
    activity_ids: set[str],
    allocation_rows: list[dict[str, str]],
    resource_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Compute per-activity resource features from planned allocations.

    Returns:
        {activity_id: {feature_name: value}}
    """
    # Resource lookup: resource_id -> attributes
    resource_attrs: dict[str, dict[str, Any]] = {}
    for row in resource_rows:
        rid = row["resource_id"]
        resource_attrs[rid] = {
            "resource_category": row["resource_category"],
            "availability": float(row["availability"]),
            "capacity": float(row["capacity"]),
        }

    # Aggregate per activity
    agg: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "labour": 0,
        "equipment": 0,
        "material": 0,
        "allocation_sum": 0.0,
        "availability_sum": 0.0,
        "availability_count": 0,
        "capacity_sum": 0.0,
        "capacity_count": 0,
    })

    for row in allocation_rows:
        aid = row["activity_id"]
        if aid not in activity_ids:
            continue
        rid = row["resource_id"]
        cat = row["resource_category"]
        frac = float(row["allocation_fraction"])

        a = agg[aid]
        a["count"] += 1
        a["allocation_sum"] += frac
        if rid in resource_attrs:
            ra = resource_attrs[rid]
            a["availability_sum"] += ra["availability"]
            a["availability_count"] += 1
            a["capacity_sum"] += ra["capacity"]
            a["capacity_count"] += 1
            if cat == "Labour":
                a["labour"] += 1
            elif cat == "Equipment":
                a["equipment"] += 1
            elif cat == "Material":
                a["material"] += 1

    result: dict[str, dict[str, Any]] = {}
    for aid, a in agg.items():
        result[aid] = {
            "resource_allocated_count": a["count"],
            "resource_labour_count": a["labour"],
            "resource_equipment_count": a["equipment"],
            "resource_material_count": a["material"],
            "resource_total_allocation_fraction": a["allocation_sum"],
            "resource_mean_availability": (
                a["availability_sum"] / a["availability_count"] if a["availability_count"] else 0.0
            ),
            "resource_mean_capacity": (
                a["capacity_sum"] / a["capacity_count"] if a["capacity_count"] else 0.0
            ),
        }
    return result


def compute_procurement_features(
    project_ids: set[str],
    procurement_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Compute project-level procurement features from planned procurement only.

    Excludes actual_lead_days and procurement_delay_days.
    """
    proj_records: dict[str, list[float]] = defaultdict(list)
    proj_supply_risk: dict[str, float] = {}

    for row in procurement_rows:
        pid = row["project_id"]
        if pid not in project_ids:
            continue
        planned = float(row["planned_lead_days"])
        proj_records[pid].append(planned)
        # supply_chain_risk is planned-side; take the max per project
        risk = float(row["supply_chain_risk"])
        if pid not in proj_supply_risk or risk > proj_supply_risk[pid]:
            proj_supply_risk[pid] = risk

    result: dict[str, dict[str, Any]] = {}
    for pid in project_ids:
        recs = proj_records.get(pid, [])
        result[pid] = {
            "procurement_record_count": len(recs),
            "procurement_planned_lead_days": recs[0] if recs else 0.0,
            "procurement_mean_planned_lead_days": (
                float(sum(recs)) / len(recs) if recs else 0.0
            ),
            "procurement_max_planned_lead_days": max(recs) if recs else 0.0,
            "procurement_supply_chain_risk": proj_supply_risk.get(pid, 0.0),
        }
    return result


# ============================================================================
# Main feature matrix builder
# ============================================================================

BuildResult = dict[str, Any]


def build_feature_matrix(
    processed_dir: Path,
) -> BuildResult:
    """Build the leakage-safe Phase 4 feature matrix.

    Args:
        processed_dir: path to data/processed/

    Returns:
        A dict containing:
        - X: list of dicts, each dict = one activity row keyed by feature name
        - y: list of floats (target_event_delay_days per activity)
        - activity_ids: list of (project_id, activity_id) in same order as X/y
        - split_assignments: dict {(project_id, activity_id): split}
        - project_splits: dict {project_id: split}
        - feature_names: list of feature column names (deterministic order)
        - categorical_feature_names: list of categorical feature names
        - numeric_feature_names: list of numeric feature names
        - metadata: dict with build info
    """
    processed_dir = Path(processed_dir)

    # Load source tables
    activities = load_csv_rows(processed_dir / "activities.csv")
    dependencies = load_csv_rows(processed_dir / "dependencies.csv")
    projects = load_csv_rows(processed_dir / "projects.csv")
    resources = load_csv_rows(processed_dir / "resources.csv")
    allocations = load_csv_rows(processed_dir / "resource_allocation.csv")
    procurement = load_csv_rows(processed_dir / "procurement.csv")
    targets = load_csv_rows(processed_dir / "targets_event_delay_days.csv")

    # Build project -> split map
    project_splits: dict[str, str] = {}
    for row in projects:
        project_splits[row["project_id"]] = row["split"]

    project_ids = set(project_splits.keys())

    # Activity-level lookup
    activity_by_key: dict[tuple[str, str], dict[str, str]] = {}
    activity_ids: set[str] = set()
    for row in activities:
        key = (row["project_id"], row["activity_id"])
        activity_by_key[key] = row
        activity_ids.add(row["activity_id"])

    # Per-project activity id sets (for dependency joins)
    project_activity_ids: dict[str, set[str]] = defaultdict(set)
    for row in activities:
        project_activity_ids[row["project_id"]].add(row["activity_id"])

    # Target lookup
    target_by_key: dict[tuple[str, str], float] = {}
    for row in targets:
        key = (row["project_id"], row["activity_id"])
        target_by_key[key] = float(row["target_event_delay_days"])

    # Network features (per activity, within each project)
    network_features: dict[str, dict[str, Any]] = {}
    for pid, aids in project_activity_ids.items():
        proj_deps = [r for r in dependencies if r["project_id"] == pid]
        net = compute_network_features(aids, proj_deps)
        network_features.update(net)

    # Project features
    project_features = compute_project_features(
        project_ids, projects, activities, dependencies
    )

    # Resource features
    resource_features = compute_resource_features(activity_ids, allocations, resources)

    # Procurement features
    procurement_features = compute_procurement_features(project_ids, procurement)

    # Assemble matrix
    X: list[dict[str, Any]] = []
    y: list[float] = []
    activity_ids_list: list[tuple[str, str]] = []
    split_assignments: dict[tuple[str, str], str] = {}

    # Deterministic ordering: sort by (project_id, activity_id)
    sorted_keys = sorted(activity_by_key.keys(), key=lambda k: (k[0], k[1]))

    for (pid, aid) in sorted_keys:
        act = activity_by_key[(pid, aid)]
        proj_feat = project_features[pid]
        res_feat = resource_features.get(aid, {})
        net_feat = network_features.get(aid, {})
        proc_feat = procurement_features[pid]

        row: dict[str, Any] = {}

        # Activity-level planned features
        row["activity_planned_duration_days"] = int(act["planned_duration_days"])
        row["activity_quantity"] = float(act["quantity"])
        row["activity_unit_cost"] = float(act["unit_cost"])
        row["activity_planned_cost"] = float(act["planned_cost"])
        row["activity_phase_order"] = int(act["phase_order"])
        row["activity_sequence"] = int(act["activity_sequence"])
        row["activity_predecessor_count"] = int(act["predecessor_count"])
        row["activity_successor_count"] = int(act["successor_count"])
        row["activity_phase"] = act["phase"]
        row["activity_resource_type"] = act["resource_type"]

        # Network features
        row.update(net_feat)

        # Project features
        row.update(proj_feat)

        # Resource features
        row.update(res_feat)

        # Procurement features
        row.update(proc_feat)

        X.append(row)
        y.append(target_by_key.get((pid, aid), 0.0))
        activity_ids_list.append((pid, aid))
        split_assignments[(pid, aid)] = project_splits[pid]

    # Metadata
    metadata: dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "num_activities": len(X),
        "num_features": len(FEATURE_COLUMN_ORDER),
        "feature_names": FEATURE_COLUMN_ORDER,
        "categorical_feature_names": CATEGORICAL_FEATURE_NAMES,
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "project_counts": dict(Counter(project_splits.values())),
        "excluded_input_columns": sorted(EXCLUDED_INPUT_COLUMNS),
        "used_source_tables": sorted(USED_SOURCE_TABLES),
        "build_timestamp": hashlib.sha256(
            str(sorted_keys).encode()
        ).hexdigest()[:16],  # deterministic fingerprint, not a real timestamp
    }

    return {
        "X": X,
        "y": y,
        "activity_ids": activity_ids_list,
        "split_assignments": split_assignments,
        "project_splits": project_splits,
        "feature_names": FEATURE_COLUMN_ORDER,
        "categorical_feature_names": CATEGORICAL_FEATURE_NAMES,
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "metadata": metadata,
    }


# ============================================================================
# Convenience accessors
# ============================================================================

def split_activity_indices(
    result: BuildResult,
) -> dict[str, list[int]]:
    """Return {split: [row indices]} for train/validation/test."""
    out: dict[str, list[int]] = defaultdict(list)
    for i, (pid, _) in enumerate(result["activity_ids"]):
        split = result["split_assignments"][(pid, _)]
        out[split].append(i)
    return dict(out)
