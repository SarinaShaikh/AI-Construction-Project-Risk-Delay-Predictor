#!/usr/bin/env python3
"""
SCOPE v0.2 — Phase 2 Data Cleaning Pipeline.

Reads ONLY from data/raw/SCOPE_v02_Public/ (never writes there), applies
deterministic, documented cleaning + validation to the 12 core downstream
tables, constructs the activity-level event-delay target, and writes results
ONLY to data/processed/.

Outputs (data/processed/):
    <table>.csv                        cleaned copy of each processed table
    targets_event_delay_days.csv       one row per activity: SUM(events.duration_days)
    prediction_time_availability.csv   per-field leakage classification
    cleaning_manifest.json             full audit trail (transformations + validations)

Exit code:
    0  all validations PASS and outputs written
    1  any validation FAIL (no processed CSVs written; manifest still written)

Usage:
    python scripts/clean_scope.py
    python scripts/clean_scope.py --raw-dir <dir> --processed-dir <dir>   # custom dirs
    python scripts/clean_scope.py --no-write                              # validate only

The script is deterministic: no random operations, original row order is
preserved, and CSV outputs are written with LF line endings.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "SCOPE_v02_Public"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Table registry — the actual schema, verified against the raw files (Phase 1/2).
# `kind` maps every column to: id / date / int / float / cat / split
# ---------------------------------------------------------------------------

TABLE_SPECS = {
    "projects": {
        "pk": ["project_id"],
        "cols": {
            "project_id": "id",
            "project_type": "cat",
            "floors": "int",
            "area_m2": "float",
            "complexity": "float",
            "contractor_capability": "float",
            "resource_availability": "float",
            "management_maturity": "float",
            "weather_exposure": "float",
            "supply_chain_exposure": "float",
            "technology_maturity": "float",
            "planned_duration_days": "int",
            "planned_cost": "float",
            "split": "split",
        },
    },
    "activities": {
        "pk": ["project_id", "activity_id"],
        "cols": {
            "project_id": "id",
            "activity_id": "id",
            "phase": "cat",
            "phase_order": "int",
            "activity_sequence": "int",
            "activity_name": "cat",
            "resource_type": "cat",
            "planned_duration_days": "int",
            "quantity": "float",
            "unit_cost": "float",
            "planned_cost": "float",
            "criticality": "float",
            "status": "cat",
            "critical_path": "int",
            "critical_path_position": "int",
            "project_network_duration": "int",
            "predecessor_count": "int",
            "successor_count": "int",
            "split": "split",
        },
    },
    "dependencies": {
        "pk": ["project_id", "predecessor_id", "successor_id"],
        "cols": {
            "project_id": "id",
            "predecessor_id": "id",
            "successor_id": "id",
            "relationship": "cat",
            "lag_days": "int",
            "split": "split",
        },
    },
    "resources": {
        "pk": ["resource_id"],
        "cols": {
            "resource_id": "id",
            "project_id": "id",
            "resource_category": "cat",
            "resource_type": "cat",
            "capacity": "float",
            "availability": "float",
            "daily_cost": "float",
            "split": "split",
        },
    },
    "resource_allocation": {
        "pk": ["project_id", "activity_id", "resource_id"],
        "cols": {
            "project_id": "id",
            "activity_id": "id",
            "resource_id": "id",
            "resource_category": "cat",
            "allocation_fraction": "float",
            "split": "split",
        },
    },
    "environment": {
        "pk": ["project_id", "day_index"],
        "cols": {
            "project_id": "id",
            "date": "date",
            "day_index": "int",
            "rainfall_mm": "float",
            "temperature_c": "float",
            "humidity_pct": "float",
            "site_access_index": "float",
            "extreme_weather": "int",
            "weather_risk": "int",
            "split": "split",
        },
    },
    "activity_states": {
        "pk": ["project_id", "activity_id", "day_index"],
        "cols": {
            "project_id": "id",
            "activity_id": "id",
            "date": "date",
            "day_index": "int",
            "productivity_index": "float",
            "weather_risk": "int",
            "site_access_index": "float",
            "event_pressure": "float",
            "split": "split",
        },
    },
    "procurement": {
        "pk": ["procurement_id"],
        "cols": {
            "procurement_id": "id",
            "project_id": "id",
            "material": "cat",
            "planned_lead_days": "int",
            "actual_lead_days": "int",
            "procurement_delay_days": "int",
            "supply_chain_risk": "float",
            "split": "split",
        },
    },
    "events": {
        "pk": ["event_id"],
        "cols": {
            "event_id": "id",
            "project_id": "id",
            "date": "date",
            "activity_id": "id",
            "event_type": "cat",
            "severity": "cat",
            "duration_days": "int",
            "cause_node": "cat",
            "impact_factor": "float",
            "split": "split",
        },
    },
    "decisions": {
        "pk": ["decision_id"],
        "cols": {
            "decision_id": "id",
            "project_id": "id",
            "activity_id": "id",
            "decision_type": "cat",
            "required_day": "int",
            "actual_decision_day": "int",
            "decision_delay_days": "int",
            "criticality": "float",
            "decision_status": "cat",
            "dependency_exposure": "float",
            "time_exposure": "float",
            "reversibility_penalty": "float",
            "decision_debt_score": "float",
            "decision_debt_level": "cat",
            "split": "split",
        },
    },
    "rework": {
        "pk": ["rework_id"],
        "cols": {
            "rework_id": "id",
            "project_id": "id",
            "event_id": "id",
            "activity_id": "id",
            "rework_days": "int",
            "rework_cost": "float",
            "split": "split",
        },
    },
    "outcomes": {
        "pk": ["project_id"],
        "cols": {
            "project_id": "id",
            "total_activities": "int",
            "total_planned_duration": "int",
            "split": "split",
        },
    },
}

# Duration-like fields that must not be negative (validation rule D).
# (value, allow_zero) — zero allowed for lag/lead differences, not for durations.
DURATION_RULES = {
    "projects": [("planned_duration_days", False)],
    "activities": [("planned_duration_days", False)],
    "dependencies": [("lag_days", True)],
    "procurement": [
        ("planned_lead_days", False),
        ("actual_lead_days", False),
        ("procurement_delay_days", True),
    ],
    "events": [("duration_days", False)],
    "decisions": [("decision_delay_days", True)],
    "rework": [("rework_days", False)],
}

# Prediction-time availability classification (STEP 7).
# availability ∈ {AVAILABLE_AT_PREDICTION_TIME, OUTCOME_OR_POST_OUTCOME, UNKNOWN}
AVAILABILITY = {
    "projects": {
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "project_type": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "floors": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "area_m2": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "complexity": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "contractor_capability": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "resource_availability": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "management_maturity": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "weather_exposure": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time risk rating."),
        "supply_chain_exposure": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time risk rating."),
        "technology_maturity": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time project characteristic."),
        "planned_duration_days": ("AVAILABLE_AT_PREDICTION_TIME", "Planned duration is known at planning time."),
        "planned_cost": ("AVAILABLE_AT_PREDICTION_TIME", "Planned cost is known at planning time."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "activities": {
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "activity_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "phase": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time activity attribute."),
        "phase_order": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time activity attribute."),
        "activity_sequence": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time activity attribute."),
        "activity_name": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time activity attribute."),
        "resource_type": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time activity attribute."),
        "planned_duration_days": ("AVAILABLE_AT_PREDICTION_TIME", "Planned duration is known at planning time."),
        "quantity": ("AVAILABLE_AT_PREDICTION_TIME", "Planned quantity."),
        "unit_cost": ("AVAILABLE_AT_PREDICTION_TIME", "Planned unit cost."),
        "planned_cost": ("AVAILABLE_AT_PREDICTION_TIME", "Planned cost."),
        "criticality": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time synthetic attribute."),
        "status": ("AVAILABLE_AT_PREDICTION_TIME", "Constant 'Planned' in SCOPE v0.2; planning metadata."),
        "critical_path": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time flag (synthetic; not yet CPM-validated)."),
        "critical_path_position": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time flag (synthetic; not yet CPM-validated)."),
        "project_network_duration": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time network aggregate."),
        "predecessor_count": ("AVAILABLE_AT_PREDICTION_TIME", "Network structure known at planning time."),
        "successor_count": ("AVAILABLE_AT_PREDICTION_TIME", "Network structure known at planning time."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "dependencies": {
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "predecessor_id": ("AVAILABLE_AT_PREDICTION_TIME", "Network structure known at planning time."),
        "successor_id": ("AVAILABLE_AT_PREDICTION_TIME", "Network structure known at planning time."),
        "relationship": ("AVAILABLE_AT_PREDICTION_TIME", "Network structure known at planning time."),
        "lag_days": ("AVAILABLE_AT_PREDICTION_TIME", "Network structure known at planning time."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "resources": {
        "resource_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "resource_category": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time resource attribute."),
        "resource_type": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time resource attribute."),
        "capacity": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time resource attribute."),
        "availability": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time resource attribute."),
        "daily_cost": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time resource attribute."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "resource_allocation": {
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "activity_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "resource_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "resource_category": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time allocation attribute."),
        "allocation_fraction": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time allocation attribute."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "environment": {
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "date": ("AVAILABLE_AT_PREDICTION_TIME", "Timeline metadata."),
        "day_index": ("AVAILABLE_AT_PREDICTION_TIME", "Timeline metadata."),
        "rainfall_mm": ("UNKNOWN", "Daily observation; exact future values are not knowable at prediction time."),
        "temperature_c": ("UNKNOWN", "Daily observation; exact future values are not knowable at prediction time."),
        "humidity_pct": ("UNKNOWN", "Daily observation; exact future values are not knowable at prediction time."),
        "site_access_index": ("UNKNOWN", "Daily observation; exact future values are not knowable at prediction time."),
        "extreme_weather": ("UNKNOWN", "Daily observation; exact future values are not knowable at prediction time."),
        "weather_risk": ("UNKNOWN", "Daily observation; exact future values are not knowable at prediction time."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "activity_states": {
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "activity_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "date": ("AVAILABLE_AT_PREDICTION_TIME", "Timeline metadata."),
        "day_index": ("AVAILABLE_AT_PREDICTION_TIME", "Timeline metadata."),
        "productivity_index": ("UNKNOWN", "Execution-time observation; not knowable before the activity runs."),
        "weather_risk": ("UNKNOWN", "Execution-time daily observation."),
        "site_access_index": ("UNKNOWN", "Execution-time daily observation."),
        "event_pressure": ("OUTCOME_OR_POST_OUTCOME", "Event-derived daily pressure; same information source as the target."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "procurement": {
        "procurement_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "material": ("AVAILABLE_AT_PREDICTION_TIME", "Planned material type."),
        "planned_lead_days": ("AVAILABLE_AT_PREDICTION_TIME", "Planned lead time is known at planning time."),
        "actual_lead_days": ("OUTCOME_OR_POST_OUTCOME", "Realized lead time; only known after procurement completes."),
        "procurement_delay_days": ("OUTCOME_OR_POST_OUTCOME", "Delay = actual - planned; only known after the fact."),
        "supply_chain_risk": ("AVAILABLE_AT_PREDICTION_TIME", "Planning-time risk rating."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "events": {
        "event_id": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source (SUM(duration_days)); never features."),
        "project_id": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "date": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "activity_id": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "event_type": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "severity": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "duration_days": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "cause_node": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "impact_factor": ("OUTCOME_OR_POST_OUTCOME", "Event records are the target source."),
        "split": ("OUTCOME_OR_POST_OUTCOME", "Event-derived table; not a predictive feature."),
    },
    "decisions": {
        "decision_id": ("UNKNOWN", "Execution-phase record; decision existence is not plan-time knowledge."),
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "activity_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "decision_type": ("UNKNOWN", "Execution-phase decision record; availability depends on prediction point."),
        "required_day": ("UNKNOWN", "Execution-phase decision record; availability depends on prediction point."),
        "actual_decision_day": ("OUTCOME_OR_POST_OUTCOME", "Decision actual; only known after the decision is made."),
        "decision_delay_days": ("OUTCOME_OR_POST_OUTCOME", "Decision actual (delay); only known after the decision is made."),
        "criticality": ("UNKNOWN", "Decision-process attribute; availability depends on prediction point."),
        "decision_status": ("OUTCOME_OR_POST_OUTCOME", "Decision actual (status); only known after the decision is made."),
        "dependency_exposure": ("UNKNOWN", "Decision-process attribute; availability depends on prediction point."),
        "time_exposure": ("UNKNOWN", "Decision-process attribute; availability depends on prediction point."),
        "reversibility_penalty": ("UNKNOWN", "Decision-process attribute; availability depends on prediction point."),
        "decision_debt_score": ("UNKNOWN", "Decision-process attribute; availability depends on prediction point."),
        "decision_debt_level": ("UNKNOWN", "Decision-process attribute; availability depends on prediction point."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
    "rework": {
        "rework_id": ("OUTCOME_OR_POST_OUTCOME", "Rework is a post-event outcome; never a feature."),
        "project_id": ("OUTCOME_OR_POST_OUTCOME", "Rework is a post-event outcome."),
        "event_id": ("OUTCOME_OR_POST_OUTCOME", "Rework is a post-event outcome."),
        "activity_id": ("OUTCOME_OR_POST_OUTCOME", "Rework is a post-event outcome."),
        "rework_days": ("OUTCOME_OR_POST_OUTCOME", "Rework is a post-event outcome."),
        "rework_cost": ("OUTCOME_OR_POST_OUTCOME", "Rework is a post-event outcome."),
        "split": ("OUTCOME_OR_POST_OUTCOME", "Rework-derived table; not a predictive feature."),
    },
    "outcomes": {
        "project_id": ("AVAILABLE_AT_PREDICTION_TIME", "Identifier."),
        "total_activities": ("AVAILABLE_AT_PREDICTION_TIME", "Planned aggregate (activity count), known at planning time."),
        "total_planned_duration": ("AVAILABLE_AT_PREDICTION_TIME", "Planned aggregate, known at planning time."),
        "split": ("AVAILABLE_AT_PREDICTION_TIME", "Dataset partition metadata, not a predictive feature."),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_raw(raw_dir: Path, table: str) -> pd.DataFrame:
    """Read a raw CSV as strings (keep_default_na=False -> '' is the missing marker)."""
    return pd.read_csv(
        raw_dir / f"{table}.csv",
        dtype="str",
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def write_processed(df: pd.DataFrame, processed_dir: Path, table: str) -> None:
    """Write a cleaned table deterministically (LF line endings, original order)."""
    out = processed_dir / f"{table}.csv"
    df.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")


def to_numeric_checked(series: pd.Series, kind: str) -> tuple[pd.Series, pd.Series]:
    """Convert a column to int64/float64; return (converted, invalid_mask)."""
    num = pd.to_numeric(series, errors="coerce")
    invalid = (series != "") & num.isna()
    if kind == "int":
        non_integral = (~num.isna()) & (num != num.round())
        invalid = invalid | non_integral
        if not invalid.any() and not num.isna().all():
            return num.astype("int64"), invalid
    return num, invalid


def check_dates(df: pd.DataFrame, cols: list[str]) -> dict[str, list[str]]:
    """Validate ISO YYYY-MM-DD dates; returns {col: [invalid raw values]}."""
    bad: dict[str, list[str]] = {}
    for c in cols:
        parsed = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
        invalid_mask = parsed.isna() & (df[c] != "")
        if invalid_mask.any():
            bad[c] = df.loc[invalid_mask, c].tolist()
    return bad


# ---------------------------------------------------------------------------
# Validation primitives
# ---------------------------------------------------------------------------


def validate_identifiers_and_duplicates(
    df: pd.DataFrame, spec: dict, table: str
) -> dict:
    """Rule A (identifiers) + Rule B (duplicates). Returns a result dict."""
    pk = spec["pk"]
    missing_ids = {}
    for c in pk:
        n = int((df[c] == "").sum())
        if n:
            missing_ids[c] = n
    pk_dups = int(df.duplicated(subset=pk).sum())
    exact_dup_rows = int(df.duplicated(keep=False).sum())
    before = len(df)
    if exact_dup_rows:
        # Exact duplicate rows are redundant; keep the first occurrence of each.
        df.drop_duplicates(inplace=True)
    removed_exact = before - len(df)
    return {
        "table": table,
        "pk": pk,
        "pk_unique": pk_dups == 0,
        "pk_duplicate_rows": pk_dups,
        "missing_id_cells": missing_ids,
        "exact_duplicate_rows": exact_dup_rows,
        "exact_duplicate_rows_removed": removed_exact,
        "exact_duplicates_handled": exact_dup_rows == 0 or removed_exact == exact_dup_rows,
    }


def validate_missing(df: pd.DataFrame, table: str) -> dict:
    """Rule H — per-column missing (empty-string) counts."""
    missing = {c: int((df[c] == "").sum()) for c in df.columns}
    return {"table": table, "missing_cells": missing, "total_missing": sum(missing.values())}


def validate_split(df: pd.DataFrame, table: str, project_split: pd.Series) -> dict:
    """Every row's split must match its project's split from projects.csv."""
    if "split" not in df.columns:
        return {"table": table, "checked": False}
    exp = df["project_id"].map(project_split)
    mism = int((df["split"] != exp).sum())
    return {"table": table, "checked": True, "mismatches": mism}


def validate_durations(df: pd.DataFrame, table: str) -> dict:
    """Rule D — negative / impossible duration values (reported, never converted)."""
    issues = []
    for col, allow_zero in DURATION_RULES.get(table, []):
        if col not in df.columns:
            continue
        s = df[col]
        num = pd.to_numeric(s, errors="coerce")
        neg = num < 0
        zeros = (num == 0) & ~neg
        if neg.any():
            issues.append(
                {
                    "column": col,
                    "negative_count": int(neg.sum()),
                    "negative_values": s[neg].tolist(),
                }
            )
        if not allow_zero and zeros.any():
            issues.append(
                {
                    "column": col,
                    "zero_count": int(zeros.sum()),
                    "zero_not_allowed": True,
                }
            )
    return {"table": table, "issues": issues, "valid": len(issues) == 0}


def validate_categorical(df: pd.DataFrame, table: str, cat_cols: list[str]) -> dict:
    """Rule G — report distinct values; no normalization performed unless trivially needed."""
    summary = {}
    for c in cat_cols:
        vals = df[c].drop_duplicates().sort_values().tolist()
        summary[c] = {"distinct": len(vals), "values": vals}
    return {"table": table, "categorical_summary": summary}


def kahn_has_cycle(nodes: set[str], edges: list[tuple[str, str]]) -> bool:
    """Return True if the directed graph has a cycle (Kahn's algorithm, O(V+E))."""
    adj: dict[str, list[str]] = collections.defaultdict(list)
    in_deg: dict[str, int] = {n: 0 for n in nodes}
    for pred, succ in edges:
        adj[pred].append(succ)
        in_deg[succ] = in_deg.get(succ, 0) + 1
    queue = collections.deque([n for n, d in in_deg.items() if d == 0])
    processed = 0
    while queue:
        node = queue.popleft()
        processed += 1
        for nb in adj[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    return processed < len(nodes)


def validate_dependencies(
    deps: pd.DataFrame,
    activities: pd.DataFrame,
    activity_index: set[tuple[str, str]],
) -> dict:
    """Rule E — dependency integrity + per-project cycle check."""
    act_proj = set(zip(activities["project_id"], activities["activity_id"]))
    edges_by_project: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    orphan_pred = []
    orphan_succ = []
    cross_project = []
    bad_rel = []
    bad_lag = []

    for row in deps.itertuples(index=False):
        pid, pred, succ, rel, lag = (
            row.project_id,
            row.predecessor_id,
            row.successor_id,
            row.relationship,
            row.lag_days,
        )
        if (pid, pred) not in act_proj:
            orphan_pred.append((pid, pred))
        if (pid, succ) not in act_proj:
            orphan_succ.append((pid, succ))
        if pred.split("_")[0] != pid or succ.split("_")[0] != pid:
            cross_project.append((pid, pred, succ))
        if rel not in {"FS", "FF", "SS"}:
            bad_rel.append((pid, pred, succ, rel))
        if lag < 0:
            bad_lag.append((pid, pred, succ, lag))
        edges_by_project[pid].append((pred, succ))

    # nodes = ALL activities of the project (isolated activities included)
    nodes_by_project: dict[str, set[str]] = collections.defaultdict(set)
    for pid, aid in activity_index:
        nodes_by_project[pid].add(aid)

    cycle_projects = []
    for pid in sorted(nodes_by_project):
        if kahn_has_cycle(nodes_by_project[pid], edges_by_project.get(pid, [])):
            cycle_projects.append(pid)

    # relationship / lag counts
    rel_counts = {str(k): int(v) for k, v in deps["relationship"].value_counts().items()}
    lag_min, lag_max = int(deps["lag_days"].min()), int(deps["lag_days"].max())

    return {
        "edges": len(deps),
        "projects_with_edges": len(edges_by_project),
        "orphan_predecessors": len(orphan_pred),
        "orphan_successors": len(orphan_succ),
        "cross_project_edges": len(cross_project),
        "invalid_relationships": len(bad_rel),
        "relationship_counts": rel_counts,
        "lag_min": lag_min,
        "lag_max": lag_max,
        "negative_lags": len(bad_lag),
        "cycle_projects": cycle_projects,
        "acyclic": len(cycle_projects) == 0,
    }


def build_target(events: pd.DataFrame, activities: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    STEP 6 — target_event_delay_days = SUM(events.duration_days) per (project_id, activity_id),
    0 when an activity has no events. One row per activity, nothing lost.
    """
    sums = (
        events.groupby(["project_id", "activity_id"], as_index=False)["duration_days"]
        .sum()
        .rename(columns={"duration_days": "target_event_delay_days"})
    )
    target = activities[["project_id", "activity_id"]].merge(
        sums, on=["project_id", "activity_id"], how="left"
    )
    target["target_event_delay_days"] = target["target_event_delay_days"].fillna(0).astype("int64")

    n_activities = len(target)
    n_zero = int((target["target_event_delay_days"] == 0).sum())
    n_pos = n_activities - n_zero
    s = target["target_event_delay_days"]
    stats = {
        "n_activities": n_activities,
        "n_zero": n_zero,
        "n_positive": n_pos,
        "zero_pct": round(100.0 * n_zero / n_activities, 4) if n_activities else None,
        "min": int(s.min()),
        "max": int(s.max()),
        "mean": round(float(s.mean()), 4),
        "median": float(s.median()),
        "std": round(float(s.std()), 4),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
    }
    return target, stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SCOPE v0.2 Phase 2 cleaning pipeline")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--no-write", action="store_true", help="validate only; do not write outputs")
    args = parser.parse_args(argv)

    raw_dir: Path = args.raw_dir.resolve()
    processed_dir: Path = args.processed_dir.resolve()
    no_write: bool = args.no_write

    if not raw_dir.is_dir():
        print(f"FATAL: raw directory not found: {raw_dir}", file=sys.stderr)
        return 1
    if processed_dir.is_relative_to(raw_dir):
        print("FATAL: processed dir must not be inside the raw dir.", file=sys.stderr)
        return 1

    print("=" * 78)
    print("SCOPE v0.2 — Phase 2 Data Cleaning Pipeline")
    print("=" * 78)
    print(f"Raw dir:      {raw_dir}")
    print(f"Processed:    {processed_dir}  ({'dry run' if no_write else 'write enabled'})")
    print(f"Python:       {sys.version.split()[0]} | pandas {pd.__version__}")
    print()

    manifest: dict = {
        "meta": {
            "script": "scripts/clean_scope.py",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "notes": (
                "Phase 2 cleaning. Raw files are never modified. "
                "All rows are preserved; no imputation; no fabricated dates."
            ),
        },
        "tables": {},
        "checks": {},
        "target": {},
        "findings": {},
    }

    all_valid = True
    failures: list[str] = []

    def record(table: str, label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_valid
        manifest.setdefault("checks", {}).setdefault(table, {})[label] = {
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }
        if not ok:
            all_valid = False
            failures.append(f"{table}: {label}")

    # ---- projects.csv first: source of truth for split + project ids ----
    print("Reading and cleaning tables...")
    projects = read_raw(raw_dir, "projects")

    tables_clean: dict[str, pd.DataFrame] = {}
    table_results: dict[str, dict] = {}

    for table, spec in TABLE_SPECS.items():
        print(f"  {table} ...", flush=True)
        df = read_raw(raw_dir, table)

        # schema drift guard: actual columns must match the registry
        expected_cols = list(spec["cols"].keys())
        actual_cols = list(df.columns)
        if actual_cols != expected_cols:
            record(table, "schema_match", False, f"expected {expected_cols}, got {actual_cols}")
            df = df[expected_cols]  # align anyway so downstream checks are stable

        res = validate_identifiers_and_duplicates(df, spec, table)
        res["raw_rows"] = int(df.shape[0])
        missing = validate_missing(df, table)
        res["missing"] = missing["missing_cells"]

        # dtype conversion (Rule F) + date validation (Rule C)
        invalid_numeric: dict[str, list] = {}
        for col, kind in spec["cols"].items():
            if kind in ("int", "float"):
                df[col], inv = to_numeric_checked(df[col], kind)
                if inv.any():
                    invalid_numeric[col] = df.loc[inv, col].tolist()
        date_cols = [c for c, k in spec["cols"].items() if k == "date"]
        invalid_dates = check_dates(df, date_cols)

        res["invalid_numeric_cells"] = invalid_numeric
        res["invalid_dates"] = invalid_dates
        res["processed_rows"] = int(df.shape[0])
        res["raw_cols"] = len(actual_cols)
        res["processed_cols"] = len(expected_cols)

        # split consistency
        split_res = validate_split(df, table, projects.set_index("project_id")["split"])
        res["split_mismatches"] = split_res["mismatches"] if split_res["checked"] else 0

        record(table, "primary_key_unique", res["pk_unique"])
        record(table, "no_missing_ids", len(res["missing_id_cells"]) == 0)
        record(table, "exact_duplicates_handled", res["exact_duplicates_handled"])
        record(table, "numeric_values_parse", len(invalid_numeric) == 0)
        record(table, "dates_parse", len(invalid_dates) == 0)
        record(table, "split_consistent", res["split_mismatches"] == 0)

        table_results[table] = res
        tables_clean[table] = df

        print(
            f"    rows {res['raw_rows']} -> {res['processed_rows']} | "
            f"pk_dups={res['pk_duplicate_rows']} exact_dups_removed={res['exact_duplicate_rows_removed']} | "
            f"missing={sum(res['missing'].values())} | "
            f"bad_numeric={sum(len(v) for v in invalid_numeric.values())} "
            f"bad_dates={sum(len(v) for v in invalid_dates.values())}"
        )

    # ---- duration validation (Rule D) ----
    print("\nDuration validation:")
    for table in DURATION_RULES:
        dur = validate_durations(tables_clean[table], table)
        table_results[table]["duration_issues"] = dur["issues"]
        record(table, "durations_valid", dur["valid"], json.dumps(dur["issues"]))
        print(f"  {table}: {'OK' if dur['valid'] else 'ISSUES: ' + json.dumps(dur['issues'])}")

    # ---- categorical summary (Rule G) ----
    for table, spec in TABLE_SPECS.items():
        cat_cols = [c for c, k in spec["cols"].items() if k == "cat"]
        if cat_cols:
            table_results[table]["categorical"] = validate_categorical(
                tables_clean[table], table, cat_cols
            )["categorical_summary"]

    # ---- foreign keys (STEP 8) ----
    print("\nForeign-key validation:")
    activities_df = tables_clean["activities"]
    activity_index = set(zip(activities_df["project_id"], activities_df["activity_id"]))
    project_ids = set(tables_clean["projects"]["project_id"])
    resource_ids = set(tables_clean["resources"]["resource_id"])
    event_ids = set(tables_clean["events"]["event_id"])

    def fk_check(label: str, values: set, universe: set, allow_empty: bool = False) -> None:
        missing = sorted(v for v in values if v not in universe)
        if allow_empty and not values:
            record("fk", label, True, "no rows to check")
            return
        record("fk", label, len(missing) == 0, f"missing={len(missing)} sample={missing[:5]}")

    deps = tables_clean["dependencies"]
    dep_res = validate_dependencies(deps, activities_df, activity_index)
    record("fk", "dependencies.predecessor -> activities", dep_res["orphan_predecessors"] == 0)
    record("fk", "dependencies.successor -> activities", dep_res["orphan_successors"] == 0)
    record("fk", "dependencies intra-project only", dep_res["cross_project_edges"] == 0)
    record("fk", "dependencies relationships valid", dep_res["invalid_relationships"] == 0)
    record("fk", "dependencies lag_days >= 0", dep_res["negative_lags"] == 0)
    record("fk", "dependencies acyclic (per project)", dep_res["acyclic"],
           f"cycle_projects={dep_res['cycle_projects']}")
    table_results["dependencies"]["dependency_graph"] = dep_res

    ra = tables_clean["resource_allocation"]
    fk_check("resource_allocation.(proj,act) -> activities",
             set(zip(ra["project_id"], ra["activity_id"])), activity_index)
    fk_check("resource_allocation.resource_id -> resources", set(ra["resource_id"]), resource_ids)
    env = tables_clean["environment"]
    fk_check("environment.project_id -> projects", set(env["project_id"]), project_ids)
    ast = tables_clean["activity_states"]
    fk_check("activity_states.(proj,act) -> activities",
             set(zip(ast["project_id"], ast["activity_id"])), activity_index)
    pro = tables_clean["procurement"]
    fk_check("procurement.project_id -> projects", set(pro["project_id"]), project_ids)
    dec = tables_clean["decisions"]
    fk_check("decisions.(proj,act) -> activities",
             set(zip(dec["project_id"], dec["activity_id"])), activity_index)
    ev = tables_clean["events"]
    fk_check("events.(proj,act) -> activities",
             set(zip(ev["project_id"], ev["activity_id"])), activity_index)
    rw = tables_clean["rework"]
    fk_check("rework.(proj,act) -> activities",
             set(zip(rw["project_id"], rw["activity_id"])), activity_index)
    fk_check("rework.event_id -> events", set(rw["event_id"]), event_ids)
    out = tables_clean["outcomes"]
    fk_check("outcomes.project_id -> projects", set(out["project_id"]), project_ids)
    fk_check("resources.project_id -> projects", set(tables_clean["resources"]["project_id"]), project_ids)

    # resource_allocation resource category vs resources category (informational)
    ra_res = ra.merge(
        tables_clean["resources"][["resource_id", "resource_category"]],
        on="resource_id",
        suffixes=("_alloc", "_res"),
    )
    cat_mismatch = int((ra_res["resource_category_alloc"] != ra_res["resource_category_res"]).sum())
    record("fk", "resource_allocation.category == resources.category", cat_mismatch == 0)

    print(f"  dependencies: {dep_res['edges']} edges, {dep_res['projects_with_edges']} projects, "
          f"orphans pred={dep_res['orphan_predecessors']} succ={dep_res['orphan_successors']}, "
          f"cross-project={dep_res['cross_project_edges']}, cycles={len(dep_res['cycle_projects'])}")
    for label, entry in manifest["checks"].get("fk", {}).items():
        print(f"  {label}: {entry['status']}")

    # ---- cross-table numeric consistency (informational checks) ----
    print("\nCross-field consistency checks:")
    dec_diff = (
        dec["actual_decision_day"] - dec["required_day"] - dec["decision_delay_days"]
    )
    ok = int((dec_diff != 0).sum()) == 0
    record("consistency", "decision_delay_days == actual_decision_day - required_day", ok,
           f"mismatches={int((dec_diff != 0).sum())}")
    print(f"  decision_delay_days == actual - required: {'PASS' if ok else 'FAIL'}")

    pro_diff = pro["actual_lead_days"] - pro["planned_lead_days"] - pro["procurement_delay_days"]
    ok = int((pro_diff != 0).sum()) == 0
    record("consistency", "procurement_delay_days == actual_lead - planned_lead", ok,
           f"mismatches={int((pro_diff != 0).sum())}")
    print(f"  procurement_delay_days == actual - planned: {'PASS' if ok else 'FAIL'}")

    cost_calc = activities_df["quantity"] * activities_df["unit_cost"]
    rel_diff = ((cost_calc - activities_df["planned_cost"]).abs() / activities_df["planned_cost"])
    n_big = int((rel_diff > 0.001).sum())
    ok = n_big == 0
    record("consistency", "planned_cost ~= quantity * unit_cost", ok,
           f"rows with relative diff > 0.1%: {n_big}; max rel diff {float(rel_diff.max()):.6e}")
    print(f"  planned_cost ~= quantity*unit_cost: {'PASS' if ok else 'FAIL'} (max rel diff {float(rel_diff.max()):.2e})")

    # predecessor/successor counts in activities vs actual edges.
    # predecessor_count == number of incoming edges (activity is the SUCCESSOR).
    # successor_count    == number of outgoing edges (activity is the PREDECESSOR).
    pred_cnt = deps.groupby(["project_id", "successor_id"]).size()
    succ_cnt = deps.groupby(["project_id", "predecessor_id"]).size()
    pred_map = pred_cnt.reset_index(name="n").set_index(["project_id", "successor_id"])["n"]
    succ_map = succ_cnt.reset_index(name="n").set_index(["project_id", "predecessor_id"])["n"]
    act_idx = activities_df.set_index(["project_id", "activity_id"])
    mapped_pred = pd.Series(
        act_idx.index.map(lambda k: pred_map.get(k, 0)).tolist(), index=act_idx.index
    )
    mapped_succ = pd.Series(
        act_idx.index.map(lambda k: succ_map.get(k, 0)).tolist(), index=act_idx.index
    )
    p_mis = int((act_idx["predecessor_count"] != mapped_pred).sum())
    s_mis = int((act_idx["successor_count"] != mapped_succ).sum())
    ok = p_mis == 0 and s_mis == 0
    record("consistency", "activities predecessor/successor counts match dependency edges", ok,
           f"predecessor mismatches={p_mis}, successor mismatches={s_mis}")
    print(f"  activity predecessor/successor counts vs edges: {'PASS' if ok else 'FAIL'}")

    # ---- STEP 5: project split ----
    print("\nProject split:")
    split_counts = {str(k): int(v) for k, v in tables_clean["projects"]["split"].value_counts().items()}
    ok_split = split_counts.get("train") == 70 and split_counts.get("validation") == 15 and split_counts.get("test") == 15
    record("split", "70/15/15 project-level split", ok_split, json.dumps(split_counts))
    print(f"  {split_counts} -> {'PASS' if ok_split else 'FAIL'}")

    # ---- STEP 6: target ----
    print("\nTarget construction (SUM(events.duration_days) per activity):")
    target, target_stats = build_target(ev, activities_df)
    n_act = len(activities_df)
    record("target", "one row per activity", len(target) == n_act,
           f"activities={n_act}, target rows={len(target)}")
    record("target", "no duplicate target rows", int(target.duplicated(["project_id", "activity_id"]).sum()) == 0)
    record("target", "no missing target values", int(target["target_event_delay_days"].isna().sum()) == 0)

    # validate target against the RAW events file (independent recomputation)
    raw_events = read_raw(raw_dir, "events")
    raw_ev_int = pd.to_numeric(raw_events["duration_days"], errors="coerce")
    raw_sums = (
        raw_events.assign(duration_days=raw_ev_int)
        .groupby(["project_id", "activity_id"], as_index=False)["duration_days"]
        .sum()
        .rename(columns={"duration_days": "target_raw"})
    )
    raw_target = activities_df[["project_id", "activity_id"]].merge(
        raw_sums, on=["project_id", "activity_id"], how="left"
    )
    raw_target["target_raw"] = raw_target["target_raw"].fillna(0)
    mismatch = int((target["target_event_delay_days"] != raw_target["target_raw"]).sum())
    record("target", "target matches recomputation from raw events", mismatch == 0,
           f"mismatches={mismatch}")
    print(f"  activities={target_stats['n_activities']} | zero={target_stats['n_zero']} "
          f"({target_stats['zero_pct']}%) | positive={target_stats['n_positive']}")
    print(f"  min={target_stats['min']} max={target_stats['max']} mean={target_stats['mean']} "
          f"median={target_stats['median']} std={target_stats['std']}")
    print(f"  p25={target_stats['p25']} p50={target_stats['p50']} p75={target_stats['p75']} "
          f"p90={target_stats['p90']} p95={target_stats['p95']} p99={target_stats['p99']}")
    manifest["target"] = {"stats": target_stats, "validated_against_raw": mismatch == 0}

    # ---- structural findings (documented, not failures) ----
    env_max = env.groupby("project_id")["day_index"].max()
    pdd = tables_clean["projects"].set_index("project_id")["planned_duration_days"]
    ratio = env_max / pdd
    ast_max = ast.groupby("project_id")["day_index"].max()
    ratio2 = ast_max / pdd
    manifest["findings"] = {
        "environment_timeline": {
            "description": "environment day_index spans ~1.25x planned duration per project (includes a post-planned buffer); structural property of the synthetic dataset.",
            "ratio_min": round(float(ratio.min()), 4),
            "ratio_max": round(float(ratio.max()), 4),
            "ratio_mean": round(float(ratio.mean()), 4),
            "projects_equal_to_planned": int((env_max == pdd).sum()),
        },
        "activity_states_timeline": {
            "description": "activity_states day_index max equals planned duration per project (1.0x); states cover the full planned horizon, not execution windows.",
            "projects_equal_to_planned": int((ast_max == pdd).sum()),
            "ratio_min": round(float(ratio2.min()), 4),
            "ratio_max": round(float(ratio2.max()), 4),
        },
        "no_reliable_actual_dates": (
            "No reliable actual activity start/finish timestamps exist in SCOPE v0.2 "
            "(activities.status = 'Planned' for all rows); event-induced delay is NOT schedule slippage."
        ),
    }
    print("\nStructural findings recorded (see manifest):")
    print(f"  environment timeline: ~{float(ratio.mean()):.3f}x planned duration (post-planned buffer)")
    print(f"  activity_states timeline: exactly {float(ratio2.min()):.1f}x planned duration")

    # ---- STEP 7: prediction-time availability ----
    avail_rows = []
    for table, spec in TABLE_SPECS.items():
        for col in spec["cols"]:
            cat, why = AVAILABILITY[table][col]
            avail_rows.append({"table": table, "column": col, "availability": cat, "rationale": why})
    avail_df = pd.DataFrame(avail_rows)
    avail_df["availability"] = pd.Categorical(
        avail_df["availability"],
        categories=["AVAILABLE_AT_PREDICTION_TIME", "UNKNOWN", "OUTCOME_OR_POST_OUTCOME"],
        ordered=True,
    )
    avail_df = avail_df.sort_values(["availability", "table", "column"])
    record("leakage", "prediction-time availability classified per field",
           len(avail_df) == sum(len(s["cols"]) for s in TABLE_SPECS.values()))

    # ---- write outputs ----
    if not no_write and all_valid:
        processed_dir.mkdir(parents=True, exist_ok=True)
        for table in TABLE_SPECS:
            write_processed(tables_clean[table], processed_dir, table)
        write_processed(target, processed_dir, "targets_event_delay_days")
        avail_df.to_csv(processed_dir / "prediction_time_availability.csv", index=False,
                        lineterminator="\n", encoding="utf-8")
        print(f"\nWrote 12 cleaned tables + targets_event_delay_days.csv "
              f"+ prediction_time_availability.csv to {processed_dir}")
    elif not all_valid:
        print("\nValidations FAILED — no processed CSVs written (see manifest).")
    else:
        print("\nDry run (--no-write): validations passed, nothing written.")

    # manifest assembly
    manifest["tables"] = table_results
    manifest["overall"] = {"all_valid": all_valid, "failures": failures}
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "cleaning_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    print()
    print("OVERALL:", "ALL VALIDATIONS PASSED" if all_valid else "VALIDATION FAILURES PRESENT")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
    print("Manifest: ", processed_dir / "cleaning_manifest.json")
    print("DONE")
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover