"""src/cpm — deterministic Critical Path Method (CPM) engine.

Calculates Early Start (ES), Early Finish (EF), Late Start (LS),
Late Finish (LF), Total Float and critical activity flags from a
directed acyclic activity graph with FS/SS/FF dependencies and lags.

This is deterministic scheduling logic. It is NOT an ML prediction
and does NOT predict future delays — it computes schedule structure
from the provided planned durations and dependencies.
"""

from __future__ import annotations

from .calculation import (
    CRITICAL_TOLERANCE,
    ActivityCpmResult,
    ProjectCpmSummary,
    build_activity_index,
    calculate_project_cpm,
    cpm_from_dfs,
    detect_cycles,
    validate_dependency_constraints,
)

__all__ = [
    "CRITICAL_TOLERANCE",
    "ActivityCpmResult",
    "ProjectCpmSummary",
    "build_activity_index",
    "calculate_project_cpm",
    "cpm_from_dfs",
    "detect_cycles",
    "validate_dependency_constraints",
]