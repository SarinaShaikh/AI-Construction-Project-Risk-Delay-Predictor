"""
Deterministic Critical Path Method (CPM) engine.

Implements the Critical Path Method for project scheduling graphs with:

- FS (finish-to-start), SS (start-to-start) and FF (finish-to-finish)
  relationships, including integer activity lags.
- Multiple terminal (sink) activities.
- Explicit cycle detection.
- Deterministic tie-breaking.
- Activity-level CPM details including ES, EF, LS, LF and Total Float.

This module is a reusable analytical engine. It is NOT an ML model and
does NOT predict delays.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRITICAL_TOLERANCE: float = 1e-6
"""Activities whose absolute total float is <= this are treated as critical."""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ActivityRef:
    """Unique identifier for an activity within a single project graph."""

    project_id: str
    activity_id: str

    def __repr__(self) -> str:
        return f"ActivityRef({self.project_id}.{self.activity_id})"


@dataclass(frozen=True)
class Dependency:
    """A single precedence relationship between two activities."""

    predecessor: ActivityRef
    successor: ActivityRef
    relationship: str
    lag_days: int = 0

    def __post_init__(self) -> None:
        if self.relationship not in {"FS", "SS", "FF"}:
            raise ValueError(
                f"Unsupported relationship: {self.relationship!r}"
            )

        if self.lag_days < 0:
            raise ValueError(
                f"lag_days must be >= 0, got {self.lag_days}"
            )


@dataclass
class ActivityCpmResult:
    """CPM results for a single activity."""

    ref: ActivityRef
    duration: int
    es: float
    ef: float
    ls: float
    lf: float
    total_float: float
    is_critical: bool
    is_virtual: bool = False
    predecessors: tuple[ActivityRef, ...] = field(default_factory=tuple)
    successors: tuple[ActivityRef, ...] = field(default_factory=tuple)


@dataclass
class ProjectCpmSummary:
    """Deterministic, per-project CPM summary."""

    project_id: str
    project_duration: float
    num_activities: int
    num_edges: int
    num_terminal_activities: int
    num_critical_activities: int
    critical_activity_ids: list[str]
    critical_paths: list[list[str]]
    has_cycles: bool = False
    cycle_description: str = ""
    dependency_violations: list[dict[str, Any]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_activity_index(
    rows: list[dict[str, Any]],
    project_id: str,
    activity_id_col: str = "activity_id",
    duration_col: str = "planned_duration_days",
) -> dict[ActivityRef, int]:
    """Build {ActivityRef: duration} for one project."""

    index: dict[ActivityRef, int] = {}

    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(
                f"Expected row dict, got {type(row)}"
            )

        if row.get("project_id") != project_id:
            continue

        aid = str(row[activity_id_col])
        dur = int(row[duration_col])

        if dur < 0:
            raise ValueError(
                f"Negative duration for "
                f"{row.get('activity_id')!r}: {dur}"
            )

        index[
            ActivityRef(project_id, aid)
        ] = dur

    return index


def build_dependencies(
    rows: list[dict[str, Any]],
    project_id: str,
    pred_col: str = "predecessor_id",
    succ_col: str = "successor_id",
    rel_col: str = "relationship",
    lag_col: str = "lag_days",
) -> list[Dependency]:
    """Build Dependency objects for one project."""

    deps: list[Dependency] = []

    for row in rows:
        if isinstance(row, Dependency):
            deps.append(row)
            continue

        if not isinstance(row, dict):
            raise TypeError(
                f"Expected row dict, got {type(row)}"
            )

        if row.get("project_id") != project_id:
            continue

        pred = ActivityRef(
            project_id,
            str(row[pred_col]),
        )

        succ = ActivityRef(
            project_id,
            str(row[succ_col]),
        )

        rel = str(row[rel_col]).upper()
        lag = int(row[lag_col])

        deps.append(
            Dependency(
                predecessor=pred,
                successor=succ,
                relationship=rel,
                lag_days=lag,
            )
        )

    return deps


def _ensure_dep_list(
    edges: list[Dependency],
) -> list[Dependency]:
    if edges and not isinstance(edges[0], Dependency):
        raise TypeError(
            "edges must be a list of Dependency objects"
        )

    return edges


def detect_cycles(
    nodes: set[ActivityRef],
    edges: list[Dependency],
) -> list[ActivityRef]:
    """Return nodes participating in a directed cycle."""

    adj: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {
        n: []
        for n in nodes
    }

    in_deg: dict[
        ActivityRef,
        int,
    ] = {
        n: 0
        for n in nodes
    }

    edges = _normalize_edges_for_project(
        edges,
        nodes,
    )

    for edge in edges:
        if (
            edge.predecessor not in nodes
            or edge.successor not in nodes
        ):
            continue

        adj[edge.predecessor].append(
            edge.successor
        )

        in_deg[edge.successor] += 1

    ready: collections.deque[ActivityRef] = collections.deque(
        sorted(
            [
                n
                for n in nodes
                if in_deg[n] == 0
            ],
            key=lambda x: x.activity_id,
        )
    )

    processed: set[ActivityRef] = set()

    while ready:
        node = ready.popleft()

        processed.add(node)

        for neighbor in adj[node]:
            in_deg[neighbor] -= 1

            if in_deg[neighbor] == 0:
                ready.append(neighbor)

    return [
        n
        for n in nodes
        if n not in processed
    ]


class CyclicGraphError(Exception):
    """Raised when a project dependency graph contains a cycle."""


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def _sorted_by_activity_id(
    items: collections.abc.Iterable[ActivityRef],
) -> list[ActivityRef]:
    """Stable deterministic sort by activity ID."""

    return sorted(
        items,
        key=lambda x: x.activity_id,
    )


def _normalize_edges(
    edges: list[Dependency],
    project_id: str,
) -> list[Dependency]:
    """Convert edge dictionaries to Dependency objects."""

    out: list[Dependency] = []

    for edge in edges:
        if isinstance(edge, Dependency):
            out.append(edge)
        else:
            out.append(
                Dependency(
                    predecessor=ActivityRef(
                        project_id,
                        str(edge["predecessor_id"]),
                    ),
                    successor=ActivityRef(
                        project_id,
                        str(edge["successor_id"]),
                    ),
                    relationship=str(
                        edge["relationship"]
                    ).upper(),
                    lag_days=int(
                        edge.get("lag_days", 0)
                    ),
                )
            )

    return out


def _normalize_edges_for_project(
    edges: list[Dependency],
    nodes: set[ActivityRef],
) -> list[Dependency]:
    """Normalize edges to Dependency objects."""

    if not edges:
        return []

    first = edges[0]

    if isinstance(first, Dependency):
        return edges

    pid = next(
        iter(nodes)
    ).project_id

    return _normalize_edges(
        edges,
        pid,
    )


def _topological_order(
    nodes: set[ActivityRef],
    edges: list[Dependency],
) -> list[ActivityRef]:
    """Kahn's topological sort with deterministic tie-breaking."""

    adj: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {
        n: []
        for n in nodes
    }

    in_deg: dict[
        ActivityRef,
        int,
    ] = {
        n: 0
        for n in nodes
    }

    edges = _normalize_edges_for_project(
        edges,
        nodes,
    )

    for edge in edges:
        if (
            edge.predecessor not in nodes
            or edge.successor not in nodes
        ):
            continue

        adj[edge.predecessor].append(
            edge.successor
        )

        in_deg[edge.successor] += 1

    ready: collections.deque[ActivityRef] = collections.deque(
        sorted(
            [
                n
                for n in nodes
                if in_deg[n] == 0
            ],
            key=lambda x: x.activity_id,
        )
    )

    order: list[ActivityRef] = []

    while ready:
        node = ready.popleft()

        order.append(node)

        newly_ready: list[ActivityRef] = []

        for neighbor in adj[node]:
            in_deg[neighbor] -= 1

            if in_deg[neighbor] == 0:
                newly_ready.append(neighbor)

        for neighbor in sorted(
            newly_ready,
            key=lambda x: x.activity_id,
        ):
            ready.append(neighbor)

    return order


# ---------------------------------------------------------------------------
# CPM computation
# ---------------------------------------------------------------------------


def calculate_project_cpm(
    project_id: str,
    activity_rows: list[dict[str, Any]],
    dependency_rows: list[dict[str, Any]],
    *,
    activity_id_col: str = "activity_id",
    duration_col: str = "planned_duration_days",
    pred_col: str = "predecessor_id",
    succ_col: str = "successor_id",
    rel_col: str = "relationship",
    lag_col: str = "lag_days",
    include_virtual_end: bool = True,
) -> ProjectCpmSummary:
    """Compute CPM for one project."""

    # 1. Build activity index.
    index = build_activity_index(
        activity_rows,
        project_id,
        activity_id_col=activity_id_col,
        duration_col=duration_col,
    )

    if not index:
        return ProjectCpmSummary(
            project_id=project_id,
            project_duration=0.0,
            num_activities=0,
            num_edges=0,
            num_terminal_activities=0,
            num_critical_activities=0,
            critical_activity_ids=[],
            critical_paths=[],
            has_cycles=False,
        )

    # 2. Build dependencies.
    edges_out = build_dependencies(
        dependency_rows,
        project_id,
        pred_col=pred_col,
        succ_col=succ_col,
        rel_col=rel_col,
        lag_col=lag_col,
    )

    _ensure_dep_list(edges_out)

    nodes = set(index.keys())

    if not edges_out:
        es = {
            n: 0.0
            for n in nodes
        }

        ef = {
            n: float(index[n])
            for n in nodes
        }

        project_duration = float(
            max(ef.values())
        )

        float_map = {
            n: project_duration - ef[n]
            for n in nodes
        }

        ls = {
            n: project_duration - float(index[n])
            for n in nodes
        }

        lf = {
            n: project_duration
            for n in nodes
        }

        preds_by = {
            n: []
            for n in nodes
        }

        succs_by = {
            n: []
            for n in nodes
        }

        results = _make_results(
            index,
            es,
            ef,
            ls,
            lf,
            float_map,
            preds_by=preds_by,
            succs_by=succs_by,
            project_duration=project_duration,
            include_virtual_end=include_virtual_end,
        )

        return _summarize(
            project_id,
            index,
            results,
            edges_out,
            project_duration=project_duration,
            float_map=float_map,
        )

    edges = edges_out

    # 3. Cycle detection.
    cycle_nodes = detect_cycles(
        nodes,
        edges,
    )

    if cycle_nodes:
        raise CyclicGraphError(
            f"Project {project_id} contains a directed cycle "
            f"involving {len(cycle_nodes)} activity nodes: "
            f"{_describe_cycle(cycle_nodes, index)}"
        )

    # 4. Topological order.
    order = _topological_order(
        nodes,
        edges,
    )

    # 5. Build adjacency.
    preds_by: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {
        n: []
        for n in nodes
    }

    succs_by: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {
        n: []
        for n in nodes
    }

    edge_map: dict[
        tuple[ActivityRef, ActivityRef],
        list[Dependency],
    ] = collections.defaultdict(list)

    for edge in edges:
        preds_by[edge.successor].append(
            edge.predecessor
        )

        succs_by[edge.predecessor].append(
            edge.successor
        )

        edge_map[
            (edge.predecessor, edge.successor)
        ].append(edge)

    # 6. Forward pass.
    es: dict[
        ActivityRef,
        float,
    ] = {
        n: 0.0
        for n in nodes
    }

    ef: dict[
        ActivityRef,
        float,
    ] = {}

    for n in order:
        duration = float(index[n])
        bound = 0.0

        for pred in preds_by[n]:
            pred_duration = float(index[pred])

            for edge in edge_map.get(
                (pred, n),
                [],
            ):
                if edge.relationship == "FS":
                    bound = max(
                        bound,
                        es[pred]
                        + pred_duration
                        + float(edge.lag_days),
                    )

                elif edge.relationship == "SS":
                    bound = max(
                        bound,
                        es[pred]
                        + float(edge.lag_days),
                    )

                elif edge.relationship == "FF":
                    bound = max(
                        bound,
                        es[pred]
                        + pred_duration
                        + float(edge.lag_days)
                        - duration,
                    )

        es[n] = bound
        ef[n] = es[n] + duration

    # 7. Backward pass.
    terminal_activities = [
        n
        for n in nodes
        if not succs_by[n]
    ]

    if not terminal_activities:
        terminal_activities = [
            n
            for n in nodes
            if es[n] == max(es.values())
        ]

    project_completion = float(
        max(
            ef[n]
            for n in terminal_activities
        )
    )

    lf: dict[
        ActivityRef,
        float,
    ] = {}

    ls: dict[
        ActivityRef,
        float,
    ] = {}

    for n in nodes:
        lf[n] = max(
            project_completion,
            ef[n],
        )

        ls[n] = (
            lf[n]
            - float(index[n])
        )

        if ls[n] < es[n]:
            ls[n] = es[n]

            lf[n] = (
                ls[n]
                + float(index[n])
            )

    rev_order = list(
        reversed(order)
    )

    changed = True

    while changed:
        changed = False

        for n in rev_order:
            current_lf = lf[n]
            bound = float(
                project_completion
            )

            for succ in succs_by[n]:
                for edge in edge_map.get(
                    (n, succ),
                    [],
                ):
                    if edge.relationship == "FS":
                        bound = min(
                            bound,
                            ls[succ]
                            - float(edge.lag_days),
                        )

                    elif edge.relationship == "SS":
                        bound = min(
                            bound,
                            ls[succ]
                            - float(index[n])
                            - float(edge.lag_days),
                        )

                    elif edge.relationship == "FF":
                        bound = min(
                            bound,
                            lf[succ]
                            - float(edge.lag_days),
                        )

            new_lf = max(
                bound,
                ef[n],
            )

            if new_lf < current_lf - 1e-9:
                lf[n] = new_lf

                ls[n] = (
                    new_lf
                    - float(index[n])
                )

                if ls[n] < es[n]:
                    ls[n] = es[n]

                    lf[n] = (
                        ls[n]
                        + float(index[n])
                    )

                changed = True

    # 8. Float.
    float_map = {
        n: ls[n] - es[n]
        for n in nodes
    }

    # 9. Results.
    results = _make_results(
        index,
        es,
        ef,
        ls,
        lf,
        float_map,
        preds_by=preds_by,
        succs_by=succs_by,
        project_duration=project_completion,
        include_virtual_end=include_virtual_end,
    )

    # 10. Validate constraints.
    violations = validate_dependency_constraints(
        edges,
        es,
        ef,
        ls,
        lf,
        index,
    )

    return _summarize(
        project_id,
        index,
        results,
        edges,
        project_duration=project_completion,
        float_map=float_map,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------


def _make_results(
    index: dict[ActivityRef, int],
    es: dict[ActivityRef, float],
    ef: dict[ActivityRef, float],
    ls: dict[ActivityRef, float],
    lf: dict[ActivityRef, float],
    float_map: dict[ActivityRef, float],
    *,
    preds_by: dict[
        ActivityRef,
        list[ActivityRef],
    ] | None = None,
    succs_by: dict[
        ActivityRef,
        list[ActivityRef],
    ] | None = None,
    project_duration: float = 0.0,
    include_virtual_end: bool = True,
) -> dict[ActivityRef, ActivityCpmResult]:
    """Build ActivityCpmResult for every real activity."""

    del project_duration
    del include_virtual_end

    preds_by = preds_by or {}
    succs_by = succs_by or {}

    results: dict[
        ActivityRef,
        ActivityCpmResult,
    ] = {}

    for n, duration in index.items():
        results[n] = ActivityCpmResult(
            ref=n,
            duration=duration,
            es=es.get(n, 0.0),
            ef=ef.get(
                n,
                float(duration),
            ),
            ls=ls.get(
                n,
                float(duration),
            ),
            lf=lf.get(
                n,
                float(duration),
            ),
            total_float=float_map.get(
                n,
                float(duration),
            ),
            is_critical=(
                abs(
                    float_map.get(
                        n,
                        float(duration),
                    )
                )
                <= CRITICAL_TOLERANCE
            ),
            is_virtual=False,
            predecessors=tuple(
                sorted(
                    preds_by.get(n, []),
                    key=lambda x: x.activity_id,
                )
            ),
            successors=tuple(
                sorted(
                    succs_by.get(n, []),
                    key=lambda x: x.activity_id,
                )
            ),
        )

    return results


def _describe_cycle(
    cycle_nodes: list[ActivityRef],
    index: dict[ActivityRef, int],
) -> str:
    """Human-readable description of a detected cycle."""

    del index

    names = sorted(
        n.activity_id
        for n in cycle_nodes
    )

    body = ", ".join(
        names[:12]
    )

    if len(names) > 12:
        body += " ..."

    return (
        f"{len(cycle_nodes)} nodes in cycle: "
        f"{body}"
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summarize(
    project_id: str,
    index: dict[ActivityRef, int],
    results: dict[ActivityRef, ActivityCpmResult],
    edges: list[Dependency],
    *,
    project_duration: float,
    float_map: dict[ActivityRef, float],
    violations: list[dict[str, Any]] | None = None,
) -> ProjectCpmSummary:
    """Build the project-level CPM summary."""

    del float_map

    real_results = {
        n: result
        for n, result in results.items()
        if not result.is_virtual
    }

    critical = [
        result
        for result in real_results.values()
        if result.is_critical
    ]

    critical_ids = sorted(
        result.ref.activity_id
        for result in critical
    )

    critical_ids_set = {
        result.ref
        for result in critical
    }

    preds_by_crit: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {}

    succs_by_crit: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {}

    for result in critical:
        preds_by_crit[result.ref] = [
            pred
            for pred in result.predecessors
            if pred in critical_ids_set
        ]

        succs_by_crit[result.ref] = [
            succ
            for succ in result.successors
            if succ in critical_ids_set
        ]

    roots = sorted(
        [
            n
            for n in critical_ids_set
            if not preds_by_crit[n]
        ],
        key=lambda n: n.activity_id,
    )

    root = roots[0] if roots else None

    path: list[ActivityRef] = []
    current = root
    visited: set[ActivityRef] = set()

    while (
        current is not None
        and current not in visited
    ):
        visited.add(current)

        path.append(current)

        successors = sorted(
            succs_by_crit.get(
                current,
                [],
            ),
            key=lambda n: n.activity_id,
        )

        next_node = (
            successors[0]
            if successors
            else None
        )

        if next_node is None:
            break

        current = next_node

    critical_path_ids = [
        n.activity_id
        for n in path
    ]

    terminal = [
        n
        for n in index
        if not results[n].successors
    ]

    return ProjectCpmSummary(
        project_id=project_id,
        project_duration=project_duration,
        num_activities=len(index),
        num_edges=len(edges),
        num_terminal_activities=len(terminal),
        num_critical_activities=len(critical),
        critical_activity_ids=critical_ids,
        critical_paths=[
            critical_path_ids
        ],
        has_cycles=False,
        dependency_violations=violations or [],
    )


def _summarize_no_edges(
    project_id: str,
    index: dict[ActivityRef, int],
    include_virtual_end: bool,
) -> ProjectCpmSummary:
    """Summarize a project with no dependencies."""

    del include_virtual_end

    nodes = set(index.keys())

    project_duration = float(
        max(index.values())
    )

    es = {
        n: 0.0
        for n in nodes
    }

    ef = {
        n: float(d)
        for n, d in index.items()
    }

    float_map = {
        n: project_duration - ef[n]
        for n in nodes
    }

    ls = {
        n: project_duration - float(index[n])
        for n in nodes
    }

    lf = {
        n: project_duration
        for n in nodes
    }

    preds_by = {
        n: []
        for n in nodes
    }

    succs_by = {
        n: []
        for n in nodes
    }

    results = _make_results(
        index,
        es,
        ef,
        ls,
        lf,
        float_map,
        preds_by=preds_by,
        succs_by=succs_by,
        project_duration=project_duration,
        include_virtual_end=True,
    )

    return _summarize(
        project_id,
        index,
        results,
        edges=[],
        project_duration=project_duration,
        float_map=float_map,
        violations=[],
    )


# ---------------------------------------------------------------------------
# Dependency constraint validation
# ---------------------------------------------------------------------------


def validate_dependency_constraints(
    edges: list[Dependency],
    es: dict[ActivityRef, float],
    ef: dict[ActivityRef, float],
    ls: dict[ActivityRef, float],
    lf: dict[ActivityRef, float],
    index: dict[ActivityRef, int] | None = None,
) -> list[dict[str, Any]]:
    """Return a list of dependency constraint violations."""

    del ls
    del lf
    del index

    _edges = _normalize_edges_for_project(
        edges,
        set(es.keys()),
    )

    violations: list[
        dict[str, Any]
    ] = []

    for edge in _edges:
        pred = edge.predecessor
        succ = edge.successor

        if (
            pred not in es
            or succ not in es
        ):
            continue

        if edge.relationship == "FS":
            required = (
                ef[pred]
                + float(edge.lag_days)
            )

            satisfied = (
                es[succ]
                >= required - 1e-9
            )

        elif edge.relationship == "SS":
            required = (
                es[pred]
                + float(edge.lag_days)
            )

            satisfied = (
                es[succ]
                >= required - 1e-9
            )

        elif edge.relationship == "FF":
            required = (
                ef[pred]
                + float(edge.lag_days)
            )

            satisfied = (
                ef[succ]
                >= required - 1e-9
            )

        else:
            required = 0.0
            satisfied = True

        if not satisfied:
            violations.append(
                {
                    "predecessor": str(pred),
                    "successor": str(succ),
                    "relationship": edge.relationship,
                    "lag_days": edge.lag_days,
                    "required": float(required),
                    "actual_es": es[succ],
                    "actual_ef": ef[succ],
                    "metric": (
                        "ES"
                        if edge.relationship != "FF"
                        else "EF"
                    ),
                }
            )

    return violations


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def cpm_from_dfs(
    project_id: str,
    activity_index: dict[ActivityRef, int],
    edges: list[Dependency],
    *,
    include_virtual_end: bool = True,
) -> ProjectCpmSummary:
    """Run CPM from an already-built activity index."""

    fake_rows = [
        {
            "project_id": node.project_id,
            "activity_id": node.activity_id,
            "planned_duration_days": duration,
        }
        for node, duration in activity_index.items()
    ]

    dep_rows: list[
        dict[str, object]
    ] = []

    for edge in edges:
        if isinstance(edge, Dependency):
            pred = edge.predecessor
            succ = edge.successor
            rel = edge.relationship
            lag = edge.lag_days

        else:
            pred = ActivityRef(
                project_id,
                str(edge["predecessor_id"]),
            )

            succ = ActivityRef(
                project_id,
                str(edge["successor_id"]),
            )

            rel = str(
                edge["relationship"]
            ).upper()

            lag = int(
                edge.get(
                    "lag_days",
                    0,
                )
            )

        dep_rows.append(
            {
                "project_id": project_id,
                "predecessor_id": pred.activity_id,
                "successor_id": succ.activity_id,
                "relationship": rel,
                "lag_days": lag,
            }
        )

    return calculate_project_cpm(
        project_id,
        fake_rows,
        dep_rows,
        activity_id_col="activity_id",
        duration_col="planned_duration_days",
        pred_col="predecessor_id",
        succ_col="successor_id",
        rel_col="relationship",
        lag_col="lag_days",
        include_virtual_end=include_virtual_end,
    )


# ---------------------------------------------------------------------------
# Detailed CPM API
# ---------------------------------------------------------------------------


def calculate_project_cpm_details(
    project_id: str,
    activity_rows: list[dict[str, Any]],
    dependency_rows: list[dict[str, Any]],
    *,
    activity_id_col: str = "activity_id",
    duration_col: str = "planned_duration_days",
    pred_col: str = "predecessor_id",
    succ_col: str = "successor_id",
    rel_col: str = "relationship",
    lag_col: str = "lag_days",
    include_virtual_end: bool = True,
) -> dict[ActivityRef, ActivityCpmResult]:
    """Return activity-level CPM results.

    This function exposes:

    - Early Start
    - Early Finish
    - Late Start
    - Late Finish
    - Total Float
    - Critical status
    - Predecessors
    - Successors

    The calculations use the same CPM rules as calculate_project_cpm().
    """

    index = build_activity_index(
        activity_rows,
        project_id,
        activity_id_col=activity_id_col,
        duration_col=duration_col,
    )

    if not index:
        return {}

    edges = build_dependencies(
        dependency_rows,
        project_id,
        pred_col=pred_col,
        succ_col=succ_col,
        rel_col=rel_col,
        lag_col=lag_col,
    )

    nodes = set(index.keys())

    # ---------------------------------------------------------------
    # No dependencies
    # ---------------------------------------------------------------

    if not edges:
        es = {
            n: 0.0
            for n in nodes
        }

        ef = {
            n: float(index[n])
            for n in nodes
        }

        project_duration = float(
            max(ef.values())
        )

        float_map = {
            n: project_duration - ef[n]
            for n in nodes
        }

        ls = {
            n: project_duration - float(index[n])
            for n in nodes
        }

        lf = {
            n: project_duration
            for n in nodes
        }

        preds_by = {
            n: []
            for n in nodes
        }

        succs_by = {
            n: []
            for n in nodes
        }

        return _make_results(
            index,
            es,
            ef,
            ls,
            lf,
            float_map,
            preds_by=preds_by,
            succs_by=succs_by,
            project_duration=project_duration,
            include_virtual_end=include_virtual_end,
        )

    # ---------------------------------------------------------------
    # Cycle detection
    # ---------------------------------------------------------------

    cycle_nodes = detect_cycles(
        nodes,
        edges,
    )

    if cycle_nodes:
        raise CyclicGraphError(
            f"Project {project_id} contains a directed cycle "
            f"involving {len(cycle_nodes)} activity nodes: "
            f"{_describe_cycle(cycle_nodes, index)}"
        )

    # ---------------------------------------------------------------
    # Topological order
    # ---------------------------------------------------------------

    order = _topological_order(
        nodes,
        edges,
    )

    # ---------------------------------------------------------------
    # Adjacency
    # ---------------------------------------------------------------

    preds_by: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {
        n: []
        for n in nodes
    }

    succs_by: dict[
        ActivityRef,
        list[ActivityRef],
    ] = {
        n: []
        for n in nodes
    }

    edge_map: dict[
        tuple[ActivityRef, ActivityRef],
        list[Dependency],
    ] = collections.defaultdict(list)

    for edge in edges:
        preds_by[edge.successor].append(
            edge.predecessor
        )

        succs_by[edge.predecessor].append(
            edge.successor
        )

        edge_map[
            (edge.predecessor, edge.successor)
        ].append(edge)

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    es: dict[
        ActivityRef,
        float,
    ] = {
        n: 0.0
        for n in nodes
    }

    ef: dict[
        ActivityRef,
        float,
    ] = {}

    for node in order:
        duration = float(
            index[node]
        )

        bound = 0.0

        for pred in preds_by[node]:
            pred_duration = float(
                index[pred]
            )

            for edge in edge_map.get(
                (pred, node),
                [],
            ):
                if edge.relationship == "FS":
                    bound = max(
                        bound,
                        es[pred]
                        + pred_duration
                        + float(edge.lag_days),
                    )

                elif edge.relationship == "SS":
                    bound = max(
                        bound,
                        es[pred]
                        + float(edge.lag_days),
                    )

                elif edge.relationship == "FF":
                    bound = max(
                        bound,
                        es[pred]
                        + pred_duration
                        + float(edge.lag_days)
                        - duration,
                    )

        es[node] = bound
        ef[node] = (
            es[node]
            + duration
        )

    # ---------------------------------------------------------------
    # Project completion
    # ---------------------------------------------------------------

    terminal_activities = [
        node
        for node in nodes
        if not succs_by[node]
    ]

    if not terminal_activities:
        terminal_activities = [
            node
            for node in nodes
            if es[node] == max(
                es.values()
            )
        ]

    project_completion = float(
        max(
            ef[node]
            for node in terminal_activities
        )
    )

    # ---------------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------------

    lf: dict[
        ActivityRef,
        float,
    ] = {}

    ls: dict[
        ActivityRef,
        float,
    ] = {}

    for node in nodes:
        lf[node] = max(
            project_completion,
            ef[node],
        )

        ls[node] = (
            lf[node]
            - float(index[node])
        )

        if ls[node] < es[node]:
            ls[node] = es[node]

            lf[node] = (
                ls[node]
                + float(index[node])
            )

    rev_order = list(
        reversed(order)
    )

    changed = True

    while changed:
        changed = False

        for node in rev_order:
            current_lf = lf[node]

            bound = float(
                project_completion
            )

            for succ in succs_by[node]:
                for edge in edge_map.get(
                    (node, succ),
                    [],
                ):
                    if edge.relationship == "FS":
                        bound = min(
                            bound,
                            ls[succ]
                            - float(edge.lag_days),
                        )

                    elif edge.relationship == "SS":
                        bound = min(
                            bound,
                            ls[succ]
                            - float(index[node])
                            - float(edge.lag_days),
                        )

                    elif edge.relationship == "FF":
                        bound = min(
                            bound,
                            lf[succ]
                            - float(edge.lag_days),
                        )

            new_lf = max(
                bound,
                ef[node],
            )

            if new_lf < current_lf - 1e-9:
                lf[node] = new_lf

                ls[node] = (
                    new_lf
                    - float(index[node])
                )

                if ls[node] < es[node]:
                    ls[node] = es[node]

                    lf[node] = (
                        ls[node]
                        + float(index[node])
                    )

                changed = True

    # ---------------------------------------------------------------
    # Float
    # ---------------------------------------------------------------

    float_map = {
        node: ls[node] - es[node]
        for node in nodes
    }

    # ---------------------------------------------------------------
    # Detailed results
    # ---------------------------------------------------------------

    return _make_results(
        index,
        es,
        ef,
        ls,
        lf,
        float_map,
        preds_by=preds_by,
        succs_by=succs_by,
        project_duration=project_completion,
        include_virtual_end=include_virtual_end,
    )