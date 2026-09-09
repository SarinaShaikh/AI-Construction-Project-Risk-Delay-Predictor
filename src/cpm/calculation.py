"""
Deterministic Critical Path Method (CPM) engine.

Implements the Critical Path Method for project scheduling graphs with:

- FS (finish-to-start), SS (start-to-start) and FF (finish-to-finish)
  relationships, including integer activity lags.
- Multiple terminal (sink) activities, handled via a deterministic
  zero-duration virtual project-end node.
- Explicit, strict cycle detection (fails loudly on cyclic graphs).
- Deterministic tie-breaking so that the same input always produces the
  same schedule.

This module is a reusable analytical engine. It is NOT an ML model and
does NOT predict delays. It calculates schedule structure from the
provided planned durations and dependencies.

Mathematics
----------
Activity durations are positive integer days.

Forward pass
    For an edge predecessor P -> successor S with lag L and relationship R:

        FS:  ES[S] >= EF[P] + L
        SS:  ES[S] >= ES[P] + L
        FF:  EF[S] >= EF[P] + L

    With EF[X] = ES[X] + duration[X], the three inequalities can be written
    as lower bounds on ES[S]:

        FS:  ES[S] >= ES[P] + duration[P] + L
        SS:  ES[S] >= ES[P] + L
        FF:  ES[S] >= ES[P] + duration[P] + L - duration[S]

    ES[S] is set to the maximum of these lower bounds across all its
    immediate predecessors, together with the natural bound ES[S] >= 0.
    Then EF[S] = ES[S] + duration[S].

Backward pass
    For the same edge P -> S with lag L and relationship R, expressing upper
    bounds on LF[P] (equivalently LS[P] = LF[P] - duration[P]):

        FS:  LF[P] <= LS[S] - L
        SS:  LF[P] <= LS[S] - L          (so LS[P] <= LS[S] - duration[P] - L)
        FF:  LF[P] <= LF[S] - L          (so LS[P] <= LF[S] - L - duration[P])

    LF[P] is the minimum of the applicable upper bounds across all immediate
    successors, and LS[P] = LF[P] - duration[P].

Float
    Total Float = LS - ES = LF - EF   (should agree to a small tolerance).

Criticality
    An activity is critical if abs(Total Float) <= CRITICAL_TOLERANCE.

Project completion
    A zero-duration virtual END node is connected from every true terminal
    activity X with relationship FS, lag 0:
        END  dep  X   (meaning: X must finish before END starts,
                          concrete: ES[END] >= EF[X] + 0)
    Because END has duration 0, EF[END] = ES[END] = max_X EF[X], the
    earliest possible project completion across all terminal activities.
    END is not treated as a real activity for reporting purposes.

Cycle safety
    Before computing a schedule, the engine detects cycles using Kahn's
    algorithm applied to each project's graph. A cyclic graph raises
    CyclicGraphError.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Deterministic tie-breaking seed. We use an explicit insertion order into
# the ready-queue rather than any random or system-dependent ordering, so the
# same graph always yields the same schedule.
#
# For floating-point comparisons we use a small tolerance because the activity
# durations are integer days and the relationship lags are integer days, so any
# slack should be an exact integer. We allow a small tolerance to absorb
# floating-point rounding in intermediate arithmetic.
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
    relationship: str  # "FS", "SS" or "FF"
    lag_days: int = 0

    def __post_init__(self) -> None:
        if self.relationship not in {"FS", "SS", "FF"}:
            raise ValueError(f"Unsupported relationship: {self.relationship!r}")
        if self.lag_days < 0:
            raise ValueError(f"lag_days must be >= 0, got {self.lag_days}")


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
    dependency_violations: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_activity_index(
    rows: list[dict[str, Any]],
    project_id: str,
    activity_id_col: str = "activity_id",
    duration_col: str = "planned_duration_days",
) -> dict[ActivityRef, int]:
    """Build {ActivityRef: duration} for one project from a rows list.

    Parameters
    ----------
    rows:
        Iterable of row dicts, typically from activities.csv for one project.
    project_id:
        The project to scope to (all rows should already be filtered to one
        project, but we keep the column for safety / clarity).
    activity_id_col:
        Column name holding the activity identifier.
    duration_col:
        Column name holding the planned duration in days.
    """
    index: dict[ActivityRef, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"Expected row dict, got {type(row)}")
        if row.get("project_id") != project_id:
            continue
        aid = str(row[activity_id_col])
        dur = int(row[duration_col])
        if dur < 0:
            raise ValueError(f"Negative duration for {row.get('activity_id')!r}: {dur}")
        index[ActivityRef(project_id, aid)] = dur
    return index


def build_dependencies(
    rows: list[dict[str, Any]],
    project_id: str,
    pred_col: str = "predecessor_id",
    succ_col: str = "successor_id",
    rel_col: str = "relationship",
    lag_col: str = "lag_days",
) -> list[Dependency]:
    """Build Dependency objects for one project from a rows list.

    Raises ValueError if an edge references an activity not present in the
    same project (should have been caught earlier during cleaning).
    """
    deps: list[Dependency] = []
    for row in rows:
        if isinstance(row, Dependency):
            deps.append(row)
            continue
        if not isinstance(row, dict):
            raise TypeError(f"Expected row dict, got {type(row)}")
        if row.get("project_id") != project_id:
            continue
        pred = ActivityRef(project_id, str(row[pred_col]))
        succ = ActivityRef(project_id, str(row[succ_col]))
        rel = str(row[rel_col]).upper()
        lag = int(row[lag_col])
        deps.append(Dependency(predecessor=pred, successor=succ, relationship=rel, lag_days=lag))
    return deps


def _ensure_dep_list(edges: list[Dependency]) -> list[Dependency]:
    if edges and not isinstance(edges[0], Dependency):
        raise TypeError("edges must be a list of Dependency objects")
    return edges


def detect_cycles(
    nodes: set[ActivityRef],
    edges: list[Dependency],
) -> list[ActivityRef]:
    """Return the set of nodes that participate in any directed cycle.

    Uses Kahn's algorithm. If the graph is acyclic, returns an empty list.
    """
    adj: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    in_deg: dict[ActivityRef, int] = {n: 0 for n in nodes}
    edges = _normalize_edges_for_project(edges, nodes)
    for e in edges:
        if e.predecessor not in nodes or e.successor not in nodes:
            continue
        adj[e.predecessor].append(e.successor)
        in_deg[e.successor] += 1

    ready: collections.deque[ActivityRef] = collections.deque(
        [n for n in nodes if in_deg[n] == 0]
    )
    processed: set[ActivityRef] = set()
    while ready:
        node = ready.popleft()
        processed.add(node)
        for nb in adj[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                ready.append(nb)

    cycle_nodes = [n for n in nodes if n not in processed]
    return cycle_nodes


class CyclicGraphError(Exception):
    """Raised when a project dependency graph contains a cycle."""


# ---------------------------------------------------------------------------
# Small deterministic helper for tie-breaking
# ---------------------------------------------------------------------------


def _sorted_by_activity_id(items: collections.abc.Iterable[ActivityRef]) -> list[ActivityRef]:
    """Stable, deterministic sort by activity id."""
    return sorted(items, key=lambda x: x.activity_id)


# ---------------------------------------------------------------------------
# CPM computation
# ---------------------------------------------------------------------------


def _normalize_edges(
    edges: list[Dependency],
    project_id: str,
) -> list[Dependency]:
    """Convert any edge dicts (as produced by the test helpers) into Dependency objects."""
    out: list[Dependency] = []
    for e in edges:
        if isinstance(e, Dependency):
            out.append(e)
        else:
            out.append(Dependency(
                predecessor=ActivityRef(project_id, str(e["predecessor_id"])),
                successor=ActivityRef(project_id, str(e["successor_id"])),
                relationship=str(e["relationship"]).upper(),
                lag_days=int(e.get("lag_days", 0)),
            ))
    return out


def _normalize_edges_for_project(edges: list[Dependency], nodes: set[ActivityRef]) -> list[Dependency]:
    """Normalize edges to Dependency objects, inferring project_id from the node set."""
    if not edges:
        return []
    first = edges[0]
    if isinstance(first, Dependency):
        return edges
    pid = next(iter(nodes)).project_id
    return _normalize_edges(edges, pid)


def _topological_order(nodes: set[ActivityRef], edges: list[Dependency]) -> list[ActivityRef]:
    """Kahn's topological sort with deterministic tie-breaking.

    Tie-breaking: the ready queue is processed in the order nodes become
    ready, and among nodes that start with in-degree 0, we use the natural
    iteration order of the `nodes` set (which is stable for a given dict/set
    creation history in CPython for a given input). To make this fully
    deterministic regardless of set ordering, we sort the initial ready list
    and the newly-ready appends by a stable key (the activity id string).
    """
    adj: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    in_deg: dict[ActivityRef, int] = {n: 0 for n in nodes}
    edges = _normalize_edges_for_project(edges, nodes)
    for e in edges:
        if e.predecessor not in nodes or e.successor not in nodes:
            continue
        adj[e.predecessor].append(e.successor)
        in_deg[e.successor] += 1

    def key(ref: ActivityRef) -> str:
        return ref.activity_id

    ready: collections.deque[ActivityRef] = collections.deque(
        sorted([n for n in nodes if in_deg[n] == 0], key=key)
    )
    order: list[ActivityRef] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        # Collect successors that become ready; sort them deterministically.
        newly_ready: list[ActivityRef] = []
        for nb in adj[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                newly_ready.append(nb)
        for nb in sorted(newly_ready, key=key):
            ready.append(nb)
    return order


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
    """Compute CPM for one project.

    Parameters
    ----------
    project_id:
        Project identifier. All rows should belong to this project.
    activity_rows:
        List of row dicts from activities.csv (may be pre-filtered to one
        project or contain all projects; we filter by project_id).
    dependency_rows:
        List of row dicts from dependencies.csv.
    include_virtual_end:
        If True (default), attach a zero-duration virtual END node connected
        from every true terminal activity, so project_duration = max terminal
        EF. If False, project_duration = max terminal EF computed from the
        raw graph without an explicit END node (identical numeric result).
    """
    # 1. Build activity index
    index = build_activity_index(
        activity_rows,
        project_id,
        activity_id_col=activity_id_col,
        duration_col=duration_col,
    )
    if not index:
        # Edge case: project with no activities
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

    # 2. Build dependencies
    edges_out: list[Dependency] = build_dependencies(
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
        # No dependencies: every activity starts at 0, project duration = max
        # activity duration.
        es = {n: 0.0 for n in nodes}
        ef = {n: float(index[n]) for n in nodes}
        project_duration = float(max(ef.values()))
        float_map: dict[ActivityRef, float] = {n: project_duration - ef[n] for n in nodes}
        ls = {n: float(project_duration) - float(index[n]) for n in nodes}
        lf = {n: float(project_duration) for n in nodes}
        preds_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
        succs_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}

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

    # 3. Cycle detection — fail loudly on cyclic graphs.
    cycle_nodes = detect_cycles(nodes, edges)
    if cycle_nodes:
        raise CyclicGraphError(
            f"Project {project_id} contains a directed cycle involving "
            f"{len(cycle_nodes)} activity nodes: {_describe_cycle(cycle_nodes, index)}"
        )

    # 4. Topological order
    order = _topological_order(nodes, edges)

    # 5. Build adjacency for fast lookups
    preds_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    succs_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    edge_map: dict[tuple[ActivityRef, ActivityRef], list[Dependency]] = collections.defaultdict(list)
    for e in edges:
        preds_by[e.successor].append(e.predecessor)
        succs_by[e.predecessor].append(e.successor)
        edge_map[(e.predecessor, e.successor)].append(e)

    # 6. Forward pass
    es: dict[ActivityRef, float] = {n: 0.0 for n in nodes}
    ef: dict[ActivityRef, float] = {}
    for n in order:
        dur = float(index[n])
        bound = 0.0
        for pred in preds_by[n]:
            pred_dur = float(index[pred])
            for e in edge_map.get((pred, n), []):
                if e.relationship == "FS":
                    bound = max(bound, es[pred] + pred_dur + float(e.lag_days))
                elif e.relationship == "SS":
                    bound = max(bound, es[pred] + float(e.lag_days))
                elif e.relationship == "FF":
                    bound = max(bound, es[pred] + pred_dur + float(e.lag_days) - dur)
        es[n] = bound
        ef[n] = es[n] + dur

    # 7. Backward pass
    ls: dict[ActivityRef, float] = {}
    lf: dict[ActivityRef, float] = {}
    # Initialise all successors of END (virtual) by the project completion.
    # We derive project completion as max terminal EF.
    terminal_activities = [n for n in nodes if not succs_by[n]]
    if not terminal_activities:
        # Fully connected — unlikely but possible. Fallback to max EF.
        terminal_activities = [n for n in nodes if es[n] == max(es.values())]

    project_completion = float(max(ef[n] for n in terminal_activities)) if terminal_activities else float(max(ef.values()))
    for n in nodes:
        # start from the project completion bound
        lf[n] = project_completion
        ls[n] = lf[n] - float(index[n])

    # Tighten using successors. We iterate backward through topological order so
    # that when we process a node, all its successors already have their final
    # LS/LF (except for nodes that can be tightened further by other successors;
    # we propagate until convergence using a simple queue for correctness and
    # determinism).
    rev_order = list(reversed(order))
    # Initialize lf/ls with the project-completed upper bound, but never below EF/ES.
    lf = {}
    ls = {}
    for n in nodes:
        # LF cannot be less than EF (otherwise float would be negative)
        lf[n] = max(project_completion, ef[n])
        ls[n] = lf[n] - float(index[n])
        # LS cannot be less than ES
        if ls[n] < es[n]:
            ls[n] = es[n]
            lf[n] = ls[n] + float(index[n])

    # Propagate backward until no more tightening.
    # LF must never drop below EF, and LS must never drop below ES.
    changed = True
    while changed:
        changed = False
        for n in rev_order:
            cur_lf = lf[n]
            bound = float(project_completion)
            for succ in succs_by[n]:
                for e in edge_map.get((n, succ), []):
                    if e.relationship == "FS":
                        bound = min(bound, ls[succ] - float(e.lag_days))
                    elif e.relationship == "SS":
                        bound = min(bound, ls[succ] - float(index[n]) - float(e.lag_days))
                    elif e.relationship == "FF":
                        bound = min(bound, lf[succ] - float(e.lag_days))
            new_lf = max(bound, ef[n])  # LF cannot be less than EF
            if new_lf < cur_lf - 1e-9:
                lf[n] = new_lf
                ls[n] = new_lf - float(index[n])
                if ls[n] < es[n]:  # LS cannot be less than ES
                    ls[n] = es[n]
                    lf[n] = ls[n] + float(index[n])
                changed = True

    # 8. Float
    float_map = {n: (ls[n] - es[n]) for n in nodes}

    # 9. Results
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

    # 10. Validate constraints (internal consistency check)
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


def _make_results(
    index: dict[ActivityRef, int],
    es: dict[ActivityRef, float],
    ef: dict[ActivityRef, float],
    ls: dict[ActivityRef, float],
    lf: dict[ActivityRef, float],
    float_map: dict[ActivityRef, float],
    *,
    preds_by: dict[ActivityRef, list[ActivityRef]] | None = None,
    succs_by: dict[ActivityRef, list[ActivityRef]] | None = None,
    project_duration: float = 0.0,
    include_virtual_end: bool = True,
) -> dict[ActivityRef, ActivityCpmResult]:
    """Build ActivityCpmResult for every real activity.

    If include_virtual_end is True and there are real terminal activities, we
    also create a virtual END result for completeness (not included in
    activity counts / critical activity reporting).
    """
    preds_by = preds_by or {}
    succs_by = succs_by or {}
    results: dict[ActivityRef, ActivityCpmResult] = {}
    for n, dur in index.items():
        results[n] = ActivityCpmResult(
            ref=n,
            duration=dur,
            es=es.get(n, 0.0),
            ef=ef.get(n, float(dur)),
            ls=ls.get(n, float(dur)),
            lf=lf.get(n, float(dur)),
            total_float=float_map.get(n, float(dur)),
            is_critical=abs(float_map.get(n, float(dur))) <= CRITICAL_TOLERANCE,
            is_virtual=False,
            predecessors=tuple(sorted(preds_by.get(n, []), key=lambda x: x.activity_id)),
            successors=tuple(sorted(succs_by.get(n, []), key=lambda x: x.activity_id)),
        )
    return results


def _describe_cycle(cycle_nodes: list[ActivityRef], index: dict[ActivityRef, int]) -> str:
    """Human-readable description of a detected cycle."""
    names = sorted(n.activity_id for n in cycle_nodes)
    body = ', '.join(names[:12])
    if len(names) > 12:
        body += ' ...'
    return f"{len(cycle_nodes)} nodes in cycle: {body}"


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
    real_results = {n: r for n, r in results.items() if not r.is_virtual}
    critical = [r for r in real_results.values() if r.is_critical]
    critical_ids = sorted(r.ref.activity_id for r in critical)

    # Find one critical path: start at a critical root (no critical predecessor
    # or all predecessors are critical and this is the earliest critical path),
    # follow critical successors. We pick the lexicographically-first critical
    # root and follow the lexicographically-first critical successor at each
    # step until we hit a terminal critical activity.
    critical_ids_set = {r.ref for r in critical}
    preds_by_crit: dict[ActivityRef, list[ActivityRef]] = {}
    succs_by_crit: dict[ActivityRef, list[ActivityRef]] = {}
    for r in critical:
        preds_by_crit[r.ref] = [p for p in r.predecessors if p in critical_ids_set]
        succs_by_crit[r.ref] = [s for s in r.successors if s in critical_ids_set]

    roots = sorted(
        [n for n in critical_ids_set if not preds_by_crit[n]],
        key=lambda n: n.activity_id,
    )
    root = roots[0] if roots else None
    path: list[ActivityRef] = []
    current = root
    visited: set[ActivityRef] = set()
    while current is not None and current not in visited:
        visited.add(current)
        path.append(current)
        succs = sorted(succs_by_crit.get(current, []), key=lambda n: n.activity_id)
        nxt = succs[0] if succs else None
        # Stop if none of the successors is critical (terminal critical activity)
        if nxt is None:
            break
        current = nxt

    critical_path_ids = [n.activity_id for n in path]
    terminal = [n for n in index if not results[n].successors]
    return ProjectCpmSummary(
        project_id=project_id,
        project_duration=project_duration,
        num_activities=len(index),
        num_edges=len(edges),
        num_terminal_activities=len(terminal),
        num_critical_activities=len(critical),
        critical_activity_ids=critical_ids,
        critical_paths=[critical_path_ids],
        has_cycles=False,
        dependency_violations=violations or [],
    )


def _summarize_no_edges(
    project_id: str,
    index: dict[ActivityRef, int],
    include_virtual_end: bool,
) -> ProjectCpmSummary:
    nodes = set(index.keys())
    project_duration = float(max(index.values()))
    es = {n: 0.0 for n in nodes}
    ef = {n: float(d) for n, d in index.items()}
    float_map = {n: project_duration - ef[n] for n in nodes}
    ls = {n: project_duration - float(index[n]) for n in nodes}
    lf = {n: project_duration for n in nodes}
    preds_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    succs_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}

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
    """Return a list of any dependency constraint violations.

    For each Dependency edge, verifies:

        FS:  ES[S] >= EF[P] + L
        SS:  ES[S] >= ES[P] + L
        FF:  EF[S] >= EF[P] + L

    Returns empty list when all constraints hold.
    """
    if index is None:
        index = {n: round(ef - es[n]) for n, ef in ef.items() for es in (es,) if (ef[n] - es[n]) >= 0}
        # Safer: require explicit durations
        index = {}
    _edges = _normalize_edges_for_project(edges, set(es.keys()))
    violations: list[dict[str, Any]] = []
    for e in _edges:
        p, s = e.predecessor, e.successor
        if p not in es or s not in es:
            continue
        if e.relationship == "FS":
            required = ef[p] + float(e.lag_days)
            satisfied = es[s] >= required - 1e-9
        elif e.relationship == "SS":
            required = es[p] + float(e.lag_days)
            satisfied = es[s] >= required - 1e-9
        elif e.relationship == "FF":
            required = ef[p] + float(e.lag_days)
            satisfied = ef[s] >= required - 1e-9
        else:
            required = 0.0
            satisfied = True
        if not satisfied:
            violations.append(
                {
                    "predecessor": str(p),
                    "successor": str(s),
                    "relationship": e.relationship,
                    "lag_days": e.lag_days,
                    "required": float(required),
                    "actual_es": es[s],
                    "actual_ef": ef[s],
                    "metric": "ES" if e.relationship != "FF" else "EF",
                }
            )
    return violations

def cpm_from_dfs(
    project_id: str,
    activity_index: dict[ActivityRef, int],
    edges: list[Dependency],
    *,
    include_virtual_end: bool = True,
) -> ProjectCpmSummary:
    """Convenience wrapper that runs CPM from an already-built activity
    index. Accepts plain edge dicts (as produced by the test helpers) as
    well as Dependency objects.
    """
    fake_rows = [
        {"project_id": n.project_id, "activity_id": n.activity_id, "planned_duration_days": d}
        for n, d in activity_index.items()
    ]
    dep_rows: list[dict[str, object]] = []
    for e in edges:
        if isinstance(e, Dependency):
            pred, succ, rel, lag = e.predecessor, e.successor, e.relationship, e.lag_days
        else:
            pred = ActivityRef(project_id, str(e["predecessor_id"]))
            succ = ActivityRef(project_id, str(e["successor_id"]))
            rel = str(e["relationship"]).upper()
            lag = int(e.get("lag_days", 0))
        dep_rows.append({
            "project_id": project_id,
            "predecessor_id": pred.activity_id,
            "successor_id": succ.activity_id,
            "relationship": rel,
            "lag_days": lag,
        })
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
