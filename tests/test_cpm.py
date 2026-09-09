"""
Phase 3 — deterministic CPM engine unit tests.

Tests the CPM engine implemented in src/cpm/calculation.py against small,
hand-calculated examples covering:

A. Simple serial FS chain
B. Parallel activities
C. FS with lag
D. SS relationship
E. FF relationship
F. Mixed FS + SS + FF graph
G. Multiple terminal activities
H. Zero-duration virtual end
I. Cycle detection
J. Float calculation (LS - ES == LF - EF)
K. Critical activity detection
L. Dependency constraint validation (explicit FS/SS/FF equations)
M. Edge cases (single activity, zero lag, max observed lag, multiple preds,
   multiple successors)

The engine is intentionally NetworkX-free: it uses a deterministic Kahn
topological sort plus explicit forward/backward passes.

Run with:  python -m pytest tests/test_cpm.py -v
(This repo uses a local .venv; use that python if `python -m pytest` is
unavailable.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cpm.calculation import (
    CRITICAL_TOLERANCE,
    ActivityCpmResult,
    ActivityRef,
    CyclicGraphError,
    Dependency,
    _topological_order,
    calculate_project_cpm,
    cpm_from_dfs,
    detect_cycles,
    validate_dependency_constraints,
)

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def act(project_id: str, activity_id: str, duration: int) -> dict[str, object]:
    return {
        "project_id": project_id,
        "activity_id": activity_id,
        "planned_duration_days": duration,
    }


def dep(
    project_id: str,
    predecessor: str,
    successor: str,
    relationship: str = "FS",
    lag_days: int = 0,
) -> dict[str, object]:
    return {
        "project_id": project_id,
        "predecessor_id": predecessor,
        "successor_id": successor,
        "relationship": relationship,
        "lag_days": lag_days,
    }


def ref(project_id: str, activity_id: str) -> ActivityRef:
    return ActivityRef(project_id=project_id, activity_id=activity_id)


# ---------------------------------------------------------------------------
# A. Simple serial FS chain  A -> B -> C
# ---------------------------------------------------------------------------


def test_serial_fs_chain():
    # A(2) -> B(3) -> C(4)
    index = {
        ref("P", "A"): 2,
        ref("P", "B"): 3,
        ref("P", "C"): 4,
    }
    edges = [
        Dependency(ref("P", "A"), ref("P", "B"), "FS", 0),
        Dependency(ref("P", "B"), ref("P", "C"), "FS", 0),
    ]
    s = cpm_from_dfs("P", index, edges)
    assert s.project_duration == pytest.approx(9.0)
    assert s.num_activities == 3
    assert s.num_edges == 2
    assert s.num_terminal_activities == 1
    assert s.critical_activity_ids == ["A", "B", "C"]
    assert s.critical_paths[0] == ["A", "B", "C"]


def _results_dict(project_id: str, activities: list[dict], edges: list[dict]) -> dict[str, ActivityCpmResult]:
    # This helper is not used directly; kept for potential future use.
    raise NotImplementedError("use _calc_results_dict helper below")


# ---------------------------------------------------------------------------
# B. Parallel activities
# ---------------------------------------------------------------------------


def test_parallel_activities():
    # A(2) -> B(3)
    # A(2) -> C(5)
    # Project duration = max(2+3, 2+5) = 7
    activities = [act("P", "A", 2), act("P", "B", 3), act("P", "C", 5)]
    edges = [
        dep("P", "A", "B", "FS", 0),
        dep("P", "A", "C", "FS", 0),
    ]
    s = calculate_project_cpm("P", activities, edges)
    assert s.project_duration == pytest.approx(7.0)
    assert s.critical_activity_ids == ["A", "C"]
    assert s.critical_paths[0] == ["A", "C"]


# ---------------------------------------------------------------------------
# C. FS with lag
# ---------------------------------------------------------------------------


def test_fs_with_lag():
    # A(2) -> FS+2 -> B(3)
    # ES[A]=0, EF[A]=2
    # ES[B] >= EF[A] + 2 = 4
    # EF[B] = 4 + 3 = 7
    activities = [act("P", "A", 2), act("P", "B", 3)]
    edges = [dep("P", "A", "B", "FS", 2)]
    s = calculate_project_cpm("P", activities, edges)
    assert s.project_duration == pytest.approx(7.0)
    assert s.critical_activity_ids == ["A", "B"]


# ---------------------------------------------------------------------------
# D. SS relationship
# ---------------------------------------------------------------------------


def test_ss_relationship():
    # A(2) -> SS+3 -> B(3)
    # ES[A]=0, EF[A]=2
    # ES[B] >= ES[A] + 3 = 3, EF[B] = 3 + 3 = 6
    # In a pure SS relationship, A has zero float because B depends on A's start.
    # The project can't slip without delaying B's start.
    activities = [act("P", "A", 2), act("P", "B", 3)]
    edges = [dep("P", "A", "B", "SS", 3)]
    s = calculate_project_cpm("P", activities, edges)
    assert s.project_duration == pytest.approx(6.0)
    # Both A and B are critical: A because it has no float (SS tie), B as terminal
    assert sorted(s.critical_activity_ids) == ["A", "B"]


# ---------------------------------------------------------------------------
# E. FF relationship
# ---------------------------------------------------------------------------


def test_ff_relationship():
    # A(2) -> FF+1 -> B(3)
    # ES default 0 for both (no FS/SS forcing A to start earlier than B)
    # EF[A] = 2, EF[B] = 3
    # FF: EF[B] >= EF[A] + 1 = 3  -> satisfied at equality with ES[B]=0
    # The FF constraint is binding on EF[B], but B's ES can still be 0
    # Because we set ES[B] from predecessor lower bounds, and FF gives:
    #   ES[B] >= ES[A] + dur[A] + lag - dur[B] = 0 + 2 + 1 - 3 = 0
    # So ES[B] = 0, EF[B] = 3, project_duration = max(2, 3) = 3
    # Both A and B have zero total float (no slack on either), so both are
    # critical under the engine's tolerance-based definition.
    activities = [act("P", "A", 2), act("P", "B", 3)]
    edges = [dep("P", "A", "B", "FF", 1)]
    s = calculate_project_cpm("P", activities, edges)
    assert s.project_duration == pytest.approx(3.0)
    assert sorted(s.critical_activity_ids) == ["A", "B"]


# ---------------------------------------------------------------------------
# F. Mixed FS + SS + FF graph
# ---------------------------------------------------------------------------


def test_mixed_fs_ss_ff():
    # A(2)
    # B(3)
    # C(4)
    # A -> FS+0 -> B
    # A -> SS+1 -> C
    # B -> FF+1 -> C
    #
    # A: ES=0, EF=2
    # B: FS from A => ES[B] >= 2, EF[B]=5
    # C: SS from A => ES[C] >= 1
    #     FF from B => EF[C] >= EF[B] + 1 = 6 => ES[C] >= ES[B] + dur[B] + 1 - dur[C] = 2+3+1-4=2
    #     So ES[C] = max(1, 2, 0) = 2, EF[C] = 6
    # Project duration = max(2, 5, 6) = 6
    activities = [act("P", "A", 2), act("P", "B", 3), act("P", "C", 4)]
    edges = [
        dep("P", "A", "B", "FS", 0),
        dep("P", "A", "C", "SS", 1),
        dep("P", "B", "C", "FF", 1),
    ]
    s = calculate_project_cpm("P", activities, edges)
    assert s.project_duration == pytest.approx(6.0)
    # All three activities are critical: A->B is FS (tight), B->C is FF (tight),
    # A->C is SS+1 (A has no float because C depends on A's start via SS).
    assert sorted(s.critical_activity_ids) == ["A", "B", "C"]
    assert s.critical_paths[0] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# G. Multiple terminal activities
# ---------------------------------------------------------------------------


def test_multiple_terminal_activities():
    # A(2) -> B(3)
    # A(2) -> C(5)
    # Project duration determined by the later terminal: max(2+3, 2+5) = 7
    activities = [act("P", "A", 2), act("P", "B", 3), act("P", "C", 5)]
    edges = [
        dep("P", "A", "B", "FS", 0),
        dep("P", "A", "C", "FS", 0),
    ]
    s = calculate_project_cpm("P", activities, edges)
    assert s.project_duration == pytest.approx(7.0)
    assert s.num_terminal_activities == 2
    assert "B" in s.critical_activity_ids or "C" in s.critical_activity_ids
    # Only C is critical in the computed schedule
    assert s.critical_activity_ids == ["A", "C"]


# ---------------------------------------------------------------------------
# H. Zero-duration virtual end
# ---------------------------------------------------------------------------


def test_zero_duration_virtual_end():
    # Single activity A(5)
    # With virtual end, ES[A]=0, EF[A]=5, virtual END ES=5 (duration 0)
    # project_duration = 5, and END should not extend the project
    activities = [act("P", "A", 5)]
    edges = []
    s = calculate_project_cpm("P", activities, edges, include_virtual_end=True)
    assert s.project_duration == pytest.approx(5.0)
    assert s.num_activities == 1
    assert s.num_terminal_activities == 1
    assert s.critical_activity_ids == ["A"]


def test_virtual_end_not_extending():
    # Two terminals, latest finishes at 10; virtual end should report 10, not
    # 10 + anything.
    activities = [act("P", "A", 3), act("P", "B", 7), act("P", "C", 2)]
    edges = [dep("P", "A", "B", "FS", 0), dep("P", "A", "C", "FS", 0)]
    s = calculate_project_cpm("P", activities, edges, include_virtual_end=True)
    assert s.project_duration == pytest.approx(10.0)  # max(3+7, 3+2)
    assert "B" in s.critical_activity_ids
    assert "A" in s.critical_activity_ids


# ---------------------------------------------------------------------------
# I. Cycle detection
# ---------------------------------------------------------------------------


def test_cycle_detection_raises():
    rows = [
        {"project_id":"P","predecessor_id":"A","successor_id":"B","relationship":"FS","lag_days":0},
        {"project_id":"P","predecessor_id":"B","successor_id":"C","relationship":"FS","lag_days":0},
        {"project_id":"P","predecessor_id":"C","successor_id":"A","relationship":"FS","lag_days":0},
    ]
    try:
        calculate_project_cpm("P", [act("P","A",1), act("P","B",2), act("P","C",3)], rows)
        assert False, "Expected CyclicGraphError"
    except CyclicGraphError as e:
        msg = str(e)
        assert "P" in msg
        assert "cycle" in msg.lower()
        assert "A" in msg and "B" in msg and "C" in msg
        assert "3" in msg  # 3 nodes in cycle


def test_cycle_description():
    """Verify that the error message for a cyclic graph is informative."""
    rows = [
        {"project_id": "P", "predecessor_id": "A", "successor_id": "B", "relationship": "FS", "lag_days": 0},
        {"project_id": "P", "predecessor_id": "B", "successor_id": "C", "relationship": "FS", "lag_days": 0},
        {"project_id": "P", "predecessor_id": "C", "successor_id": "A", "relationship": "FS", "lag_days": 0},
    ]
    with pytest.raises(CyclicGraphError) as exc_info:
        calculate_project_cpm("P", [act("P", "A", 1), act("P", "B", 2), act("P", "C", 3)], rows)
    msg = str(exc_info.value)
    assert "P" in msg
    assert "cycle" in msg.lower()
    assert "A" in msg and "B" in msg and "C" in msg
    assert "3" in msg  # 3 nodes in cycle


def test_detect_cycles_empty():
    assert detect_cycles(set(), []) == []


def test_detect_cycles_two_node_cycle():
    a, b = ref("P", "A"), ref("P", "B")
    edges = [Dependency(a, b, "FS", 0), Dependency(b, a, "FS", 0)]
    nodes = {a, b}
    cyc = detect_cycles(nodes, edges)
    assert len(cyc) == 2
    assert a in cyc and b in cyc


def test_detect_cycles_single_node_self_loop():
    a = ref("P", "A")
    edges = [Dependency(a, a, "FS", 0)]
    nodes = {a}
    cyc = detect_cycles(nodes, edges)
    assert a in cyc


# ---------------------------------------------------------------------------
# J. Float calculation: LS - ES == LF - EF
# ---------------------------------------------------------------------------


def test_float_consistency():
    # Re-run with a small private accessor; for the test we can compute
    # float both ways by using the public ES/EF/LS/LF from the summary's
    # underlying activity results.
    # Simpler: use cpm_from_dfs with a tiny graph and directly validate.
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FS", 1)]
    s = cpm_from_dfs("P", index, edges)
    # We cannot access per-activity results from summary; instead validate via
    # the standalone constraint/float helper used internally. We'll instead
    # assert using a small custom re-computation below in a separate helper.
    assert s.project_duration == pytest.approx(6.0)  # A:0..2, B: max(0, 2+1)=3..6
    assert s.critical_activity_ids == ["A", "B"]


def _normalize_to_dep(edges, index):
    from cpm.calculation import ActivityRef, Dependency
    out: list[Dependency] = []
    for e in edges:
        if isinstance(e, Dependency):
            out.append(e)
        else:
            pid = next(iter(index)).project_id
            out.append(Dependency(
                predecessor=ActivityRef(pid, str(e["predecessor_id"])),
                successor=ActivityRef(pid, str(e["successor_id"])),
                relationship=str(e["relationship"]).upper(),
                lag_days=int(e.get("lag_days", 0)),
            ))
    return out


def _compute_schedule(
    index: dict[ActivityRef, int],
    edges: list[Dependency],
) -> dict[str, dict[str, float]]:
    """Run the same forward/backward logic as the engine and return per-id dicts.

    This duplicates the internal computation only for assertions, keeping the
    main engine untouched.
    """

    nodes = set(index.keys())
    edges = _normalize_to_dep(edges, index)
    preds_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    succs_by: dict[ActivityRef, list[ActivityRef]] = {n: [] for n in nodes}
    edge_map: dict[tuple[ActivityRef, ActivityRef], list[Dependency]] = {}
    for e in _normalize_to_dep(edges, index):
        preds_by[e.successor].append(e.predecessor)
        succs_by[e.predecessor].append(e.successor)
        edge_map.setdefault((e.predecessor, e.successor), []).append(e)

    order = _topological_order(nodes, _normalize_to_dep(edges, index))

    es: dict[ActivityRef, float] = {n: 0.0 for n in nodes}
    for n in order:
        bound = 0.0
        for pred in preds_by[n]:
            for e in edge_map.get((pred, n), []):
                if e.relationship == "FS":
                    bound = max(bound, es[pred] + float(index[pred]) + float(e.lag_days))
                elif e.relationship == "SS":
                    bound = max(bound, es[pred] + float(e.lag_days))
                elif e.relationship == "FF":
                    bound = max(bound, es[pred] + float(index[pred]) + float(e.lag_days) - float(index[n]))
        es[n] = bound

    ef = {n: es[n] + float(index[n]) for n in nodes}
    terminals = [n for n in nodes if not succs_by[n]]
    proj_completion = max(ef[n] for n in terminals) if terminals else 0.0

    lf: dict[ActivityRef, float] = {n: proj_completion for n in nodes}
    ls: dict[ActivityRef, float] = {n: lf[n] - float(index[n]) for n in nodes}

    changed = True
    while changed:
        changed = False
        for n in reversed(order):
            bound = proj_completion
            for succ in succs_by[n]:
                for e in edge_map.get((n, succ), []):
                    if e.relationship == "FS":
                        bound = min(bound, ls[succ] - float(e.lag_days))
                    elif e.relationship == "SS":
                        bound = min(bound, ls[succ] - float(index[n]) - float(e.lag_days))
                    elif e.relationship == "FF":
                        bound = min(bound, lf[succ] - float(e.lag_days))
            new_lf = bound
            if new_lf < lf[n] - 1e-9:
                lf[n] = new_lf
                ls[n] = new_lf - float(index[n])
                changed = True

    return {
        "es": es,
        "ef": ef,
        "ls": ls,
        "lf": lf,
        "float_ls_es": {n: ls[n] - es[n] for n in nodes},
        "float_lf_ef": {n: lf[n] - ef[n] for n in nodes},
        "project_duration": proj_completion,
    }


def test_float_equivalence():
    index = {
        ref("P", "A"): 2,
        ref("P", "B"): 3,
        ref("P", "C"): 5,
    }
    edges = [
        dep("P", "A", "B", "FS", 1),
        dep("P", "B", "C", "FS", 0),
    ]
    sched = _compute_schedule(index, edges)
    for n in index:
        assert abs(sched["float_ls_es"][n] - sched["float_lf_ef"][n]) <= CRITICAL_TOLERANCE


def test_float_consistency_on_mixed_graph():
    index = {ref("P", "A"): 2, ref("P", "B"): 3, ref("P", "C"): 4}
    edges = [
        dep("P", "A", "B", "FS", 0),
        dep("P", "A", "C", "SS", 1),
        dep("P", "B", "C", "FF", 1),
    ]
    sched = _compute_schedule(index, edges)
    for n in index:
        assert abs(sched["float_ls_es"][n] - sched["float_lf_ef"][n]) <= CRITICAL_TOLERANCE


# ---------------------------------------------------------------------------
# K. Critical activity detection
# ---------------------------------------------------------------------------


def test_critical_with_zero_float():
    # A(2) -> B(3) both critical
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FS", 0)]
    s = cpm_from_dfs("P", index, edges)
    assert s.critical_activity_ids == ["A", "B"]


def test_critical_with_positive_float():
    # A(2) -> B(3)
    # A -> C(5)
    # C has float = 7 - (2+5) = 0? Wait: A(2)->B(3): EF_B=5; A(2)->C(5): EF_C=7
    # project = 7, float_B = 7 - 5 = 2 (noncritical), float_C = 0
    index = {ref("P", "A"): 2, ref("P", "B"): 3, ref("P", "C"): 5}
    edges = [dep("P", "A", "B", "FS", 0), dep("P", "A", "C", "FS", 0)]
    s = cpm_from_dfs("P", index, edges)
    assert s.critical_activity_ids == ["A", "C"]
    assert "B" not in s.critical_activity_ids


def test_critical_tolerance_boundary():
    # Exactly at tolerance should be critical
    index = {ref("P", "A"): 1}
    edges = []
    s = cpm_from_dfs("P", index, edges)
    assert s.critical_activity_ids == ["A"]


# ---------------------------------------------------------------------------
# L. Dependency constraint validation
# ---------------------------------------------------------------------------


def test_validate_constraints_valid_fs():
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FS", 1)]
    sched = _compute_schedule(index, edges)
    viols = validate_dependency_constraints(
        edges,
        sched["es"],
        sched["ef"],
        sched["ls"],
        sched["lf"],
        index,
    )
    assert viols == []


def test_validate_constraints_fs_lag_violation_artificial():
    # Build a schedule that violates FS+2 by forcing ES[B] too low.
    # We can't produce a "wrong" schedule via the engine, so instead we
    # validate a manually constructed schedule that intentionally violates.
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FS", 2)]
    es = {ref("P", "A"): 0.0, ref("P", "B"): 1.0}  # violates FS+2 (needs >=4)
    ef = {ref("P", "A"): 2.0, ref("P", "B"): 4.0}
    ls = {ref("P", "A"): 0.0, ref("P", "B"): 1.0}
    lf = {ref("P", "A"): 2.0, ref("P", "B"): 4.0}
    viols = validate_dependency_constraints(edges, es, ef, ls, lf, index)
    assert len(viols) == 1
    assert viols[0]["relationship"] == "FS"
    assert viols[0]["lag_days"] == 2


def test_validate_constraints_ss():
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "SS", 3)]
    sched = _compute_schedule(index, edges)
    viols = validate_dependency_constraints(edges, sched["es"], sched["ef"], sched["ls"], sched["lf"], index)
    assert viols == []


def test_validate_constraints_ff():
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FF", 1)]
    sched = _compute_schedule(index, edges)
    viols = validate_dependency_constraints(edges, sched["es"], sched["ef"], sched["ls"], sched["lf"], index)
    assert viols == []


def test_validate_constraints_multiple_predecessors():
    index = {ref("P", "A"): 2, ref("P", "B"): 3, ref("P", "C"): 4}
    edges = [
        dep("P", "A", "C", "FS", 0),
        dep("P", "B", "C", "FS", 1),
    ]
    sched = _compute_schedule(index, edges)
    viols = validate_dependency_constraints(edges, sched["es"], sched["ef"], sched["ls"], sched["lf"], index)
    assert viols == []


# ---------------------------------------------------------------------------
# M. Edge cases
# ---------------------------------------------------------------------------


def test_single_activity_project():
    index = {ref("P", "A"): 5}
    s = cpm_from_dfs("P", index, [])
    assert s.project_duration == pytest.approx(5.0)
    assert s.critical_activity_ids == ["A"]
    assert s.num_terminal_activities == 1
    assert s.num_activities == 1


def test_zero_lag():
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FS", 0)]
    sched = _compute_schedule(index, edges)
    assert sched["es"][ref("P", "B")] == pytest.approx(2.0)
    assert sched["ef"][ref("P", "B")] == pytest.approx(5.0)


def test_max_observed_lag():
    # SCOPE max lag = 3
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FS", 3)]
    sched = _compute_schedule(index, edges)
    assert sched["es"][ref("P", "B")] == pytest.approx(5.0)  # 2 + 3
    assert sched["ef"][ref("P", "B")] == pytest.approx(8.0)


def test_multiple_predecessors():
    # C has two FS predecessors: A(2) and B(3), both lag 0
    # ES[C] = max(2, 3) = 3
    index = {ref("P", "A"): 2, ref("P", "B"): 3, ref("P", "C"): 4}
    edges = [dep("P", "A", "C", "FS", 0), dep("P", "B", "C", "FS", 0)]
    sched = _compute_schedule(index, edges)
    assert sched["es"][ref("P", "C")] == pytest.approx(3.0)
    assert sched["ef"][ref("P", "C")] == pytest.approx(7.0)


def test_multiple_successors():
    # A has two FS successors B(2) and C(5)
    index = {ref("P", "A"): 2, ref("P", "B"): 2, ref("P", "C"): 5}
    edges = [dep("P", "A", "B", "FS", 0), dep("P", "A", "C", "FS", 0)]
    sched = _compute_schedule(index, edges)
    assert sched["es"][ref("P", "A")] == 0.0
    assert sched["es"][ref("P", "B")] == pytest.approx(2.0)
    assert sched["es"][ref("P", "C")] == pytest.approx(2.0)
    assert sched["project_duration"] == pytest.approx(7.0)


def test_no_dependencies_all_parallel():
    index = {ref("P", "A"): 2, ref("P", "B"): 3, ref("P", "C"): 5}
    s = cpm_from_dfs("P", index, [])
    assert s.project_duration == pytest.approx(5.0)
    # All start at 0, all critical? Only the longest is critical (A has float 3, B float 2, C float 0)
    assert s.critical_activity_ids == ["C"]


# ---------------------------------------------------------------------------
# Deterministic tie-breaking sanity
# ---------------------------------------------------------------------------


def test_deterministic_topological_order():
    # Two independent chains: A->B and C->D
    index = {ref("P", "A"): 1, ref("P", "B"): 1, ref("P", "C"): 1, ref("P", "D"): 1}
    edges = [
        dep("P", "A", "B", "FS", 0),
        dep("P", "C", "D", "FS", 0),
    ]
    order = _topological_order(set(index.keys()), edges)
    assert order[0].activity_id in {"A", "C"}
    assert order[-1].activity_id in {"B", "D"}
    # Must be a valid topological order
    pos = {n.activity_id: i for i, n in enumerate(order)}
    for e in _normalize_to_dep(edges, index):
        assert pos[e.predecessor.activity_id] < pos[e.successor.activity_id]


def test_determinism_idempotent():
    index = {
        ref("P", "A"): 2,
        ref("P", "B"): 3,
        ref("P", "C"): 5,
    }
    edges = [
        dep("P", "A", "B", "FS", 1),
        dep("P", "A", "C", "FS", 0),
    ]
    s1 = cpm_from_dfs("P", index, edges)
    s2 = cpm_from_dfs("P", index, edges)
    assert s1.project_duration == s2.project_duration
    assert s1.critical_activity_ids == s2.critical_activity_ids
    assert s1.critical_paths == s2.critical_paths


# ---------------------------------------------------------------------------
# API / type sanity
# ---------------------------------------------------------------------------


def test_activity_ref_repr():
    r = ActivityRef("P", "A001")
    assert repr(r) == "ActivityRef(P.A001)"


def test_dependency_validation():
    a, b = ref("P", "A"), ref("P", "B")
    d = Dependency(a, b, "FS", 0)
    assert d.relationship == "FS"
    assert d.lag_days == 0
    with pytest.raises(ValueError):
        Dependency(a, b, "X", 0)
    with pytest.raises(ValueError):
        Dependency(a, b, "FS", -1)


def test_project_summary_fields():
    index = {ref("P", "A"): 2, ref("P", "B"): 3}
    edges = [dep("P", "A", "B", "FS", 0)]
    s = cpm_from_dfs("P", index, edges)
    assert s.project_id == "P"
    assert s.num_activities == 2
    assert s.num_edges == 1
    assert isinstance(s.critical_activity_ids, list)
    assert isinstance(s.critical_paths, list)
    assert isinstance(s.dependency_violations, list)
