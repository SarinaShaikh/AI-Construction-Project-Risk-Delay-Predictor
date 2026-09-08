"""
Phase 2 — focused tests for the SCOPE v0.2 cleaning pipeline (scripts/clean_scope.py).

Two layers:

1. A module-scoped fixture runs the pipeline once into a fresh temporary
   processed dir. It asserts the run exits 0, that raw files are untouched,
   and that the fresh outputs are byte-identical to data/processed/ (the
   committed outputs) — proving the pipeline is deterministic/reproducible.

2. Data-integrity tests read the processed outputs and independently verify:
   row counts, primary-key uniqueness, foreign keys, dependency integrity,
   acyclicity, the 70/15/15 project split, and the event-delay target.

Run with:  uv run pytest tests/ -v        (or)   python -m pytest tests/ -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import clean_scope

RAW_DIR = clean_scope.DEFAULT_RAW_DIR
PROCESSED_DIR = clean_scope.DEFAULT_PROCESSED_DIR
TABLES = list(clean_scope.TABLE_SPECS.keys())
PK = {t: clean_scope.TABLE_SPECS[t]["pk"] for t in TABLES}

RAW_FILES = sorted(p.name for p in RAW_DIR.glob("*.csv"))


def _snapshot_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_raw() -> dict[str, tuple[int, int, str]]:
    """Return {filename: (size, mtime_ns, sha256)} for every raw file."""
    snap = {}
    for p in RAW_DIR.glob("*.csv"):
        st = p.stat()
        snap[p.name] = (st.st_size, st.st_mtime_ns, _snapshot_sha256(p))
    return snap


def _read(table: str, usecols=None) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{table}.csv"
    return pd.read_csv(path, dtype="str", keep_default_na=False, usecols=usecols)


# ---------------------------------------------------------------------------
# Fixture: one fresh full run per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("p2_fresh")
    raw_before = _snapshot_raw()
    rc = clean_scope.main(["--processed-dir", str(out)])
    raw_after = _snapshot_raw()
    return {"out": out, "rc": rc, "raw_before": raw_before, "raw_after": raw_after}


# ---------------------------------------------------------------------------
# 1. Pipeline execution / reproducibility
# ---------------------------------------------------------------------------


def test_pipeline_exits_zero(fresh_run):
    assert fresh_run["rc"] == 0


def test_output_files_exist(fresh_run):
    expected = [f"{t}.csv" for t in TABLES] + [
        "targets_event_delay_days.csv",
        "prediction_time_availability.csv",
        "cleaning_manifest.json",
    ]
    missing = [f for f in expected if not (fresh_run["out"] / f).is_file()]
    assert missing == [], f"missing outputs: {missing}"


def test_raw_files_unchanged(fresh_run):
    assert fresh_run["raw_before"] == fresh_run["raw_after"], (
        "raw files changed during the pipeline run"
    )


def test_deterministic_outputs(fresh_run):
    """Fresh run outputs must be byte-identical to the data/processed outputs."""
    for f in [f"{t}.csv" for t in TABLES] + [
        "targets_event_delay_days.csv",
        "prediction_time_availability.csv",
    ]:
        assert (PROCESSED_DIR / f).is_file(), f"data/processed/{f} missing"
        assert (PROCESSED_DIR / f).read_bytes() == (fresh_run["out"] / f).read_bytes(), (
            f"output {f} differs between runs — pipeline is not deterministic"
        )


def test_manifest_records_all_pass(fresh_run):
    manifest = json.loads((fresh_run["out"] / "cleaning_manifest.json").read_text())
    assert manifest["overall"]["all_valid"] is True
    assert manifest["overall"]["failures"] == []


# ---------------------------------------------------------------------------
# 2. Row counts (no unexpected row loss)
# ---------------------------------------------------------------------------

RAW_COUNTS = {
    "projects": 100,
    "activities": 9279,
    "dependencies": 18176,
    "resources": 2748,
    "resource_allocation": 9279,
    "environment": 52332,
    "activity_states": 4060623,
    "procurement": 800,
    "events": 27989,
    "decisions": 3374,
    "rework": 1858,
    "outcomes": 100,
}


@pytest.mark.parametrize("table", TABLES)
def test_row_counts_preserved(table):
    assert len(_read(table)) == RAW_COUNTS[table]


# ---------------------------------------------------------------------------
# 3. Primary-key uniqueness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", TABLES)
def test_primary_keys_unique(table):
    df = _read(table, usecols=PK[table])
    assert int(df.duplicated(subset=PK[table]).sum()) == 0


def test_ids_not_empty():
    for table in TABLES:
        df = _read(table)
        for c in PK[table]:
            assert int((df[c] == "").sum()) == 0, f"{table}.{c} has empty values"


# ---------------------------------------------------------------------------
# 4. Foreign keys
# ---------------------------------------------------------------------------


def _key_set(df, cols):
    return set(zip(*(df[c] for c in cols))) if len(cols) > 1 else set(df[cols[0]])


def test_fk_dependencies_reference_activities():
    deps = _read("dependencies", usecols=["project_id", "predecessor_id", "successor_id"])
    acts = _read("activities", usecols=["project_id", "activity_id"])
    act_keys = _key_set(acts, ["project_id", "activity_id"])
    assert all(k in act_keys for k in _key_set(deps, ["project_id", "predecessor_id"]))
    assert all(k in act_keys for k in _key_set(deps, ["project_id", "successor_id"]))


def test_fk_resource_allocation():
    ra = _read("resource_allocation", usecols=["project_id", "activity_id", "resource_id"])
    acts = _read("activities", usecols=["project_id", "activity_id"])
    res = _read("resources", usecols=["resource_id"])
    act_keys = _key_set(acts, ["project_id", "activity_id"])
    res_keys = set(res["resource_id"])
    assert all(k in act_keys for k in _key_set(ra, ["project_id", "activity_id"]))
    assert set(ra["resource_id"]) <= res_keys


def test_fk_events_decisions_rework():
    acts = _read("activities", usecols=["project_id", "activity_id"])
    act_keys = _key_set(acts, ["project_id", "activity_id"])
    ev = _read("events", usecols=["project_id", "activity_id"])
    dec = _read("decisions", usecols=["project_id", "activity_id"])
    rw = _read("rework", usecols=["project_id", "activity_id", "event_id"])
    ev_ids = set(_read("events", usecols=["event_id"])["event_id"])
    assert all(k in act_keys for k in _key_set(ev, ["project_id", "activity_id"]))
    assert all(k in act_keys for k in _key_set(dec, ["project_id", "activity_id"]))
    assert all(k in act_keys for k in _key_set(rw, ["project_id", "activity_id"]))
    assert set(rw["event_id"]) <= ev_ids


def test_fk_project_ids():
    proj = set(_read("projects", usecols=["project_id"])["project_id"])
    for table in ["resources", "environment", "procurement", "outcomes"]:
        vals = set(_read(table, usecols=["project_id"])["project_id"])
        assert vals <= proj, f"{table}.project_id references unknown project"


# ---------------------------------------------------------------------------
# 5. Dependency integrity + acyclicity
# ---------------------------------------------------------------------------


def test_dependencies_relationships_and_lags():
    deps = _read("dependencies")
    assert set(deps["relationship"]) <= {"FS", "FF", "SS"}
    lags = pd.to_numeric(deps["lag_days"])
    assert int((lags < 0).sum()) == 0
    assert lags.min() == 0 and lags.max() == 3


def test_dependencies_intra_project():
    deps = _read("dependencies")
    wrong = deps[deps["project_id"] != deps["predecessor_id"].str.split("_").str[0]]
    wrong2 = deps[deps["project_id"] != deps["successor_id"].str.split("_").str[0]]
    assert len(wrong) == 0 and len(wrong2) == 0


def test_no_cycles_in_any_project():
    deps = _read("dependencies")
    activities = _read("activities", usecols=["project_id", "activity_id"])
    nodes: dict[str, set[str]] = {}
    for pid, aid in zip(activities["project_id"], activities["activity_id"]):
        nodes.setdefault(pid, set()).add(aid)
    edges_by_project: dict[str, list[tuple[str, str]]] = {}
    for row in deps.itertuples(index=False):
        edges_by_project.setdefault(row.project_id, []).append(
            (row.predecessor_id, row.successor_id)
        )
    for pid, edges in edges_by_project.items():
        assert not clean_scope.kahn_has_cycle(nodes[pid], edges), f"cycle in {pid}"


def test_kahn_cycle_unit():
    assert clean_scope.kahn_has_cycle({"a", "b", "c"}, [("a", "b"), ("b", "c"), ("c", "a")])
    assert not clean_scope.kahn_has_cycle({"a", "b", "c"}, [("a", "b"), ("b", "c")])
    assert not clean_scope.kahn_has_cycle({"a", "b", "c"}, [])


# ---------------------------------------------------------------------------
# 6. Project split (70/15/15, project-level, consistent)
# ---------------------------------------------------------------------------


def test_project_split_counts():
    proj = _read("projects")
    counts = proj["split"].value_counts().to_dict()
    assert counts == {"train": 70, "validation": 15, "test": 15}


def test_split_consistent_across_tables():
    proj = _read("projects", usecols=["project_id", "split"])
    expected = dict(zip(proj["project_id"], proj["split"]))
    for table in TABLES:
        df = _read(table, usecols=["project_id", "split"])
        mism = int((df["split"] != df["project_id"].map(expected)).sum())
        assert mism == 0, f"{table} has split inconsistencies"


def test_split_is_project_level():
    proj = _read("projects", usecols=["project_id", "split"])
    assert int(proj.duplicated(subset=["project_id"]).sum()) == 0


# ---------------------------------------------------------------------------
# 7. Target: one value per activity, correct aggregation, no leakage of events
# ---------------------------------------------------------------------------


def test_target_one_row_per_activity():
    acts = _read("activities", usecols=["project_id", "activity_id"])
    tgt = _read("targets_event_delay_days")
    assert set(tgt.columns) == {"project_id", "activity_id", "target_event_delay_days"}
    assert len(tgt) == len(acts)
    assert int(tgt.duplicated(["project_id", "activity_id"]).sum()) == 0
    assert set(zip(tgt["project_id"], tgt["activity_id"])) == set(
        zip(acts["project_id"], acts["activity_id"])
    )


def test_target_values_valid():
    tgt = _read("targets_event_delay_days")
    vals = pd.to_numeric(tgt["target_event_delay_days"])
    assert int(vals.isna().sum()) == 0
    assert int((vals < 0).sum()) == 0
    assert vals.min() == 0 and vals.max() == 168


def test_target_matches_raw_events_aggregation():
    """Independent recomputation from the RAW events file must match exactly."""
    raw_events = pd.read_csv(
        RAW_DIR / "events.csv", usecols=["project_id", "activity_id", "duration_days"]
    )
    raw_events["duration_days"] = pd.to_numeric(raw_events["duration_days"])
    sums = (
        raw_events.groupby(["project_id", "activity_id"], as_index=False)["duration_days"]
        .sum()
        .rename(columns={"duration_days": "expected_raw"})
    )
    tgt = _read("targets_event_delay_days")
    tgt_vals = pd.to_numeric(tgt["target_event_delay_days"])
    merged = tgt.merge(sums, on=["project_id", "activity_id"], how="left")
    merged["expected_raw"] = merged["expected_raw"].fillna(0)
    assert int((merged["expected_raw"] != tgt_vals).sum()) == 0


def test_zero_target_activities_have_no_events():
    raw_events = pd.read_csv(RAW_DIR / "events.csv", usecols=["activity_id"])
    evented = set(raw_events["activity_id"])
    tgt = _read("targets_event_delay_days")
    zeros = tgt[pd.to_numeric(tgt["target_event_delay_days"]) == 0]
    assert len(zeros) == 523
    assert not (set(zeros["activity_id"]) & evented)


def test_target_statistics():
    tgt = _read("targets_event_delay_days")
    vals = pd.to_numeric(tgt["target_event_delay_days"])
    assert vals.mean() == pytest.approx(16.7741, abs=1e-3)
    assert vals.median() == 13.0
    assert vals.quantile(0.99) == 67.0


# ---------------------------------------------------------------------------
# 8. Availability classification sanity
# ---------------------------------------------------------------------------


def test_prediction_time_availability_classification():
    avail = pd.read_csv(PROCESSED_DIR / "prediction_time_availability.csv", dtype="str")
    assert len(avail) == sum(len(s["cols"]) for s in clean_scope.TABLE_SPECS.values())
    assert set(avail["availability"]) <= {
        "AVAILABLE_AT_PREDICTION_TIME",
        "OUTCOME_OR_POST_OUTCOME",
        "UNKNOWN",
    }
    # event-derived table must be classified as outcome info (leakage rule)
    ev_cols = avail[avail["table"] == "events"]
    assert set(ev_cols["availability"]) == {"OUTCOME_OR_POST_OUTCOME"}
    # rework (post-event outcomes) and decision actuals must not be "available"
    rw_outcome = avail[avail["table"] == "rework"]
    assert set(rw_outcome["availability"]) == {"OUTCOME_OR_POST_OUTCOME"}