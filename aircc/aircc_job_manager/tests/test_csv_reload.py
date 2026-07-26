"""Unit tests for live CSV reload: the manager re-reads (and re-validates) the
CSVs before EVERY claim, and pushes CSV priority/epochs onto still-pending DB
rows, so editing a CSV while the sbatch runs reaches every model not yet claimed
-- see csv_spec.load_spec, db.sync_pending_spec and JobManager._worker.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

import pytest

from aircc.aircc_job_manager import job_manager as jm
from aircc.aircc_job_manager.csv_spec import ALL_COLUMNS, load_spec
from aircc.aircc_job_manager.db import AirccDB


def _row(name: str, *, priority: int = 0, epochs: int = 100, dep: str = "", **extra) -> dict:
    row = {c: "" for c in ALL_COLUMNS}
    row.update(
        model_name=name,
        arch="convnext_base",
        init="0",
        init_mode="scratch",
        dependency_model_name=dep,
        priority=str(priority),
        **{"training.epochs": str(epochs), "training.batch_size": "256"},
    )
    row.update(extra)
    return row


def _write_csv(csv_dir: Path, rows: list[dict], arch: str = "convnext_base") -> Path:
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"{arch}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture()
def db(tmp_path):
    return AirccDB(str(tmp_path / "jobs.sqlite"))


# ---- 1. strict loader -----------------------------------------------------
def test_load_spec_happy_path(tmp_path):
    _write_csv(tmp_path / "csv", [_row("A"), _row("B", priority=1, dep="A")])
    rows, by_name, deps = load_spec(tmp_path / "csv")
    assert len(rows) == 2
    assert set(by_name) == {"A", "B"}
    assert by_name["B"]["training.epochs"] == "100"
    assert deps == {"A": "", "B": "A"}


def test_load_spec_rejects_empty_dir(tmp_path):
    (tmp_path / "csv").mkdir()
    with pytest.raises(ValueError, match="no arch CSVs"):
        load_spec(tmp_path / "csv")


def test_load_spec_rejects_blank_model_name(tmp_path):
    _write_csv(tmp_path / "csv", [_row("A"), _row("")])
    with pytest.raises(ValueError, match="blank model_name"):
        load_spec(tmp_path / "csv")


def test_load_spec_rejects_duplicate_model_name(tmp_path):
    _write_csv(tmp_path / "csv", [_row("A"), _row("A", priority=1)])
    with pytest.raises(ValueError, match="duplicate model_name"):
        load_spec(tmp_path / "csv")


def test_load_spec_rejects_non_int_priority(tmp_path):
    bad = _row("B")
    bad["priority"] = ""
    _write_csv(tmp_path / "csv", [_row("A"), bad])
    with pytest.raises(ValueError, match="priority"):
        load_spec(tmp_path / "csv")


def test_load_spec_rejects_non_int_epochs(tmp_path):
    bad = _row("A")
    bad["training.epochs"] = "many"
    _write_csv(tmp_path / "csv", [bad])
    with pytest.raises(ValueError, match="training.epochs"):
        load_spec(tmp_path / "csv")


# ---- 2. diff-only DB sync -------------------------------------------------
def test_sync_pending_spec_only_touches_pending_unclaimed(db):
    for name in ("pend", "run", "done", "parked"):
        db.upsert_pending(name, 100, 5)
    db._write("UPDATE jobs SET status='running', owner_task=7 WHERE model_name='run'", ())
    db._write("UPDATE jobs SET status='finished' WHERE model_name='done'", ())
    db._write("UPDATE jobs SET owner_task=-1 WHERE model_name='parked'", ())

    spec = {n: (200, 10) for n in ("pend", "run", "done", "parked")}
    assert db.sync_pending_spec(spec) == ["pend"]

    assert (db.get("pend").total_epochs, db.get("pend").priority) == (200, 10)
    for name in ("run", "done", "parked"):
        j = db.get(name)
        assert (j.total_epochs, j.priority) == (100, 5), name

    # idempotent: a second identical sync writes nothing
    assert db.sync_pending_spec(spec) == []


def test_sync_pending_spec_never_inserts(db):
    db.upsert_pending("seeded", 100, 5)
    assert db.sync_pending_spec({"seeded": (100, 5), "unseeded": (50, 0)}) == []
    assert db.get("unseeded") is None


# ---- 3. end-to-end: reload between claims ---------------------------------
class _BoundedStop(threading.Event):
    """Stop event that trips itself after ``max_waits`` idle waits (and never
    actually sleeps), so a worker loop with nothing claimable terminates."""

    def __init__(self, max_waits: int = 1):
        super().__init__()
        self.max_waits = max_waits
        self.waits = 0

    def wait(self, timeout=None):  # type: ignore[override]
        self.waits += 1
        if self.waits >= self.max_waits:
            self.set()
        return super().wait(0)


class _Recorder:
    """Stands in for lifecycle.run: records the row it was handed, finishes the
    model, and stops the worker so each _run_worker does at most one launch."""

    def __init__(self, db: AirccDB):
        self.db = db
        self.mgr: jm.JobManager | None = None
        self.seen: list[dict] = []

    def __call__(self, row, models_root, db, **kw):
        self.seen.append(dict(row))
        self.db.mark_finished(row["model_name"])
        assert self.mgr is not None
        self.mgr.stop.set()
        return True


def _manager(db: AirccDB, csv_dir: Path, tmp_path: Path, rec: _Recorder) -> jm.JobManager:
    mgr = jm.JobManager(db, csv_dir, tmp_path / "models", None)
    rec.mgr = mgr
    return mgr


def _run_worker(mgr: jm.JobManager, max_waits: int = 1) -> None:
    """Run one bounded pass of _worker: it exits after a launch or ``max_waits``."""
    mgr.stop = _BoundedStop(max_waits)
    t = threading.Thread(target=mgr._worker, args=(0,), daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "worker did not stop"


def test_csv_edit_reaches_the_next_claim(tmp_path, monkeypatch, db):
    csv_dir = tmp_path / "csv"
    _write_csv(csv_dir, [_row("A"), _row("B", priority=1)])
    db.upsert_pending("A", 100, 0)
    db.upsert_pending("B", 100, 1)

    rec = _Recorder(db)
    monkeypatch.setattr(jm.lifecycle, "run", rec)
    mgr = _manager(db, csv_dir, tmp_path, rec)

    # pass 1: claims A (priority 0) with the CSV as it stands
    _run_worker(mgr)
    assert [r["model_name"] for r in rec.seen] == ["A"]
    assert rec.seen[0]["training.epochs"] == "100"

    # edit the CSV while the SAME manager object is alive -- no restart
    _write_csv(csv_dir, [_row("A"),
                         _row("B", priority=1, epochs=250, **{"training.batch_size": "128"})])
    _run_worker(mgr)

    assert [r["model_name"] for r in rec.seen] == ["A", "B"]
    assert rec.seen[1]["training.epochs"] == "250"   # fresh row, not the startup snapshot
    assert rec.seen[1]["training.batch_size"] == "128"
    assert db.get("B").total_epochs == 250           # synced onto the DB before the claim


def test_csv_priority_edit_reorders_the_next_claim(tmp_path, monkeypatch, db):
    csv_dir = tmp_path / "csv"
    _write_csv(csv_dir, [_row("A"), _row("B", priority=1), _row("C", priority=2)])
    for name, prio in (("A", 0), ("B", 1), ("C", 2)):
        db.upsert_pending(name, 100, prio)

    rec = _Recorder(db)
    monkeypatch.setattr(jm.lifecycle, "run", rec)
    mgr = _manager(db, csv_dir, tmp_path, rec)

    _run_worker(mgr)
    assert [r["model_name"] for r in rec.seen] == ["A"]

    # demote B in the CSV only -- C must be claimed next
    _write_csv(csv_dir, [_row("A"), _row("B", priority=9), _row("C", priority=2)])
    _run_worker(mgr)
    assert [r["model_name"] for r in rec.seen] == ["A", "C"]
    assert db.get("B").priority == 9


def test_dependency_edit_reaches_the_next_claim(tmp_path, monkeypatch, db):
    """A dependency added in the CSV gates the very next claim."""
    csv_dir = tmp_path / "csv"
    _write_csv(csv_dir, [_row("A"), _row("B", priority=1)])
    db.upsert_pending("A", 100, 0)
    db.upsert_pending("B", 100, 1)

    rec = _Recorder(db)
    monkeypatch.setattr(jm.lifecycle, "run", rec)
    mgr = _manager(db, csv_dir, tmp_path, rec)

    _run_worker(mgr)
    assert [r["model_name"] for r in rec.seen] == ["A"]

    # A is finished but has no best_checkpoint, so B -- now dependent on A -- is blocked
    _write_csv(csv_dir, [_row("A"), _row("B", priority=1, dep="A")])
    _run_worker(mgr, max_waits=2)
    assert [r["model_name"] for r in rec.seen] == ["A"]
    assert db.get("B").status == "pending"


def test_dry_run_skips_parked_rows(tmp_path, capsys, db):
    """--dry-run must mirror claim_next: parked rows (owner_task=-1) are not eligible."""
    csv_dir = tmp_path / "csv"
    _write_csv(csv_dir, [_row("claimable"), _row("held_back", priority=1)])
    db.upsert_pending("claimable", 100, 0)
    db.upsert_pending("held_back", 100, 1)
    db._write("UPDATE jobs SET owner_task=-1 WHERE model_name='held_back'", ())

    jm.JobManager(db, csv_dir, tmp_path / "models", None).dry_run(8)
    out = capsys.readouterr().out
    assert "claimable" in out
    assert "held_back" not in out


# ---- 4. loud failure ------------------------------------------------------
def test_torn_csv_blocks_claiming_and_fails_nothing(tmp_path, monkeypatch, db):
    csv_dir = tmp_path / "csv"
    path = _write_csv(csv_dir, [_row("A"), _row("B", priority=1)])
    db.upsert_pending("A", 100, 0)
    db.upsert_pending("B", 100, 1)

    rec = _Recorder(db)
    monkeypatch.setattr(jm.lifecycle, "run", rec)
    mgr = _manager(db, csv_dir, tmp_path, rec)

    # simulate a half-written file: header + one good row + a truncated row
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2] + ["," * (len(ALL_COLUMNS) - 1)]) + "\n")

    _run_worker(mgr, max_waits=2)
    assert rec.seen == []                                   # nothing claimed
    assert {j.status for j in db.all_jobs()} == {"pending"}  # nothing failed, nothing running

    # a clean rewrite unblocks the very next pass
    _write_csv(csv_dir, [_row("A"), _row("B", priority=1)])
    _run_worker(mgr)
    assert [r["model_name"] for r in rec.seen] == ["A"]
