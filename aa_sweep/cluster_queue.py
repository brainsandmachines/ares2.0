"""The Slurm lane's work queue: a SQLite DB of (model, kind) units, drained by two job arrays.

    python3 -m aa_sweep.cluster_queue feed [--launch] [--dry-run]  < units.json   # nightly driver
    python3 -m aa_sweep.cluster_queue claim --log <path>                          # array task
    python3 -m aa_sweep.cluster_queue finish <id> <rc>                            # array task
    python3 -m aa_sweep.cluster_queue release <id>                                # SIGTERM trap
    python3 -m aa_sweep.cluster_queue status [--all]
    python3 -m aa_sweep.cluster_queue reset <id> | drop <id>

Runs **on the BGU cluster** (login node or array task) against ``config.CLUSTER_QUEUE_DB``, with the
login node's stock ``python3`` (3.9) -- stdlib only, so a task can claim before loading conda and a
task with nothing to do exits in seconds.

Division of labour:

* The Botero cron (``aa_sweep.submit``) censuses every Slurm-lane model and pipes the units here
  over one ssh (``feed --launch``). ``feed`` is idempotent; ``--launch`` then submits the array of
  each lane that has claimable work and no task of its own still waiting to start.
* Each array task (``scripts/aa_queue_task.sh``) claims ONE unit, runs the engine on it, and hands
  the exit code to ``finish``, which re-censuses the model dir on disk -- the CSV, not the exit
  code, decides whether the unit is done. A task that finds nothing to claim cancels its array's
  remaining pending tasks, so an empty queue means nothing running.

Ownership invariant, as in ``slurm_job_manager/db.py``: ``slurm_job_id IS NULL`` <=> claimable. A
running row whose owner Slurm no longer runs is released by ``requeue_dead`` (at every claim and
every feed). All writes are one ``BEGIN IMMEDIATE`` transaction with a bounded retry, so two tasks
can never claim the same unit.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from aa_sweep import config
from aa_sweep.census import grid_cells, kind_status

PENDING = "pending"
RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"

# A claim younger than this is exempt from the dead-owner check: a job id that young may not be
# visible to squeue/sacct yet, and calling it dead would hand its unit to a second writer.
REQUEUE_MIN_AGE_S = 300

# Owner states that may still be writing the CSV. PENDING is deliberately *not* here: a NODE_FAIL
# requeue keeps the job id but the restarted task claims afresh, so its old row is orphaned.
_OWNER_LIVE_STATES = frozenset({"RUNNING", "SUSPENDED", "COMPLETING", "CONFIGURING", "RESIZING"})

_BUSY_TIMEOUT_MS = 30_000
_MAX_RETRIES = 8
_RETRY_SLEEP_S = 0.25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    model_dir    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    missing      INTEGER NOT NULL DEFAULT 0,
    priority     INTEGER NOT NULL DEFAULT 100,
    slurm_job_id INTEGER,
    partition    TEXT,
    claimed_ts   INTEGER,
    attempts     INTEGER NOT NULL DEFAULT 0,
    requeued     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    log_path     TEXT,
    created_ts   INTEGER,
    updated_ts   INTEGER,
    UNIQUE (model_name, kind)
);
"""


def _is_transient(err: sqlite3.OperationalError) -> bool:
    msg = str(err).lower()
    return "locked" in msg or "busy" in msg or "unable to open database file" in msg


def missing_cells(model_dir: str, kind: str) -> int:
    """Grid cells this kind still lacks, read from the model dir's own CSV -- the same census the
    Botero-side planner runs over its ssh probe, so the two can never disagree about 'done'."""
    d = Path(model_dir)
    files = {p.name for p in d.iterdir() if p.is_file()} if d.is_dir() else set()
    csv_path = d / config.CSV_FOR_KIND[kind]
    try:
        text = csv_path.read_text() if csv_path.is_file() else ""
    except OSError:
        text = ""
    status = kind_status(
        kind=kind,
        ckpt_filename=config.CKPT_FILE_FOR_KIND[kind],
        model_name=d.name,
        grid=grid_cells(config.NORMS, config.EPS_INPUTS),
        files=files,
        csv_text=text,
    )
    return len(status.missing)


def owner_alive(job_id: int, run=subprocess.run) -> bool:
    """True if Slurm may still be running this task -- or if we could not find out.

    Fails open like ``slurm_job_manager.controller._default_slurm_active``: a squeue/sacct hiccup
    must never read as 'dead', or a healthy task's CSV gets a second writer. Only an answered probe
    that shows no live state releases a row.
    """
    answered = False
    for argv in (["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                 ["sacct", "-j", str(job_id), "-X", "-n", "-o", "State"]):
        try:
            proc = run(argv, capture_output=True, text=True, timeout=60)
        except Exception:
            continue
        states = {line.split()[0].upper() for line in proc.stdout.splitlines() if line.strip()}
        if states & _OWNER_LIVE_STATES:
            return True
        if proc.returncode == 0 or "invalid job id" in (proc.stderr or "").lower():
            answered = True
    return not answered


class QueueDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._atomic(lambda conn: conn.executescript(_SCHEMA), begin=False)

    # ---- plumbing -----------------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000.0)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        try:
            yield conn
        finally:
            conn.close()

    def _atomic(self, op, begin: bool = True, commit: bool = True):
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                with self._connect() as conn:
                    if begin:
                        conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = op(conn)
                        if commit:
                            conn.commit()
                        else:
                            conn.rollback()
                        return result
                    except Exception:
                        conn.rollback()
                        raise
            except sqlite3.OperationalError as e:
                last_err = e
                if _is_transient(e):
                    time.sleep(_RETRY_SLEEP_S * (attempt + 1))
                    continue
                raise
        raise sqlite3.OperationalError(f"DB write failed after {_MAX_RETRIES} retries: {last_err}")

    def rows(self, where: str = "1", params: tuple = ()) -> list:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                f"SELECT * FROM units WHERE {where} ORDER BY priority, missing DESC, id", params)]

    def get(self, unit_id: int) -> Optional[dict]:
        found = self.rows("id=?", (int(unit_id),))
        return found[0] if found else None

    def counts(self) -> dict:
        with self._connect() as conn:
            return {r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM units GROUP BY status")}

    def claimable(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM units WHERE status='pending' AND slurm_job_id IS NULL"
            ).fetchone()[0]

    # ---- the nightly feed ---------------------------------------------------------------------
    def feed(self, units: list, dry_run: bool = False) -> dict:
        """Upsert the census. Each unit: ``{model_name, kind, model_dir, missing}``.

        * new with missing cells       -> pending
        * pending/failed, now complete -> finished (something else completed it)
        * finished, cells missing again -> pending (the grid grew; not a failure)
        * pending                      -> refresh ``missing``/``model_dir``
        * running                      -> untouched: its task owns the CSV and will ``finish`` it
        * failed with cells missing    -> untouched; ``reset`` re-arms it by hand
        """
        now = int(time.time())
        tally = {"inserted": 0, "reopened": 0, "closed": 0, "refreshed": 0, "untouched": 0}

        def op(conn: sqlite3.Connection) -> dict:
            for u in units:
                name, kind, mdir, miss = u["model_name"], u["kind"], u["model_dir"], int(u["missing"])
                row = conn.execute("SELECT * FROM units WHERE model_name=? AND kind=?",
                                   (name, kind)).fetchone()
                if row is None:
                    if miss > 0:
                        conn.execute(
                            "INSERT INTO units (model_name, kind, model_dir, missing, created_ts, "
                            "updated_ts) VALUES (?, ?, ?, ?, ?, ?)", (name, kind, mdir, miss, now, now))
                        tally["inserted"] += 1
                    continue
                status = row["status"]
                if miss == 0 and status in (PENDING, FAILED) and row["slurm_job_id"] is None:
                    conn.execute("UPDATE units SET status='finished', missing=0, updated_ts=? "
                                 "WHERE id=?", (now, row["id"]))
                    tally["closed"] += 1
                elif miss > 0 and status == FINISHED:
                    conn.execute("UPDATE units SET status='pending', missing=?, model_dir=?, "
                                 "attempts=0, updated_ts=? WHERE id=?", (miss, mdir, now, row["id"]))
                    tally["reopened"] += 1
                elif miss > 0 and status == PENDING:
                    conn.execute("UPDATE units SET missing=?, model_dir=? WHERE id=?",
                                 (miss, mdir, row["id"]))
                    tally["refreshed"] += 1
                else:
                    tally["untouched"] += 1
            return tally

        return self._atomic(op, commit=not dry_run)

    # ---- the array task -----------------------------------------------------------------------
    def claim(self, owner: int, partition: str = "", log_path: str = "") -> Optional[dict]:
        """Atomically take the top pending unit for this array task, or None when there is none.

        Any row this very job id still holds is released first: a task restarted under the same id
        (NODE_FAIL requeue) is a new process, and its previous claim is no longer being worked.
        """
        now = int(time.time())

        def op(conn: sqlite3.Connection) -> Optional[dict]:
            conn.execute("UPDATE units SET status='pending', slurm_job_id=NULL, claimed_ts=NULL, "
                         "requeued=requeued+1, updated_ts=? WHERE slurm_job_id=? AND status='running'",
                         (now, int(owner)))
            cand = conn.execute(
                "SELECT id FROM units WHERE status='pending' AND slurm_job_id IS NULL "
                "ORDER BY priority, missing DESC, id LIMIT 1").fetchone()
            if cand is None:
                return None
            conn.execute(
                "UPDATE units SET status='running', slurm_job_id=?, partition=?, claimed_ts=?, "
                "log_path=?, updated_ts=? WHERE id=? AND slurm_job_id IS NULL",
                (int(owner), partition, now, log_path, now, cand["id"]))
            return dict(conn.execute("SELECT * FROM units WHERE id=?", (cand["id"],)).fetchone())

        return self._atomic(op)

    def finish(self, unit_id: int, owner: Optional[int], rc: int,
               missing: Optional[int] = None) -> Optional[str]:
        """Close out a claim from the CSV on disk. Returns the new status, or None if this task no
        longer owns the unit (it was released and re-claimed meanwhile -- then it is not ours)."""
        unit = self.get(unit_id)
        if unit is None:
            return None
        if missing is None:
            missing = missing_cells(unit["model_dir"], unit["kind"])
        now = int(time.time())

        def op(conn: sqlite3.Connection) -> Optional[str]:
            sql_guard = " AND status='running'"
            params: tuple = (int(unit_id),)
            if owner is not None:
                sql_guard += " AND slurm_job_id=?"
                params += (int(owner),)
            if missing == 0:
                cur = conn.execute(
                    "UPDATE units SET status='finished', missing=0, slurm_job_id=NULL, "
                    f"last_error=NULL, updated_ts={now} WHERE id=?" + sql_guard, params)
                return FINISHED if cur.rowcount else None
            attempts = unit["attempts"] + 1
            status = FAILED if attempts >= config.QUEUE_MAX_ATTEMPTS else PENDING
            error = f"rc={rc}, {missing} cell(s) still missing (log {unit['log_path'] or '?'})"
            cur = conn.execute(
                "UPDATE units SET status=?, missing=?, attempts=?, last_error=?, slurm_job_id=NULL, "
                f"claimed_ts=NULL, updated_ts={now} WHERE id=?" + sql_guard,
                (status, missing, attempts, error) + params)
            return status if cur.rowcount else None

        return self._atomic(op)

    def release(self, unit_id: int, owner: Optional[int] = None) -> int:
        """Back to the pool without counting an attempt (time limit / scancel). The engine flushes
        the CSV after every cell, so whoever claims it next resumes where this task stopped."""
        sql = ("UPDATE units SET status='pending', slurm_job_id=NULL, claimed_ts=NULL, "
               "requeued=requeued+1, updated_ts=? WHERE id=? AND status='running'")
        params: tuple = (int(time.time()), int(unit_id))
        if owner is not None:
            sql += " AND slurm_job_id=?"
            params += (int(owner),)
        return self._atomic(lambda conn: conn.execute(sql, params).rowcount)

    def requeue_dead(self, alive: Callable[[int], bool] = owner_alive,
                     min_age_s: int = REQUEUE_MIN_AGE_S, now: Optional[int] = None) -> list:
        """Release running rows whose owning task Slurm no longer runs. Returns their ids."""
        now = int(time.time()) if now is None else int(now)
        released = []
        for row in self.rows("status='running' AND slurm_job_id IS NOT NULL"):
            if row["claimed_ts"] is not None and now - row["claimed_ts"] < min_age_s:
                continue
            if not alive(int(row["slurm_job_id"])) and self.release(row["id"], owner=row["slurm_job_id"]):
                released.append(row["id"])
        return released

    # ---- manual ops ---------------------------------------------------------------------------
    def reset(self, unit_id: int) -> int:
        return self._atomic(lambda conn: conn.execute(
            "UPDATE units SET status='pending', attempts=0, slurm_job_id=NULL, claimed_ts=NULL, "
            "updated_ts=? WHERE id=? AND status!='running'", (int(time.time()), int(unit_id))).rowcount)

    def drop(self, unit_id: int) -> int:
        return self._atomic(lambda conn: conn.execute(
            "DELETE FROM units WHERE id=?", (int(unit_id),)).rowcount)


# ---- launching the arrays ---------------------------------------------------------------------
def launch(claimable: int, dry_run: bool = False, run=subprocess.run) -> tuple:
    """Submit each lane's array if it has work to pick up and nothing of its own waiting to start.

    ``--dependency=singleton`` makes a fresh array wait for the lane's previous one to end, so a lane
    whose old array still has running tasks never exceeds its ``%8``. An array that is merely
    throttled (``JobArrayTaskLimit``) or still waiting on singleton shows up as PENDING and blocks a
    duplicate. Returns ``(messages, errors)``.
    """
    messages, errors = [], []
    for lane, (script, name) in config.QUEUE_ARRAYS.items():
        if claimable <= 0:
            messages.append(f"launch {lane}: skip, nothing pending")
            continue
        try:
            sq = run(["squeue", "-u", config.SLURM_USER, "-h", "-t", "PENDING", "-n", name, "-o", "%i"],
                     capture_output=True, text=True, timeout=60)
        except Exception as exc:
            errors.append(f"launch {lane}: squeue failed: {exc}")
            continue
        if sq.returncode != 0:
            errors.append(f"launch {lane}: squeue rc={sq.returncode}: {sq.stderr.strip()}")
            continue
        if sq.stdout.strip():
            messages.append(f"launch {lane}: skip, {name} already has pending tasks "
                            f"({sq.stdout.split()[0]})")
            continue
        if dry_run:
            messages.append(f"launch {lane}: DRY-RUN would sbatch {script}")
            continue
        try:
            sb = run(["sbatch", "--parsable", "--dependency=singleton", script],
                     capture_output=True, text=True, timeout=120, cwd=config.SLURM_REPO)
        except Exception as exc:
            errors.append(f"launch {lane}: sbatch failed: {exc}")
            continue
        if sb.returncode != 0:
            errors.append(f"launch {lane}: sbatch rc={sb.returncode}: {sb.stderr.strip()}")
            continue
        messages.append(f"launch {lane}: submitted {name} job={sb.stdout.strip().split(';')[0]}")
    return messages, errors


# ---- CLI --------------------------------------------------------------------------------------
def _say(msg: str, err: bool = False) -> None:
    print(f"[aa_queue] {msg}", file=sys.stderr if err else sys.stdout, flush=True)


def _env_owner() -> Optional[int]:
    job = os.environ.get("SLURM_JOB_ID")
    return int(job) if job else None


def _age(ts: Optional[int]) -> str:
    if not ts:
        return "-"
    s = int(time.time()) - int(ts)
    return f"{s // 86400}d{s % 86400 // 3600:02d}h" if s >= 86400 else f"{s // 3600}h{s % 3600 // 60:02d}m"


def _print_status(db: QueueDB, show_all: bool) -> None:
    counts = db.counts()
    _say("counts " + " ".join(f"{k}={counts.get(k, 0)}" for k in (PENDING, RUNNING, FINISHED, FAILED)))
    wanted = "1" if show_all else "status IN ('running','failed')"
    for r in db.rows(wanted):
        print(f"  {r['id']:>5}  {r['status']:<8} {r['model_name']}:{r['kind']}  missing={r['missing']} "
              f"job={r['slurm_job_id'] or '-'} part={r['partition'] or '-'} age={_age(r['claimed_ts'])} "
              f"attempts={r['attempts']}" + (f"  err={r['last_error']}" if r["last_error"] else ""))
    if not show_all:
        pending = db.rows("status='pending'")
        for r in pending[:10]:
            print(f"  {r['id']:>5}  pending  {r['model_name']}:{r['kind']}  missing={r['missing']}")
        if len(pending) > 10:
            print(f"  ... {len(pending) - 10} more pending (--all to list)")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="aa_sweep.cluster_queue", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=config.CLUSTER_QUEUE_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("feed", help="upsert units (JSON list on stdin)")
    p.add_argument("--launch", action="store_true", help="then submit any lane array that is due")
    p.add_argument("--dry-run", action="store_true", help="write nothing, submit nothing")
    p = sub.add_parser("claim", help="claim one unit; prints 'id<TAB>kind<TAB>model_dir' or nothing")
    p.add_argument("--log", default="")
    for name in ("finish", "release", "reset", "drop"):
        p = sub.add_parser(name)
        p.add_argument("id", type=int)
        if name == "finish":
            p.add_argument("rc", type=int)
    p = sub.add_parser("status")
    p.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    db = QueueDB(args.db)

    if args.cmd == "feed":
        raw = sys.stdin.read().strip()
        units = json.loads(raw) if raw else []
        released = [] if args.dry_run else db.requeue_dead()
        tally = db.feed(units, dry_run=args.dry_run)
        tag = "DRY-RUN " if args.dry_run else ""
        claimable = db.claimable() + (tally["inserted"] + tally["reopened"] if args.dry_run else 0)
        _say(f"{tag}feed units={len(units)} " + " ".join(f"{k}={v}" for k, v in tally.items())
             + f" released_dead={len(released)} claimable={claimable}")
        errors = []
        if args.launch or args.dry_run:
            messages, errors = launch(claimable, dry_run=args.dry_run)
            for m in messages + errors:
                _say(m)
        counts = db.counts()
        _say(f"{tag}summary " + " ".join(f"{k}={counts.get(k, 0)}" for k in (PENDING, RUNNING, FINISHED, FAILED))
             + f" launch_errors={len(errors)}")
        return 1 if errors else 0

    if args.cmd == "claim":
        owner = _env_owner()
        if owner is None:
            _say("claim needs $SLURM_JOB_ID (run it from an array task)", err=True)
            return 2
        # stdout is the machine-read unit line (aa_queue_task.sh parses it); notes go to stderr.
        released = db.requeue_dead()
        if released:
            _say(f"released dead-owner unit(s) {released}", err=True)
        unit = db.claim(owner, os.environ.get("SLURM_JOB_PARTITION", ""), args.log)
        if unit is not None:
            print(f"{unit['id']}\t{unit['kind']}\t{unit['model_dir']}")
        return 0

    if args.cmd == "finish":
        status = db.finish(args.id, _env_owner(), args.rc)
        unit = db.get(args.id) or {}
        _say(f"finish {args.id} {unit.get('model_name')}:{unit.get('kind')} rc={args.rc} -> "
             f"{status or 'no-op (not the owner any more)'} missing={unit.get('missing')}")
        return 0

    if args.cmd == "release":
        n = db.release(args.id, owner=_env_owner())
        _say(f"release {args.id}: {'released' if n else 'no-op (not owner / not running)'}")
        return 0

    if args.cmd == "reset":
        _say(f"reset {args.id}: {'pending' if db.reset(args.id) else 'no-op (missing or running)'}")
        return 0

    if args.cmd == "drop":
        _say(f"drop {args.id}: {'deleted' if db.drop(args.id) else 'no such unit'}")
        return 0

    _print_status(db, args.all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
