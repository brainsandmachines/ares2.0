"""The Botero lane: a local, strictly serial GPU slot for the AutoAttack sweep.

Botero has one RTX 4090 -- the same 24GB class ``sbatches/aa_sweep_completion.sbatch`` demands via
``--constraint`` -- and local copies of every model from both clusters under
``/mnt/data4t/{slurm,aircc}_archive/models`` (mirrored onto the QNAP at ``/mnt/botero``). The BGU
lane is chronically jammed behind ``QOSMaxGRESPerUser``, so this module makes that GPU a real slot:

    queue (this module)  ->  botero_runner.py  ->  data_analysis/autoattack_array_eval.py

Design points that matter:

* **Depth 5, serial.** The queue holds up to ``config.BOTERO_SLOTS`` units of work; the runner runs
  exactly one at a time. Depth is a *backlog*, not a concurrency level.
* **Results stay local.** A local run writes its CSV rows into the archive model dir it read the
  checkpoint from, and nothing is ever pushed back to a cluster. That is why
  ``census.kind_status`` grew a third source -- see ``plan.build_plan``.
* **Moving beats adding.** Top-up prefers to *move* work that is already queued on Slurm (cancel it
  there, run it here), taking from the back of the queue: those are the jobs Slurm would pick up
  last. Only when there is nothing left to move does it enqueue brand-new work.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from aa_sweep import config

# Statuses this lane owns a unit under. `parked` means "ours, but do not run it yet" -- it still
# counts against the slot budget and still suppresses the cluster submission, so parking a unit
# never hands it back to Slurm behind your back. Only `queued` is claimable.
ACTIVE = ("queued", "running", "parked")

SCHEMA = """
CREATE TABLE IF NOT EXISTS botero_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    checkpoint_kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    enqueued_ts INTEGER NOT NULL,
    started_ts INTEGER,
    finished_ts INTEGER,
    pid INTEGER,
    origin TEXT,
    model_dir TEXT,
    cells_at_enqueue INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
-- A (model, kind) can be active at most once, but a finished row must not block a later
-- re-enqueue if new grid cells appear. A partial index is exactly that rule.
CREATE UNIQUE INDEX IF NOT EXISTS ux_botero_active
    ON botero_jobs(model_name, checkpoint_kind) WHERE status IN ('queued', 'running', 'parked');
CREATE INDEX IF NOT EXISTS ix_botero_status ON botero_jobs(status, id);
"""


@dataclass
class Job:
    """One queued unit of work: sweep the missing grid cells of one (model, checkpoint kind)."""

    id: int
    model_name: str
    checkpoint_kind: str
    status: str
    model_dir: str
    origin: str = ""
    attempts: int = 0
    pid: int | None = None
    cells_at_enqueue: int | None = None
    last_error: str | None = None

    @property
    def slug(self) -> str:
        return f"{self.model_name.strip('/').replace('/', '__')}_{self.checkpoint_kind}"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the local queue. Local disk, so WAL is safe here."""
    db_path = Path(db_path or config.BOTERO_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        model_name=row["model_name"],
        checkpoint_kind=row["checkpoint_kind"],
        status=row["status"],
        model_dir=row["model_dir"] or "",
        origin=row["origin"] or "",
        attempts=row["attempts"],
        pid=row["pid"],
        cells_at_enqueue=row["cells_at_enqueue"],
        last_error=row["last_error"],
    )


# --- the local model archive ------------------------------------------------------------------


def resolve_model_dir(
    model_name: str, ckpt_filename: str, roots: tuple[Path, ...] | None = None
) -> Path | None:
    """First archive root holding this model's checkpoint *and* its image selection.

    The selection json is not optional: it is what pins the same 1024 images the cluster rows were
    computed on, and without it a local run would produce numbers that are not comparable.
    """
    for root in roots or config.BOTERO_MODEL_ROOTS:
        candidate = Path(root) / model_name
        try:
            if (candidate / ckpt_filename).is_file() and (candidate / config.SELECTION_JSON).is_file():
                return candidate
        except OSError:
            # A dropped CIFS/sshfs root must not take the whole run down; just try the next one.
            continue
    return None


def qnap_counterpart(model_dir: Path, model_name: str) -> Path | None:
    """Where on the QNAP this model's artifacts should be echoed, or None if it is not there.

    Usually the straight ``/mnt/data4t`` -> ``/mnt/botero`` mirror of the path. But the two archives
    are not consistently split by originating cluster -- ``convnext_base_v1_l2_2_init1`` lives under
    ``slurm_archive`` locally and ``aircc_archive`` on the QNAP -- so fall back to whichever QNAP
    root already holds the model rather than creating a second copy of it under the other one.
    """
    try:
        mirrored = config.BOTERO_QNAP_ROOT / Path(model_dir).relative_to(config.BOTERO_LOCAL_ROOT)
    except ValueError:
        mirrored = None
    if mirrored is not None and mirrored.is_dir():
        return mirrored
    for root in config.BOTERO_MODEL_ROOTS:
        candidate = Path(root) / model_name
        if config.BOTERO_QNAP_ROOT in Path(root).parents and candidate.is_dir():
            return candidate
    return None


# --- queue operations -------------------------------------------------------------------------


def active_units(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    rows = conn.execute(
        f"SELECT model_name, checkpoint_kind FROM botero_jobs WHERE status IN {ACTIVE}"
    ).fetchall()
    return {(r["model_name"], r["checkpoint_kind"]) for r in rows}


def active_job_names(conn: sqlite3.Connection | None = None) -> set[str]:
    """Slurm-style job names for everything this lane owns, for ``submit.live_job_names``."""
    own = conn is None
    conn = conn or connect()
    try:
        return {config.job_name(m, k) for m, k in active_units(conn)}
    finally:
        if own:
            conn.close()


def active_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM botero_jobs WHERE status IN {ACTIVE}"
    ).fetchone()[0]


def enqueue(
    conn: sqlite3.Connection,
    model_name: str,
    checkpoint_kind: str,
    model_dir: Path,
    origin: str = "new",
    cells: int | None = None,
) -> int | None:
    """Add one unit of work. Returns None if it is already active (the partial index refuses it)."""
    try:
        cur = conn.execute(
            "INSERT INTO botero_jobs (model_name, checkpoint_kind, status, enqueued_ts, origin,"
            " model_dir, cells_at_enqueue) VALUES (?, ?, 'queued', ?, ?, ?, ?)",
            (model_name, checkpoint_kind, int(time.time()), origin, str(model_dir), cells),
        )
    except sqlite3.IntegrityError:
        return None
    return cur.lastrowid


def claim(conn: sqlite3.Connection, pid: int) -> Job | None:
    """Take the oldest queued row, atomically. Mirrors slurm_job_manager/db.py's claim."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM botero_jobs WHERE status='queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        now = int(time.time())
        conn.execute(
            "UPDATE botero_jobs SET status='running', pid=?, started_ts=?, attempts=attempts+1,"
            " finished_ts=NULL WHERE id=? AND status='queued'",
            (pid, now, row["id"]),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    job = _row_to_job(conn.execute("SELECT * FROM botero_jobs WHERE id=?", (row["id"],)).fetchone())
    return job


def finish(conn: sqlite3.Connection, job_id: int, ok: bool, error: str | None = None) -> None:
    conn.execute(
        "UPDATE botero_jobs SET status=?, finished_ts=?, pid=NULL, last_error=? WHERE id=?",
        ("finished" if ok else "failed", int(time.time()), None if ok else (error or "")[-4000:], job_id),
    )


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_stale(conn: sqlite3.Connection, alive=pid_alive) -> list[str]:
    """Return rows whose runner died back to the queue (or fail them past the attempt cap).

    Safe to requeue: the eval engine flushes the sweep CSV after every setting, so an interrupted
    run has already banked every completed cell and simply resumes at the next one.
    """
    notes: list[str] = []
    for row in conn.execute("SELECT * FROM botero_jobs WHERE status='running'").fetchall():
        if alive(row["pid"]):
            continue
        if row["attempts"] >= config.BOTERO_MAX_ATTEMPTS:
            conn.execute(
                "UPDATE botero_jobs SET status='failed', finished_ts=?, pid=NULL, last_error=?"
                " WHERE id=?",
                (int(time.time()), f"runner died {row['attempts']}x (attempt cap)", row["id"]),
            )
            notes.append(f"job {row['id']} {row['model_name']}:{row['checkpoint_kind']} FAILED (attempt cap)")
        else:
            conn.execute(
                "UPDATE botero_jobs SET status='queued', pid=NULL, started_ts=NULL WHERE id=?",
                (row["id"],),
            )
            notes.append(f"job {row['id']} {row['model_name']}:{row['checkpoint_kind']} requeued (runner gone)")
    return notes


# --- topping the queue back up to full ---------------------------------------------------------


def pending_slurm_jobs(run=subprocess.run) -> list[tuple[int, str, str]]:
    """``(jobid, model_name, kind)`` for every PENDING aaswp job, back of the queue first.

    Descending job id is the point: those are the ones Slurm will start last, so they are the ones
    worth taking. ``squeue -o '%j'`` is unwidthed for the same reason as in ``submit.py`` -- a
    width would truncate long names and every decode would miss.
    """
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST,
         f"squeue -u {config.SLURM_USER} -h -t PD -o '%i|%j'"],
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 2,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"squeue -t PD failed rc={proc.returncode}: {proc.stderr.strip()}")
    out: list[tuple[int, str, str]] = []
    for line in proc.stdout.splitlines():
        if "|" not in line:
            continue
        raw_id, raw_name = line.split("|", 1)
        parsed = config.parse_job_name(raw_name.strip())
        if parsed is None:
            continue
        try:
            job_id = int(raw_id.strip().split("_")[0])
        except ValueError:
            continue
        out.append((job_id, parsed[0], parsed[1]))
    out.sort(key=lambda item: -item[0])
    return out


def scancel(job_id: int, run=subprocess.run) -> None:
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST,
         f"scancel {job_id}"],
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 2,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scancel {job_id} failed rc={proc.returncode}: {proc.stderr.strip()}")


def topup(
    works,
    conn: sqlite3.Connection | None = None,
    dry_run: bool = False,
    slots: int | None = None,
    pending=pending_slurm_jobs,
    cancel=scancel,
    log=print,
) -> list[str]:
    """Refill the Botero queue to ``slots`` units. Returns one description per unit taken.

    ``works`` is the ``plan.ModelWork`` list the nightly run already built, used both to know how
    many cells a candidate still needs and -- for source B -- to find work that was never queued
    anywhere.
    """
    own = conn is None
    conn = conn or connect()
    slots = config.BOTERO_SLOTS if slots is None else slots
    taken: list[str] = []
    try:
        have = active_count(conn)
        need = slots - have
        if need <= 0:
            log(f"botero: queue full ({have}/{slots}), nothing to move")
            return taken

        by_name = {w.model_name: w for w in works}
        active = active_units(conn)

        def eligible(model_name: str, kind: str) -> tuple[Path, int] | None:
            """Local dir + remaining cell count, or None with the reason logged."""
            if (model_name, kind) in active:
                return None
            work = by_name.get(model_name)
            status = work.kinds.get(kind) if work else None
            if status is None:
                log(f"botero: skip {model_name}:{kind}, not a finished model in this plan")
                return None
            if not status.missing:
                log(f"botero: skip {model_name}:{kind}, no missing cells")
                return None
            model_dir = resolve_model_dir(model_name, config.CKPT_FILE_FOR_KIND[kind])
            if model_dir is None:
                log(f"botero: skip {model_name}:{kind}, no local archive dir with"
                    f" {config.CKPT_FILE_FOR_KIND[kind]} + {config.SELECTION_JSON}")
                return None
            return model_dir, len(status.missing)

        # --- source A: move work off the back of the Slurm queue ---------------------------
        try:
            candidates = pending()
        except Exception as exc:
            log(f"botero: cannot read the pending Slurm queue ({exc}); falling back to new work")
            candidates = []

        for job_id, model_name, kind in candidates:
            if len(taken) >= need:
                break
            hit = eligible(model_name, kind)
            if hit is None:
                continue
            model_dir, cells = hit
            if dry_run:
                log(f"botero: DRY-RUN would scancel {job_id} and enqueue {model_name}:{kind}"
                    f" ({cells} cells) from {model_dir}")
                taken.append(f"{model_name}:{kind}")
                active.add((model_name, kind))
                continue
            try:
                cancel(job_id)
            except Exception as exc:
                log(f"botero: scancel {job_id} failed ({exc}), leaving {model_name}:{kind} on Slurm")
                continue
            # Enqueue only after the cancel lands, so the unit is never queued in both lanes.
            if enqueue(conn, model_name, kind, model_dir, origin=f"moved:{job_id}", cells=cells) is None:
                log(f"botero: {model_name}:{kind} became active concurrently, skipping")
                continue
            log(f"botero: moved {model_name}:{kind} ({cells} cells) from slurm job {job_id}")
            taken.append(f"{model_name}:{kind}")
            active.add((model_name, kind))

        # --- source B: nothing left to move, so take work that was never queued at all ------
        #
        # Anything still pending on Slurm is off limits here even though source A did not take it:
        # a scancel that failed, or a unit past the `need` cut, is still queued over there, and
        # enqueueing it locally too would run it in both lanes at once.
        on_slurm = {(model_name, kind) for _, model_name, kind in candidates}
        if len(taken) < need:
            for work in works:
                if len(taken) >= need:
                    break
                for kind in config.CHECKPOINT_KINDS:
                    if len(taken) >= need:
                        break
                    status = work.kinds.get(kind)
                    if status is None or not status.missing:
                        continue
                    if (work.model_name, kind) in on_slurm:
                        continue
                    hit = eligible(work.model_name, kind)
                    if hit is None:
                        continue
                    model_dir, cells = hit
                    if dry_run:
                        log(f"botero: DRY-RUN would enqueue new {work.model_name}:{kind}"
                            f" ({cells} cells) from {model_dir}")
                    elif enqueue(conn, work.model_name, kind, model_dir, origin="new", cells=cells) is None:
                        continue
                    else:
                        log(f"botero: enqueued new {work.model_name}:{kind} ({cells} cells)")
                    taken.append(f"{work.model_name}:{kind}")
                    active.add((work.model_name, kind))

        log(f"botero: queue {have + (0 if dry_run else len(taken))}/{slots}"
            f" after {'considering' if dry_run else 'taking'} {len(taken)} unit(s)")
        return taken
    finally:
        if own:
            conn.close()


# --- CLI --------------------------------------------------------------------------------------


def _cmd_status(conn: sqlite3.Connection, args) -> int:
    rows = conn.execute(
        "SELECT * FROM botero_jobs ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1"
        " WHEN 'parked' THEN 2 ELSE 3 END, id"
    ).fetchall()
    if not rows:
        print(f"botero queue is empty (0/{config.BOTERO_SLOTS})")
        return 0
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"botero queue {active_count(conn)}/{config.BOTERO_SLOTS} active  "
          + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"{'id':>4} {'status':<9} {'pid':>7} {'cells':>5} {'try':>3}  model:kind")
    for row in rows:
        if args.all or row["status"] in ACTIVE:
            print(f"{row['id']:>4} {row['status']:<9} {row['pid'] or '-':>7} "
                  f"{row['cells_at_enqueue'] if row['cells_at_enqueue'] is not None else '-':>5} "
                  f"{row['attempts']:>3}  {row['model_name']}:{row['checkpoint_kind']}"
                  + (f"  [{row['origin']}]" if row["origin"] else ""))
            if row["last_error"]:
                print(f"{'':>4} error: {row['last_error'].strip().splitlines()[-1][:140]}")
    return 0


def _cmd_enqueue(conn: sqlite3.Connection, args) -> int:
    model_dir = resolve_model_dir(args.model, config.CKPT_FILE_FOR_KIND[args.kind])
    if model_dir is None:
        print(f"no local archive dir for {args.model} with {config.CKPT_FILE_FOR_KIND[args.kind]}",
              file=sys.stderr)
        return 1
    job_id = enqueue(conn, args.model, args.kind, model_dir, origin="manual")
    if job_id is None:
        print(f"{args.model}:{args.kind} is already queued or running", file=sys.stderr)
        return 1
    print(f"queued job {job_id}: {args.model}:{args.kind} from {model_dir}")
    return 0


def _cmd_park(conn: sqlite3.Connection, args) -> int:
    """Hold a queued unit back without giving it up to Slurm.

    Used when the runner has to be recycled: a long-lived runner imported `config` at start-up, so
    parking whatever is still queued lets it drain and exit instead of claiming the next unit with
    the old settings.
    """
    cur = conn.execute(
        "UPDATE botero_jobs SET status='parked' WHERE id=? AND status='queued'", (args.job_id,))
    print(f"parked {cur.rowcount} row(s)" if cur.rowcount else "not queued, nothing parked")
    return 0 if cur.rowcount else 1


def _cmd_unpark(conn: sqlite3.Connection, args) -> int:
    ids = [args.job_id] if args.job_id else [
        r["id"] for r in conn.execute("SELECT id FROM botero_jobs WHERE status='parked' ORDER BY id")]
    for job_id in ids:
        conn.execute("UPDATE botero_jobs SET status='queued' WHERE id=? AND status='parked'", (job_id,))
    print(f"unparked {len(ids)} row(s)")
    return 0 if ids else 1


def _cmd_reset(conn: sqlite3.Connection, args) -> int:
    cur = conn.execute(
        "UPDATE botero_jobs SET status='queued', pid=NULL, started_ts=NULL, finished_ts=NULL,"
        " attempts=0, last_error=NULL WHERE id=?",
        (args.job_id,),
    )
    print(f"reset {cur.rowcount} row(s)")
    return 0 if cur.rowcount else 1


def _cmd_drop(conn: sqlite3.Connection, args) -> int:
    cur = conn.execute("DELETE FROM botero_jobs WHERE id=?", (args.job_id,))
    print(f"deleted {cur.rowcount} row(s)")
    return 0 if cur.rowcount else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and hand-edit the Botero AA-sweep queue.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show the queue.")
    status.add_argument("--all", action="store_true", help="Include finished/failed rows.")
    status.set_defaults(func=_cmd_status)

    add = sub.add_parser("enqueue", help="Queue one (model, kind) by hand.")
    add.add_argument("model")
    add.add_argument("kind", choices=config.CHECKPOINT_KINDS)
    add.set_defaults(func=_cmd_enqueue)

    park = sub.add_parser("park", help="Hold a queued unit back (still counts against the slots).")
    park.add_argument("job_id", type=int)
    park.set_defaults(func=_cmd_park)

    unpark = sub.add_parser("unpark", help="Return parked units to the queue (all, or one id).")
    unpark.add_argument("job_id", type=int, nargs="?", default=None)
    unpark.set_defaults(func=_cmd_unpark)

    reset = sub.add_parser("reset", help="Put a failed/stuck row back in the queue.")
    reset.add_argument("job_id", type=int)
    reset.set_defaults(func=_cmd_reset)

    drop = sub.add_parser("drop", help="Delete a row outright.")
    drop.add_argument("job_id", type=int)
    drop.set_defaults(func=_cmd_drop)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = connect()
    try:
        return args.func(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
