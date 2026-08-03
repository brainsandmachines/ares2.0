#!/usr/bin/env python3
"""Botero-side AIRCC backup + DB health monitor.

This is intended to run directly after ``scripts/backup_aircc_models.sh`` from
the same cron entry. It validates the just-finished rsync block, checks the live
AIRCC DB on the sshfs mount, repairs finished rows whose best-checkpoint write
was lost, and escalates failed model rows to a read-only LLM diagnosis saved as
markdown.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shlex
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from aircc.aircc_job_manager.db import AirccDB, Job
from aircc.aircc_job_manager.notify import make_emailer

logger = logging.getLogger("aircc.daily_monitor")

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOUNT_REPO = Path.home() / "aircc_mount" / "ashtomer" / "ares"
DEFAULT_DB = DEFAULT_MOUNT_REPO / "aircc" / "aircc_job_manager" / "aircc_jobs.sqlite"
DEFAULT_BACKUP_LOG = Path("/mnt/data/robustness_models/aircc_models/backup.log")
DEFAULT_RECOMMENDATIONS_DIR = (
    DEFAULT_REPO_ROOT / "aircc" / "aircc_job_manager" / "recommendations"
)
DEFAULT_EXPECTED_RUNNING = 32
# The repair below has to run ON AIRCC: the mount is read-only, so the fix
# cannot be applied through DEFAULT_DB.
DEFAULT_SSH_HOST = os.environ.get("AIRCC_SSH_HOST", "aircc")
DEFAULT_REMOTE_REPO = "/shared/cycle2_bgu_golan_prj/ashtomer/ares"
REMOTE_PYTHON = "/usr/bin/python3"
BACKFILL_MODULE = "aircc.aircc_job_manager.scripts.backfill_best_checkpoints"

Emailer = Callable[[str, str], None]
LLMClient = Callable[[str], str]
Runner = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass
class BackupCheck:
    ok: bool
    message: str
    block: str = ""


@dataclass
class DBCheck:
    ok: bool
    running_count: int = 0
    failed_jobs: list[Job] | None = None
    message: str = ""


@dataclass
class RepairCheck:
    ok: bool
    models: list[str]
    message: str
    output: str = ""


def _extract_backup_blocks(log_text: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in log_text.splitlines():
        if line.startswith("[backup] ") and " rsync " in line:
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(block).strip() for block in blocks]


def check_backup_log(log_path: Path, backup_rc: int) -> BackupCheck:
    if backup_rc != 0:
        return BackupCheck(False, f"rsync exited with rc={backup_rc}")
    if not log_path.is_file():
        return BackupCheck(False, f"backup log is missing: {log_path}")

    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        return BackupCheck(False, f"cannot read backup log {log_path}: {exc}")

    blocks = _extract_backup_blocks(text)
    if not blocks:
        return BackupCheck(False, f"backup log has no rsync block: {log_path}")
    latest = blocks[-1]
    if "ERROR:" in latest or "[backup] ERROR" in latest:
        return BackupCheck(False, "latest backup block contains ERROR", latest)
    required = ("Number of files:", "Total file size:", "sent ", "speedup is")
    missing = [item for item in required if item not in latest]
    if missing:
        return BackupCheck(False, f"latest backup block is missing {', '.join(missing)}", latest)
    if not any(line.startswith("[backup] ") and line.endswith(" done")
               for line in latest.splitlines()):
        return BackupCheck(False, "latest backup block has no done marker", latest)
    return BackupCheck(True, "backup log validates", latest)


def check_db(db_path: Path, expected_running: int) -> DBCheck:
    if not db_path.is_file():
        return DBCheck(False, message=f"AIRCC DB is missing: {db_path}")
    try:
        jobs = AirccDB(str(db_path)).all_jobs()
    except sqlite3.Error as exc:
        return DBCheck(False, message=f"cannot read AIRCC DB {db_path}: {exc}")

    running_count = sum(1 for job in jobs if job.status == "running")
    failed = [job for job in jobs if job.status == "failed"]
    problems = []
    if running_count != expected_running:
        problems.append(f"running={running_count}, expected={expected_running}")
    if failed:
        problems.append(f"failed={len(failed)}")
    return DBCheck(
        ok=not problems,
        running_count=running_count,
        failed_jobs=failed,
        message="; ".join(problems) if problems else "AIRCC DB validates",
    )


def find_missing_best_checkpoints(db_path: Path) -> list[str]:
    """Finished models whose best_checkpoint never made it into the DB.

    ``progress.write_best_checkpoint`` gets one shot per run inside the training
    process and swallows DB errors, so a transient sqlite CANTOPEN on the shared
    FS leaves a finished row with a NULL best_checkpoint -- which then silently
    gates every continuation row depending on it. Nothing fails, nothing is
    logged where anyone looks, so this check is the only thing that surfaces it.

    Read with ``immutable=1``: no locking, ignores the -wal, safe on the
    read-only mount. That can lag the live DB by seconds, so a row that was
    repaired moments ago may still show up here -- harmless, the repair is
    idempotent and reports "nothing to backfill".
    """
    if not db_path.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT model_name FROM jobs WHERE status='finished' "
            "AND (best_checkpoint IS NULL OR TRIM(best_checkpoint)='') "
            "ORDER BY model_name"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("cannot scan for missing best checkpoints: %s", exc)
        return []
    finally:
        conn.close()
    return [r[0] for r in rows]


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=300)


def repair_missing_best_checkpoints(
    models: list[str],
    *,
    ssh_host: str = DEFAULT_SSH_HOST,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    runner: Optional[Runner] = None,
) -> RepairCheck:
    """Re-derive the missing best checkpoints from on-disk AutoAttack output.

    Runs ``scripts/backfill_best_checkpoints.py --apply`` over ssh on the AIRCC
    login node. Idempotent and conservative: it only touches finished rows whose
    best_checkpoint is NULL, and takes the score from the same
    ``autoattack_eps_norm_scores.json`` the in-training hook would have used --
    it never re-evaluates anything.

    Slurm/login-node commands need a login shell, and the whole thing must be one
    ssh argv item (see the aircc-status skill notes).
    """
    if not models:
        return RepairCheck(True, [], "no missing best checkpoints")

    runner = runner or _default_runner
    remote = f"cd {remote_repo} && {REMOTE_PYTHON} -m {BACKFILL_MODULE} --apply"
    argv = ["ssh", "-o", "ConnectTimeout=15", ssh_host, f"bash -lc {shlex.quote(remote)}"]
    try:
        proc = runner(argv)
    except Exception as exc:
        return RepairCheck(False, models, f"backfill could not run: {exc}")

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return RepairCheck(False, models, f"backfill exited rc={proc.returncode}", output)
    # The remote script decides what it actually wrote (the mount read that
    # flagged these models lags the live DB), so quote its own summary rather
    # than claiming len(models) rows were repaired.
    summary = next((ln for ln in reversed(output.splitlines())
                    if ln.startswith(("wrote ", "nothing to backfill"))), "see output")
    return RepairCheck(True, models, f"backfill ran for {len(models)} flagged row(s): {summary}",
                       output)


def _job_signature(job: Job) -> str:
    raw = "\n".join([
        job.model_name,
        job.status,
        str(job.current_epoch),
        str(job.total_epochs),
        job.last_error or "",
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def make_llm_client(cmd_name: Optional[str] = None) -> Optional[LLMClient]:
    cmd_name = cmd_name or os.environ.get("ORCH_LLM_CMD", "codex")
    cmd = shutil.which(cmd_name)
    if not cmd:
        logger.info("no `%s` CLI on PATH; AIRCC reports will be raw-only", cmd_name)
        return None
    argv = [cmd, "exec", "--skip-git-repo-check"] if os.path.basename(cmd) == "codex" else [cmd, "-p"]

    def _client(prompt: str) -> str:
        proc = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{cmd_name} CLI failed: {proc.stderr[:500]}")
        return proc.stdout.strip()

    return _client


def _failure_prompt(job: Job, mount_repo: Path) -> str:
    return (
        "You are inspecting a failed AIRCC ARES training run through the mounted "
        "AIRCC repo. Do not modify files. Produce concise markdown with: "
        "(1) one-line root cause, (2) suggested fix, (3) exact files/configs/logs "
        "to inspect or change.\n\n"
        f"Mounted repo: {mount_repo}\n"
        f"Model: {job.model_name}\n"
        f"Status: {job.status}\n"
        f"Epoch: {job.current_epoch}/{job.total_epochs}\n"
        f"Owner task: {job.owner_task}\n"
        f"Checkpoint: {job.checkpoint or 'n/a'}\n"
        f"Best checkpoint: {job.best_checkpoint or 'n/a'}\n"
        f"Last error:\n```\n{job.last_error or '(empty)'}\n```\n\n"
        "Likely log directory: "
        f"{mount_repo / 'aircc' / 'aircc_job_manager' / 'logs'}\n"
    )


def diagnose_failed_job(
    job: Job,
    recommendations_dir: Path,
    mount_repo: Path,
    llm_client: Optional[LLMClient],
) -> Path:
    recommendations_dir.mkdir(parents=True, exist_ok=True)
    sig = _job_signature(job)
    report_path = recommendations_dir / f"{job.model_name}.{sig[:12]}.md"
    if report_path.exists():
        return report_path

    report_md = None
    if llm_client is not None:
        try:
            report_md = llm_client(_failure_prompt(job, mount_repo))
        except Exception as exc:
            logger.exception("AIRCC LLM diagnosis failed for %s: %s", job.model_name, exc)

    if not report_md:
        report_md = (
            f"# AIRCC failure report\n\n"
            f"- model: `{job.model_name}`\n"
            f"- signature: `{sig}`\n"
            f"- time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- mounted repo: `{mount_repo}`\n\n"
            "## Last error\n\n"
            f"```\n{job.last_error or '(empty)'}\n```\n"
        )
    report_path.write_text(report_md)
    return report_path


def _send(emailer: Optional[Emailer], subject: str, body: str) -> None:
    if emailer is None:
        logger.warning("no emailer configured; would send %s\n%s", subject, body)
        return
    emailer(subject, body)


def run_once(
    *,
    db_path: Path = DEFAULT_DB,
    backup_log: Path = DEFAULT_BACKUP_LOG,
    backup_rc: int = 0,
    expected_running: int = DEFAULT_EXPECTED_RUNNING,
    recommendations_dir: Path = DEFAULT_RECOMMENDATIONS_DIR,
    mount_repo: Path = DEFAULT_MOUNT_REPO,
    emailer: Optional[Emailer] = None,
    llm_client: Optional[LLMClient] = None,
    ssh_host: str = DEFAULT_SSH_HOST,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    repair: bool = True,
    runner: Optional[Runner] = None,
) -> int:
    backup = check_backup_log(backup_log, backup_rc)
    db = check_db(db_path, expected_running)
    missing_best = find_missing_best_checkpoints(db_path)

    rc = 0
    if not backup.ok:
        rc = 1
        _send(
            emailer,
            "[aircc] backup rsync problem",
            f"AIRCC backup problem: {backup.message}\n\n"
            f"backup log: {backup_log}\n\n"
            f"latest block:\n{backup.block or '(unavailable)'}\n",
        )
    if not db.ok:
        rc = 1
        _send(
            emailer,
            "[aircc] DB health problem",
            f"AIRCC DB health problem: {db.message}\n\n"
            f"db: {db_path}\n"
            f"running: {db.running_count}\n"
            f"failed: {len(db.failed_jobs or [])}\n",
        )

    if missing_best:
        listing = "\n".join(f"  - {m}" for m in missing_best)
        if not repair:
            rc = 1
            _send(
                emailer,
                "[aircc] finished rows with no best checkpoint",
                f"{len(missing_best)} finished model(s) have no best_checkpoint, so any "
                f"continuation row depending on them is silently gated:\n{listing}\n\n"
                f"repair is disabled (--no-repair); run on {ssh_host}:\n"
                f"  cd {remote_repo} && {REMOTE_PYTHON} -m {BACKFILL_MODULE} --apply\n",
            )
        else:
            fix = repair_missing_best_checkpoints(
                missing_best, ssh_host=ssh_host, remote_repo=remote_repo, runner=runner)
            if not fix.ok:
                rc = 1
            _send(
                emailer,
                f"[aircc] best-checkpoint backfill {'ok' if fix.ok else 'FAILED'}",
                f"{len(missing_best)} finished model(s) had no best_checkpoint, gating any "
                f"continuation row depending on them:\n{listing}\n\n"
                f"result: {fix.message}\n\n{fix.output or '(no output)'}\n",
            )
            logger.info("AIRCC best-checkpoint repair: %s", fix.message)

    for job in db.failed_jobs or []:
        report_path = diagnose_failed_job(job, recommendations_dir, mount_repo, llm_client)
        _send(
            emailer,
            f"[aircc] failure on {job.model_name}",
            f"there is a failure on {job.model_name}, suggested fix is saved "
            f"as a markdown file at {report_path}\n\n"
            f"last_error: {job.last_error or '(empty)'}\n",
        )

    logger.info("AIRCC daily monitor done: backup=%s db=%s", backup.message, db.message)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="AIRCC backup-linked daily monitor.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--backup-log", type=Path, default=DEFAULT_BACKUP_LOG)
    ap.add_argument("--backup-rc", type=int, default=0)
    ap.add_argument("--expected-running", type=int, default=DEFAULT_EXPECTED_RUNNING)
    ap.add_argument("--recommendations-dir", type=Path, default=DEFAULT_RECOMMENDATIONS_DIR)
    ap.add_argument("--mount-repo", type=Path, default=DEFAULT_MOUNT_REPO)
    ap.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    ap.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    ap.add_argument("--no-repair", dest="repair", action="store_false",
                    help="only report finished rows missing a best_checkpoint, "
                         "do not run the backfill on AIRCC")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_once(
        db_path=args.db,
        backup_log=args.backup_log,
        backup_rc=args.backup_rc,
        expected_running=args.expected_running,
        recommendations_dir=args.recommendations_dir,
        mount_repo=args.mount_repo,
        emailer=make_emailer(),
        llm_client=make_llm_client(),
        ssh_host=args.ssh_host,
        remote_repo=args.remote_repo,
        repair=args.repair,
    )


if __name__ == "__main__":
    raise SystemExit(main())
