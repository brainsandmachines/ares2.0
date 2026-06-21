"""Botero-side hourly monitor: keep the controller array pools fed + healthy.

Run once per hour from cron. In the pull model the controller array tasks claim
and process work themselves, so the monitor only has to:

  1. Top up: if a partition has fewer than ORCH_MIN_REMAINING controller tasks
     left (and there is claimable DB work, and no dependent pool is already
     queued), submit the next pool as an ``afterany`` dependency of the latest
     one so pools run back-to-back without stacking concurrency.
  2. Settle: if a new pool was submitted, wait 30s so step 3 sees steady state.
  3. Capacity check: confirm the number of running controller tasks matches the
     expected cap (min(partition capacity, work available)); if not, wait 30s
     and re-check once (the just-submitted pool may be spinning up).
  4. Failures: map recently-FAILED controller tasks back to their DB rows and
     act deterministically (requeue transient / mark FAILED), escalating only
     unrecognized signatures to the codex diagnosis CLI (deduped, once each).

Everything before the codex escalation is deterministic. The SlurmClient runner
is injectable, so the whole monitor is testable offline with canned squeue/sacct.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .classify import ACTION_FAIL, ACTION_REQUEUE, ACTION_UNKNOWN, classify
from .config import Config
from .core import ACTIVE_STATES, CONTROLLER_JOB_NAME, SlurmClient
from .db import ACTIVE_STATUSES, OrchestratorDB
from .failure_analyzer import analyze_failure
from . import notify

logger = logging.getLogger("orchestrator.monitor")

_RUNNING_STATES = {"RUNNING", "CONFIGURING", "COMPLETING", "RESIZING"}
_SETTLE_SECONDS = 30


@dataclass
class PartitionState:
    """Aggregated controller-array state for one partition."""
    running: int = 0
    pending: int = 0
    array_job_ids: set[int] = field(default_factory=set)
    latest_array_id: Optional[int] = None
    has_queued_dependent: bool = False

    @property
    def remaining(self) -> int:
        """Tasks not yet consumed (running + still-queued)."""
        return self.running + self.pending


def _array_task_count(jobid_field: str) -> int:
    """Number of tasks a squeue JOBID field represents.

    ``123_4`` -> 1; a pending pack ``123_[7-200%6]`` -> 194 (sum of ranges,
    ignoring the ``%N`` throttle); ``123_[3,5,9]`` -> 3.
    """
    m = re.search(r"_\[(.+?)\]", jobid_field)
    if not m:
        return 1
    spec = m.group(1).split("%", 1)[0]
    total = 0
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                total += int(hi) - int(lo) + 1
            except ValueError:
                total += 1
        elif part:
            total += 1
    return total


def parse_controller_state(squeue_text: str) -> dict[str, PartitionState]:
    """Parse ``squeue_raw`` output into per-partition controller state."""
    states: dict[str, PartitionState] = {}
    for line in squeue_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        fields = line.split("|")
        if len(fields) < 4:
            continue
        jobid_field, partition, state, name = fields[0], fields[1], fields[2], fields[3]
        reason = fields[4] if len(fields) > 4 else ""
        if name.strip() != CONTROLLER_JOB_NAME:
            continue
        ps = states.setdefault(partition.strip(), PartitionState())
        count = _array_task_count(jobid_field)
        st = state.strip().upper()
        if st in _RUNNING_STATES:
            ps.running += count
        elif st == "PENDING":
            ps.pending += count
            if "dependency" in reason.strip().lower():
                ps.has_queued_dependent = True
        try:
            array_id = int(jobid_field.split("_", 1)[0])
            ps.array_job_ids.add(array_id)
            ps.latest_array_id = max(ps.latest_array_id or 0, array_id)
        except ValueError:
            pass
    return states


def _array_spec(cfg: Config, partition: str) -> str:
    size = int(cfg.array_size)
    concurrency = int(cfg.node_capacity.get(partition, 6))
    return f"1-{size}%{concurrency}"


def maybe_topup(
    db: OrchestratorDB,
    slurm: SlurmClient,
    cfg: Config,
    partition: str,
    state: PartitionState,
    dry_run: bool = False,
) -> Optional[int]:
    """Submit the next controller pool for ``partition`` if warranted.

    Returns the new array job id (or None if nothing was submitted).
    """
    claimable = db.claimable_count(partition)
    if state.remaining >= cfg.min_remaining:
        return None
    if claimable <= 0:
        logger.info("%s: %d tasks left but no claimable work; not topping up",
                    partition, state.remaining)
        return None
    if state.has_queued_dependent:
        logger.info("%s: dependent pool already queued; not topping up", partition)
        return None
    spec = _array_spec(cfg, partition)
    dep = state.latest_array_id
    if dry_run:
        logger.info("[dry-run] %s: would submit %s (afterany=%s)", partition, spec, dep)
        return None
    new_id = slurm.submit_controller_array(partition, spec, dependency_job_id=dep)
    logger.info("%s: submitted controller pool %s array=%s afterany=%s "
                "(remaining=%d, claimable=%d)",
                partition, new_id, spec, dep, state.remaining, claimable)
    return new_id


def capacity_check(
    db: OrchestratorDB, cfg: Config, partition: str, state: PartitionState
) -> bool:
    """True if running controller tasks match the expected cap for the partition."""
    cap = int(cfg.node_capacity.get(partition, 0))
    expected = min(cap, db.claimable_count(partition) + state.running)
    ok = state.running >= expected
    if not ok:
        logger.warning("%s: capacity low (running=%d expected=%d)",
                       partition, state.running, expected)
    return ok


def inspect_failures(
    db: OrchestratorDB,
    slurm: SlurmClient,
    cfg: Config,
    dry_run: bool = False,
) -> int:
    """Map recently-FAILED controller tasks to DB rows and act deterministically."""
    llm = notify.make_llm_client() if cfg.enable_failure_analyzer else None
    emailer = notify.make_emailer(cfg)
    acted = 0
    for task_id in slurm.failed_controller_tasks(since_hours=cfg.failure_lookback_hours):
        row = db.get_by_slurm_job_id(task_id)
        # Only a still-owned, still-active row needs handling; if the controller
        # already classified it (ownership cleared / terminal), skip.
        if row is None or row.status not in ACTIVE_STATUSES:
            continue
        log_text = slurm.read_log(task_id)
        action = classify(log_text)
        if dry_run:
            logger.info("[dry-run] %s (task %s) -> %s", row.model_id, task_id, action)
            continue
        if action == ACTION_REQUEUE:
            db.requeue(row.model_id)
            logger.info("monitor requeued %s (task %s, transient)", row.model_id, task_id)
        elif action == ACTION_FAIL:
            analyze_failure(db, row.model_id, log_text, cfg.recommendations_dir,
                            llm_client=None, emailer=emailer)
            logger.info("monitor marked %s FAILED (deterministic)", row.model_id)
        else:  # ACTION_UNKNOWN -> escalate to codex (deduped, once per signature)
            analyze_failure(db, row.model_id, log_text, cfg.recommendations_dir,
                            llm_client=llm, emailer=emailer)
            logger.info("monitor escalated %s (task %s, unknown signature)",
                        row.model_id, task_id)
        acted += 1
    return acted


def run_once(cfg: Optional[Config] = None, dry_run: bool = False) -> None:
    cfg = cfg or Config.load()
    db = OrchestratorDB(cfg.db_path)
    slurm = SlurmClient(cfg)

    states = parse_controller_state(slurm.squeue_raw())
    submitted_any = False
    for partition in cfg.node_capacity:
        state = states.get(partition, PartitionState())
        if maybe_topup(db, slurm, cfg, partition, state, dry_run=dry_run):
            submitted_any = True

    # 2. settle, then 3. capacity check (re-read squeue once if a pool was added).
    if submitted_any and not dry_run:
        time.sleep(_SETTLE_SECONDS)
        states = parse_controller_state(slurm.squeue_raw())
    for partition in cfg.node_capacity:
        state = states.get(partition, PartitionState())
        if not capacity_check(db, cfg, partition, state) and not dry_run:
            time.sleep(_SETTLE_SECONDS)
            state = parse_controller_state(slurm.squeue_raw()).get(partition, PartitionState())
            capacity_check(db, cfg, partition, state)

    # 4. failures
    inspect_failures(db, slurm, cfg, dry_run=dry_run)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_once()


if __name__ == "__main__":
    main()
