"""Botero orchestrator core: squeue accounting + GPU top-up over SSH.

Everything that talks to the cluster goes through a ``runner`` callable so the
logic is fully testable offline (inject a fake runner; no real ssh needed).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from .config import Config
from .db import OrchestratorDB

logger = logging.getLogger("orchestrator.core")

# A runner takes an argv list and returns stdout (raising on non-zero exit).
Runner = Callable[[list[str]], str]

# Slurm states that occupy or reserve a slot for top-up accounting.
_ACTIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESIZING"}
# Deterministic node fill order.
_NODE_ORDER = ("rtx_pro_6000", "rtx6000")


def subprocess_runner(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return proc.stdout


@dataclass
class Submission:
    model_id: str
    node: str
    job_name: str
    launcher: str
    slurm_job_id: Optional[int] = None


# Launcher key -> sbatch path (kept in sync with seed.LAUNCHER_SBATCH).
LAUNCHER_SBATCH = {
    "golan-trainmodels": "sbatches/golan-trainmodels.sbatch",
    "eps_curriculum": "sbatches/epsilon_curriculum.sbatch",
}


class SlurmClient:
    """Thin SSH wrapper around squeue / sbatch / sacct."""

    def __init__(self, cfg: Config, runner: Runner = subprocess_runner):
        self.cfg = cfg
        self.runner = runner

    def _ssh(self, remote_cmd: str) -> str:
        return self.runner(["ssh", self.cfg.ssh_host, remote_cmd])

    def squeue_counts(self) -> dict[str, int]:
        """Count my active (running+pending) jobs per partition."""
        out = self._ssh(
            f"squeue -u {self.cfg.cluster_user} -h -o '%P|%T'"
        )
        counts: dict[str, int] = {node: 0 for node in self.cfg.node_capacity}
        for line in out.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            partition, _, state = line.partition("|")
            partition = partition.strip()
            if state.strip().upper() in _ACTIVE_STATES and partition in counts:
                counts[partition] += 1
        return counts

    def submit(self, launcher: str, job_name: str, node: str, model_id: str) -> int:
        """Submit a job over ssh; return the Slurm job id (via --parsable)."""
        sbatch = LAUNCHER_SBATCH[launcher]
        export = (
            f"ALL,ORCH_MODEL_ID={model_id},ORCH_DB={self.cfg.db_path_cluster}"
        )
        remote = (
            f"cd {self.cfg.cluster_repo} && "
            f"sbatch --parsable --job-name={job_name} --partition={node} "
            f"--export={export} {sbatch}"
        )
        out = self._ssh(remote).strip()
        # --parsable prints "<jobid>" or "<jobid>;<cluster>"
        return int(out.split(";")[0].split()[0])

    def job_state(self, slurm_job_id: int) -> Optional[str]:
        """Return the Slurm state of a job via sacct, or None if unknown."""
        out = self._ssh(
            f"sacct -j {slurm_job_id} -n -X -o State%30 2>/dev/null"
        )
        states = [s.strip() for s in out.splitlines() if s.strip()]
        return states[0].split()[0] if states else None

    def is_active(self, slurm_job_id: int) -> bool:
        state = self.job_state(slurm_job_id)
        return state is not None and state.upper() in _ACTIVE_STATES

    def read_log(self, slurm_job_id: int, tail_lines: int = 300) -> str:
        """Best-effort fetch of a job's stdout tail (for failure analysis).

        Uses ``scontrol`` to resolve the StdOut path while the job is still
        known, then tails it. Returns "" if the path can't be resolved (the
        daemon treats an empty log as a non-traceback timeout, never a false
        FAILED).
        """
        try:
            info = self._ssh(f"scontrol show job {slurm_job_id} -o 2>/dev/null")
        except Exception:
            return ""
        path = None
        for tok in info.split():
            if tok.startswith("StdOut="):
                path = tok[len("StdOut="):]
                break
        if not path:
            return ""
        try:
            return self._ssh(f"tail -n {tail_lines} {path} 2>/dev/null")
        except Exception:
            return ""


class Orchestrator:
    def __init__(self, db: OrchestratorDB, slurm: SlurmClient, cfg: Config):
        self.db = db
        self.slurm = slurm
        self.cfg = cfg

    def free_slots(self, counts: dict[str, int]) -> dict[str, int]:
        return {
            node: max(0, self.cfg.node_capacity.get(node, 0) - counts.get(node, 0))
            for node in self.cfg.node_capacity
        }

    def tick(self, dry_run: bool = False) -> list[Submission]:
        """One scheduling pass: top up free GPU slots from the PENDING queue.

        Returns the submissions made (or that *would* be made, when dry_run).
        Deterministic: nodes filled in fixed order, picks ordered by
        (priority, model_id), and no model is picked twice in a tick.
        """
        counts = self.slurm.squeue_counts()
        free = self.free_slots(counts)
        submissions: list[Submission] = []
        claimed: set[str] = set()

        nodes = [n for n in _NODE_ORDER if n in free] + [
            n for n in free if n not in _NODE_ORDER
        ]
        for node in nodes:
            for _ in range(free[node]):
                pick = self._next_unclaimed(node, claimed)
                if pick is None:
                    break  # no eligible PENDING work for this node
                claimed.add(pick.model_id)
                sub = Submission(
                    model_id=pick.model_id,
                    node=node,
                    job_name=pick.job_name,
                    launcher=pick.launcher,
                )
                if not dry_run:
                    job_id = self.slurm.submit(
                        pick.launcher, pick.job_name, node, pick.model_id
                    )
                    self.db.mark_running(pick.model_id, job_id, node)
                    sub.slurm_job_id = job_id
                    logger.info(
                        "submitted %s -> %s (job %s)", pick.model_id, node, job_id
                    )
                else:
                    logger.info("[dry-run] would submit %s -> %s", pick.model_id, node)
                submissions.append(sub)
        return submissions

    def _next_unclaimed(self, node: str, claimed: set[str]):
        """Pick next PENDING for node, skipping already-claimed models this tick."""
        return self.db.next_pending_for_node(node, exclude=claimed or None)

    def requeue_stale(self) -> int:
        seconds = int(self.cfg.stale_running_hours * 3600)
        n = self.db.requeue_stale_running(seconds)
        if n:
            logger.info("requeued %d stale RUNNING rows", n)
        return n
