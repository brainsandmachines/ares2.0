"""Botero-side Slurm client: squeue/sacct accounting + array submission over SSH.

Everything that talks to the cluster goes through a ``runner`` callable so the
logic is fully testable offline (inject a fake runner; no real ssh needed). In
the pull model the cluster array tasks claim their own work from the DB, so this
module no longer pushes per-model jobs -- it submits *controller array pools* and
reports their state to the monitor.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Callable, Optional

from .config import Config

logger = logging.getLogger("orchestrator.core")

# A runner takes an argv list and returns stdout (raising on non-zero exit).
Runner = Callable[[list[str]], str]

# Slurm states that occupy or reserve a slot.
ACTIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESIZING"}

# The single array launcher used by every controller pool.
CONTROLLER_SBATCH = "sbatches/controller.sbatch"
# Job name carried by every controller array (lets squeue/sacct filter to ours).
CONTROLLER_JOB_NAME = "orch-controller"


def subprocess_runner(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return proc.stdout


class SlurmClient:
    """Thin SSH wrapper around squeue / sbatch / sacct / scontrol."""

    def __init__(self, cfg: Config, runner: Runner = subprocess_runner):
        self.cfg = cfg
        self.runner = runner

    def _ssh(self, remote_cmd: str) -> str:
        return self.runner(["ssh", self.cfg.ssh_host, remote_cmd])

    # ---- accounting ------------------------------------------------------ #
    def squeue_raw(self) -> str:
        """Expanded per-task listing: ``JOBID|PARTITION|STATE|NAME|REASON`` lines.

        ``-r`` expands array elements so each running/pending task is its own
        line; a not-yet-started array pack still appears as a single
        ``<id>_[lo-hi%N]`` line (the monitor parses its range). The REASON field
        flags a queued dependent pool (``Dependency``).
        """
        return self._ssh(
            f"squeue -u {self.cfg.cluster_user} -h -r -o '%i|%P|%T|%j|%r'"
        )

    def failed_controller_tasks(self, since_hours: float = 2.0) -> list[int]:
        """Unique per-task ids (JobIDRaw) of controller tasks that ended FAILED
        within the recent window, for mapping back to owning DB rows."""
        import time as _time

        start = _time.strftime(
            "%Y-%m-%dT%H:%M:%S", _time.localtime(_time.time() - since_hours * 3600)
        )
        out = self._ssh(
            f"sacct -u {self.cfg.cluster_user} --name {CONTROLLER_JOB_NAME} "
            f"--starttime {start} -n -X -o JobIDRaw,State 2>/dev/null"
        )
        failed: list[int] = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].upper().startswith(("FAILED", "OUT_OF_ME", "NODE_FAIL")):
                try:
                    failed.append(int(parts[0]))
                except ValueError:
                    continue
        return failed

    def submit_controller_array(self, partition: str, array_spec: str) -> int:
        """Submit one controller array pool; return its (array) job id.

        ``array_spec`` is e.g. ``1-200%6``. Pools are submitted immediately;
        Slurm's array throttle controls concurrency.
        """
        export = f"ALL,ORCH_DB={self.cfg.db_path_cluster}"
        remote = (
            f"cd {self.cfg.cluster_repo} && "
            f"sbatch --parsable --job-name={CONTROLLER_JOB_NAME} "
            f"--partition={partition} --array={array_spec} "
            f"--export={export} {CONTROLLER_SBATCH}"
        )
        out = self._ssh(remote).strip()
        return int(out.split(";")[0].split()[0])

    def job_state(self, slurm_job_id: int) -> Optional[str]:
        """Return the Slurm state of a job via sacct, or None if unknown."""
        out = self._ssh(f"sacct -j {slurm_job_id} -n -X -o State%30 2>/dev/null")
        states = [s.strip() for s in out.splitlines() if s.strip()]
        return states[0].split()[0] if states else None

    def is_active(self, slurm_job_id: int) -> bool:
        state = self.job_state(slurm_job_id)
        return state is not None and state.upper() in ACTIVE_STATES

    def failed_tasks(self, array_job_id: int) -> list[int]:
        """Per-task JobIDs (e.g. ``123_4`` -> step ids) that ended FAILED."""
        out = self._ssh(
            f"sacct -j {array_job_id} -n -X -o JobID,State 2>/dev/null"
        )
        failed: list[int] = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].upper().startswith("FAILED"):
                failed.append(parts[0])
        return failed

    def read_log(self, slurm_job_id: int, tail_lines: int = 300) -> str:
        """Best-effort fetch of a job's stdout tail (for failure analysis)."""
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
