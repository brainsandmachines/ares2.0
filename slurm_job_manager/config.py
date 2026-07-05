"""Configuration for the Slurm job manager, loaded from ``slurm_job_manager/.env``.

Only the botero-side **monitor** (failure escalation) and the ``codex``/email
transports need config; the cluster controller reads just ``SJM_DB`` from its
sbatch. All values are overridable via the ``.env`` file or the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Job name carried by every manager array (lets squeue/sacct filter to ours).
MANAGER_JOB_NAME = "sjm-manager"


def _load_dotenv(path: str) -> dict[str, str]:
    """Minimal KEY=VALUE .env parser (no external dependency)."""
    out: dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            out[key.strip()] = val.strip().strip('"').strip("'")
    return out


@dataclass
class Config:
    # DB path as seen from BOTERO (the monitor reads/writes here).
    db_path: str = "/home/tomer_a/Documents/ares/slurm_job_manager/jobs.sqlite"
    # DB path as seen from a CLUSTER job (exported to the sbatch as SJM_DB).
    db_path_cluster: str = "/home/ashtomer/projects/ares/slurm_job_manager/jobs.sqlite"

    ssh_host: str = "slurm"           # ssh alias for the submit/login node
    cluster_user: str = "ashtomer"
    cluster_repo: str = "/home/ashtomer/projects/ares"
    logs_root: str = "/home/ashtomer/projects/ares/outs"

    # failure analyzer / codex escalation
    enable_failure_analyzer: bool = True
    failure_lookback_hours: float = 2.0
    recommendations_dir: str = "/home/tomer_a/Documents/ares/slurm_job_manager/recommendations"
    alert_email: Optional[str] = "tomerash98@gmail.com"

    @classmethod
    def load(cls, dotenv_path: Optional[str] = None) -> "Config":
        here = os.path.dirname(os.path.abspath(__file__))
        env = _load_dotenv(dotenv_path or os.path.join(here, ".env"))
        # Expose .env to os.environ-based readers (SMTP + LLM) without overriding
        # anything already set in the real env.
        for _k, _v in env.items():
            os.environ.setdefault(_k, _v)

        def g(key: str, default):
            return env.get(key, default)

        return cls(
            db_path=g("SJM_DB", cls.db_path),
            db_path_cluster=g("SJM_DB_CLUSTER", cls.db_path_cluster),
            ssh_host=g("SJM_SSH_HOST", cls.ssh_host),
            cluster_user=g("SJM_CLUSTER_USER", cls.cluster_user),
            cluster_repo=g("SJM_CLUSTER_REPO", cls.cluster_repo),
            logs_root=g("SJM_LOGS_ROOT", cls.logs_root),
            enable_failure_analyzer=str(g("SJM_ENABLE_FAILURE_ANALYZER", "1")).lower()
            in ("1", "true", "yes"),
            failure_lookback_hours=float(g("SJM_FAILURE_LOOKBACK_HOURS", cls.failure_lookback_hours)),
            recommendations_dir=g("SJM_RECOMMENDATIONS_DIR", cls.recommendations_dir),
            alert_email=g("SJM_ALERT_EMAIL", cls.alert_email) or None,
        )
