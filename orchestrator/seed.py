"""Seed the ``models_queue`` table from a version-controlled experiment list.

Deterministic: re-running upserts the same rows (existing runtime state such as
status/current_epoch is preserved by ``OrchestratorDB.upsert_model``).

Usage:
    python -m orchestrator.seed --db /mnt/slurm/orchestrator.db \
        --experiments orchestrator/experiments.yaml
"""

from __future__ import annotations

import argparse
import os

import yaml

from .db import OrchestratorDB
from . import naming

# Launcher name -> sbatch script that the orchestrator submits over SSH.
LAUNCHER_SBATCH = {
    "golan-trainmodels": "sbatches/golan-trainmodels.sbatch",
    "eps_curriculum": "sbatches/epsilon_curriculum.sbatch",
}


def seed(db_path: str, experiments_path: str) -> int:
    with open(experiments_path) as fh:
        doc = yaml.safe_load(fh)

    cfg = doc.get("config", {})
    models_root = cfg.get("models_root", "/home/ashtomer/projects/ares/results/models")
    default_node = cfg.get("default_node", None)
    default_priority = int(cfg.get("default_priority", 100))

    db = OrchestratorDB(db_path)
    count = 0
    for exp in doc.get("experiments", []):
        launcher = exp["launcher"]
        if launcher not in LAUNCHER_SBATCH:
            raise ValueError(f"unknown launcher '{launcher}' in {experiments_path}")
        job_name = exp["job_name"]
        model_id = exp.get("model_id", job_name)
        model_dir = exp.get("model_dir") or naming.model_dir_for(job_name, models_root)
        # arch/protocol/eps: explicit if given, else derived from the job name.
        arch = exp.get("arch") or naming.arch_from_name(job_name)
        protocol = exp.get("protocol") or naming.protocol_from_name(job_name)
        eps = exp.get("eps")
        if eps is None:
            eps = naming.eps_from_name(job_name)
        db.upsert_model(
            model_id=model_id,
            launcher=launcher,
            job_name=job_name,
            model_dir=model_dir,
            warmup_epochs=int(exp.get("warmup_epochs", 0)),
            linear_epochs=int(exp.get("linear_epochs", 0)),
            plateau_epochs=int(exp.get("plateau_epochs", 0)),
            cluster_node=exp.get("node", default_node),
            priority=int(exp.get("priority", default_priority)),
            arch=arch,
            protocol=protocol,
            eps=eps,
        )
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed models_queue from experiments.yaml")
    ap.add_argument("--db", required=True, help="Path to the orchestrator SQLite DB")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        "--experiments",
        default=os.path.join(here, "experiments.yaml"),
        help="Path to the experiment list YAML",
    )
    args = ap.parse_args()
    n = seed(args.db, args.experiments)
    print(f"[seed] upserted {n} experiments into {args.db}")


if __name__ == "__main__":
    main()
