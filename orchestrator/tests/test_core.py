"""Offline tests for the scheduling core (fake ssh runner; no real cluster)."""

from orchestrator.config import Config
from orchestrator.core import Orchestrator, SlurmClient
from orchestrator.db import OrchestratorDB


class FakeRunner:
    """Records argv, simulates squeue/sbatch/sacct deterministically."""

    def __init__(self, running=None):
        self.running = running or {"rtx_pro_6000": 0, "rtx6000": 0}
        self.calls = []
        self._next_job = 5000

    def __call__(self, args):
        self.calls.append(args)
        remote = args[-1]
        if "squeue" in remote:
            lines = []
            for node, n in self.running.items():
                lines += [f"{node}|RUNNING"] * n
            return "\n".join(lines) + ("\n" if lines else "")
        if "sbatch" in remote:
            self._next_job += 1
            return f"{self._next_job}\n"
        if "sacct" in remote:
            return "COMPLETED\n"
        return ""


def make(tmp_path, running=None, n_models=20):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    for i in range(n_models):
        db.upsert_model(f"m{i:02d}", "golan-trainmodels", f"job{i}", f"/d/m{i}", 0, 0, 250)
    cfg = Config(db_path=str(tmp_path / "o.db"), db_path_cluster="/cluster/o.db")
    runner = FakeRunner(running)
    orch = Orchestrator(db, SlurmClient(cfg, runner=runner), cfg)
    return db, orch, runner


def test_free_slots_math(tmp_path):
    db, orch, runner = make(tmp_path, running={"rtx_pro_6000": 2, "rtx6000": 8})
    free = orch.free_slots(orch.slurm.squeue_counts())
    assert free == {"rtx_pro_6000": 4, "rtx6000": 0}


def test_tick_fills_only_free_slots(tmp_path):
    db, orch, runner = make(tmp_path, running={"rtx_pro_6000": 2, "rtx6000": 8})
    subs = orch.tick()
    assert len(subs) == 4
    assert all(s.node == "rtx_pro_6000" for s in subs)
    assert len(db.list_models(status="RUNNING")) == 4


def test_tick_empty_cluster_fills_all_14(tmp_path):
    db, orch, runner = make(tmp_path, running={"rtx_pro_6000": 0, "rtx6000": 0})
    subs = orch.tick()
    assert len(subs) == 14  # 6 + 8


def test_tick_deterministic_and_no_double_claim(tmp_path):
    db, orch, runner = make(tmp_path, running={"rtx_pro_6000": 5, "rtx6000": 7})
    subs = orch.tick()
    ids = [s.model_id for s in subs]
    assert ids == sorted(ids)            # ordered by model_id
    assert len(ids) == len(set(ids))     # no model picked twice in a tick


def test_sbatch_export_carries_model_and_db(tmp_path):
    db, orch, runner = make(tmp_path, running={"rtx_pro_6000": 5, "rtx6000": 8})
    orch.tick()
    sbatch = [a[-1] for a in runner.calls if "sbatch" in a[-1]][0]
    assert "ORCH_MODEL_ID=" in sbatch
    assert "ORCH_DB=/cluster/o.db" in sbatch
    assert "--partition=rtx_pro_6000" in sbatch
    assert "--parsable" in sbatch


def test_dry_run_submits_nothing(tmp_path):
    db, orch, runner = make(tmp_path, running={"rtx_pro_6000": 0, "rtx6000": 0})
    subs = orch.tick(dry_run=True)
    assert len(subs) == 14
    assert not any("sbatch" in a[-1] for a in runner.calls)
    assert len(db.list_models(status="RUNNING")) == 0


def test_no_pending_means_no_submit(tmp_path):
    db, orch, runner = make(tmp_path, running={"rtx_pro_6000": 0, "rtx6000": 0}, n_models=0)
    assert orch.tick() == []
