"""Monitor tests: squeue parsing, top-up decisions, capacity, failure mapping.

A fake SSH runner serves canned squeue/sbatch/sacct/scontrol/tail output so the
whole monitor is exercised without a cluster.
"""

from orchestrator import monitor
from orchestrator.config import Config
from orchestrator.core import SlurmClient
from orchestrator.db import OrchestratorDB, STATUS_TRAINING


class FakeRunner:
    def __init__(self, squeue="", failed_raw="", log="boom"):
        self.squeue = squeue
        self.failed_raw = failed_raw
        self.log = log
        self.calls = []
        self._next = 9000

    def __call__(self, args):
        remote = args[-1]
        self.calls.append(remote)
        if "squeue" in remote:
            return self.squeue
        if remote.startswith("cd ") and "sbatch" in remote:
            self._next += 1
            return f"{self._next}\n"
        if "sacct" in remote and "--name" in remote:
            return self.failed_raw
        if "scontrol show job" in remote:
            return "JobId=1 StdOut=/tmp/x.out JobName=orch-controller"
        if remote.startswith("tail"):
            return self.log
        return ""


def _cfg(tmp_path):
    return Config(db_path=str(tmp_path / "o.db"), db_path_cluster="/cluster/o.db",
                  enable_failure_analyzer=False, alert_email=None)


# --------------------------------------------------------------------------
# squeue parsing
# --------------------------------------------------------------------------
def test_array_task_count():
    assert monitor._array_task_count("123_4") == 1
    assert monitor._array_task_count("123_[7-200%6]") == 194
    assert monitor._array_task_count("123_[3,5,9]") == 3
    assert monitor._array_task_count("123_[10-12,20]") == 4


def test_parse_controller_state_counts_and_dependency():
    sq = "\n".join([
        "501_1|rtx_pro_6000|RUNNING|orch-controller|None",
        "501_2|rtx_pro_6000|RUNNING|orch-controller|None",
        "501_[3-200%6]|rtx_pro_6000|PENDING|orch-controller|Resources",
        "777_[1-200%6]|rtx_pro_6000|PENDING|orch-controller|Dependency",
        "999_1|rtx6000|RUNNING|some-other-job|None",        # ignored (name)
    ])
    states = monitor.parse_controller_state(sq)
    pro = states["rtx_pro_6000"]
    assert pro.running == 2
    assert pro.pending == 198 + 200
    assert pro.has_queued_dependent is True
    assert pro.latest_array_id == 777
    assert "rtx6000" not in states            # other-job filtered out


# --------------------------------------------------------------------------
# top-up decisions
# --------------------------------------------------------------------------
def _seed(db, n, status=None):
    for i in range(n):
        db.upsert_model(f"m{i:02d}", "golan-trainmodels", f"j{i}", f"/d/{i}", 0, 0, 250)
        if status:
            db.force_status(f"m{i:02d}", status)


def test_topup_submits_when_low_and_work_exists(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    _seed(db, 5)
    cfg = _cfg(tmp_path)
    runner = FakeRunner()
    slurm = SlurmClient(cfg, runner=runner)
    state = monitor.PartitionState(running=2, pending=5, latest_array_id=501)
    new_id = monitor.maybe_topup(db, slurm, cfg, "rtx_pro_6000", state)
    assert new_id == 9001
    sbatch = [c for c in runner.calls if "sbatch" in c][0]
    assert "--array=1-200%6" in sbatch
    assert "--partition=rtx_pro_6000" in sbatch
    assert "--dependency=afterany:501" in sbatch


def test_topup_skips_when_enough_remaining(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    _seed(db, 5)
    cfg = _cfg(tmp_path)
    slurm = SlurmClient(cfg, runner=FakeRunner())
    state = monitor.PartitionState(running=6, pending=30, latest_array_id=501)
    assert monitor.maybe_topup(db, slurm, cfg, "rtx_pro_6000", state) is None


def test_topup_skips_when_no_claimable_work(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))   # empty queue
    cfg = _cfg(tmp_path)
    slurm = SlurmClient(cfg, runner=FakeRunner())
    state = monitor.PartitionState(running=0, pending=0)
    assert monitor.maybe_topup(db, slurm, cfg, "rtx_pro_6000", state) is None


def test_topup_skips_when_dependent_already_queued(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    _seed(db, 5)
    cfg = _cfg(tmp_path)
    slurm = SlurmClient(cfg, runner=FakeRunner())
    state = monitor.PartitionState(running=1, pending=1, latest_array_id=501,
                                   has_queued_dependent=True)
    assert monitor.maybe_topup(db, slurm, cfg, "rtx_pro_6000", state) is None


def test_topup_bootstrap_without_dependency(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    _seed(db, 3)
    cfg = _cfg(tmp_path)
    runner = FakeRunner()
    slurm = SlurmClient(cfg, runner=runner)
    state = monitor.PartitionState()              # no arrays at all
    new_id = monitor.maybe_topup(db, slurm, cfg, "rtx6000", state)
    assert new_id == 9001
    sbatch = [c for c in runner.calls if "sbatch" in c][0]
    assert "--array=1-200%8" in sbatch            # rtx6000 concurrency
    assert "--dependency" not in sbatch           # nothing to chain to


def test_capacity_check(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    _seed(db, 10)
    cfg = _cfg(tmp_path)
    assert monitor.capacity_check(db, cfg, "rtx_pro_6000",
                                  monitor.PartitionState(running=6)) is True
    assert monitor.capacity_check(db, cfg, "rtx_pro_6000",
                                  monitor.PartitionState(running=3)) is False


# --------------------------------------------------------------------------
# failure mapping
# --------------------------------------------------------------------------
def test_inspect_failures_requeues_transient(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("m1", "golan-trainmodels", "l2_8_init1", "/d/m1", 0, 0, 250)
    db.claim_next("rtx6000", 12345)               # owned by task 12345, TRAINING
    cfg = _cfg(tmp_path)
    runner = FakeRunner(failed_raw="12345 FAILED\n", log="CANCELLED DUE TO TIME LIMIT")
    slurm = SlurmClient(cfg, runner=runner)
    n = monitor.inspect_failures(db, slurm, cfg)
    assert n == 1
    row = db.get_model_state("m1")
    assert row.slurm_job_id is None and row.status == STATUS_TRAINING and row.requeued == 1


def test_inspect_failures_marks_deterministic_failed(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("m1", "golan-trainmodels", "l2_8_init1", "/d/m1", 0, 0, 250)
    db.claim_next("rtx6000", 22222)
    cfg = _cfg(tmp_path)
    runner = FakeRunner(failed_raw="22222 FAILED\n",
                        log="ModuleNotFoundError: No module named 'dvd'")
    slurm = SlurmClient(cfg, runner=runner)
    n = monitor.inspect_failures(db, slurm, cfg)
    assert n == 1
    assert db.get_model_state("m1").status == "FAILED"


def test_inspect_failures_skips_unmapped_or_unowned(tmp_path):
    db = OrchestratorDB(str(tmp_path / "o.db"))
    db.upsert_model("m1", "golan-trainmodels", "l2_8_init1", "/d/m1", 0, 0, 250)
    # task id not owning any row -> nothing to do
    cfg = _cfg(tmp_path)
    runner = FakeRunner(failed_raw="88888 FAILED\n", log="whatever")
    slurm = SlurmClient(cfg, runner=runner)
    assert monitor.inspect_failures(db, slurm, cfg) == 0
