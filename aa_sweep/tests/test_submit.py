import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aa_sweep import config, plan as plan_mod, submit as submit_mod  # noqa: E402
from aa_sweep.tests.test_census import csv_text  # noqa: E402

ALL_CELLS = [(n, e) for n in config.NORMS for e in config.EPS_INPUTS]
CKPT = config.CKPT_FILE_FOR_KIND
BIG = 1_418_062_559


def _install_fakes(monkeypatch, probe, live_names=(), aircc_finished=(), sjm_finished=()):
    """Point submit.main() at in-memory fixtures instead of the mounts and the cluster."""
    submitted = []

    monkeypatch.setattr(submit_mod, "check_mounts", lambda: [])
    monkeypatch.setattr(
        plan_mod, "finished_models",
        lambda db: list(aircc_finished) if db == config.AIRCC_DB else list(sjm_finished),
    )
    monkeypatch.setattr(plan_mod, "probe_slurm", lambda names: probe)
    monkeypatch.setattr(plan_mod, "_read_local_dir", lambda d: ({}, {}))
    monkeypatch.setattr(submit_mod, "live_job_names", lambda: set(live_names))
    monkeypatch.setattr(
        submit_mod.stage_mod, "stage_model",
        lambda work, a, s, **kw: submit_mod.stage_mod.StageResult(model_name=work.model_name),
    )

    def fake_submit(model_dir, model_name, kind, run=subprocess.run):
        submitted.append((model_name, kind))
        return "12345"

    monkeypatch.setattr(submit_mod, "submit_job", fake_submit)
    return submitted


def _complete_probe(name="m"):
    csvs = {k: csv_text(name, f"/x/{name}/{CKPT[k]}", ALL_CELLS) for k in config.CHECKPOINT_KINDS}
    return {name: {"exists": True, "files": {v: BIG for v in CKPT.values()}, "csvs": csvs}}


def _empty_probe(name="m"):
    return {name: {"exists": True, "files": {v: BIG for v in CKPT.values()}, "csvs": {}}}


def test_complete_model_produces_no_jobs(monkeypatch, capsys):
    submitted = _install_fakes(monkeypatch, _complete_probe(), sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check"]) == 0

    assert submitted == []
    assert "complete (nothing to do, nothing staged): 1" in capsys.readouterr().out


def test_incomplete_model_submits_one_job_per_kind(monkeypatch):
    submitted = _install_fakes(monkeypatch, _empty_probe(), sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check"]) == 0

    assert submitted == [("m", "best"), ("m", "last"), ("m", "advbest")]


def test_jobs_already_in_flight_are_not_resubmitted(monkeypatch):
    """What makes a *daily* cron safe: a 30h job started yesterday is left alone."""
    live = {config.job_name("m", "best"), config.job_name("m", "last")}
    submitted = _install_fakes(monkeypatch, _empty_probe(), live_names=live, sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check"]) == 0

    assert submitted == [("m", "advbest")]


def test_dry_run_submits_nothing(monkeypatch, capsys):
    submitted = _install_fakes(monkeypatch, _empty_probe(), sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check", "--dry-run"]) == 0

    assert submitted == []
    out = capsys.readouterr().out
    assert "DRY-RUN would submit aaswp_m_best" in out
    assert "would submit 3 jobs" in out


def test_limit_caps_submissions(monkeypatch):
    submitted = _install_fakes(monkeypatch, _empty_probe(), sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check", "--limit", "2"]) == 0

    assert len(submitted) == 2


def test_missing_mount_aborts_before_touching_the_cluster(monkeypatch, capsys):
    monkeypatch.setattr(submit_mod, "check_mounts", lambda: ["aircc mount is not mounted at /x"])
    monkeypatch.setattr(submit_mod, "notify", lambda subject, body: None)

    def explode(*a, **k):
        raise AssertionError("must not reach the cluster when a mount is down")

    monkeypatch.setattr(plan_mod, "finished_models", explode)

    assert submit_mod.main([]) == 1
    assert "ABORT" in capsys.readouterr().out


def test_submit_failure_is_reported_and_exits_nonzero(monkeypatch):
    _install_fakes(monkeypatch, _empty_probe(), sjm_finished=["m"])
    notified = []
    monkeypatch.setattr(submit_mod, "notify", lambda subject, body: notified.append(subject))

    def failing_submit(*a, **k):
        raise RuntimeError("sbatch: Invalid partition")

    monkeypatch.setattr(submit_mod, "submit_job", failing_submit)

    assert submit_mod.main(["--skip-mount-check"]) == 1
    assert notified and "failure" in notified[0]


def test_submit_job_builds_the_expected_sbatch_command():
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="998877;cluster\n", stderr="")

    job_id = submit_mod.submit_job("/models/m", "m", "advbest", run=fake_run)

    assert job_id == "998877"
    remote = captured["cmd"][-1]
    assert f"cd {config.SLURM_REPO} && sbatch --parsable" in remote
    assert "--job-name=aaswp_m_advbest" in remote
    assert "AA_MODEL_DIR=/models/m,AA_CHECKPOINT_KIND=advbest" in remote
    assert remote.endswith(config.SBATCH_SCRIPT)
