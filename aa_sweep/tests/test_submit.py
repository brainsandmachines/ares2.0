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


class _NoBotero:
    """Stands in for aa_sweep.botero inside plan.build_plan: no local archive, no filesystem."""

    @staticmethod
    def resolve_model_dir(*_args, **_kwargs):
        return None


_NO_BOTERO = _NoBotero()


def _install_fakes(monkeypatch, probe, live_names=(), aircc_finished=(), sjm_finished=()):
    """Point submit.main() at in-memory fixtures instead of the mounts and the cluster."""
    submitted = []

    monkeypatch.setattr(submit_mod, "check_paths", lambda: [])
    monkeypatch.setattr(
        plan_mod, "finished_models",
        lambda db: list(aircc_finished) if db == config.AIRCC_DB else list(sjm_finished),
    )
    monkeypatch.setattr(plan_mod, "probe_slurm", lambda names: probe)
    monkeypatch.setattr(plan_mod, "_read_local_dir", lambda d: ({}, {}))
    monkeypatch.setattr(plan_mod, "botero", _NO_BOTERO)
    monkeypatch.setattr(submit_mod, "live_job_names", lambda: set(live_names))
    # The Botero lane has its own suite; here it must not reach ssh or the real local queue.
    monkeypatch.setattr(submit_mod.botero_mod, "topup", lambda works, **kw: [])

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
    assert "slurm lane: 1 complete" in capsys.readouterr().out


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


def test_a_pending_job_blocks_resubmission_just_like_a_running_one(monkeypatch):
    """squeue's default states include PENDING; a job queued for days must not be duplicated."""
    live = {config.job_name("m", "best")}
    submitted = _install_fakes(monkeypatch, _empty_probe(), live_names=live, sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check"]) == 0

    assert ("m", "best") not in submitted


def test_a_foreign_job_mentioning_the_model_blocks_every_kind(monkeypatch, capsys):
    """A hand-launched eval on the same model could touch any of its CSVs."""
    submitted = _install_fakes(
        monkeypatch, _empty_probe(), live_names={"manual_aa_eval_m_linf"}, sjm_finished=["m"]
    )

    assert submit_mod.main(["--skip-mount-check"]) == 0

    assert submitted == []
    assert "manual_aa_eval_m_linf' is already queued/running" in capsys.readouterr().out


def test_our_own_other_kinds_do_not_block_each_other(monkeypatch):
    """The three kinds write three different CSVs, so they are safe to run concurrently."""
    submitted = _install_fakes(
        monkeypatch, _empty_probe(), live_names={config.job_name("m", "best")}, sjm_finished=["m"]
    )

    assert submit_mod.main(["--skip-mount-check"]) == 0

    assert submitted == [("m", "last"), ("m", "advbest")]


def test_unrelated_jobs_are_ignored(monkeypatch):
    """`sjm-manager` contains the model name `m` as a substring -- it must not block."""
    submitted = _install_fakes(
        monkeypatch, _empty_probe(), live_names={"sjm-manager", "aaswp_other_model_best"},
        sjm_finished=["m"],
    )

    assert submit_mod.main(["--skip-mount-check"]) == 0

    assert len(submitted) == 3


def test_model_name_must_match_on_token_boundaries():
    conflicting = submit_mod.conflicting_job
    # substring but not a token -> not a conflict
    assert conflicting("m", "best", {"sjm-manager"}) is None
    assert conflicting("linf_1_init1", "best", {"aaswp_xlinf_1_init1_last"}) is None
    assert conflicting("l2_1_init1", "best", {"train_l2_1_init10_probe"}) is None
    # delimited occurrence -> conflict
    assert conflicting("linf_1_init1", "best", {"manual_linf_1_init1_eval"}) == "manual_linf_1_init1_eval"
    assert conflicting("linf_1_init1", "best", {"eval-linf_1_init1"}) == "eval-linf_1_init1"


def test_a_nested_model_is_not_blocked_by_a_flat_model_that_ends_the_same_way():
    """The starvation bug: swin_b/<x> reduces to <x>, which is a `_`-delimited suffix of a
    different, flat convnext_base_<x>. Those blockers sit PENDING behind QOSMaxGRESPerUser for
    weeks, so the "next night picks it up" escape hatch never fires and the model is skipped
    forever. Any aaswp_* name belongs to us and only the exact (model, kind) match may block."""
    flat = config.job_name("convnext_base_linf_cont4to6_init1", "advbest")
    assert submit_mod.conflicting_job("swin_b/linf_cont4to6_init1", "best", {flat}) is None
    assert submit_mod.conflicting_job("vit_b_cvst/linf_cont4to6_init1", "last", {flat}) is None
    # ... and the reverse direction, where our own dir-name token is the whole flat name.
    nested = config.job_name("swin_b/linf_cont4to6_init1", "best")
    assert submit_mod.conflicting_job("convnext_base_linf_cont4to6_init1", "advbest", {nested}) is None


def test_our_own_other_kinds_still_do_not_block():
    """Three kinds write three different CSVs, so they may run at once."""
    other = config.job_name("swin_b/linf_cont4to6_init1", "last")
    assert submit_mod.conflicting_job("swin_b/linf_cont4to6_init1", "best", {other}) is None


def test_the_exact_same_unit_still_blocks():
    mine = config.job_name("swin_b/linf_cont4to6_init1", "best")
    assert submit_mod.conflicting_job("swin_b/linf_cont4to6_init1", "best", {mine}) == mine


def test_a_nested_unit_already_RUNNING_under_its_renamed_form_still_blocks():
    """sbatches/aa_sweep_completion.sbatch renames a job to `aaswp_$(basename dir)_<kind>` the
    moment it starts, so a *running* nested unit loses its `vit_b_cvst__` segment in squeue.

    Matching only the submitted form would miss it and put a second job on the same CSV. Jobs
    20201886/20201887 sat in the real queue in exactly this state.
    """
    renamed = "aaswp_l2_cont4to6_init1_advbest"
    assert config.job_name("vit_b_cvst/l2_cont4to6_init1", "advbest") != renamed
    assert submit_mod.conflicting_job(
        "vit_b_cvst/l2_cont4to6_init1", "advbest", {renamed}
    ) == renamed


def test_the_renamed_form_does_not_block_an_unrelated_flat_model_of_that_name():
    """The other half of the same coin: `aaswp_l2_cont4to6_init1_advbest` is genuinely the flat
    model's own name too, so it must block that one -- and only for the same kind."""
    renamed = "aaswp_l2_cont4to6_init1_advbest"
    assert submit_mod.conflicting_job("l2_cont4to6_init1", "advbest", {renamed}) == renamed
    assert submit_mod.conflicting_job("l2_cont4to6_init1", "best", {renamed}) is None
    # A *differently* nested model reducing to the same basename is blocked as well. Two lanes
    # cannot both be right about who owns that CSV, and over-blocking is the safe direction.
    assert submit_mod.conflicting_job("swin_b/l2_cont4to6_init1", "advbest", {renamed}) == renamed


def test_a_foreign_job_touching_the_model_dir_still_blocks():
    """Not one of ours -> no way to know it will not write the same CSV. Fail closed."""
    assert submit_mod.conflicting_job(
        "swin_b/linf_cont4to6_init1", "best", {"manual_linf_cont4to6_init1_eval"}
    ) == "manual_linf_cont4to6_init1_eval"


def test_conflicting_job_matches_full_untruncated_names():
    """A %.40j-style truncated name must not be what we compare against."""
    full = config.job_name("convnext_base_linftrades_2_init0", "last")
    assert len(full) > 40
    assert submit_mod.conflicting_job("convnext_base_linftrades_2_init0", "last", {full}) == full
    # The truncated form is a substring of the model dir name check, so it still blocks -- but via
    # the foreign-job path, which is the conservative outcome we want rather than a silent miss.
    assert submit_mod.conflicting_job("convnext_base_linftrades_2_init0", "last", {full[:40]}) is not None


def test_squeue_failure_fails_closed(monkeypatch):
    """No queue view means no way to know what is running; skip the night rather than duplicate."""
    def failing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="slurm_load_jobs error")

    monkeypatch.setattr(submit_mod, "notify", lambda *a, **kw: None)
    try:
        submit_mod.live_job_names(run=failing)
    except RuntimeError as exc:
        assert "squeue failed" in str(exc)
    else:
        raise AssertionError("live_job_names must raise when squeue fails")


def test_dry_run_submits_nothing(monkeypatch, capsys):
    submitted = _install_fakes(monkeypatch, _empty_probe(), sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check", "--dry-run"]) == 0

    assert submitted == []
    out = capsys.readouterr().out
    assert "DRY-RUN would submit aaswp_m_best" in out
    assert "would submit 3 sbatch jobs" in out


def test_limit_caps_submissions(monkeypatch):
    submitted = _install_fakes(monkeypatch, _empty_probe(), sjm_finished=["m"])

    assert submit_mod.main(["--skip-mount-check", "--limit", "2"]) == 0

    assert len(submitted) == 2


def test_missing_mount_aborts_before_touching_the_cluster(monkeypatch, capsys):
    monkeypatch.setattr(submit_mod, "check_paths", lambda: ["qnap mount is not mounted at /x"])
    monkeypatch.setattr(submit_mod, "notify", lambda subject, body, **kw: None)

    def explode(*a, **k):
        raise AssertionError("must not reach the cluster when a mount is down")

    monkeypatch.setattr(plan_mod, "finished_models", explode)

    assert submit_mod.main([]) == 1
    assert "ABORT" in capsys.readouterr().out


def test_submit_failure_is_reported_and_exits_nonzero(monkeypatch):
    _install_fakes(monkeypatch, _empty_probe(), sjm_finished=["m"])
    notified = []
    monkeypatch.setattr(submit_mod, "notify", lambda subject, body, **kw: notified.append(subject))

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
