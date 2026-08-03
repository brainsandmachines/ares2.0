import csv
import io
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aa_sweep import config, plan as plan_mod, stage as stage_mod  # noqa: E402
from aa_sweep.tests.test_census import csv_text  # noqa: E402


def work_for(tmp_path, staging_files, aircc=True):
    work = plan_mod.ModelWork(
        model_name="m",
        sources={"aircc"} if aircc else {"sjm"},
        slurm_dir=f"{config.SLURM_MODELS_ROOT}/m",
        aircc_dir=(tmp_path / "m") if aircc else None,
    )
    for kind in config.CHECKPOINT_KINDS:
        work.kinds[kind] = plan_mod.KindStatus(
            kind=kind,
            ckpt_on_aircc=config.CKPT_FILE_FOR_KIND[kind] in staging_files,
            missing={("l2", 1.0)} if config.CKPT_FILE_FOR_KIND[kind] in staging_files else set(),
        )
    return work


class Recorder:
    def __init__(self, returncode=0):
        self.calls = []
        self.returncode = returncode

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr="boom")


# --- rsync command shape ---------------------------------------------------------------------

def test_rsync_never_overwrites_and_carries_only_the_named_checkpoints(tmp_path):
    cmd = stage_mod.rsync_command("m", tmp_path / "m", ["model_best_adv.pth.tar"])

    assert "--ignore-existing" in cmd
    assert cmd[-1] == f"{config.SLURM_SSH_HOST}:{config.SLURM_MODELS_ROOT}/m/"
    includes = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--include"]
    assert "model_best_adv.pth.tar" in includes
    assert "last.pth.tar" not in includes
    assert "model_best.pth.tar" not in includes
    assert "autoattack_sweep_results*.csv" in includes
    # The catch-all exclude is what keeps the 1.4GB rolling checkpoints out.
    assert cmd[cmd.index("--exclude") + 1] == "*"


def test_rsync_carries_no_checkpoint_when_all_kinds_are_already_on_the_cluster(tmp_path):
    includes = stage_mod.rsync_command("m", tmp_path / "m", [])
    assert not [tok for tok in includes if tok.endswith(".pth.tar")]


# --- CSV merge -------------------------------------------------------------------------------

def test_merge_adds_aircc_only_rows_and_keeps_the_cluster_row_on_a_tie():
    slurm = csv_text("m", "/x/m/last.pth.tar", [("l2", 1.0), ("l2", 2.0)])
    aircc = csv_text("m", "results/models/m/last.pth.tar", [("l2", 2.0), ("l1", 8.0)])

    merged = stage_mod.merge_csv_rows(aircc, slurm)

    rows = list(csv.DictReader(io.StringIO(merged)))
    cells = {(r["attack_norm"], float(r["epsilon_input"])) for r in rows}
    assert cells == {("l2", 1.0), ("l2", 2.0), ("l1", 8.0)}
    assert len(rows) == 3  # the duplicated (l2, 2.0) was not appended twice
    kept = [r for r in rows if (r["attack_norm"], float(r["epsilon_input"])) == ("l2", 2.0)]
    assert len(kept) == 1 and kept[0]["checkpoint_path"] == "/x/m/last.pth.tar"


def test_merge_returns_none_when_there_is_nothing_to_add():
    same = csv_text("m", "/x/m/last.pth.tar", [("l2", 1.0)])
    assert stage_mod.merge_csv_rows(same, same) is None
    # Nothing on one side means rsync already handles it -- no merge needed.
    assert stage_mod.merge_csv_rows(same, "") is None
    assert stage_mod.merge_csv_rows("", same) is None


# --- stage_model -----------------------------------------------------------------------------

def test_dry_run_reports_sizes_without_running_anything(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "model_best_adv.pth.tar").write_bytes(b"x" * 2048)
    work = work_for(tmp_path, ["model_best_adv.pth.tar"])
    recorder = Recorder()

    res = stage_mod.stage_model(work, {}, {}, run=recorder, dry_run=True)

    assert recorder.calls == []
    assert res.files == ["model_best_adv.pth.tar"]
    assert res.bytes_planned == 2048
    assert res.ok


def test_stage_runs_rsync_then_pushes_merged_csvs(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "last.pth.tar").write_bytes(b"x")
    work = work_for(tmp_path, ["last.pth.tar"])
    aircc_csvs = {"last": csv_text("m", "results/models/m/last.pth.tar", [("l1", 8.0)])}
    slurm_csvs = {"last": csv_text("m", "/x/m/last.pth.tar", [("l2", 1.0)])}
    recorder = Recorder()

    res = stage_mod.stage_model(work, aircc_csvs, slurm_csvs, run=recorder)

    assert res.ok
    assert res.merged_csvs == ["last"]
    assert recorder.calls[0][0][0] == "rsync"
    ssh_cmd = recorder.calls[1][0]
    assert ssh_cmd[0] == "ssh" and ssh_cmd[-1].endswith("autoattack_sweep_results_last.csv")
    assert "l1" in recorder.calls[1][1]["input"]


def test_rsync_failure_is_reported_and_stops_the_model(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "last.pth.tar").write_bytes(b"x")
    work = work_for(tmp_path, ["last.pth.tar"])

    res = stage_mod.stage_model(work, {}, {}, run=Recorder(returncode=12))

    assert not res.ok
    assert "rc=12" in res.error


def test_vanished_file_return_codes_are_forgiven(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "last.pth.tar").write_bytes(b"x")
    work = work_for(tmp_path, ["last.pth.tar"])

    for rc in (23, 24):
        assert stage_mod.stage_model(work, {}, {}, run=Recorder(returncode=rc)).ok


def test_sjm_only_model_is_never_staged(tmp_path):
    work = work_for(tmp_path, [], aircc=False)
    recorder = Recorder()

    res = stage_mod.stage_model(work, {}, {}, run=recorder)

    assert recorder.calls == []
    assert res.files == [] and res.ok
