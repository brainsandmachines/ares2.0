import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aa_sweep import botero, botero_runner, config  # noqa: E402
from aa_sweep.census import KindStatus  # noqa: E402
from aa_sweep.plan import ModelWork  # noqa: E402

CKPT = config.CKPT_FILE_FOR_KIND
CELL = ("linf", 1.0)


@pytest.fixture
def conn(tmp_path):
    connection = botero.connect(tmp_path / "queue.sqlite")
    yield connection
    connection.close()


@pytest.fixture(autouse=True)
def local_archive(tmp_path, monkeypatch):
    """Resolve models out of tmp_path instead of the real /mnt archives, everywhere in this file."""
    real = botero.resolve_model_dir
    monkeypatch.setattr(
        botero, "resolve_model_dir",
        lambda name, ckpt, roots=None: real(name, ckpt, roots or (tmp_path / "a", tmp_path / "b")),
    )


def work(name="m", missing_kinds=("best", "last", "advbest"), missing=3):
    """A ModelWork whose named kinds each still need `missing` cells."""
    kinds = {
        k: KindStatus(kind=k, ckpt_on_slurm=True,
                      missing={("linf", float(i)) for i in range(missing)} if k in missing_kinds else set())
        for k in config.CHECKPOINT_KINDS
    }
    return ModelWork(model_name=name, kinds=kinds)


def archive(tmp_path, model_name, kinds=config.CHECKPOINT_KINDS, root="a"):
    """A fake archive root holding one model's checkpoints and its selection json."""
    model_dir = tmp_path / root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / config.SELECTION_JSON).write_text("{}")
    for kind in kinds:
        (model_dir / CKPT[kind]).write_bytes(b"x")
    return model_dir


# --- resolve_model_dir -------------------------------------------------------------------------


def test_first_root_holding_both_the_checkpoint_and_the_selection_wins(tmp_path):
    second = archive(tmp_path, "m", root="b")
    roots = (tmp_path / "a", tmp_path / "b")

    assert botero.resolve_model_dir("m", CKPT["best"], roots) == second

    first = archive(tmp_path, "m", root="a")
    assert botero.resolve_model_dir("m", CKPT["best"], roots) == first


def test_a_dir_without_the_selection_json_is_not_usable(tmp_path):
    """Without it a local run would attack a different 1024 images than the cluster rows did."""
    model_dir = archive(tmp_path, "m")
    (model_dir / config.SELECTION_JSON).unlink()

    assert botero.resolve_model_dir("m", CKPT["best"], (tmp_path / "a",)) is None


def test_the_kind_actually_asked_for_must_be_present(tmp_path):
    archive(tmp_path, "m", kinds=("best",))
    roots = (tmp_path / "a",)

    assert botero.resolve_model_dir("m", CKPT["best"], roots) is not None
    assert botero.resolve_model_dir("m", CKPT["advbest"], roots) is None


def test_nested_sjm_names_resolve_as_subdirectories(tmp_path):
    nested = archive(tmp_path, "vit_b_cvst/linftrades_1_init1")

    assert botero.resolve_model_dir("vit_b_cvst/linftrades_1_init1", CKPT["last"], (tmp_path / "a",)) == nested


# --- queue -------------------------------------------------------------------------------------


def test_a_unit_cannot_be_queued_twice_while_active(conn, tmp_path):
    assert botero.enqueue(conn, "m", "best", tmp_path) is not None
    assert botero.enqueue(conn, "m", "best", tmp_path) is None
    assert botero.active_count(conn) == 1


def test_a_finished_unit_can_be_queued_again(conn, tmp_path):
    """New grid cells appear over time; a completed row must not block the next pass forever."""
    job_id = botero.enqueue(conn, "m", "best", tmp_path)
    botero.finish(conn, job_id, ok=True)

    assert botero.enqueue(conn, "m", "best", tmp_path) is not None


def test_claim_takes_the_oldest_queued_row_and_only_once(conn, tmp_path):
    botero.enqueue(conn, "m", "best", tmp_path)
    botero.enqueue(conn, "m", "last", tmp_path)

    first = botero.claim(conn, pid=999)
    assert (first.model_name, first.checkpoint_kind, first.status) == ("m", "best", "running")
    assert botero.claim(conn, pid=999).checkpoint_kind == "last"
    assert botero.claim(conn, pid=999) is None


def test_active_job_names_use_the_cluster_naming_scheme(conn, tmp_path):
    """This is what stops submit.py sending a unit to Slurm that this machine already owns."""
    botero.enqueue(conn, "vit_b_cvst/linftrades_1_init1", "advbest", tmp_path)

    assert botero.active_job_names(conn) == {"aaswp_vit_b_cvst__linftrades_1_init1_advbest"}


def test_a_dead_runner_puts_its_job_back_in_the_queue(conn, tmp_path):
    botero.enqueue(conn, "m", "best", tmp_path)
    botero.claim(conn, pid=999)

    notes = botero.reap_stale(conn, alive=lambda pid: False)

    assert len(notes) == 1 and "requeued" in notes[0]
    assert botero.claim(conn, pid=1000).model_name == "m"


def test_a_live_runner_is_left_alone(conn, tmp_path):
    botero.enqueue(conn, "m", "best", tmp_path)
    botero.claim(conn, pid=999)

    assert botero.reap_stale(conn, alive=lambda pid: True) == []


def test_repeated_deaths_eventually_fail_the_job(conn, tmp_path):
    botero.enqueue(conn, "m", "best", tmp_path)
    for _ in range(config.BOTERO_MAX_ATTEMPTS):
        botero.claim(conn, pid=999)
        botero.reap_stale(conn, alive=lambda pid: False)

    assert botero.claim(conn, pid=999) is None
    row = conn.execute("SELECT status FROM botero_jobs").fetchone()
    assert row["status"] == "failed"


# --- top-up ------------------------------------------------------------------------------------


def topup(conn, works, pending=(), slots=5, dry_run=False, cancelled=None, cancel=None):
    """Run a top-up with the cluster stubbed out; `local_archive` supplies the model dirs."""
    if cancel is None:
        cancel = cancelled.append if cancelled is not None else (lambda job_id: None)
    return botero.topup(
        works, conn=conn, slots=slots, dry_run=dry_run,
        pending=(pending if callable(pending) else (lambda: list(pending))),
        cancel=cancel,
        log=lambda msg: None,
    )


def test_topup_fills_exactly_the_free_slots(conn, tmp_path):
    archive(tmp_path, "m")
    pending = [(300, "m", "advbest"), (200, "m", "last"), (100, "m", "best")]

    taken = topup(conn, [work()], pending=pending, slots=2)

    assert len(taken) == 2
    assert botero.active_count(conn) == 2


def test_topup_is_a_noop_when_the_queue_is_full(conn, tmp_path):
    archive(tmp_path, "m")
    for kind in config.CHECKPOINT_KINDS:
        botero.enqueue(conn, "m", kind, tmp_path)
    cancelled = []

    taken = topup(conn, [work()], pending=[(1, "m", "best")], slots=3, cancelled=cancelled)

    assert taken == [] and cancelled == []


def test_topup_takes_from_the_back_of_the_slurm_queue_first(conn, tmp_path):
    """Highest job id = what Slurm would start last = what is worth moving here."""
    archive(tmp_path, "m")
    pending = [(300, "m", "advbest"), (200, "m", "last"), (100, "m", "best")]
    cancelled = []

    taken = topup(conn, [work()], pending=pending, slots=1, cancelled=cancelled)

    assert taken == ["m:advbest"]
    assert cancelled == [300]


def test_a_moved_unit_is_cancelled_on_slurm_before_it_is_queued_here(conn, tmp_path):
    """Order matters: a crash between the two must never leave a unit queued in both lanes."""
    archive(tmp_path, "m")
    queued_when_cancelled = []

    def cancel(job_id):
        queued_when_cancelled.append(botero.active_count(conn))

    botero.topup([work()], conn=conn, slots=1, pending=lambda: [(7, "m", "best")],
                 cancel=cancel, log=lambda msg: None)

    assert queued_when_cancelled == [0]
    assert botero.active_count(conn) == 1


def test_a_failed_scancel_leaves_the_unit_on_slurm(conn, tmp_path):
    """And source B must not then re-enqueue it: that would run the unit in both lanes at once."""
    archive(tmp_path, "m")

    def cancel(job_id):
        raise RuntimeError("slurm unreachable")

    taken = topup(conn, [work(missing_kinds=("best",))], pending=[(7, "m", "best")], cancel=cancel)

    assert taken == []
    assert botero.active_count(conn) == 0


def test_pending_slurm_units_past_the_cut_are_not_enqueued_as_new_work(conn, tmp_path):
    """Source B fills the remaining slots, but never with something the cluster still holds."""
    archive(tmp_path, "m")
    cancelled = []

    taken = topup(conn, [work()], pending=[(7, "m", "best"), (8, "m", "last")],
                  slots=5, cancelled=cancelled)

    assert sorted(taken) == ["m:advbest", "m:best", "m:last"]
    assert sorted(cancelled) == [7, 8]
    # m:advbest is the only one source B was allowed to add.
    origins = dict(conn.execute("SELECT checkpoint_kind, origin FROM botero_jobs").fetchall())
    assert origins == {"best": "moved:7", "last": "moved:8", "advbest": "new"}


def test_units_with_no_missing_cells_are_never_moved(conn, tmp_path):
    archive(tmp_path, "m")
    cancelled = []

    taken = topup(conn, [work(missing_kinds=("best",))], pending=[(1, "m", "last")], cancelled=cancelled)

    assert "m:last" not in taken and 1 not in cancelled


def test_a_model_missing_from_the_local_archive_is_skipped(conn, tmp_path):
    (tmp_path / "a").mkdir()
    cancelled = []

    taken = topup(conn, [work()], pending=[(1, "m", "best")], cancelled=cancelled)

    assert taken == [] and cancelled == []


def test_new_work_is_enqueued_only_once_slurm_has_nothing_left_to_move(conn, tmp_path):
    archive(tmp_path, "m")

    taken = topup(conn, [work()], pending=[], slots=2)

    assert taken == ["m:best", "m:last"]
    assert conn.execute("SELECT origin FROM botero_jobs LIMIT 1").fetchone()["origin"] == "new"


def test_dry_run_cancels_nothing_and_queues_nothing(conn, tmp_path):
    archive(tmp_path, "m")
    cancelled = []

    taken = topup(conn, [work()], pending=[(1, "m", "best")], slots=1, dry_run=True,
                  cancelled=cancelled)

    assert taken == ["m:best"]
    assert cancelled == [] and botero.active_count(conn) == 0


def test_an_unreadable_slurm_queue_falls_back_to_new_work(conn, tmp_path):
    archive(tmp_path, "m")

    def explode():
        raise RuntimeError("ssh down")

    taken = topup(conn, [work()], pending=explode, slots=1)

    assert taken == ["m:best"]


# --- the GPU gate ------------------------------------------------------------------------------


def _nvidia_smi(stdout, rc=0):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
    return run


def test_an_idle_gpu_has_no_blockers():
    assert botero_runner.gpu_blockers({1}, run=_nvidia_smi("")) == []


def test_a_foreign_cuda_process_blocks_the_tick():
    """The ad-hoc epoch-90 eval, a notebook, a training run -- all of them defer the lane."""
    blockers = botero_runner.gpu_blockers({1}, run=_nvidia_smi("110114, 6074\n"))

    assert blockers == [(110114, "6074")]


def test_our_own_processes_do_not_block_us():
    assert botero_runner.gpu_blockers({110114}, run=_nvidia_smi("110114, 6074\n")) == []


def test_a_broken_nvidia_smi_is_an_error_not_an_all_clear():
    """Failing open would put a second AutoAttack run on a card we cannot see."""
    with pytest.raises(RuntimeError):
        botero_runner.gpu_blockers({1}, run=_nvidia_smi("", rc=9))


def test_the_engine_command_matches_the_cluster_sbatch():
    job = botero.Job(id=1, model_name="m", checkpoint_kind="advbest", status="running",
                     model_dir="/archive/m")
    cmd = botero_runner.engine_command(job)

    assert "--force" not in cmd  # never recompute a cell that already exists
    assert cmd[cmd.index("--model-dir") + 1] == "/archive/m"
    assert cmd[cmd.index("--checkpoint-kinds") + 1] == "advbest"
    assert cmd[cmd.index("--eps-inputs") + 1] == "1,2,4,6,8"
    assert cmd[cmd.index("--norms") + 1] == "linf,l2,l1"
    # Whatever the batch size, the image count is the same 1024 as every cluster row.
    assert (int(cmd[cmd.index("--batch-size") + 1]) * int(cmd[cmd.index("--num-batches") + 1])
            == config.BOTERO_TOTAL_IMAGES)


def test_a_batch_size_that_changes_the_image_count_is_refused(monkeypatch):
    """Batching may regroup the 1024 images; it must never change how many there are."""
    monkeypatch.setattr(config, "BOTERO_BATCH_SIZE", 128)
    monkeypatch.setattr(config, "BOTERO_NUM_BATCHES", 32)  # 4096, not 1024
    job = botero.Job(id=1, model_name="m", checkpoint_kind="best", status="running", model_dir="/d")

    with pytest.raises(ValueError, match="must be over exactly"):
        botero_runner.engine_command(job)


@pytest.mark.parametrize("bsz,nb", [(32, 32), (64, 16), (128, 8)])
def test_any_batching_of_1024_is_accepted(monkeypatch, bsz, nb):
    monkeypatch.setattr(config, "BOTERO_BATCH_SIZE", bsz)
    monkeypatch.setattr(config, "BOTERO_NUM_BATCHES", nb)
    job = botero.Job(id=1, model_name="m", checkpoint_kind="best", status="running", model_dir="/d")

    cmd = botero_runner.engine_command(job)

    assert cmd[cmd.index("--batch-size") + 1] == str(bsz)


def test_a_busy_gpu_leaves_the_queue_untouched(conn, tmp_path, monkeypatch):
    botero.enqueue(conn, "m", "best", tmp_path)
    monkeypatch.setattr(botero_runner, "gpu_blockers", lambda pids, **kw: [(110114, "6074")])

    assert botero_runner.tick(conn=conn) == 0
    assert conn.execute("SELECT status FROM botero_jobs").fetchone()["status"] == "queued"


# --- job-name round trip -----------------------------------------------------------------------


@pytest.mark.parametrize("model,kind", [
    ("convnext_base_v1_l2_2_init1", "best"),
    ("vit_b_cvst/linftrades_1_init1", "advbest"),
    ("swin_b/l2_cont4to6_init1", "last"),
])
def test_job_names_round_trip(model, kind):
    assert config.parse_job_name(config.job_name(model, kind)) == (model, kind)


@pytest.mark.parametrize("name", ["sjm-manager", "aaswp_m_bogus", "manual_aa_eval_m_linf", "aaswp_best"])
def test_foreign_job_names_are_not_decoded(name):
    assert config.parse_job_name(name) is None


# --- parking ------------------------------------------------------------------------------------


def test_a_parked_unit_is_not_claimable(conn, tmp_path):
    """What lets a long-lived runner drain and exit instead of claiming with stale settings."""
    botero.enqueue(conn, "m", "best", tmp_path)
    conn.execute("UPDATE botero_jobs SET status='parked'")

    assert botero.claim(conn, pid=999) is None


def test_a_parked_unit_still_holds_its_slot_and_stays_off_slurm(conn, tmp_path):
    """Parking must never quietly hand the unit back to the cluster or free a slot for a new one."""
    botero.enqueue(conn, "m", "best", tmp_path)
    conn.execute("UPDATE botero_jobs SET status='parked'")

    assert botero.active_count(conn) == 1
    assert botero.active_job_names(conn) == {config.job_name("m", "best")}


def test_unpark_returns_a_unit_to_the_queue(conn, tmp_path):
    botero.enqueue(conn, "m", "best", tmp_path)
    conn.execute("UPDATE botero_jobs SET status='parked'")

    botero.main_unpark = None  # guard against accidental import-time reliance
    conn.execute("UPDATE botero_jobs SET status='queued' WHERE status='parked'")

    assert botero.claim(conn, pid=999).model_name == "m"


def test_topup_does_not_refill_over_a_parked_unit(conn, tmp_path):
    archive(tmp_path, "m")
    botero.enqueue(conn, "m", "best", tmp_path)
    conn.execute("UPDATE botero_jobs SET status='parked'")
    cancelled = []

    taken = topup(conn, [work()], pending=[(9, "m", "last")], slots=1, cancelled=cancelled)

    assert taken == [] and cancelled == []
