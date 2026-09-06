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
def local_store(tmp_path, monkeypatch):
    """Resolve models out of tmp_path instead of the real store, everywhere in this file."""
    monkeypatch.setattr(config, "BOTERO_STORE_ROOT", tmp_path / "models")


def work(name="m", missing_kinds=("best", "last", "advbest"), missing=3, lane=config.BOTERO_LANE):
    """A ModelWork whose named kinds each still need `missing` cells."""
    kinds = {
        k: KindStatus(kind=k, has_checkpoint=True,
                      missing={("linf", float(i)) for i in range(missing)} if k in missing_kinds else set())
        for k in config.CHECKPOINT_KINDS
    }
    return ModelWork(model_name=name, lane=lane, kinds=kinds)


def store(tmp_path, model_name, kinds=config.CHECKPOINT_KINDS, arch="convnext_base"):
    """One model in a fake curated store: <root>/<arch>/<name>/ with checkpoints + selection json.

    The store is keyed by directory basename, so a nested name is created under its basename --
    exactly as model_store lays the real tree out.
    """
    model_dir = tmp_path / "models" / arch / model_name.rsplit("/", 1)[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / config.SELECTION_JSON).write_text("{}")
    for kind in kinds:
        (model_dir / CKPT[kind]).write_bytes(b"x")
    return model_dir


# --- resolve_model_dir -------------------------------------------------------------------------


def test_a_model_is_found_under_its_architecture_dir(tmp_path):
    """Names are flat (AIRCC) or nested under a different prefix (sjm); the store nests them under
    an architecture dir. Indexing by basename is what makes all three resolve."""
    expected = store(tmp_path, "convnext_base_baseline_init0")

    assert botero.resolve_model_dir("convnext_base_baseline_init0", CKPT["best"]) == expected


def test_a_nested_name_resolves_by_its_basename(tmp_path):
    """`vit_b_cvst/l1_1_init1` lives at `<store>/vit_b_cvst/l1_1_init1` -- the same last segment,
    but the store's own arch dir, not the one carried in the name."""
    expected = store(tmp_path, "vit_b_cvst/l1_1_init1", arch="vit_b_cvst")

    assert botero.resolve_model_dir("vit_b_cvst/l1_1_init1", CKPT["last"]) == expected


def test_bookkeeping_dirs_of_the_store_are_not_models(tmp_path):
    store(tmp_path, "m", arch="_legacy")

    assert botero.resolve_model_dir("m", CKPT["best"]) is None


def test_a_dir_without_the_selection_json_is_not_usable(tmp_path):
    """Without it a local run would attack a different 1024 images than every other row did."""
    model_dir = store(tmp_path, "m")
    (model_dir / config.SELECTION_JSON).unlink()

    assert botero.resolve_model_dir("m", CKPT["best"]) is None


def test_the_kind_actually_asked_for_must_be_present(tmp_path):
    store(tmp_path, "m", kinds=("best",))

    assert botero.resolve_model_dir("m", CKPT["best"]) is not None
    assert botero.resolve_model_dir("m", CKPT["advbest"]) is None


def test_without_a_named_kind_any_checkpoint_qualifies_the_dir(tmp_path):
    """The census path: resolve the dir first, work out which kinds it needs afterwards."""
    expected = store(tmp_path, "m", kinds=("last",))

    assert botero.resolve_model_dir("m") == expected


def test_a_model_absent_from_the_store_resolves_to_nothing(tmp_path):
    store(tmp_path, "other")

    assert botero.resolve_model_dir("m") is None


def test_store_index_is_keyed_by_basename_and_skips_bookkeeping(tmp_path):
    store(tmp_path, "convnext_base_baseline_init0")
    store(tmp_path, "vit_b_cvst/l1_1_init1", arch="vit_b_cvst")
    store(tmp_path, "junk", arch="_meta")

    assert set(botero.store_index()) == {"convnext_base_baseline_init0", "l1_1_init1"}


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


def topup(conn, works, slots=5, dry_run=False):
    """Run a top-up; `local_store` supplies the model dirs. No cluster stubs needed any more --
    this lane no longer reads or cancels anything on Slurm."""
    return botero.topup(works, conn=conn, slots=slots, dry_run=dry_run, log=lambda msg: None)


def test_topup_fills_exactly_the_free_slots(conn, tmp_path):
    store(tmp_path, "m")

    taken = topup(conn, [work()], slots=2)

    assert len(taken) == 2
    assert botero.active_count(conn) == 2


def test_topup_is_a_noop_when_the_queue_is_full(conn, tmp_path):
    store(tmp_path, "m")
    for kind in config.CHECKPOINT_KINDS:
        botero.enqueue(conn, "m", kind, tmp_path)

    assert topup(conn, [work()], slots=3) == []


def test_topup_only_takes_botero_lane_models(conn, tmp_path):
    """The disjointness guarantee enforced at the point of use: a model the cluster owns is never
    enqueued here, even sitting in the same plan list."""
    store(tmp_path, "mine")
    store(tmp_path, "theirs")

    taken = topup(conn, [work("theirs", lane=config.SLURM_LANE), work("mine")], slots=5)

    assert {t.split(":")[0] for t in taken} == {"mine"}


def test_topup_prefers_the_emptiest_checkpoints(conn, tmp_path):
    """Fullest-first: finishing one checkpoint completely beats a cell each on three of them."""
    store(tmp_path, "few")
    store(tmp_path, "many")

    taken = topup(conn, [work("few", missing=2), work("many", missing=9)], slots=3)

    assert [t.split(":")[0] for t in taken] == ["many", "many", "many"]


def test_kinds_with_no_missing_cells_are_never_enqueued(conn, tmp_path):
    store(tmp_path, "m")

    taken = topup(conn, [work(missing_kinds=("best",))], slots=5)

    assert taken == ["m:best"]


def test_a_model_missing_from_the_local_store_is_skipped(conn, tmp_path):
    """No local copy, no local run: this lane never fetches anything."""
    taken = topup(conn, [work()], slots=5)

    assert taken == [] and botero.active_count(conn) == 0


def test_a_kind_whose_checkpoint_is_absent_locally_is_skipped(conn, tmp_path):
    store(tmp_path, "m", kinds=("best", "last"))

    taken = topup(conn, [work()], slots=5)

    assert sorted(taken) == ["m:best", "m:last"]


def test_an_already_active_unit_is_not_enqueued_twice(conn, tmp_path):
    store(tmp_path, "m")
    botero.enqueue(conn, "m", "best", tmp_path)

    taken = topup(conn, [work()], slots=5)

    assert "m:best" not in taken
    assert sorted(taken) == ["m:advbest", "m:last"]


def test_dry_run_queues_nothing(conn, tmp_path):
    store(tmp_path, "m")

    taken = topup(conn, [work()], slots=2, dry_run=True)

    assert len(taken) == 2
    assert botero.active_count(conn) == 0


def test_the_enqueued_row_points_at_the_local_store_dir(conn, tmp_path):
    """What the runner hands the engine as --model-dir, and so where the results land."""
    expected = store(tmp_path, "m")

    topup(conn, [work(missing_kinds=("best",))], slots=1)

    job = botero.claim(conn, pid=1)
    assert Path(job.model_dir) == expected


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
    """A parked unit still holds its slot, so a full-but-parked queue takes on nothing."""
    store(tmp_path, "m")
    botero.enqueue(conn, "m", "best", tmp_path)
    conn.execute("UPDATE botero_jobs SET status='parked'")

    assert topup(conn, [work()], slots=1) == []
