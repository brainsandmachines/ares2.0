import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aa_sweep import cluster_queue as q, config  # noqa: E402
from aa_sweep.tests.test_census import csv_text  # noqa: E402

ALL_CELLS = [(n, e) for n in config.NORMS for e in config.EPS_INPUTS]


def _db(tmp_path):
    return q.QueueDB(str(tmp_path / "q.sqlite"))


def _unit(name="m", kind="best", missing=15, model_dir="/models/m"):
    return {"model_name": name, "kind": kind, "model_dir": model_dir, "missing": missing}


def _model_dir(tmp_path, cells=()):
    d = tmp_path / "m"
    d.mkdir(exist_ok=True)
    (d / "model_best.pth.tar").write_bytes(b"x")
    ckpt = str(d / "model_best.pth.tar")
    (d / config.CSV_FOR_KIND["best"]).write_text(csv_text("m", ckpt, list(cells)) if cells else "")
    return d


def test_feed_is_idempotent(tmp_path):
    db = _db(tmp_path)
    assert db.feed([_unit()])["inserted"] == 1
    assert db.feed([_unit()])["inserted"] == 0
    assert db.counts() == {"pending": 1}


def test_complete_units_are_never_inserted_and_close_a_pending_row(tmp_path):
    db = _db(tmp_path)
    assert db.feed([_unit(missing=0)])["inserted"] == 0
    db.feed([_unit()])
    assert db.feed([_unit(missing=0)])["closed"] == 1
    assert db.counts() == {"finished": 1}


def test_a_running_row_is_left_to_its_task(tmp_path):
    db = _db(tmp_path)
    db.feed([_unit()])
    db.claim(111)
    assert db.feed([_unit(missing=0)])["untouched"] == 1
    assert db.counts() == {"running": 1}


def test_claim_order_and_no_double_claim(tmp_path):
    db = _db(tmp_path)
    db.feed([_unit("a", missing=3), _unit("b", missing=15)])
    first = db.claim(111, "main")
    assert first["model_name"] == "b" and first["partition"] == "main"  # fullest first
    assert db.claim(222)["model_name"] == "a"
    assert db.claim(333) is None


def test_a_restarted_job_id_gives_up_its_old_claim(tmp_path):
    db = _db(tmp_path)
    db.feed([_unit("a"), _unit("b")])
    old = db.claim(111)
    db.claim(111)
    # It may well take the same (top) unit again -- what matters is it never holds two.
    assert len(db.rows("slurm_job_id=111")) == 1
    assert db.get(old["id"])["requeued"] == 1


def test_finish_is_decided_by_the_csv_not_the_exit_code(tmp_path):
    db = _db(tmp_path)
    d = _model_dir(tmp_path, ALL_CELLS)
    db.feed([_unit(model_dir=str(d))])
    unit = db.claim(111)
    assert db.finish(unit["id"], 111, rc=1) == "finished"


def test_finish_with_cells_missing_retries_then_parks(tmp_path):
    db = _db(tmp_path)
    d = _model_dir(tmp_path, ALL_CELLS[:5])
    db.feed([_unit(model_dir=str(d))])
    for attempt in range(1, config.QUEUE_MAX_ATTEMPTS + 1):
        unit = db.claim(100 + attempt)
        status = db.finish(unit["id"], 100 + attempt, rc=0)
    assert status == "failed"
    row = db.get(unit["id"])
    assert row["attempts"] == config.QUEUE_MAX_ATTEMPTS and row["missing"] == 10
    assert db.claim(999) is None
    assert db.reset(unit["id"]) == 1 and db.claim(999) is not None


def test_finish_by_a_task_that_lost_the_claim_is_a_noop(tmp_path):
    db = _db(tmp_path)
    db.feed([_unit()])
    unit = db.claim(111)
    assert db.finish(unit["id"], 222, rc=0, missing=0) is None
    assert db.get(unit["id"])["status"] == "running"


def test_release_is_owner_guarded_and_costs_no_attempt(tmp_path):
    db = _db(tmp_path)
    db.feed([_unit()])
    unit = db.claim(111)
    assert db.release(unit["id"], owner=222) == 0
    assert db.release(unit["id"], owner=111) == 1
    row = db.get(unit["id"])
    assert row["status"] == "pending" and row["attempts"] == 0 and row["requeued"] == 1


def test_requeue_dead_respects_the_grace_period(tmp_path):
    db = _db(tmp_path)
    db.feed([_unit()])
    unit = db.claim(111)
    assert db.requeue_dead(alive=lambda j: False, now=unit["claimed_ts"] + 10) == []
    assert db.requeue_dead(alive=lambda j: True, now=unit["claimed_ts"] + 10_000) == []
    assert db.requeue_dead(alive=lambda j: False, now=unit["claimed_ts"] + 10_000) == [unit["id"]]


def test_owner_alive_fails_open_when_slurm_does_not_answer():
    def broken(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 60)

    assert q.owner_alive(1, run=broken) is True


def test_owner_alive_false_once_slurm_forgot_it():
    def gone(argv, **kw):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Invalid job id specified")
        return subprocess.CompletedProcess(argv, 0, stdout="CANCELLED by 1\n", stderr="")

    assert q.owner_alive(1, run=gone) is False


class _FakeSlurm:
    def __init__(self, pending_names=()):
        self.pending_names = set(pending_names)
        self.sbatched = []

    def __call__(self, argv, **kw):
        if argv[0] == "squeue":
            name = argv[argv.index("-n") + 1]
            out = "123_[9-200]\n" if name in self.pending_names else ""
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        if argv[0] == "sbatch":
            self.sbatched.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="555\n", stderr="")
        raise AssertionError(argv)


def test_launch_nothing_when_nothing_is_pending():
    slurm = _FakeSlurm()
    q.launch(0, run=slurm)
    assert slurm.sbatched == []


def test_launch_both_arrays_as_singletons():
    slurm = _FakeSlurm()
    messages, errors = q.launch(5, run=slurm)
    assert errors == []
    assert [a[-1] for a in slurm.sbatched] == [s for s, _ in config.QUEUE_ARRAYS.values()]
    assert all("--dependency=singleton" in a for a in slurm.sbatched)


def test_launch_skips_a_lane_whose_array_still_has_pending_tasks():
    slurm = _FakeSlurm(pending_names={"aaq-main"})
    q.launch(5, run=slurm)
    assert [a[-1] for a in slurm.sbatched] == [config.QUEUE_ARRAYS["rtx6000"][0]]


def test_launch_dry_run_submits_nothing():
    slurm = _FakeSlurm()
    messages, _ = q.launch(5, dry_run=True, run=slurm)
    assert slurm.sbatched == [] and all("DRY-RUN" in m for m in messages)


def test_claim_stdout_is_only_the_unit_line_even_when_it_releases_dead_owners(tmp_path, monkeypatch, capsys):
    """aa_queue_task.sh reads the claim's stdout as `<id>\\t<kind>\\t<dir>`; a release note there
    once became the unit id, crashed the engine and left the unit stuck `running`."""
    db_path = str(tmp_path / "q.sqlite")
    q.QueueDB(db_path).feed([_unit()])
    monkeypatch.setattr(q.QueueDB, "requeue_dead", lambda self, *a, **k: [6])
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "main")
    assert q.main(["--db", db_path, "claim"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == ["1\tbest\t/models/m"]
    assert "released dead-owner unit(s) [6]" in err
