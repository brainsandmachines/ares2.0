"""AA results travel only with the checkpoint they were computed on.

Backfill decides each file on its own -- checkpoints on epoch, metadata on mtime -- so
without a guard a source dir's sweep CSVs reach a store dir holding a *different*
checkpoint of the same kind. That is what put a failed first attempt's 0.0-robust grids
beside the AIRCC rerun's checkpoints in five store dirs (aircc_copy_audit.md,
2026-10-07). These tests pin the guard: results whose source checkpoint differs from the
one the destination will hold are held back and reported, while a rerun's results still
arrive with its checkpoints, and results beside identical checkpoints still flow.

Checkpoint epochs are faked (first line ``epoch=N``) so no torch file is needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_store import backfill
from model_store.census import ModelRecord
from model_store.hashes import HashCache
from model_store.naming import ModelIdentity

NAME = "convnext_base_dvd_b_linf_2_init1"
OLD_MTIME, NEW_MTIME = 1_780_000_000, 1_790_000_000
CSVS = ("autoattack_sweep_results.csv", "autoattack_sweep_results_last.csv",
        "autoattack_sweep_results_advbest.csv")
RESULTS = set(CSVS) | {"autoattack_eps_norm_scores.json",
                       f"autoattack_eval_comparation_{NAME}.png"}

# basename -> (epoch, fill byte). The rerun is what the store holds; the failed attempt
# is what the stale cluster copy (and so the Slurm archive) held: lower or equal epochs.
RERUN = {"model_best.pth.tar": (199, b"n"), "last.pth.tar": (199, b"n"),
         "model_best_adv.pth.tar": (199, b"n")}
FAILED = {"model_best.pth.tar": (160, b"o"), "last.pth.tar": (199, b"o"),
          "model_best_adv.pth.tar": (151, b"o")}


def _fake_epoch(path):
    try:
        first = Path(path).read_bytes().split(b"\n", 1)[0]
    except OSError:
        return None
    return int(first.split(b"=")[1]) if first.startswith(b"epoch=") else None


@pytest.fixture(autouse=True)
def _epochs(monkeypatch):
    monkeypatch.setattr(backfill, "checkpoint_epoch", _fake_epoch)


def _write(path: Path, data: bytes, mtime: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (mtime, mtime))


def _ckpts(d: Path, spec: dict, mtime: int) -> None:
    for name, (epoch, fill) in spec.items():
        _write(d / name, b"epoch=%03d\n" % epoch + fill * 64, mtime)


def _results(d: Path, tag: bytes, mtime: int) -> None:
    for name in RESULTS:
        _write(d / name, tag + b" " + name.encode(), mtime)


def _setup(tmp_path, source="aircc", status="finished"):
    ident = ModelIdentity(canonical=NAME, arch="convnext_base", protocol="madry",
                          norm="linf", eps=2.0, init="1", source="csv")
    qdir = tmp_path / "qnap" / NAME
    store = tmp_path / "models"
    rec = ModelRecord(identity=ident, dirs={"qnap-slurm": qdir},
                      db_source=source, db_status=status)
    return {NAME: rec}, qdir, store / ident.store_relpath, store


def _plan(tmp_path, records, store):
    held: list = []
    items = backfill.plan(records, store, labels=("qnap-slurm",),
                          cache=HashCache(tmp_path / "sha256.jsonl"), skipped=held)
    return {it.rel: it.reason for it in items}, {it.rel: it for it in held}


def test_results_of_another_attempt_are_held_back(tmp_path):
    """The incident: store holds the rerun, the source holds the failed attempt and
    fuller, newer sweep CSVs. No checkpoint is pulled (no higher epoch, not an sjm
    rerun) -- and now none of the results computed on them is either."""
    records, qdir, sdir, store = _setup(tmp_path)
    _ckpts(qdir, FAILED, NEW_MTIME)
    _results(qdir, b"failed", NEW_MTIME)
    _ckpts(sdir, RERUN, OLD_MTIME)
    _results(sdir, b"rerun", OLD_MTIME)

    pulls, held = _plan(tmp_path, records, store)
    assert pulls == {}
    assert set(held) == RESULTS
    assert all(it.reason == backfill.SKIP_OTHER_CHECKPOINT for it in held.values())
    # `last` is ep199 on both sides, same size: only the bytes tell them apart.
    assert "last.pth.tar differs" in held["autoattack_sweep_results_last.csv"].detail


def test_missing_results_are_held_back_too(tmp_path):
    records, qdir, sdir, store = _setup(tmp_path)
    _ckpts(qdir, FAILED, NEW_MTIME)
    _results(qdir, b"failed", NEW_MTIME)
    _ckpts(sdir, RERUN, OLD_MTIME)          # the store has no result files at all

    pulls, held = _plan(tmp_path, records, store)
    assert pulls == {}
    assert set(held) == RESULTS


def test_a_finished_rerun_brings_its_results_with_it(tmp_path):
    """The checkpoints are pulled in the same pass, so the results describe what the
    store will hold: they must still arrive (the qnap-slurm-rerun path)."""
    records, qdir, sdir, store = _setup(tmp_path, source="sjm")
    _ckpts(qdir, RERUN, NEW_MTIME)
    _results(qdir, b"rerun", NEW_MTIME)
    _ckpts(sdir, FAILED, OLD_MTIME)
    _results(sdir, b"failed", OLD_MTIME)

    pulls, held = _plan(tmp_path, records, store)
    assert held == {}
    assert {r for r in pulls if r.endswith(".pth.tar")} == set(RERUN)
    assert RESULTS <= set(pulls)


def test_newer_results_beside_an_identical_checkpoint_still_flow(tmp_path):
    """The everyday case: the sweep added cells on the cluster, same checkpoint."""
    records, qdir, sdir, store = _setup(tmp_path)
    _ckpts(qdir, RERUN, OLD_MTIME)
    _ckpts(sdir, RERUN, OLD_MTIME)
    _results(qdir, b"more cells", NEW_MTIME)
    _results(sdir, b"fewer", OLD_MTIME)

    pulls, held = _plan(tmp_path, records, store)
    assert held == {}
    assert pulls == {name: "qnap-newer" for name in RESULTS}


def test_identical_bytes_under_a_different_mtime_count_as_the_same_checkpoint(tmp_path):
    """mtime lies here (the 2026-08-10 bulk rewrite): sha256 decides, not the stamp."""
    records, qdir, sdir, store = _setup(tmp_path)
    _ckpts(qdir, RERUN, NEW_MTIME)
    _ckpts(sdir, RERUN, OLD_MTIME)
    _results(qdir, b"more cells", NEW_MTIME)

    pulls, held = _plan(tmp_path, records, store)
    assert held == {}
    assert set(pulls) == RESULTS


def test_a_result_without_its_checkpoint_cannot_vouch_for_the_destinations(tmp_path):
    """A source dir missing a keeper (the cluster's partial copies) cannot show its CSV
    describes the store's checkpoint; one that neither side holds is harmless."""
    records, qdir, sdir, store = _setup(tmp_path)
    _ckpts(qdir, {"last.pth.tar": RERUN["last.pth.tar"]}, OLD_MTIME)
    _ckpts(sdir, {"model_best.pth.tar": RERUN["model_best.pth.tar"],
                  "last.pth.tar": RERUN["last.pth.tar"]}, OLD_MTIME)
    _results(qdir, b"src", NEW_MTIME)

    pulls, held = _plan(tmp_path, records, store)
    assert "no model_best.pth.tar" in held["autoattack_sweep_results.csv"].detail
    assert "autoattack_sweep_results_last.csv" in pulls       # last is identical
    assert "autoattack_sweep_results_advbest.csv" in pulls    # no advbest anywhere
    # json/png describe all three kinds, and model_best cannot be vouched for
    assert {"autoattack_eps_norm_scores.json", f"autoattack_eval_comparation_{NAME}.png"} <= set(held)


def test_other_metadata_is_not_guarded(tmp_path):
    records, qdir, sdir, store = _setup(tmp_path)
    _ckpts(qdir, FAILED, NEW_MTIME)
    _ckpts(sdir, RERUN, OLD_MTIME)
    _write(qdir / "log.txt", b"new log", NEW_MTIME)
    _write(qdir / "pgd_eval" / "autoattack_sweep_results.csv", b"nested", NEW_MTIME)

    pulls, held = _plan(tmp_path, records, store)
    assert held == {}
    assert pulls == {"log.txt": "missing", "pgd_eval/autoattack_sweep_results.csv": "missing"}
