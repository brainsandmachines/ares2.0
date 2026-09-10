"""Backfill of a finished Slurm rerun whose checkpoints are not at a higher epoch.

A run reset and retrained on Slurm ends at the same final epoch as the run already
on Botero, so the higher-epoch rule alone never lets its ``last`` arrive -- while
its AA results (metadata) do. These tests pin the exception and every guard on it:
only a finished sjm row unlocks it, a part-way snapshot never replaces a complete
run, identical bytes are never re-copied, the AIRCC root keeps the strict rule,
and the rsync leg does not get ``--update``.

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

NAME = "convnext_base_dvd_b_linftrades_4_init0"
OLD_MTIME, NEW_MTIME = 1_780_000_000, 1_790_000_000

# basename -> (epoch, fill byte); the real epochs of this model (Botero vs Slurm rerun)
OLD_RUN = {"model_best.pth.tar": (176, b"o"), "last.pth.tar": (199, b"o"),
           "model_best_adv.pth.tar": (199, b"o")}
RERUN = {"model_best.pth.tar": (196, b"n"), "last.pth.tar": (199, b"n"),
         "model_best_adv.pth.tar": (197, b"n")}


def _fake_epoch(path):
    try:
        first = Path(path).read_bytes().split(b"\n", 1)[0]
    except OSError:
        return None
    return int(first.split(b"=")[1]) if first.startswith(b"epoch=") else None


@pytest.fixture(autouse=True)
def _epochs(monkeypatch):
    monkeypatch.setattr(backfill, "checkpoint_epoch", _fake_epoch)


def _ckpt(path: Path, epoch: int, fill: bytes, mtime: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"epoch=%03d\n" % epoch + fill * 64)
    os.utime(path, (mtime, mtime))


def _setup(tmp_path, qnap, local, source="sjm", status="finished", label="qnap-slurm"):
    ident = ModelIdentity(canonical=NAME, arch="convnext_base", protocol="trades",
                          norm="linf", eps=4.0, init="0", source="csv")
    qdir = tmp_path / "qnap" / NAME
    store = tmp_path / "models"
    for name, (epoch, fill) in qnap.items():
        _ckpt(qdir / name, epoch, fill, NEW_MTIME)
    for name, (epoch, fill) in local.items():
        _ckpt(store / ident.store_relpath / name, epoch, fill, OLD_MTIME)
    rec = ModelRecord(identity=ident, dirs={label: qdir}, db_source=source, db_status=status)
    return {NAME: rec}, store


def _plan(tmp_path, records, store, label="qnap-slurm"):
    items = backfill.plan(records, store, labels=(label,),
                          cache=HashCache(tmp_path / "sha256.jsonl"))
    return {(it.rel, it.reason) for it in items}


def test_finished_rerun_replaces_equal_and_lower_epoch_checkpoints(tmp_path):
    records, store = _setup(tmp_path, RERUN, OLD_RUN)
    assert _plan(tmp_path, records, store) == {
        ("model_best.pth.tar", "qnap-higher-epoch"),
        ("last.pth.tar", "qnap-slurm-rerun"),          # 199 -> 199
        ("model_best_adv.pth.tar", "qnap-slurm-rerun"),  # 199 -> 197
    }


@pytest.mark.parametrize("source,status", [
    ("sjm", "running"), ("sjm", "pending"), ("sjm", "failed"), ("aircc", "finished"),
])
def test_only_a_finished_sjm_row_unlocks_it(tmp_path, source, status):
    records, store = _setup(tmp_path, RERUN, OLD_RUN, source, status)
    assert _plan(tmp_path, records, store) == {("model_best.pth.tar", "qnap-higher-epoch")}


def test_a_part_way_snapshot_never_replaces_a_complete_run(tmp_path):
    partial = {"model_best.pth.tar": (110, b"n"), "last.pth.tar": (120, b"n"),
               "model_best_adv.pth.tar": (115, b"n")}
    records, store = _setup(tmp_path, partial, OLD_RUN)
    assert _plan(tmp_path, records, store) == set()


def test_identical_bytes_with_a_different_mtime_are_not_recopied(tmp_path):
    records, store = _setup(tmp_path, OLD_RUN, OLD_RUN)
    assert _plan(tmp_path, records, store) == set()


def test_the_aircc_root_keeps_the_strict_rule(tmp_path):
    records, store = _setup(tmp_path, RERUN, OLD_RUN, label="qnap-aircc")
    assert _plan(tmp_path, records, store, label="qnap-aircc") == {
        ("model_best.pth.tar", "qnap-higher-epoch")}


def test_a_rerun_leg_does_not_get_rsync_update(tmp_path, monkeypatch):
    records, store = _setup(tmp_path, RERUN, OLD_RUN)
    items = [it for it in backfill.plan(records, store, labels=("qnap-slurm",),
                                        cache=HashCache(tmp_path / "sha256.jsonl"))
             if it.reason == "qnap-slurm-rerun"]
    cmds = []

    class _Done:
        returncode, stderr = 0, ""

    monkeypatch.setattr(backfill.subprocess, "run",
                        lambda cmd, **kw: cmds.append(cmd) or _Done())
    assert backfill.apply_pull(items, dry_run=False) == 0
    (cmd,) = cmds
    assert "--update" not in cmd
