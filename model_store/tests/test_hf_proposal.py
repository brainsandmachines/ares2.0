"""hf_proposal: Monday proposes (and mails) only what awaits upload and has finished its AA
sweep; "approved mail #N" uploads exactly those models, and only if their checkpoints are
still the ones mailed and their sweep is still complete."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from model_store import hf_export, hf_proposal, verify_hf
from model_store.tests.test_hf_export import _args, _ckpt, _manifest, _row

GRID = [(n, e) for n in ("linf", "l2", "l1") for e in (1.0, 2.0, 4.0, 6.0, 8.0)]


def _aa_csv(target: Path, cells=GRID) -> Path:
    """The sweep CSV for the fixture's ``last`` checkpoint, as the cluster wrote it: the
    model_name and checkpoint_path are the cluster's, not the renamed zoo dir's."""
    path = target.parent / "autoattack_sweep_results_last.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model_name", "checkpoint_path", "epoch", "attack_norm",
                                           "epsilon_input", "clean_acc", "robust_acc"])
        w.writeheader()
        for norm, eps in cells:
            w.writerow({"model_name": "cluster_name",
                        "checkpoint_path": "results/models/arch/cluster_name/last.pth.tar",
                        "epoch": 7, "attack_norm": norm, "epsilon_input": eps,
                        "clean_acc": 70.0, "robust_acc": 30.0})
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    mails, uploads, verifies = [], [], []
    monkeypatch.setattr("aircc.aircc_job_manager.notify.make_emailer",
                        lambda source: lambda subject, body, **kw: mails.append((subject, body, kw)))
    monkeypatch.setattr(hf_export, "upload", lambda mirror, repo, nw: uploads.append(mirror) or 0)
    monkeypatch.setattr(verify_hf, "main", lambda argv: verifies.append(argv) or 0)
    a, _ = _row(tmp_path, "a_l2_4_init0")
    b, _ = _row(tmp_path, "b_l2_4_init0", seed=1)
    for row in (a, b):
        _aa_csv(Path(row["target"]))
    manifest = _manifest(tmp_path, [a, b])
    excludes = tmp_path / "ex.csv"
    excludes.write_text("pattern,reason\n")

    def run(*argv: str) -> int:
        return hf_proposal.main(["--manifest", str(manifest), "--mirror", str(tmp_path / "mirror"),
                                 "--excludes", str(excludes), "--hash-cache", str(tmp_path / "sha.jsonl"),
                                 "--proposals", str(tmp_path / "props"), *argv])

    class Env:
        pass
    e = Env()
    e.__dict__.update(tmp=tmp_path, a=a, b=b, manifest=manifest, excludes=excludes, run=run,
                      mails=mails, uploads=uploads, verifies=verifies, props=tmp_path / "props")
    return e


def _status(env, n: int) -> dict:
    return json.loads((env.props / f"HF-{n}.json").read_text())


def test_nothing_to_propose_sends_no_mail(env):
    assert hf_export.main(_args(env.tmp, env.manifest, "--apply",
                                "--excludes", str(env.excludes))) == 0  # both already exported
    assert env.run("propose") == 0
    assert env.mails == [] and not env.props.exists()


def test_propose_numbers_mails_and_supersedes(env):
    assert env.run("propose") == 0
    [(subject, body, kw)] = env.mails
    assert subject == "[ares] HF upload proposal #1: 2 model(s)"
    assert "approved mail #1" in body and "a_l2_4_init0" in body and kw == {"urgent": True}
    assert _status(env, 1)["status"] == "open"

    assert env.run("propose") == 0
    assert env.mails[1][0].startswith("[ares] HF upload proposal #2")
    assert "supersedes open proposal(s): #1" in env.mails[1][1]
    assert _status(env, 1)["status"] == "superseded" and _status(env, 2)["status"] == "open"
    assert env.run("approve", "1") == 2  # superseded proposals cannot be approved
    assert env.uploads == []


def test_excluded_models_are_not_proposed(env):
    env.excludes.write_text("pattern,reason\nb_*,collapsed\n")
    assert env.run("propose") == 0
    assert hf_proposal.load_status(env.props, 1)["models"] == ["a_l2_4_init0"]


def test_approve_uploads_exactly_the_proposed_models(env):
    assert env.run("propose") == 0
    assert env.run("approve", "1", "--device", "cpu") == 0

    mirror = env.tmp / "mirror" / "convnext_base/madry/l2"
    assert (mirror / "a_l2_4_init0" / "model.safetensors").is_file()
    assert (mirror / "b_l2_4_init0" / "model.safetensors").is_file()
    assert len(env.uploads) == 1
    [vargv] = env.verifies
    assert "--forward" in vargv and vargv[vargv.index("--only") + 1:] == [
        env.a["relpath"], env.b["relpath"]]
    assert _status(env, 1)["status"] == "uploaded"
    assert env.run("approve", "1") == 2  # not twice


def test_approve_refused_when_a_checkpoint_changed_after_the_mail(env, capsys):
    assert env.run("propose") == 0
    _ckpt(Path(env.a["target"]), seed=5)  # retrained / re-synced: new bytes, new inode

    assert env.run("approve", "1") == 1
    assert env.uploads == [] and not (env.tmp / "mirror").exists()
    assert "a_l2_4_init0: checkpoint changed since the proposal" in capsys.readouterr().err
    assert _status(env, 1)["status"] == "open"


def test_dry_run_writes_and_mails_nothing(env, capsys):
    assert env.run("propose", "--dry-run") == 0
    assert env.mails == [] and not env.props.exists()
    assert "approved mail #1" in capsys.readouterr().out


def test_model_with_incomplete_aa_sweep_is_held_not_proposed(env, capsys):
    _aa_csv(Path(env.b["target"]), cells=GRID[:3])  # sweep still running when it was synced
    assert env.run("propose") == 0
    [(subject, body, _)] = env.mails
    assert subject == "[ares] HF upload proposal #1: 1 model(s)"
    assert "Held back until their AutoAttack sweep reaches Botero (1):" in body
    assert "b_l2_4_init0  AA sweep incomplete: last 12 cells missing" in body
    st = _status(env, 1)
    assert st["models"] == ["a_l2_4_init0"]
    assert st["held"] == [{"model": "b_l2_4_init0",
                           "reason": "AA sweep incomplete: last 12 cells missing"}]
    assert "HELD b_l2_4_init0" in capsys.readouterr().out


def test_nothing_mailed_when_every_candidate_is_held(env, capsys):
    for row in (env.a, env.b):
        (Path(row["target"]).parent / "autoattack_sweep_results_last.csv").unlink()
    assert env.run("propose") == 0
    assert env.mails == [] and not env.props.exists()
    assert "(2 held)" in capsys.readouterr().out


def test_approve_refused_when_the_sweep_is_incomplete(env, capsys):
    """A proposal mailed before the gate existed (HF-6) must not upload an unswept model."""
    assert env.run("propose") == 0
    _aa_csv(Path(env.a["target"]), cells=[])

    assert env.run("approve", "1") == 1
    assert env.uploads == [] and not (env.tmp / "mirror").exists()
    assert "a_l2_4_init0: AA sweep incomplete: last 15 cells missing" in capsys.readouterr().err
    assert _status(env, 1)["status"] == "open"
