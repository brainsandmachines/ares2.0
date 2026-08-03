"""Tests for the sweep-completion additions to the AutoAttack engine.

Covers the three behaviours the daily aa_sweep cron depends on:
  * per-setting CSV flush, so a job killed mid-sweep keeps what it finished;
  * --norms / --model-dir plumbing, so an arbitrary run dir can be swept over an explicit grid;
  * existing rows (notably an eps_norm result) being reused rather than recomputed.
"""

import argparse
import csv
import sys
import types
from pathlib import Path

import pytest


class _DummyAutoAttack:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.modules.setdefault("autoattack", types.SimpleNamespace(AutoAttack=_DummyAutoAttack))

from data_analysis import autoattack_array_eval as engine  # noqa: E402


FIELDS = [
    "run_id", "timestamp", "model_name", "checkpoint_kind", "checkpoint_path", "state_dict_used",
    "epoch", "attack_norm", "epsilon_input", "epsilon_eval", "clean_acc", "robust_acc",
    "num_images", "batch_size", "num_batches", "seed", "selection_json",
]


def _row(model_name, ckpt_path, norm, eps, robust_acc=50.0, timestamp="2026-01-01T00:00:00"):
    return {
        "run_id": "run", "timestamp": timestamp, "model_name": model_name,
        "checkpoint_kind": "last", "checkpoint_path": ckpt_path, "state_dict_used": "state_dict",
        "epoch": 40, "attack_norm": norm, "epsilon_input": eps,
        "epsilon_eval": eps, "clean_acc": 70.0, "robust_acc": robust_acc,
        "num_images": 1024, "batch_size": 128, "num_batches": 8, "seed": 0,
        "selection_json": "results/models/m/autoattack_sweep_selection.json",
    }


def _read(csv_path):
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _cells(rows):
    return {(r["attack_norm"], float(r["epsilon_input"])) for r in rows}


# --- --norms parsing -------------------------------------------------------------------------

def test_parse_norms_accepts_subsets_and_normalises_case():
    assert engine.parse_norms("linf,l2,l1") == ("linf", "l2", "l1")
    assert engine.parse_norms(" L2 ") == ("l2",)


def test_parse_norms_rejects_unknown_norm():
    with pytest.raises(ValueError, match="l3"):
        engine.parse_norms("linf,l3")


def test_parse_norms_rejects_empty():
    with pytest.raises(ValueError):
        engine.parse_norms(" , ")


# --- grid construction -----------------------------------------------------------------------

def test_expected_settings_uses_requested_norms_and_eps():
    settings = engine.expected_settings(None, eps_inputs=(1.0, 2.0), norms=("linf", "l2"))
    assert {(norm, eps) for norm, eps, _ in settings} == {
        ("linf", 1.0), ("linf", 2.0), ("l2", 1.0), ("l2", 2.0)
    }


def test_is_complete_output_ignores_eps_12_rows_outside_the_grid(tmp_path):
    """eps 12 is no longer wanted; its presence must not affect completeness either way."""
    csv_path = tmp_path / "autoattack_sweep_results_last.csv"
    ckpt = "/models/m/last.pth.tar"
    engine.write_rows(csv_path, [_row("m", ckpt, "linf", eps) for eps in (1.0, 2.0, 12.0)])

    assert engine.is_complete_output(
        csv_path, None, checkpoint_path=ckpt, model_name="m",
        eps_inputs=(1.0, 2.0), norms=("linf",),
    )
    assert not engine.is_complete_output(
        csv_path, None, checkpoint_path=ckpt, model_name="m",
        eps_inputs=(1.0, 2.0, 4.0), norms=("linf",),
    )


# --- per-setting flush -----------------------------------------------------------------------

def _patch_sweep_deps(monkeypatch, tmp_path, attack_results):
    """Stub out everything the sweep needs beyond CSV bookkeeping (model load, data, attacks)."""
    monkeypatch.setattr(engine, "load_model", lambda ckpt, device: (object(), object(), "state_dict"))
    monkeypatch.setattr(
        engine, "build_selected_loader",
        lambda *a, **k: (object(), list(range(1024)), [0] * 1024),
    )
    monkeypatch.setattr(engine, "validate_raw_batch", lambda loader: None)
    monkeypatch.setattr(engine, "clean_accuracy", lambda *a, **k: 0.70)
    monkeypatch.setattr(engine, "set_seed", lambda seed: None)
    monkeypatch.setattr(engine.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(engine, "extract_epoch_from_checkpoint", lambda p: 40)

    calls = []

    def fake_attack(model, loader, device, norm, epsilon, seed, logger):
        calls.append((norm, epsilon))
        return attack_results(norm, epsilon)

    monkeypatch.setattr(engine, "run_autoattack_setting", fake_attack)
    return calls


def test_rows_are_flushed_after_every_setting(monkeypatch, tmp_path):
    """A sweep interrupted mid-way must leave the settings it finished on disk."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    ckpt = model_dir / "last.pth.tar"
    ckpt.write_text("x")
    csv_path = model_dir / "autoattack_sweep_results_last.csv"

    def blow_up_on_third(norm, epsilon):
        done = len(_read(csv_path)) if csv_path.exists() else 0
        if done >= 2:
            raise RuntimeError("simulated Slurm time limit")
        return 0.5

    _patch_sweep_deps(monkeypatch, tmp_path, blow_up_on_third)

    with pytest.raises(RuntimeError, match="time limit"):
        engine.run_autoattack_sweep_for_checkpoint(
            checkpoint_path=ckpt, model_dir=model_dir, val_dir=tmp_path, device="cpu",
            output_csv="autoattack_sweep_results_last.csv", checkpoint_kind="last",
            eps_inputs=(1.0, 2.0, 4.0), norms=("linf",),
        )

    # Two completed settings survived the crash; the old write-once-at-the-end code left zero.
    rows = _read(csv_path)
    assert len(rows) == 2
    assert _cells(rows) == {("linf", 1.0), ("linf", 2.0)}


def test_interrupted_sweep_resumes_without_recomputing(monkeypatch, tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    ckpt = model_dir / "last.pth.tar"
    ckpt.write_text("x")
    csv_path = model_dir / "autoattack_sweep_results_last.csv"
    engine.write_rows(csv_path, [_row("m", str(ckpt), "linf", 1.0), _row("m", str(ckpt), "linf", 2.0)])

    calls = _patch_sweep_deps(monkeypatch, tmp_path, lambda norm, eps: 0.5)
    engine.run_autoattack_sweep_for_checkpoint(
        checkpoint_path=ckpt, model_dir=model_dir, val_dir=tmp_path, device="cpu",
        output_csv="autoattack_sweep_results_last.csv", checkpoint_kind="last",
        eps_inputs=(1.0, 2.0, 4.0), norms=("linf",),
    )

    assert calls == [("linf", 4.0 / engine.LINF_DIVISOR)]
    assert _cells(_read(csv_path)) == {("linf", 1.0), ("linf", 2.0), ("linf", 4.0)}


def test_existing_eps_norm_row_is_reused_not_recomputed(monkeypatch, tmp_path):
    """The whole point of the daily sweep: an eps_norm result already on disk is kept as-is."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    ckpt = model_dir / "last.pth.tar"
    ckpt.write_text("x")
    csv_path = model_dir / "autoattack_sweep_results_last.csv"
    # An eps_norm eval wrote exactly one cell, with a relative path as it does on the cluster.
    engine.write_rows(csv_path, [
        _row("m", "results/models/m/last.pth.tar", "l2", 4.0, robust_acc=41.5, timestamp="EPSNORM")
    ])

    calls = _patch_sweep_deps(monkeypatch, tmp_path, lambda norm, eps: 0.5)
    engine.run_autoattack_sweep_for_checkpoint(
        checkpoint_path=ckpt, model_dir=model_dir, val_dir=tmp_path, device="cpu",
        output_csv="autoattack_sweep_results_last.csv", checkpoint_kind="last",
        eps_inputs=(1.0, 2.0, 4.0), norms=("l2",),
    )

    # 3 cells expected, 1 already present -> only 2 attacks run, and the original row is untouched
    # despite having been written under a different (relative) checkpoint path.
    assert len(calls) == 2
    assert ("l2", 4.0) not in calls
    rows = _read(csv_path)
    assert _cells(rows) == {("l2", 1.0), ("l2", 2.0), ("l2", 4.0)}
    kept = [r for r in rows if float(r["epsilon_input"]) == 4.0][0]
    assert kept["timestamp"] == "EPSNORM"
    assert float(kept["robust_acc"]) == 41.5


def test_force_recomputes_and_replaces_existing_rows(monkeypatch, tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    ckpt = model_dir / "last.pth.tar"
    ckpt.write_text("x")
    csv_path = model_dir / "autoattack_sweep_results_last.csv"
    engine.write_rows(csv_path, [_row("m", str(ckpt), "l2", 1.0, robust_acc=11.0, timestamp="OLD")])

    _patch_sweep_deps(monkeypatch, tmp_path, lambda norm, eps: 0.25)
    engine.run_autoattack_sweep_for_checkpoint(
        checkpoint_path=ckpt, model_dir=model_dir, val_dir=tmp_path, device="cpu",
        output_csv="autoattack_sweep_results_last.csv", checkpoint_kind="last",
        eps_inputs=(1.0,), norms=("l2",), force=True,
    )

    rows = _read(csv_path)
    assert len(rows) == 1  # replaced, not duplicated
    assert rows[0]["timestamp"] != "OLD"
    assert float(rows[0]["robust_acc"]) == 25.0


# --- --model-dir plumbing --------------------------------------------------------------------

def test_run_sweep_for_model_dir_passes_norms_and_skips_missing_kinds(monkeypatch, tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "last.pth.tar").write_text("x")  # no model_best / model_best_adv

    seen = []

    def fake_sweep(**kwargs):
        seen.append((kwargs["checkpoint_kind"], kwargs["norms"], kwargs["eps_inputs"]))

    monkeypatch.setattr(engine, "run_autoattack_sweep_for_checkpoint", fake_sweep)
    monkeypatch.setattr(engine, "extract_epoch_from_summary", lambda d: 40)
    monkeypatch.setattr(engine, "extract_epoch_from_checkpoint", lambda p: 40)

    args = argparse.Namespace(
        val_dir=tmp_path, device="cpu", batch_size=32, num_batches=32, num_workers=1, seed=0,
        output_csv="autoattack_sweep_results.csv", selection_json="autoattack_sweep_selection.json",
        run_id="run", force=False, dry_run=False, max_settings=None, plot_comparison=False,
    )

    engine.run_sweep_for_model_dir(
        model_dir, args, eps_inputs=(1.0, 2.0), norms=("linf", "l2"),
        checkpoint_kinds=("best", "last", "advbest"),
    )

    assert seen == [("last", ("linf", "l2"), (1.0, 2.0))]
