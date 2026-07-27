"""Unit tests for the eps_norm JSON fast path in best_checkpoint_for_threat."""

from __future__ import annotations

import json

from aircc.aircc_job_manager.best_checkpoint import best_checkpoint_for_threat


def test_best_checkpoint_uses_eps_norm_json_when_it_matches_threat(tmp_path):
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()
    (model_dir / "model_best.pth.tar").write_bytes(b"best")
    (model_dir / "last.pth.tar").write_bytes(b"last")
    (model_dir / "autoattack_eps_norm_scores.json").write_text(
        json.dumps(
            {
                "attack_norm": "linf",
                "epsilon_input": 8.0,
                "scores": {"model_best.pth.tar": 42.0, "last.pth.tar": 39.5},
            }
        )
    )

    path, score = best_checkpoint_for_threat(model_dir, "linf", 8.0)

    assert path == str((model_dir / "model_best.pth.tar").resolve())
    assert score == 42.0


def test_best_checkpoint_ignores_eps_norm_json_on_threat_mismatch(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()
    (model_dir / "autoattack_eps_norm_scores.json").write_text(
        json.dumps(
            {
                "attack_norm": "linf",
                "epsilon_input": 8.0,
                "scores": {"model_best.pth.tar": 42.0},
            }
        )
    )

    # No CSV grid exists either, so a mismatched threat model must fall through
    # to the CSV-based path and come back empty rather than misusing the JSON.
    path, score = best_checkpoint_for_threat(model_dir, "l2", 4.0)

    assert path is None
    assert score is None
