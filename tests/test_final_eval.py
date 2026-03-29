import logging
from pathlib import Path

import data_analysis.final_eval as fe


def test_make_validate_args_exposes_l1_apgd_settings():
    eval_cfg = fe.SimpleNamespace(std=(0.229, 0.224, 0.225), mean=(0.485, 0.456, 0.406))

    args = fe.make_validate_args(
        eval_cfg,
        norm="l1",
        eps_eval=255.0,
        attack_steps=10,
        l1_step_mode="l1_apgd",
        l1_apgd_rho=0.05,
        l1_apgd_use_halving=True,
        l1_apgd_min_step_scale=0.01,
    )

    assert args.attack_norm == "l1"
    assert args.l1_step_mode == "l1_apgd"
    assert args.l1_apgd_rho == 0.05
    assert args.l1_apgd_use_halving is True
    assert args.l1_apgd_min_step_scale == 0.01


def test_run_final_evaluation_writes_custom_pgd_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(fe, "setup_logger", lambda _out_dir: logging.getLogger("test_final_eval"))
    monkeypatch.setattr(fe, "set_seed", lambda _seed: None)

    captured = {}

    def _evaluate_pgd_sweep(**kwargs):
        captured.update(kwargs)
        return [
            {
                "timestamp": "2026-03-29T00:00:00",
                "model_name": "model_best",
                "checkpoint_path": str(kwargs["ckpt_path"]),
                "category": "madry",
                "train_norm": "l1",
                "init": "1",
                "attack_norm": "l1",
                "epsilon_input": 1.0,
                "epsilon_eval": 127.5,
                "attack_steps": 10,
                "attack_step": 25.5,
                "l1_step_mode": "l1_apgd",
                "l1_apgd_rho": 0.05,
                "clean_top1": 80.0,
                "clean_top5": 95.0,
                "adv_top1": 60.0,
                "adv_top5": 85.0,
                "clean_loss": 1.0,
                "adv_loss": 2.0,
                "state_dict_used": "state_dict_ema",
            }
        ]

    monkeypatch.setattr(fe, "evaluate_pgd_sweep", _evaluate_pgd_sweep)

    out_dir = tmp_path / "eval_out"
    outputs = fe.run_final_evaluation(
        checkpoint_path=str(tmp_path / "model_best.pth.tar"),
        models_dir=None,
        val_dir="/tmp/val",
        device="cuda",
        out_dir=str(out_dir),
        aa=False,
        pgd=True,
        aa_batch_size=32,
        aa_norm=None,
        aa_eps=None,
        aa_max_batches=None,
        pgd_batch_size=64,
        pgd_eps=[1.0],
        pgd_norms=["l1"],
        pgd_attack_steps=10,
        pgd_max_batches=None,
        pgd_output_csv="pgd_validation_results_l1_apgd.csv",
        l1_step_mode="l1_apgd",
        l1_apgd_rho=0.05,
        l1_apgd_use_halving=True,
        l1_apgd_min_step_scale=0.01,
        plots=False,
        plot_x_col="epsilon_input",
        num_workers=8,
    )

    csv_path = out_dir / "pgd_validation_results_l1_apgd.csv"
    assert outputs["pgd_csv"] == str(csv_path)
    assert csv_path.exists()
    assert captured["l1_step_mode"] == "l1_apgd"
    assert captured["l1_apgd_rho"] == 0.05
    assert captured["l1_apgd_use_halving"] is True
    assert captured["l1_apgd_min_step_scale"] == 0.01
    assert "l1_step_mode" in csv_path.read_text()
