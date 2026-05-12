import logging
import importlib
from types import SimpleNamespace
from pathlib import Path

import torch

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


def test_create_model_from_checkpoint_uses_timm_for_standard_model(monkeypatch):
    captured = {}
    expected = object()

    def _create_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(fe, "create_model", _create_model)

    eval_cfg = SimpleNamespace(num_classes=17, input_size=224)
    model = fe.create_model_from_checkpoint("convnext_small", SimpleNamespace(), eval_cfg)

    assert model is expected
    assert captured["args"] == ("convnext_small",)
    assert captured["kwargs"] == {"pretrained": False, "num_classes": 17}


def test_create_model_from_checkpoint_builds_v1_from_checkpoint_args(monkeypatch):
    v1_convnext = importlib.import_module("ares.model.v1_convnext")

    captured = {}

    class _FakeV1(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(v1_convnext, "V1ConvNeXt", _FakeV1)

    ckpt_args = SimpleNamespace(
        drop=0.1,
        drop_path=0.2,
        gp="avg",
        bn_momentum=0.01,
        bn_eps=0.001,
        v1_noise_train_only=False,
        v1_visual_degrees=9,
        v1_stride=2,
        v1_ksize=17,
        v1_sf_corr=0.5,
        v1_sf_max=8,
        v1_sf_min=1,
        v1_rand_param=True,
        v1_gabor_seed=11,
        v1_simple_channels=32,
        v1_complex_channels=64,
        v1_noise_mode="neuronal",
        v1_noise_scale=0.4,
        v1_noise_level=0.08,
        v1_k_exc=19,
    )
    eval_cfg = SimpleNamespace(num_classes=1000, input_size=224)

    model = fe.create_model_from_checkpoint("convnext_small_v1", ckpt_args, eval_cfg)

    assert isinstance(model, _FakeV1)
    assert captured["backbone_name"] == "convnext_small"
    assert captured["input_size"] == 224
    assert captured["num_classes"] == 1000
    assert captured["drop_rate"] == 0.1
    assert captured["drop_path_rate"] == 0.2
    assert captured["global_pool"] == "avg"
    assert captured["bn_momentum"] == 0.01
    assert captured["bn_eps"] == 0.001
    assert captured["visual_degrees"] == 9
    assert captured["stride"] == 2
    assert captured["ksize"] == 17
    assert captured["sf_corr"] == 0.5
    assert captured["sf_max"] == 8
    assert captured["sf_min"] == 1
    assert captured["rand_param"] is True
    assert captured["gabor_seed"] == 11
    assert captured["simple_channels"] == 32
    assert captured["complex_channels"] == 64
    assert captured["noise_mode"] == "neuronal"
    assert captured["noise_scale"] == 0.4
    assert captured["noise_level"] == 0.08
    assert captured["k_exc"] == 19


def test_final_eval_v1_gabor_seed_defaults_to_checkpoint_seed():
    ckpt_args = SimpleNamespace(seed=123456, v1_gabor_seed=None)

    assert fe.resolve_v1_gabor_seed(ckpt_args) == 123456


def test_load_model_from_ckpt_loads_v1_state_dict_ema(monkeypatch, tmp_path):
    v1_convnext = importlib.import_module("ares.model.v1_convnext")

    class _FakeV1(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(v1_convnext, "V1ConvNeXt", _FakeV1)

    ckpt_path = tmp_path / "model_best.pth.tar"
    torch.save(
        {
            "arch": "convnext_small_v1",
            "args": SimpleNamespace(input_size=224, num_classes=1000),
            "state_dict": {"weight": torch.tensor([2.0])},
            "state_dict_ema": {"weight": torch.tensor([3.0])},
        },
        ckpt_path,
    )

    model, eval_cfg, state_key = fe.load_model_from_ckpt(ckpt_path, torch.device("cpu"))

    assert isinstance(model, _FakeV1)
    assert state_key == "state_dict_ema"
    assert eval_cfg.input_size == 224
    assert torch.equal(model.weight.detach(), torch.tensor([3.0]))


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
