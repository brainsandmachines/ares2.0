import contextlib
import csv
import math
import types

import pytest
import torch
from omegaconf import OmegaConf

import robust_training.adversarial_training as advt
from ares.utils.logger import _auto_experiment_name
from ares.utils.model import resolve_v1_gabor_seed


def _compose_base_cfg():
    root = "robust_training/configs"
    top = OmegaConf.load(f"{root}/config.yaml")
    top_dict = OmegaConf.to_container(top, resolve=True)
    top_dict.pop("defaults", None)

    return OmegaConf.create(
        {
            **top_dict,
            "training": OmegaConf.load(f"{root}/training/convnext_small.yaml"),
            "model": OmegaConf.load(f"{root}/model/convnext_small.yaml"),
            "dataset": OmegaConf.load(f"{root}/dataset/imagenet.yaml"),
            "optimizer": OmegaConf.load(f"{root}/optimizer/adamw.yaml"),
            "attacks": OmegaConf.load(f"{root}/attacks/adv.yaml"),
            "dist": OmegaConf.load(f"{root}/dist/default.yaml"),
            "lr_scheduler": OmegaConf.load(f"{root}/lr_scheduler/cosine.yaml"),
            "continuation": OmegaConf.load(f"{root}/continuation/default.yaml"),
            "epsilon_schedule": OmegaConf.load(f"{root}/epsilon_schedule/default.yaml"),
            "checkpointing": OmegaConf.load(f"{root}/checkpointing/default.yaml"),
        }
    )


def _runtime_test_cfg(cfg):
    return OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "dataset": {"train_dir": "/tmp/train", "eval_dir": "/tmp/val"},
                "dist": {
                    "world_size": 1,
                    "rank": 0,
                    "local_rank": 0,
                    "device_id": 0,
                    "distributed": False,
                },
                "output_dir": "/tmp/out",
                "final_eval": False,
            }
        ),
    )


def _sbatch_mode_from_jobname(jobname):
    attack_norm = "linf"
    advtrain = False
    gradnorm = False
    gradnorm_penalty_norm = "l1"
    criterion = "madry"

    if "gradnorm" in jobname:
        gradnorm = True
        if "gradnorm_l2" in jobname:
            gradnorm_penalty_norm = "l2"
    elif "linf" in jobname:
        attack_norm = "linf"
        advtrain = True
    elif "l2" in jobname:
        attack_norm = "l2"
        advtrain = True
    elif "l1" in jobname:
        attack_norm = "l1"
        advtrain = True

    if "trades" in jobname:
        criterion = "trades"

    return {
        "attack_norm": attack_norm,
        "advtrain": advtrain,
        "gradnorm": gradnorm,
        "gradnorm_penalty_norm": gradnorm_penalty_norm,
        "attack_criterion": criterion,
    }


@pytest.mark.parametrize(
    "jobname,eps,exp_num,expected",
    [
        (
            "linf_16_init1",
            16,
            1,
            {
                "attack_norm": "linf",
                "advtrain": True,
                "gradnorm": False,
                "gradnorm_penalty_norm": "l1",
                "attack_criterion": "madry",
            },
        ),
        (
            "l2_16_trades_init2",
            16,
            2,
            {
                "attack_norm": "l2",
                "advtrain": True,
                "gradnorm": False,
                "gradnorm_penalty_norm": "l1",
                "attack_criterion": "trades",
            },
        ),
        (
            "l1_8_init3",
            8,
            3,
            {
                "attack_norm": "l1",
                "advtrain": True,
                "gradnorm": False,
                "gradnorm_penalty_norm": "l1",
                "attack_criterion": "madry",
            },
        ),
        (
            "gradnorm_16_init4",
            16,
            4,
            {
                "attack_norm": "linf",
                "advtrain": False,
                "gradnorm": True,
                "gradnorm_penalty_norm": "l1",
                "attack_criterion": "madry",
            },
        ),
        (
            "gradnorm_l1_16_init4",
            16,
            4,
            {
                "attack_norm": "linf",
                "advtrain": False,
                "gradnorm": True,
                "gradnorm_penalty_norm": "l1",
                "attack_criterion": "madry",
            },
        ),
        (
            "gradnorm_l2_16_init4",
            16,
            4,
            {
                "attack_norm": "linf",
                "advtrain": False,
                "gradnorm": True,
                "gradnorm_penalty_norm": "l2",
                "attack_criterion": "madry",
            },
        ),
    ],
)
def test_sbatch_launch_modes_are_launchable(jobname, eps, exp_num, expected):
    parsed = _sbatch_mode_from_jobname(jobname)
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "attacks": {
                    "attack_eps": float(eps),
                    "attack_norm": parsed["attack_norm"],
                    "advtrain": parsed["advtrain"],
                    "gradnorm": parsed["gradnorm"],
                    "gradnorm_penalty_norm": parsed["gradnorm_penalty_norm"],
                    "attack_criterion": parsed["attack_criterion"],
                },
                "model": {"experiment_num": int(exp_num)},
            }
        ),
    )

    assert cfg.attacks.attack_eps == float(eps)
    assert cfg.model.experiment_num == exp_num
    assert cfg.attacks.attack_norm == expected["attack_norm"]
    assert cfg.attacks.advtrain is expected["advtrain"]
    assert cfg.attacks.gradnorm is expected["gradnorm"]
    assert cfg.attacks.gradnorm_penalty_norm == expected["gradnorm_penalty_norm"]
    assert cfg.attacks.attack_criterion == expected["attack_criterion"]


@pytest.mark.parametrize(
    "mode,eps,criterion,expect_regnorm,expected_eps,expected_random_start,gradnorm_penalty_norm",
    [
        ("linf", 16.0, "madry", False, 16.0 / 255.0, False, "l1"),
        ("l1", 2.0, "madry", False, 2.0 * 255.0 / 2.0, False, "l1"),
        ("l2", 4.0, "trades", False, 4.0, True, "l1"),
        ("gradnorm", 8.0, "madry", True, 8.0 / 255.0, False, "l1"),
        ("gradnorm", 8.0, "madry", True, 8.0 / 255.0, False, "l2"),
    ],
)
def test_main_one_epoch_modes(
    monkeypatch,
    mode,
    eps,
    criterion,
    expect_regnorm,
    expected_eps,
    expected_random_start,
    gradnorm_penalty_norm,
):
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {"epochs": 1, "model_ema": False, "batch_size": 2},
                "attacks": {
                    "attack_eps": float(eps),
                    "attack_norm": "linf" if mode == "gradnorm" else mode,
                    "advtrain": mode in {"linf", "l2", "l1"},
                    "gradnorm": mode == "gradnorm",
                    "gradnorm_penalty_norm": gradnorm_penalty_norm,
                    "attack_criterion": criterion,
                },
                "model": {"experiment_num": 1, "resume": ""},
            }
        ),
    )
    cfg = _runtime_test_cfg(cfg)

    class _Logger:
        def info(self, *_a, **_k):
            return None

        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def exception(self, *_a, **_k):
            return None

    class _DummySched:
        def step(self, *_a, **_k):
            return None

        def step_update(self, *_a, **_k):
            return None

    class _DummySaver:
        def save_checkpoint(self, *_a, **_k):
            return (0.9, 0)

    train_call = {}

    def _train_one_epoch_stub(epoch, model, loader, optimizer, train_loss_fn, in_cfg, reg_loss_fn=None, **kwargs):
        train_call["epoch"] = epoch
        train_call["attack_eps"] = in_cfg.attacks.attack_eps
        train_call["random_start"] = in_cfg.attacks.get("random_start", False)
        train_call["reg_loss_fn"] = reg_loss_fn
        train_call["gradnorm_start_epoch"] = kwargs.get("gradnorm_start_epoch")
        return {"loss": 0.1}

    def _fake_loader():
        x = torch.zeros(2, 3, 8, 8)
        y = torch.zeros(2, dtype=torch.long)
        return [(x, y)]

    monkeypatch.setattr(advt, "distributed_init", lambda _args: None)
    monkeypatch.setattr(advt, "random_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "setup_logger", lambda *a, **k: _Logger())
    monkeypatch.setattr(advt, "resolve_amp", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "build_model", lambda *_a, **_k: torch.nn.Linear(8, 3))
    monkeypatch.setattr(advt, "optimizer_kwargs", lambda **_k: {})
    monkeypatch.setattr(advt, "create_optimizer_v2", lambda model, **_k: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(advt, "build_loss_scaler", lambda *_a, **_k: (contextlib.nullcontext, None))
    monkeypatch.setattr(advt, "build_dataset", lambda *_a, **_k: (_fake_loader(), _fake_loader()))
    monkeypatch.setattr(advt, "build_loss", lambda *_a, **_k: (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()))
    monkeypatch.setattr(advt, "create_scheduler_v2", lambda *_a, **_k: (_DummySched(), 1))
    monkeypatch.setattr(advt, "scheduler_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(advt, "CheckpointSaver", lambda *a, **k: _DummySaver())
    monkeypatch.setattr(advt, "validate", lambda *_a, **_k: {"top1": 1.0, "loss": 0.1})
    monkeypatch.setattr(advt, "update_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "resume_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "load_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "distribute_bn", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "_maybe_run_final_eval", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "train_one_epoch", _train_one_epoch_stub)

    monkeypatch.setattr(advt.wandb, "init", lambda **_k: None)
    monkeypatch.setattr(advt.wandb, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(advt.wandb.util, "generate_id", lambda: "test-run-id")

    advt.main(cfg)

    assert train_call["epoch"] == 0
    assert math.isclose(train_call["attack_eps"], expected_eps, rel_tol=1e-8)
    assert train_call["random_start"] is expected_random_start
    if expect_regnorm:
        assert isinstance(train_call["reg_loss_fn"], advt.DBP)
        assert train_call["reg_loss_fn"].penalty_norm == cfg.attacks.gradnorm_penalty_norm
        assert train_call["gradnorm_start_epoch"] == cfg.attacks.alpha_start_epoch
    else:
        assert train_call["reg_loss_fn"] is None


def test_final_eval_defaults_are_pgd_without_autoattack_with_plots():
    cfg = _compose_base_cfg()

    assert cfg.final_eval is True
    assert cfg.final_eval_pgd is True
    assert cfg.final_eval_autoattack is True
    assert cfg.final_eval_plots is True


def test_maybe_run_final_eval_default_call(monkeypatch, tmp_path):
    calls = []
    fake_module = types.ModuleType("data_analysis.final_eval")

    def _run_final_evaluation(**kwargs):
        calls.append(kwargs)

    fake_module.run_final_evaluation = _run_final_evaluation
    monkeypatch.setitem(__import__("sys").modules, "data_analysis.final_eval", fake_module)
    monkeypatch.setattr(advt.torch.cuda, "is_available", lambda: True)

    class _Logger:
        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def exception(self, *_a, **_k):
            return None

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_best.pth.tar").write_bytes(b"x")

    cfg = OmegaConf.merge(
        _runtime_test_cfg(_compose_base_cfg()),
        OmegaConf.create(
            {
                "final_eval": True,
                "final_eval_autoattack": False,
                "final_eval_pgd": True,
                "final_eval_ckpt_name": "model_best.pth.tar",
                "final_eval_val_dir": "",
                "final_eval_out_dir": "",
                "final_eval_aa_batch_size": None,
                "training": {"batch_size": 2},
                "final_eval_aa_norm": None,
                "final_eval_aa_eps": None,
                "final_eval_aa_max_batches": None,
                "final_eval_pgd_batch_size": None,
                "final_eval_pgd_eps": "0.5,1,2,4,8,16",
                "final_eval_pgd_norms": "linf,l2,l1",
                "final_eval_pgd_attack_steps": 10,
                "final_eval_pgd_max_batches": None,
                "final_eval_plots": True,
                "final_eval_plot_x_col": "epsilon_input",
                "final_eval_num_workers": 8,
            }
        ),
    )

    advt._maybe_run_final_eval(cfg, str(output_dir), _Logger())

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["aa"] is False
    assert kwargs["pgd"] is True
    assert kwargs["plots"] is True


def test_maybe_run_final_eval_skips_when_pgd_csv_is_in_pgd_eval_subdir(monkeypatch, tmp_path):
    calls = []
    fake_module = types.ModuleType("data_analysis.final_eval")

    def _run_final_evaluation(**kwargs):
        calls.append(kwargs)

    fake_module.run_final_evaluation = _run_final_evaluation
    monkeypatch.setitem(__import__("sys").modules, "data_analysis.final_eval", fake_module)
    monkeypatch.setattr(advt.torch.cuda, "is_available", lambda: True)

    class _Logger:
        def info(self, *_a, **_k):
            return None

        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def exception(self, *_a, **_k):
            return None

    output_dir = tmp_path / "out"
    pgd_eval_dir = output_dir / "pgd_eval"
    pgd_eval_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "model_best.pth.tar"
    ckpt.write_bytes(b"x")

    csv_path = pgd_eval_dir / "pgd_validation_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint_path", "attack_norm", "epsilon_input"])
        w.writeheader()
        for norm in ("linf", "l2", "l1"):
            for eps in (0.5, 1, 2, 4, 8, 16):
                w.writerow(
                    {
                        "checkpoint_path": str(ckpt),
                        "attack_norm": norm,
                        "epsilon_input": eps,
                    }
                )

    cfg = OmegaConf.merge(
        _runtime_test_cfg(_compose_base_cfg()),
        OmegaConf.create(
            {
                "final_eval": True,
                "final_eval_autoattack": False,
                "final_eval_pgd": True,
                "final_eval_ckpt_name": "model_best.pth.tar",
                "final_eval_val_dir": "",
                "final_eval_out_dir": "",
                "final_eval_aa_batch_size": None,
                "training": {"batch_size": 2},
                "final_eval_aa_norm": None,
                "final_eval_aa_eps": None,
                "final_eval_aa_max_batches": None,
                "final_eval_pgd_batch_size": None,
                "final_eval_pgd_eps": "0.5,1,2,4,8,16",
                "final_eval_pgd_norms": "linf,l2,l1",
                "final_eval_pgd_attack_steps": 10,
                "final_eval_pgd_max_batches": None,
                "final_eval_plots": True,
                "final_eval_plot_x_col": "epsilon_input",
                "final_eval_num_workers": 8,
                "final_eval_skip_if_complete": True,
            }
        ),
    )

    advt._maybe_run_final_eval(cfg, str(output_dir), _Logger())

    assert len(calls) == 0


def test_main_v1_feature_attack_defaults(monkeypatch):
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {"epochs": 1, "model_ema": False, "batch_size": 2},
                "attacks": {
                    "attack_domain": "v1_feature",
                    "attack_norm": "linf",
                    "advtrain": True,
                    "attack_criterion": "madry",
                    "v1_attack_eps": 9.0,
                    "v1_attack_step": None,
                    "v1_attack_it": 3,
                },
                "model": {"experiment_num": 1, "resume": "", "model": "convnext_small_v1"},
            }
        ),
    )
    cfg = _runtime_test_cfg(cfg)

    class _Logger:
        def info(self, *_a, **_k):
            return None

        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def exception(self, *_a, **_k):
            return None

    class _DummySched:
        def step(self, *_a, **_k):
            return None

        def step_update(self, *_a, **_k):
            return None

    class _DummySaver:
        def save_checkpoint(self, *_a, **_k):
            return (0.9, 0)

    train_call = {}

    def _train_one_epoch_stub(epoch, model, loader, optimizer, train_loss_fn, in_cfg, reg_loss_fn=None, **kwargs):
        train_call["attack_domain"] = in_cfg.attacks.attack_domain
        train_call["v1_attack_step"] = in_cfg.attacks.v1_attack_step
        return {"loss": 0.1}

    def _fake_loader():
        x = torch.zeros(2, 3, 8, 8)
        y = torch.zeros(2, dtype=torch.long)
        return [(x, y)]

    monkeypatch.setattr(advt, "distributed_init", lambda _args: None)
    monkeypatch.setattr(advt, "random_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "setup_logger", lambda *a, **k: _Logger())
    monkeypatch.setattr(advt, "resolve_amp", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "build_model", lambda *_a, **_k: torch.nn.Linear(8, 3))
    monkeypatch.setattr(advt, "optimizer_kwargs", lambda **_k: {})
    monkeypatch.setattr(advt, "create_optimizer_v2", lambda model, **_k: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(advt, "build_loss_scaler", lambda *_a, **_k: (contextlib.nullcontext, None))
    monkeypatch.setattr(advt, "build_dataset", lambda *_a, **_k: (_fake_loader(), _fake_loader()))
    monkeypatch.setattr(advt, "build_loss", lambda *_a, **_k: (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()))
    monkeypatch.setattr(advt, "create_scheduler_v2", lambda *_a, **_k: (_DummySched(), 1))
    monkeypatch.setattr(advt, "scheduler_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(advt, "CheckpointSaver", lambda *a, **k: _DummySaver())
    monkeypatch.setattr(advt, "validate", lambda *_a, **_k: {"top1": 1.0, "loss": 0.1})
    monkeypatch.setattr(advt, "update_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "resume_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "load_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "distribute_bn", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "_maybe_run_final_eval", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "train_one_epoch", _train_one_epoch_stub)
    monkeypatch.setattr(advt.wandb, "init", lambda **_k: None)
    monkeypatch.setattr(advt.wandb, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(advt.wandb.util, "generate_id", lambda: "test-run-id")

    advt.main(cfg)

    assert train_call["attack_domain"] == "v1_feature"
    assert math.isclose(train_call["v1_attack_step"], (9.0 / 255.0) / 3.0, rel_tol=1e-8)


@pytest.mark.parametrize("criterion", ["madry", "trades"])
def test_main_rejects_v1_noise_with_advtrain(monkeypatch, criterion):
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {"epochs": 1, "model_ema": False, "batch_size": 2},
                "attacks": {
                    "attack_domain": "v1_feature",
                    "attack_norm": "linf",
                    "advtrain": True,
                    "attack_criterion": criterion,
                    "v1_attack_eps": 6.0,
                    "v1_attack_step": None,
                    "v1_attack_it": 3,
                },
                "model": {
                    "experiment_num": 1,
                    "resume": "",
                    "model": "convnext_small_v1",
                    "v1_noise_mode": "neuronal",
                },
            }
        ),
    )
    cfg = _runtime_test_cfg(cfg)

    monkeypatch.setattr(advt, "distributed_init", lambda _args: None)

    with pytest.raises(ValueError, match="Adversarial training with V1 noise is not supported"):
        advt.main(cfg)


def test_main_v1_feature_trades_defaults(monkeypatch):
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {"epochs": 1, "model_ema": False, "batch_size": 2},
                "attacks": {
                    "attack_domain": "v1_feature",
                    "attack_norm": "linf",
                    "advtrain": True,
                    "attack_criterion": "trades",
                    "v1_attack_eps": 6.0,
                    "v1_attack_step": None,
                    "v1_attack_it": 3,
                },
                "model": {"experiment_num": 1, "resume": "", "model": "convnext_small_v1"},
            }
        ),
    )
    cfg = _runtime_test_cfg(cfg)

    class _Logger:
        def info(self, *_a, **_k):
            return None

        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def exception(self, *_a, **_k):
            return None

    class _DummySched:
        def step(self, *_a, **_k):
            return None

        def step_update(self, *_a, **_k):
            return None

    class _DummySaver:
        def save_checkpoint(self, *_a, **_k):
            return (0.9, 0)

    train_call = {}

    def _train_one_epoch_stub(epoch, model, loader, optimizer, train_loss_fn, in_cfg, reg_loss_fn=None, **kwargs):
        train_call["attack_domain"] = in_cfg.attacks.attack_domain
        train_call["attack_criterion"] = in_cfg.attacks.attack_criterion
        train_call["v1_attack_step"] = in_cfg.attacks.v1_attack_step
        train_call["random_start"] = in_cfg.attacks.get("random_start", False)
        return {"loss": 0.1}

    def _fake_loader():
        x = torch.zeros(2, 3, 8, 8)
        y = torch.zeros(2, dtype=torch.long)
        return [(x, y)]

    monkeypatch.setattr(advt, "distributed_init", lambda _args: None)
    monkeypatch.setattr(advt, "random_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "setup_logger", lambda *a, **k: _Logger())
    monkeypatch.setattr(advt, "resolve_amp", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "build_model", lambda *_a, **_k: torch.nn.Linear(8, 3))
    monkeypatch.setattr(advt, "optimizer_kwargs", lambda **_k: {})
    monkeypatch.setattr(advt, "create_optimizer_v2", lambda model, **_k: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(advt, "build_loss_scaler", lambda *_a, **_k: (contextlib.nullcontext, None))
    monkeypatch.setattr(advt, "build_dataset", lambda *_a, **_k: (_fake_loader(), _fake_loader()))
    monkeypatch.setattr(advt, "build_loss", lambda *_a, **_k: (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()))
    monkeypatch.setattr(advt, "create_scheduler_v2", lambda *_a, **_k: (_DummySched(), 1))
    monkeypatch.setattr(advt, "scheduler_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(advt, "CheckpointSaver", lambda *a, **k: _DummySaver())
    monkeypatch.setattr(advt, "validate", lambda *_a, **_k: {"top1": 1.0, "loss": 0.1})
    monkeypatch.setattr(advt, "update_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "resume_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "load_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "distribute_bn", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "_maybe_run_final_eval", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "train_one_epoch", _train_one_epoch_stub)
    monkeypatch.setattr(advt.wandb, "init", lambda **_k: None)
    monkeypatch.setattr(advt.wandb, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(advt.wandb.util, "generate_id", lambda: "test-run-id")

    advt.main(cfg)

    assert train_call["attack_domain"] == "v1_feature"
    assert train_call["attack_criterion"] == "trades"
    assert math.isclose(train_call["v1_attack_step"], (6.0 / 255.0) / 3.0, rel_tol=1e-8)
    assert train_call["random_start"] is True


@pytest.mark.parametrize(
    "criterion,expected_random_start",
    [("madry", False), ("trades", True)],
)
def test_main_v1_feature_l2_defaults(monkeypatch, criterion, expected_random_start):
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {"epochs": 1, "model_ema": False, "batch_size": 2},
                "attacks": {
                    "attack_domain": "v1_feature",
                    "attack_norm": "l2",
                    "advtrain": True,
                    "attack_criterion": criterion,
                    "v1_attack_eps": 6.0,
                    "v1_attack_step": None,
                    "v1_attack_it": 3,
                },
                "model": {"experiment_num": 1, "resume": "", "model": "convnext_small_v1"},
            }
        ),
    )
    cfg = _runtime_test_cfg(cfg)

    class _Logger:
        def info(self, *_a, **_k):
            return None

        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def exception(self, *_a, **_k):
            return None

    class _DummySched:
        def step(self, *_a, **_k):
            return None

        def step_update(self, *_a, **_k):
            return None

    class _DummySaver:
        def save_checkpoint(self, *_a, **_k):
            return (0.9, 0)

    train_call = {}

    def _train_one_epoch_stub(epoch, model, loader, optimizer, train_loss_fn, in_cfg, reg_loss_fn=None, **kwargs):
        train_call["attack_domain"] = in_cfg.attacks.attack_domain
        train_call["attack_criterion"] = in_cfg.attacks.attack_criterion
        train_call["v1_attack_step"] = in_cfg.attacks.v1_attack_step
        train_call["random_start"] = in_cfg.attacks.get("random_start", False)
        return {"loss": 0.1}

    def _fake_loader():
        x = torch.zeros(2, 3, 8, 8)
        y = torch.zeros(2, dtype=torch.long)
        return [(x, y)]

    monkeypatch.setattr(advt, "distributed_init", lambda _args: None)
    monkeypatch.setattr(advt, "random_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "setup_logger", lambda *a, **k: _Logger())
    monkeypatch.setattr(advt, "resolve_amp", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "build_model", lambda *_a, **_k: torch.nn.Linear(8, 3))
    monkeypatch.setattr(advt, "optimizer_kwargs", lambda **_k: {})
    monkeypatch.setattr(advt, "create_optimizer_v2", lambda model, **_k: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(advt, "build_loss_scaler", lambda *_a, **_k: (contextlib.nullcontext, None))
    monkeypatch.setattr(advt, "build_dataset", lambda *_a, **_k: (_fake_loader(), _fake_loader()))
    monkeypatch.setattr(advt, "build_loss", lambda *_a, **_k: (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()))
    monkeypatch.setattr(advt, "create_scheduler_v2", lambda *_a, **_k: (_DummySched(), 1))
    monkeypatch.setattr(advt, "scheduler_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(advt, "CheckpointSaver", lambda *a, **k: _DummySaver())
    monkeypatch.setattr(advt, "validate", lambda *_a, **_k: {"top1": 1.0, "loss": 0.1})
    monkeypatch.setattr(advt, "update_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "resume_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "load_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "distribute_bn", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "_maybe_run_final_eval", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "train_one_epoch", _train_one_epoch_stub)
    monkeypatch.setattr(advt.wandb, "init", lambda **_k: None)
    monkeypatch.setattr(advt.wandb, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(advt.wandb.util, "generate_id", lambda: "test-run-id")

    advt.main(cfg)

    assert train_call["attack_domain"] == "v1_feature"
    assert train_call["attack_criterion"] == criterion
    assert math.isclose(train_call["v1_attack_step"], 4.0, rel_tol=1e-8)
    assert train_call["random_start"] is expected_random_start


def test_mixup_turns_off_and_rebuilds_loader(monkeypatch):
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {"epochs": 2, "model_ema": False, "batch_size": 2},
                "dataset": {
                    "mixup_active": True,
                    "mixup_off_epoch": 1,
                },
                "attacks": {
                    "advtrain": False,
                    "gradnorm": False,
                },
                "model": {"experiment_num": 1, "resume": ""},
            }
        ),
    )
    cfg = _runtime_test_cfg(cfg)

    class _Logger:
        def info(self, *_a, **_k):
            return None

        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def exception(self, *_a, **_k):
            return None

    class _DummySched:
        def step(self, *_a, **_k):
            return None

        def step_update(self, *_a, **_k):
            return None

    class _DummySaver:
        def save_checkpoint(self, *_a, **_k):
            return (0.9, 0)

    class _FakeSampler:
        def set_epoch(self, *_a, **_k):
            return None

    class _FakeLoader:
        def __init__(self):
            self.sampler = _FakeSampler()

        def __iter__(self):
            x = torch.zeros(2, 3, 8, 8)
            y = torch.zeros(2, dtype=torch.long)
            yield x, y

        def __len__(self):
            return 1

    build_states = []
    epoch_mixup_states = []

    def _build_dataset_stub(in_cfg, *_a, **_k):
        build_states.append(bool(in_cfg.dataset.mixup_active))
        return _FakeLoader(), _FakeLoader()

    def _train_one_epoch_stub(epoch, model, loader, optimizer, train_loss_fn, in_cfg, reg_loss_fn=None, **kwargs):
        epoch_mixup_states.append((epoch, bool(in_cfg.dataset.mixup_active)))
        return {"loss": 0.1}

    monkeypatch.setattr(advt, "distributed_init", lambda _args: None)
    monkeypatch.setattr(advt, "random_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "setup_logger", lambda *a, **k: _Logger())
    monkeypatch.setattr(advt, "resolve_amp", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "build_model", lambda *_a, **_k: torch.nn.Linear(8, 3))
    monkeypatch.setattr(advt, "optimizer_kwargs", lambda **_k: {})
    monkeypatch.setattr(advt, "create_optimizer_v2", lambda model, **_k: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(advt, "build_loss_scaler", lambda *_a, **_k: (contextlib.nullcontext, None))
    monkeypatch.setattr(advt, "build_dataset", _build_dataset_stub)
    monkeypatch.setattr(advt, "build_loss", lambda *_a, **_k: (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()))
    monkeypatch.setattr(advt, "create_scheduler_v2", lambda *_a, **_k: (_DummySched(), 2))
    monkeypatch.setattr(advt, "scheduler_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(advt, "CheckpointSaver", lambda *a, **k: _DummySaver())
    monkeypatch.setattr(advt, "validate", lambda *_a, **_k: {"top1": 1.0, "loss": 0.1})
    monkeypatch.setattr(advt, "update_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "resume_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "load_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "distribute_bn", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "_maybe_run_final_eval", lambda *_a, **_k: None)
    monkeypatch.setattr(advt, "train_one_epoch", _train_one_epoch_stub)
    monkeypatch.setattr(advt.wandb, "init", lambda **_k: None)
    monkeypatch.setattr(advt.wandb, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(advt.wandb.util, "generate_id", lambda: "test-run-id")

    advt.main(cfg)

    assert build_states == [True, False]
    assert epoch_mixup_states == [(0, True), (1, False)]


def test_convnext_small_v1_default_noise_mode_is_null():
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "model": OmegaConf.load("robust_training/configs/model/convnext_small_v1.yaml"),
            }
        ),
    )
    assert cfg.model.model == "convnext_small_v1"
    assert cfg.model.v1_noise_mode is None
    assert cfg.model.v1_gabor_seed is None


def test_v1_gabor_seed_defaults_to_run_seed():
    cfg = OmegaConf.create({"seed": 123456, "model": {"v1_gabor_seed": None}})

    assert resolve_v1_gabor_seed(cfg) == 123456


def test_v1_gabor_seed_explicit_override_is_preserved():
    cfg = OmegaConf.create({"seed": 123456, "model": {"v1_gabor_seed": 7}})

    assert resolve_v1_gabor_seed(cfg) == 7


def test_auto_experiment_name_uses_v1_attack_eps_for_feature_domain():
    cfg = OmegaConf.create(
        {
            "model": {"model": "convnext_small_v1", "v1_noise_mode": None, "experiment_num": 1, "experiment_name": None},
            "attacks": {
                "advtrain": True,
                "attack_domain": "v1_feature",
                "attack_norm": "linf",
                "attack_criterion": "madry",
                "attack_eps": 1.0,
                "v1_attack_eps": 16.0 / 255.0,
                "gradnorm": False,
            },
        }
    )

    experiment_name, group_name = _auto_experiment_name(cfg)

    assert experiment_name == "convnext_small_v1_clean_linf_16_init1"
    assert group_name == "v1feat_linf_madry"


def test_auto_experiment_name_marks_v1_noise_runs():
    cfg = OmegaConf.create(
        {
            "model": {"model": "convnext_small_v1", "v1_noise_mode": "neuronal", "experiment_num": 1, "experiment_name": None},
            "attacks": {
                "advtrain": False,
                "attack_domain": "pixel",
                "attack_norm": "linf",
                "attack_criterion": "madry",
                "attack_eps": 1.0,
                "v1_attack_eps": 16.0 / 255.0,
                "gradnorm": False,
            },
        }
    )

    experiment_name, group_name = _auto_experiment_name(cfg)

    assert experiment_name == "convnext_small_v1_noise_init1"
    assert group_name == "default"


@pytest.mark.parametrize(
    "penalty_norm,expected_name,expected_group",
    [
        ("l1", "convnext_small_gradnorm_l1_8_init2", "gradnorm_l1"),
        ("l2", "convnext_small_gradnorm_l2_8_init2", "gradnorm_l2"),
    ],
)
def test_auto_experiment_name_includes_gradnorm_penalty_norm(
    penalty_norm, expected_name, expected_group
):
    cfg = OmegaConf.create(
        {
            "model": {"model": "convnext_small", "v1_noise_mode": None, "experiment_num": 2, "experiment_name": None},
            "attacks": {
                "advtrain": False,
                "attack_domain": "pixel",
                "attack_norm": "linf",
                "attack_criterion": "madry",
                "attack_eps": 8.0 / 255.0,
                "v1_attack_eps": 1.0,
                "gradnorm": True,
                "gradnorm_penalty_norm": penalty_norm,
            },
        }
    )

    experiment_name, group_name = _auto_experiment_name(cfg)

    assert experiment_name == expected_name
    assert group_name == expected_group
