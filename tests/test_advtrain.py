import argparse
import contextlib
import math
import types

import pytest
import torch
from omegaconf import OmegaConf

import robust_training.adversarial_training as advt


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
        }
    )


def _sbatch_mode_from_jobname(jobname):
    attack_norm = "linf"
    advtrain = False
    gradnorm = False
    criterion = "madry"

    if "gradnorm" in jobname:
        gradnorm = True
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
        "attack_criterion": criterion,
    }


@pytest.mark.parametrize(
    "jobname,eps,exp_num,expected",
    [
        (
            "linf_16_init1",
            16,
            1,
            {"attack_norm": "linf", "advtrain": True, "gradnorm": False, "attack_criterion": "madry"},
        ),
        (
            "l2_16_trades_init2",
            16,
            2,
            {"attack_norm": "l2", "advtrain": True, "gradnorm": False, "attack_criterion": "trades"},
        ),
        (
            "l1_8_init3",
            8,
            3,
            {"attack_norm": "l1", "advtrain": True, "gradnorm": False, "attack_criterion": "madry"},
        ),
        (
            "gradnorm_16_init4",
            16,
            4,
            {"attack_norm": "linf", "advtrain": False, "gradnorm": True, "attack_criterion": "madry"},
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
                    "attack_criterion": parsed["attack_criterion"],
                },
                "model": {"experiment_num": int(exp_num)},
            }
        ),
    )

    merged = advt._merge_groups_for_hydra(cfg)
    args = argparse.Namespace(**merged)

    assert args.attack_eps == float(eps)
    assert args.experiment_num == exp_num
    assert args.attack_norm == expected["attack_norm"]
    assert args.advtrain is expected["advtrain"]
    assert args.gradnorm is expected["gradnorm"]
    assert args.attack_criterion == expected["attack_criterion"]


@pytest.mark.parametrize(
    "mode,eps,criterion,expect_regnorm,expected_eps,expected_random_start",
    [
        ("linf", 16.0, "madry", False, 16.0 / 255.0, False),
        ("l1", 2.0, "madry", False, 2.0 * 255.0 / 2.0, False),
        ("l2", 4.0, "trades", False, 4.0, True),
        ("gradnorm", 8.0, "madry", True, 8.0 / 255.0, False),
    ],
)
def test_main_one_epoch_modes(monkeypatch, mode, eps, criterion, expect_regnorm, expected_eps, expected_random_start):
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
                    "attack_criterion": criterion,
                },
                "model": {"experiment_num": 1, "resume": ""},
            }
        ),
    )
    args = argparse.Namespace(**advt._merge_groups_for_hydra(cfg))

    args.train_dir = "/tmp/train"
    args.eval_dir = "/tmp/val"
    args.output_dir = "/tmp/out"
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.device_id = 0
    args.final_eval = False

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

    def _train_one_epoch_stub(epoch, model, loader, optimizer, train_loss_fn, in_args, reg_loss_fn=None, **kwargs):
        train_call["epoch"] = epoch
        train_call["attack_eps"] = in_args.attack_eps
        train_call["random_start"] = getattr(in_args, "random_start", False)
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

    advt.main(args)

    assert train_call["epoch"] == 0
    assert math.isclose(train_call["attack_eps"], expected_eps, rel_tol=1e-8)
    assert train_call["random_start"] is expected_random_start
    if expect_regnorm:
        assert isinstance(train_call["reg_loss_fn"], advt.DBP)
        assert train_call["gradnorm_start_epoch"] == args.alpha_start_epoch
    else:
        assert train_call["reg_loss_fn"] is None


def test_final_eval_defaults_are_pgd_without_autoattack_with_plots():
    cfg = _compose_base_cfg()
    merged = advt._merge_groups_for_hydra(cfg)

    assert merged["final_eval"] is True
    assert merged["final_eval_pgd"] is True
    assert merged["final_eval_autoattack"] is False
    assert merged["final_eval_plots"] is True


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

    args = argparse.Namespace(
        final_eval=True,
        final_eval_autoattack=False,
        final_eval_pgd=True,
        final_eval_ckpt_name="model_best.pth.tar",
        final_eval_val_dir="",
        eval_dir="/tmp/val",
        final_eval_out_dir="",
        final_eval_aa_batch_size=None,
        batch_size=2,
        final_eval_aa_norm=None,
        final_eval_aa_eps=None,
        final_eval_aa_max_batches=None,
        final_eval_pgd_batch_size=None,
        final_eval_pgd_eps="0.5,1,2,4,8,16",
        final_eval_pgd_norms="linf,l2,l1",
        final_eval_pgd_attack_steps=10,
        final_eval_pgd_max_batches=None,
        final_eval_plots=True,
        final_eval_plot_x_col="epsilon_input",
        final_eval_num_workers=8,
    )

    advt._maybe_run_final_eval(args, str(output_dir), _Logger())

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["aa"] is False
    assert kwargs["pgd"] is True
    assert kwargs["plots"] is True
