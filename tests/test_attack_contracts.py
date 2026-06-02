import contextlib
import math

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
            "continuation": OmegaConf.load(f"{root}/continuation/default.yaml"),
            "epsilon_schedule": OmegaConf.load(f"{root}/epsilon_schedule/default.yaml"),
            "checkpointing": OmegaConf.load(f"{root}/checkpointing/default.yaml"),
        }
    )


def test_attack_defaults_contract():
    cfg = OmegaConf.load("robust_training/configs/attacks/adv.yaml")

    assert cfg.attack_it == 3
    assert cfg.v1_attack_it == 3
    assert cfg.l1_step_mode == "l2_norm"


@pytest.mark.parametrize(
    "attack_domain,attack_norm,eps_field,eps_value,expected_step",
    [
        ("pixel", "l2", "attack_eps", 4.0, 8.0 / 3.0),
        ("v1_feature", "l2", "v1_attack_eps", 6.0, 12.0 / 3.0),
    ],
)
def test_l2_default_step_derivation_contract(
    monkeypatch,
    attack_domain,
    attack_norm,
    eps_field,
    eps_value,
    expected_step,
):
    cfg = _compose_base_cfg()
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "training": {"epochs": 1, "batch_size": 2, "model_ema": False},
                "attacks": {
                    "attack_domain": attack_domain,
                    "attack_norm": attack_norm,
                    "advtrain": True,
                    "attack_criterion": "madry",
                    eps_field: eps_value,
                    "attack_step": None,
                    "v1_attack_step": None,
                    "attack_it": 3,
                    "v1_attack_it": 3,
                },
                "model": {
                    "model": "convnext_small_v1" if attack_domain == "v1_feature" else "convnext_small",
                    "resume": "",
                    "experiment_num": 1,
                },
            }
        ),
    )
    cfg = OmegaConf.merge(
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

    recorded = {}

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
            return (0.0, 0)

    def _fake_loader():
        x = torch.zeros(2, 3, 8, 8)
        y = torch.zeros(2, dtype=torch.long)
        return [(x, y)]

    def _train_one_epoch_stub(epoch, model, loader, optimizer, train_loss_fn, in_cfg, **kwargs):
        recorded["attack_domain"] = in_cfg.attacks.attack_domain
        recorded["attack_step"] = in_cfg.attacks.get("attack_step", None)
        recorded["v1_attack_step"] = in_cfg.attacks.get("v1_attack_step", None)
        return {"loss": 0.1}

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

    if attack_domain == "pixel":
        assert math.isclose(recorded["attack_step"], expected_step, rel_tol=1e-8)
    else:
        assert math.isclose(recorded["v1_attack_step"], expected_step, rel_tol=1e-8)
