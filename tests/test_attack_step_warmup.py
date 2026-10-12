"""The training PGD step ramps over epochs 0-4 unless attacks.disable_attack_step_warmup is set.

A continuation restarts its epoch counter at 0, so with the ramp its first epoch trains at
step 0 (clean images). The flag gives it the full step from epoch 0.
"""
import pytest
import torch
from omegaconf import OmegaConf

import ares.utils.train_loop as train_loop


class _HostTensor(torch.Tensor):
    """A CPU tensor whose .cuda() is a no-op, so train_one_epoch runs without a GPU."""

    def cuda(self, *args, **kwargs):
        return self


class _Stop(Exception):
    pass


def _cfg(disable_warmup):
    attacks = {
        "advtrain": True, "attack_criterion": "madry", "attack_domain": "pixel",
        "attack_eps": 4.0, "attack_step": 0.5, "attack_it": 7,
    }
    if disable_warmup is not None:
        attacks["disable_attack_step_warmup"] = disable_warmup
    return OmegaConf.create({
        "attacks": attacks,
        "training": {"epochs": 40},
        "dataset": {},
        "optimizer": {},
        "dist": {"rank": 0, "distributed": False, "world_size": 1},
        "lr_scheduler": {"cooldown_epochs": 0},
        "model": {"channels_last": False},
    })


def _first_attack_step(monkeypatch, epoch, disable_warmup):
    """Run train_one_epoch until its first adversarial-example call; return that call's step."""
    seen = {}

    def _fake_adv_generator(cfg, images, target, model, eps, attack_steps, attack_lr, random_start, **_k):
        seen["step"] = attack_lr
        seen["it"] = attack_steps
        raise _Stop

    monkeypatch.setattr(train_loop, "adv_generator", _fake_adv_generator)
    monkeypatch.setattr(train_loop, "build_dvd_state", lambda *_a, **_k: None)
    images = torch.zeros(2, 3, 4, 4).as_subclass(_HostTensor)
    target = torch.zeros(2, dtype=torch.long).as_subclass(_HostTensor)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(_Stop):
        train_loop.train_one_epoch(
            epoch, model, [(images, target)], optimizer, torch.nn.CrossEntropyLoss(),
            _cfg(disable_warmup),
        )
    assert seen["it"] == 7
    return seen["step"]


@pytest.mark.parametrize("disable_warmup", [None, False])
def test_default_ramps_the_step_from_zero(monkeypatch, disable_warmup):
    assert _first_attack_step(monkeypatch, 0, disable_warmup) == 0.0
    assert _first_attack_step(monkeypatch, 2, disable_warmup) == pytest.approx(0.5 * 2 / 5)
    assert _first_attack_step(monkeypatch, 5, disable_warmup) == pytest.approx(0.5)


def test_disabled_warmup_uses_the_full_step_from_epoch_zero(monkeypatch):
    for epoch in (0, 1, 4, 10):
        assert _first_attack_step(monkeypatch, epoch, True) == pytest.approx(0.5)
