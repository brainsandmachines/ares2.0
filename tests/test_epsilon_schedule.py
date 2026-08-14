import math
from pathlib import Path

from omegaconf import OmegaConf

from ares.utils.continuation import (
    current_active_epsilon_user,
    maybe_save_best_adv_checkpoint,
    maybe_save_periodic_checkpoint,
    set_active_epsilon,
)
from ares.utils.epsilon_schedule import (
    L1_EPS_MULTIPLIER,
    V1_FEATURE_L2_EPS_MULTIPLIER,
    EpsilonSchedule,
    default_step_for_epsilon,
    denormalize_epsilon,
    normalize_epsilon,
)


def test_fixed_schedule_returns_target_every_epoch():
    schedule = EpsilonSchedule("fixed", source_epsilon=None, target_epsilon=8)

    assert schedule.value_for_epoch(0) == 8
    assert schedule.value_for_epoch(14) == 8


def test_warmup_ramp_fixed_schedule_values_are_one_based():
    schedule = EpsilonSchedule(
        "warmup_ramp_fixed",
        source_epsilon=4,
        target_epsilon=8,
        warmup_epochs=3,
        ramp_start_epoch=4,
        ramp_end_epoch=15,
        fixed_start_epoch=16,
    )

    assert schedule.value_for_epoch(0) == 4
    assert schedule.value_for_epoch(2) == 4
    assert schedule.value_for_epoch(3) == 4
    assert math.isclose(schedule.value_for_epoch(9), 4 + (10 - 4) / (15 - 4) * 4)
    assert schedule.value_for_epoch(14) == 8
    assert schedule.value_for_epoch(15) == 8


def test_norm_specific_epsilon_conversion():
    assert math.isclose(normalize_epsilon(8, "linf"), 8 / 255)
    assert normalize_epsilon(8, "l2") == 8
    assert normalize_epsilon(8, "l2", attack_domain="v1_feature") == 8 * V1_FEATURE_L2_EPS_MULTIPLIER
    assert normalize_epsilon(8, "l1") == 8 * L1_EPS_MULTIPLIER
    assert denormalize_epsilon(8 * L1_EPS_MULTIPLIER, "l1") == 8
    assert denormalize_epsilon(8 * V1_FEATURE_L2_EPS_MULTIPLIER, "l2", attack_domain="v1_feature") == 8


def test_default_step_tracks_epsilon_for_linf_and_l2():
    assert math.isclose(default_step_for_epsilon(8 / 255, "linf", 4), 2 / 255)
    assert default_step_for_epsilon(8, "l2", 4) == 4
    assert default_step_for_epsilon(8 * L1_EPS_MULTIPLIER, "l1", 4) is None


def test_v1_l2_continuation_uses_user_facing_epsilon():
    cfg = OmegaConf.create(
        {
            "runtime": {"active_attack_step_auto": True},
            "attacks": {
                "attack_domain": "v1_feature",
                "attack_norm": "l2",
                "v1_attack_eps": 0.0,
                "v1_attack_step": None,
                "v1_attack_it": 4,
            },
        }
    )

    eps_internal = set_active_epsilon(cfg, 2)

    assert eps_internal == 2 * V1_FEATURE_L2_EPS_MULTIPLIER
    assert cfg.attacks.v1_attack_eps == 20
    assert cfg.attacks.v1_attack_step == 10
    assert current_active_epsilon_user(cfg) == 2


def test_best_adv_checkpoint_tracks_advtop1(tmp_path):
    class _Saver:
        checkpoint_dir = str(tmp_path)
        extension = ".pth.tar"

        def _save(self, path, epoch, metric):
            Path(path).write_text(f"{epoch},{metric}")

    saver = _Saver()
    best = maybe_save_best_adv_checkpoint(saver, 0, {"advtop1": 10.0}, None, None)
    assert best == (10.0, 0)
    assert (tmp_path / "model_best_adv.pth.tar").read_text() == "0,10.0"

    best = maybe_save_best_adv_checkpoint(saver, 1, {"advtop1": 9.0}, *best)
    assert best == (10.0, 0)
    assert (tmp_path / "model_best_adv.pth.tar").read_text() == "0,10.0"

    best = maybe_save_best_adv_checkpoint(saver, 2, {"advtop1": 11.0}, *best)
    assert best == (11.0, 2)
    assert (tmp_path / "model_best_adv.pth.tar").read_text() == "2,11.0"


def test_periodic_checkpoint_saves_on_interval_only(tmp_path):
    class _Saver:
        checkpoint_dir = str(tmp_path)
        extension = ".pth.tar"

        def _save(self, path, epoch, metric):
            Path(path).write_text(f"{epoch},{metric}")

    saver = _Saver()
    # epoch is 0-based: epochs 0..18 are no-ops, epoch 19 = 20 completed epochs.
    for epoch in range(19):
        assert maybe_save_periodic_checkpoint(saver, epoch, 20, 1.0) is None
    assert not (tmp_path / "periodic").exists()

    path = maybe_save_periodic_checkpoint(saver, 19, 20, 42.0)
    assert path == str(tmp_path / "periodic" / "epoch_0020.pth.tar")
    assert Path(path).read_text() == "19,42.0"

    assert maybe_save_periodic_checkpoint(saver, 39, 20, 43.0) is not None
    assert (tmp_path / "periodic" / "epoch_0040.pth.tar").exists()

    # disabled / missing config
    assert maybe_save_periodic_checkpoint(saver, 19, 0, 1.0) is None
    assert maybe_save_periodic_checkpoint(saver, 19, None, 1.0) is None
    assert maybe_save_periodic_checkpoint(None, 19, 20, 1.0) is None
