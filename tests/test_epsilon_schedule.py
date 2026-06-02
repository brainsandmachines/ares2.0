import math
from pathlib import Path

from ares.utils.continuation import maybe_save_best_adv_checkpoint
from ares.utils.epsilon_schedule import EpsilonSchedule, default_step_for_epsilon, normalize_epsilon


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
    assert normalize_epsilon(8, "l1") == 8 * 255 / 2


def test_default_step_tracks_epsilon_for_linf_and_l2():
    assert math.isclose(default_step_for_epsilon(8 / 255, "linf", 4), 2 / 255)
    assert default_step_for_epsilon(8, "l2", 4) == 4
    assert default_step_for_epsilon(8 * 255 / 2, "l1", 4) is None


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
