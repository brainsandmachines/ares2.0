"""Decomposing a run from its own config, for the models that are in no CSV.

The shapes here are copied from real dirs in the Slurm archive: the flat
``args.yaml`` of the older convnext_small runs (where ``experiment_name`` is null
and the name lives in ``output_dir``), and the nested ``hydra_config.yaml`` of the
newer ones.
"""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from model_store.config_reader import read_identity  # noqa: E402


def _write(tmp_path, name, payload):
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def write_args_yaml(tmp_path, name, **over):
    """Older generation: flat argparse dump, experiment_name is null."""
    model_dir = _write(tmp_path, name, None)
    payload = {
        "model": "convnext_small",
        "experiment_name": None,
        "output_dir": f"/home/ashtomer/projects/ares/results/models/{name}",
        "attack_norm": "l2",
        "attack_eps": 2,
        "attack_criterion": "mixup",
        "gradnorm": False,
        "trades_beta": 1.0,
    }
    payload.update(over)
    (model_dir / "args.yaml").write_text(yaml.safe_dump(payload))
    return model_dir


def write_hydra_config(tmp_path, name, **over):
    """Newer generation: nested Hydra config."""
    model_dir = _write(tmp_path, name, None)
    payload = {
        "model": {"model": "convnext_small", "experiment_name": name,
                  "v1_noise_mode": None},
        "attacks": {"attack_norm": "l2", "attack_eps": 2, "advtrain": True,
                    "attack_criterion": "trades", "gradnorm": False},
        "dataset": {"dvd": {"enabled": False, "variant": None}},
    }
    for key, value in over.items():
        section, _, field = key.partition(".")
        if field:
            payload.setdefault(section, {})[field] = value
        else:
            payload[section] = value
    (model_dir / "hydra_config.yaml").write_text(yaml.safe_dump(payload))
    return model_dir


class TestOldGeneration:
    def test_name_comes_from_output_dir_when_experiment_name_is_null(self, tmp_path):
        d = write_args_yaml(tmp_path, "convnext_small_l2_2_init1")
        ident = read_identity(d)
        assert ident.canonical == "convnext_small_l2_2_init1"
        assert ident.source == "config:args.yaml"

    def test_madry_from_mixup_criterion(self, tmp_path):
        ident = read_identity(write_args_yaml(tmp_path, "convnext_small_l2_2_init1"))
        assert (ident.arch, ident.protocol, ident.norm, ident.eps) == (
            "convnext_small", "madry", "l2", 2.0)
        assert ident.init == "1"
        assert not ident.legacy

    def test_trades_from_criterion(self, tmp_path):
        d = write_args_yaml(tmp_path, "convnext_small_l2trades_2_init1",
                            attack_criterion="trades")
        assert read_identity(d).protocol == "trades"

    def test_gradnorm_wins_over_the_norm(self, tmp_path):
        """gradnorm is a gradient penalty, not an eps-bounded protocol."""
        d = write_args_yaml(tmp_path, "convnext_small_gradnorm_l1_1_init1",
                            gradnorm=True, attack_norm="l1", attack_eps=1)
        assert read_identity(d).protocol == "gradnorm"

    def test_zero_eps_is_a_baseline_and_drops_norm_and_eps(self, tmp_path):
        d = write_args_yaml(tmp_path, "convnext_small_baseline_init1", attack_eps=0)
        ident = read_identity(d)
        assert ident.protocol == "baseline"
        assert ident.norm is None and ident.eps is None

    def test_experiment_relpath_omits_norm_for_a_baseline(self, tmp_path):
        d = write_args_yaml(tmp_path, "convnext_small_baseline_init1", attack_eps=0)
        assert read_identity(d).experiment_relpath == (
            "convnext_small/baseline/convnext_small_baseline_init1.pth.tar")


class TestNewGeneration:
    def test_nested_keys(self, tmp_path):
        d = write_hydra_config(tmp_path, "convnext_small_l2trades_2_init1")
        ident = read_identity(d)
        assert (ident.canonical, ident.arch, ident.protocol, ident.norm, ident.eps) == (
            "convnext_small_l2trades_2_init1", "convnext_small", "trades", "l2", 2.0)
        assert ident.source == "config:hydra_config.yaml"

    def test_dvd_without_adversarial_training_is_a_dvd_baseline(self, tmp_path):
        d = write_hydra_config(
            tmp_path, "convnext_small_dvd_b_baseline_init1",
            **{"dataset.dvd": {"enabled": True, "variant": "dvd-b"},
               "attacks.advtrain": False, "attacks.attack_eps": 0})
        assert read_identity(d).protocol == "dvd_b_baseline"

    def test_dvd_with_adversarial_training_keeps_its_protocol(self, tmp_path):
        """Matches how the CSVs name these: dvd_b_l2trades_* is protocol=trades."""
        d = write_hydra_config(
            tmp_path, "convnext_small_dvd_b_l2trades_2_init1",
            **{"dataset.dvd": {"enabled": True, "variant": "dvd-b"}})
        assert read_identity(d).protocol == "trades"

    def test_v1_arch_suffix_is_stripped_from_the_store_arch(self, tmp_path):
        d = write_hydra_config(tmp_path, "convnext_base_v1_l2_2_init1",
                               **{"model.model": "convnext_base_v1"})
        ident = read_identity(d)
        assert ident.arch == "convnext_base"
        assert ident.protocol == "v1"

    def test_v1_noise_mode_selects_the_noise_protocol(self, tmp_path):
        d = write_hydra_config(
            tmp_path, "convnext_base_v1_noise_init0",
            **{"model.model": "convnext_base_v1", "model.v1_noise_mode": "gaussian",
               "attacks.advtrain": False, "attacks.attack_eps": 0})
        assert read_identity(d).protocol == "v1_noise"

    def test_v1_clean_when_not_adversarial(self, tmp_path):
        d = write_hydra_config(
            tmp_path, "convnext_base_v1_clean_init0",
            **{"model.model": "convnext_base_v1", "attacks.advtrain": False,
               "attacks.attack_eps": 0})
        assert read_identity(d).protocol == "v1_clean"


class TestRefusals:
    def test_no_config_returns_none(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert read_identity(tmp_path / "empty") is None

    def test_arch_disagreement_is_flagged_not_guessed(self, tmp_path):
        """Config says one arch, the folder name says another -> _legacy/unparsed."""
        d = write_hydra_config(tmp_path, "convnext_small_l2_2_init1",
                               **{"model.model": "convnext_base"})
        ident = read_identity(d)
        assert ident.legacy and ident.notes == "unparsed"
        assert ident.experiment_relpath is None
        assert ident.store_relpath.startswith("_legacy/unparsed/")

    def test_odd_init_suffix_yields_no_init(self, tmp_path):
        d = write_args_yaml(tmp_path, "convnext_small_linf_16_initbad7",
                            attack_norm="linf", attack_eps=16)
        assert read_identity(d).init is None

    def test_hydra_config_preferred_over_args_yaml(self, tmp_path):
        """Both present: the newer, richer shape wins (CONFIG_CANDIDATES order)."""
        name = "convnext_small_l2_2_init1"
        write_args_yaml(tmp_path, name, attack_criterion="mixup")
        write_hydra_config(tmp_path, name, **{"attacks.attack_criterion": "trades"})
        assert read_identity(tmp_path / name).protocol == "trades"
