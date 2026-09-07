"""Naming rules -- the part that, if wrong, silently loses a model.

Two hazards are covered here specifically:

* the 31 leaf-name collisions between ``swin_b`` and ``vit_b_cvst`` (every
  ``swin_b`` leaf name is also a ``vit_b_cvst`` leaf name), which is why the
  canonical name must carry the arch;
* the ``checkpoint-N`` glob, which must match every digit count -- a 2- or
  3-digit epoch escaping the filter would put ~1 TB of intermediates into the
  curated tree.
"""

from __future__ import annotations

import pytest

from model_store.naming import (
    CKPT_FILE_FOR_KIND, KIND_FOR_CKPT_FILE, ModelIdentity, arch_from_name,
    canonical_name, flat_id, is_intermediate, is_keeper_checkpoint,
)


@pytest.mark.parametrize("raw,expected", [
    # nested Slurm lanes: the arch lives in the parent dir and must be pulled in
    ("swin_b/l2_4_init1", "swin_b_l2_4_init1"),
    ("vit_b_cvst/l2_4_init1", "vit_b_cvst_l2_4_init1"),
    ("vit_m_cvst/l2trades_2_init1", "vit_m_cvst_l2trades_2_init1"),
    # flat AIRCC / convnext: already self-describing, left alone
    ("convnext_base_l2_4_init1", "convnext_base_l2_4_init1"),
    ("convnext_small_dvd_b_l2trades_2_init1", "convnext_small_dvd_b_l2trades_2_init1"),
    # legacy containers whose leaf already names the arch: container dropped
    ("old_models/convnext_small_l2_1_init1", "convnext_small_l2_1_init1"),
    ("old_models/madry/convnext_small_eps-1_l2_seed-1", "convnext_small_eps-1_l2_seed-1"),
    # a container that is not an arch keeps its name, rather than inventing one
    ("vit_b_cvst_broken/l1_1_init1", "vit_b_cvst_broken_l1_1_init1"),
])
def test_canonical_name(raw, expected):
    assert canonical_name(raw) == expected


def test_swin_and_vit_leaves_collide_but_canonical_names_do_not():
    """The whole reason the arch has to be in the name."""
    leaves = ["l1_1_init1", "l2_4_init1", "linftrades_cont4to8_init1", "baseline_init1"]
    swin = {canonical_name(f"swin_b/{leaf}") for leaf in leaves}
    vit = {canonical_name(f"vit_b_cvst/{leaf}") for leaf in leaves}
    assert swin.isdisjoint(vit)
    # ... whereas the raw leaf names are identical, which a flat export would merge
    assert {leaf for leaf in leaves} == {leaf for leaf in leaves}
    assert len(swin) == len(vit) == len(leaves)


@pytest.mark.parametrize("arch", ["convnext_base", "convnext_small", "swin_b",
                                  "vit_b_cvst", "vit_m_cvst"])
def test_arch_from_name_roundtrip(arch):
    assert arch_from_name(f"{arch}_l2_4_init1") == arch


def test_arch_from_name_prefers_the_longest_match():
    # 'convnext_small' must not be shadowed by a shorter 'convnext_*' entry
    assert arch_from_name("convnext_small_l2_1_init1") == "convnext_small"
    assert arch_from_name("convnext_base_l2_1_init1") == "convnext_base"


def test_arch_from_name_unknown():
    assert arch_from_name("resnet50_l2_eps1") is None


@pytest.mark.parametrize("name", [
    "checkpoint-0.pth.tar", "checkpoint-1.pth.tar", "checkpoint-7.pth.tar",
    "checkpoint-12.pth.tar", "checkpoint-99.pth.tar", "checkpoint-100.pth.tar",
    "checkpoint-203.pth.tar", "checkpoint-1234.pth.tar", "tmp.pth.tar",
])
def test_intermediates_are_excluded_at_every_digit_count(name):
    assert is_intermediate(name)
    assert not is_keeper_checkpoint(name)


@pytest.mark.parametrize("name", [
    "last.pth.tar", "model_best.pth.tar", "model_best_adv.pth.tar",
    "epoch_0090.pth.tar", "epoch_0120.pth.tar",
])
def test_keepers_are_never_treated_as_intermediates(name):
    assert is_keeper_checkpoint(name)
    assert not is_intermediate(name)


def test_checkpoint_glob_does_not_eat_lookalikes():
    """The pattern is anchored, so these must survive."""
    for name in ("checkpoint.pth.tar", "my-checkpoint-1.pth.tar",
                 "checkpoint-best.pth.tar"):
        assert not is_intermediate(name), name


def test_kind_mapping_is_a_bijection():
    """The DB stores one of these basenames; a drift here mislabels every entry."""
    assert set(KIND_FOR_CKPT_FILE) == set(CKPT_FILE_FOR_KIND.values())
    for kind, filename in CKPT_FILE_FOR_KIND.items():
        assert KIND_FOR_CKPT_FILE[filename] == kind


def test_flat_id_matches_aa_sweep_convention():
    assert flat_id("vit_b_cvst/linf_1_init1") == "vit_b_cvst__linf_1_init1"
    assert flat_id("convnext_base_l2_4_init1") == "convnext_base_l2_4_init1"


class TestExperimentRelpath:
    def test_adversarial_includes_the_norm_level(self):
        ident = ModelIdentity("convnext_base_l2_4_init1", "convnext_base", "madry",
                              "l2", 4.0, "1", "csv")
        assert ident.experiment_relpath == (
            "convnext_base/madry/l2/convnext_base_l2_4_init1.pth.tar")

    def test_baseline_omits_the_norm_level(self):
        ident = ModelIdentity("convnext_base_baseline_init0", "convnext_base",
                              "baseline", None, None, "0", "csv")
        assert ident.experiment_relpath == (
            "convnext_base/baseline/convnext_base_baseline_init0.pth.tar")

    def test_nested_arch_carries_into_the_filename(self):
        ident = ModelIdentity("swin_b_l2_4_init1", "swin_b", "madry", "l2", 4.0,
                              "1", "csv")
        assert ident.experiment_relpath == "swin_b/madry/l2/swin_b_l2_4_init1.pth.tar"

    def test_legacy_gets_no_experiment_entry(self):
        ident = ModelIdentity("convnext_small_l2_1_init1", "convnext_small", "madry",
                              "l2", 1.0, "1", "config", legacy=True, notes="old_models")
        assert ident.experiment_relpath is None
        assert ident.store_relpath == "_legacy/old_models/convnext_small_l2_1_init1"

    def test_undecomposed_gets_no_experiment_entry(self):
        ident = ModelIdentity("weird_run", None, None, None, None, None, "name",
                              legacy=True, notes="unparsed")
        assert ident.experiment_relpath is None
        assert ident.store_relpath == "_legacy/unparsed/weird_run"

    def test_store_relpath_groups_by_arch(self):
        ident = ModelIdentity("swin_b_l2_4_init1", "swin_b", "madry", "l2", 4.0,
                              "1", "csv")
        assert ident.store_relpath == "swin_b/swin_b_l2_4_init1"


class TestLegacyKeying:
    """Two real models under old_models/ share a leaf name.

    ``old_models/madry/convnext_small_eps-1_linf_seed-1`` and
    ``old_models/gradnorm/convnext_small_eps-1_linf_seed-1`` differ only by the
    protocol subdir. Keying legacy models on the canonical leaf collapsed them
    into one record and dropped a model; keying on the archive path does not.
    """

    def _ident(self, relpath):
        return ModelIdentity(
            canonical=canonical_name(relpath), arch="convnext_small",
            protocol=None, norm=None, eps=None, init=None, source="config",
            legacy=True, notes="old_models", legacy_relpath=relpath)

    def test_same_leaf_under_different_protocols_gets_distinct_keys(self):
        a = self._ident("old_models/madry/convnext_small_eps-1_linf_seed-1")
        b = self._ident("old_models/gradnorm/convnext_small_eps-1_linf_seed-1")
        assert a.canonical == b.canonical           # the leaf really does collide
        assert a.record_key != b.record_key         # ... but the records do not
        assert a.store_relpath != b.store_relpath

    def test_legacy_store_path_mirrors_the_archive_path(self):
        ident = self._ident("old_models/madry/convnext_small_eps-1_linf_seed-1")
        assert ident.store_relpath == (
            "_legacy/old_models/madry/convnext_small_eps-1_linf_seed-1")

    def test_legacy_without_a_relpath_still_gets_a_bucket(self):
        ident = ModelIdentity("weird", None, None, None, None, None, "name",
                              legacy=True, notes="unparsed")
        assert ident.store_relpath == "_legacy/unparsed/weird"
        assert ident.record_key == "weird"

    def test_non_legacy_key_is_the_canonical_name(self):
        ident = ModelIdentity("swin_b_l2_4_init1", "swin_b", "madry", "l2", 4.0,
                              "1", "csv")
        assert ident.record_key == "swin_b_l2_4_init1"


class TestEpochReader:
    """The epoch decides conflicts, so reading it must be exact and cheap."""

    def test_reads_epoch_from_a_real_checkpoint_without_loading_weights(self, tmp_path):
        torch = pytest.importorskip("torch")
        import time
        from model_store.epochs import checkpoint_epoch

        ckpt = tmp_path / "last.pth.tar"
        torch.save({"epoch": 199, "arch": "convnext_base",
                    "state_dict": {"w": torch.zeros(4096, 512)}}, ckpt)

        start = time.time()
        assert checkpoint_epoch(ckpt) == 199
        # Metadata-only: must not be paying for the 8 MB of weights.
        assert time.time() - start < 1.0

    def test_missing_epoch_key_returns_none(self, tmp_path):
        torch = pytest.importorskip("torch")
        from model_store.epochs import checkpoint_epoch
        ckpt = tmp_path / "last.pth.tar"
        torch.save({"state_dict": {"w": torch.zeros(8)}}, ckpt)
        assert checkpoint_epoch(ckpt) is None

    def test_nested_epoch_is_found(self, tmp_path):
        torch = pytest.importorskip("torch")
        from model_store.epochs import checkpoint_epoch
        ckpt = tmp_path / "last.pth.tar"
        torch.save({"meta": {"epoch": 42}, "state_dict": {}}, ckpt)
        assert checkpoint_epoch(ckpt) == 42

    def test_unreadable_file_returns_none_rather_than_raising(self, tmp_path):
        from model_store.epochs import checkpoint_epoch
        bad = tmp_path / "truncated.pth.tar"
        bad.write_bytes(b"not a checkpoint")
        assert checkpoint_epoch(bad) is None
