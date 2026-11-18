import csv
import json

import pytest

from experiments.geirhos_ood.models import ZOO_FILES, family_of, load_models, select_models
from experiments.geirhos_ood.tasks import EXCLUDED_TASKS, TASKS, select_tasks
from model_store.hf_export import MIRROR_ROOT

UPSTREAM_16 = {"colour", "contrast", "high-pass", "low-pass", "phase-scrambling", "power-equalisation",
               "false-colour", "rotation", "eidolonI", "eidolonII", "eidolonIII", "uniform-noise",
               "sketch", "stylized", "edge", "silhouette"}


def test_exactly_the_16_upstream_tasks():
    assert set(TASKS) == UPSTREAM_16
    assert len(TASKS) == 16
    assert sum(t.kind == "distortion" for t in TASKS.values()) == 12


def test_cue_conflict_is_excluded_everywhere():
    assert "cue-conflict" in EXCLUDED_TASKS and "cue-conflict" not in TASKS
    with pytest.raises(ValueError, match="already scored"):
        select_tasks(["cue-conflict"])
    with pytest.raises(ValueError, match="Unknown"):
        select_tasks(["shape-bias"])


def test_scored_conditions_drop_upstream_exclusions():
    assert TASKS["contrast"].scored_conditions == ("c50", "c30", "c15", "c10", "c05")
    assert TASKS["false-colour"].scored_conditions == ("False",)
    assert TASKS["sketch"].scored_conditions is None
    for t in TASKS.values():
        assert set(t.excluded_conditions) <= set(t.data_conditions)


def test_family_mapping_and_order(tiny_models):
    fams = [m.family for m in tiny_models]
    assert fams == ["clean baselines", "clean baselines", "convnext_base madry_l2", "convnext_base madry_l2",
                    "vit_b_cvst madry_l2", "vit_b_cvst trades_l2"]
    assert [m.key for m in tiny_models[:2]] == ["convnext_base_baseline_init0", "convnext_base_baseline_init1"]
    assert [m.generator for m in tiny_models] == [True, False, True, False, True, False]
    other = tiny_models[3]
    assert other.kind == "last" and other.model_id == "convnext_base_l2_1_init1__last"
    assert {m.source_json for m in tiny_models} == {"models_main.json", "models_other.json"}


def test_hf_excluded_and_resetepoch_runs_are_dropped(tiny_models):
    keys = {m.key for m in tiny_models}
    assert "convnext_base_l2_2_init0" not in keys
    assert not any("_resetepoch" in k for k in keys)
    assert len(tiny_models) == 6


def test_family_names():
    def fam(**kw):
        return family_of({"model_name": "x", "arch": "convnext_base", **kw})[0]

    assert fam(protocol="dvd_b", norm="linftrades") == "convnext_base dvdb_trades_linf"
    assert fam(protocol="v1", norm="l2") == "convnext_base v1_madry_l2"
    assert fam(protocol="gradnorm", norm="l1") == "convnext_base gradnorm_l1"
    assert fam(protocol="v1_baseline", norm=None) == "clean baselines"


def test_unplaceable_model_raises(tiny_zoo, tmp_path):
    files, excludes = tiny_zoo
    data = json.loads(files[0].read_text())
    data["models"].append({**data["models"][0], "model_name": "convnext_base/mystery", "protocol": "mystery",
                           "checkpoint": str(tmp_path / "nowhere.pth.tar")})
    files[0].write_text(json.dumps(data))
    with pytest.raises(KeyError, match="mystery"):
        load_models(files, excludes)


def test_duplicate_checkpoint_raises(tiny_zoo):
    files, excludes = tiny_zoo
    data = json.loads(files[0].read_text())
    data["models"].append({**data["models"][0], "model_name": "convnext_base/l2_1_init1_again"})
    files[0].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="listed twice"):
        load_models(files, excludes)


def test_select_models(tiny_models):
    assert len(select_models(tiny_models, ["convnext_base_l2_1_init1"])) == 1
    with pytest.raises(KeyError):
        select_models(tiny_models, ["nope"])


@pytest.mark.skipif(not all(p.exists() for p in ZOO_FILES), reason="model zoo JSONs not present")
def test_real_zoo_is_the_hf_published_set():
    models = load_models()
    assert len(models) == len({m.checkpoint for m in models}) == len({m.key for m in models})
    assert models[0].family == "clean baselines"
    assert not any("_resetepoch" in m.key for m in models)
    manifest = MIRROR_ROOT / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="") as fh:
            assert {m.key for m in models} == {r["name"] for r in csv.DictReader(fh)}
