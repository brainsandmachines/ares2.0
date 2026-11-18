import json

import pytest

from experiments.geirhos_ood.models import ADV_JSON, BLACKBOX_JSON, load_models, select_models
from experiments.geirhos_ood.tasks import EXCLUDED_TASKS, TASKS, select_tasks

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
    assert [m.in_adv_json for m in tiny_models] == [True, False, True, False, True, True]
    other = tiny_models[3]
    assert other.kind == "last" and other.model_id == "convnext_base_l2_1_init1__last"


def test_unplaceable_model_raises(tiny_zoo, tmp_path):
    adv, bb, ref = tiny_zoo
    data = json.loads(adv.read_text())
    data["models"].append({**data["models"][1], "model_name": "convnext_base/mystery",
                           "checkpoint": str(tmp_path / "nowhere.pth.tar")})
    adv.write_text(json.dumps(data))
    with pytest.raises(KeyError, match="mystery"):
        load_models(adv, bb, ref)


def test_select_models(tiny_models):
    assert len(select_models(tiny_models, ["convnext_base_l2_1_init1"])) == 1
    with pytest.raises(KeyError):
        select_models(tiny_models, ["nope"])


@pytest.mark.skipif(not (ADV_JSON.exists() and BLACKBOX_JSON.exists()), reason="model zoo JSONs not present")
def test_real_zoo_has_199_unique_checkpoints():
    models = load_models()
    assert len(models) == 199
    assert len({m.checkpoint for m in models}) == 199
    assert models[0].family == "clean baselines"
    assert sum(m.family.endswith(("trades_l1", "trades_l2", "trades_linf")) and m.arch != "convnext_base"
               for m in models) == 30
