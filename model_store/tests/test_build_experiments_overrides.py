"""Blessing overrides and excludes -- the hand corrections the Monday rebuild must not revert.

``build_experiments`` regenerates every symlink from the job DBs each week, so a
link fixed by hand lasts only until the next run. Rows in the checked-in CSVs are
re-applied on every rebuild; these tests pin that a checkpoint override beats the
DB record, that a rename publishes the source checkpoint under the new name and
drops the model that had it, that excludes skip what they match, that a malformed
row fails loudly instead of silently falling back, and that the real CSVs parse.
"""

from __future__ import annotations

import pytest

from model_store.build_experiments import (
    EXCLUDES_CSV, OVERRIDES_CSV, Override, load_excludes, load_overrides, plan,
)
from model_store.census import ModelRecord
from model_store.naming import CKPT_FILE_FOR_KIND, ModelIdentity

NAME = "convnext_base_gradnorm_l1_2_init0"
DB_SCORE = 54.39453125
ORIGINAL = "convnext_base_l1_cont4to6_init0"
RERUN = "convnext_base_l1_cont4to6_lr1e4_pgd5_init0"


def _record(store, name=NAME, protocol="gradnorm", norm="linf", eps=2.0,
            best="last.pth.tar", score=DB_SCORE, files=tuple(CKPT_FILE_FOR_KIND.values())):
    ident = ModelIdentity(canonical=name, arch="convnext_base", protocol=protocol,
                          norm=norm, eps=eps, init="0", source="csv")
    model_dir = store / ident.store_relpath
    model_dir.mkdir(parents=True)
    for basename in files:
        (model_dir / basename).write_bytes(b"")
    return ModelRecord(
        identity=ident, dirs={"data4t-aircc": model_dir}, db_source="aircc",
        db_status="finished",
        best_checkpoint=f"/shared/cycle2_bgu_golan_prj/ashtomer/ares/results/models/{name}/{best}",
        best_score=score,
    )


def _pair(store, rerun_files=tuple(CKPT_FILE_FOR_KIND.values())):
    return {
        ORIGINAL: _record(store, ORIGINAL, "madry", "l1", 6.0, score=24.0),
        RERUN: _record(store, RERUN, "madry", "l1", 6.0, score=27.8, files=rerun_files),
    }


# --- checkpoint overrides ---------------------------------------------------
def test_without_override_the_db_record_wins(tmp_path):
    entries, gaps = plan({NAME: _record(tmp_path)}, tmp_path)
    (e,) = entries
    assert (e.kind, e.rule, e.target.name) == ("last", "db", "last.pth.tar")
    assert not gaps


def test_override_beats_the_db_record(tmp_path):
    ov = {NAME: Override("model_best_adv.pth.tar", None, 54.0, "advbest scores higher")}
    entries, gaps = plan({NAME: _record(tmp_path)}, tmp_path, overrides=ov)
    (e,) = entries
    assert (e.kind, e.rule, e.target.name) == ("advbest", "override", "model_best_adv.pth.tar")
    assert e.note == "advbest scores higher"
    assert e.best_score == 54.0
    assert not gaps


def test_override_without_score_keeps_the_db_score(tmp_path):
    ov = {NAME: Override("model_best_adv.pth.tar", None, None, "why")}
    (e,), _ = plan({NAME: _record(tmp_path)}, tmp_path, overrides=ov)
    assert e.best_score == DB_SCORE


def test_unmatched_override_is_reported(tmp_path):
    ov = {"convnext_base_no_such_model": Override("last.pth.tar", None, None, "why")}
    _, gaps = plan({NAME: _record(tmp_path)}, tmp_path, overrides=ov)
    assert [(g.canonical, g.why) for g in gaps] == [
        ("convnext_base_no_such_model", "override-unused")]


def test_checkpoint_override_publishes_a_failed_row_known_only_to_the_store(tmp_path):
    # The real case: the row ended 'failed' (so is_trained is False) and the model
    # reached models/ by backfill, so no census root holds a dir for it.
    name = "convnext_base_dvd_b_l2trades_1_init1"
    rec = _record(tmp_path, name, "trades", "l2", 1.0, score=None)
    rec.best_checkpoint, rec.db_status, rec.dirs = None, "failed", {}
    assert not rec.is_trained
    ov = {name: Override("model_best.pth.tar", None, 59.47, "crashed after finishing")}
    entries, gaps = plan({name: rec}, tmp_path, overrides=ov)
    (e,) = entries
    assert (e.kind, e.rule, e.best_score) == ("best", "override", 59.47)
    assert not gaps


# --- renames ----------------------------------------------------------------
def test_rename_alone_does_not_bless(tmp_path):
    rec = _record(tmp_path, RERUN, "madry", "l1", 6.0)
    rec.best_checkpoint = None
    ov = {RERUN: Override(None, ORIGINAL, None, "why")}
    entries, gaps = plan({RERUN: rec}, tmp_path, overrides=ov)
    assert not entries
    assert {g.why for g in gaps} == {"not-db-blessed", "override-unused"}


def test_rename_publishes_the_rerun_under_the_original_name(tmp_path):
    ov = {RERUN: Override(None, ORIGINAL, None, "rerun is canonical")}
    entries, gaps = plan(_pair(tmp_path), tmp_path, overrides=ov)
    (e,) = entries
    assert e.canonical == ORIGINAL
    assert e.relpath == f"convnext_base/madry/l1/{ORIGINAL}.pth.tar"
    assert e.target == tmp_path / "convnext_base" / RERUN / "last.pth.tar"
    assert (e.source, e.rule, e.best_score) == (RERUN, "db", 27.8)
    assert [(g.canonical, g.why) for g in gaps] == [(ORIGINAL, "superseded")]


def test_rename_that_fails_to_resolve_keeps_the_original(tmp_path):
    ov = {RERUN: Override(None, ORIGINAL, None, "rerun is canonical")}
    entries, gaps = plan(_pair(tmp_path, rerun_files=()), tmp_path, overrides=ov)
    (e,) = entries
    assert (e.canonical, e.source) == (ORIGINAL, "")
    assert {g.why for g in gaps} == {"target-missing", "override-unused"}


# --- excludes ---------------------------------------------------------------
def test_exclude_skips_matching_relpaths_in_any_protocol(tmp_path):
    records = {
        "a": _record(tmp_path, "convnext_base_dvd_b_linf_cont4to6_init0_resetepoch", "madry", "linf", 6.0),
        "b": _record(tmp_path, "convnext_base_dvd_b_l1trades_cont4to6_init0_resetepoch", "trades", "l1", 6.0),
        "c": _record(tmp_path, "convnext_base_dvd_b_linf_cont4to6_init0_contepoch", "madry", "linf", 6.0),
    }
    entries, gaps = plan(records, tmp_path, excludes={"*_resetepoch.pth.tar": "unused"})
    assert [e.canonical for e in entries] == ["convnext_base_dvd_b_linf_cont4to6_init0_contepoch"]
    assert sorted(g.why for g in gaps) == ["excluded", "excluded"]


def test_unmatched_exclude_is_reported(tmp_path):
    _, gaps = plan({NAME: _record(tmp_path)}, tmp_path, excludes={"*_nothing.pth.tar": "why"})
    assert [(g.canonical, g.why) for g in gaps] == [("*_nothing.pth.tar", "exclude-unused")]


# --- loading ----------------------------------------------------------------
HEADER = "model,checkpoint,publish_as,score,reason\n"


@pytest.mark.parametrize("body", [
    f"{NAME},checkpoint-120.pth.tar,,,why\n",                          # not a keeper
    f"{NAME},model_best_adv.pth.tar,,,\n",                             # no reason
    f"{NAME},last.pth.tar,,,a\n{NAME},model_best.pth.tar,,,b\n",       # duplicate model
    f"{NAME},,,,why\n",                                                # sets nothing
    f"{NAME},,{NAME},,why\n",                                          # renamed to itself
    f"a_init0,,x_init0,,why\nb_init0,,x_init0,,why\n",                 # two onto one name
    f"a_init0,,b_init0,,why\nb_init0,,c_init0,,why\n",                 # chain
])
def test_malformed_overrides_fail_loudly(tmp_path, body):
    path = tmp_path / "overrides.csv"
    path.write_text(HEADER + body)
    with pytest.raises(ValueError):
        load_overrides(path)


@pytest.mark.parametrize("body", [
    "*_resetepoch.pth.tar,\n",                                         # no reason
    "*_resetepoch.pth.tar,a\n*_resetepoch.pth.tar,b\n",                # duplicate
])
def test_malformed_excludes_fail_loudly(tmp_path, body):
    path = tmp_path / "excludes.csv"
    path.write_text("pattern,reason\n" + body)
    with pytest.raises(ValueError):
        load_excludes(path)


def test_missing_files_are_empty(tmp_path):
    assert load_overrides(tmp_path / "absent.csv") == {}
    assert load_excludes(tmp_path / "absent.csv") == {}


def test_checked_in_files_parse():
    ov = load_overrides(OVERRIDES_CSV)
    assert ov[NAME].checkpoint == "model_best_adv.pth.tar"
    assert ov[RERUN].publish_as == ORIGINAL
    assert "*_resetepoch.pth.tar" in load_excludes(EXCLUDES_CSV)
