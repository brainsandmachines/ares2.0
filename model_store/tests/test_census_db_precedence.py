"""Which DB row a model gets when both job DBs name it.

AIRCC froze several runs as ``running`` that Slurm later finished and blessed. When
the last row read simply won, the stale AIRCC row (read after sjm) hid the Slurm
blessing and the model never got a link. Both row orders are tested, so the rule
holds regardless of how ``load_db_rows`` concatenates the DBs.
"""

from __future__ import annotations

import pytest

from model_store.census import ModelRecord, apply_db_rows
from model_store.naming import ModelIdentity

NAME = "convnext_base_dvd_b_linftrades_2_init0"


def _records():
    ident = ModelIdentity(canonical=NAME, arch="convnext_base", protocol="trades",
                          norm="linf", eps=2.0, init="0", source="csv")
    return {NAME: ModelRecord(identity=ident)}


def _row(source, status, best=None, score=None):
    path = f"/home/ashtomer/projects/ares/results/models/{NAME}/{best}" if best else None
    return {"model_name": NAME, "status": status, "best_checkpoint": path,
            "best_score": score, "_source": source}


SJM_DONE = _row("sjm", "finished", "model_best.pth.tar", 43.9)
SJM_PENDING = _row("sjm", "pending")
AIRCC_DONE = _row("aircc", "finished", "last.pth.tar", 6.3)
AIRCC_RUNNING = _row("aircc", "running")


def _merged(rows):
    records = _records()
    apply_db_rows(records, rows, {})
    rec = records[NAME]
    return rec.db_source, rec.db_status, rec.best_basename, rec.best_score


@pytest.mark.parametrize("rows", [[SJM_DONE, AIRCC_RUNNING], [AIRCC_RUNNING, SJM_DONE]])
def test_slurm_blessing_beats_a_stale_aircc_row(rows):
    assert _merged(rows) == ("sjm", "finished", "model_best.pth.tar", 43.9)


@pytest.mark.parametrize("rows", [[SJM_PENDING, AIRCC_DONE], [AIRCC_DONE, SJM_PENDING]])
def test_aircc_blessing_survives_an_unblessed_sjm_row(rows):
    assert _merged(rows) == ("aircc", "finished", "last.pth.tar", 6.3)


@pytest.mark.parametrize("rows", [[SJM_DONE, AIRCC_DONE], [AIRCC_DONE, SJM_DONE]])
def test_both_blessed_the_live_sjm_row_wins(rows):
    assert _merged(rows)[0] == "sjm"


@pytest.mark.parametrize("rows", [[SJM_PENDING, AIRCC_RUNNING], [AIRCC_RUNNING, SJM_PENDING]])
def test_neither_blessed_the_live_sjm_row_wins(rows):
    assert _merged(rows)[:2] == ("sjm", "pending")


def test_a_single_row_is_taken_as_is():
    assert _merged([AIRCC_RUNNING]) == ("aircc", "running", None, None)
