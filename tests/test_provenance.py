"""Seed provenance (ares/utils/provenance.py) + per-epoch reseed resume equivalence."""

import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ares.utils import provenance
from ares.utils.dist import random_seed
from ares.utils.provenance import derive_epoch_seed


def test_derive_epoch_seed_is_deterministic_distinct_and_in_range():
    assert derive_epoch_seed(1, 5) == derive_epoch_seed(1, 5)
    seeds = {derive_epoch_seed(s, e) for s in range(4) for e in range(50)}
    assert len(seeds) == 200
    assert all(0 <= s < 2**31 for s in seeds)


def _write_log(run_dir, seeds):
    lines = []
    for i, s in enumerate(seeds):
        lines.append(f"[2026-05-0{i + 1} 10:00:00 ARES 2.0] INFO: Experiment: x\n")
        lines.append(f"[2026-05-0{i + 1} 10:00:01 ARES 2.0] INFO: Using seed: {s}\n")
    (run_dir / "log.txt").write_text("".join(lines))


def test_prior_launches_backfills_log_once(tmp_path):
    _write_log(tmp_path, [111, 222])
    prior = provenance.prior_launches(tmp_path)
    assert [p["seed"] for p in prior] == [111, 222]
    assert all(p["source"] == "backfill:log.txt" for p in prior)
    assert prior[0]["ts"] == "2026-05-01 10:00:01"

    # The jsonl now exists. A later launch reads it and doesn't re-backfill, even
    # though log.txt has grown (this launch's own seed line, etc.).
    with (tmp_path / "log.txt").open("a") as fh:
        fh.write("[2026-05-09 10:00:01 ARES 2.0] INFO: Using seed: 333\n")
    assert [p["seed"] for p in provenance.prior_launches(tmp_path)] == [111, 222]


def test_prior_launches_fresh_dir_is_empty(tmp_path):
    assert provenance.prior_launches(tmp_path) == []
    assert not (tmp_path / provenance.SEED_HISTORY_FILE).exists()


def test_record_launch_appends_and_returns_full_history(tmp_path):
    _write_log(tmp_path, [7])
    prior = provenance.prior_launches(tmp_path)
    hist = provenance.record_launch(
        tmp_path, prior=prior, seed=42, seed_source="config",
        start_epoch=3, total_epochs=10, epoch_reseed=True,
    )
    assert [h["seed"] for h in hist] == [7, 42]
    last = hist[-1]
    assert last["source"] == "config" and last["start_epoch"] == 3 and last["trains"] is True
    assert last["epoch_reseed"] is True
    env = last["env"]
    assert env["torch"] == torch.__version__ and "git" in env and "argv" in env

    on_disk = [json.loads(l) for l in (tmp_path / provenance.SEED_HISTORY_FILE).read_text().splitlines()]
    assert [h["seed"] for h in on_disk] == [7, 42]

    # eval-only relaunch (sjm-eval lane): start_epoch == total -> trains False
    hist2 = provenance.record_launch(
        tmp_path, prior=hist, seed=42, seed_source="config",
        start_epoch=10, total_epochs=10, epoch_reseed=True,
    )
    assert hist2[-1]["trains"] is False and len(hist2) == 3


def test_checkpoint_seed_history_is_compact(tmp_path):
    hist = provenance.record_launch(
        tmp_path, prior=[{"ts": "t0", "seed": 1, "source": "backfill:log.txt"}], seed=2,
        seed_source="drawn", start_epoch=0, total_epochs=5, epoch_reseed=False,
    )
    compact = provenance.checkpoint_seed_history(hist)
    assert [c["seed"] for c in compact] == [1, 2]
    assert compact[0]["git_commit"] is None  # backfilled entries have no env
    assert "env" not in compact[1] and "argv" not in json.dumps(compact)
    from omegaconf import OmegaConf  # must be embeddable in the checkpoint args
    assert OmegaConf.to_container(OmegaConf.create({"seed_history": compact}))["seed_history"] == compact


class _RandomAugDataset(Dataset):
    """Each item depends on the worker's torch RNG, like a random crop/flip."""

    def __len__(self):
        return 16

    def __getitem__(self, i):
        return torch.tensor([float(i)]), torch.rand(1)


def _epoch_stream(loader, seed, epoch):
    random_seed(derive_epoch_seed(seed, epoch), 0)
    idx, aug = [], []
    for x, a in loader:
        idx += x.flatten().tolist()
        aug += a.flatten().tolist()
    idx.append(float(torch.rand(1).cuda().item() if torch.cuda.is_available() else torch.rand(1).item()))
    return idx, aug


@pytest.mark.parametrize("num_workers", [0, 2])
def test_epoch_reseed_makes_resume_match_uninterrupted(num_workers):
    def loader():
        return DataLoader(_RandomAugDataset(), batch_size=4, shuffle=True, num_workers=num_workers)

    seed = 4
    # Uninterrupted: epochs 0, 1, 2 in one process, with RNG consumed in between
    # (model init, validation PGD, ...).
    random_seed(seed, 0)
    l = loader()
    uninterrupted = []
    for epoch in range(3):
        uninterrupted.append(_epoch_stream(l, seed, epoch))
        torch.rand(1000)  # validation / anything else between epochs

    # Requeued: a fresh process starts at epoch 2 with a different RNG history.
    random_seed(seed, 0)
    torch.rand(12345)
    resumed = _epoch_stream(loader(), seed, 2)

    assert resumed == uninterrupted[2]
    assert uninterrupted[1] != uninterrupted[2]  # epochs still differ from each other
