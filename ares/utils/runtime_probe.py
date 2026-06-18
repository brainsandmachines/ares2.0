"""Lightweight per-phase timing for the runtime-arch-eval inspection.

Used only when ``cfg.runtime_probe`` is set. A :class:`PhaseTimer` is passed into
``train_one_epoch`` / ``validate``; it skips a warmup window of batches, times a
steady-state window, and extrapolates to a full epoch. Results are appended to a
shared CSV with an exclusive file lock so concurrent SLURM array tasks don't
corrupt each other's writes.
"""

import csv
import fcntl
import os
import time

import torch


# CSV column order (kept fixed so all array tasks agree on the header).
FIELDNAMES = [
    "protocol",
    "arch",
    "bsz",
    "optimizer",
    "compile",
    "fullepochruntime",
    "train_runtime",
    "eval_runtime",
]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhaseTimer:
    """Times a steady-state window of batches and extrapolates to a full epoch.

    Timing starts once ``batch_idx == warmup`` (so compile/cuDNN-autotune warmup
    on the first batches is excluded) and stops after ``measure`` batches. Call
    :meth:`maybe_start` at the top of each loop iteration and :meth:`maybe_stop`
    at the bottom; break the loop when it returns ``True``.
    """

    def __init__(self, warmup, measure, total_batches):
        self.warmup = int(warmup)
        self.measure = int(measure)
        self.total_batches = int(total_batches)
        self._t0 = None
        self.measured_s = 0.0
        self.measured_batches = 0

    def maybe_start(self, batch_idx):
        if batch_idx == self.warmup:
            _sync()
            self._t0 = time.time()

    def maybe_stop(self, batch_idx):
        if self._t0 is None:
            return False
        if batch_idx >= self.warmup + self.measure - 1:
            _sync()
            self.measured_s = time.time() - self._t0
            self.measured_batches = batch_idx - self.warmup + 1
            return True
        return False

    def full_estimate(self):
        """Estimated seconds for a full epoch over ``total_batches``."""
        if self.measured_batches <= 0:
            return 0.0
        return self.measured_s / self.measured_batches * self.total_batches


def append_runtime_row(csv_path, row):
    """Append one result row to the shared CSV under an exclusive lock."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    ordered = {k: row.get(k, "") for k in FIELDNAMES}
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            write_header = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(ordered)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
