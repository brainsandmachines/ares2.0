#!/usr/bin/env python3
"""Standalone GPU-memory cleanup validation (run manually, never in the manager).

Verifies that after a short training-like child process exits:
  1. the child is fully finished (returncode set),
  2. no leftover python compute process remains on the GPU,
  3. GPU free memory returns to ~the pre-launch level.

    srun --gpus=1 python aircc/aircc_job_manager/tests/test_gpu_cleanup.py

The parent intentionally never initializes CUDA, so its own measurement does not
perturb GPU memory. All GPU memory is measured via nvidia-smi.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Child: allocate a real model + optimizer on the GPU, do a few fwd/bwd steps,
# then exit so we can confirm the memory is released.
CHILD_CODE = r"""
import torch
dev = torch.device("cuda")
x = torch.randn(64, 3, 224, 224, device=dev)
try:
    from timm.models import create_model
    model = create_model("convnext_small", num_classes=1000).to(dev)
except Exception:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 256, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(256, 1000)).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(3):
    opt.zero_grad()
    out = model(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    opt.step()
torch.cuda.synchronize()
print("child: allocated", torch.cuda.memory_allocated() // (1024*1024), "MB")
"""


def _gpu_index() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip()
    if first.isdigit():
        return int(first)
    return 0


def free_mb(idx: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(idx)],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def compute_pids(idx: int) -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits", "-i", str(idx)],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [p.strip() for p in out.splitlines() if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU memory cleanup test.")
    ap.add_argument("--gpu-index", type=int, default=None)
    ap.add_argument("--tolerance-mb", type=int, default=500)
    ap.add_argument("--settle-seconds", type=float, default=5.0)
    args = ap.parse_args()

    idx = args.gpu_index if args.gpu_index is not None else _gpu_index()
    print(f"[gpu-cleanup] using GPU index {idx}, tolerance {args.tolerance_mb} MB")

    baseline = free_mb(idx)
    print(f"[gpu-cleanup] baseline free: {baseline} MB")

    proc = subprocess.run([sys.executable, "-c", CHILD_CODE])
    rc = proc.returncode
    time.sleep(args.settle_seconds)  # let the driver reclaim memory

    after = free_mb(idx)
    pids = compute_pids(idx)
    delta = baseline - after

    check1 = rc is not None
    check2 = len(pids) == 0
    check3 = abs(delta) <= args.tolerance_mb

    print(f"[gpu-cleanup] child returncode: {rc}  -> {'PASS' if check1 else 'FAIL'}")
    print(f"[gpu-cleanup] leftover compute pids: {pids or 'none'}  -> {'PASS' if check2 else 'FAIL'}")
    print(f"[gpu-cleanup] free after: {after} MB (delta {delta:+d} MB)  -> {'PASS' if check3 else 'FAIL'}")

    ok = check1 and check2 and check3
    print(f"[gpu-cleanup] {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
