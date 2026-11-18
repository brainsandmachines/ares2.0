"""Live check that the OOD evaluation fits on the GPU next to another running job.

Opt-in (touches the real GPU and loads real checkpoints):

    GEIRHOS_GPU_COEXIST=1 [GEIRHOS_OTHER_PID=<pid>] [GEIRHOS_BUDGET_GB=8] [GEIRHOS_BATCH=64] \
        python -m pytest experiments/geirhos_ood/tests/test_gpu_coexist.py -q -s

1. Sample the other processes' GPU memory with nvidia-smi for GEIRHOS_SAMPLE_S seconds (their peak).
2. Under the budget, load one real checkpoint per arch (convnext_base, vit_b_cvst, swin_b) and push
   full batches through the exact inference path, sampling this process's own nvidia-smi usage.
3. Assert our peak + their peak + margin < total, and that the other job is still alive.
Measurements go to results/logs/gpu_coexist_<time>.json; they set the real run's budget/batch.
"""

import datetime as dt
import json
import os
import subprocess
import threading
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("GEIRHOS_GPU_COEXIST") != "1" or not torch.cuda.is_available(),
                                reason="opt-in live GPU test: set GEIRHOS_GPU_COEXIST=1")

MB = 1024 ** 2


def gpu_apps() -> dict[int, int]:
    """{pid: used MiB} for all compute processes on the GPU."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    apps = {}
    for line in out.strip().splitlines():
        pid, used = (s.strip() for s in line.split(","))
        apps[int(pid)] = int(used)
    return apps


def total_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class Sampler(threading.Thread):
    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period, self.stop, self.peaks = period, threading.Event(), {}

    def run(self):
        while not self.stop.is_set():
            for pid, used in gpu_apps().items():
                self.peaks[pid] = max(self.peaks.get(pid, 0), used)
            time.sleep(self.period)


def test_coexists_with_running_gpu_job():
    from data_analysis.autoattack_array_eval import load_model
    from experiments.geirhos_ood.evaluate import RESULTS_DIR, _decide
    from experiments.geirhos_ood.gpu import apply_budget
    from experiments.geirhos_ood.models import load_models

    me = os.getpid()
    other_pid = int(os.environ["GEIRHOS_OTHER_PID"]) if os.environ.get("GEIRHOS_OTHER_PID") else None
    budget = float(os.environ.get("GEIRHOS_BUDGET_GB", "8"))
    batch = int(os.environ.get("GEIRHOS_BATCH", "64"))
    margin_mib = int(float(os.environ.get("GEIRHOS_MARGIN_GB", "2")) * 1024)
    sample_s = float(os.environ.get("GEIRHOS_SAMPLE_S", "60"))
    if other_pid is not None:
        assert pid_alive(other_pid), f"other job {other_pid} is not running"

    before = Sampler()
    before.start()
    time.sleep(sample_s)
    before.stop.set()
    before.join()
    other_peak = sum(v for p, v in before.peaks.items() if p != me)

    device = torch.device("cuda")
    apply_budget(budget, device)
    during = Sampler()
    during.start()
    per_arch = {}
    models = load_models()
    for arch in ("convnext_base", "vit_b_cvst", "swin_b"):
        entry = next(m for m in models if m.arch == arch)
        torch.cuda.reset_peak_memory_stats(device)
        model, _cfg, _key = load_model(entry.checkpoint, device)
        x = torch.rand(batch, 3, 224, 224)
        for _ in range(5):
            out = _decide(model, x, device)
        assert out.shape == (batch, 2, 16)
        torch.cuda.synchronize()
        per_arch[arch] = {"checkpoint": str(entry.checkpoint),
                          "torch_max_reserved_mib": torch.cuda.max_memory_reserved(device) // MB}
        del model
        torch.cuda.empty_cache()
    time.sleep(2)
    during.stop.set()
    during.join()

    our_peak = during.peaks.get(me, 0)
    other_peak_during = sum(v for p, v in during.peaks.items() if p != me)
    total = total_mib()
    other_peak = max(other_peak, other_peak_during)
    report = {
        "time": dt.datetime.now().isoformat(timespec="seconds"), "total_mib": total, "budget_gb": budget,
        "batch": batch, "our_peak_mib_nvidia_smi": our_peak, "other_peak_mib": other_peak,
        "other_pid": other_pid, "margin_mib": margin_mib, "per_arch": per_arch,
        "headroom_mib": total - our_peak - other_peak,
    }
    (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "logs" / f"gpu_coexist_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    assert our_peak > 0, "nvidia-smi never saw this process on the GPU"
    assert our_peak + other_peak + margin_mib < total, report
    if other_pid is not None:
        assert pid_alive(other_pid), "the other GPU job died during the test"
