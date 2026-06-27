"""
torch.compile debug script for B200 (sm_100).
Uses torchvision.models.convnext_small — no timm required.

Tests backends in order; prints PASS/FAIL for each.
Run with TORCHINDUCTOR_COMPILE_THREADS=1 to get the real crash
from the inductor subprocess instead of the vague "exited unexpectedly" message.
"""

import os
import subprocess
import sys
import traceback

import torch
import torchvision.models as tvm

DEVICE = "cuda"
BSZ = 16  # small enough to be fast, large enough to trigger real kernel paths


def print_env():
    print("=" * 60)
    print(f"Python   : {sys.version.split()[0]}")
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    try:
        import triton
        print(f"Triton   : {triton.__version__}")
    except ImportError:
        print("Triton   : not installed")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU      : {props.name}  sm_{props.major}{props.minor}  {props.total_memory // 1024**3} GB")
    else:
        print("GPU      : NONE")
    print(f"TORCHINDUCTOR_COMPILE_THREADS: {os.environ.get('TORCHINDUCTOR_COMPILE_THREADS', 'unset')}")
    print("=" * 60)
    print()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("nvidia-smi:", result.stdout.strip())
    print()


def make_model():
    model = tvm.convnext_small(weights=None).to(DEVICE)
    model.train()
    return model


def run_forward_backward(model):
    x = torch.randn(BSZ, 3, 224, 224, device=DEVICE)
    y = torch.randint(0, 1000, (BSZ,), device=DEVICE)
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    return loss.item()


def test(label, backend=None, extra_env=None):
    print(f"--- {label} ---")
    old_env = {}
    if extra_env:
        for k, v in extra_env.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v
    try:
        model = make_model()
        if backend is not None:
            model = torch.compile(model, backend=backend)
        else:
            model = torch.compile(model)
        loss = run_forward_backward(model)
        print(f"PASS  loss={loss:.4f}\n")
        return True
    except Exception:
        print("FAIL")
        traceback.print_exc()
        print()
        return False
    finally:
        for k, old in old_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def main():
    print_env()

    results = {}

    # 1. No compile baseline — must pass
    results["eager (no compile)"] = test(
        "eager (no compile)", backend="eager"
    )

    # 2. aot_eager — AOT autograd, no Triton kernel generation
    results["aot_eager"] = test(
        "aot_eager  [no Triton, should always pass]", backend="aot_eager"
    )

    # 3. cudagraphs — CUDA Graph capture, no Triton
    results["cudagraphs"] = test(
        "cudagraphs  [no Triton, should always pass]", backend="cudagraphs"
    )

    # 4. inductor (default) with TORCHINDUCTOR_COMPILE_THREADS=1
    #    Forces compilation in the main process → real crash stacktrace instead of
    #    "subprocess exited unexpectedly"
    results["inductor+threads=1"] = test(
        "inductor  TORCHINDUCTOR_COMPILE_THREADS=1  [main-process compile, shows real error]",
        backend=None,
        extra_env={"TORCHINDUCTOR_COMPILE_THREADS": "1"},
    )

    # 5. inductor default (subprocess mode) — the failing case
    results["inductor (default)"] = test(
        "inductor default  [this is the one that fails on B200 with CUDA 12.8]",
        backend=None,
    )

    print("=" * 60)
    print("SUMMARY")
    for label, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print("=" * 60)


if __name__ == "__main__":
    main()
