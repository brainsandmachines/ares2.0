"""The Botero lane's worker: run queued AutoAttack units on the local 4090, one at a time.

Invoked every 10 minutes from cron through ``scripts/aa_sweep_botero_runner.sh``, which holds a
``flock`` for the whole -- possibly multi-day -- lifetime of a tick. That lock is what makes the
lane serial: a tick that lands while a job is running exits immediately.

A tick:

    reap dead runners -> GPU free? -> claim oldest queued -> run the engine -> echo to QNAP -> repeat

The GPU gate is deliberately conservative: *any* foreign CUDA compute process (the ad-hoc epoch-90
eval, an interactive notebook, a training run) defers the tick. Botero is a workstation first, and
a sweep cell is worth less than whatever a human is doing on the card.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from aa_sweep import botero, config


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[aa_botero] {_now()} {msg}", flush=True)


def gpu_users(run=subprocess.run) -> list[tuple[int, str]]:
    """``(pid, used_memory)`` for every CUDA compute process currently on the GPU.

    Graphics contexts (Xorg) do not appear in ``--query-compute-apps``, which is what we want: a
    desktop session is not a reason to defer a sweep.
    """
    proc = run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed rc={proc.returncode}: {proc.stderr.strip()}")
    users: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        users.append((int(parts[0]), parts[1]))
    return users


def gpu_blockers(own_pids: set[int], run=subprocess.run) -> list[tuple[int, str]]:
    """CUDA processes that are not ours. Non-empty means: do not start a job this tick."""
    return [(pid, mem) for pid, mem in gpu_users(run=run) if pid not in own_pids]


def _own_pids(proc: subprocess.Popen | None) -> set[int]:
    """Our own process tree, so the engine we launched never looks like a blocker to us."""
    pids = {os.getpid()}
    if proc is not None and proc.pid:
        pids.add(proc.pid)
        # The dataloader workers are children of the engine and hold their own CUDA contexts.
        try:
            out = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True, timeout=30)
            pids.update(int(p) for p in out.stdout.split() if p.isdigit())
        except Exception:  # pragma: no cover - pgrep missing is not fatal
            pass
    return pids


def engine_command(job: botero.Job) -> list[str]:
    """The same invocation as ``sbatches/aa_sweep_completion.sbatch``, minus Slurm.

    Differences from the cluster, and only these: the local ImageNet val root, and ``nice`` so an
    interactive session on this workstation always wins the CPU. Notably **no** ``--force``: the
    engine diffs the CSV's existing rows against the grid and attacks only the difference, so
    everything already computed on either cluster is reused, exactly as on the cluster side.
    """
    images = config.BOTERO_BATCH_SIZE * config.BOTERO_NUM_BATCHES
    if images != config.BOTERO_TOTAL_IMAGES:
        raise ValueError(
            f"batch_size {config.BOTERO_BATCH_SIZE} x num_batches {config.BOTERO_NUM_BATCHES}"
            f" = {images} images, but every sweep row must be over exactly"
            f" {config.BOTERO_TOTAL_IMAGES}. Rows attacking a different number of images are not"
            f" comparable to the cluster's."
        )
    return [
        "nice", "-n", "5", config.BOTERO_PYTHON,
        str(config.BOTERO_REPO / "data_analysis/autoattack_array_eval.py"),
        "--model-dir", job.model_dir,
        "--checkpoint-kinds", job.checkpoint_kind,
        "--val-dir", str(config.BOTERO_VAL_DIR),
        "--norms", ",".join(config.NORMS),
        "--eps-inputs", ",".join(str(int(e) if float(e).is_integer() else e) for e in config.EPS_INPUTS),
        "--batch-size", str(config.BOTERO_BATCH_SIZE),
        "--num-batches", str(config.BOTERO_NUM_BATCHES),
        "--num-workers", str(config.BOTERO_NUM_WORKERS),
        "--seed", "0",
        "--device", "cuda",
        "--plot-comparison",
    ]


def echo_to_qnap(job: botero.Job) -> list[str]:
    """Copy the run's small artifacts to the QNAP twin of the archive dir.

    Only the KB-sized outputs -- the checkpoints are already there and are not touched by an eval.
    The weekly ``mirror_archives_to_qnap.sh`` cron that would otherwise do this is disabled, so the
    runner keeps the share current itself. Best effort: a share that is down must not fail a job
    whose real result is already safely on local disk.
    """
    source = Path(job.model_dir)
    target = botero.qnap_counterpart(source, job.model_name)
    if target is None:
        return []
    copied: list[str] = []
    patterns = ("autoattack_sweep_results*.csv", "autoattack_eval_comparation_*.png",
                "autoattack_eps_norm_scores.json")
    try:
        target.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for path in sorted(source.glob(pattern)):
                subprocess.run(["cp", "-p", str(path), str(target / path.name)],
                               check=True, capture_output=True, timeout=600)
                copied.append(path.name)
    except Exception as exc:
        log(f"{job.model_name}:{job.checkpoint_kind}: QNAP echo failed ({exc}); local results are intact")
    return copied


def run_job(job: botero.Job, conn) -> bool:
    """Run one unit to completion in the foreground. Returns True on a clean exit."""
    config.BOTERO_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.BOTERO_LOG_DIR / f"{job.slug}.log"
    cmd = engine_command(job)
    log(f"job {job.id} START {job.model_name}:{job.checkpoint_kind} "
        f"({job.cells_at_enqueue} cells at enqueue) dir={job.model_dir} log={log_path}")

    with log_path.open("a") as fh:
        fh.write(f"\n===== {_now()} job {job.id} {job.model_name}:{job.checkpoint_kind} =====\n")
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(config.BOTERO_REPO))
        # The queue row must name the *engine*, not this wrapper: a runner killed mid-job then
        # leaves a row whose pid is genuinely gone, which is what reap_stale looks for.
        conn.execute("UPDATE botero_jobs SET pid=? WHERE id=?", (proc.pid, job.id))
        rc = proc.wait()

    if rc == 0:
        copied = echo_to_qnap(job)
        botero.finish(conn, job.id, ok=True)
        log(f"job {job.id} DONE {job.model_name}:{job.checkpoint_kind}"
            + (f" (echoed {len(copied)} file(s) to the QNAP)" if copied else ""))
        return True

    tail = ""
    try:
        tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-40:])
    except OSError:
        pass
    botero.finish(conn, job.id, ok=False, error=f"rc={rc}\n{tail}")
    log(f"job {job.id} FAILED rc={rc} {job.model_name}:{job.checkpoint_kind} (see {log_path})")
    return False


def tick(once: bool = False, conn=None) -> int:
    """One cron tick: drain the queue until it is empty or the GPU is taken."""
    own = conn is None
    conn = conn or botero.connect()
    try:
        for note in botero.reap_stale(conn):
            log(note)

        ran = 0
        while True:
            blockers = gpu_blockers(_own_pids(None))
            if blockers:
                pretty = ", ".join(f"pid {pid} ({mem} MiB)" for pid, mem in blockers)
                log(f"GPU busy [{pretty}], deferring (queue {botero.active_count(conn)}"
                    f"/{config.BOTERO_SLOTS})")
                return 0

            # A runner holds its lock for days across several jobs, so it would otherwise keep
            # running every one of them with the settings it imported at start-up. Re-read config
            # before each claim: an edited batch size takes effect at the next job, not the next
            # restart.
            importlib.reload(config)

            job = botero.claim(conn, os.getpid())
            if job is None:
                if ran == 0:
                    log(f"nothing queued (0 active of {config.BOTERO_SLOTS} slots)")
                return 0

            if not Path(job.model_dir).is_dir():
                botero.finish(conn, job.id, ok=False, error=f"model dir vanished: {job.model_dir}")
                log(f"job {job.id} FAILED, model dir vanished: {job.model_dir}")
                continue

            run_job(job, conn)
            ran += 1
            if once:
                return 0
    finally:
        if own:
            conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true",
                        help="Run at most one job, then exit (debugging).")
    parser.add_argument("--check-gpu", action="store_true",
                        help="Report whether the GPU is free and exit without touching the queue.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_gpu:
        try:
            blockers = gpu_blockers({os.getpid()})
        except Exception as exc:
            log(f"cannot query the GPU: {exc}")
            return 1
        if blockers:
            log("GPU busy [" + ", ".join(f"pid {p} ({m} MiB)" for p, m in blockers) + "]")
            return 1
        log("GPU is free")
        return 0
    return tick(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
