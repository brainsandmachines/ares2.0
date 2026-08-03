"""Daily driver: complete the AutoAttack sweep grid for every finished model on both clusters.

    python -m aa_sweep.submit --dry-run     # show the plan, submit nothing
    python -m aa_sweep.submit               # stage + submit

Run from a Botero cron via ``aa_sweep/scripts/aa_sweep_daily.sh``. Read-only against both job DBs;
the only writes are the staging rsync onto the BGU cluster and the sbatch submissions.

Order matters: census first, stage second, submit third. Censusing before staging is what keeps
already-complete models free -- they cost one directory listing and zero bytes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from aa_sweep import config, plan as plan_mod, stage as stage_mod


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[aa_sweep] {_now()} {msg}", flush=True)


def check_mounts() -> list[str]:
    """The sshfs mounts are known to drop silently; a missing one looks like 'no work to do'."""
    problems = []
    try:
        mounts = subprocess.run(["mount"], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # pragma: no cover - environment failure
        return [f"could not run `mount`: {exc}"]
    for label, path in (("slurm", config.SLURM_MOUNT), ("aircc", config.AIRCC_MOUNT)):
        if f" {path} " not in mounts:
            problems.append(f"{label} mount is not mounted at {path}")
    return problems


def live_job_names(run=subprocess.run) -> set[str]:
    """Job names already pending/running on the cluster, so a 30h job is not resubmitted daily."""
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST,
         f"squeue -u {config.SLURM_USER} -h -o '%j'"],
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 2,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"squeue failed rc={proc.returncode}: {proc.stderr.strip()}")
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def submit_job(model_dir: str, model_name: str, kind: str, run=subprocess.run) -> str:
    remote = (
        f"cd {config.SLURM_REPO} && sbatch --parsable "
        f"--job-name={config.job_name(model_name, kind)} "
        f"--export=ALL,AA_MODEL_DIR={model_dir},AA_CHECKPOINT_KIND={kind} "
        f"{config.SBATCH_SCRIPT}"
    )
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST, remote],
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 2,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed rc={proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout.strip().split(";")[0]


def notify(subject: str, body: str) -> None:
    """Email on real breakage only, matching the other two Botero cron scripts."""
    try:
        from aircc.aircc_job_manager.notify import make_emailer

        emailer = make_emailer()
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[notify] emailer unavailable ({exc}); would send: {subject}", file=sys.stderr)
        return
    if emailer is None:
        print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
        return
    emailer(subject, body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; stage and submit nothing.")
    parser.add_argument("--limit", type=int, default=None, help="Debugging knob: submit at most N jobs.")
    parser.add_argument("--model", action="append", default=None,
                        help="Restrict to this model name (repeatable). Debugging knob.")
    parser.add_argument("--skip-mount-check", action="store_true", help="For testing on a host without the mounts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.skip_mount_check:
        problems = check_mounts()
        if problems:
            msg = "; ".join(problems)
            log(f"ABORT: {msg}")
            notify("[aa_sweep] mounts down", f"Cannot plan the AutoAttack sweep:\n\n{msg}")
            return 1

    try:
        aircc_finished = plan_mod.finished_models(config.AIRCC_DB)
        sjm_finished = plan_mod.finished_models(config.SJM_DB)
    except Exception as exc:
        log(f"ABORT: reading job DBs failed: {exc}")
        notify("[aa_sweep] job DB read failed", str(exc))
        return 1

    if args.model:
        wanted = set(args.model)
        aircc_finished = [m for m in aircc_finished if m in wanted]
        sjm_finished = [m for m in sjm_finished if m in wanted]

    candidates = sorted(set(aircc_finished) | set(sjm_finished))
    log(f"finished models: aircc={len(aircc_finished)} sjm={len(sjm_finished)} total={len(candidates)}")

    try:
        probe = plan_mod.probe_slurm(candidates)
    except Exception as exc:
        log(f"ABORT: cluster probe failed: {exc}")
        notify("[aa_sweep] cluster probe failed", str(exc))
        return 1

    works = plan_mod.build_plan(aircc_finished, sjm_finished, probe)

    complete = [w for w in works if w.is_complete]
    pending = [w for w in works if not w.is_complete]
    log(f"complete (nothing to do, nothing staged): {len(complete)}")
    log(f"needing work: {len(pending)} models, {sum(w.missing_cell_count for w in pending)} grid cells")

    for work in works:
        for line in work.conflicts:
            log(f"CONFLICT {work.model_name}: {line}")

    try:
        running = live_job_names()
    except Exception as exc:
        log(f"ABORT: squeue check failed: {exc}")
        notify("[aa_sweep] squeue check failed", str(exc))
        return 1

    submitted: list[str] = []
    skipped_live = 0
    staged_bytes = 0
    failures: list[str] = []

    for work in pending:
        kinds = [k for k in work.runnable_kinds if config.job_name(work.model_name, k) not in running]
        skipped_live += len(work.runnable_kinds) - len(kinds)
        if not kinds:
            log(f"{work.model_name}: all runnable kinds already in flight, skipping")
            continue

        if work.staging_files or work.aircc_dir is not None:
            res = stage_mod.stage_model(work, work.aircc_csvs, work.slurm_csvs, dry_run=args.dry_run)
            staged_bytes += res.bytes_planned
            if not res.ok:
                failures.append(f"{work.model_name}: {res.error}")
                log(f"{work.model_name}: STAGING FAILED ({res.error}), not submitting")
                continue
            if res.files:
                log(f"{work.model_name}: staged {', '.join(res.files)} ({res.bytes_planned / 1e9:.1f} GB)")
            if res.merged_csvs:
                log(f"{work.model_name}: merged aircc-only rows into {', '.join(res.merged_csvs)} csv(s)")

        for kind in kinds:
            if args.limit is not None and len(submitted) >= args.limit:
                log(f"--limit {args.limit} reached, stopping")
                break
            missing = len(work.kinds[kind].missing)
            if args.dry_run:
                log(f"DRY-RUN would submit {config.job_name(work.model_name, kind)} ({missing} cells)")
                submitted.append(f"{work.model_name}:{kind}")
                continue
            try:
                job_id = submit_job(work.slurm_dir, work.model_name, kind)
            except Exception as exc:
                failures.append(f"{work.model_name}:{kind}: {exc}")
                log(f"{work.model_name}:{kind}: SUBMIT FAILED {exc}")
                continue
            log(f"submitted {config.job_name(work.model_name, kind)} job={job_id} ({missing} cells)")
            submitted.append(f"{work.model_name}:{kind}")
        if args.limit is not None and len(submitted) >= args.limit:
            break

    verb = "would submit" if args.dry_run else "submitted"
    log(
        f"summary: {verb} {len(submitted)} jobs; {skipped_live} already in flight; "
        f"{staged_bytes / 1e9:.1f} GB staged; {len(failures)} failures"
    )

    if failures:
        notify(f"[aa_sweep] {len(failures)} failure(s)", "\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
