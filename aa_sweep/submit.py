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
import re
import subprocess
import sys
from datetime import datetime

from aa_sweep import botero as botero_mod, config, mirror, plan as plan_mod, stage as stage_mod


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
    """Job names already in flight, so a 30h job is not resubmitted daily.

    squeue's default state filter covers PENDING as well as RUNNING, which matters here: a job can
    sit pending for days behind the queue and must not be submitted again in the meantime.

    ``-o '%j'`` is deliberately unwidthed. A width like ``%.40j`` truncates
    ``aaswp_convnext_base_linftrades_2_init0_last`` to 40 characters, and every comparison against
    a full job name would then miss.

    The Botero lane's queue is folded in under the same naming scheme, so a unit this machine owns
    is never also sent to the cluster. It has to be a union rather than a separate check: the
    dedupe below reasons about one set of names, and a unit belongs to exactly one lane.
    """
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST,
         f"squeue -u {config.SLURM_USER} -h -o '%j'"],
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 2,
    )
    if proc.returncode != 0:
        # Fail closed: without a reliable queue view we cannot tell what is already running, and
        # submitting duplicates of a multi-day job is worse than skipping a night.
        raise RuntimeError(f"squeue failed rc={proc.returncode}: {proc.stderr.strip()}")
    names = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return names | botero_mod.active_job_names()


def conflicting_job(model_name: str, kind: str, live_names: set[str]) -> str | None:
    """Name of a live job that means this (model, kind) must not be submitted, else None."""
    mine = config.job_name(model_name, kind)
    if mine in live_names:
        return mine

    # Our own three kinds may run concurrently -- they write three different CSVs. A job we did
    # NOT name but which mentions this model carries no such guarantee (a hand-launched eval, a
    # re-training run), so treat it as a conflict and let the next night pick the work up.
    #
    # Match on token boundaries, not raw substring: a plain `in` test makes the model dir name
    # `m` match `sjm-manager`, and short nested names like `linf_1_init1` would be similarly
    # trigger-happy. Job names are `_`/`-` delimited, so requiring non-alphanumeric neighbours is
    # enough.
    ours = {config.job_name(model_name, k) for k in config.CHECKPOINT_KINDS}
    dir_name = model_name.rsplit("/", 1)[-1]
    token = re.compile(rf"(?<![0-9A-Za-z]){re.escape(dir_name)}(?![0-9A-Za-z])")
    for name in live_names:
        if name not in ours and token.search(name):
            return name
    return None


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


def notify(subject: str, body: str, dedup_key: str | None = None) -> None:
    """Email on real breakage only, matching the other two Botero cron scripts.

    ``dedup_key`` lets the morning digest collapse a repeat of the same
    condition to one line -- these are mostly transient cluster/mount problems
    that repeat verbatim for days.
    """
    try:
        from aircc.aircc_job_manager.notify import make_emailer

        emailer = make_emailer(source="aa_sweep")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[notify] emailer unavailable ({exc}); would send: {subject}", file=sys.stderr)
        return
    if emailer is None:
        print(f"[notify] no emailer configured; would send: {subject}", file=sys.stderr)
        return
    emailer(subject, body, dedup_key=dedup_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; stage and submit nothing.")
    parser.add_argument("--limit", type=int, default=None, help="Debugging knob: submit at most N jobs.")
    parser.add_argument("--model", action="append", default=None,
                        help="Restrict to this model name (repeatable). Debugging knob.")
    parser.add_argument("--skip-mount-check", action="store_true", help="For testing on a host without the mounts.")
    parser.add_argument("--no-botero", action="store_true",
                        help="Skip the Botero-lane top-up; submit to the cluster only.")
    parser.add_argument("--botero-topup-only", action="store_true",
                        help="Only top the Botero queue up; stage nothing and submit no sbatch.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.skip_mount_check:
        problems = check_mounts()
        if problems:
            msg = "; ".join(problems)
            log(f"ABORT: {msg}")
            notify("[aa_sweep] mounts down", f"Cannot plan the AutoAttack sweep:\n\n{msg}",
               dedup_key="aa_sweep-mounts-down")
            return 1

    try:
        aircc_finished = plan_mod.finished_models(config.AIRCC_DB)
        sjm_finished = plan_mod.finished_models(config.SJM_DB)
    except Exception as exc:
        log(f"ABORT: reading job DBs failed: {exc}")
        notify("[aa_sweep] job DB read failed", str(exc), dedup_key="aa_sweep-db-read-failed")
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
        notify("[aa_sweep] cluster probe failed", str(exc), dedup_key="aa_sweep-probe-failed")
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
        notify("[aa_sweep] squeue check failed", str(exc), dedup_key="aa_sweep-squeue-failed")
        return 1

    # One check per run, not per model: whether the nightly backup that fills the local mirror
    # completed cleanly. If it did not, every model falls back to reading the AIRCC mount.
    backup_ok, backup_message = mirror.backup_log_ok()
    log(f"mirror source: {'usable' if backup_ok else 'NOT usable'} ({backup_message})")

    submitted: list[str] = []
    skipped_live = 0
    staged_bytes = 0
    failures: list[str] = []
    moved: list[str] = []

    if args.botero_topup_only:
        log("--botero-topup-only: staging nothing and submitting no sbatch")
        try:
            moved = botero_mod.topup(works, dry_run=args.dry_run, log=log)
        except Exception as exc:
            log(f"botero top-up failed: {exc}")
            notify("[aa_sweep] botero top-up failed", str(exc), dedup_key="aa_sweep-botero-topup-failed")
            return 1
        verb = "would move" if args.dry_run else "moved"
        log(f"summary: {verb} {len(moved)} unit(s) to the Botero lane")
        return 0

    for work in pending:
        kinds = []
        for kind in work.runnable_kinds:
            blocker = conflicting_job(work.model_name, kind, running)
            if blocker is None:
                kinds.append(kind)
            else:
                skipped_live += 1
                log(f"{work.model_name}:{kind}: skipping, '{blocker}' is already queued/running")
        if not kinds:
            continue

        if work.staging_files or work.aircc_dir is not None:
            res = stage_mod.stage_model(
                work, work.aircc_csvs, work.slurm_csvs, dry_run=args.dry_run,
                backup_ok=backup_ok, backup_message=backup_message,
            )
            staged_bytes += res.bytes_planned
            if not res.ok:
                failures.append(f"{work.model_name}: {res.error}")
                log(f"{work.model_name}: STAGING FAILED ({res.error}), not submitting")
                continue
            if res.files:
                via = res.source + (f" [{res.source_reason}]" if res.source_reason else "")
                log(f"{work.model_name}: staged {', '.join(res.files)} "
                    f"({res.bytes_planned / 1e9:.1f} GB) via {via}")
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

    # Last, deliberately: the Botero lane prefers to *move* work already queued on Slurm, so it
    # should see tonight's submissions as candidates too rather than a queue that is one day stale.
    if not args.no_botero:
        try:
            moved = botero_mod.topup(works, dry_run=args.dry_run, log=log)
        except Exception as exc:
            # A local-lane problem must not cost the cluster submissions that already succeeded.
            failures.append(f"botero top-up: {exc}")
            log(f"botero top-up FAILED {exc}")

    verb = "would submit" if args.dry_run else "submitted"
    log(
        f"summary: {verb} {len(submitted)} jobs; {skipped_live} already in flight; "
        f"{staged_bytes / 1e9:.1f} GB staged; {len(moved)} moved to botero; {len(failures)} failures"
    )

    if failures:
        notify(f"[aa_sweep] {len(failures)} failure(s)", "\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
