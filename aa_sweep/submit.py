"""Daily driver: keep two independent AutoAttack sweep lanes fed.

    python -m aa_sweep.submit --dry-run     # show the plan, write and submit nothing
    python -m aa_sweep.submit               # feed the cluster queue + top up the local queue

Run from a Botero cron via ``aa_sweep/scripts/aa_sweep_daily.sh``. Read-only against both job DBs
and against the BGU cluster's filesystem; the only writes are rows in the two queue DBs -- the
cluster's (``cluster_queue.py``, over one ssh, which also submits a partition's array when it is
due) and this machine's own.

**No model is ever copied.** Each lane evaluates the copy of the model its own machine already
holds -- the cluster from ``results/models``, this machine from ``config.BOTERO_STORE_ROOT`` -- and
writes its results back beside it. Propagating those results between machines is the weekly rsync's
job, not this package's.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime

from aa_sweep import botero as botero_mod, config, plan as plan_mod


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[aa_sweep] {_now()} {msg}", flush=True)


def check_paths() -> list[str]:
    """A missing filesystem looks exactly like 'no work to do', so fail loudly instead.

    Two things have to be there: the sshfs mount of the BGU cluster (for the sjm DB) and the QNAP
    share (for the frozen AIRCC DB -- the model *list*, not the models). Plus the local store the
    Botero lane evaluates from, which is ordinary local disk rather than a mount but is worth the
    same check: an empty store would silently retire the whole local lane.
    """
    problems = []
    try:
        mounts = subprocess.run(["mount"], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # pragma: no cover - environment failure
        return [f"could not run `mount`: {exc}"]
    for label, path in (("slurm", config.SLURM_MOUNT), ("qnap", config.AIRCC_ARCHIVE.parent)):
        if f" {path} " not in mounts:
            problems.append(f"{label} mount is not mounted at {path}")
    # Mounted is not the same as populated: a share that reconnected empty reads as "nothing to do".
    if not config.AIRCC_DB.is_file():
        problems.append(f"frozen aircc job DB is missing at {config.AIRCC_DB}")
    if not config.BOTERO_STORE_ROOT.is_dir():
        problems.append(f"local model store is missing at {config.BOTERO_STORE_ROOT}")
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
    # Both forms this unit's own job can wear. A *nested* model is submitted as
    # `aaswp_vit_b_cvst__l2_cont4to6_init1_best` but renames itself to
    # `aaswp_l2_cont4to6_init1_best` the moment it starts running (the `scontrol update` in
    # sbatches/aa_sweep_completion.sbatch), so checking only the submitted form would let a second
    # job open a CSV a running one already owns.
    for mine in config.own_job_names(model_name, kind):
        if mine in live_names:
            return mine

    # Our own three kinds may run concurrently -- they write three different CSVs. A job we did
    # NOT name but which mentions this model carries no such guarantee (a hand-launched eval, a
    # re-training run), so treat it as a conflict and let the next night pick the work up.
    #
    # Any `aaswp_*` name is one of ours by construction, whichever model it belongs to, so the
    # exact tests above are the only conflicts it can be -- decode and skip the rest. Doing this by
    # name-shape rather than by comparing against this model's own names is what fixes the
    # collision between a nested model and a flat one: `swin_b/linf_cont4to6_init1` reduces to the
    # dir name `linf_cont4to6_init1`, which is a suffix of the *unrelated*
    # `convnext_base_linf_cont4to6_init1` with a `_` on either side. Those blockers were jobs stuck
    # PENDING behind QOSMaxGRESPerUser for weeks, so "next night" never came and the nested models
    # were skipped every single run.
    #
    # For foreign names there is nothing to decode, so fall back to the dir name and match on token
    # boundaries rather than raw substring: a plain `in` test makes the model dir name `m` match
    # `sjm-manager`. Names are `_`/`-` delimited, so requiring non-alphanumeric neighbours is
    # enough. Still deliberately broad -- against an unknown job, over-blocking beats two writers
    # on one CSV.
    dir_name = model_name.rsplit("/", 1)[-1]
    token = re.compile(rf"(?<![0-9A-Za-z]){re.escape(dir_name)}(?![0-9A-Za-z])")
    for name in live_names:
        if config.parse_job_name(name) is not None:
            continue
        if token.search(name):
            return name
    return None


def feed_queue(units: list[dict], dry_run: bool = False, run=subprocess.run) -> str:
    """Hand the Slurm lane's census to the cluster queue in one ssh round trip.

    Runs ``python3 -m aa_sweep.cluster_queue feed --launch`` on the login node with the units as
    JSON on stdin: that upserts them into the queue DB and submits each partition's array if it is
    due (see cluster_queue.launch). Returns the remote output; raises on a non-zero exit, which
    covers both a failed DB write and a failed array sbatch.
    """
    flag = "--dry-run" if dry_run else "--launch"
    remote = (f"cd {config.SLURM_REPO} && PYTHONPATH={config.SLURM_REPO} "
              f"python3 -m aa_sweep.cluster_queue feed {flag}")
    proc = run(
        ["ssh", "-o", f"ConnectTimeout={config.SSH_TIMEOUT_SECONDS}", config.SLURM_SSH_HOST, remote],
        input=json.dumps(units),
        capture_output=True,
        text=True,
        timeout=config.SSH_TIMEOUT_SECONDS * 4,
    )
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"queue feed failed rc={proc.returncode}: {out}\n{proc.stderr.strip()}")
    return out


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
    parser.add_argument("--limit", type=int, default=None, help="Debugging knob: feed at most N runnable units.")
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
        problems = check_paths()
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

    slurm_works = [w for w in works if w.lane == config.SLURM_LANE]
    botero_works = [w for w in works if w.lane == config.BOTERO_LANE]
    log(f"lanes: slurm={len(slurm_works)} models (own copy on the cluster), "
        f"botero={len(botero_works)} models (own copy in {config.BOTERO_STORE_ROOT})")

    pending = [w for w in slurm_works if not w.is_complete]
    log(f"slurm lane: {len(slurm_works) - len(pending)} complete, {len(pending)} needing "
        f"{sum(w.missing_cell_count for w in pending)} grid cells")
    botero_pending = [w for w in botero_works if not w.is_complete]
    log(f"botero lane: {len(botero_works) - len(botero_pending)} complete, {len(botero_pending)} "
        f"needing {sum(w.missing_cell_count for w in botero_pending)} grid cells")

    try:
        running = live_job_names()
    except Exception as exc:
        log(f"ABORT: squeue check failed: {exc}")
        notify("[aa_sweep] squeue check failed", str(exc), dedup_key="aa_sweep-squeue-failed")
        return 1

    fed: list[dict] = []
    runnable: list[dict] = []
    skipped_live = 0
    failures: list[str] = []
    moved: list[str] = []

    if args.botero_topup_only:
        log("--botero-topup-only: submitting no sbatch")
        try:
            moved = botero_mod.topup(botero_works, dry_run=args.dry_run, log=log)
        except Exception as exc:
            log(f"botero top-up failed: {exc}")
            notify("[aa_sweep] botero top-up failed", str(exc), dedup_key="aa_sweep-botero-topup-failed")
            return 1
        verb = "would enqueue" if args.dry_run else "enqueued"
        log(f"summary: {verb} {len(moved)} unit(s) in the Botero lane")
        return 0

    # Every Slurm-lane unit with a checkpoint goes to the queue, complete ones included (missing=0):
    # that is how the queue closes a pending row some other job already finished. The only units
    # held back are those a *standalone* job is still working on -- a hand-submitted eval or a
    # pre-queue `aaswp_*` sbatch -- so the arrays never put a second writer on that CSV.
    for work in slurm_works:
        for kind in config.CHECKPOINT_KINDS:
            status = work.kinds.get(kind)
            if status is None or not status.has_checkpoint:
                continue
            if status.missing:
                blocker = conflicting_job(work.model_name, kind, running)
                if blocker is not None:
                    skipped_live += 1
                    log(f"{work.model_name}:{kind}: skipping, '{blocker}' is already queued/running")
                    continue
            fed.append({"model_name": work.model_name, "kind": kind,
                        "model_dir": work.slurm_dir, "missing": len(status.missing)})

    runnable = [u for u in fed if u["missing"]]
    if args.limit is not None and len(runnable) > args.limit:
        log(f"--limit {args.limit}: feeding {args.limit} of {len(runnable)} runnable unit(s)")
        runnable = runnable[: args.limit]
        fed = [u for u in fed if not u["missing"]] + runnable
    for unit in runnable:
        log(f"{'DRY-RUN ' if args.dry_run else ''}feed {unit['model_name']}:{unit['kind']} "
            f"({unit['missing']} cells)")
    if fed:
        try:
            for line in feed_queue(fed, dry_run=args.dry_run).splitlines():
                log(f"cluster: {line}")
        except Exception as exc:
            failures.append(f"queue feed: {exc}")
            log(f"queue feed FAILED {exc}")

    # The local lane is independent of the cluster submissions above -- it draws from a disjoint set
    # of models -- so its failures are collected but never cost a submission that already succeeded.
    if not args.no_botero:
        try:
            moved = botero_mod.topup(botero_works, dry_run=args.dry_run, log=log)
        except Exception as exc:
            failures.append(f"botero top-up: {exc}")
            log(f"botero top-up FAILED {exc}")

    verb = "would feed" if args.dry_run else "fed"
    log(
        f"summary: {verb} {len(runnable)} runnable unit(s) to the cluster queue "
        f"({len(fed)} censused); {skipped_live} held by a standalone job; "
        f"{len(moved)} enqueued on botero; {len(failures)} failures"
    )

    if failures:
        notify(f"[aa_sweep] {len(failures)} failure(s)", "\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
