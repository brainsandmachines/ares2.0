"""Measure GPU-hours per training epoch from our own training logs (read-only).

Sources
-------
* Slurm (sjm): ``<sjm_outs>/**/*.out``. Each ``sjm.controller: claimed <model>`` line opens a
  segment; the ``[sjm] ... part=<partition>`` header gives the GPU; the ``[lifecycle]`` command gives
  the Hydra overrides (``attacks.attack_it``). One model per GPU, so wall-clock == GPU-hours.
* AIRCC (B200): ``<aircc_archive>/{models,models_failed}/*/log.txt``, one launch per
  ``Runtime distributed=`` line. AIRCC ran 2 training processes per GPU
  (``N_SLOTS = 2`` in ``aircc/aircc_job_manager/job_manager.py``), so GPU-hours = wall-clock / 2.
  PGD steps come from the run dir's ``runtime_config.yaml``; every launch before 2026-08-01 11:46
  (commit 0721d965, the first time any non-DVD row was given ``attack_it`` != 3) is PGD-3.

Epoch time = start of epoch e+1 minus start of epoch e (first ``Train: [e/E]`` line), within one
segment/launch. It therefore includes per-epoch validation + checkpointing, and excludes AutoAttack.
The first epoch of each segment (torch.compile warm-up) is excluded from the rates.

Waste ("redone work")
---------------------
For each model, every epoch occurrence (including first epochs and the partial tail of an
interrupted segment) is costed. When the same epoch index is trained more than once, only the
latest occurrence is productive; earlier ones are waste. This captures interruption rollbacks and
from-scratch relaunches of collapsed / masked models alike. Start-up time before the first Train
line (data loading, compile) is not counted on either side.

Scope: convnext_base, vit_b_cvst, swin_b; madry / trades / clean-baseline runs. DVD, V1 and
GradNorm variants are excluded (different recipes).

Outputs (in --out-dir): epoch_rates.csv, epoch_rates_per_model.csv, waste_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

IMAGENET_TRAIN = 1_281_167
PGD_CHANGE_TS = datetime(2026, 8, 1, 11, 46)
ARCHS = ("convnext_base", "vit_b_cvst", "swin_b")
EXCLUDE_TOKENS = ("dvd", "gradnorm", "_v1", "v1_")

TRAIN_RE = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ARES 2\.0\] INFO: Train: \[(\d+)/(\d+)\] \[\s*(\d+)/(\d+)"
)
CLAIM_RE = re.compile(r"sjm\.controller: claimed (\S+)")
PART_RE = re.compile(r"^\[sjm\] .*\bpart=(\S+)")
ATTACK_IT_RE = re.compile(r"attacks\.attack_it=(\d+)")
LAUNCH_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ARES 2\.0\] INFO: Runtime distributed=")
EXPERIMENT_RE = re.compile(r"INFO: Experiment: (\S+)")


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


@dataclass
class Segment:
    system: str  # slurm | aircc
    model: str
    gpu: str
    pgd: int
    source: str
    gpu_share: float  # GPU-hours per wall-hour (1.0 slurm, 0.5 aircc)
    epoch_start: dict = field(default_factory=dict)  # epoch -> first ts
    iters: int = 0
    total_epochs: int = 0
    last_ts: datetime | None = None

    def add_train(self, ts: datetime, ep: int, tot: int, n_iter: int) -> None:
        self.epoch_start.setdefault(ep, ts)
        self.iters = n_iter
        self.total_epochs = tot
        self.last_ts = ts

    def occurrences(self):
        """(epoch, gpu_hours, is_first_in_segment, is_tail) for every epoch seen."""
        eps = sorted(self.epoch_start)
        out = []
        for i, ep in enumerate(eps):
            start = self.epoch_start[ep]
            if i + 1 < len(eps):
                end, tail = self.epoch_start[eps[i + 1]], False
                if eps[i + 1] != ep + 1:  # gap in the log: cannot time this epoch
                    continue
            else:
                end, tail = self.last_ts, True
            hours = (end - start).total_seconds() / 3600 * self.gpu_share
            out.append((ep, hours, i == 0, tail))
        return out


def classify(model: str):
    """-> (arch, proto) or None if out of scope."""
    m = model.lower()
    if any(t in m for t in EXCLUDE_TOKENS):
        return None
    arch = next((a for a in ARCHS if m.startswith(a)), None)
    if arch is None:
        return None
    leaf = m.split("/")[-1]
    if "baseline" in leaf or "clean" in leaf:
        proto = "clean"
    elif "trades" in leaf:
        proto = "trades"
    else:
        proto = "madry"
    return arch, proto


def canonical_model(name: str) -> str:
    """Strip archive suffixes such as ``__failed_2026...`` / ``__attempt_...``."""
    return name.split("__")[0]


# --------------------------------------------------------------------------- slurm
def read_slurm(outs: Path) -> list[Segment]:
    pat = r"Train: \[|^\[sjm\] |sjm\.controller: claimed|^\[lifecycle\]"
    proc = subprocess.run(
        ["grep", "-rH", "--include=*.out", "-E", pat, str(outs)],
        capture_output=True, text=True, errors="replace", check=False,
    )
    segs: list[Segment] = []
    cur: dict[str, Segment | None] = {}
    part: dict[str, str] = {}
    for raw in proc.stdout.splitlines():
        fname, _, line = raw.partition(":")
        if m := PART_RE.match(line):
            part[fname] = m.group(1)
            continue
        if m := CLAIM_RE.search(line):
            model = m.group(1)
            seg = Segment("slurm", model, part.get(fname, "?"), 3, fname, 1.0)
            cur[fname] = seg
            segs.append(seg)
            continue
        seg = cur.get(fname)
        if seg is None:
            continue
        if line.startswith("[lifecycle]"):
            if m := ATTACK_IT_RE.search(line):
                seg.pgd = int(m.group(1))
            continue
        if m := TRAIN_RE.match(line):
            seg.add_train(parse_ts(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(5)))
    return [s for s in segs if s.epoch_start]


# --------------------------------------------------------------------------- aircc
def runtime_attack_it(run_dir: Path) -> int:
    cfg = run_dir / "runtime_config.yaml"
    if cfg.exists():
        for line in cfg.read_text(errors="replace").splitlines():
            s = line.strip()
            if s.startswith("attack_it:"):
                try:
                    return int(s.split(":", 1)[1])
                except ValueError:
                    pass
    return 3


def read_aircc(archive: Path) -> list[Segment]:
    segs: list[Segment] = []
    seen: set[tuple[str, datetime]] = set()
    for sub in ("models", "models_failed"):
        for run_dir in sorted((archive / sub).glob("*")):
            log = run_dir / "log.txt"
            if not log.exists():
                continue
            cfg_pgd = runtime_attack_it(run_dir)
            seg = None
            dir_model = canonical_model(run_dir.name)
            with log.open(errors="replace") as fh:
                for line in fh:
                    if m := LAUNCH_RE.match(line):
                        ts = parse_ts(m.group(1))
                        pgd = 3 if ts < PGD_CHANGE_TS else cfg_pgd
                        seg = Segment("aircc", dir_model, "B200", pgd, str(log), 0.5)
                        seg.launch_ts = ts
                        segs.append(seg)
                        continue
                    if seg is None:
                        continue
                    if m := EXPERIMENT_RE.search(line):
                        seg.model = canonical_model(m.group(1))
                        continue
                    if "Train: [" in line and (m := TRAIN_RE.match(line)):
                        seg.add_train(parse_ts(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(5)))
    out = []
    for s in segs:  # drop launches duplicated across archive copies
        key = (s.model, getattr(s, "launch_ts", None))
        if s.epoch_start and key not in seen:
            seen.add(key)
            out.append(s)
    return out


# --------------------------------------------------------------------------- analysis
def pctl(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def rates(segs: list[Segment]):
    groups = defaultdict(list)  # key -> [(model, hours)]
    sources = defaultdict(set)
    for s in segs:
        c = classify(s.model)
        if c is None or s.iters == 0:
            continue
        arch, proto = c
        pgd = 0 if proto == "clean" else s.pgd
        bs = round(IMAGENET_TRAIN / s.iters)
        key = (arch, proto, pgd, s.gpu, bs)
        for ep, h, first, tail in s.occurrences():
            if first or tail or h <= 0:
                continue
            groups[key].append((s.model, h))
        sources[key].add(s.source)
    rows, per_model = [], []
    for key, vals in sorted(groups.items()):
        if not vals:
            continue
        hs = [h for _, h in vals]
        models = sorted({m for m, _ in vals})
        arch, proto, pgd, gpu, bs = key
        rows.append(dict(
            arch=arch, proto=proto, pgd=pgd, gpu=gpu, batch=bs,
            median_gpu_h_per_epoch=round(statistics.median(hs), 4),
            q25=round(pctl(hs, 0.25), 4), q75=round(pctl(hs, 0.75), 4),
            n_models=len(models), n_epochs=len(hs),
            gpu_share=("wall/2 (2 models per B200)" if gpu == "B200" else "wall (1 model per GPU)"),
            example_sources=";".join(sorted(sources[key])[:3]),
        ))
        by_model = defaultdict(list)
        for m, h in vals:
            by_model[m].append(h)
        for m, v in sorted(by_model.items()):
            per_model.append(dict(arch=arch, proto=proto, pgd=pgd, gpu=gpu, batch=bs, model=m,
                                  median_gpu_h_per_epoch=round(statistics.median(v), 4), n_epochs=len(v)))
    return rows, per_model


def load_discarded(repo: Path) -> dict[str, datetime]:
    """Models whose run was thrown away (collapse / masking), with the date it was discarded.

    Read from ``aa_sweep/sweep_excludes.csv`` and ``model_store/hf_excludes.csv``; the first
    ``YYYY-MM-DD`` in the reason is the discard date. Names are normalised ``arch/x`` -> ``arch_x``.
    """
    out: dict[str, datetime] = {}
    for rel, col in (("aa_sweep/sweep_excludes.csv", "model_name"), ("model_store/hf_excludes.csv", "pattern")):
        path = repo / rel
        if not path.exists():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                m = re.search(r"(\d{4}-\d\d-\d\d)", row.get("reason", ""))
                if not m:
                    continue
                name = row[col].replace("/", "_")
                d = datetime.strptime(m.group(1), "%Y-%m-%d")
                out[name] = min(d, out.get(name, d))
    return out


def waste(segs: list[Segment], system: str, discarded: dict[str, datetime], proto: str = "all"):
    by_model = defaultdict(list)
    for s in segs:
        c = classify(s.model)
        if s.system == system and c is not None and proto in ("all", c[1]):
            by_model[canonical_model(s.model)].append(s)
    total = wasted = 0.0
    relaunched = 0
    for model, ss in by_model.items():
        cutoff = discarded.get(model.replace("/", "_"))
        occ = []  # (time, epoch, hours)
        for s in ss:
            for ep, h, _first, _tail in s.occurrences():
                if h > 0:
                    occ.append((s.epoch_start[ep], ep, h))
        occ.sort()
        last_idx = {}
        for i, (_, ep, _) in enumerate(occ):
            last_idx[ep] = i
        model_wasted = 0.0
        for i, (t, ep, h) in enumerate(occ):
            total += h
            if last_idx[ep] != i or (cutoff is not None and t < cutoff):
                model_wasted += h
        wasted += model_wasted
        # a model counts as relaunched if >= 5 epochs of its work were thrown away
        if model_wasted > 0 and (cutoff is not None or sum(1 for i, (_, ep, _) in enumerate(occ) if last_idx[ep] != i) >= 5):
            relaunched += 1
    n = len(by_model)
    return dict(system=system, proto=proto, models=n, total_gpu_h=round(total, 1), wasted_gpu_h=round(wasted, 1),
                waste_frac_of_total=round(wasted / total, 4) if total else 0,
                overhead_vs_useful=round(wasted / (total - wasted), 4) if total > wasted else 0,
                models_with_discarded_run=relaunched,
                relaunch_frac=round(relaunched / n, 4) if n else 0)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sjm-outs", type=Path, default=Path.home() / "slurm_mount/projects/ares/outs/sjm")
    ap.add_argument("--aircc-archive", type=Path, default=Path("/mnt/botero/aircc_archive"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    segs = read_slurm(args.sjm_outs) + read_aircc(args.aircc_archive)
    rows, per_model = rates(segs)
    write_csv(args.out_dir / "epoch_rates.csv", rows)
    write_csv(args.out_dir / "epoch_rates_per_model.csv", per_model)
    discarded = load_discarded(Path(__file__).resolve().parents[2])
    w = [waste(segs, "slurm", discarded), waste(segs, "aircc", discarded)]
    by_proto = [waste(segs, sysname, discarded, pr) for sysname in ("slurm", "aircc") for pr in ("madry", "trades", "clean")]
    tot = sum(x["total_gpu_h"] for x in w)
    wst = sum(x["wasted_gpu_h"] for x in w)
    w.append(dict(system="combined", proto="all", models=sum(x["models"] for x in w), total_gpu_h=round(tot, 1),
                  wasted_gpu_h=round(wst, 1), waste_frac_of_total=round(wst / tot, 4),
                  overhead_vs_useful=round(wst / (tot - wst), 4),
                  models_with_discarded_run=sum(x["models_with_discarded_run"] for x in w),
                  relaunch_frac=round(sum(x["models_with_discarded_run"] for x in w)
                                      / max(1, sum(x["models"] for x in w)), 4)))
    madry = [x for x in by_proto if x["proto"] == "madry"]
    mt, mw = sum(x["total_gpu_h"] for x in madry), sum(x["wasted_gpu_h"] for x in madry)
    by_proto.append(dict(system="combined", proto="madry", models=sum(x["models"] for x in madry),
                         total_gpu_h=round(mt, 1), wasted_gpu_h=round(mw, 1),
                         waste_frac_of_total=round(mw / mt, 4), overhead_vs_useful=round(mw / (mt - mw), 4),
                         models_with_discarded_run=sum(x["models_with_discarded_run"] for x in madry),
                         relaunch_frac=round(sum(x["models_with_discarded_run"] for x in madry)
                                             / max(1, sum(x["models"] for x in madry)), 4)))
    w += by_proto
    write_csv(args.out_dir / "waste_summary.csv", w)
    for r in rows:
        print(f"{r['arch']:14s} {r['proto']:6s} pgd{r['pgd']} {r['gpu']:13s} bs{r['batch']:<4d} "
              f"{r['median_gpu_h_per_epoch']:.3f} ({r['q25']:.2f}-{r['q75']:.2f}) "
              f"models={r['n_models']} epochs={r['n_epochs']}")
    for x in w:
        print(x)


if __name__ == "__main__":
    main()
