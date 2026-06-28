#!/usr/bin/env python3
"""Generate the permanent AIRCC model-specification CSVs (the ground truth).

One CSV per architecture. Each row fully specifies one model with **one column
per Hydra override** (header = the Hydra key; see ``csv_spec.OVERRIDE_COLUMNS``)
plus a handful of manager-metadata columns. The job manager builds the training
command from the non-empty override cells -- there is no free-text override blob.
Dynamic checkpoint args (``continuation.checkpoint_path`` / ``model.resume``) and
``+machine=aircc`` are added at launch time, not stored here.

Per init (0,1,2), per architecture (see the approved plan Round 2 §3):

  * baseline (1)                         clean, advtrain=False, 150 ep
  * dvd_b_baseline (1)                   DVD-B aug, advtrain=False, 200 ep
  * madry x {l1,l2,linf} (18)            scratch 1/2/4 (150 ep); cont 4to6,4to8 (40);
                                         cont 8to12 (50, "+10 ramp")
  * trades x {l1,l2,linf} (18)           same, criterion=trades, no compile
  * dvd_b+madry x {l1,l2,linf} (27)      scratch 1/2/4 (200 ep); cont 4to6/4to8/8to12
                                         in resetepoch & contepoch variants
  * dvd_b+trades x {l1,l2,linf} (27)     same, criterion=trades, no compile
  * v1 (l2 only) (6)                     V1 feature-space, scratch + conts
  * gradnorm_l1 (6)                      eps 1..12, resume same-init baseline (150->200)
  => 104 rows/init x 3 inits = 312 rows/CSV.

Per-protocol batch sizes (your choice): madry/baseline/dvd_baseline/dvd_madry=512,
trades/gradnorm/dvd_trades=256, v1=256. compile only for madry & dvd_madry.
checkpointing.save_best_adv=true for everything except the two baselines.

Continuation init mode (ares/utils/continuation.py + adversarial_training.py:168):
  * continuation.enabled -> weights-only, epoch resets to 0 (resetepoch)
  * model.resume         -> inherits epoch+optimizer (contepoch, gradnorm)
Default warmup_ramp_fixed = 40 ep {warmup 4, ramp 5..25, fixed 26}; "+10 ramp"
(eps 12) = 50 ep {ramp 5..35, fixed 36}. For contepoch the boundaries are offset
by the source model's final epoch O.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aircc.aircc_job_manager.csv_spec import ALL_COLUMNS, OVERRIDE_COLUMNS

ARCHES = ("convnext_small", "convnext_base", "convnext_large")
INITS = (0, 1, 2)
NORMS = ("l1", "l2", "linf")

BASELINE_EPOCHS = 150
DVD_SCRATCH_EPOCHS = 200
ADV_SCRATCH_EPOCHS = 150
GRADNORM_EXTRA_EPOCHS = 50

CONT_WARMUP = 4
CONT_RAMP_START = 5
CONT_RAMP_END_68 = 25
CONT_FIXED_START_68 = 26
CONT_LEN_68 = 40
CONT_RAMP_END_12 = 35
CONT_FIXED_START_12 = 36
CONT_LEN_12 = 50


def batch_size(*, protocol: str, dvd: bool, criterion: str) -> int:
    """512 for madry/baseline/dvd_baseline/dvd_madry; 256 for trades/gradnorm/dvd_trades/v1."""
    if protocol in ("baseline", "dvd_b_baseline"):
        return 512
    if protocol == "gradnorm" or protocol == "v1":
        return 256
    if criterion == "trades":  # trades or dvd_trades
        return 256
    return 512  # madry or dvd_madry


# --------------------------------------------------------------------------
class RowSink:
    def __init__(self):
        self.rows: list[dict] = []
        self._priority = 0

    def add(self, **kw):
        row = {c: "" for c in ALL_COLUMNS}
        for k, v in kw.items():
            if k not in row:
                raise KeyError(f"unknown column: {k}")
            row[k] = v
        row["priority"] = self._priority
        self._priority += 1
        self.rows.append(row)


def _common(row: dict, *, arch: str, name: str, init: int, epochs: int, bs: int, v1: bool):
    """Fill the always-present override columns."""
    row["model.experiment_name"] = name
    row["model.experiment_num"] = init
    row["output_dir"] = "results/models"
    row["training.epochs"] = epochs
    row["training.batch_size"] = bs
    if v1:
        row["model"] = f"{arch}_v1"
        row["model.v1_noise_mode"] = "null"
    elif arch != "convnext_small":
        row["model"] = arch


def _schedule_cols(row: dict, source_eps: int, target_eps: int, offset: int) -> int:
    """Fill epsilon_schedule.* columns (offset by O for contepoch); return total epochs."""
    if target_eps == 12:
        ramp_end = offset + CONT_RAMP_END_12
        fixed_start = offset + CONT_FIXED_START_12
        epochs = offset + CONT_LEN_12
    else:
        ramp_end = offset + CONT_RAMP_END_68
        fixed_start = offset + CONT_FIXED_START_68
        epochs = offset + CONT_LEN_68
    row["epsilon_schedule.enabled"] = "true"
    row["epsilon_schedule.type"] = "warmup_ramp_fixed"
    row["epsilon_schedule.source_epsilon"] = source_eps
    row["epsilon_schedule.target_epsilon"] = target_eps
    row["epsilon_schedule.warmup_epochs"] = offset + CONT_WARMUP
    row["epsilon_schedule.ramp_start_epoch"] = offset + CONT_RAMP_START
    row["epsilon_schedule.ramp_end_epoch"] = ramp_end
    row["epsilon_schedule.fixed_start_epoch"] = fixed_start
    return epochs


# --------------------------------------------------------------------------
def _adv_family(sink: RowSink, arch: str, init: int, *, criterion: str, dvd: bool):
    crit_tag = "trades" if criterion == "trades" else ""
    dvd_prefix = "dvd_b_" if dvd else ""
    dvd_variant = "dvd-b" if dvd else ""
    scratch_epochs = DVD_SCRATCH_EPOCHS if dvd else ADV_SCRATCH_EPOCHS
    compile_ok = criterion == "madry"  # compile only for madry & dvd_madry
    protocol = "trades" if criterion == "trades" else "madry"
    bs = batch_size(protocol=protocol, dvd=dvd, criterion=criterion)

    def attack_cols(row: dict, norm: str, eps, *, target: bool = False):
        row["attacks.advtrain"] = "True"
        row["attacks.attack_criterion"] = criterion
        row["attacks.attack_norm"] = norm
        row["attacks.attack_eps"] = eps
        if dvd:
            row["dataset.dvd.enabled"] = "true"
            row["dataset.dvd.variant"] = "dvd-b"
        if compile_ok:
            row["model.compile_model"] = "True"
        row["checkpointing.save_best_adv"] = "true"

    for norm in NORMS:
        nt = f"{norm}{crit_tag}"
        for eps in (1, 2, 4):
            name = f"{arch}_{dvd_prefix}{nt}_{eps}_init{init}"
            kw = {c: "" for c in OVERRIDE_COLUMNS}
            _common(kw, arch=arch, name=name, init=init, epochs=scratch_epochs, bs=bs, v1=False)
            attack_cols(kw, norm, eps)
            sink.add(model_name=name, arch=arch, init=init, protocol=protocol,
                     init_mode="scratch", threat_norm=norm, threat_eps=eps,
                     notes=f"{protocol}{'+dvd_b' if dvd else ''} {norm} eps{eps} scratch", **kw)

        scratch4 = f"{arch}_{dvd_prefix}{nt}_4_init{init}"
        variants = (("resetepoch", "continuation"), ("contepoch", "resume")) if dvd \
            else (("", "continuation"),)

        for variant_tag, init_mode in variants:
            suffix = f"_{variant_tag}" if variant_tag else ""
            cont48_name = None
            cont48_epochs = scratch_epochs
            for src, tgt in ((4, 6), (4, 8), (8, 12)):
                dep = scratch4 if src == 4 else cont48_name
                offset = (scratch_epochs if src == 4 else cont48_epochs) if init_mode == "resume" else 0
                name = f"{arch}_{dvd_prefix}{nt}_cont{src}to{tgt}_init{init}{suffix}"
                kw = {c: "" for c in OVERRIDE_COLUMNS}
                attack_cols(kw, norm, tgt, target=True)
                if init_mode == "continuation":
                    kw["continuation.enabled"] = "true"
                kw["continuation.use_ema"] = "true"
                epochs = _schedule_cols(kw, src, tgt, offset)
                _common(kw, arch=arch, name=name, init=init, epochs=epochs, bs=bs, v1=False)
                sink.add(model_name=name, arch=arch, init=init, protocol=protocol,
                         init_mode=init_mode, epoch_variant=variant_tag,
                         dependency_model_name=dep, threat_norm=norm, threat_eps=tgt,
                         notes=f"cont {src}->{tgt} from {dep} ({init_mode})", **kw)
                if tgt == 8:
                    cont48_name, cont48_epochs = name, epochs


def _v1_family(sink: RowSink, arch: str, init: int):
    bs = batch_size(protocol="v1", dvd=False, criterion="madry")
    norm = "l2"

    def base_cols(row: dict, eps):
        row["attacks.advtrain"] = "True"
        row["attacks.attack_criterion"] = "madry"
        row["attacks.attack_domain"] = "v1_feature"
        row["attacks.attack_norm"] = norm
        row["attacks.v1_attack_eps"] = eps
        row["checkpointing.save_best_adv"] = "true"

    for eps in (1, 2, 4):
        name = f"{arch}_v1_l2_{eps}_init{init}"
        kw = {c: "" for c in OVERRIDE_COLUMNS}
        _common(kw, arch=arch, name=name, init=init, epochs=ADV_SCRATCH_EPOCHS, bs=bs, v1=True)
        base_cols(kw, eps)
        sink.add(model_name=name, arch=arch, init=init, protocol="v1", init_mode="scratch",
                 threat_norm=norm, threat_eps=eps, notes=f"v1 l2 eps{eps} scratch", **kw)

    scratch4 = f"{arch}_v1_l2_4_init{init}"
    cont48 = None
    for src, tgt in ((4, 6), (4, 8), (8, 12)):
        dep = scratch4 if src == 4 else cont48
        name = f"{arch}_v1_l2_cont{src}to{tgt}_init{init}"
        kw = {c: "" for c in OVERRIDE_COLUMNS}
        base_cols(kw, tgt)
        kw["continuation.enabled"] = "true"
        kw["continuation.use_ema"] = "true"
        epochs = _schedule_cols(kw, src, tgt, 0)
        _common(kw, arch=arch, name=name, init=init, epochs=epochs, bs=bs, v1=True)
        sink.add(model_name=name, arch=arch, init=init, protocol="v1", init_mode="continuation",
                 dependency_model_name=dep, threat_norm=norm, threat_eps=tgt,
                 notes=f"v1 cont {src}->{tgt} from {dep}", **kw)
        if tgt == 8:
            cont48 = name


def _gradnorm_family(sink: RowSink, arch: str, init: int, baseline_name: str):
    bs = batch_size(protocol="gradnorm", dvd=False, criterion="madry")
    total = BASELINE_EPOCHS + GRADNORM_EXTRA_EPOCHS
    for eps in (1, 2, 4, 6, 8, 12):
        name = f"{arch}_gradnorm_l1_{eps}_init{init}"
        kw = {c: "" for c in OVERRIDE_COLUMNS}
        kw["attacks.gradnorm"] = "True"
        kw["attacks.gradnorm_penalty_norm"] = "l1"
        kw["attacks.attack_eps"] = eps
        kw["checkpointing.save_best_adv"] = "true"
        _common(kw, arch=arch, name=name, init=init, epochs=total, bs=bs, v1=False)
        sink.add(model_name=name, arch=arch, init=init, protocol="gradnorm", init_mode="resume",
                 dependency_model_name=baseline_name, threat_norm="linf", threat_eps=eps,
                 notes=f"gradnorm l1 eps{eps} resume from {baseline_name} (150->200)", **kw)


def build_arch_rows(arch: str) -> list[dict]:
    sink = RowSink()
    for init in INITS:
        bname = f"{arch}_baseline_init{init}"
        kw = {c: "" for c in OVERRIDE_COLUMNS}
        _common(kw, arch=arch, name=bname, init=init, epochs=BASELINE_EPOCHS,
                bs=batch_size(protocol="baseline", dvd=False, criterion=""), v1=False)
        kw["attacks.advtrain"] = "False"
        sink.add(model_name=bname, arch=arch, init=init, protocol="baseline",
                 init_mode="scratch", notes="clean baseline (no adv)", **kw)

        dname = f"{arch}_dvd_b_baseline_init{init}"
        kw = {c: "" for c in OVERRIDE_COLUMNS}
        _common(kw, arch=arch, name=dname, init=init, epochs=DVD_SCRATCH_EPOCHS,
                bs=batch_size(protocol="dvd_b_baseline", dvd=True, criterion=""), v1=False)
        kw["attacks.advtrain"] = "False"
        kw["dataset.dvd.enabled"] = "true"
        kw["dataset.dvd.variant"] = "dvd-b"
        sink.add(model_name=dname, arch=arch, init=init, protocol="dvd_b_baseline",
                 init_mode="scratch", notes="DVD-B augmentation, no adv", **kw)

        _adv_family(sink, arch, init, criterion="madry", dvd=False)
        _adv_family(sink, arch, init, criterion="trades", dvd=False)
        _adv_family(sink, arch, init, criterion="madry", dvd=True)
        _adv_family(sink, arch, init, criterion="trades", dvd=True)
        _v1_family(sink, arch, init)
        _gradnorm_family(sink, arch, init, bname)
    return sink.rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the permanent AIRCC model CSVs.")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "csv")
    args = ap.parse_args()
    for arch in ARCHES:
        rows = build_arch_rows(arch)
        seen: set[str] = set()
        for row in rows:
            dep = row["dependency_model_name"]
            if dep and dep not in seen:
                raise AssertionError(f"{arch}: dependency '{dep}' for '{row['model_name']}' not seeded earlier")
            seen.add(row["model_name"])
        out = args.out_dir / f"{arch}.csv"
        write_csv(out, rows)
        print(f"[generate_csvs] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
