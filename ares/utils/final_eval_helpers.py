import csv
import json
import os
from pathlib import Path

import torch

from ares.utils.continuation import current_active_epsilon_user
from ares.utils.epsilon_schedule import DEFAULT_EPS_INPUTS


DEFAULT_EPS_INPUTS_CSV = ",".join(str(int(eps)) for eps in DEFAULT_EPS_INPUTS)
AA_FULL_SWEEP_NORMS = ("linf", "l2", "l1")


def _parse_csv_list(value):
    if value is None:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _parse_float_list(value):
    return [float(v) for v in _parse_csv_list(value)]


def _parse_aatype(value):
    aatype = str(value or "eps_norm").strip().lower()
    allowed = {"full_sweep", "eps_norm"}
    if aatype not in allowed:
        raise ValueError(f"Unsupported final_eval_aatype: {aatype}")
    return aatype


def _parse_checkpoint_kinds(value):
    kinds = [v.lower() for v in _parse_csv_list(value)]
    kinds = kinds or ["best"]
    allowed = {"best", "last", "advbest"}
    unknown = [kind for kind in kinds if kind not in allowed]
    if unknown:
        raise ValueError(f"Unsupported final_eval_aa_checkpoint_kinds: {', '.join(unknown)}")
    return kinds


def _to_abs_norm(path):
    return os.path.normpath(os.path.abspath(path))


def _float_eq(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def _checkpoint_matches(row_checkpoint_path, target_checkpoint_path):
    row_raw = str(row_checkpoint_path or "").strip()
    if not row_raw:
        return False

    row_norm = os.path.normpath(row_raw)
    row_abs = _to_abs_norm(row_raw)
    target_norm = os.path.normpath(str(target_checkpoint_path))
    target_abs = _to_abs_norm(str(target_checkpoint_path))

    if row_abs == target_abs or row_norm == target_norm:
        return True
    if row_abs.endswith(target_norm) or target_abs.endswith(row_norm):
        return True
    return os.path.basename(row_norm) == os.path.basename(target_norm)


def _read_csv_rows(csv_path, _logger):
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        _logger.warning("Failed reading final eval csv %s: %s", csv_path, exc)
        return []


def _pgd_completion_csv_candidates(out_dir, csv_name):
    candidates = [
        os.path.join(out_dir, csv_name),
        os.path.join(out_dir, "pgd_eval", csv_name),
    ]
    deduped = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def _is_pgd_eval_complete(cfg, out_dir, ckpt_path, _logger):
    csv_name = cfg.get("final_eval_pgd_completion_csv", "pgd_validation_results.csv")

    expected_norms = [n.lower() for n in _parse_csv_list(cfg.get("final_eval_pgd_norms", "linf,l2,l1"))]
    expected_eps = _parse_float_list(cfg.get("final_eval_pgd_eps", DEFAULT_EPS_INPUTS_CSV))
    expected_pairs = {(n, round(float(e), 10)) for n in expected_norms for e in expected_eps}

    observed_pairs = set()
    found_any_rows = False
    for csv_path in _pgd_completion_csv_candidates(out_dir, csv_name):
        rows = _read_csv_rows(csv_path, _logger)
        if not rows:
            continue
        found_any_rows = True
        for row in rows:
            if not _checkpoint_matches(row.get("checkpoint_path"), ckpt_path):
                continue
            attack_norm = str(row.get("attack_norm", "")).strip().lower()
            eps_input = row.get("epsilon_input")
            if not attack_norm or eps_input in (None, ""):
                continue
            try:
                observed_pairs.add((attack_norm, round(float(eps_input), 10)))
            except (TypeError, ValueError):
                continue

    if not found_any_rows:
        return False

    missing_count = len(expected_pairs - observed_pairs)
    if missing_count:
        _logger.info(
            "PGD final eval incomplete for %s: missing %d/%d expected norm/eps pairs.",
            ckpt_path,
            missing_count,
            len(expected_pairs),
        )
        return False

    _logger.info("PGD final eval already complete for %s. Skipping PGD.", ckpt_path)
    return True


def _resolve_expected_aa_attack(cfg, ckpt_path, parse_attack_from_path):
    aa_norm = cfg.get("final_eval_aa_norm", None)
    aa_eps = cfg.get("final_eval_aa_eps", None)
    if aa_norm is not None or aa_eps is not None:
        if aa_norm is None or aa_eps is None:
            return False, None, None
        return False, str(aa_norm), float(aa_eps)
    return parse_attack_from_path(str(ckpt_path))


def _is_aa_eval_complete(cfg, out_dir, ckpt_path, parse_attack_from_path, _logger):
    csv_name = cfg.get("final_eval_aa_completion_csv", "autoattack_results.csv")
    csv_path = os.path.join(out_dir, csv_name)
    rows = _read_csv_rows(csv_path, _logger)
    if not rows:
        return False

    skip_auto, expected_norm, expected_eps = _resolve_expected_aa_attack(cfg, ckpt_path, parse_attack_from_path)
    if skip_auto:
        _logger.info("AA baseline checkpoint detected for %s. Skipping AA.", ckpt_path)
        return True
    if expected_norm is None or expected_eps is None:
        return False

    expected_norm = str(expected_norm).strip().lower()

    for row in rows:
        if not _checkpoint_matches(row.get("checkpoint_path"), ckpt_path):
            continue
        row_norm = str(row.get("attack_norm", "")).strip().lower()
        row_eps = row.get("epsilon_eval")
        if not row_norm or row_eps in (None, ""):
            continue
        try:
            if row_norm == expected_norm and _float_eq(float(row_eps), expected_eps, tol=1e-9):
                _logger.info(
                    "AA final eval already complete for %s (norm=%s eps=%s). Skipping AA.",
                    ckpt_path,
                    expected_norm,
                    expected_eps,
                )
                return True
        except (TypeError, ValueError):
            continue

    _logger.info(
        "AA final eval incomplete for %s (need norm=%s eps=%s).",
        ckpt_path,
        expected_norm,
        expected_eps,
    )
    return False


def _maybe_plot_aa_checkpoint_comparison(output_dir, aa_eps_inputs, _logger):
    """Generate the best/last/advbest AA heatmap plot right after the sweep completes."""
    try:
        from data_analysis.plot_autoattack_comparation import (
            load_autoattack_results,
            model_entries,
            plot_comparation,
        )
    except Exception as exc:
        _logger.warning("AA checkpoint comparison plot skipped (import failed): %s", exc)
        return

    try:
        models_root = Path(output_dir).parent
        model_name = Path(output_dir).name
        eval_eps_order = sorted({0.0} | set(aa_eps_inputs))

        # Collect whichever checkpoint CSVs actually exist.
        entries = model_entries(models_root, [model_name], "all")
        if not entries:
            _logger.info("AA comparison plot: no checkpoint CSVs found under %s, skipping.", output_dir)
            return

        data_by_norm = load_autoattack_results(models_root, entries, eval_eps_order)
        model_labels = [label for _name, label, _csv in entries]
        out_png = Path(output_dir) / f"autoattack_eval_comparation_{model_name}.png"
        plot_comparation(data_by_norm, model_labels, eval_eps_order, model_name, out_png)
        _logger.info("Saved AA checkpoint comparison plot to %s", out_png)
    except Exception as exc:
        _logger.warning("AA checkpoint comparison plot failed (ignored): %s", exc)


def _write_eps_norm_score_summary(
    output_dir,
    checkpoint_kinds,
    aa_csv_name,
    aa_norm,
    aa_eps_input,
    find_checkpoint_for_kind,
    output_csv_for_checkpoint_kind,
    _logger,
):
    """Write {checkpoint_filename: robust_acc} for the trained norm/eps AA eval."""
    scores = {}
    for checkpoint_kind in checkpoint_kinds:
        kind_ckpt = find_checkpoint_for_kind(Path(output_dir), checkpoint_kind)
        if kind_ckpt is None:
            continue
        kind_csv_name = output_csv_for_checkpoint_kind(aa_csv_name, checkpoint_kind)
        rows = _read_csv_rows(os.path.join(output_dir, kind_csv_name), _logger)
        for row in rows:
            if not _checkpoint_matches(row.get("checkpoint_path"), kind_ckpt):
                continue
            row_norm = str(row.get("attack_norm", "")).strip().lower()
            row_eps = row.get("epsilon_input")
            if row_norm != aa_norm or row_eps in (None, ""):
                continue
            try:
                if not _float_eq(float(row_eps), aa_eps_input, tol=1e-9):
                    continue
                scores[os.path.basename(str(kind_ckpt))] = float(row.get("robust_acc"))
            except (TypeError, ValueError):
                continue
            break

    if not scores:
        _logger.info("No trained-norm/eps AutoAttack scores found under %s; skipping JSON summary.", output_dir)
        return

    json_path = os.path.join(output_dir, "autoattack_eps_norm_scores.json")
    with open(json_path, "w") as f:
        json.dump({"attack_norm": aa_norm, "epsilon_input": aa_eps_input, "scores": scores}, f, indent=2)
    _logger.info("Wrote trained-norm/eps AutoAttack score summary to %s", json_path)


def _maybe_run_final_eval(cfg, output_dir, _logger, did_train_this_run=True):
    if not cfg.get("final_eval", False):
        return

    if not (cfg.get("final_eval_autoattack", False) or cfg.get("final_eval_pgd", False)):
        _logger.warning("Final eval enabled but no eval type selected. Use --final-eval-autoattack and/or --final-eval-pgd.")
        return

    ckpt_name = cfg.get("final_eval_ckpt_name", "model_best.pth.tar")
    ckpt_path = os.path.join(output_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        _logger.error("Final eval checkpoint not found: %s", ckpt_path)
        return

    val_dir = cfg.get("final_eval_val_dir", "") or cfg.dataset.eval_dir
    if not val_dir:
        _logger.error("Final eval requires --eval-dir or --final-eval-val-dir.")
        return

    configured_out_dir = str(cfg.get("final_eval_out_dir", "") or "").strip()
    if configured_out_dir:
        # Relative final_eval_out_dir should live under the model output dir.
        out_dir = configured_out_dir if os.path.isabs(configured_out_dir) else os.path.join(output_dir, configured_out_dir)
    else:
        out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        _logger.error("Final eval requires CUDA (validate() uses .cuda()).")
        return

    try:
        aa_bs = cfg.get("final_eval_aa_batch_size", None) or 64
        aa_num_batches = cfg.get("final_eval_aa_max_batches", None) or 2
        aa_eps_inputs = _parse_float_list(cfg.get("final_eval_aa_eps_inputs", DEFAULT_EPS_INPUTS_CSV))
        aa_csv_name = cfg.get("final_eval_aa_completion_csv", "autoattack_sweep_results.csv")
        aa_checkpoint_kinds = _parse_checkpoint_kinds(cfg.get("final_eval_aa_checkpoint_kinds", "best,last,advbest"))
        pgd_bs = cfg.get("final_eval_pgd_batch_size", None) or cfg.training.batch_size
        run_aa_sweep = bool(cfg.get("final_eval_autoattack", False))
        run_pgd = bool(cfg.get("final_eval_pgd", False))
        aa_runner = None
        aa_import_failed = False
        aa_norms = list(AA_FULL_SWEEP_NORMS)
        resolved_trained_norm_eps = False
        all_aa_checkpoint_kinds = list(aa_checkpoint_kinds)

        if cfg.get("final_eval_require_train_this_run", False) and not did_train_this_run:
            _logger.info("No epochs trained in this run; skipping final eval due to final_eval_require_train_this_run=true.")
            return

        if run_aa_sweep:
            aatype = _parse_aatype(cfg.get("final_eval_aatype", "eps_norm"))
            if aatype == "eps_norm":
                try:
                    aa_norms = [str(cfg.attacks.attack_norm)]
                    aa_eps_inputs = [float(current_active_epsilon_user(cfg))]
                    resolved_trained_norm_eps = True
                except Exception as exc:
                    _logger.warning(
                        "final_eval_aatype=eps_norm but the trained attack norm/eps could not be "
                        "resolved (%s); running the full AA sweep instead.", exc,
                    )

            if cfg.get("final_eval_aa_norm", None) is not None or cfg.get("final_eval_aa_eps", None) is not None:
                _logger.warning(
                    "final_eval_aa_norm/final_eval_aa_eps are ignored for final AutoAttack sweep; "
                    "running %s x eps %s.",
                    ",".join(aa_norms),
                    ",".join(str(eps).rstrip("0").rstrip(".") for eps in aa_eps_inputs),
                )
            try:
                from data_analysis.autoattack_array_eval import (
                    find_checkpoint_for_kind,
                    is_complete_output,
                    output_csv_for_checkpoint_kind,
                    run_autoattack_sweep_for_checkpoint,
                )

                aa_runner = run_autoattack_sweep_for_checkpoint
            except Exception as exc:
                _logger.exception("Failed to import AutoAttack sweep runner: %s", exc)
                aa_import_failed = True
                run_aa_sweep = False

        if cfg.get("final_eval_skip_if_complete", True):
            if run_pgd and _is_pgd_eval_complete(cfg, out_dir, ckpt_path, _logger):
                run_pgd = False
            if run_aa_sweep:
                pending_kinds = []
                for checkpoint_kind in aa_checkpoint_kinds:
                    kind_ckpt = find_checkpoint_for_kind(Path(output_dir), checkpoint_kind)
                    if kind_ckpt is None:
                        _logger.info("AutoAttack checkpoint kind=%s missing under %s. Skipping.", checkpoint_kind, output_dir)
                        continue
                    kind_csv_name = output_csv_for_checkpoint_kind(aa_csv_name, checkpoint_kind)
                    if is_complete_output(
                        Path(output_dir) / kind_csv_name,
                        max_settings=None,
                        checkpoint_path=kind_ckpt,
                        model_name=os.path.basename(output_dir),
                        eps_inputs=aa_eps_inputs,
                        norms=aa_norms,
                    ):
                        _logger.info("AutoAttack sweep already complete for %s. Skipping AutoAttack.", kind_ckpt)
                        continue
                    pending_kinds.append(checkpoint_kind)
                aa_checkpoint_kinds = pending_kinds
                run_aa_sweep = bool(aa_checkpoint_kinds)

        if not run_aa_sweep and not run_pgd:
            if aa_import_failed:
                _logger.error("No final AutoAttack sweep ran because the AutoAttack sweep runner failed to import.")
                return
            _logger.info("All requested final eval outputs already complete for %s. Skipping final eval.", ckpt_path)
            return

        if run_pgd:
            try:
                from data_analysis.final_eval import run_final_evaluation
            except Exception as exc:
                _logger.exception("Failed to import final evaluation runner: %s", exc)
                run_pgd = False
            else:
                run_final_evaluation(
                    checkpoint_path=ckpt_path,
                    models_dir=None,
                    val_dir=val_dir,
                    device=device,
                    out_dir=out_dir,
                    aa=False,
                    pgd=True,
                    aa_batch_size=int(aa_bs),
                    aa_norm=None,
                    aa_eps=None,
                    aa_max_batches=None,
                    pgd_batch_size=int(pgd_bs),
                    pgd_eps=_parse_float_list(cfg.get("final_eval_pgd_eps", DEFAULT_EPS_INPUTS_CSV)),
                    pgd_norms=_parse_csv_list(cfg.get("final_eval_pgd_norms", "linf,l2,l1")),
                    pgd_attack_steps=int(cfg.get("final_eval_pgd_attack_steps", 10)),
                    pgd_max_batches=cfg.get("final_eval_pgd_max_batches", None),
                    pgd_output_csv=cfg.get("final_eval_pgd_completion_csv", "pgd_validation_results.csv"),
                    l1_step_mode="l1_apgd",
                    l1_apgd_rho=float(cfg.attacks.get("l1_apgd_rho", 0.05)),
                    l1_apgd_use_halving=bool(cfg.attacks.get("l1_apgd_use_halving", True)),
                    l1_apgd_min_step_scale=float(cfg.attacks.get("l1_apgd_min_step_scale", 0.01)),
                    plots=bool(cfg.get("final_eval_plots", False)),
                    plot_x_col=cfg.get("final_eval_plot_x_col", "pgd_constrained_eps"),
                    num_workers=int(cfg.get("final_eval_num_workers", 8)),
                )

        if run_aa_sweep and aa_runner is not None:
            for checkpoint_kind in aa_checkpoint_kinds:
                kind_ckpt = find_checkpoint_for_kind(Path(output_dir), checkpoint_kind)
                if kind_ckpt is None:
                    _logger.info("AutoAttack checkpoint kind=%s missing under %s. Skipping.", checkpoint_kind, output_dir)
                    continue
                kind_csv_name = output_csv_for_checkpoint_kind(aa_csv_name, checkpoint_kind)
                aa_runner(
                    checkpoint_path=kind_ckpt,
                    model_dir=output_dir,
                    val_dir=val_dir,
                    device=device,
                    batch_size=int(aa_bs),
                    num_batches=int(aa_num_batches),
                    num_workers=int(cfg.get("final_eval_num_workers", 8)),
                    seed=int(cfg.seed or 0),
                    output_csv=kind_csv_name,
                    checkpoint_kind=checkpoint_kind,
                    eps_inputs=aa_eps_inputs,
                    norms=aa_norms,
                    force=not bool(cfg.get("final_eval_skip_if_complete", True)),
                    logger=_logger,
                )
            if resolved_trained_norm_eps:
                _write_eps_norm_score_summary(
                    output_dir=output_dir,
                    checkpoint_kinds=all_aa_checkpoint_kinds,
                    aa_csv_name=aa_csv_name,
                    aa_norm=aa_norms[0],
                    aa_eps_input=aa_eps_inputs[0],
                    find_checkpoint_for_kind=find_checkpoint_for_kind,
                    output_csv_for_checkpoint_kind=output_csv_for_checkpoint_kind,
                    _logger=_logger,
                )
            else:
                _maybe_plot_aa_checkpoint_comparison(output_dir, aa_eps_inputs, _logger)
            # aircc job manager: pick the best checkpoint by the model's threat
            # model and record it in the AIRCC DB. No-op when the job is untracked
            # (AIRCC_MODEL_ID/AIRCC_DB unset).
            try:
                from aircc.aircc_job_manager import progress as _aircc_progress
                _aircc_progress.write_best_checkpoint(output_dir)
            except Exception:  # pragma: no cover - never block on the handoff
                _logger.warning("aircc best-checkpoint write failed (ignored)")
    except Exception as exc:
        _logger.exception("Final eval failed: %s", exc)
