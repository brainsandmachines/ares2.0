import csv
import os
from pathlib import Path

import torch


def _parse_csv_list(value):
    if value is None:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _parse_float_list(value):
    return [float(v) for v in _parse_csv_list(value)]


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


def _is_pgd_eval_complete(args, out_dir, ckpt_path, _logger):
    csv_name = getattr(args, "final_eval_pgd_completion_csv", "pgd_validation_results.csv")

    expected_norms = [n.lower() for n in _parse_csv_list(getattr(args, "final_eval_pgd_norms", "linf,l2,l1"))]
    expected_eps = _parse_float_list(getattr(args, "final_eval_pgd_eps", "0.5,1,2,4,8,16"))
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


def _resolve_expected_aa_attack(args, ckpt_path, parse_attack_from_path):
    aa_norm = getattr(args, "final_eval_aa_norm", None)
    aa_eps = getattr(args, "final_eval_aa_eps", None)
    if aa_norm is not None or aa_eps is not None:
        if aa_norm is None or aa_eps is None:
            return False, None, None
        return False, str(aa_norm), float(aa_eps)
    return parse_attack_from_path(str(ckpt_path))


def _is_aa_eval_complete(args, out_dir, ckpt_path, parse_attack_from_path, _logger):
    csv_name = getattr(args, "final_eval_aa_completion_csv", "autoattack_results.csv")
    csv_path = os.path.join(out_dir, csv_name)
    rows = _read_csv_rows(csv_path, _logger)
    if not rows:
        return False

    skip_auto, expected_norm, expected_eps = _resolve_expected_aa_attack(args, ckpt_path, parse_attack_from_path)
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


def _maybe_run_final_eval(args, output_dir, _logger, did_train_this_run=True):
    if not getattr(args, "final_eval", False):
        return

    if not (getattr(args, "final_eval_autoattack", False) or getattr(args, "final_eval_pgd", False)):
        _logger.warning("Final eval enabled but no eval type selected. Use --final-eval-autoattack and/or --final-eval-pgd.")
        return

    ckpt_name = getattr(args, "final_eval_ckpt_name", "model_best.pth.tar")
    ckpt_path = os.path.join(output_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        _logger.error("Final eval checkpoint not found: %s", ckpt_path)
        return

    val_dir = getattr(args, "final_eval_val_dir", "") or getattr(args, "eval_dir", "")
    if not val_dir:
        _logger.error("Final eval requires --eval-dir or --final-eval-val-dir.")
        return

    configured_out_dir = str(getattr(args, "final_eval_out_dir", "") or "").strip()
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
        aa_bs = getattr(args, "final_eval_aa_batch_size", None) or 64
        aa_num_batches = getattr(args, "final_eval_aa_max_batches", None) or 2
        aa_csv_name = getattr(args, "final_eval_aa_completion_csv", "autoattack_sweep_results.csv")
        pgd_bs = getattr(args, "final_eval_pgd_batch_size", None) or getattr(args, "batch_size", 64)
        run_aa_sweep = bool(getattr(args, "final_eval_autoattack", False))
        run_pgd = bool(getattr(args, "final_eval_pgd", False))
        aa_runner = None
        aa_import_failed = False

        if getattr(args, "final_eval_require_train_this_run", False) and not did_train_this_run:
            _logger.info("No epochs trained in this run; skipping final eval due to final_eval_require_train_this_run=true.")
            return

        if run_aa_sweep:
            if getattr(args, "final_eval_aa_norm", None) is not None or getattr(args, "final_eval_aa_eps", None) is not None:
                _logger.warning(
                    "final_eval_aa_norm/final_eval_aa_eps are ignored for final AutoAttack sweep; "
                    "running linf,l2,l1 x eps 1,2,4,8,16."
                )
            try:
                from data_analysis.autoattack_array_eval import is_complete_output, run_autoattack_sweep_for_checkpoint

                aa_runner = run_autoattack_sweep_for_checkpoint
            except Exception as exc:
                _logger.exception("Failed to import AutoAttack sweep runner: %s", exc)
                aa_import_failed = True
                run_aa_sweep = False

        if getattr(args, "final_eval_skip_if_complete", True):
            if run_pgd and _is_pgd_eval_complete(args, out_dir, ckpt_path, _logger):
                run_pgd = False
            if run_aa_sweep and is_complete_output(
                Path(output_dir) / aa_csv_name,
                max_settings=None,
                checkpoint_path=ckpt_path,
                model_name=os.path.basename(output_dir),
            ):
                _logger.info("AutoAttack sweep already complete for %s. Skipping AutoAttack.", ckpt_path)
                run_aa_sweep = False

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
                    pgd_eps=_parse_float_list(getattr(args, "final_eval_pgd_eps", "0.5,1,2,4,8,16")),
                    pgd_norms=_parse_csv_list(getattr(args, "final_eval_pgd_norms", "linf,l2,l1")),
                    pgd_attack_steps=int(getattr(args, "final_eval_pgd_attack_steps", 10)),
                    pgd_max_batches=getattr(args, "final_eval_pgd_max_batches", None),
                    pgd_output_csv=getattr(args, "final_eval_pgd_completion_csv", "pgd_validation_results.csv"),
                    l1_step_mode="l1_apgd",
                    l1_apgd_rho=float(getattr(args, "l1_apgd_rho", 0.05)),
                    l1_apgd_use_halving=bool(getattr(args, "l1_apgd_use_halving", True)),
                    l1_apgd_min_step_scale=float(getattr(args, "l1_apgd_min_step_scale", 0.01)),
                    plots=bool(getattr(args, "final_eval_plots", False)),
                    plot_x_col=getattr(args, "final_eval_plot_x_col", "pgd_constrained_eps"),
                    num_workers=int(getattr(args, "final_eval_num_workers", 8)),
                )

        if run_aa_sweep and aa_runner is not None:
            aa_runner(
                checkpoint_path=ckpt_path,
                model_dir=output_dir,
                val_dir=val_dir,
                device=device,
                batch_size=int(aa_bs),
                num_batches=int(aa_num_batches),
                num_workers=int(getattr(args, "final_eval_num_workers", 8)),
                seed=int(getattr(args, "seed", 0) or 0),
                output_csv=aa_csv_name,
                force=not bool(getattr(args, "final_eval_skip_if_complete", True)),
                logger=_logger,
            )
    except Exception as exc:
        _logger.exception("Final eval failed: %s", exc)
