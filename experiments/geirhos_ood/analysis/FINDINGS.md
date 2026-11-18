# Geirhos OOD: Madry vs TRADES, threat norm, eps and shape bias

Produced by `python -m experiments.geirhos_ood.analyze` from `../results/task_metrics.csv`, the
zoo JSONs (arch, protocol, norm, eps, full-val clean accuracy with bilinear resize, AA robust
accuracy) and `data_analysis/shape_bias_results_botero.csv`.

- **Checkpoints:** 201, one per run: the Hugging Face-published set of the five contstim model
  JSONs (since 2026-11-18; until then the 199 of the retired adversarial/blackbox JSONs, see the
  README). 169 of them have shape bias. The 32 without it are the 27 vit_b_cvst/swin_b TRADES
  runs and 5 convnext_base init2 runs added on 2026-11-18 (`l2_cont4to8_init2` and four
  `dvd_b_l1/l2` init2), none of which `shape_bias_results_botero.csv` has scored.
- **Tables:** `tables/`. `model_table.csv` is the joined per-checkpoint table everything else
  is computed from.
- **Figures:** `plots/` (`.pdf` + `.png`).

## 1. TRADES keeps contrast and noise robustness; Madry loses it as eps grows

Pooled over the three archs (standard variant), eps level 1 → 8:

| | contrast | uniform noise | high-pass | shape bias | clean acc |
|---|---|---|---|---|---|
| Madry L1 | 0.95 → 0.73 | 0.67 → 0.60 | 0.55 → 0.41 | 0.60 → 0.74 | 0.78 → 0.73 |
| Madry L2 | 0.82 → 0.40 | 0.65 → 0.36 | 0.54 → 0.30 | 0.70 → 0.83 | 0.77 → 0.61 |
| Madry L∞ | 0.84 → 0.45 | 0.65 → 0.42 | 0.60 → 0.40 | 0.66 → 0.79 | 0.79 → 0.70 |
| TRADES L1 | 0.99 → 0.99 | 0.68 → 0.69 | 0.57 → 0.51 | 0.53 → 0.69 | 0.78 → 0.75 |
| TRADES L2 | 0.98 → 0.99 | 0.71 → 0.76 | 0.57 → 0.58 | 0.62 → 0.74 | 0.78 → 0.72 |
| TRADES L∞ | 0.97 → 0.99 | 0.70 → 0.75 | 0.61 → 0.61 | 0.59 → 0.72 | 0.79 → 0.74 |
| clean baselines | 0.98–0.99 | 0.55–0.69 | 0.61–0.80 | 0.27–0.57 | |
| humans | 0.66 | 0.68 | 0.63 | ~0.96 | |

- **Same pattern on every arch and variant.** In `madry_minus_trades.png`, 12 of 12
  (group × norm) rows are negative on contrast (−0.05 to −0.50) and on uniform noise (−0.03 to −0.26).
- **The gap is largest for L2, then L∞, and small for L1.** It is smallest for DVD-B L1:
  −0.05 on contrast.
- **Madry also loses power equalisation, phase noise and stylized as eps grows;** TRADES stays
  flat or improves (`eps_curves_*.png`, `eps_change.png`).
- **TRADES L2/L∞ end above the clean baseline on uniform noise** (0.75 vs 0.55–0.69). Madry
  ends well below it.

**It is not a clean-accuracy or robustness artefact.** At the same eps, Madry is *more* robust
than TRADES (higher AA at the same threat) and has slightly lower clean accuracy, so both
needed controlling for. Two checks:

- **Regression** (`clean_acc_control.csv`). It covers the 177 Madry/TRADES checkpoints of the
  standard and DVD-B variants, across 3 archs. Accuracy is regressed on TRADES, DVD-B, norm,
  arch, clean accuracy, eps and AA robust accuracy (one slope per norm).
- **Matched pairs** (`robustness_matched.png`). Each point is one matched-eps Madry/TRADES pair,
  with the task gap plotted against the AA gap; the line's value at AA gap = 0 is the gap at
  equal robustness.

| | TRADES − Madry, raw | given clean acc | given clean + eps + AA (regression) | matched pairs at equal AA |
|---|---|---|---|---|
| contrast | +0.29 | +0.22 | +0.18 ± 0.02 | +0.26 ± 0.03 |
| uniform noise | +0.16 | +0.11 | +0.09 ± 0.01 | +0.15 ± 0.02 |
| high-pass | +0.12 | +0.08 | +0.07 ± 0.01 | +0.12 ± 0.01 |
| power equalisation | +0.08 | +0.04 | +0.03 ± 0.01 | +0.08 ± 0.01 |
| phase noise | +0.06 | +0.04 | +0.04 ± 0.00 | +0.06 ± 0.01 |
| edge | −0.01 | −0.02 | −0.03 ± 0.01 | −0.01 ± 0.01 |
| silhouette | −0.01 | −0.02 | −0.03 ± 0.01 | −0.01 ± 0.01 |
| shape bias | −0.05 | −0.03 | −0.04 ± 0.01 | −0.05 ± 0.00 |

On contrast, the pairs where Madry is much more robust show an even *larger* gap, not a smaller one.

**Contrast and uniform noise go together.** Across the 201 checkpoints their Spearman ρ is
0.81 (`task_correlations.png`). Only three task pairs are tighter: phase noise/power
equalisation and eidolon II/III (0.84 each), and high-pass/power equalisation (0.82). Uniform
noise/phase noise is 0.76. Together these form one cluster: low-contrast, noise and spectrum distortions. Madry is weak on all of them; TRADES is
not.

## 2. Threat norm

- **Within Madry, L2 hurts most** on contrast, noise, high-pass, power equalisation,
  opponent colour and rotation. L∞ hurts a little less; L1 is mild. L2 also costs the most clean
  accuracy (0.77 → 0.61).
- **Within TRADES, norm barely matters for the OOD tasks.** L1 TRADES is a bit lower on uniform
  noise and high-pass than L2/L∞.
- **Shape bias rises with eps for every norm.** Madry L2 is the highest (up to 0.83), and TRADES
  L1 the lowest.

## 3. Shape bias: what it does and doesn't predict

**Raw correlations mostly reflect clean accuracy.** Shape bias is strongly anti-correlated with
clean accuracy (ρ = −0.88): the models that lose the most accuracy are the most shape-biased. So
raw shape bias also correlates negatively with most tasks: rotation −0.80, opponent colour
−0.72, high-pass −0.71, contrast −0.59.

**Controlling for clean accuracy** (partial Spearman, `shape_bias_correlations.png`) recovers the
connection Geirhos et al. expected, but only on the shape-only tasks:

| task | raw ρ | partial ρ, clean acc held fixed |
|---|---|---|
| sketch | −0.05 | **+0.52** |
| silhouette | +0.42 | **+0.46** |
| edge | −0.06 | **+0.31** |
| eidolon I | +0.50 | +0.33 |
| contrast | −0.59 | **−0.48** |
| uniform noise | −0.44 | −0.22 |
| stylized | −0.30 | +0.04 |
| mean OOD accuracy | −0.55 | −0.10 |

At equal clean accuracy, a more shape-biased model recognises sketches, silhouettes and edge
maps better, but low-contrast images worse. Shape bias does **not** predict better OOD accuracy
overall.

**Madry and TRADES raise shape bias by different routes:**
- **Within Madry:** shape bias goes with collapse on contrast (ρ −0.80), noise (−0.85),
  power equalisation (−0.81) and the mean OOD score (−0.82). Madry's extra shape bias comes
  together with the loss of contrast and noise robustness from §1.
- **Within TRADES:** shape bias is unrelated to the mean OOD score (−0.05), positive with
  uniform noise (+0.24) and strongly positive with eidolon I/II/III (+0.86 / +0.77 / +0.57).
- **The overall trade-off:** Madry gets about +0.05 more shape bias and +0.01–0.03 on
  edge/silhouette/sketch, and pays −0.09 to −0.26 on contrast and noise.

**Error consistency (human-like mistakes).** Shape bias is the best predictor of how human-like
a model's errors are: mean error consistency ρ +0.66, still +0.58 with clean accuracy held fixed.
It holds where kappa is meaningful: stylized +0.47, eidolon I/II/III +0.64/+0.61/+0.68, sketch +0.32.

## Caveats

- **eps 6 and 8 are continuation runs** that start from the eps-4 model, so the top of each eps
  curve also carries the continuation protocol.
- **eps is the level in the run name.** Its physical size differs per norm, so compare norms by
  trend, not by eps value.
- **Error consistency is near 0 when a model is near 100% correct** (there are almost no errors
  to share). That depresses TRADES's contrast kappa and every model's colour/opponent-colour
  kappa. The mean error-consistency figures include those tasks, so the TRADES −0.024 mean-EC
  deficit is partly this ceiling effect.
- **Checkpoints are not independent.** There are init pairs, and in the full sweep the eps-6/8
  runs share their eps-4 parent. Treat the OLS standard errors and p-values as optimistic; the
  effect sizes, and their consistency across 12 independent (group × norm) rows, are the
  robust evidence.
- **The vit_b_cvst/swin_b TRADES runs have no shape bias,** so every shape-bias figure is driven by
  convnext_base TRADES plus Madry from all three archs.
