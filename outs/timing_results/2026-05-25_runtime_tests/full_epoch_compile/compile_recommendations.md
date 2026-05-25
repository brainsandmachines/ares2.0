# Runtime Compile Decisions

Use `full_epoch_compile_comparisons.csv` as the clean decision table.

- Enable compile for pixel Madry.
- Enable compile for V1 TRADES on `rtx_pro_6000` only.
- Do not enable compile for pixel TRADES.
- Do not enable compile for V1 Madry.

The detailed row-level data, including failed attempts, is in `full_epoch_compile_summary.csv`; timing phase splits are in `full_epoch_phase_breakdown.csv`.
