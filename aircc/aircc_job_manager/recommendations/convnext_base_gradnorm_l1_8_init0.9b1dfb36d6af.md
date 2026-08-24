1. **Root cause:** GradNorm training numerically diverged at epoch 101—both CE and L1 regularization became `NaN` at batch 4658; this repeated in July at batch 4620, so it is not an AIRCC manager failure.

2. **Suggested fix:** Resume from `last.pth.tar`/`checkpoint-100.pth.tar` with a lower job-specific learning rate—set `lr_scheduler.lrb=1e-4` in the CSV row. If instability persists, ramp GradNorm from resumed epoch 101 or reduce `gradnorm_max_reg_to_ce_ratio` below `1.0`. Do not retry unchanged.

3. **Inspect/change:**

   - Failure traceback: [jm_154597_1.err](</home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/logs/jm_154597_1.err:449>)
   - Launch and pre-failure progression: [jm_154597_1.out](</home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/logs/jm_154597_1.out:63049>)
   - Loss increase near failure: [log.txt](</home/tomer_a/aircc_mount/ashtomer/ares/results/models/convnext_base_gradnorm_l1_8_init0/log.txt>)
   - Change the model row’s final `lr_scheduler.lrb` field from blank to `1e-4`: [convnext_base.csv](</home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/csv/convnext_base.csv:104>)
   - Verify effective LR, AMP, no gradient clipping, and GradNorm cap: [runtime_config.yaml](</home/tomer_a/aircc_mount/ashtomer/ares/results/models/convnext_base_gradnorm_l1_8_init0/runtime_config.yaml:95>)
   - Resume checkpoint: [last.pth.tar](</home/tomer_a/aircc_mount/ashtomer/ares/results/models/convnext_base_gradnorm_l1_8_init0/last.pth.tar>)
   - GradNorm computation/NaN guard: [train_loop.py](</home/tomer_a/aircc_mount/ashtomer/ares/ares/utils/train_loop.py:128>)

No files were modified.