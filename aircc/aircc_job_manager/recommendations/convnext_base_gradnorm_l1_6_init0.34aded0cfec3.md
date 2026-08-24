### 1. Root cause

The training subprocess received external `SIGTERM` (`rc=-15`) at 2026-08-14 14:59:11; training was healthy immediately beforehand, with no traceback, CUDA OOM, or configuration error.

### 2. Suggested fix

Requeue the DB row as `pending`; it should resume from `last.pth.tar`—saved after epoch 125—losing only the unfinished portion of epoch 126. No CSV/Hydra change appears necessary.

```sql
UPDATE jobs
SET status='pending', owner_task=NULL, claimed_ts=NULL, last_error=NULL
WHERE model_name='convnext_base_gradnorm_l1_6_init0';
```

If SIGTERM recurs, inspect AIRCC/Slurm accounting for job `153201_7`; the manager remained alive and immediately claimed another model, so the allocation itself did not end.

### 3. Exact inspection/change targets

- Failure record: [jm_153201_7.out](</home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/logs/jm_153201_7.out:40696>)
- Stderr—no corresponding failure: [jm_153201_7.err](</home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/logs/jm_153201_7.err:48>)
- Last healthy batch: [log.txt](</home/tomer_a/aircc_mount/ashtomer/ares/results/models/convnext_base_gradnorm_l1_6_init0/log.txt:3478>)
- Resume checkpoint: `/home/tomer_a/aircc_mount/ashtomer/ares/results/models/convnext_base_gradnorm_l1_6_init0/last.pth.tar`
- DB row to requeue: `/home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/aircc_jobs.sqlite`
- CSV specification—inspect, but do not change: [convnext_base.csv](</home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/csv/convnext_base.csv:103>)
- Effective resume/epoch config: [runtime_config.yaml](</home/tomer_a/aircc_mount/ashtomer/ares/results/models/convnext_base_gradnorm_l1_6_init0/runtime_config.yaml:25>)
- Resume-offset metadata: [.aircc_resume_target.json](</home/tomer_a/aircc_mount/ashtomer/ares/results/models/convnext_base_gradnorm_l1_6_init0/.aircc_resume_target.json>)
- Return-code handling: [lifecycle.py](</home/tomer_a/aircc_mount/ashtomer/ares/aircc/aircc_job_manager/lifecycle.py:273>)