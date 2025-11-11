#!/home/ashtomer/.conda/envs/tomer_advtrain/bin/python
import sys, os
# sys.path.append("/home/ashtomer/projects/ares")
import time
from model_scheduler import Model_scheduler

if __name__ == "__main__":
    try:
        db_path = "/home/ashtomer/projects/ares/job_manager/model_scheduler.db"
        sch = Model_scheduler(db_path=db_path)
        affected = sch.requeue_stale_trainings(threshold_hours=10)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Requeued {affected} stale jobs.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
