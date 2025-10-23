#!/home/ashtomer/.conda/envs/ares/bin/python
import sys
import time
from adv_scheduler import TaskScheduler

if __name__ == "__main__":
    try:
        db_path = "/home/ashtomer/projects/ares/robust_training/adv_scheduler.db"
        sch = TaskScheduler(db_path=db_path)
        affected = sch.requeue_stale_trainings(threshold_hours=10)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Requeued {affected} stale jobs.")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
