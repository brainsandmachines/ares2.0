
from adv_scheduler import TaskScheduler

# Connect to your scheduler database
sch = TaskScheduler("adv_scheduler.db")

# Update all models in the table
sch._execute_sqlite("UPDATE models SET status = 'waiting'")


print("[DONE] Selected models have been reset.")
