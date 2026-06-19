from model_scheduler import Model_scheduler

# Connect to your scheduler database
sch = Model_scheduler("/home/ashtomer/projects/ares/job_manager/model_scheduler.db")

# Reset all models to 'waiting'
sch._execute_sqlite("UPDATE models SET status = 'waiting'")

print("[DONE] Reset all models to 'waiting'.")
