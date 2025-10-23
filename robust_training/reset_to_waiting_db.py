# from adv_scheduler import TaskScheduler

# sch = TaskScheduler("adv_scheduler.db")

# # List of model IDs you want to reset
# models_to_reset = [
#     "norm=linf|c=4|adv=1|init=2",
#     "norm=l2|c=1|adv=1|init=2",
#     "c=0|adv=0"
# ]

# for mid in models_to_reset:
#     sch._execute_sqlite(
#         "UPDATE models SET status = 'waiting' WHERE model_id = ?",
#         (mid,)
#     )
#     print(f"[INFO] Reset {mid} to waiting")

# print("[DONE] Selected models have been reset.")
