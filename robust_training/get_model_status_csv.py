from adv_scheduler import TaskScheduler

sch = TaskScheduler("adv_scheduler.db")
df = sch.list_models_df()
df.to_csv("model_status.csv", index=False)
print(df)
