from model_scheduler import Model_scheduler

sch = Model_scheduler("model_scheduler.db")
df = sch.list_models_df()
df.to_csv("model_status.csv", index=False)
print(df)
