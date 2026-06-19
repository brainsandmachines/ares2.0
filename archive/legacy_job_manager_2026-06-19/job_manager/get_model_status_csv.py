from model_scheduler import Model_scheduler

sch = Model_scheduler("/home/ashtomer/projects/ares/job_manager/model_scheduler.db")
df = sch.list_models_df()
df.to_csv("/home/ashtomer/projects/ares/job_manager/model_status.csv", index=False)
print(df)
