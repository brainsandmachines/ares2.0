import sqlite3, time
conn = sqlite3.connect('/home/ashtomer/projects/ares/orchestrator/orchestrator.db')
ts = int(time.time())
for model_id, job_id, node in [
    ('l2_2_init1', 18258867, 'rtx6000'),
    ('l2_4_init1', 18258933, 'rtx6000'),
]:
    conn.execute(
        "UPDATE models_queue SET status='RUNNING', slurm_job_id=?, cluster_node=?, last_update_ts=? WHERE model_id=?",
        (job_id, node, ts, model_id)
    )
    print(f"  marked {model_id} RUNNING job={job_id} node={node}")
conn.commit()
conn.close()
