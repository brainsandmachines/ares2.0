import sqlite3, subprocess

MODELS_ROOT = "/home/ashtomer/projects/ares/results/models"
OLD_MODELS  = f"{MODELS_ROOT}/old_models"

TO_REMOVE_DB = [
    "gradnorm_l1_1_init1",
    "gradnorm_l2_2_init1",
    "gradnorm_l2_4_init1",
    "gradnorm_l2_8_init1",
    "l2_cont4to8_direct_init1",
    "l2_cont8to16_direct_init1",
    "l2trades_cont4to8_direct_init1",
    "linf_cont4to8_direct_init1",
]

conn = sqlite3.connect('/home/ashtomer/projects/ares/orchestrator/orchestrator.db')

moved, failed = [], []
for model_id in TO_REMOVE_DB:
    # Look up model_dir from DB
    row = conn.execute("SELECT model_dir FROM models_queue WHERE model_id=?", (model_id,)).fetchone()
    if row:
        src = row[0]
        import os
        dirname = os.path.basename(src)
        dst = f"{OLD_MODELS}/{dirname}"
        r = subprocess.run(["mv", src, dst], capture_output=True, text=True)
        if r.returncode == 0:
            moved.append((model_id, src))
        else:
            failed.append((model_id, r.stderr.strip()))
    else:
        failed.append((model_id, "not found in DB"))

if failed:
    print("FAILED moves:")
    for m, e in failed: print(f"  {m}: {e}")
    print("Aborting DB delete.")
else:
    placeholders = ",".join("?" * len(TO_REMOVE_DB))
    conn.execute(f"DELETE FROM models_queue WHERE model_id IN ({placeholders})", TO_REMOVE_DB)
    conn.commit()
    remaining = conn.execute("SELECT count(*) FROM models_queue").fetchone()[0]
    print(f"Moved {len(moved)} directories to old_models and removed from DB. DB now has {remaining} rows.")
    for m, s in moved:
        print(f"  moved {m}")

conn.close()
