import os
import sqlite3
import torch

from robust_training.model_scheduler import Model_scheduler

# === CONFIG ===
DB_PATH = "/home/ashtomer/projects/ares/robust_training/model_scheduler.db"
RESULTS_ROOT = "/home/ashtomer/projects/ares/robust_training/results/convnext_small"
CHECKPOINT_NAME = "last.pth.tar"

# === Initialize scheduler ===
sch = Model_scheduler(DB_PATH)

# === Load DB into dictionary ===
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
rows = c.execute("SELECT model_id, current_epoch FROM models").fetchall()
db_epochs = {model_id: current_epoch for model_id, current_epoch in rows}
conn.close()

# === Scan all model result folders ===
for model_dir in sorted(os.listdir(RESULTS_ROOT)):
    ckpt_path = os.path.join(RESULTS_ROOT, model_dir, CHECKPOINT_NAME)
    if not os.path.isfile(ckpt_path):
        print(f"[MISSING] {model_dir} → no {CHECKPOINT_NAME}")
        continue

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # timm CheckpointSaver stores the epoch index (0-based)
        ckpt_epoch = ckpt.get("epoch", ckpt.get("start_epoch", None))
        if ckpt_epoch is not None:
            ckpt_epoch += 1  # Convert to 1-based epoch
    except Exception as e:
        print(f"[ERROR] {model_dir}: cannot read checkpoint ({e})")
        continue

    # match model_id format to DB key
    # convnext_small_eps-1_l2_seed-1  ->  norm=l2|c=1|adv=1|gradnorm=0|init=1
    parts = model_dir.split("_")
    # Example: ['convnext', 'small', 'eps-1', 'l2', 'seed-1']
    eps = parts[2].split("-")[1]
    norm = parts[3] if parts[3] != "None" else None
    seed = parts[4].split("-")[1] if "seed" in parts[4] else None
    adv = 0 if eps == "0" else 1
    if adv == 0 and eps == "0":  # baseline model (no init)
        model_id = f"c=0|adv=0|gradnorm=0"
    else:
        model_id = f"norm={norm}|c={eps}|adv={adv}|gradnorm=0|init={seed}"

    db_epoch = db_epochs.get(model_id, None)

    if db_epoch is None:
        print(f"[WARN] {model_dir} → model_id not found in DB ({model_id})")
        continue

    if ckpt_epoch == db_epoch:
        print(f"[OK]   {model_dir}: epoch {ckpt_epoch} ✅ matches DB")
    else:
        print(f"[UPDATE] {model_dir}: checkpoint={ckpt_epoch}, DB={db_epoch} → updating DB")
        sch.update_epochs({model_id: ckpt_epoch})
