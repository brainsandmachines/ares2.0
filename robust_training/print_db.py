#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import os

DB_PATH = "/home/ashtomer/projects/ares/robust_training/adv_scheduler.db"

print("[INFO] Updated epoch values:")

if not os.path.exists(DB_PATH):
    print(f"[ERROR] Database not found: {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Fetch all model info
c.execute("SELECT model_id, current_epoch, status FROM models ORDER BY model_id;")
rows = c.fetchall()

for row in rows:
    print(row)

conn.close()
print(f"[INFO] Changes saved to {DB_PATH}")
