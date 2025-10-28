#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import os

DB_PATH = "/home/ashtomer/projects/ares/robust_training/model_scheduler.db"

print("[INFO] Reading model table...\n")

if not os.path.exists(DB_PATH):
    print(f"[ERROR] Database not found: {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Fetch all model info
c.execute("""
    SELECT model_id, current_epoch, status, job_id
    FROM models
    ORDER BY model_id;
""")
rows = c.fetchall()

# Print header
print(f"{'MODEL_ID':<35} {'EPOCH':<7} {'STATUS':<10} {'JOBID':<12}")
print("-" * 70)

# Print each row formatted in columns
for model_id, epoch, status, jobid in rows:
    jobid = jobid if jobid is not None else "-"
    print(f"{model_id:<35} {epoch:<7} {status:<10} {jobid:<12}")

print("\n[INFO] Done reading from", DB_PATH)

conn.close()
