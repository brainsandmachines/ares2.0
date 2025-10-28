#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import json, os, time, math
import pandas as pd

class TaskScheduler():
    def __init__(self, db_path='adv_scheduler.db', max_job_time_in_seconds=12*3600,max_epochs=250):
        self.db_path = db_path
        self.max_job_time_in_seconds = max_job_time_in_seconds
        self.max_epochs = max_epochs

        is_new = not os.path.isfile(self.db_path)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # --- Jobs table ---
        c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT UNIQUE,
            model_id TEXT,
            is_completed INTEGER,
            time_started INTEGER,
            results TEXT
        )
        """)

        # --- Models table ---
        c.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            norm TEXT,
            constraint_val INTEGER,
            adv_train INTEGER,
            init_id INTEGER,
            created_at INTEGER,
            current_epoch INTEGER DEFAULT 0,
            last_progress_ts INTEGER,
            last_epoch_duration_sec REAL,
            final_accuracy REAL,
            final_adv_accuracy REAL,
            status TEXT DEFAULT 'waiting'
        )
        """)

        # --- Epoch logs ---
        c.execute("""
        CREATE TABLE IF NOT EXISTS epoch_logs (
            model_id TEXT,
            epoch INTEGER,
            duration_sec REAL,
            ts INTEGER,
            PRIMARY KEY (model_id, epoch)
        )
        """)

        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_models_norm_cons_adv_init
        ON models(norm, constraint_val, adv_train, init_id)
        """)

        conn.commit()
        conn.close()

        if is_new:
            self.seed_models()

    # ---------------- internal helpers ----------------
    def _execute_sqlite(self, query, parameters=None, quiet_duplicate=False):
        if parameters is None:
            parameters = ()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(query, parameters)
            conn.commit()
            success = True
        except Exception as e:
            if not (quiet_duplicate and "UNIQUE constraint failed" in str(e)):
                print('sqlite error:', str(e))
            success = False
        finally:
            conn.close()
        return success

    def get_model_id(self, norm, constraint_val, adv_train, grad_norm, init_id=None):
        """
        Return the model_id string if it exists in the database.
        Otherwise, return None.
        """
        model_id = self._make_model_id(norm, constraint_val, adv_train, init_id)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(1) FROM models WHERE model_id = ?", (model_id,))
        exists = c.fetchone()[0] > 0
        conn.close()
        return model_id if exists else None
    
    def _sqlite_fetchone(self, query, parameters=None):
        if parameters is None:
            parameters = ()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(query, parameters)
            row = c.fetchone()
            fetched = row[0] if row else None
        except Exception as e:
            print('sqlite error:', str(e))
            fetched = None
        finally:
            conn.close()
        return fetched

    # ---------------- model registry ----------------
    @staticmethod
    def _make_model_id(norm, constraint_val, adv_train,grad_norm, init_id):
        parts = []
        if norm is not None: parts.append(f"norm={norm}")
        parts.append(f"c={constraint_val}")
        parts.append(f"adv={int(bool(adv_train))}")
        parts.append(f"gradnorm={int(bool(grad_norm))}")
        if init_id is not None: parts.append(f"init={init_id}")
        return "|".join(parts)

    def seed_models(self):
        now = int(time.time())
        models_to_insert = []

        # 16 adv-trained models
        for norm in ["linf", "l2"]:
            for constraint_val in [1, 2, 4, 8]:
                for init_id in [1, 2]:
                    model_id = self._make_model_id(norm, constraint_val, True, init_id)
                    models_to_insert.append((model_id, norm, constraint_val, 1, init_id, now))

        # 1 baseline
        baseline_id = self._make_model_id(None, 0, False, None)
        models_to_insert.append((baseline_id, None, 0, 0, None, now))

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany("""
            INSERT OR IGNORE INTO models
            (model_id, norm, constraint_val, adv_train, init_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, models_to_insert)
        conn.commit()
        conn.close()
        print("[INFO] Seeded 17 models into the database.")

    def list_models_df(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT model_id, norm, constraint_val, adv_train, grad_norm, init_id,
               current_epoch, status,
               last_progress_ts, last_time_selected
            FROM models
            ORDER BY adv_train DESC, norm IS NULL, norm, constraint_val, init_id
        """, conn)
        conn.close()
        
        # Convert timestamps (seconds since epoch) into readable datetimes
        for col in ["last_progress_ts", "last_time_selected"]:
            df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")

        # Optional: show timestamps in local time (instead of UTC)
        df["last_progress_ts"] = df["last_progress_ts"].dt.tz_localize("UTC").dt.tz_convert("Asia/Jerusalem")
        df["last_time_selected"] = df["last_time_selected"].dt.tz_localize("UTC").dt.tz_convert("Asia/Jerusalem")
        return df

    def update_status(self, model_id, status):
        assert status in ['waiting', 'training', 'finished']
        if status == 'training':
            return self._execute_sqlite(
                "UPDATE models SET status = ?, last_progress_ts = COALESCE(last_progress_ts, ?) WHERE model_id = ?",
                (status, int(time.time()), model_id)
            )
        return self._execute_sqlite("UPDATE models SET status = ? WHERE model_id = ?", (status, model_id))

    def log_epoch(self, model_id, epoch, duration_sec):
        now = int(time.time())
        return self._execute_sqlite("""
            INSERT OR REPLACE INTO epoch_logs (model_id, epoch, duration_sec, ts)
            VALUES (?, ?, ?, ?)
        """, (model_id, int(epoch), float(duration_sec), now))

    def update_progress_epoch_end(self, model_id):
        now = int(time.time())
        max_epoch = self.max_epochs

        # Increment epoch count and update timestamp
        self._execute_sqlite("""
            UPDATE models
            SET current_epoch = current_epoch + 1,
                last_progress_ts = ?
            WHERE model_id = ?
        """, (now, model_id))

        # Check if training finished
        current_epoch = self._sqlite_fetchone(
            "SELECT current_epoch FROM models WHERE model_id = ?", (model_id,)
        )
        if current_epoch is not None and current_epoch >= max_epoch:
            self._execute_sqlite(
                "UPDATE models SET status = 'finished' WHERE model_id = ?", (model_id,)
            )
        else:
            self._execute_sqlite(
                "UPDATE models SET status = 'training' WHERE model_id = ?", (model_id,)
            )

    
    def requeue_stale_trainings(self, threshold_hours=10):
        """
        Requeue models stuck in 'training' for longer than threshold_hours
        (and still under max_epoch). Sets them back to 'waiting'.
        """
        now = int(time.time())
        max_epoch = self.max_epochs
        cutoff = now - int(threshold_hours * 3600)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            UPDATE models
            SET status = 'waiting'
            WHERE status = 'training'
            AND current_epoch < ?
            AND (last_progress_ts IS NULL OR last_progress_ts < ?)
        """, (max_epoch, cutoff))
        affected = c.rowcount
        conn.commit()
        conn.close()

        if affected > 0:
            print(f"[INFO] Requeued {affected} stale training jobs (> {threshold_hours}h inactive).")
        return affected

    def claim_next_waiting_model(self, cooldown_minutes=2, retries=5):
        """
        Claim the next available waiting model for training.
        Retries a few times if the database is temporarily locked.
        No WAL, no timeout, no manual BEGIN statements.
        """
        cooldown_secs = int(cooldown_minutes * 60)
        slurm_array_jobid = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_jobid       = os.environ.get("SLURM_JOB_ID")  # fallback for non-array jobs
        slurm_array_task  = os.environ.get("SLURM_ARRAY_TASK_ID")
        job_identifier = None
        base_id = slurm_array_jobid or slurm_jobid
        if base_id:
            job_identifier = f"{base_id}_{slurm_array_task}" if slurm_array_task else base_id

        for attempt in range(retries):
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                
                # Ensure JOBID column exists (auto add if missing)
                c.execute("PRAGMA table_info(models)")
                cols = [r[1] for r in c.fetchall()]
                if "JOBID" not in cols:
                    c.execute("ALTER TABLE models ADD COLUMN JOBID TEXT")
                    conn.commit()


                # Find the next waiting model
                c.execute("""
                    SELECT model_id, norm, constraint_val, adv_train, init_id, current_epoch
                    FROM models
                    WHERE status = 'waiting'
                    AND (
                        last_time_selected IS NULL
                        OR (strftime('%s','now') - last_time_selected) > ?
                    )
                    ORDER BY current_epoch DESC
                    LIMIT 1
                """, (cooldown_secs,))
                row = c.fetchone()

                if not row:
                    conn.close()
                    return None  # no waiting models available

                # Extract info
                model_id, norm, constraint_val, adv_train, init_id, epoch = row
                now = int(time.time())
                
                if epoch >= self.max_epochs:
                    self.update_status(model_id, 'finished')
                    continue  # try again

                # Mark it as "training" and record the Slurm JOBID
                c.execute("""
                    UPDATE models
                    SET status = 'training',
                        last_time_selected = ?,
                        JOBID = ?
                    WHERE model_id = ?
                """, (now, job_identifier, model_id))
                conn.commit()
                conn.close()

                # Return info as dict
                return {
                    "model_id": model_id,
                    "norm": norm,
                    "constraint_val": constraint_val,
                    "adv_train": adv_train,
                    "init_id": init_id,
                    "epochs": self.max_epochs-epoch,
                    "JOBID": job_identifier,
                }

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    print(f"[WARN] Database is locked. Retry {attempt+1}/{retries} in 2s...")
                    time.sleep(2)
                    continue  # try again
                else:
                    print(f"[ERROR] SQLite operational error: {e}")
                    break

            finally:
                try:
                    conn.close()
                except:
                    pass

        print(f"[ERROR] Failed to claim model after {retries} retries.")
        return None
