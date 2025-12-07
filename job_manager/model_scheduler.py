import sqlite3
import os, time
import pandas as pd

class Model_scheduler():
    def __init__(self, db_path='model_scheduler.db', num_epochs=250):
        print("[DEBUG] Initializing Model_scheduler...")
        self.db_path = db_path
        self.num_epochs = num_epochs

        is_new = not os.path.isfile(self.db_path)
        print(f"[DEBUG] Database path: {self.db_path}, is_new: {is_new}")

        conn = sqlite3.connect(self.db_path, timeout=5.0)
        c = conn.cursor()
        print("[DEBUG] Connected to SQLite database.")

        # --- Models table ---
        c.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            norm TEXT,
            constraint_val REAL,
            adv_train INTEGER,
            grad_norm INTEGER DEFAULT 0,
            init_id INTEGER,
            current_epoch INTEGER DEFAULT 0,
            last_progress_ts INTEGER,
            last_time_selected INTEGER,
            status TEXT DEFAULT 'waiting',
            job_id TEXT
        )
        """)
        print("[DEBUG] Models table created or already exists.")

        conn.commit()
        conn.close()
        print("[DEBUG] Database connection closed.")

        if is_new:
            print("[DEBUG] Seeding models...")
            self.seed_models()
            print("[DEBUG] Models seeded.")
            
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
        Otherwise, create a new entry and return the model_id.
        """
        model_id = self._make_model_id(norm, constraint_val, adv_train, init_id=init_id, grad_norm=grad_norm)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Check if model already exists
        c.execute("SELECT COUNT(1) FROM models WHERE model_id = ?", (model_id,))
        exists = c.fetchone()[0] > 0

        # If not found, insert a new entry
        if not exists:
            try:
                c.execute(
                    """
                    INSERT INTO models (model_id, norm, constraint_val, adv_train, grad_norm, init_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (model_id, norm, constraint_val, adv_train, grad_norm, init_id),
                )
                conn.commit()
                print(f"Created new model entry: {model_id}")
            except Exception as e:
                print("Error creating model entry:", e)
                conn.rollback()

        conn.close()
        return model_id
    
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
    def _make_model_id(norm, constraint_val, adv_train, init_id=None, grad_norm=0):
        """Build a stable model id string.

        Signature: (norm, constraint_val, adv_train, init_id=None, grad_norm=0)
        Historically some callers passed init_id as the 4th arg; make that the
        canonical position and keep grad_norm as an optional trailing param.
        """
        parts = []
        if norm is not None:
            parts.append(f"norm={norm}")
        parts.append(f"c={constraint_val}")
        parts.append(f"adv={int(bool(adv_train))}")
        parts.append(f"gradnorm={int(bool(grad_norm))}")
        if init_id is not None:
            parts.append(f"init={init_id}")
        return "|".join(parts)
    
    def seed_models(self):
        models_to_insert = []

        # 16 adv-trained models
        for norm in ["linf", "l2"]:
            for constraint_val in [1, 2, 4, 8]:
                for init_id in [1, 2]:
                    model_id = self._make_model_id(norm, constraint_val, True, init_id=init_id, grad_norm=0)
                    models_to_insert.append((model_id, norm, constraint_val, 1, 0, init_id))

        # 1 baseline
        baseline_id = self._make_model_id(None, 0, False, init_id=None, grad_norm=0)
        models_to_insert.append((baseline_id, None, 0, 0, 0, None))

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany("""
            INSERT OR IGNORE INTO models
            (model_id, norm, constraint_val, adv_train, grad_norm, init_id)
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

    def update_progress_epoch_end(self, model_id, epoch):
        now = int(time.time())
        num_epochs = self.num_epochs

        # Increment epoch count and update timestamp
        self._execute_sqlite("""
            UPDATE models
            SET current_epoch = ?,
                last_progress_ts = ?
            WHERE model_id = ?
        """, (epoch,now, model_id))

         # Update status based on epoch
        if epoch >= num_epochs:
            self._execute_sqlite(
            "UPDATE models SET status = 'finished' WHERE model_id = ?",
            (model_id,)
        )
        else:
            self._execute_sqlite(
            "UPDATE models SET status = 'training' WHERE model_id = ?",
            (model_id,)
        )

    
    def requeue_stale_trainings(self, threshold_hours=10):
        """
        Requeue models stuck in 'training' for longer than threshold_hours
        (and still under max_epoch). Sets them back to 'waiting'.
        """
        now = int(time.time())
        num_epochs = self.num_epochs
        cutoff = now - int(threshold_hours * 3600)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            UPDATE models
            SET status = 'waiting'
            WHERE status = 'training'
            AND current_epoch < ?
            AND (last_progress_ts IS NULL OR last_progress_ts < ?)
        """, (num_epochs, cutoff))
        affected = c.rowcount
        conn.commit()
        conn.close()

        if affected > 0:
            print(f"[INFO] Requeued {affected} stale training jobs (> {threshold_hours}h inactive).")
        return affected

    def claim_next_waiting_model(self, cooldown_minutes=2):
        """
        Atomically claim the next available waiting model for training.
        Prevents duplicate claims under concurrent access.
        """
        cooldown_secs = int(cooldown_minutes * 60)
        job_identifier = None

        slurm_array_jobid = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_jobid = os.environ.get("SLURM_JOB_ID")
        slurm_array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        base_id = slurm_array_jobid or slurm_jobid
        if base_id:
            job_identifier = f"{base_id}_{slurm_array_task}" if slurm_array_task else base_id

        now = int(time.time())

        for _ in range(5):  # retry up to 5 times if database is locked
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
                c = conn.cursor()

                # Start an IMMEDIATE transaction (prevents concurrent writes)
                c.execute("BEGIN IMMEDIATE")

                # Select one waiting model that’s eligible for claiming
                c.execute("""
                    SELECT model_id, norm, constraint_val, adv_train, grad_norm, init_id, current_epoch
                    FROM models
                    WHERE status = 'waiting'
                    AND (last_time_selected IS NULL OR (strftime('%s','now') - last_time_selected) > ?)
                    ORDER BY grad_norm DESC, current_epoch DESC
                    LIMIT 1
                """, (cooldown_secs,))
                row = c.fetchone()

                if not row:
                    conn.rollback()
                    conn.close()
                    return None

                model_id, norm, constraint_val, adv_train, grad_norm, init_id, epoch = row

                if epoch >= self.num_epochs:
                    c.execute("UPDATE models SET status='finished' WHERE model_id=?", (model_id,))
                    conn.commit()
                    conn.close()
                    return None

                # Atomically update status → 'training'
                c.execute("""
                    UPDATE models
                    SET status='training',
                        last_time_selected=?,
                        job_id=?
                    WHERE model_id=?
                    AND status='waiting'
                """, (now, job_identifier, model_id))

                if c.rowcount == 0:
                    # Someone else already claimed it; retry
                    conn.rollback()
                    conn.close()
                    time.sleep(0.1)
                    continue

                conn.commit()
                conn.close()

                print(f"[DEBUG] Claimed model: {model_id}, norm: {norm}, constraint_val: {constraint_val}, adv_train: {adv_train}, grad_norm: {grad_norm}, init_id: {init_id}, epochs_left: {self.num_epochs - epoch}")
                return {
                    "model_id": model_id,
                    "norm": norm,
                    "constraint_val": constraint_val,
                    "adv_train": adv_train,
                    "init_id": init_id,
                    "epochs": self.num_epochs,
                    "epochs_left": self.num_epochs - epoch,
                    "JOBID": job_identifier,
                    "grad_norm": grad_norm
                }

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    time.sleep(0.1)  # brief pause before retry
                    continue
                else:
                    print(f"[ERROR] SQLite error: {e}")
                    break
            finally:
                try:
                    conn.close()
                except:
                    pass

        print("[WARN] Failed to claim model after multiple retries.")
        return None
    
    def update_epochs(self, epoch_updates):
        """
        Update the current_epoch for specific model_ids.

        Args:
            epoch_updates (dict): A dictionary where keys are model_ids and values are the new epoch numbers.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for model_id, epoch in epoch_updates.items():
            c.execute(
                """
                UPDATE models
                SET current_epoch = ?
                WHERE model_id = ?
                """,
                (epoch, model_id),
            )

        conn.commit()
        conn.close()
        print("[INFO] Updated epochs for models.")
