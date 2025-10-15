import sqlite3

db_path = "adv_scheduler.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("UPDATE models SET status = 'waiting';")
conn.commit()
conn.close()

print("✅ All models reset to waiting.")
