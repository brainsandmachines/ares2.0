import sys
import os

# Add the parent directory of 'ares' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_manager.model_scheduler import Model_scheduler

# Path to the database file
db_path = '/home/ashtomer/projects/ares/job_manager/model_scheduler.db'

# Delete the existing database file if it exists
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Deleted existing database: {db_path}")

# Instantiate the Model_scheduler class to recreate the database
print("Initializing Model_scheduler...")
scheduler = Model_scheduler(db_path=db_path)
print("Model_scheduler initialized.")

# Seed the database with predefined models
print("Seeding models into the database...")
scheduler.seed_models()
print("Database initialized and seeded successfully.")