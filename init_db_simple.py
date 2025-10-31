
import sqlite3
import os

# Create database directory
os.makedirs("database", exist_ok=True)

# Connect to database
conn = sqlite3.connect("database/asr_training.db")

# Create basic tables
conn.execute("""
CREATE TABLE IF NOT EXISTS AudioFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    duration_seconds REAL,
    sample_rate INTEGER,
    transcript TEXT,
    split_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
print(" Database initialized successfully!")
