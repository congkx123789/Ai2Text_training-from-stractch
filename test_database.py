import sqlite3
import os

# Check if database exists
db_path = "database/asr_training.db"
if os.path.exists(db_path):
    print(f"✅ Database found: {db_path}")
    
    # Connect and query
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\n📊 Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
        
        # Count rows in each table
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"    Rows: {count}")
    
    conn.close()
    print("\n✅ Database test completed successfully!")
else:
    print(f"❌ Database not found: {db_path}")


