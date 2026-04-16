import logging
import sqlite3
import os
from karierai.database import init_sqlite, get_connection

logging.basicConfig(level=logging.INFO)

def test_init():
    print("Testing init_sqlite...")
    try:
        init_sqlite()
        print("Success: init_sqlite completed.")
    except Exception as e:
        print(f"Error in init_sqlite: {e}")

def test_readonly():
    print("\nTesting read-only connection...")
    try:
        with get_connection(read_only=True) as conn:
            row = conn.execute("SELECT 1").fetchone()
            print(f"Success: Read-only query returned {row[0]}")
    except Exception as e:
        print(f"Error in read-only connection: {e}")

if __name__ == "__main__":
    # Ensure we are in the right directory to import karierai
    import sys
    sys.path.append('src')
    test_init()
    test_readonly()
