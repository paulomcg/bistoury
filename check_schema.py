#!/usr/bin/env python3
"""
Check the actual database schema to understand the interval field issue.
"""

from src.bistoury.database import DatabaseManager
from src.bistoury.config import Config

config = Config.load_from_env()
db = DatabaseManager(config)

print("🔍 CHECKING DATABASE SCHEMA")
print("=" * 50)

# Check table existence and schema
tables = ['candles_1m', 'trades', 'orderbook_snapshots']

for table in tables:
    print(f"\n📋 Table: {table}")
    try:
        # Get table info from DuckDB
        result = db.execute(f"DESCRIBE {table}")
        if result:
            print("   Columns:")
            for row in result:
                column_name = row[0]
                column_type = row[1]
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                print(f"     - {column_name}: {column_type} ({nullable})")
        else:
            print("   No schema info available")
    except Exception as e:
        print(f"   Error: {e}")

print(f"\n🗂️ Checking if interval field exists in candles_1m:")
try:
    # Try to query the interval field specifically
    result = db.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'candles_1m' AND column_name = 'interval'")
    if result:
        print("   ✅ interval field EXISTS in candles_1m table")
    else:
        print("   ❌ interval field DOES NOT EXIST in candles_1m table")
except Exception as e:
    print(f"   Error checking interval field: {e}")

print(f"\n📊 Sample data from candles_1m (if any):")
try:
    result = db.execute("SELECT * FROM candles_1m LIMIT 3")
    if result:
        print(f"   Found {len(result)} sample records")
        for i, row in enumerate(result):
            print(f"     Row {i+1}: {row}")
    else:
        print("   No data found")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n🔧 Full CREATE TABLE statement for candles_1m:")
try:
    # Get the full table creation statement
    result = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='candles_1m'")
    if result:
        print(f"   {result[0][0]}")
    else:
        # Try DuckDB way
        result = db.execute("SHOW CREATE TABLE candles_1m")
        if result:
            print(f"   {result[0][1]}")
        else:
            print("   Could not retrieve CREATE TABLE statement")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50) 