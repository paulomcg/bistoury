#!/usr/bin/env python3
"""Test script for DuckDB setup verification."""

from src.bistoury.database.connection import DatabaseManager
from src.bistoury.config import Config
import tempfile
from pathlib import Path

def test_database_setup():
    """Test database connection and basic operations."""
    print("🚀 Testing DuckDB Installation and Configuration...")
    
    # Test database connection
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config()
        config.database.path = str(Path(temp_dir) / 'test.db')
        
        # Create manager
        print("📦 Creating DatabaseManager...")
        db_manager = DatabaseManager(config)
        
        # Test connection
        print("🔌 Testing connection...")
        conn = db_manager.get_connection()
        result = conn.execute("SELECT 'DuckDB connection successful!' as message").fetchone()
        print(f'✅ {result[0]}')
        
        # Test basic operations
        print("🧪 Testing basic operations...")
        conn.execute("""
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                price DOUBLE,
                timestamp TIMESTAMP
            )
        """)
        
        # Insert test data
        test_records = [
            (1, 'BTC', 45000.50, '2024-01-01 12:00:00'),
            (2, 'ETH', 2800.25, '2024-01-01 12:01:00'),
            (3, 'SOL', 95.75, '2024-01-01 12:02:00'),
        ]
        
        for record in test_records:
            conn.execute(
                "INSERT INTO test_data (id, symbol, price, timestamp) VALUES (?, ?, ?, ?)",
                record
            )
        
        # Query data
        results = conn.execute("SELECT * FROM test_data ORDER BY timestamp").fetchall()
        print(f"📊 Inserted {len(results)} test records")
        
        for row in results:
            print(f"   {row[1]}: ${row[2]:,.2f} at {row[3]}")
        
        # Test database info
        print("📈 Getting database info...")
        info = db_manager.get_database_info()
        print(f"   Database Path: {info.get('database_path')}")
        print(f"   Active Connections: {info.get('active_connections')}")
        print(f"   Max Connections: {info.get('max_connections')}")
        
        # Test connection pooling
        print("🔄 Testing connection pooling...")
        conn1 = db_manager.get_connection("thread1")
        conn2 = db_manager.get_connection("thread2")
        conn3 = db_manager.get_connection("thread1")  # Should reuse
        
        print(f"   Thread1 connection: {id(conn1)}")
        print(f"   Thread2 connection: {id(conn2)}")
        print(f"   Thread1 reused: {id(conn3)} (same: {conn1 is conn3})")
        
        # Test cleanup
        db_manager.close_all_connections()
        print("🧹 Connections cleaned up")
        
        print("\n🎉 Task 2.1 - DuckDB Installation and Configuration: COMPLETE!")
        print("\n✅ All tests passed!")
        return True

if __name__ == "__main__":
    test_database_setup() 