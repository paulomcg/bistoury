"""Tests for database connection and management."""

import tempfile
import threading
import pytest
from pathlib import Path

from bistoury.database.connection import DatabaseManager, initialize_database, get_connection
from bistoury.config import Config, DatabaseConfig


def test_database_manager_initialization():
    """Test database manager initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        assert db_manager.db_path == db_path
        assert db_manager.max_connections == config.database.max_connections
        assert db_path.parent.exists()


def test_database_connection():
    """Test getting a database connection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        conn = db_manager.get_connection()
        
        assert conn is not None
        
        # Test basic query
        result = conn.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1


def test_connection_pooling():
    """Test connection pooling with multiple threads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        # Get connections from different threads
        conn1 = db_manager.get_connection("thread1")
        conn2 = db_manager.get_connection("thread2")
        conn3 = db_manager.get_connection("thread1")  # Should reuse thread1's connection
        
        assert conn1 is not conn2  # Different threads should get different connections
        assert conn1 is conn3     # Same thread should reuse connection


def test_database_operations():
    """Test basic database operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        # Create test table
        db_manager.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value DOUBLE
            )
        """)
        
        # Insert data
        db_manager.execute(
            "INSERT INTO test_table (id, name, value) VALUES (?, ?, ?)",
            (1, "test", 123.45)
        )
        
        # Query data
        result = db_manager.execute("SELECT * FROM test_table")
        assert len(result) == 1
        assert result[0][1] == "test"
        assert result[0][2] == 123.45


def test_batch_operations():
    """Test batch database operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        # Create test table
        db_manager.execute("""
            CREATE TABLE batch_test (
                id INTEGER PRIMARY KEY,
                value INTEGER
            )
        """)
        
        # Insert batch data
        data = [(i, i) for i in range(100)]  # (id, value) pairs
        db_manager.execute_many("INSERT INTO batch_test (id, value) VALUES (?, ?)", data)
        
        # Verify data
        result = db_manager.execute("SELECT COUNT(*) FROM batch_test")
        assert result[0][0] == 100


def test_global_database_functions():
    """Test global database management functions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        # Initialize global database
        db_manager = initialize_database(config)
        assert db_manager is not None
        
        # Get connection through global function
        conn = get_connection()
        assert conn is not None
        
        # Test basic operation
        result = conn.execute("SELECT 'global test' as msg").fetchone()
        assert result[0] == "global test"


def test_database_info():
    """Test database information retrieval."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        # Create a test table
        db_manager.execute("""
            CREATE TABLE info_test (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        
        # Insert a test record
        db_manager.execute("INSERT INTO info_test (id, data) VALUES (?, ?)", (1, "test"))
        
        info = db_manager.get_database_info()
        
        assert "database_path" in info
        assert "table_count" in info
        assert "active_connections" in info
        assert info["table_count"] >= 1  # At least our test table


def test_connection_cleanup():
    """Test connection cleanup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        # Get connections
        conn1 = db_manager.get_connection("test_thread")
        assert "test_thread" in db_manager._connections
        
        # Close specific connection
        db_manager.close_connection("test_thread")
        assert "test_thread" not in db_manager._connections
        
        # Get new connections
        conn2 = db_manager.get_connection("thread1")
        conn3 = db_manager.get_connection("thread2")
        assert len(db_manager._connections) == 2
        
        # Close all connections
        db_manager.close_all_connections()
        assert len(db_manager._connections) == 0


def test_cursor_context_manager():
    """Test cursor context manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("CREATE TABLE cursor_test (id INTEGER, value TEXT)")
            cursor.execute("INSERT INTO cursor_test VALUES (1, 'test')")
        
        # Verify data persisted
        result = db_manager.execute("SELECT * FROM cursor_test")
        assert len(result) == 1
        assert result[0] == (1, 'test')


@pytest.mark.parametrize("thread_count", [1, 3, 5])
def test_concurrent_access(thread_count):
    """Test concurrent database access."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        config = Config()
        config.database.path = str(db_path)
        
        db_manager = DatabaseManager(config)
        
        # Create test table
        db_manager.execute("""
            CREATE TABLE concurrent_test (
                id INTEGER PRIMARY KEY,
                thread_id TEXT,
                value INTEGER
            )
        """)
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                # Each thread inserts data
                for i in range(10):
                    # Use unique IDs: thread_id * 100 + i to avoid conflicts
                    unique_id = thread_id * 100 + i
                    db_manager.execute(
                        "INSERT INTO concurrent_test VALUES (?, ?, ?)",
                        (unique_id, f"thread_{thread_id}", i)
                    )
                results.append(f"thread_{thread_id}")
            except Exception as e:
                errors.append(str(e))
        
        # Start threads
        threads = []
        for i in range(thread_count):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == thread_count
        
        # Check data in database
        total_rows = db_manager.execute("SELECT COUNT(*) FROM concurrent_test")[0][0]
        assert total_rows == thread_count * 10 