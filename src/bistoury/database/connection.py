"""DuckDB connection management and pooling."""

import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager, contextmanager
import duckdb
import os

from ..config import Config
from ..logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages DuckDB connections with pooling and configuration."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.db_path = Path(self.config.database.path)
        self.max_connections = self.config.database.max_connections
        self.connection_timeout = self.config.database.connection_timeout
        
        # Connection pool
        self._connections: Dict[str, duckdb.DuckDBPyConnection] = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Only log if not in live mode (check environment flag)
        if not os.getenv('BISTOURY_LIVE_MODE'):
            logger.info(f"Database manager initialized: {self.db_path}")
    
    def get_connection(self, thread_name: Optional[str] = None) -> duckdb.DuckDBPyConnection:
        """Get a thread-local DuckDB connection.
        
        Args:
            thread_name: Optional thread identifier for connection pooling
            
        Returns:
            DuckDB connection instance
        """
        if thread_name is None:
            thread_name = threading.current_thread().name
        
        with self._lock:
            if thread_name not in self._connections:
                logger.debug(f"Creating new database connection for thread: {thread_name}")
                conn = duckdb.connect(str(self.db_path))
                
                # Configure connection
                self._configure_connection(conn)
                
                self._connections[thread_name] = conn
                
        return self._connections[thread_name]
    
    def _configure_connection(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Configure a DuckDB connection with optimal settings.
        
        Args:
            conn: DuckDB connection to configure
        """
        # Enable parallel processing (use available CPU cores)
        cpu_count = os.cpu_count() or 4
        conn.execute(f"SET threads TO {cpu_count}")
        
        # Memory settings for large datasets
        conn.execute("SET memory_limit = '2GB'")
        conn.execute("SET max_memory = '3GB'")
        
        # Optimize for analytics workload
        conn.execute("SET enable_progress_bar = false")
        conn.execute("SET enable_object_cache = true")
        
        # Enable advanced compression
        conn.execute("SET preserve_insertion_order = false")
        
        # Optimize for time-series data
        conn.execute("SET enable_external_access = false")  # Security
        
        logger.debug("Database connection configured")
    
    @contextmanager
    def get_cursor(self, thread_name: Optional[str] = None):
        """Get a cursor for database operations.
        
        Args:
            thread_name: Optional thread identifier
            
        Yields:
            DuckDB cursor
        """
        conn = self.get_connection(thread_name)
        try:
            yield conn.cursor()
        except Exception as e:
            logger.error(f"Database cursor error: {e}")
            raise
    
    def execute(self, query: str, params: Optional[tuple] = None, 
               thread_name: Optional[str] = None) -> Any:
        """Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            thread_name: Optional thread identifier
            
        Returns:
            Query results
        """
        conn = self.get_connection(thread_name)
        try:
            if params:
                return conn.execute(query, params).fetchall()
            else:
                return conn.execute(query).fetchall()
        except Exception as e:
            logger.error(f"Database query error: {query[:100]}... - {e}")
            raise
    
    def execute_many(self, query: str, params_list: list, 
                    thread_name: Optional[str] = None) -> None:
        """Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            thread_name: Optional thread identifier
        """
        conn = self.get_connection(thread_name)
        try:
            conn.executemany(query, params_list)
        except Exception as e:
            logger.error(f"Database batch query error: {query[:100]}... - {e}")
            raise
    
    def close_connection(self, thread_name: Optional[str] = None) -> None:
        """Close a specific connection.
        
        Args:
            thread_name: Thread identifier for connection to close
        """
        if thread_name is None:
            thread_name = threading.current_thread().name
            
        with self._lock:
            if thread_name in self._connections:
                try:
                    self._connections[thread_name].close()
                    del self._connections[thread_name]
                    logger.debug(f"Closed database connection for thread: {thread_name}")
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
    
    def close_all_connections(self) -> None:
        """Close all database connections."""
        with self._lock:
            for thread_name, conn in self._connections.items():
                try:
                    conn.close()
                    logger.debug(f"Closed database connection for thread: {thread_name}")
                except Exception as e:
                    logger.error(f"Error closing connection for {thread_name}: {e}")
            
            self._connections.clear()
            logger.info("All database connections closed")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics.
        
        Returns:
            Dictionary with database info
        """
        conn = self.get_connection()
        
        try:
            # Database file size (if it exists)
            db_size = "Unknown"
            if self.db_path.exists():
                size_bytes = self.db_path.stat().st_size
                if size_bytes < 1024:
                    db_size = f"{size_bytes} bytes"
                elif size_bytes < 1024 * 1024:
                    db_size = f"{size_bytes / 1024:.1f} KB"
                elif size_bytes < 1024 * 1024 * 1024:
                    db_size = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    db_size = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
            
            # Table count
            table_count = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchone()[0]
            
            # Active connections
            active_connections = len(self._connections)
            
            return {
                "database_path": str(self.db_path),
                "database_size": db_size,
                "table_count": table_count,
                "active_connections": active_connections,
                "max_connections": self.max_connections
            }
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {
                "database_path": str(self.db_path),
                "error": str(e)
            }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(config: Optional[Config] = None) -> DatabaseManager:
    """Initialize the global database manager.
    
    Args:
        config: Configuration object
        
    Returns:
        Database manager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(config)
        # Only log if not in live mode (check environment flag)
        if not os.getenv('BISTOURY_LIVE_MODE'):
            logger.info("Database manager initialized")
    
    return _db_manager


def get_connection(thread_name: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Get a database connection from the global manager.
    
    Args:
        thread_name: Optional thread identifier
        
    Returns:
        DuckDB connection
        
    Raises:
        RuntimeError: If database manager is not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database() first.")
    
    return _db_manager.get_connection(thread_name)


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance.
    
    Returns:
        Database manager instance
        
    Raises:
        RuntimeError: If database manager is not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database() first.")
    
    return _db_manager


def shutdown_database() -> None:
    """Shutdown the database manager and close all connections."""
    global _db_manager
    
    if _db_manager is not None:
        _db_manager.close_all_connections()
        _db_manager = None
        logger.info("Database manager shutdown complete") 