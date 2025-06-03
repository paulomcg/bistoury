"""
Database switcher and compatibility layer for multiple database files.
Allows seamless switching between production and test databases.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import json
import duckdb

from ..config import Config, DatabaseConfig
from .connection import DatabaseManager
from .schema import MarketDataSchema, DataInsertion, DataQuery
from ..logger import get_logger

logger = get_logger(__name__)


class DatabaseSwitcher:
    """Manages switching between different database files and schemas."""
    
    def __init__(self):
        self.available_databases = {
            'production': 'data/bistoury.db',
            'test': 'data/test.duckdb',
            'memory': ':memory:'
        }
        self.current_db = None
        self.current_manager = None
        
    def list_available_databases(self) -> Dict[str, Dict[str, Any]]:
        """List all available databases with their info."""
        db_info = {}
        
        for name, path in self.available_databases.items():
            if path == ':memory:':
                db_info[name] = {
                    'path': path,
                    'exists': True,
                    'size': 'N/A',
                    'type': 'memory'
                }
                continue
                
            path_obj = Path(path)
            if path_obj.exists():
                size_bytes = path_obj.stat().st_size
                size_str = self._format_size(size_bytes)
                
                # Try to get table info
                try:
                    conn = duckdb.connect(str(path_obj))
                    tables = conn.execute("""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_schema = 'main'
                    """).fetchone()[0]
                    conn.close()
                    
                    # Detect schema type
                    schema_type = self._detect_schema_type(path_obj)
                    
                    db_info[name] = {
                        'path': str(path_obj),
                        'exists': True,
                        'size': size_str,
                        'tables': tables,
                        'schema_type': schema_type,
                        'type': 'file'
                    }
                except Exception as e:
                    db_info[name] = {
                        'path': str(path_obj),
                        'exists': True,
                        'size': size_str,
                        'error': str(e),
                        'type': 'file'
                    }
            else:
                db_info[name] = {
                    'path': str(path_obj),
                    'exists': False,
                    'size': '0',
                    'type': 'file'
                }
                
        return db_info
        
    def switch_to_database(self, database_name: str) -> DatabaseManager:
        """Switch to a specific database."""
        if database_name not in self.available_databases:
            raise ValueError(f"Unknown database: {database_name}. Available: {list(self.available_databases.keys())}")
            
        # Close current connection
        if self.current_manager:
            self.current_manager.close_all_connections()
            
        # Create config for the target database
        config = Config()
        config.database = DatabaseConfig(path=self.available_databases[database_name])
        
        # Create new manager
        self.current_manager = DatabaseManager(config)
        self.current_db = database_name
        
        logger.info(f"Switched to database: {database_name} ({self.available_databases[database_name]})")
        return self.current_manager
        
    def get_current_database(self) -> Optional[str]:
        """Get the name of the currently active database."""
        return self.current_db
        
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
            
    def _detect_schema_type(self, db_path: Path) -> str:
        """Detect the schema type of a database file."""
        try:
            conn = duckdb.connect(str(db_path))
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchall()
            table_names = {row[0] for row in tables}
            conn.close()
            
            # Check for production schema
            production_tables = {
                'symbols', 'trades', 'orderbook_snapshots', 'funding_rates',
                'candles_1m', 'candles_5m', 'candles_15m', 
                'candles_1h', 'candles_4h', 'candles_1d'
            }
            
            # Check for test schema (from exploration)
            test_tables = {
                'all_mids', 'candles', 'order_books', 'raw_messages', 'trades'
            }
            
            if production_tables.issubset(table_names):
                return 'production'
            elif test_tables.issubset(table_names):
                return 'test_legacy'
            else:
                return 'unknown'
                
        except Exception:
            return 'error'


class TestDatabaseCompatibilityLayer:
    """Provides a compatibility layer for the test database schema."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def get_symbols_compatible(self) -> List[Dict[str, Any]]:
        """Get symbols from test database in production format."""
        try:
            # Extract unique symbols from trades table
            result = self.db_manager.execute("""
                SELECT DISTINCT 
                    coin as symbol,
                    coin as name,
                    0 as sz_decimals,
                    50.0 as max_leverage,
                    0 as margin_table_id,
                    false as is_delisted
                FROM trades 
                WHERE coin IS NOT NULL
                ORDER BY coin
            """)
            
            symbols = []
            for i, row in enumerate(result, 1):
                symbols.append({
                    'id': i,
                    'symbol': row[0],
                    'name': row[1],
                    'sz_decimals': row[2],
                    'max_leverage': row[3],
                    'margin_table_id': row[4],
                    'is_delisted': row[5],
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc)
                })
                
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting compatible symbols: {e}")
            return []
            
    def get_trades_compatible(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trades from test database in production format."""
        try:
            result = self.db_manager.execute("""
                SELECT 
                    id,
                    symbol,
                    timestamp,
                    px as price,
                    sz as size,
                    side,
                    tid as trade_id,
                    coin
                FROM trades 
                WHERE coin = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))
            
            trades = []
            for row in result:
                trades.append({
                    'id': row[0],
                    'symbol': row[1],
                    'timestamp': row[2],
                    'price': row[3],
                    'size': row[4],
                    'side': row[5],
                    'trade_id': row[6],
                    'hash': None,
                    'user1': None,
                    'user2': None,
                    'created_at': row[2]  # Use trade timestamp
                })
                
            return trades
            
        except Exception as e:
            logger.error(f"Error getting compatible trades: {e}")
            return []
            
    def get_orderbook_compatible(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest orderbook from test database in production format."""
        try:
            result = self.db_manager.execute("""
                SELECT 
                    timestamp,
                    levels,
                    time
                FROM order_books 
                WHERE coin = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            
            if not result:
                return None
                
            row = result[0]
            levels = json.loads(row[1]) if row[1] else [[], []]
            
            return {
                'coin': symbol,
                'timestamp': row[0],
                'time': row[2],
                'levels': levels,
                'created_at': row[0]
            }
            
        except Exception as e:
            logger.error(f"Error getting compatible orderbook: {e}")
            return None
            
    def get_raw_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the raw data in test database."""
        try:
            stats = {}
            
            # Get date range
            date_range = self.db_manager.execute("""
                SELECT 
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date
                FROM raw_messages
            """)
            
            if date_range:
                stats['date_range'] = {
                    'start': date_range[0][0],
                    'end': date_range[0][1]
                }
            
            # Get channel breakdown
            channels = self.db_manager.execute("""
                SELECT 
                    channel,
                    COUNT(*) as message_count
                FROM raw_messages
                GROUP BY channel
                ORDER BY message_count DESC
            """)
            
            stats['channels'] = dict(channels)
            
            # Get symbol breakdown
            symbols = self.db_manager.execute("""
                SELECT 
                    coin,
                    COUNT(*) as trade_count
                FROM trades
                WHERE coin IS NOT NULL
                GROUP BY coin
                ORDER BY trade_count DESC
            """)
            
            stats['symbols'] = dict(symbols)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting raw data stats: {e}")
            return {}


class CompatibleDataQuery:
    """Data query class that works with both production and test schemas."""
    
    def __init__(self, db_manager: DatabaseManager, schema_type: str = 'production'):
        self.db_manager = db_manager
        self.schema_type = schema_type
        
        if schema_type == 'test_legacy':
            self.compat_layer = TestDatabaseCompatibilityLayer(db_manager)
        else:
            self.production_query = DataQuery(db_manager)
            
    def get_symbols(self, include_delisted: bool = False) -> List[Dict]:
        """Get symbols - compatible across schemas."""
        if self.schema_type == 'test_legacy':
            return self.compat_layer.get_symbols_compatible()
        else:
            return self.production_query.get_symbols(include_delisted)
            
    def get_latest_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest trades - compatible across schemas."""
        if self.schema_type == 'test_legacy':
            return self.compat_layer.get_trades_compatible(symbol, limit)
        else:
            return self.production_query.get_latest_trades(symbol, limit)
            
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest orderbook - compatible across schemas."""
        if self.schema_type == 'test_legacy':
            return self.compat_layer.get_orderbook_compatible(symbol)
        else:
            return self.production_query.get_latest_orderbook(symbol)
            
    def get_candles(self, timeframe: str, symbol: str, **kwargs) -> List[Dict]:
        """Get candles - only available in production schema."""
        if self.schema_type == 'test_legacy':
            logger.warning("Candles not available in test_legacy schema")
            return []
        else:
            return self.production_query.get_candles(timeframe, symbol, **kwargs)


# Global database switcher instance
_db_switcher = DatabaseSwitcher()


def get_database_switcher() -> DatabaseSwitcher:
    """Get the global database switcher instance."""
    return _db_switcher


def switch_database(database_name: str) -> DatabaseManager:
    """Convenience function to switch databases."""
    return _db_switcher.switch_to_database(database_name)


def get_compatible_query(db_manager: DatabaseManager, schema_type: str = 'production') -> CompatibleDataQuery:
    """Get a compatible data query instance."""
    return CompatibleDataQuery(db_manager, schema_type) 