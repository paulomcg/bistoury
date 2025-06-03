"""
Database Schema for HyperLiquid Market Data Storage

This module defines the database schema based on HyperLiquid API exploration.
Designed for high-performance data storage and efficient querying.
"""

from typing import Dict, List, Optional
import duckdb
from datetime import datetime, timezone
import logging
import json

logger = logging.getLogger(__name__)

class MarketDataSchema:
    """Manages database schema for market data storage."""
    
    def __init__(self, db_manager):
        """Initialize schema manager with database connection."""
        self.db_manager = db_manager
        
    def create_all_tables(self) -> None:
        """Create all market data tables in the correct order."""
        logger.info("Creating market data schema...")
        
        # Create tables in dependency order
        self.create_symbols_table()
        self.create_candles_tables()
        self.create_trades_table()
        self.create_orderbook_snapshots_table()
        self.create_funding_rates_table()
        self.create_indices()
        
        logger.info("Market data schema created successfully")
        
    def create_symbols_table(self) -> None:
        """Create symbols metadata table."""
        sql = """
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY,
            symbol TEXT UNIQUE NOT NULL,
            name TEXT,
            sz_decimals INTEGER,
            max_leverage DOUBLE,
            margin_table_id INTEGER,
            is_delisted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.db_manager.execute(sql)
        
        # Create sequence for auto-incrementing IDs 
        seq_sql = "CREATE SEQUENCE IF NOT EXISTS symbols_seq START 1"
        self.db_manager.execute(seq_sql)
        
        logger.info("Created symbols table")
        
    def create_candles_tables(self) -> None:
        """Create candlestick tables for different timeframes."""
        
        # Base candlestick table structure
        base_candles_sql = """
        CREATE TABLE IF NOT EXISTS {} (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp_start TIMESTAMP NOT NULL,
            timestamp_end TIMESTAMP NOT NULL,
            open_price DECIMAL(20,8) NOT NULL,
            close_price DECIMAL(20,8) NOT NULL,
            high_price DECIMAL(20,8) NOT NULL,
            low_price DECIMAL(20,8) NOT NULL,
            volume DECIMAL(20,8) NOT NULL,
            trade_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp_start)
        )
        """
        
        # Create tables for different timeframes
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        for timeframe in timeframes:
            table_name = f"candles_{timeframe}"
            sql = base_candles_sql.format(table_name)
            self.db_manager.execute(sql)
            
            # Create sequence for each table
            seq_sql = f"CREATE SEQUENCE IF NOT EXISTS {table_name}_seq START 1"
            self.db_manager.execute(seq_sql)
            
            logger.info(f"Created {table_name} table")
            
    def create_trades_table(self) -> None:
        """Create trades table for individual trade records."""
        sql = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            price DECIMAL(20,8) NOT NULL,
            size DECIMAL(20,8) NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('A', 'B')), -- A=Ask/Sell, B=Bid/Buy
            trade_id BIGINT UNIQUE,
            hash TEXT,
            user1 TEXT,
            user2 TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.db_manager.execute(sql)
        
        # Create sequence for trades
        seq_sql = "CREATE SEQUENCE IF NOT EXISTS trades_seq START 1"
        self.db_manager.execute(seq_sql)
        
        logger.info("Created trades table")
        
    def create_orderbook_snapshots_table(self) -> None:
        """Create orderbook snapshots table."""
        sql = """
        CREATE TABLE IF NOT EXISTS orderbook_snapshots (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            bids JSON NOT NULL,  -- Array of {px, sz, n} objects
            asks JSON NOT NULL,  -- Array of {px, sz, n} objects
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.db_manager.execute(sql)
        
        # Create sequence for orderbook snapshots
        seq_sql = "CREATE SEQUENCE IF NOT EXISTS orderbook_snapshots_seq START 1"
        self.db_manager.execute(seq_sql)
        
        logger.info("Created orderbook_snapshots table")
        
    def create_funding_rates_table(self) -> None:
        """Create funding rates table for perpetual contracts."""
        sql = """
        CREATE TABLE IF NOT EXISTS funding_rates (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            funding_rate DECIMAL(20,8) NOT NULL,
            predicted_rate DECIMAL(20,8),
            open_interest DECIMAL(20,8),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
        """
        self.db_manager.execute(sql)
        
        # Create sequence for funding rates
        seq_sql = "CREATE SEQUENCE IF NOT EXISTS funding_rates_seq START 1"
        self.db_manager.execute(seq_sql)
        
        logger.info("Created funding_rates table")
        
    def create_indices(self) -> None:
        """Create performance indices on frequently queried columns."""
        indices = [
            # Symbols indices
            "CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_symbols_is_delisted ON symbols(is_delisted)",
            
            # Trades indices
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id)",
            
            # Orderbook indices
            "CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_timestamp ON orderbook_snapshots(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_orderbook_timestamp ON orderbook_snapshots(timestamp)",
            
            # Funding rates indices
            "CREATE INDEX IF NOT EXISTS idx_funding_symbol_timestamp ON funding_rates(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_funding_timestamp ON funding_rates(timestamp)",
        ]
        
        # Create indices for all candlestick tables
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            table_name = f"candles_{timeframe}"
            indices.extend([
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_timestamp ON {table_name}(symbol, timestamp_start)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp_start)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)",
            ])
            
        # Execute all index creation statements
        for sql in indices:
            self.db_manager.execute(sql)
            
        logger.info("Created performance indices")
        
    def drop_all_tables(self) -> None:
        """Drop all market data tables (for testing/reset)."""
        tables = [
            'funding_rates',
            'orderbook_snapshots', 
            'trades',
            'candles_1m', 'candles_5m', 'candles_15m', 
            'candles_1h', 'candles_4h', 'candles_1d',
            'symbols'
        ]
        
        for table in tables:
            sql = f"DROP TABLE IF EXISTS {table}"
            self.db_manager.execute(sql)
            
        logger.info("Dropped all market data tables")
        
    def get_schema_info(self) -> Dict:
        """Get information about the current schema."""
        tables_info = {}
        
        # Get list of tables
        tables_result = self.db_manager.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        )
        
        for (table_name,) in tables_result:
            # Get column info for each table
            columns_result = self.db_manager.execute(
                f"DESCRIBE {table_name}"
            )
            tables_info[table_name] = {
                'columns': columns_result,
                'row_count': self._get_table_row_count(table_name)
            }
            
        return tables_info
        
    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        try:
            result = self.db_manager.execute(f"SELECT COUNT(*) FROM {table_name}")
            return result[0][0] if result else 0
        except Exception:
            return 0
            
    def validate_schema(self) -> bool:
        """Validate that all required tables and indices exist."""
        required_tables = [
            'symbols', 'trades', 'orderbook_snapshots', 'funding_rates',
            'candles_1m', 'candles_5m', 'candles_15m', 
            'candles_1h', 'candles_4h', 'candles_1d'
        ]
        
        existing_tables_result = self.db_manager.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        )
        existing_tables = {row[0] for row in existing_tables_result}
        
        missing_tables = set(required_tables) - existing_tables
        
        if missing_tables:
            logger.error(f"Missing required tables: {missing_tables}")
            return False
            
        logger.info("Schema validation passed")
        return True


class DataInsertion:
    """Handles efficient data insertion operations."""
    
    def __init__(self, db_manager):
        """Initialize data insertion handler."""
        self.db_manager = db_manager
        
    def insert_symbol(self, symbol_data: Dict) -> int:
        """Insert or update symbol metadata."""
        # First try to get existing symbol ID
        existing_sql = "SELECT id FROM symbols WHERE symbol = ?"
        existing_result = self.db_manager.execute(existing_sql, (symbol_data.get('name'),))
        
        if existing_result:
            # Update existing symbol
            sql = """
            UPDATE symbols SET
                name = ?,
                sz_decimals = ?,
                max_leverage = ?,
                margin_table_id = ?,
                is_delisted = ?,
                updated_at = ?
            WHERE symbol = ?
            """
            
            current_time = datetime.now(timezone.utc)
            
            values = (
                symbol_data.get('name'),  # name
                symbol_data.get('szDecimals', 0),
                symbol_data.get('maxLeverage', 1),
                symbol_data.get('marginTableId'),
                symbol_data.get('isDelisted', False),
                current_time,
                symbol_data.get('name')  # symbol for WHERE clause
            )
            
            self.db_manager.execute(sql, values)
            return existing_result[0][0]  # Return existing ID
        else:
            # Insert new symbol with auto-generated ID
            sql = """
            INSERT INTO symbols 
            (id, symbol, name, sz_decimals, max_leverage, margin_table_id, is_delisted, updated_at)
            VALUES (nextval('symbols_seq'), ?, ?, ?, ?, ?, ?, ?)
            """
            
            current_time = datetime.now(timezone.utc)
            
            values = (
                symbol_data.get('name'),  # symbol
                symbol_data.get('name'),  # name (same as symbol)
                symbol_data.get('szDecimals', 0),
                symbol_data.get('maxLeverage', 1),
                symbol_data.get('marginTableId'),
                symbol_data.get('isDelisted', False),
                current_time
            )
            
            self.db_manager.execute(sql, values)
            
            # Get the inserted ID
            new_id_result = self.db_manager.execute("SELECT currval('symbols_seq')")
            return new_id_result[0][0]
        
    def insert_candle(self, timeframe: str, candle_data: Dict) -> int:
        """Insert candlestick data."""
        table_name = f"candles_{timeframe}"
        
        # Convert timestamps from milliseconds to datetime
        start_ts = datetime.fromtimestamp(candle_data['t'] / 1000, tz=timezone.utc)
        end_ts = datetime.fromtimestamp(candle_data['T'] / 1000, tz=timezone.utc)
        
        # Check if candle already exists
        existing_sql = f"SELECT id FROM {table_name} WHERE symbol = ? AND timestamp_start = ?"
        existing_result = self.db_manager.execute(existing_sql, (candle_data['s'], start_ts))
        
        if existing_result:
            # Update existing candle
            sql = f"""
            UPDATE {table_name} SET
                timestamp_end = ?,
                open_price = ?,
                close_price = ?,
                high_price = ?,
                low_price = ?,
                volume = ?,
                trade_count = ?
            WHERE symbol = ? AND timestamp_start = ?
            """
            
            values = (
                end_ts,
                float(candle_data['o']),  # open
                float(candle_data['c']),  # close
                float(candle_data['h']),  # high
                float(candle_data['l']),  # low
                float(candle_data['v']),  # volume
                candle_data.get('n', 0),   # trade_count
                candle_data['s'],  # symbol
                start_ts
            )
            
            self.db_manager.execute(sql, values)
            return existing_result[0][0]
        else:
            # Insert new candle
            sql = f"""
            INSERT INTO {table_name}
            (id, symbol, timestamp_start, timestamp_end, open_price, close_price, 
             high_price, low_price, volume, trade_count)
            VALUES (nextval('{table_name}_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                candle_data['s'],  # symbol
                start_ts,
                end_ts,
                float(candle_data['o']),  # open
                float(candle_data['c']),  # close
                float(candle_data['h']),  # high
                float(candle_data['l']),  # low
                float(candle_data['v']),  # volume
                candle_data.get('n', 0)   # trade_count
            )
            
            self.db_manager.execute(sql, values)
            
            # Get the inserted ID
            new_id_result = self.db_manager.execute(f"SELECT currval('{table_name}_seq')")
            return new_id_result[0][0]
        
    def insert_trade(self, trade_data: Dict) -> int:
        """Insert trade data."""
        # Convert timestamp from milliseconds
        timestamp = datetime.fromtimestamp(trade_data['time'] / 1000, tz=timezone.utc)
        
        # Extract users safely
        users = trade_data.get('users', [])
        user1 = users[0] if len(users) > 0 else None
        user2 = users[1] if len(users) > 1 else None
        
        # Check if trade already exists (by trade_id)
        if 'tid' in trade_data:
            existing_sql = "SELECT id FROM trades WHERE trade_id = ?"
            existing_result = self.db_manager.execute(existing_sql, (trade_data['tid'],))
            
            if existing_result:
                return existing_result[0][0]  # Trade already exists, return existing ID
        
        # Insert new trade
        sql = """
        INSERT INTO trades
        (id, symbol, timestamp, price, size, side, trade_id, hash, user1, user2)
        VALUES (nextval('trades_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            trade_data['coin'],
            timestamp,
            float(trade_data['px']),
            float(trade_data['sz']),
            trade_data['side'],
            trade_data.get('tid'),
            trade_data.get('hash'),
            user1,
            user2
        )
        
        self.db_manager.execute(sql, values)
        
        # Get the inserted ID
        new_id_result = self.db_manager.execute("SELECT currval('trades_seq')")
        return new_id_result[0][0]
        
    def insert_orderbook_snapshot(self, orderbook_data: Dict) -> int:
        """Insert orderbook snapshot."""
        # Convert timestamp from milliseconds
        timestamp = datetime.fromtimestamp(orderbook_data['time'] / 1000, tz=timezone.utc)
        
        # Extract bids and asks from levels array
        levels = orderbook_data.get('levels', [[], []])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        
        # Insert new orderbook snapshot
        sql = """
        INSERT INTO orderbook_snapshots
        (id, symbol, timestamp, bids, asks)
        VALUES (nextval('orderbook_snapshots_seq'), ?, ?, ?, ?)
        """
        
        values = (
            orderbook_data['coin'],
            timestamp,
            json.dumps(bids),  # Convert to proper JSON string
            json.dumps(asks)   # Convert to proper JSON string
        )
        
        self.db_manager.execute(sql, values)
        
        # Get the inserted ID
        new_id_result = self.db_manager.execute("SELECT currval('orderbook_snapshots_seq')")
        return new_id_result[0][0]
        
    def insert_funding_rate(self, funding_data: Dict) -> int:
        """Insert funding rate data."""
        timestamp = datetime.fromtimestamp(funding_data['timestamp'] / 1000, tz=timezone.utc)
        
        # Check if funding rate already exists
        existing_sql = "SELECT id FROM funding_rates WHERE symbol = ? AND timestamp = ?"
        existing_result = self.db_manager.execute(existing_sql, (funding_data['symbol'], timestamp))
        
        if existing_result:
            # Update existing funding rate
            sql = """
            UPDATE funding_rates SET
                funding_rate = ?,
                predicted_rate = ?,
                open_interest = ?
            WHERE symbol = ? AND timestamp = ?
            """
            
            values = (
                float(funding_data['funding_rate']),
                funding_data.get('predicted_rate'),
                funding_data.get('open_interest'),
                funding_data['symbol'],
                timestamp
            )
            
            self.db_manager.execute(sql, values)
            return existing_result[0][0]
        else:
            # Insert new funding rate
            sql = """
            INSERT INTO funding_rates
            (id, symbol, timestamp, funding_rate, predicted_rate, open_interest)
            VALUES (nextval('funding_rates_seq'), ?, ?, ?, ?, ?)
            """
            
            values = (
                funding_data['symbol'],
                timestamp,
                float(funding_data['funding_rate']),
                funding_data.get('predicted_rate'),
                funding_data.get('open_interest')
            )
            
            self.db_manager.execute(sql, values)
            
            # Get the inserted ID
            new_id_result = self.db_manager.execute("SELECT currval('funding_rates_seq')")
            return new_id_result[0][0]
        
    def bulk_insert_candles(self, timeframe: str, candles: List[Dict]) -> int:
        """Bulk insert candlestick data for better performance."""
        if not candles:
            return 0
            
        table_name = f"candles_{timeframe}"
        
        sql = f"""
        INSERT INTO {table_name}
        (id, symbol, timestamp_start, timestamp_end, open_price, close_price, 
         high_price, low_price, volume, trade_count)
        VALUES (nextval('{table_name}_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Prepare data for bulk insert
        values_list = []
        for candle in candles:
            start_ts = datetime.fromtimestamp(candle['t'] / 1000, tz=timezone.utc)
            end_ts = datetime.fromtimestamp(candle['T'] / 1000, tz=timezone.utc)
            
            values = (
                candle['s'],  # symbol
                start_ts,
                end_ts,
                float(candle['o']),  # open
                float(candle['c']),  # close
                float(candle['h']),  # high
                float(candle['l']),  # low
                float(candle['v']),  # volume
                candle.get('n', 0)   # trade_count
            )
            values_list.append(values)
            
        # Execute bulk insert
        self.db_manager.execute_many(sql, values_list)
        return len(values_list)


class DataQuery:
    """Handles data querying operations."""
    
    def __init__(self, db_manager):
        """Initialize data query handler."""
        self.db_manager = db_manager
        
    def get_symbols(self, include_delisted: bool = False) -> List[Dict]:
        """Get all symbols."""
        sql = "SELECT * FROM symbols"
        if not include_delisted:
            sql += " WHERE is_delisted = FALSE"
        sql += " ORDER BY symbol"
        
        results = self.db_manager.execute(sql)
        
        # Convert to dictionaries
        columns = ['id', 'symbol', 'name', 'sz_decimals', 'max_leverage', 
                  'margin_table_id', 'is_delisted', 'created_at', 'updated_at']
        
        return [dict(zip(columns, row)) for row in results]
        
    def get_candles(self, timeframe: str, symbol: str, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[Dict]:
        """Get candlestick data."""
        table_name = f"candles_{timeframe}"
        
        sql = f"SELECT * FROM {table_name} WHERE symbol = ?"
        params = [symbol]
        
        if start_time:
            sql += " AND timestamp_start >= ?"
            params.append(start_time)
            
        if end_time:
            sql += " AND timestamp_start <= ?"
            params.append(end_time)
            
        sql += " ORDER BY timestamp_start"
        
        if limit:
            sql += f" LIMIT {limit}"
            
        results = self.db_manager.execute(sql, tuple(params))
        
        # Convert to dictionaries
        columns = ['id', 'symbol', 'timestamp_start', 'timestamp_end',
                  'open_price', 'close_price', 'high_price', 'low_price',
                  'volume', 'trade_count', 'created_at']
        
        return [dict(zip(columns, row)) for row in results]
        
    def get_latest_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get latest trades for a symbol."""
        sql = """
        SELECT * FROM trades 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        results = self.db_manager.execute(sql, (symbol, limit))
        
        columns = ['id', 'symbol', 'timestamp', 'price', 'size', 'side',
                  'trade_id', 'hash', 'user1', 'user2', 'created_at']
        
        return [dict(zip(columns, row)) for row in results]
        
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get the latest orderbook snapshot for a symbol."""
        sql = """
        SELECT * FROM orderbook_snapshots 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        
        results = self.db_manager.execute(sql, (symbol,))
        
        if not results:
            return None
            
        result = results[0]
        columns = ['id', 'symbol', 'timestamp', 'bids', 'asks', 'created_at']
        orderbook = dict(zip(columns, result))
        
        # Parse JSON fields
        try:
            orderbook['bids'] = json.loads(orderbook['bids'])
            orderbook['asks'] = json.loads(orderbook['asks'])
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, keep as string
            pass
            
        return orderbook 