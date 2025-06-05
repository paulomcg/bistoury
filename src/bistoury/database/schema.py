"""
Database Schema for HyperLiquid Market Data Storage

This module defines the database schema based on HyperLiquid API exploration.
Designed for high-performance data storage and efficient querying.
"""

from typing import Dict, List, Optional, Any
import duckdb
from datetime import datetime, timezone, timedelta
import logging
import json
from decimal import Decimal, InvalidOperation

from ..logger import get_logger

logger = get_logger(__name__)

class MarketDataSchema:
    """Manages database schema for market data storage."""
    
    def __init__(self, db_manager):
        """Initialize schema manager with database connection."""
        self.db_manager = db_manager
        
    def create_all_tables(self) -> None:
        """Create all market data tables in the correct order."""
        logger.debug("Creating market data schema...")
        
        # Create tables in dependency order
        self.create_symbols_table()
        self.create_candles_tables()
        self.create_trades_table()
        self.create_orderbook_snapshots_table()
        self.create_funding_rates_table()
        self.create_indices()
        
        logger.debug("Market data schema created successfully")
        
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
        
        logger.debug("Created symbols table")
        
    def create_candles_tables(self) -> None:
        """Create candlestick tables for different timeframes."""
        
        # Base candlestick table structure
        base_candles_sql = """
        CREATE TABLE IF NOT EXISTS {} (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time_ms BIGINT NOT NULL,
            close_time_ms BIGINT NOT NULL,
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
            
            logger.debug(f"Created {table_name} table")
            
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
        
        logger.debug("Created trades table")
        
    def create_orderbook_snapshots_table(self) -> None:
        """Create orderbook snapshots table."""
        # Order book snapshots table - Level 2 data with bids/asks
        # HyperLiquid L2Book structure: 
        # {"coin": "BTC", "time": 1748989278468, "levels": [[bids], [asks]]}
        # Each level: {"px": "105699.0", "sz": "12.97221", "n": 30}
        orderbook_snapshot_sql = """
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                id BIGINT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                time_ms BIGINT NOT NULL,  -- HyperLiquid's original timestamp
                bids JSON NOT NULL,       -- Array of {px, sz, n} objects
                asks JSON NOT NULL,       -- Array of {px, sz, n} objects
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, time_ms)
            )
        """
        
        self.db_manager.execute(orderbook_snapshot_sql)
        
        # Create indices separately for DuckDB compatibility  
        self.db_manager.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_time ON orderbook_snapshots(symbol, timestamp)")
        self.db_manager.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_time_ms ON orderbook_snapshots(time_ms)")
        
        # Create sequence for orderbook snapshots
        self.db_manager.execute("CREATE SEQUENCE IF NOT EXISTS orderbook_snapshots_seq START 1")
        
        # Create sequence for funding rates
        self.db_manager.execute("CREATE SEQUENCE IF NOT EXISTS funding_rates_seq START 1")
        
        logger.debug("Created orderbook_snapshots table")
        
    def create_funding_rates_table(self) -> None:
        """Create funding rates table for perpetual contracts."""
        # Funding rates table - Enhanced for HyperLiquid format
        # HyperLiquid fundingHistory structure:
        # {"coin": "BTC", "fundingRate": "0.0000125", "premium": "0.0004683623", "time": 1748905200002}
        funding_rates_sql = """
            CREATE TABLE IF NOT EXISTS funding_rates (
                id BIGINT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                time_ms BIGINT NOT NULL,         -- HyperLiquid's original timestamp
                funding_rate TEXT NOT NULL,      -- Stored as string for precision (like "0.0000125")
                funding_rate_decimal DECIMAL(20, 10), -- Converted decimal for calculations
                premium TEXT,                    -- Premium as string (like "0.0004683623")
                premium_decimal DECIMAL(20, 10), -- Premium as decimal for calculations
                predicted_rate TEXT,             -- For future predictions if available
                open_interest TEXT,              -- Open interest if available
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, time_ms)
            )
        """
        
        self.db_manager.execute(funding_rates_sql)
        
        # Create indices separately for DuckDB compatibility
        self.db_manager.execute("CREATE INDEX IF NOT EXISTS idx_funding_symbol_time ON funding_rates(symbol, timestamp)")
        self.db_manager.execute("CREATE INDEX IF NOT EXISTS idx_funding_time_ms ON funding_rates(time_ms)")
        
        logger.debug("Created funding_rates table")
        
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
            
        logger.debug("Created performance indices")
        
    def drop_all_tables(self) -> None:
        """Drop all tables for clean recreation."""
        tables = [
            'orderbook_snapshots',
            'funding_rates',
            'trades',
            'candles_1m', 'candles_5m', 'candles_15m', 
            'candles_1h', 'candles_4h', 'candles_1d',
            'symbols'
        ]
        
        for table in tables:
            try:
                self.db_manager.execute(f"DROP TABLE IF EXISTS {table}")
                logger.debug(f"Dropped table: {table}")
            except Exception as e:
                logger.warning(f"Could not drop table {table}: {e}")
                
        # Drop sequences
        sequences = [
            'symbol_seq', 'candle_seq', 'trade_seq', 
            'orderbook_snapshots_seq', 'funding_rates_seq'
        ]
        
        for seq in sequences:
            try:
                self.db_manager.execute(f"DROP SEQUENCE IF EXISTS {seq}")
                logger.debug(f"Dropped sequence: {seq}")
            except Exception as e:
                logger.warning(f"Could not drop sequence {seq}: {e}")
                
    def recreate_all_tables(self) -> None:
        """Drop and recreate all tables with fresh schema."""
        logger.debug("Recreating all tables with fresh schema...")
        self.drop_all_tables()
        self.create_all_tables()
        logger.debug("All tables recreated successfully")
        
    def get_schema_debug(self) -> Dict:
        """Get debugrmation about the current schema."""
        tables_debug = {}
        
        # Get list of tables
        tables_result = self.db_manager.execute(
            "SELECT table_name FROM debugrmation_schema.tables WHERE table_schema = 'main'"
        )
        
        for (table_name,) in tables_result:
            # Get column debug for each table
            columns_result = self.db_manager.execute(
                f"DESCRIBE {table_name}"
            )
            tables_debug[table_name] = {
                'columns': columns_result,
                'row_count': self._get_table_row_count(table_name)
            }
            
        return tables_debug
        
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
            "SELECT table_name FROM debugrmation_schema.tables WHERE table_schema = 'main'"
        )
        existing_tables = {row[0] for row in existing_tables_result}
        
        missing_tables = set(required_tables) - existing_tables
        
        if missing_tables:
            logger.error(f"Missing required tables: {missing_tables}")
            return False
            
        logger.debug("Schema validation passed")
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
        
        # Calculate end timestamp if not provided
        if 'T' in candle_data:
            end_ts = datetime.fromtimestamp(candle_data['T'] / 1000, tz=timezone.utc)
        else:
            # Calculate end time based on timeframe
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
            }
            minutes = timeframe_minutes.get(timeframe, 1)
            end_ts = start_ts + timedelta(minutes=minutes)
        
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
        
    def insert_orderbook_snapshot(self, data: Dict[str, Any]) -> None:
        """
        Insert a single order book snapshot from HyperLiquid L2Book format.
        
        Expected data format:
        {
            "coin": "BTC",
            "time": 1748989278468,
            "levels": [
                [{"px": "105699.0", "sz": "12.97221", "n": 30}, ...],  # bids
                [{"px": "105700.0", "sz": "5.5", "n": 15}, ...]        # asks
            ]
        }
        """
        if not isinstance(data, dict) or 'coin' not in data or 'time' not in data or 'levels' not in data:
            raise ValueError("Invalid orderbook data format")
        
        levels = data['levels']
        if not isinstance(levels, list) or len(levels) != 2:
            raise ValueError("Invalid levels format - expected [bids, asks]")
        
        # Convert millisecond timestamp to datetime
        timestamp = datetime.fromtimestamp(data['time'] / 1000, tz=timezone.utc)
        
        sql = """
            INSERT INTO orderbook_snapshots 
            (id, symbol, timestamp, time_ms, bids, asks, created_at)
            VALUES (nextval('orderbook_snapshots_seq'), ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, time_ms) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                bids = EXCLUDED.bids,
                asks = EXCLUDED.asks,
                created_at = EXCLUDED.created_at
        """
        
        # Store bids and asks as JSON strings
        bids_json = json.dumps(levels[0]) if levels[0] else '[]'
        asks_json = json.dumps(levels[1]) if levels[1] else '[]'
        
        self.db_manager.execute(sql, (
            data['coin'],
            timestamp,
            data['time'],
            bids_json,
            asks_json,
            datetime.now(timezone.utc)
        ))

    def insert_funding_rate(self, data: Dict[str, Any]) -> None:
        """
        Insert a single funding rate record from HyperLiquid fundingHistory format.
        
        Expected data format:
        {
            "coin": "BTC",
            "fundingRate": "0.0000125",
            "premium": "0.0004683623", 
            "time": 1748905200002
        }
        """
        if not isinstance(data, dict) or 'coin' not in data or 'time' not in data:
            raise ValueError("Invalid funding rate data format")
        
        # Convert millisecond timestamp to datetime
        timestamp = datetime.fromtimestamp(data['time'] / 1000, tz=timezone.utc)
        
        # Convert string values to decimals for calculations (handle empty/null values)
        funding_rate_decimal = None
        premium_decimal = None
        
        if 'fundingRate' in data and data['fundingRate']:
            try:
                funding_rate_decimal = Decimal(data['fundingRate'])
            except (ValueError, TypeError, InvalidOperation):
                pass
                
        if 'premium' in data and data['premium']:
            try:
                premium_decimal = Decimal(data['premium'])
            except (ValueError, TypeError, InvalidOperation):
                pass
        
        sql = """
            INSERT INTO funding_rates 
            (id, symbol, timestamp, time_ms, funding_rate, funding_rate_decimal, 
             premium, premium_decimal, predicted_rate, open_interest, created_at)
            VALUES (nextval('funding_rates_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, time_ms) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                funding_rate = EXCLUDED.funding_rate,
                funding_rate_decimal = EXCLUDED.funding_rate_decimal,
                premium = EXCLUDED.premium,
                premium_decimal = EXCLUDED.premium_decimal,
                predicted_rate = EXCLUDED.predicted_rate,
                open_interest = EXCLUDED.open_interest,
                created_at = EXCLUDED.created_at
        """
        
        self.db_manager.execute(sql, (
            data['coin'],
            timestamp,
            data['time'],
            data.get('fundingRate', ''),
            funding_rate_decimal,
            data.get('premium', ''),
            premium_decimal,
            data.get('predictedRate', ''),
            data.get('openInterest', ''),
            datetime.now(timezone.utc)
        ))
        
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
        
    def get_orderbook_snapshots(
        self, 
        symbol: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get orderbook snapshots for a symbol within a time range.
        
        Returns data in HyperLiquid-compatible format with parsed bids/asks.
        """
        sql = """
            SELECT symbol, timestamp, time_ms, bids, asks, created_at
            FROM orderbook_snapshots 
            WHERE symbol = ?
        """
        params = [symbol]
        
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
            
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        results = self.db_manager.execute(sql, params)
        
        # Parse JSON data and format results
        snapshots = []
        for row in results:
            try:
                bids = json.loads(row[3]) if row[3] else []
                asks = json.loads(row[4]) if row[4] else []
                
                snapshots.append({
                    'coin': row[0],
                    'timestamp': row[1],
                    'time': row[2],  # Original HyperLiquid timestamp
                    'levels': [bids, asks],
                    'created_at': row[5]
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for orderbook snapshot: {e}")
                continue
                
        return snapshots

    def get_funding_rates(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get funding rates for a symbol within a time range.
        
        Returns data in HyperLiquid-compatible format.
        """
        sql = """
            SELECT symbol, timestamp, time_ms, funding_rate, funding_rate_decimal,
                   premium, premium_decimal, predicted_rate, open_interest, created_at
            FROM funding_rates 
            WHERE symbol = ?
        """
        params = [symbol]
        
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
            
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        results = self.db_manager.execute(sql, params)
        
        # Format results in HyperLiquid format
        rates = []
        for row in results:
            rates.append({
                'coin': row[0],
                'timestamp': row[1],
                'time': row[2],  # Original HyperLiquid timestamp
                'fundingRate': row[3],  # String value as received from API
                'fundingRateDecimal': float(row[4]) if row[4] is not None else None,
                'premium': row[5],
                'premiumDecimal': float(row[6]) if row[6] is not None else None,
                'predictedRate': row[7],
                'openInterest': row[8],
                'created_at': row[9]
            })
                
        return rates
        
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent orderbook snapshot for a symbol."""
        sql = """
            SELECT symbol, timestamp, time_ms, bids, asks, created_at
            FROM orderbook_snapshots 
            WHERE symbol = ?
            ORDER BY time_ms DESC 
            LIMIT 1
        """
        
        results = self.db_manager.execute(sql, [symbol])
        if not results:
            return None
            
        row = results[0]
        try:
            bids = json.loads(row[3]) if row[3] else []
            asks = json.loads(row[4]) if row[4] else []
            
            return {
                'coin': row[0],
                'timestamp': row[1],
                'time': row[2],
                'levels': [bids, asks],
                'created_at': row[5]
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for latest orderbook: {e}")
            return None
            
    def get_latest_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent funding rate for a symbol."""
        sql = """
            SELECT symbol, timestamp, time_ms, funding_rate, funding_rate_decimal,
                   premium, premium_decimal, predicted_rate, open_interest, created_at
            FROM funding_rates 
            WHERE symbol = ?
            ORDER BY time_ms DESC 
            LIMIT 1
        """
        
        results = self.db_manager.execute(sql, [symbol])
        if not results:
            return None
            
        row = results[0]
        return {
            'coin': row[0],
            'timestamp': row[1],
            'time': row[2],
            'fundingRate': row[3],
            'fundingRateDecimal': float(row[4]) if row[4] is not None else None,
            'premium': row[5],
            'premiumDecimal': float(row[6]) if row[6] is not None else None,
            'predictedRate': row[7],
            'openInterest': row[8],
            'created_at': row[9]
        } 