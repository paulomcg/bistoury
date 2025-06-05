"""
Enhanced HyperLiquid data collector for Bistoury.
Collects market data from HyperLiquid and stores it using optimized database models.

This implementation uses the database entity models from task 4.7 for:
- Type safety and data validation  
- Optimized batch processing
- Compression-aware serialization
- Performance monitoring
- Data integrity validation
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field

from ..database import DatabaseManager
from ..logger import get_logger
from ..models.database import (
    DBCandlestickData, 
    DBTradeData, 
    DBOrderBookSnapshot,
    DBFundingRateData,
    DBBatchOperation
)
from ..models.serialization import (
    BatchProcessor,
    DatabaseSerializer,
    ModelConverter,
    DataIntegrityValidator,
    SerializationFormat,
    CompressionLevel
)
from ..models.market_data import CandlestickData, Timeframe
from .client import HyperLiquidIntegration

logger = get_logger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for the data collector."""
    symbols: Set[str] = field(default_factory=set)
    intervals: Set[str] = field(default_factory=lambda: {'1m', '5m', '15m', '1h', '4h', '1d'})
    buffer_size: int = 1000
    flush_interval: float = 30.0
    max_batch_size: int = 5000
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    enable_validation: bool = True
    enable_monitoring: bool = True
    orderbook_symbols: Set[str] = field(default_factory=lambda: {'BTC', 'ETH', 'SOL'})
    max_concurrent_subscriptions: int = 20


@dataclass 
class CollectorStats:
    """Statistics for data collection operations."""
    candles_collected: int = 0
    trades_collected: int = 0
    orderbooks_collected: int = 0
    funding_rates_collected: int = 0
    historical_requests: int = 0
    total_candles_stored: int = 0
    errors: int = 0
    validation_errors: int = 0
    batches_processed: int = 0
    last_activity: Optional[datetime] = None
    start_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'candles_collected': self.candles_collected,
            'trades_collected': self.trades_collected,
            'orderbooks_collected': self.orderbooks_collected,
            'funding_rates_collected': self.funding_rates_collected,
            'historical_requests': self.historical_requests,
            'total_candles_stored': self.total_candles_stored,
            'errors': self.errors,
            'validation_errors': self.validation_errors,
            'batches_processed': self.batches_processed,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (
                (datetime.now(timezone.utc) - self.start_time).total_seconds()
                if self.start_time else 0
            )
        }


class EnhancedDataCollector:
    """
    Enhanced data collector using database entity models for optimal performance.
    
    Features:
    - Type-safe data handling with Pydantic models
    - Batch processing with compression
    - Data validation and integrity checks
    - Performance monitoring and metrics
    - Graceful error handling and recovery
    - Efficient database operations
    """
    
    def __init__(
        self, 
        hyperliquid: HyperLiquidIntegration, 
        db_manager: DatabaseManager,
        config: Optional[CollectorConfig] = None
    ):
        self.hyperliquid = hyperliquid
        self.db_manager = db_manager
        self.config = config or CollectorConfig()
        
        # Initialize processing components
        self.batch_processor = BatchProcessor(
            batch_size=self.config.buffer_size,
            enable_validation=self.config.enable_validation
        )
        self.serializer = DatabaseSerializer(
            compression_level=self.config.compression_level,
            batch_size=self.config.buffer_size
        )
        self.converter = ModelConverter()
        self.validator = DataIntegrityValidator()
        
        # Collection status
        self.running = False
        self.active_subscriptions: Set[str] = set()
        self.collection_tasks: List[asyncio.Task] = []
        
        # Enhanced data buffers with type safety
        self.candle_buffer: List[Tuple[DBCandlestickData, str]] = []
        self.trade_buffer: List[DBTradeData] = []
        self.orderbook_buffer: List[DBOrderBookSnapshot] = []
        self.funding_rate_buffer: List[DBFundingRateData] = []
        
        # Statistics and monitoring
        self.stats = CollectorStats()
        self.batch_operations: Dict[str, DBBatchOperation] = {}
        
        logger.info(f"EnhancedDataCollector initialized with {len(self.config.symbols)} symbols")
        
    async def start(self) -> bool:
        """
        Start the enhanced data collection process.
        
        Returns:
            bool: True if started successfully
        """
        if self.running:
            logger.warning("Enhanced DataCollector is already running")
            return True
        
        try:
            # Ensure HyperLiquid is connected
            if not self.hyperliquid.is_connected():
                logger.info("Connecting to HyperLiquid...")
                if not await self.hyperliquid.connect():
                    logger.error("Failed to connect to HyperLiquid")
                    return False
            
            # Auto-discover symbols if none specified
            if not self.config.symbols:
                await self._discover_symbols()
            
            # Initialize database tables
            await self._initialize_database_tables()
            
            # Start collection
            self.running = True
            self.stats.start_time = datetime.now(timezone.utc)
            
            # Start background tasks
            self.collection_tasks = [
                asyncio.create_task(self._periodic_flush()),
                asyncio.create_task(self._periodic_stats()),
                asyncio.create_task(self._health_monitor()),
            ]
            
            # Subscribe to data feeds
            await self._subscribe_to_enhanced_feeds()
            
            logger.info("Enhanced DataCollector started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Enhanced DataCollector: {e}")
            self.running = False
            return False
    
    async def stop(self) -> None:
        """Stop the data collection process gracefully."""
        logger.info("Stopping Enhanced DataCollector...")
        
        self.running = False
        
        # Cancel all background tasks
        for task in self.collection_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        # Flush all remaining data
        await self._flush_all_buffers()
        
        # Complete any ongoing batch operations
        await self._complete_batch_operations()
        
        # Clear subscriptions
        self.active_subscriptions.clear()
        
        logger.info("Enhanced DataCollector stopped gracefully")
    
    async def _discover_symbols(self) -> None:
        """Discover available trading symbols from HyperLiquid."""
        try:
            logger.info("Discovering trading symbols...")
            
            meta = await self.hyperliquid.get_meta()
            universe = meta.get('universe', [])
            
            discovered_symbols = set()
            for symbol_info in universe:
                symbol = symbol_info.get('name')
                if symbol:
                    discovered_symbols.add(symbol)
            
            self.config.symbols = discovered_symbols
            logger.info(f"Discovered {len(self.config.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            # Use default symbols if discovery fails
            self.config.symbols = {'BTC', 'ETH', 'SOL', 'AVAX', 'ARB'}
            logger.info(f"Using default symbols: {self.config.symbols}")
    
    async def _initialize_database_tables(self) -> None:
        """Initialize database tables for enhanced data storage."""
        try:
            # Use the standard schema from database.schema module
            from ..database.schema import MarketDataSchema
            
            schema = MarketDataSchema(self.db_manager)
            schema.create_all_tables()
            
            logger.info("Database tables initialized successfully using standard schema")
            
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise
    
    async def _create_optimized_table(self, table_name: str, description: str) -> None:
        """Create optimized table schema based on data type."""
        try:
            if 'candles' in table_name:
                schema = """
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,         -- HyperLiquid 's' field
                    interval TEXT NOT NULL,       -- HyperLiquid 'i' field 
                    open_time_ms BIGINT NOT NULL, -- HyperLiquid 't' field (open time millis)
                    close_time_ms BIGINT NOT NULL,-- HyperLiquid 'T' field (close time millis)
                    open_price TEXT NOT NULL,     -- HyperLiquid 'o' field
                    high_price TEXT NOT NULL,     -- HyperLiquid 'h' field
                    low_price TEXT NOT NULL,      -- HyperLiquid 'l' field
                    close_price TEXT NOT NULL,    -- HyperLiquid 'c' field
                    volume TEXT NOT NULL,         -- HyperLiquid 'v' field
                    trade_count INTEGER,          -- HyperLiquid 'n' field
                    timestamp_start TIMESTAMP NOT NULL,  -- Computed from open_time_ms for indexing
                    timestamp_end TIMESTAMP NOT NULL,    -- Computed from close_time_ms for indexing
                    UNIQUE(symbol, open_time_ms)  -- Unique on symbol and open time
                )
                """.format(table_name=table_name)
                
            elif table_name == 'trades':
                schema = """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    trade_id BIGINT NOT NULL,
                    price TEXT NOT NULL,
                    size TEXT NOT NULL,
                    side TEXT NOT NULL,
                    hash TEXT,
                    user1 TEXT,
                    user2 TEXT,
                    UNIQUE(trade_id)
                )
                """
                
            elif table_name == 'order_book_snapshots':
                schema = """
                CREATE TABLE IF NOT EXISTS order_book_snapshots (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    time_ms BIGINT NOT NULL,
                    bids TEXT NOT NULL,
                    asks TEXT NOT NULL,
                    UNIQUE(symbol, time_ms)
                )
                """
                
            elif table_name == 'funding_rates':
                schema = """
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    time_ms BIGINT NOT NULL,
                    funding_rate TEXT NOT NULL,
                    premium TEXT,
                    UNIQUE(symbol, time_ms)
                )
                """
                
            elif table_name == 'batch_operations':
                schema = """
                CREATE TABLE IF NOT EXISTS batch_operations (
                    id INTEGER PRIMARY KEY,
                    batch_id TEXT UNIQUE NOT NULL,
                    operation_type TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    error_message TEXT,
                    metrics_json TEXT
                )
                """
            else:
                return
                
            self.db_manager.execute(schema)
            
            # Create indices for performance
            await self._create_table_indices(table_name)
            
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise
    
    async def _create_table_indices(self, table_name: str) -> None:
        """Create optimized indices for tables."""
        try:
            indices = []
            
            if table_name.startswith('candles_'):
                # Candle tables use timestamp_start/timestamp_end
                indices = [
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time ON {table_name}(symbol, open_time_ms)",
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {table_name}(timestamp_start)",
                ]
            elif table_name == "trades":
                # Trades table uses timestamp (not time_ms)
                indices = [
                    f"CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)",
                    f"CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
                ]
            elif table_name in ["orderbook_snapshots", "order_book_snapshots", "funding_rates"]:
                # Other data tables use timestamp
                indices = [
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time ON {table_name}(symbol, timestamp)",
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {table_name}(timestamp)",
                ]
            elif table_name == "batch_operations":
                # Batch operations table has different schema - only create relevant indexes
                indices = [
                    f"CREATE INDEX IF NOT EXISTS idx_batch_operations_table ON batch_operations(table_name)",
                    f"CREATE INDEX IF NOT EXISTS idx_batch_operations_status ON batch_operations(status)",
                    f"CREATE INDEX IF NOT EXISTS idx_batch_operations_start_time ON batch_operations(start_time)",
                ]
            # Skip index creation for other tables that don't follow the standard schema
            
            for index_sql in indices:
                self.db_manager.execute(index_sql)
                
        except Exception as e:
            logger.error(f"Failed to create indices for {table_name}: {e}")
    
    async def _subscribe_to_enhanced_feeds(self) -> None:
        """Subscribe to enhanced data feeds with better management."""
        try:
            subscription_count = 0
            max_subscriptions = self.config.max_concurrent_subscriptions
            
            # Subscribe to all mid prices (global price updates)
            success = await self.hyperliquid.subscribe_all_mids(self._handle_enhanced_price_update)
            if success:
                self.active_subscriptions.add('allMids')
                logger.debug("Subscribed to enhanced all mid prices")
            
            # Subscribe to individual symbol feeds with limits
            symbol_list = list(self.config.symbols)[:max_subscriptions]
            
            for symbol in symbol_list:
                if subscription_count >= max_subscriptions:
                    break
                    
                # Subscribe to trades
                trade_success = await self.hyperliquid.subscribe_trades(
                    symbol, self._handle_enhanced_trade_update
                )
                if trade_success:
                    self.active_subscriptions.add(f'trades_{symbol}')
                    subscription_count += 1
                
                # Subscribe to candles for configured intervals
                for interval in self.config.intervals:
                    if subscription_count >= max_subscriptions:
                        break
                    
                    candle_success = await self.hyperliquid.subscribe_candle(
                        symbol, interval, self._handle_enhanced_candle_update
                    )
                    if candle_success:
                        self.active_subscriptions.add(f'candle_{symbol}_{interval}')
                        subscription_count += 1
                        logger.debug(f"Subscribed to {interval} candles for {symbol}")
                
                # Subscribe to order book for key symbols only
                if symbol in self.config.orderbook_symbols and subscription_count < max_subscriptions:
                    ob_success = await self.hyperliquid.subscribe_orderbook(
                        symbol, self._handle_enhanced_orderbook_update
                    )
                    if ob_success:
                        self.active_subscriptions.add(f'orderbook_{symbol}')
                        subscription_count += 1
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.1)
            
            logger.info(f"Successfully subscribed to {len(self.active_subscriptions)} enhanced feeds")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to enhanced feeds: {e}")
            self.stats.errors += 1
    
    async def _handle_enhanced_price_update(self, message: Dict[str, Any]) -> None:
        """Handle enhanced mid price updates with validation."""
        try:
            data = message.get('data', {})
            
            if isinstance(data, dict):
                timestamp = datetime.now(timezone.utc)
                processed_count = 0
                
                # Process each symbol's price with validation
                for symbol, price_str in data.items():
                    if symbol in self.config.symbols:
                        try:
                            # Note: Price updates are typically stored as part of candle generation
                            # For real-time price tracking, we could store in a separate prices table
                            # or update the most recent candle data
                            
                            processed_count += 1
                            
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid price data for {symbol}: {price_str}, error: {e}")
                            self.stats.validation_errors += 1
                            continue
                
                self.stats.last_activity = timestamp
                
        except Exception as e:
            logger.error(f"Error handling enhanced price update: {e}")
            self.stats.errors += 1
    
    async def _handle_enhanced_trade_update(self, message: Dict[str, Any]) -> None:
        """Handle enhanced trade updates with database models."""
        try:
            logger.debug(f"💹 Received trade message: {json.dumps(message, indent=2)}")
            
            data = message.get('data', [])
            
            if isinstance(data, list):
                timestamp = datetime.now(timezone.utc)
                
                for trade_data in data:
                    try:
                        logger.debug(f"🔄 Processing trade data: {trade_data}")
                        
                        symbol = trade_data.get('coin', '')
                        
                        logger.debug(f"🎯 Trade symbol: {symbol}")
                        
                        # Only process trades for symbols we're interested in
                        if symbol not in self.config.symbols:
                            logger.info(f"⏭️ Skipping trade for {symbol} - not in config")
                            continue
                            
                        # Create enhanced trade model with RAW HyperLiquid fields
                        raw_side = trade_data.get('side', '')
                        # Convert HyperLiquid notation to database side notation
                        if raw_side == 'B':
                            side = 'B'  # Keep HyperLiquid notation for raw storage
                        elif raw_side == 'A':
                            side = 'A'  # Keep HyperLiquid notation for raw storage
                        else:
                            side = raw_side
                        
                        # Extract raw HyperLiquid fields for preservation
                        trade_id = trade_data.get('tid', trade_data.get('time', int(timestamp.timestamp() * 1000)))
                        trade_hash = trade_data.get('hash', None)
                        user1 = trade_data.get('user1', None)  
                        user2 = trade_data.get('user2', None)
                        
                        logger.debug(f"💰 Trade - Price: {trade_data.get('px')}, Size: {trade_data.get('sz')}, Side: {side}")
                        
                        db_trade = DBTradeData(
                            symbol=symbol,
                            timestamp=timestamp,
                            price=str(trade_data.get('px', '0')),
                            size=str(trade_data.get('sz', '0')),
                            side=side,
                            trade_id=trade_id,
                            hash=trade_hash,
                            user1=user1,
                            user2=user2
                        )
                        
                        logger.debug(f"✅ Created trade model for {symbol}")
                        
                        # Validate if enabled
                        if self.config.enable_validation:
                            if not self._validate_trade_data(db_trade):
                                logger.warning(f"❌ Trade validation failed for {symbol}")
                                continue
                        
                        self.trade_buffer.append(db_trade)
                        self.stats.trades_collected += 1
                        
                        logger.debug(f"📈 Added trade to buffer. Total in buffer: {len(self.trade_buffer)}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process trade data: {e}")
                        self.stats.validation_errors += 1
                        continue
                
                self.stats.last_activity = timestamp
                
                # Flush if buffer is full
                if len(self.trade_buffer) >= self.config.buffer_size:
                    logger.info(f"🚰 Flushing trade buffer - {len(self.trade_buffer)} trades")
                    await self._flush_trade_buffer()
                    
        except Exception as e:
            logger.error(f"Error handling enhanced trade update: {e}")
            self.stats.errors += 1
    
    async def _handle_enhanced_orderbook_update(self, message: Dict[str, Any]) -> None:
        """Handle enhanced order book updates with compression."""
        try:
            data = message.get('data', {})
            
            if isinstance(data, dict):
                timestamp = datetime.now(timezone.utc)
                symbol = data.get('coin', '')
                
                if symbol in self.config.orderbook_symbols:
                    # Extract raw bids/asks from HyperLiquid format
                    levels = data.get('levels', [])
                    raw_bids = levels[0] if len(levels) > 0 else []
                    raw_asks = levels[1] if len(levels) > 1 else []
                    
                    # Create enhanced orderbook model with RAW format
                    db_orderbook = DBOrderBookSnapshot(
                        symbol=symbol,
                        timestamp=timestamp,
                        time_ms=int(timestamp.timestamp() * 1000),
                        bids=json.dumps(raw_bids),
                        asks=json.dumps(raw_asks)
                    )
                    
                    # Validate if enabled
                    if self.config.enable_validation:
                        if not self._validate_orderbook_data(db_orderbook):
                            return
                    
                    self.orderbook_buffer.append(db_orderbook)
                    self.stats.orderbooks_collected += 1
                    self.stats.last_activity = timestamp
                    
                    # Flush if buffer is full
                    if len(self.orderbook_buffer) >= self.config.buffer_size:
                        await self._flush_orderbook_buffer()
                        
        except Exception as e:
            logger.error(f"Error handling enhanced orderbook update: {e}")
            self.stats.errors += 1
    
    async def _handle_enhanced_candle_update(self, message: Dict[str, Any]) -> None:
        """Handle enhanced candle/kline updates from WebSocket."""
        try:
            logger.debug(f"🕯️ Received candle message: {json.dumps(message, indent=2)}")
            
            data = message.get('data', {})
            
            # Data is a single candle object, not a list
            if isinstance(data, dict) and data:
                timestamp = datetime.now(timezone.utc)
                
                try:
                    logger.debug(f"📊 Processing candle data: {data}")
                    
                    # Extract raw HyperLiquid candle fields based on API documentation
                    symbol = data.get('s', '')  # coin symbol
                    interval = data.get('i', '')  # interval
                    
                    logger.debug(f"🎯 Symbol: {symbol}, Interval: {interval}")
                    
                    # Only process candles for symbols and intervals we're interested in
                    if symbol not in self.config.symbols or interval not in self.config.intervals:
                        logger.info(f"⏭️ Skipping {symbol} {interval} - not in config")
                        return
                    
                    # Extract candle fields according to HyperLiquid format:
                    # t: open millis, T: close millis, o: open, c: close, h: high, l: low, v: volume, n: trade count
                    open_time = data.get('t', 0)  # open millis
                    close_time = data.get('T', 0)  # close millis
                    open_price = str(data.get('o', '0'))  # open price
                    high_price = str(data.get('h', '0'))  # high price
                    low_price = str(data.get('l', '0'))   # low price
                    close_price = str(data.get('c', '0')) # close price
                    volume = str(data.get('v', '0'))      # volume (base unit)
                    trade_count = data.get('n', 0)        # number of trades
                    
                    logger.debug(f"💰 Prices - O:{open_price} H:{high_price} L:{low_price} C:{close_price} V:{volume}")
                    
                    # Convert timestamp from milliseconds to datetime
                    candle_timestamp = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
                    
                    # Create enhanced candle model with correct field mapping
                    db_candle = DBCandlestickData(
                        symbol=symbol,
                        timestamp_start=candle_timestamp,
                        timestamp_end=datetime.fromtimestamp(close_time / 1000, tz=timezone.utc),
                        open_price=open_price,
                        high_price=high_price,
                        low_price=low_price,
                        close_price=close_price,
                        volume=volume,
                        trade_count=trade_count
                    )
                    
                    logger.debug(f"✅ Created candle model for {symbol} {interval}")
                    
                    # Validate if enabled
                    if self.config.enable_validation:
                        if not self._validate_candle_data(db_candle):
                            logger.warning(f"❌ Validation failed for {symbol} {interval}")
                            return
                    
                    # Store the interval with the candle for later table routing
                    # We'll use a tuple: (candle, interval)
                    self.candle_buffer.append((db_candle, interval))
                    self.stats.candles_collected += 1
                    
                    logger.debug(f"📈 Added candle to buffer. Total in buffer: {len(self.candle_buffer)}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process candle data: {e}")
                    self.stats.validation_errors += 1
                
                self.stats.last_activity = timestamp
                
                # Flush if buffer is full
                if len(self.candle_buffer) >= self.config.buffer_size:
                    logger.info(f"🚰 Flushing candle buffer - {len(self.candle_buffer)} candles")
                    await self._flush_candle_buffer()
                    
        except Exception as e:
            logger.error(f"Error handling enhanced candle update: {e}")
            self.stats.errors += 1
    
    def _validate_trade_data(self, trade: DBTradeData) -> bool:
        """Validate trade data using the data integrity validator."""
        try:
            # Basic validation checks
            if not trade.symbol or not trade.price or not trade.size:
                return False
                
            price = Decimal(trade.price)
            size = Decimal(trade.size)
            
            if price <= 0 or size <= 0:
                return False
                
            # Handle HyperLiquid side notation (B/A) and standard notation (buy/sell)
            valid_sides = ['buy', 'sell', 'B', 'A']
            if trade.side not in valid_sides:
                return False
                
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _validate_orderbook_data(self, orderbook: DBOrderBookSnapshot) -> bool:
        """Validate order book data."""
        try:
            return bool(
                orderbook.symbol and 
                orderbook.timestamp and 
                orderbook.bids and 
                orderbook.asks
            )
        except Exception:
            return False

    def _validate_candle_data(self, candle: DBCandlestickData) -> bool:
        """Validate candle data using basic checks."""
        try:
            # Basic validation checks
            if not candle.symbol:
                return False
                
            # Validate price data
            try:
                open_price = Decimal(candle.open_price)
                high_price = Decimal(candle.high_price)
                low_price = Decimal(candle.low_price)
                close_price = Decimal(candle.close_price)
                volume = Decimal(candle.volume)
                
                # Prices must be positive
                if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                    return False
                
                # Volume must be non-negative
                if volume < 0:
                    return False
                
                # High must be >= all other prices, low must be <= all other prices
                if high_price < max(open_price, close_price, low_price):
                    return False
                if low_price > min(open_price, close_price, high_price):
                    return False
                
            except (ValueError, TypeError, InvalidOperation):
                return False
            
            # Validate timestamps
            if not candle.timestamp_start or not candle.timestamp_end:
                return False
            
            # End time should be after start time
            if candle.timestamp_end <= candle.timestamp_start:
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Candle validation error: {e}")
            return False

    async def _flush_trade_buffer(self) -> None:
        """Flush trade data to database using enhanced models."""
        if not self.trade_buffer:
            return
        
        try:
            batch_id = f"trades_{int(datetime.now(timezone.utc).timestamp())}"
            batch_op = DBBatchOperation(
                batch_id=batch_id,
                operation_type="INSERT",
                table_name="trades",
                record_count=len(self.trade_buffer),
                start_time=datetime.now(timezone.utc)
            )
            
            # Get the next ID for insertion
            result = self.db_manager.execute("SELECT COALESCE(max(id), 0) + 1 as next_id FROM trades")
            start_id = result[0][0] if result and result[0] else 1

            # Serialize and insert using RAW SCHEMA (preserve HyperLiquid format)
            insert_query = """
                INSERT INTO trades (id, symbol, timestamp, trade_id, price, size, side, hash, user1, user2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (trade_id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                timestamp = EXCLUDED.timestamp,
                price = EXCLUDED.price,
                size = EXCLUDED.size,
                side = EXCLUDED.side,
                hash = EXCLUDED.hash,
                user1 = EXCLUDED.user1,
                user2 = EXCLUDED.user2
            """
            
            params = [
                (
                    start_id + i,  # Explicit ID assignment
                    trade.symbol,
                    trade.timestamp,
                    trade.trade_id,
                    trade.price,
                    trade.size,
                    trade.side,
                    trade.hash,
                    trade.user1,
                    trade.user2
                )
                for i, trade in enumerate(self.trade_buffer)
            ]
            
            self.db_manager.execute_many(insert_query, params)
            
            # Complete batch operation
            batch_op.mark_completed(success=True)
            self.batch_operations[batch_id] = batch_op
            self.stats.batches_processed += 1
            
            logger.debug(f"Enhanced flush: {len(self.trade_buffer)} trade records to database")
            self.trade_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush enhanced trade buffer: {e}")
            self.stats.errors += 1
    
    async def _flush_orderbook_buffer(self) -> None:
        """Flush order book data to database using enhanced models."""
        if not self.orderbook_buffer:
            return
        
        try:
            batch_id = f"orderbooks_{int(datetime.now(timezone.utc).timestamp())}"
            batch_op = DBBatchOperation(
                batch_id=batch_id,
                operation_type="INSERT",
                table_name="orderbook_snapshots",
                record_count=len(self.orderbook_buffer),
                start_time=datetime.now(timezone.utc)
            )
            
            # Get the next ID for insertion
            result = self.db_manager.execute("SELECT COALESCE(max(id), 0) + 1 as next_id FROM orderbook_snapshots")
            start_id = result[0][0] if result and result[0] else 1

            # Serialize and insert using RAW SCHEMA (preserve HyperLiquid format)
            insert_query = """
                INSERT INTO orderbook_snapshots (id, symbol, timestamp, time_ms, bids, asks)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, time_ms) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                bids = EXCLUDED.bids,
                asks = EXCLUDED.asks
            """
            
            params = [
                (
                    start_id + i,  # Explicit ID assignment
                    ob.symbol,
                    ob.timestamp,
                    ob.time_ms,
                    ob.bids,
                    ob.asks
                )
                for i, ob in enumerate(self.orderbook_buffer)
            ]
            
            self.db_manager.execute_many(insert_query, params)
            
            # Complete batch operation
            batch_op.mark_completed(success=True)
            self.batch_operations[batch_id] = batch_op
            self.stats.batches_processed += 1
            
            stored_count = len(self.orderbook_buffer)
            logger.info(f"✅ Flushed {stored_count} orderbook snapshots to orderbook_snapshots")
            self.orderbook_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush enhanced orderbook buffer: {e}")
            self.stats.errors += 1
    
    async def _flush_candle_buffer(self) -> None:
        """Flush candlestick data to database using enhanced models."""
        if not self.candle_buffer:
            return
        
        try:
            # Group candles by timeframe for separate table insertion
            candles_by_timeframe = {}
            for candle, interval in self.candle_buffer:
                if interval not in candles_by_timeframe:
                    candles_by_timeframe[interval] = []
                candles_by_timeframe[interval].append((candle, interval))
            
            # Insert into appropriate timeframe tables
            for interval, candles in candles_by_timeframe.items():
                table_name = f"candles_{interval}"
                
                    # Use the new schema (matches DBCandlestickData and MarketDataSchema)
                # Get the next ID for insertion to avoid sequence issues
                result = self.db_manager.execute(f"SELECT COALESCE(max(id), 0) + 1 as next_id FROM {table_name}")
                start_id = result[0][0] if result and result[0] else 1
                
                query = f"""
                    INSERT OR IGNORE INTO {table_name} 
                    (id, symbol, interval, open_time_ms, close_time_ms, timestamp_start, timestamp_end, open_price, high_price, low_price, close_price, volume, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Prepare bulk insert data
                insert_data = []
                for i, (candle_model, candle_interval) in enumerate(candles):
                    try:
                        # Validate the model before insertion
                        if not self._validate_candle_data(candle_model):
                            logger.warning(f"Skipping invalid candle: {candle_model}")
                            continue
                        
                        # Calculate millisecond timestamps from datetime fields
                        open_time_ms = int(candle_model.timestamp_start.timestamp() * 1000)
                        close_time_ms = int(candle_model.timestamp_end.timestamp() * 1000)
                        
                        insert_data.append((
                            start_id + i,  # Explicit ID assignment
                            candle_model.symbol,
                            candle_interval,  # Add the interval field!
                            open_time_ms,  # Add the open_time_ms field!
                            close_time_ms,  # Add the close_time_ms field!
                            candle_model.timestamp_start,
                            candle_model.timestamp_end,
                            candle_model.open_price,
                            candle_model.high_price,
                            candle_model.low_price,
                            candle_model.close_price,
                            candle_model.volume,
                            candle_model.trade_count or 0
                        ))
                    except Exception as e:
                        logger.error(f"Error preparing candle data: {e}")
                        continue
                
                if insert_data:
                    # Execute the insert
                    self.db_manager.execute_many(query, insert_data)
                    stored_count = len(insert_data)
                    
                    self.stats.candles_collected += stored_count
                    logger.info(f"✅ Flushed {stored_count} candles to {table_name}")
            
            logger.debug(f"Enhanced flush: {len(self.candle_buffer)} candle records to database")
            self.candle_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush enhanced candle buffer: {e}")
            self.stats.errors += 1
    
    async def _flush_funding_rate_buffer(self) -> None:
        """Flush funding rate data to database using enhanced models."""
        if not self.funding_rate_buffer:
            return
        
        try:
            batch_id = f"funding_rates_{int(datetime.now(timezone.utc).timestamp())}"
            batch_op = DBBatchOperation(
                batch_id=batch_id,
                operation_type="INSERT",
                table_name="funding_rates",
                record_count=len(self.funding_rate_buffer),
                start_time=datetime.now(timezone.utc)
            )
            
            insert_query = """
                INSERT INTO funding_rates (symbol, timestamp, time_ms, funding_rate, funding_rate_decimal, premium, premium_decimal)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, time_ms) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                funding_rate = EXCLUDED.funding_rate,
                funding_rate_decimal = EXCLUDED.funding_rate_decimal,
                premium = EXCLUDED.premium,
                premium_decimal = EXCLUDED.premium_decimal
            """
            
            params = [
                (
                    fr.symbol,
                    fr.timestamp,
                    fr.time_ms,
                    fr.funding_rate,
                    float(fr.funding_rate) if fr.funding_rate else None,  # Convert to decimal
                    fr.premium,
                    float(fr.premium) if fr.premium else None  # Convert to decimal
                )
                for fr in self.funding_rate_buffer
            ]
            
            self.db_manager.execute_many(insert_query, params)
            
            # Complete batch operation
            batch_op.mark_completed(success=True)
            self.batch_operations[batch_id] = batch_op
            self.stats.batches_processed += 1
            
            self.stats.funding_rates_collected += len(self.funding_rate_buffer)
            logger.debug(f"Enhanced flush: {len(self.funding_rate_buffer)} funding rate records to database")
            self.funding_rate_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush enhanced funding rate buffer: {e}")
            self.stats.errors += 1
    
    async def _flush_all_buffers(self) -> None:
        """Flush all data buffers to database."""
        await asyncio.gather(
            self._flush_trade_buffer(),
            self._flush_orderbook_buffer(),
            self._flush_candle_buffer(),
            self._flush_funding_rate_buffer(),
            return_exceptions=True
        )
    
    async def _complete_batch_operations(self) -> None:
        """Complete any ongoing batch operations."""
        try:
            for batch_id, batch_op in self.batch_operations.items():
                if batch_op.status == "running":
                    batch_op.mark_completed(success=True)
                    
            # Store batch operation records in database
            if self.batch_operations:
                await self._store_batch_operations()
                
        except Exception as e:
            logger.error(f"Error completing batch operations: {e}")
    
    async def _store_batch_operations(self) -> None:
        """Store batch operation records in database."""
        try:
            # Get the next ID for insertion
            result = self.db_manager.execute("SELECT COALESCE(max(id), 0) + 1 as next_id FROM batch_operations")
            start_id = result[0][0] if result and result[0] else 1

            insert_query = """
                INSERT INTO batch_operations 
                (id, batch_id, operation_type, table_name, record_count, start_time, end_time, status, error_message, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (batch_id) DO UPDATE SET
                operation_type = EXCLUDED.operation_type,
                table_name = EXCLUDED.table_name,
                record_count = EXCLUDED.record_count,
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                metrics_json = EXCLUDED.metrics_json
            """
            
            params = [
                (
                    start_id + i,  # Explicit ID assignment
                    batch_op.batch_id,
                    batch_op.operation_type,
                    batch_op.table_name,
                    batch_op.record_count,
                    batch_op.start_time,
                    batch_op.end_time,
                    batch_op.status,
                    batch_op.error_message,
                    batch_op.metrics_json
                )
                for i, batch_op in enumerate(self.batch_operations.values())
            ]
            
            self.db_manager.execute_many(insert_query, params)
            self.batch_operations.clear()
            
        except Exception as e:
            logger.error(f"Failed to store batch operations: {e}")
    
    async def _health_monitor(self) -> None:
        """Monitor collector health and performance."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check for stale data
                if self.stats.last_activity:
                    time_since_activity = datetime.now(timezone.utc) - self.stats.last_activity
                    if time_since_activity.total_seconds() > 600:  # 10 minutes
                        logger.warning(f"No activity for {time_since_activity.total_seconds()} seconds")
                
                # Check buffer sizes
                total_buffer_size = (
                    len(self.trade_buffer) + 
                    len(self.orderbook_buffer) + 
                    len(self.candle_buffer) +
                    len(self.funding_rate_buffer)
                )
                
                if total_buffer_size > self.config.max_batch_size:
                    logger.warning(f"Large buffer size detected: {total_buffer_size} records")
                    await self._flush_all_buffers()
                
                # Check error rates
                if self.stats.errors > 100:
                    logger.error(f"High error count detected: {self.stats.errors}")
                
                # Log health status
                logger.info(f"Health check: {len(self.active_subscriptions)} subscriptions, "
                          f"{total_buffer_size} buffered records, "
                          f"{self.stats.errors} errors")
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
    
    async def collect_enhanced_historical_data(
        self, 
        symbol: str, 
        days_back: int = 7,
        intervals: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Collect historical candlestick data using enhanced models.
        
        Args:
            symbol: Trading symbol
            days_back: Number of days to go back
            intervals: List of intervals to collect
            
        Returns:
            Dictionary mapping interval to number of candles collected
        """
        try:
            intervals = intervals or ['1m', '5m', '15m', '1h', '4h', '1d']
            results = {}
            
            logger.info(f"Starting enhanced historical collection for {symbol}: {days_back} days")
            
            for interval in intervals:
                try:
                    # Use existing historical collection method from HyperLiquid client
                    candles = await self.hyperliquid.get_historical_candles_bulk(
                        symbol=symbol,
                        interval=interval,
                        days_back=days_back
                    )
                    
                    # Convert to enhanced database models
                    db_candles = []
                    for candle_data in candles:
                        try:
                            # Convert to business model first
                            business_candle = CandlestickData(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(candle_data['t'] / 1000, tz=timezone.utc),
                                timeframe=Timeframe(interval),
                                open=Decimal(str(candle_data['o'])),
                                high=Decimal(str(candle_data['h'])),
                                low=Decimal(str(candle_data['l'])),
                                close=Decimal(str(candle_data['c'])),
                                volume=Decimal(str(candle_data['v']))
                            )
                            
                            # Convert to database model
                            db_candle = self.converter.candlestick_to_db(business_candle)
                            
                            # Validate if enabled
                            if self.config.enable_validation:
                                if self.validator.validate_candlestick_data(db_candle):
                                    db_candles.append(db_candle)
                                else:
                                    self.stats.validation_errors += 1
                            else:
                                db_candles.append(db_candle)
                                
                        except Exception as e:
                            logger.warning(f"Failed to process candle data for {symbol} {interval}: {e}")
                            self.stats.validation_errors += 1
                            continue
                    
                    # Store enhanced candles
                    if db_candles:
                        stored_count = await self._store_enhanced_historical_candles(
                            db_candles, symbol, interval
                        )
                        results[interval] = stored_count
                        self.stats.candles_collected += stored_count
                        
                        logger.info(f"Enhanced collection: {stored_count} {interval} candles for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to collect {interval} data for {symbol}: {e}")
                    self.stats.errors += 1
                    results[interval] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced historical collection failed for {symbol}: {e}")
            self.stats.errors += 1
            return {}
    
    async def _store_enhanced_historical_candles(
        self, 
        candles: List[DBCandlestickData], 
        symbol: str, 
        interval: str
    ) -> int:
        """Store enhanced historical candles in database."""
        try:
            if not candles:
                return 0
            
            table_name = f"candles_{interval}"
            
            # Create batch operation record
            batch_id = f"historical_{symbol}_{interval}_{int(datetime.now(timezone.utc).timestamp())}"
            batch_op = DBBatchOperation(
                batch_id=batch_id,
                operation_type="INSERT",
                table_name=table_name,
                record_count=len(candles),
                start_time=datetime.now(timezone.utc)
            )
            
            # Use the new schema (matches DBCandlestickData and MarketDataSchema)
            # Get the next ID for insertion to avoid sequence issues  
            result = self.db_manager.execute(f"SELECT COALESCE(max(id), 0) + 1 as next_id FROM {table_name}")
            start_id = result[0][0] if result and result[0] else 1
            
            query = f"""
                INSERT OR IGNORE INTO {table_name} 
                (id, symbol, interval, open_time_ms, close_time_ms, timestamp_start, timestamp_end, open_price, high_price, low_price, close_price, volume, trade_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Prepare bulk insert data
            insert_data = []
            for i, candle in enumerate(candles):
                # Calculate millisecond timestamps from datetime fields
                open_time_ms = int(candle.timestamp_start.timestamp() * 1000)
                close_time_ms = int(candle.timestamp_end.timestamp() * 1000)
                
                insert_data.append((
                    start_id + i,  # Explicit ID assignment
                    candle.symbol,
                    interval,  # Add the interval field!
                    open_time_ms,  # Add the open_time_ms field!
                    close_time_ms,  # Add the close_time_ms field!
                    candle.timestamp_start,
                    candle.timestamp_end,
                    candle.open_price,
                    candle.high_price,
                    candle.low_price,
                    candle.close_price,
                    candle.volume,
                    candle.trade_count
                ))
            
            # Execute the insert
            self.db_manager.execute_many(query, insert_data)
            stored_count = len(insert_data)
            
            # Complete batch operation
            batch_op.mark_completed(success=True)
            self.batch_operations[batch_id] = batch_op
            self.stats.batches_processed += 1
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store enhanced historical candles: {e}")
            self.stats.errors += 1
            return 0
    
    async def collect_funding_rates(self, symbols: Optional[List[str]] = None) -> int:
        """Collect current funding rates for symbols."""
        try:
            target_symbols = symbols or list(self.config.symbols)[:10]  # Limit to avoid rate limits
            collected_count = 0
            
            for symbol in target_symbols:
                try:
                    # Only collect for symbols we're configured to track
                    if symbol not in self.config.symbols:
                        continue
                        
                    # Get funding rate from HyperLiquid
                    funding_data = await self.hyperliquid.get_funding_rate(symbol)
                    
                    if funding_data:
                        # Create enhanced funding rate model
                        timestamp = datetime.now(timezone.utc)
                        
                        db_funding_rate = DBFundingRateData(
                            symbol=symbol,
                            timestamp=timestamp,
                            time_ms=int(timestamp.timestamp() * 1000),
                            funding_rate=str(funding_data.get('fundingRate', '0')),
                            premium=str(funding_data.get('premium', '0')) if funding_data.get('premium') else None
                        )
                        
                        self.funding_rate_buffer.append(db_funding_rate)
                        collected_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to collect funding rate for {symbol}: {e}")
                    self.stats.errors += 1
                    continue
            
            # Flush if we have data
            if self.funding_rate_buffer:
                await self._flush_funding_rate_buffer()
            
            logger.info(f"Collected {collected_count} funding rates")
            return collected_count
            
        except Exception as e:
            logger.error(f"Failed to collect funding rates: {e}")
            self.stats.errors += 1
            return 0

    async def _periodic_flush(self) -> None:
        """Periodically flush buffers to database."""
        while self.running:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush_all_buffers()
                
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                self.stats.errors += 1
    
    async def _periodic_stats(self) -> None:
        """Periodically log collection statistics."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Log stats every minute
                
                logger.info(
                    f"DataCollector stats: "
                    f"trades={self.stats.trades_collected}, "
                    f"orderbooks={self.stats.orderbooks_collected}, "
                    f"errors={self.stats.errors}, "
                    f"subscriptions={len(self.active_subscriptions)}"
                )
                
            except Exception as e:
                logger.error(f"Error logging stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            **self.stats.to_dict(),
            'active_subscriptions': len(self.active_subscriptions),
            'subscription_types': list(self.active_subscriptions),
            'symbols_count': len(self.config.symbols),
            'buffer_sizes': {
                'trades': len(self.trade_buffer),
                'orderbooks': len(self.orderbook_buffer)
            },
            'running': self.running
        }
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the collection list."""
        if symbol not in self.config.symbols:
            self.config.symbols.add(symbol)
            logger.info(f"Added symbol {symbol} to collection list")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the collection list."""
        if symbol in self.config.symbols:
            self.config.symbols.remove(symbol)
            logger.info(f"Removed symbol {symbol} from collection list")
    
    async def collect_historical_data(
        self, 
        symbol: str, 
        days_back: int = 7,
        interval: str = "1m"
    ) -> int:
        """
        Collect historical candlestick data for a symbol and store in database.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            days_back: Number of days to go back
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            
        Returns:
            Number of candles collected and stored
        """
        try:
            logger.info(f"Starting historical data collection for {symbol}: {days_back} days of {interval} candles")
            
            def progress_callback(current, total):
                percentage = (current / total) * 100 if total > 0 else 0
                logger.info(f"Historical collection progress for {symbol}: {current}/{total} ({percentage:.1f}%)")
            
            # Use the enhanced bulk collection method
            candles = await self.hyperliquid.get_historical_candles_bulk(
                symbol=symbol,
                interval=interval,
                days_back=days_back,
                progress_callback=progress_callback
            )
            
            if not candles:
                logger.warning(f"No historical candles received for {symbol}")
                return 0
            
            logger.info(f"Received {len(candles)} historical candles for {symbol}")
            
            # Store in database
            stored_count = await self._store_historical_candles(candles, symbol, interval)
            
            logger.info(f"Successfully stored {stored_count} historical candles for {symbol}")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to collect historical data for {symbol}: {e}")
            return 0

    async def collect_historical_data_bulk(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 7,
        intervals: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Collect historical data for multiple symbols and intervals efficiently.
        
        Args:
            symbols: List of symbols to collect (defaults to self.symbols)
            days_back: Number of days to go back
            intervals: List of intervals to collect (defaults to ['1h'])
            
        Returns:
            Dictionary mapping symbol -> interval -> count of candles collected
        """
        try:
            target_symbols = symbols or list(self.config.symbols)
            target_intervals = intervals or ['1h']
            
            if not target_symbols:
                logger.warning("No symbols specified for bulk historical collection")
                return {}
            
            logger.info(f"Starting bulk historical collection:")
            logger.info(f"  Symbols: {len(target_symbols)} ({target_symbols[:5]}...)")
            logger.info(f"  Intervals: {target_intervals}")
            logger.info(f"  Days back: {days_back}")
            
            results = {}
            total_operations = len(target_symbols) * len(target_intervals)
            completed_operations = 0
            
            for symbol in target_symbols:
                symbol_results = {}
                
                for interval in target_intervals:
                    completed_operations += 1
                    
                    logger.info(f"Collecting {symbol} {interval} data ({completed_operations}/{total_operations})")
                    
                    count = await self.collect_historical_data(
                        symbol=symbol,
                        days_back=days_back,
                        interval=interval
                    )
                    
                    symbol_results[interval] = count
                    
                    # Small delay between requests to respect rate limits
                    await asyncio.sleep(0.5)
                
                results[symbol] = symbol_results
            
            total_candles = sum(
                sum(interval_counts.values()) 
                for interval_counts in results.values()
            )
            
            logger.info(f"Bulk historical collection complete: {total_candles} total candles")
            return results
            
        except Exception as e:
            logger.error(f"Failed bulk historical collection: {e}")
            return {}

    async def _store_historical_candles(
        self, 
        candles: List[Dict[str, Any]], 
        symbol: str, 
        interval: str
    ) -> int:
        """
        Store historical candles in the appropriate database table.
        
        Args:
            candles: List of candle data from HyperLiquid
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            Number of candles successfully stored
        """
        try:
            if not candles:
                return 0
            
            # Determine the correct table based on interval
            table_map = {
                '1m': 'candles_1m',
                '5m': 'candles_5m',
                '15m': 'candles_15m',
                '1h': 'candles_1h',
                '4h': 'candles_4h',
                '1d': 'candles_1d'
            }
            
            table_name = table_map.get(interval, 'candles_1m')
            
            # Convert HyperLiquid format to database format using NEW SCHEMA
            db_candles = []
            for candle in candles:
                try:
                    # HyperLiquid candle format: {'t': timestamp_ms, 's': symbol, 'o': open, 'c': close, 'h': high, 'l': low, 'v': volume, 'n': count}
                    timestamp_ms = int(candle.get('t', 0))
                    
                    if timestamp_ms == 0:
                        continue
                    
                    # Convert to datetime
                    timestamp_start = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    
                    # Calculate end timestamp based on timeframe
                    timeframe_minutes = {
                        '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
                    }
                    minutes = timeframe_minutes.get(interval, 1)
                    timestamp_end = timestamp_start + timedelta(minutes=minutes)
                    
                    db_candle = {
                        'symbol': symbol,
                        'timestamp_start': timestamp_start,
                        'timestamp_end': timestamp_end,
                        'open_price': str(candle.get('o', 0)),
                        'high_price': str(candle.get('h', 0)),
                        'low_price': str(candle.get('l', 0)),
                        'close_price': str(candle.get('c', 0)),
                        'volume': str(candle.get('v', 0)),
                        'trade_count': int(candle.get('n', 0))
                    }
                    
                    db_candles.append(db_candle)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid candle data: {e}")
                    continue
            
            if not db_candles:
                logger.warning(f"No valid candles to store for {symbol}")
                return 0
            
            # Use database-specific insertion with NEW SCHEMA
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert using the new schema (matches DBCandlestickData and MarketDataSchema)
                # Get the next ID for insertion to avoid sequence issues
                result = self.db_manager.execute(f"SELECT COALESCE(max(id), 0) + 1 as next_id FROM {table_name}")
                start_id = result[0][0] if result and result[0] else 1
                
                query = f"""
                    INSERT OR IGNORE INTO {table_name} 
                    (id, symbol, interval, open_time_ms, close_time_ms, timestamp_start, timestamp_end, open_price, high_price, low_price, close_price, volume, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Prepare insert data to match the new schema
                insert_data = []
                for i, candle in enumerate(db_candles):
                    # Calculate millisecond timestamps from datetime fields
                    open_time_ms = int(candle['timestamp_start'].timestamp() * 1000)
                    close_time_ms = int(candle['timestamp_end'].timestamp() * 1000)
                    
                    insert_data.append((
                        start_id + i,  # Explicit ID assignment
                        candle['symbol'],
                        interval,  # Add the interval field!
                        open_time_ms,  # Add the open_time_ms field!
                        close_time_ms,  # Add the close_time_ms field!
                        candle['timestamp_start'],
                        candle['timestamp_end'],
                        candle['open_price'],
                        candle['high_price'],
                        candle['low_price'],
                        candle['close_price'],
                        candle['volume'],
                        candle.get('trade_count', 0)  # Default to 0 if not provided
                    ))
                
                # Execute the insert
                self.db_manager.execute_many(query, insert_data)
                stored_count = len(insert_data)
                
                self.stats.historical_requests += 1
                self.stats.total_candles_stored += stored_count
                
                logger.info(f"Stored {stored_count} historical candles for {symbol} ({interval}) in {table_name}")
                return stored_count
                
        except Exception as e:
            logger.error(f"Database error storing candles: {e}")
            return 0

    async def backfill_missing_data(
        self,
        symbol: str,
        interval: str = "1h",
        max_days_back: int = 30
    ) -> int:
        """
        Identify and backfill missing data gaps for a symbol.
        
        Args:
            symbol: Trading symbol
            interval: Time interval 
            max_days_back: Maximum days to look back for gaps
            
        Returns:
            Number of candles backfilled
        """
        try:
            logger.info(f"Checking for missing data gaps in {symbol} {interval}")
            
            # Get the table name
            table_map = {
                '1m': 'candles_1m',
                '5m': 'candles_5m',
                '15m': 'candles_15m', 
                '1h': 'candles_1h',
                '4h': 'candles_4h',
                '1d': 'candles_1d'
            }
            
            table_name = table_map.get(interval, 'candles_1h')
            
            # Query for existing data to find gaps
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get earliest and latest timestamps
                cursor.execute(f"""
                    SELECT EXTRACT(EPOCH FROM MIN(timestamp_start)) * 1000 as earliest, 
                           EXTRACT(EPOCH FROM MAX(timestamp_start)) * 1000 as latest, 
                           COUNT(*) as count
                    FROM {table_name}
                    WHERE symbol = ?
                """, (symbol,))
                
                result = cursor.fetchone()
                
                if not result or result[0] is None:
                    logger.info(f"No existing data for {symbol}, collecting initial historical data")
                    return await self.collect_historical_data(symbol, max_days_back, interval)
                
                earliest_ms, latest_ms, existing_count = result
                
                # Calculate expected number of candles
                interval_minutes = self._interval_to_minutes(interval)
                time_span_minutes = (latest_ms - earliest_ms) / (1000 * 60)
                expected_count = int(time_span_minutes / interval_minutes)
                
                missing_percentage = ((expected_count - existing_count) / expected_count) * 100 if expected_count > 0 else 0
                
                logger.info(f"Data analysis for {symbol} {interval}:")
                logger.info(f"  Existing candles: {existing_count}")
                logger.info(f"  Expected candles: {expected_count}")
                logger.info(f"  Missing: {missing_percentage:.1f}%")
                
                if missing_percentage < 5:
                    logger.info(f"Data completeness acceptable for {symbol} ({missing_percentage:.1f}% missing)")
                    return 0
                
                # If significant gaps, do a targeted backfill
                logger.info(f"Backfilling missing data for {symbol} (target: last {max_days_back} days)")
                
                return await self.collect_historical_data(symbol, max_days_back, interval)
                
        except Exception as e:
            logger.error(f"Failed to backfill missing data for {symbol}: {e}")
            return 0
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        interval_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return interval_map.get(interval, 1) 


# Keep the original DataCollector class for backward compatibility
class DataCollector(EnhancedDataCollector):
    """
    Backward compatibility wrapper for the enhanced data collector.
    
    This maintains the original API while using the enhanced implementation.
    """
    
    def __init__(
        self, 
        hyperliquid: HyperLiquidIntegration, 
        db_manager: DatabaseManager,
        symbols: Optional[List[str]] = None
    ):
        config = CollectorConfig()
        if symbols:
            config.symbols = set(symbols)
        
        super().__init__(hyperliquid, db_manager, config)
        
        # Map old stats format for compatibility
        self._old_stats = {
            'prices_collected': 0,
            'trades_collected': 0,
            'orderbooks_collected': 0,
            'errors': 0,
            'last_activity': None
        }
    
    @property
    def symbols(self) -> Set[str]:
        """Get symbols for backward compatibility."""
        return self.config.symbols
    
    @symbols.setter
    def symbols(self, value: Union[Set[str], List[str]]) -> None:
        """Set symbols for backward compatibility."""
        if isinstance(value, list):
            self.config.symbols = set(value)
        else:
            self.config.symbols = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats in old format for compatibility."""
        enhanced_stats = super().get_stats()
        
        # Map to old format
        return {
            'prices_collected': 0,  # Not tracked separately in enhanced version
            'trades_collected': enhanced_stats.get('trades_collected', 0),
            'orderbooks_collected': enhanced_stats.get('orderbooks_collected', 0),
            'errors': enhanced_stats.get('errors', 0),
            'last_activity': enhanced_stats.get('last_activity'),
            'active_subscriptions': enhanced_stats.get('active_subscriptions', 0),
            'subscription_types': enhanced_stats.get('subscription_types', []),
            'symbols_count': enhanced_stats.get('symbols_count', 0),
            'buffer_sizes': enhanced_stats.get('buffer_sizes', {}),
            'running': enhanced_stats.get('running', False)
        } 