"""
HyperLiquid data collector for Bistoury.
Collects market data from HyperLiquid and stores it in the database.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any
from decimal import Decimal

from ..database import DatabaseManager
from ..logger import get_logger
from .client import HyperLiquidIntegration

logger = get_logger(__name__)


class DataCollector:
    """
    Collects real-time market data from HyperLiquid and stores it in the database.
    
    Handles:
    - Price updates (all mids)
    - Order book snapshots and updates
    - Trade executions
    - Periodic data collection tasks
    """
    
    def __init__(
        self, 
        hyperliquid: HyperLiquidIntegration, 
        db_manager: DatabaseManager,
        symbols: Optional[List[str]] = None
    ):
        self.hyperliquid = hyperliquid
        self.db_manager = db_manager
        self.symbols = set(symbols) if symbols else set()
        
        # Collection status
        self.running = False
        self.active_subscriptions: Set[str] = set()
        
        # Data buffers for batch insertion
        self.price_buffer: List[Dict[str, Any]] = []
        self.trade_buffer: List[Dict[str, Any]] = []
        self.orderbook_buffer: List[Dict[str, Any]] = []
        
        # Buffer limits and flush intervals
        self.buffer_size = 100
        self.flush_interval = 30.0  # seconds
        
        # Statistics
        self.stats = {
            'prices_collected': 0,
            'trades_collected': 0,
            'orderbooks_collected': 0,
            'errors': 0,
            'last_activity': None
        }
        
        logger.info(f"DataCollector initialized for {len(self.symbols)} symbols")
    
    async def start(self) -> bool:
        """
        Start the data collection process.
        
        Returns:
            bool: True if started successfully
        """
        if self.running:
            logger.warning("DataCollector is already running")
            return True
        
        try:
            # Ensure HyperLiquid is connected
            if not self.hyperliquid.is_connected():
                logger.info("Connecting to HyperLiquid...")
                if not await self.hyperliquid.connect():
                    logger.error("Failed to connect to HyperLiquid")
                    return False
            
            # Auto-discover symbols if none specified
            if not self.symbols:
                await self._discover_symbols()
            
            # Start data collection
            self.running = True
            
            # Start periodic tasks
            asyncio.create_task(self._periodic_flush())
            asyncio.create_task(self._periodic_stats())
            
            # Subscribe to data feeds
            await self._subscribe_to_feeds()
            
            logger.info("DataCollector started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start DataCollector: {e}")
            self.running = False
            return False
    
    async def stop(self) -> None:
        """Stop the data collection process."""
        logger.info("Stopping DataCollector...")
        
        self.running = False
        
        # Flush remaining data
        await self._flush_buffers()
        
        # Clear subscriptions
        self.active_subscriptions.clear()
        
        logger.info("DataCollector stopped")
    
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
            
            self.symbols = discovered_symbols
            logger.info(f"Discovered {len(self.symbols)} symbols: {list(self.symbols)[:10]}...")
            
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            # Use some default symbols if discovery fails
            self.symbols = {'BTC', 'ETH', 'SOL', 'AVAX', 'ARB'}
            logger.info(f"Using default symbols: {self.symbols}")
    
    async def _subscribe_to_feeds(self) -> None:
        """Subscribe to all required data feeds."""
        try:
            # Subscribe to all mid prices (global price updates)
            success = await self.hyperliquid.subscribe_all_mids(self._handle_price_update)
            if success:
                self.active_subscriptions.add('allMids')
                logger.info("Subscribed to all mid prices")
            
            # Subscribe to individual symbol feeds
            for symbol in list(self.symbols)[:10]:  # Limit to first 10 symbols to avoid overwhelming
                # Subscribe to trades
                trade_success = await self.hyperliquid.subscribe_trades(symbol, self._handle_trade_update)
                if trade_success:
                    self.active_subscriptions.add(f'trades_{symbol}')
                
                # Subscribe to order book (limited to key symbols)
                if symbol in {'BTC', 'ETH', 'SOL'}:
                    ob_success = await self.hyperliquid.subscribe_orderbook(symbol, self._handle_orderbook_update)
                    if ob_success:
                        self.active_subscriptions.add(f'orderbook_{symbol}')
                
                # Small delay to avoid overwhelming the connection
                await asyncio.sleep(0.1)
            
            logger.info(f"Successfully subscribed to {len(self.active_subscriptions)} feeds")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to feeds: {e}")
    
    async def _handle_price_update(self, message: Dict[str, Any]) -> None:
        """Handle mid price updates."""
        try:
            data = message.get('data', {})
            
            if isinstance(data, dict):
                timestamp = datetime.now(timezone.utc)
                
                # Process each symbol's price
                for symbol, price_str in data.items():
                    if symbol in self.symbols:
                        price_data = {
                            'symbol': symbol,
                            'price': float(price_str),
                            'timestamp': timestamp,
                            'type': 'mid'
                        }
                        self.price_buffer.append(price_data)
                
                self.stats['prices_collected'] += len(data)
                self.stats['last_activity'] = timestamp
                
                # Flush if buffer is full
                if len(self.price_buffer) >= self.buffer_size:
                    await self._flush_price_buffer()
            
        except Exception as e:
            logger.error(f"Error handling price update: {e}")
            self.stats['errors'] += 1
    
    async def _handle_trade_update(self, message: Dict[str, Any]) -> None:
        """Handle trade updates."""
        try:
            data = message.get('data', [])
            
            if isinstance(data, list):
                timestamp = datetime.now(timezone.utc)
                
                for trade in data:
                    if isinstance(trade, dict):
                        symbol = trade.get('coin')
                        if symbol and symbol in self.symbols:
                            trade_data = {
                                'symbol': symbol,
                                'price': float(trade.get('px', 0)),
                                'size': float(trade.get('sz', 0)),
                                'side': 'buy' if trade.get('side') == 'B' else 'sell',
                                'timestamp': datetime.fromtimestamp(trade.get('time', 0) / 1000, tz=timezone.utc),
                                'trade_id': trade.get('tid')
                            }
                            self.trade_buffer.append(trade_data)
                
                self.stats['trades_collected'] += len(data)
                self.stats['last_activity'] = timestamp
                
                # Flush if buffer is full
                if len(self.trade_buffer) >= self.buffer_size:
                    await self._flush_trade_buffer()
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
            self.stats['errors'] += 1
    
    async def _handle_orderbook_update(self, message: Dict[str, Any]) -> None:
        """Handle order book updates."""
        try:
            data = message.get('data', {})
            
            if isinstance(data, dict) and 'levels' in data:
                timestamp = datetime.now(timezone.utc)
                symbol = data.get('coin')
                
                if symbol and symbol in self.symbols:
                    # Convert HyperLiquid order book format to our database format
                    levels = data.get('levels', [])
                    if len(levels) >= 2:
                        bids = levels[0] if len(levels) > 0 else []
                        asks = levels[1] if len(levels) > 1 else []
                        
                        orderbook_data = {
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'bids': json.dumps(bids),
                            'asks': json.dumps(asks),
                            'time_ms': data.get('time', int(timestamp.timestamp() * 1000))
                        }
                        self.orderbook_buffer.append(orderbook_data)
                
                self.stats['orderbooks_collected'] += 1
                self.stats['last_activity'] = timestamp
                
                # Flush if buffer is full
                if len(self.orderbook_buffer) >= self.buffer_size:
                    await self._flush_orderbook_buffer()
            
        except Exception as e:
            logger.error(f"Error handling orderbook update: {e}")
            self.stats['errors'] += 1
    
    async def _flush_price_buffer(self) -> None:
        """Flush price data to database."""
        if not self.price_buffer:
            return
        
        try:
            # Insert price data (could be candles table or separate prices table)
            # For now, we'll create a simple prices table structure
            await self.db_manager.execute_query(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    type TEXT DEFAULT 'mid'
                )
                """
            )
            
            # Batch insert
            insert_query = """
                INSERT INTO prices (symbol, price, timestamp, type)
                VALUES (?, ?, ?, ?)
            """
            
            params = [
                (row['symbol'], row['price'], row['timestamp'], row['type'])
                for row in self.price_buffer
            ]
            
            await self.db_manager.execute_many(insert_query, params)
            
            logger.debug(f"Flushed {len(self.price_buffer)} price records to database")
            self.price_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush price buffer: {e}")
            self.stats['errors'] += 1
    
    async def _flush_trade_buffer(self) -> None:
        """Flush trade data to database."""
        if not self.trade_buffer:
            return
        
        try:
            # Use existing trades table from schema
            insert_query = """
                INSERT INTO trades (symbol, price, size, side, timestamp, time_ms, trade_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = [
                (
                    row['symbol'], 
                    row['price'], 
                    row['size'], 
                    row['side'], 
                    row['timestamp'],
                    int(row['timestamp'].timestamp() * 1000),
                    row['trade_id']
                )
                for row in self.trade_buffer
            ]
            
            await self.db_manager.execute_many(insert_query, params)
            
            logger.debug(f"Flushed {len(self.trade_buffer)} trade records to database")
            self.trade_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush trade buffer: {e}")
            self.stats['errors'] += 1
    
    async def _flush_orderbook_buffer(self) -> None:
        """Flush order book data to database."""
        if not self.orderbook_buffer:
            return
        
        try:
            # Use existing order_books table from schema
            insert_query = """
                INSERT OR REPLACE INTO order_books (symbol, timestamp, bids, asks, time_ms)
                VALUES (?, ?, ?, ?, ?)
            """
            
            params = [
                (
                    row['symbol'], 
                    row['timestamp'], 
                    row['bids'], 
                    row['asks'], 
                    row['time_ms']
                )
                for row in self.orderbook_buffer
            ]
            
            await self.db_manager.execute_many(insert_query, params)
            
            logger.debug(f"Flushed {len(self.orderbook_buffer)} orderbook records to database")
            self.orderbook_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush orderbook buffer: {e}")
            self.stats['errors'] += 1
    
    async def _flush_buffers(self) -> None:
        """Flush all data buffers to database."""
        await asyncio.gather(
            self._flush_price_buffer(),
            self._flush_trade_buffer(),
            self._flush_orderbook_buffer(),
            return_exceptions=True
        )
    
    async def _periodic_flush(self) -> None:
        """Periodically flush buffers to database."""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffers()
                
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                self.stats['errors'] += 1
    
    async def _periodic_stats(self) -> None:
        """Periodically log collection statistics."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Log stats every minute
                
                logger.info(
                    f"DataCollector stats: "
                    f"prices={self.stats['prices_collected']}, "
                    f"trades={self.stats['trades_collected']}, "
                    f"orderbooks={self.stats['orderbooks_collected']}, "
                    f"errors={self.stats['errors']}, "
                    f"subscriptions={len(self.active_subscriptions)}"
                )
                
            except Exception as e:
                logger.error(f"Error logging stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            **self.stats,
            'active_subscriptions': len(self.active_subscriptions),
            'subscription_types': list(self.active_subscriptions),
            'symbols_count': len(self.symbols),
            'buffer_sizes': {
                'prices': len(self.price_buffer),
                'trades': len(self.trade_buffer),
                'orderbooks': len(self.orderbook_buffer)
            },
            'running': self.running
        }
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the collection list."""
        if symbol not in self.symbols:
            self.symbols.add(symbol)
            logger.info(f"Added symbol {symbol} to collection list")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the collection list."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
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
            target_symbols = symbols or list(self.symbols)
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
            
            # Convert HyperLiquid format to database format
            db_candles = []
            for candle in candles:
                try:
                    # HyperLiquid candle format: {'t': timestamp_ms, 's': symbol, 'o': open, 'c': close, 'h': high, 'l': low, 'v': volume, 'n': count}
                    timestamp_ms = int(candle.get('t', 0))
                    
                    if timestamp_ms == 0:
                        continue
                    
                    # Convert to datetime
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    
                    db_candle = {
                        'symbol': symbol,
                        'open_price': float(candle.get('o', 0)),
                        'high_price': float(candle.get('h', 0)),
                        'low_price': float(candle.get('l', 0)),
                        'close_price': float(candle.get('c', 0)),
                        'volume': float(candle.get('v', 0)),
                        'trade_count': int(candle.get('n', 0)),
                        'timestamp_start': timestamp,
                        'timestamp_end': timestamp  # For interval data, start and end are typically the same
                    }
                    
                    db_candles.append(db_candle)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid candle data: {e}")
                    continue
            
            if not db_candles:
                logger.warning(f"No valid candles to store for {symbol}")
                return 0
            
            # Use database manager to insert
            # This assumes the database manager has a method to insert candles
            # We'll need to use the data insertion component
            try:
                # Get database connection
                with self.db_manager.get_connection() as conn:
                    # Prepare INSERT query with ON CONFLICT handling
                    sequence_name = f"{table_name}_seq"
                    query = f"""
                    INSERT INTO {table_name} 
                    (id, symbol, timestamp_start, timestamp_end, open_price, close_price, high_price, low_price, volume, trade_count)
                    VALUES (nextval('{sequence_name}'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (symbol, timestamp_start) DO UPDATE SET
                        timestamp_end = EXCLUDED.timestamp_end,
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        trade_count = EXCLUDED.trade_count
                    """
                    
                    # Execute batch insert
                    cursor = conn.cursor()
                    insert_data = [
                        (
                            candle['symbol'],
                            candle['timestamp_start'],
                            candle['timestamp_end'],
                            candle['open_price'],
                            candle['close_price'],
                            candle['high_price'],
                            candle['low_price'],
                            candle['volume'],
                            candle['trade_count']
                        )
                        for candle in db_candles
                    ]
                    
                    cursor.executemany(query, insert_data)
                    conn.commit()
                    
                    inserted_count = len(db_candles)
                    logger.debug(f"Stored {inserted_count} candles in {table_name}")
                    
                    return inserted_count
                    
            except Exception as db_error:
                logger.error(f"Database error storing candles: {db_error}")
                return 0
            
        except Exception as e:
            logger.error(f"Failed to store historical candles: {e}")
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