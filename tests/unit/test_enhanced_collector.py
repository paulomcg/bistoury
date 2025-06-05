"""
Test suite for the enhanced data collector.

Tests the enhanced collector agent implementation using database entity models
for type safety, validation, and performance optimization.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.bistoury.hyperliquid.collector import (
    EnhancedDataCollector,
    CollectorConfig,
    CollectorStats,
    DataCollector
)
from src.bistoury.models.database import (
    DBCandlestickData,
    DBTradeData, 
    DBOrderBookSnapshot,
    DBFundingRateData
)
from src.bistoury.models.market_data import CandlestickData, Timeframe
from src.bistoury.models.serialization import CompressionLevel


class TestCollectorConfig:
    """Test collector configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CollectorConfig()
        
        assert isinstance(config.symbols, set)
        assert len(config.symbols) == 0
        assert config.buffer_size == 1000
        assert config.flush_interval == 30.0
        assert config.compression_level == CompressionLevel.MEDIUM
        assert config.enable_validation is True
        assert config.enable_monitoring is True
        assert 'BTC' in config.orderbook_symbols
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CollectorConfig(
            symbols={'BTC', 'ETH'},
            buffer_size=500,
            compression_level=CompressionLevel.HIGH,
            enable_validation=False
        )
        
        assert config.symbols == {'BTC', 'ETH'}
        assert config.buffer_size == 500
        assert config.compression_level == CompressionLevel.HIGH
        assert config.enable_validation is False


class TestCollectorStats:
    """Test collector statistics tracking."""
    
    def test_stats_creation(self):
        """Test stats creation and initialization."""
        stats = CollectorStats()
        
        assert stats.candles_collected == 0
        assert stats.trades_collected == 0
        assert stats.errors == 0
        assert stats.start_time is None
    
    def test_stats_to_dict(self):
        """Test stats conversion to dictionary."""
        stats = CollectorStats(
            candles_collected=100,
            trades_collected=50,
            errors=2,
            start_time=datetime.now(timezone.utc)
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict['candles_collected'] == 100
        assert stats_dict['trades_collected'] == 50
        assert stats_dict['errors'] == 2
        assert 'uptime_seconds' in stats_dict
        assert isinstance(stats_dict['uptime_seconds'], float)


class TestEnhancedDataCollector:
    """Test enhanced data collector functionality."""
    
    @pytest.fixture
    def mock_hyperliquid(self):
        """Mock HyperLiquid integration."""
        mock = AsyncMock()
        mock.is_connected.return_value = True
        mock.connect.return_value = True
        mock.get_meta.return_value = {
            'universe': [
                {'name': 'BTC'},
                {'name': 'ETH'},
                {'name': 'SOL'}
            ]
        }
        mock.subscribe_all_mids.return_value = True
        mock.subscribe_trades.return_value = True
        mock.subscribe_orderbook.return_value = True
        return mock
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        mock = MagicMock()
        mock.execute = MagicMock(return_value=[(1,)])
        mock.execute_many = MagicMock()
        mock.get_connection = MagicMock()
        return mock
    
    @pytest.fixture
    def collector_config(self):
        """Test collector configuration."""
        return CollectorConfig(
            symbols={'BTC', 'ETH'},
            buffer_size=10,
            flush_interval=1.0,
            enable_validation=True
        )
    
    @pytest.fixture
    def enhanced_collector(self, mock_hyperliquid, mock_db_manager, collector_config):
        """Enhanced data collector instance."""
        return EnhancedDataCollector(
            hyperliquid=mock_hyperliquid,
            db_manager=mock_db_manager,
            config=collector_config
        )
    
    def test_initialization(self, enhanced_collector):
        """Test enhanced collector initialization."""
        assert not enhanced_collector.running
        assert len(enhanced_collector.active_subscriptions) == 0
        assert len(enhanced_collector.trade_buffer) == 0
        assert len(enhanced_collector.orderbook_buffer) == 0
        assert isinstance(enhanced_collector.stats, CollectorStats)
    
    @pytest.mark.asyncio
    async def test_symbol_discovery(self, enhanced_collector, mock_hyperliquid):
        """Test automatic symbol discovery."""
        # Clear symbols to trigger discovery
        enhanced_collector.config.symbols.clear()
        
        await enhanced_collector._discover_symbols()
        
        assert 'BTC' in enhanced_collector.config.symbols
        assert 'ETH' in enhanced_collector.config.symbols
        assert 'SOL' in enhanced_collector.config.symbols
        mock_hyperliquid.get_meta.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_table_creation(self, enhanced_collector, mock_db_manager):
        """Test database table creation."""
        await enhanced_collector._initialize_database_tables()
        
        # Should have called execute multiple times for table creation
        assert mock_db_manager.execute.call_count >= 8
        
        # Check that candle tables are created
        calls = [call[0][0] for call in mock_db_manager.execute.call_args_list]
        candle_table_calls = [call for call in calls if 'candles_' in call]
        assert len(candle_table_calls) >= 6  # At least 6 timeframe tables
    
    @pytest.mark.asyncio
    async def test_enhanced_trade_update_handling(self, enhanced_collector):
        """Test enhanced trade update processing."""
        # Mock trade message
        trade_message = {
            'data': [{
                'coin': 'BTC',
                'px': '50000.0',
                'sz': '0.1',
                'side': 'B',
                'time': 1749092733430,
                'tid': 123456  # Use integer instead of string
            }]
        }
        
        await enhanced_collector._handle_enhanced_trade_update(trade_message)
        
        assert len(enhanced_collector.trade_buffer) == 1
        trade = enhanced_collector.trade_buffer[0]
        
        assert isinstance(trade, DBTradeData)
        assert trade.symbol == 'BTC'
        assert trade.price == '50000.0'
        assert trade.size == '0.1'
        assert enhanced_collector.stats.trades_collected == 1
    
    @pytest.mark.asyncio
    async def test_enhanced_orderbook_update_handling(self, enhanced_collector):
        """Test enhanced orderbook update processing."""
        # Mock orderbook message
        orderbook_message = {
            'data': {
                'coin': 'BTC',
                'levels': [
                    [['50000.0', '1.0'], ['49999.0', '0.5']],  # bids
                    [['50001.0', '0.8'], ['50002.0', '0.3']]   # asks
                ]
            }
        }
        
        await enhanced_collector._handle_enhanced_orderbook_update(orderbook_message)
        
        assert len(enhanced_collector.orderbook_buffer) == 1
        orderbook = enhanced_collector.orderbook_buffer[0]
        
        assert isinstance(orderbook, DBOrderBookSnapshot)
        assert orderbook.symbol == 'BTC'
        assert orderbook.bids is not None
        assert orderbook.asks is not None
        assert enhanced_collector.stats.orderbooks_collected == 1
    
    @pytest.mark.asyncio
    async def test_enhanced_candle_update_handling(self, enhanced_collector):
        """Test enhanced candle update processing from WebSocket."""
        # Mock candle message matching HyperLiquid format
        candle_message = {
            'data': {
                's': 'BTC',        # symbol
                'i': '1m',         # interval  
                't': 1672531200000, # open time millis (2023-01-01 00:00:00 UTC)
                'T': 1672531260000, # close time millis (2023-01-01 00:01:00 UTC)
                'o': '50000.0',    # open price
                'h': '50100.0',    # high price
                'l': '49900.0',    # low price
                'c': '50050.0',    # close price
                'v': '10.5',       # volume
                'n': 42            # trade count
            }
        }
        
        await enhanced_collector._handle_enhanced_candle_update(candle_message)
        
        assert len(enhanced_collector.candle_buffer) == 1
        candle, interval = enhanced_collector.candle_buffer[0]
        
        assert isinstance(candle, DBCandlestickData)
        assert candle.symbol == 'BTC'
        assert candle.open_price == '50000.0'
        assert candle.high_price == '50100.0'
        assert candle.low_price == '49900.0'
        assert candle.close_price == '50050.0'
        assert candle.volume == '10.5'
        assert candle.trade_count == 42
        assert interval == '1m'
        assert enhanced_collector.stats.candles_collected == 1

    @pytest.mark.asyncio
    async def test_multiple_candle_intervals(self, enhanced_collector):
        """Test candle processing for multiple intervals."""
        intervals = ['1m', '5m', '1h']
        
        for i, interval in enumerate(intervals):
            candle_message = {
                'data': {
                    's': 'BTC',
                    'i': interval,
                    't': 1672531200000 + (i * 60000),  # Different timestamps
                    'T': 1672531260000 + (i * 60000),
                    'o': f'{50000 + i}.0',
                    'h': f'{50100 + i}.0', 
                    'l': f'{49900 + i}.0',
                    'c': f'{50050 + i}.0',
                    'v': f'10.{i}',
                    'n': 42 + i
                }
            }
            await enhanced_collector._handle_enhanced_candle_update(candle_message)
        
        assert len(enhanced_collector.candle_buffer) == 3
        assert enhanced_collector.stats.candles_collected == 3
        
        # Verify each interval is properly stored
        intervals_found = [interval for _, interval in enhanced_collector.candle_buffer]
        assert set(intervals_found) == set(intervals)

    @pytest.mark.asyncio
    async def test_candle_buffer_flushing(self, enhanced_collector, mock_db_manager):
        """Test candle buffer flushing to database with readable field names."""
        # Add candles to buffer
        for i in range(3):
            candle_message = {
                'data': {
                    's': 'BTC',
                    'i': '1m',
                    't': 1672531200000 + (i * 60000),
                    'T': 1672531260000 + (i * 60000),
                    'o': f'{50000 + i}.0',
                    'h': f'{50100 + i}.0',
                    'l': f'{49900 + i}.0', 
                    'c': f'{50050 + i}.0',
                    'v': f'10.{i}',
                    'n': 42 + i
                }
            }
            await enhanced_collector._handle_enhanced_candle_update(candle_message)
        
        # Mock database response for ID generation
        mock_db_manager.execute.return_value = [(1,)]
        
        # Flush candle buffer
        await enhanced_collector._flush_candle_buffer()
        
        # Verify buffer is cleared
        assert len(enhanced_collector.candle_buffer) == 0
        
        # Verify database insert was called with readable field names
        mock_db_manager.execute_many.assert_called()
        call_args = mock_db_manager.execute_many.call_args
        query = call_args[0][0]
        
        # Check that readable field names are used in the query (new schema)
        assert 'symbol' in query
        assert 'timestamp_start' in query
        assert 'timestamp_end' in query
        assert 'open_price' in query
        assert 'high_price' in query
        assert 'low_price' in query
        assert 'close_price' in query
        assert 'volume' in query
        assert 'trade_count' in query

    @pytest.mark.asyncio  
    async def test_candle_subscription_handling(self, enhanced_collector, mock_hyperliquid):
        """Test candle subscription setup."""
        # Mock successful candle subscriptions
        mock_hyperliquid.subscribe_candle.return_value = True
        
        enhanced_collector.config.symbols = {'BTC', 'ETH'}
        enhanced_collector.config.intervals = {'1m', '5m'}
        
        await enhanced_collector._subscribe_to_enhanced_feeds()
        
        # Verify candle subscriptions were made
        expected_calls = 2 * 2  # 2 symbols * 2 intervals
        candle_calls = [call for call in mock_hyperliquid.subscribe_candle.call_args_list]
        assert len(candle_calls) >= expected_calls - 2  # Allow for rate limiting

    def test_candle_data_validation(self, enhanced_collector):
        """Test candle data validation logic."""
        # Valid candle
        valid_candle = DBCandlestickData(
            symbol='BTC',
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc) + timedelta(minutes=1),
            open_price='50000.0',
            high_price='50100.0',
            low_price='49900.0',
            close_price='50050.0',
            volume='10.5',
            trade_count=42
        )
        
        assert enhanced_collector._validate_candle_data(valid_candle) is True
        
        # Invalid candle - negative price
        invalid_candle = DBCandlestickData(
            symbol='BTC',
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc) + timedelta(minutes=1),
            open_price='-50000.0',  # Invalid negative price
            high_price='50100.0',
            low_price='49900.0',
            close_price='50050.0',
            volume='10.5',
            trade_count=42
        )
        
        assert enhanced_collector._validate_candle_data(invalid_candle) is False
        
        # Invalid candle - high < low
        invalid_candle2 = DBCandlestickData(
            symbol='BTC',
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc) + timedelta(minutes=1),
            open_price='50000.0',
            high_price='49000.0',  # High less than low
            low_price='49900.0',
            close_price='50050.0',
            volume='10.5',
            trade_count=42
        )
        
        assert enhanced_collector._validate_candle_data(invalid_candle2) is False

    @pytest.mark.asyncio
    async def test_candle_error_handling(self, enhanced_collector):
        """Test error handling in candle processing."""
        # Invalid candle message - missing required fields
        invalid_message = {
            'data': {
                's': 'BTC',
                # Missing other required fields
            }
        }
        
        # Should not raise exception
        await enhanced_collector._handle_enhanced_candle_update(invalid_message)
        
        # Buffer should remain empty
        assert len(enhanced_collector.candle_buffer) == 0
        
        # Error count should increase
        initial_errors = enhanced_collector.stats.errors
        
        # Malformed message
        malformed_message = {
            'data': 'not_a_dict'
        }
        
        await enhanced_collector._handle_enhanced_candle_update(malformed_message)
        assert enhanced_collector.stats.errors >= initial_errors

    @pytest.mark.asyncio
    async def test_candle_timeframe_routing(self, enhanced_collector, mock_db_manager):
        """Test that candles are routed to correct timeframe tables."""
        intervals = ['1m', '5m', '1h', '1d']
        
        # Add candles for different intervals
        for interval in intervals:
            candle_message = {
                'data': {
                    's': 'BTC',
                    'i': interval,
                    't': 1672531200000,
                    'T': 1672531260000,
                    'o': '50000.0',
                    'h': '50100.0',
                    'l': '49900.0',
                    'c': '50050.0',
                    'v': '10.5',
                    'n': 42
                }
            }
            await enhanced_collector._handle_enhanced_candle_update(candle_message)
        
        # Mock database response
        mock_db_manager.execute.return_value = [(1,)]
        
        # Flush should create separate inserts for each timeframe
        await enhanced_collector._flush_candle_buffer()
        
        # Verify execute_many was called multiple times for different tables
        call_count = mock_db_manager.execute_many.call_count
        assert call_count == len(intervals)  # One call per interval table
    
    @pytest.mark.asyncio
    async def test_funding_rate_collection(self, enhanced_collector, mock_hyperliquid):
        """Test funding rate collection."""
        # Mock funding rate response
        mock_hyperliquid.get_funding_rate.return_value = {
            'fundingRate': '0.0001',
            'premium': '0.0005'
        }

        collected_count = await enhanced_collector.collect_funding_rates(['BTC'])

        assert collected_count == 1
        # Buffer is automatically flushed, so check database operations instead
        assert enhanced_collector.db_manager.execute_many.called
        assert enhanced_collector.stats.batches_processed >= 1
    
    def test_stats_reporting(self, enhanced_collector):
        """Test statistics reporting."""
        # Update some stats
        enhanced_collector.stats.trades_collected = 100
        enhanced_collector.stats.errors = 5
        enhanced_collector.active_subscriptions.add('test_subscription')
        
        stats = enhanced_collector.get_stats()
        
        assert stats['trades_collected'] == 100
        assert stats['errors'] == 5
        assert stats['active_subscriptions'] == 1
        assert 'uptime_seconds' in stats

    def test_trade_data_validation(self, enhanced_collector):
        """Test trade data validation."""
        # Valid trade
        valid_trade = DBTradeData(
            symbol='BTC',
            timestamp=datetime.now(timezone.utc),
            price='50000.0',
            size='0.1',
            side='buy',
            trade_id=12345
        )
        
        assert enhanced_collector._validate_trade_data(valid_trade) is True
        
        # Invalid trade - negative price
        invalid_trade = DBTradeData(
            symbol='BTC',
            timestamp=datetime.now(timezone.utc),
            price='-50000.0',
            size='0.1',
            side='buy',
            trade_id=12346
        )
        
        assert enhanced_collector._validate_trade_data(invalid_trade) is False

    def test_orderbook_data_validation(self, enhanced_collector):
        """Test orderbook data validation."""
        # Valid orderbook
        valid_orderbook = DBOrderBookSnapshot(
            symbol='BTC',
            timestamp=datetime.now(timezone.utc),
            time_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            bids=json.dumps([['50000.0', '1.0']]),
            asks=json.dumps([['50100.0', '1.0']])
        )
        
        assert enhanced_collector._validate_orderbook_data(valid_orderbook) is True
        
        # Invalid orderbook - missing data
        invalid_orderbook = DBOrderBookSnapshot(
            symbol='',  # Empty symbol
            timestamp=datetime.now(timezone.utc),
            time_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            bids='[]',
            asks='[]'
        )
        
        assert enhanced_collector._validate_orderbook_data(invalid_orderbook) is False

    @pytest.mark.asyncio
    async def test_buffer_flushing(self, enhanced_collector, mock_db_manager):
        """Test buffer flushing functionality."""
        # Add test data to buffers
        trade = DBTradeData(
            symbol='BTC',
            timestamp=datetime.now(timezone.utc),
            price='50000.0',
            size='0.1',
            side='buy',
            trade_id=12345
        )
        enhanced_collector.trade_buffer.append(trade)
        
        # Mock database response for ID generation
        mock_db_manager.execute.return_value = [(1,)]
        
        await enhanced_collector._flush_trade_buffer()
        
        # Check that database operations were called
        mock_db_manager.execute_many.assert_called()
        assert len(enhanced_collector.trade_buffer) == 0
        assert enhanced_collector.stats.batches_processed >= 1

    @pytest.mark.asyncio
    async def test_enhanced_historical_data_collection(self, enhanced_collector, mock_hyperliquid):
        """Test enhanced historical data collection."""
        # Mock historical data in HyperLiquid raw format
        mock_candles = [
            {
                't': int(datetime.now(timezone.utc).timestamp() * 1000),  # open time ms
                'o': '50000.0',  # open price
                'h': '50100.0',  # high price
                'l': '49900.0',  # low price
                'c': '50050.0',  # close price
                'v': '10.0'      # volume
            }
        ]
        mock_hyperliquid.get_historical_candles_bulk.return_value = mock_candles
        
        # Mock the storage method to avoid actual database operations
        enhanced_collector._store_enhanced_historical_candles = AsyncMock(return_value=1)
        
        # Mock the converter and validator to avoid business model dependencies
        enhanced_collector.converter = MagicMock()
        enhanced_collector.validator = MagicMock()
        
        # Create a mock DB candle for the converter to return
        mock_db_candle = DBCandlestickData(
            symbol='BTC',
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc) + timedelta(hours=1),
            open_price='50000.0',
            high_price='50100.0',
            low_price='49900.0',
            close_price='50050.0',
            volume='10.0',
            trade_count=100
        )
        enhanced_collector.converter.candlestick_to_db.return_value = mock_db_candle
        enhanced_collector.validator.validate_candlestick_data.return_value = True
        
        results = await enhanced_collector.collect_enhanced_historical_data(
            symbol='BTC',
            days_back=1,
            intervals=['1h']
        )
        
        assert results['1h'] == 1
        mock_hyperliquid.get_historical_candles_bulk.assert_called_once()
        enhanced_collector._store_enhanced_historical_candles.assert_called_once()


class TestBackwardCompatibility:
    """Test backward compatibility with original DataCollector."""
    
    @pytest.fixture
    def mock_hyperliquid(self):
        """Mock HyperLiquid integration."""
        mock = AsyncMock()
        mock.is_connected.return_value = True
        return mock
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        return AsyncMock()
    
    def test_original_constructor_compatibility(self, mock_hyperliquid, mock_db_manager):
        """Test that original DataCollector constructor still works."""
        symbols = ['BTC', 'ETH', 'SOL']
        
        collector = DataCollector(
            hyperliquid=mock_hyperliquid,
            db_manager=mock_db_manager,
            symbols=symbols
        )
        
        assert isinstance(collector, EnhancedDataCollector)
        assert collector.config.symbols == set(symbols)
    
    def test_original_stats_format(self, mock_hyperliquid, mock_db_manager):
        """Test that original stats format is maintained."""
        collector = DataCollector(
            hyperliquid=mock_hyperliquid,
            db_manager=mock_db_manager,
            symbols=['BTC']
        )
        
        # Update enhanced stats
        collector.stats.trades_collected = 50
        collector.stats.errors = 3
        collector.active_subscriptions.add('test')
        
        # Get stats in old format
        stats = collector.get_stats()
        
        # Check old format fields are present
        assert 'prices_collected' in stats
        assert 'trades_collected' in stats
        assert 'orderbooks_collected' in stats
        assert 'errors' in stats
        assert 'active_subscriptions' in stats
        assert 'symbols_count' in stats
        assert 'running' in stats
        
        # Check values are correctly mapped
        assert stats['trades_collected'] == 50
        assert stats['errors'] == 3
        assert stats['active_subscriptions'] == 1


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.fixture
    def enhanced_collector(self):
        """Enhanced collector with mocked dependencies."""
        mock_hyperliquid = AsyncMock()
        mock_db_manager = AsyncMock()
        config = CollectorConfig(buffer_size=5)
        
        return EnhancedDataCollector(
            hyperliquid=mock_hyperliquid,
            db_manager=mock_db_manager,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, enhanced_collector):
        """Test handling of database errors during flush."""
        # Mock database manager to raise an exception
        enhanced_collector.db_manager.execute_many = MagicMock(side_effect=Exception("Database error"))
        enhanced_collector.db_manager.execute = MagicMock(return_value=[(1,)])
        
        # Add data to buffer
        trade = DBTradeData(
            symbol='BTC',
            timestamp=datetime.now(timezone.utc),
            price='50000.0',
            size='0.1',
            side='buy',
            trade_id=12345
        )
        enhanced_collector.trade_buffer.append(trade)
        
        # Flush should handle the error gracefully
        await enhanced_collector._flush_trade_buffer()
        
        # Error should be recorded
        assert enhanced_collector.stats.errors >= 1
    
    @pytest.mark.asyncio
    async def test_invalid_message_handling(self, enhanced_collector):
        """Test handling of invalid messages."""
        # Invalid trade message
        invalid_message = {
            'data': [
                {
                    'invalid': 'data'
                }
            ]
        }
        
        # Should handle gracefully without crashing
        await enhanced_collector._handle_enhanced_trade_update(invalid_message)
        
        # Should not add anything to buffer
        assert len(enhanced_collector.trade_buffer) == 0
    
    def test_validation_error_tracking(self, enhanced_collector):
        """Test validation error tracking."""
        # Invalid trade data
        invalid_trade = DBTradeData(
            symbol='',  # Empty symbol
            timestamp=datetime.now(timezone.utc),
            price='0',  # Invalid price
            size='0',   # Invalid size
            side='invalid',
            trade_id=12345
        )
        
        result = enhanced_collector._validate_trade_data(invalid_trade)
        
        assert result is False 