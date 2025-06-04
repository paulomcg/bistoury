"""
Integration tests for HyperLiquid API functionality.
Tests the actual HyperLiquid integration with real API calls.
"""

import pytest
import asyncio
import warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from src.bistoury.hyperliquid.client import HyperLiquidIntegration
from src.bistoury.hyperliquid.collector import DataCollector
from src.bistoury.database import DatabaseManager
from src.bistoury.config import Config
from src.bistoury.logger import get_logger

logger = get_logger(__name__)


class TestHyperLiquidIntegration:
    """Integration tests for HyperLiquid API client."""
    
    @pytest.fixture
    async def hl_client(self):
        """Create HyperLiquid client for testing."""
        config = Config.load_from_env()
        client = HyperLiquidIntegration(config)
        yield client
        # Cleanup
        if hasattr(client, 'ws_manager') and client.ws_manager:
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_health_check(self, hl_client):
        """Test HyperLiquid API health check."""
        result = await hl_client.health_check()
        assert result is True, "Health check should pass"
    
    @pytest.mark.asyncio
    async def test_get_all_mids(self, hl_client):
        """Test getting all mid prices."""
        mids = await hl_client.get_all_mids()
        
        assert isinstance(mids, dict), "Mids should be a dictionary"
        assert len(mids) > 0, "Should have at least some mid prices"
        
        # Check that values are price strings
        sample_key = next(iter(mids.keys()))
        sample_value = mids[sample_key]
        assert isinstance(sample_value, str), "Mid prices should be strings"
        
        # Try to convert to float to verify it's a valid price
        float(sample_value)  # Should not raise ValueError
        
        # Verify we have major cryptocurrencies
        major_symbols = {'BTC', 'ETH', 'SOL'}
        found_symbols = set(mids.keys())
        overlap = major_symbols.intersection(found_symbols)
        assert len(overlap) >= 2, f"Should have at least 2 major symbols, found: {overlap}"
    
    @pytest.mark.asyncio
    async def test_get_metadata(self, hl_client):
        """Test getting market metadata."""
        metadata = await hl_client.get_meta()
        
        assert isinstance(metadata, dict), "Metadata should be a dict"
        assert len(metadata) > 0, "Should have metadata"
        
        # Check that it has expected structure
        assert 'universe' in metadata, "Should have universe data"
        
        universe = metadata['universe']
        assert isinstance(universe, list), "Universe should be a list"
        assert len(universe) > 100, "Should have substantial number of symbols"
        
        # Verify structure of symbol info
        if universe:
            sample_symbol = universe[0]
            assert 'name' in sample_symbol, "Symbol should have name"
            
        logger.info(f"Retrieved metadata with {len(universe)} symbols")
    
    @pytest.mark.asyncio
    async def test_get_candles_btc(self, hl_client):
        """Test getting candlestick data for BTC."""
        candles = await hl_client.get_candles('BTC', '5m')
        
        assert isinstance(candles, list), "Candles should be a list"
        assert len(candles) > 0, "Should have at least some candles"
        
        # Check structure of first candle
        first_candle = candles[0]
        required_fields = ['t', 's', 'o', 'c', 'h', 'l', 'v']
        for field in required_fields:
            assert field in first_candle, f"Candle should have {field} field"
        
        assert first_candle['s'] == 'BTC', "Symbol should match request"
        
        # Verify price data is valid
        ohlc_fields = ['o', 'h', 'l', 'c']
        for field in ohlc_fields:
            price = float(first_candle[field])
            assert price > 0, f"{field} price should be positive"
        
        # Verify high >= low
        assert float(first_candle['h']) >= float(first_candle['l']), "High should be >= Low"
    
    @pytest.mark.asyncio
    async def test_get_candles_multiple_intervals(self, hl_client):
        """Test getting candlestick data for multiple intervals."""
        intervals = ['1m', '5m', '15m', '1h']
        
        for interval in intervals:
            candles = await hl_client.get_candles('BTC', interval)
            assert isinstance(candles, list), f"Candles for {interval} should be a list"
            assert len(candles) > 0, f"Should have candles for {interval}"
            
            if candles:
                assert candles[0]['s'] == 'BTC', f"Symbol should be BTC for {interval}"
            
            # Small delay to avoid overwhelming API
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_get_candles_date_range(self, hl_client):
        """Test getting candlestick data for specific date range."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=2)
        
        candles = await hl_client.get_candles('BTC', '5m', start_time, end_time)
        
        assert isinstance(candles, list), "Candles should be a list"
        assert len(candles) > 0, "Should have candles in time range"
        
        # Verify timestamps are within a reasonable range
        # Note: Candles may start slightly before the requested time due to interval alignment
        if candles:
            first_timestamp = int(candles[0]['t'])
            last_timestamp = int(candles[-1]['t'])
            
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            # Allow 5 minutes tolerance for candle alignment (5 * 60 * 1000 ms)
            tolerance_ms = 5 * 60 * 1000
            
            assert first_timestamp >= (start_ms - tolerance_ms), f"First candle should be close to start time. Got {first_timestamp}, expected >= {start_ms - tolerance_ms}"
            assert last_timestamp <= end_ms, f"Last candle should be before end time. Got {last_timestamp}, expected <= {end_ms}"
    
    @pytest.mark.asyncio
    async def test_get_order_book_btc(self, hl_client):
        """Test getting order book for BTC."""
        order_book = await hl_client.get_order_book('BTC')
        
        assert isinstance(order_book, dict), "Order book should be a dictionary"
        assert len(order_book) > 0, "Order book should have data"
        
        # Order book structure varies, but should have some key fields
        assert 'coin' in order_book or 'levels' in order_book or 'bids' in order_book, \
            "Order book should have expected structure"
    
    @pytest.mark.asyncio
    async def test_connection_info(self, hl_client):
        """Test getting connection information."""
        info = hl_client.get_connection_info()
        
        assert isinstance(info, dict), "Connection info should be a dict"
        assert 'base_url' in info, "Should have base URL"
        assert 'ws_url' in info, "Should have WebSocket URL"
        assert 'testnet' in info, "Should have testnet status"
        
        # Verify URL format
        assert info['base_url'].startswith('https://'), "Base URL should be HTTPS"
        assert info['ws_url'].startswith('wss://'), "WebSocket URL should be WSS"


class TestHyperLiquidWebSocket:
    """Integration tests for HyperLiquid WebSocket functionality."""
    
    @pytest.fixture
    async def hl_client(self):
        """Create HyperLiquid client for WebSocket testing."""
        config = Config.load_from_env()
        client = HyperLiquidIntegration(config)
        yield client
        # Cleanup
        if hasattr(client, 'ws_manager') and client.ws_manager:
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, hl_client):
        """Test WebSocket connection establishment."""
        result = await hl_client.connect()
        assert result is True, "WebSocket connection should succeed"
        assert hl_client.is_connected(), "Client should report as connected"
        
        # Test disconnect
        await hl_client.disconnect()
        assert not hl_client.is_connected(), "Client should report as disconnected"
    
    @pytest.mark.asyncio
    async def test_websocket_subscription_all_mids(self, hl_client):
        """Test subscribing to all mid price updates."""
        # Connect first
        connected = await hl_client.connect()
        assert connected, "Must be connected before subscribing"
        
        # Store received messages
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
            logger.debug(f"Received mid price message: {len(message.get('data', {})) if message else 0} symbols")
        
        # Subscribe
        subscribed = await hl_client.subscribe_all_mids(message_handler)
        assert subscribed, "Subscription should succeed"
        
        # Wait for messages
        await asyncio.sleep(3)
        
        # Verify subscription tracking
        assert 'allMids' in hl_client.subscriptions, "Should be subscribed to allMids"
        
        # In test environment, might not get real-time data immediately
        logger.info(f"Received {len(received_messages)} mid price messages")
    
    @pytest.mark.asyncio
    async def test_websocket_subscription_orderbook(self, hl_client):
        """Test subscribing to order book updates."""
        # Connect first
        connected = await hl_client.connect()
        assert connected, "Must be connected before subscribing"
        
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
            logger.debug(f"Received order book message for: {message.get('data', {}).get('coin', 'unknown')}")
        
        # Subscribe to BTC order book
        subscribed = await hl_client.subscribe_orderbook('BTC', message_handler)
        assert subscribed, "Order book subscription should succeed"
        
        subscription_key = 'l2Book_BTC'
        assert subscription_key in hl_client.subscriptions, f"Should be subscribed to {subscription_key}"
        
        # Wait for potential messages
        await asyncio.sleep(2)
        
        logger.info(f"Received {len(received_messages)} order book messages")
    
    @pytest.mark.asyncio 
    async def test_websocket_subscription_trades(self, hl_client):
        """Test subscribing to trade updates."""
        # Connect first
        connected = await hl_client.connect()
        assert connected, "Must be connected before subscribing"
        
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
            trades = message.get('data', [])
            logger.debug(f"Received {len(trades)} trade updates")
        
        # Subscribe to BTC trades
        subscribed = await hl_client.subscribe_trades('BTC', message_handler)
        assert subscribed, "Trades subscription should succeed"
        
        subscription_key = 'trades_BTC'
        assert subscription_key in hl_client.subscriptions, f"Should be subscribed to {subscription_key}"
        
        # Wait for potential messages
        await asyncio.sleep(2)
        
        logger.info(f"Received {len(received_messages)} trade messages")
    
    @pytest.mark.asyncio
    async def test_websocket_multiple_subscriptions(self, hl_client):
        """Test subscribing to multiple feeds simultaneously."""
        # Connect first
        connected = await hl_client.connect()
        assert connected, "Must be connected before subscribing"
        
        # Subscribe to multiple feeds
        symbols = ['BTC', 'ETH']
        
        for symbol in symbols:
            trades_success = await hl_client.subscribe_trades(symbol)
            assert trades_success, f"Should subscribe to {symbol} trades"
            
            orderbook_success = await hl_client.subscribe_orderbook(symbol)
            assert orderbook_success, f"Should subscribe to {symbol} order book"
            
            # Small delay between subscriptions
            await asyncio.sleep(0.2)
        
        # Also subscribe to all mids
        mids_success = await hl_client.subscribe_all_mids()
        assert mids_success, "Should subscribe to all mids"
        
        # Verify all subscriptions are tracked
        expected_subscriptions = {'allMids', 'trades_BTC', 'trades_ETH', 'l2Book_BTC', 'l2Book_ETH'}
        actual_subscriptions = set(hl_client.subscriptions.keys())
        
        missing = expected_subscriptions - actual_subscriptions
        assert len(missing) == 0, f"Missing subscriptions: {missing}"
        
        logger.info(f"Successfully subscribed to {len(actual_subscriptions)} feeds")


class TestDataCollectorIntegration:
    """Integration tests for the DataCollector component."""
    
    @pytest.fixture
    async def setup_collector(self, test_database):
        """Create DataCollector with test database."""
        config = Config.load_from_env()
        hl_client = HyperLiquidIntegration(config)
        collector = DataCollector(hl_client, test_database, symbols=['BTC', 'ETH'])
        
        yield collector
        
        # Cleanup
        await collector.stop()
        if hl_client.is_connected():
            await hl_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self, setup_collector):
        """Test DataCollector initialization."""
        collector = setup_collector
        
        assert not collector.running, "Collector should not be running initially"
        assert len(collector.symbols) == 2, "Should have specified symbols"
        assert 'BTC' in collector.symbols, "Should have BTC"
        assert 'ETH' in collector.symbols, "Should have ETH"
        
        stats = collector.get_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert stats['prices_collected'] == 0, "Should start with zero prices collected"
    
    @pytest.mark.asyncio
    async def test_collector_start_stop(self, setup_collector):
        """Test DataCollector start and stop functionality."""
        collector = setup_collector
        
        # Test start
        started = await collector.start()
        assert started, "Collector should start successfully"
        assert collector.running, "Collector should be running"
        
        # Wait a bit for initialization
        await asyncio.sleep(1)
        
        # Test stop
        await collector.stop()
        assert not collector.running, "Collector should be stopped"
    
    @pytest.mark.asyncio
    async def test_collector_symbol_discovery(self, setup_collector):
        """Test automatic symbol discovery."""
        collector = setup_collector
        
        # Clear symbols to test discovery
        collector.symbols = set()
        
        # Start collector (should trigger discovery)
        started = await collector.start()
        assert started, "Collector should start successfully"
        
        # Should have discovered symbols
        assert len(collector.symbols) > 0, "Should have discovered symbols"
        assert len(collector.symbols) > 10, "Should have discovered many symbols"
        
        logger.info(f"Discovered {len(collector.symbols)} symbols")
    
    @pytest.mark.asyncio
    async def test_collector_historical_data(self, setup_collector):
        """Test historical data collection."""
        collector = setup_collector
        
        # Start collector first
        await collector.start()
        
        # Collect historical data for BTC
        count = await collector.collect_historical_data('BTC', days_back=1, interval='1h')
        
        assert count > 0, "Should collect some historical data"
        assert count <= 26, "Should not significantly exceed expected candle count for 1 day/1h"
        
        logger.info(f"Collected {count} historical candles")


class TestErrorHandlingAndResilience:
    """Test error handling and connection resilience."""
    
    @pytest.fixture
    async def hl_client(self):
        """Create HyperLiquid client for error testing."""
        config = Config.load_from_env()
        client = HyperLiquidIntegration(config)
        yield client
        if client.is_connected():
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, hl_client):
        """Test handling of invalid symbols."""
        # Test with invalid symbol
        candles = await hl_client.get_candles('INVALID_SYMBOL_123', '1m')
        
        # Should return empty list or handle gracefully
        assert isinstance(candles, list), "Should return list even for invalid symbol"
        
        # Order book test
        order_book = await hl_client.get_order_book('INVALID_SYMBOL_123')
        assert isinstance(order_book, dict), "Should return dict even for invalid symbol"
    
    @pytest.mark.asyncio
    async def test_subscription_without_connection(self, hl_client):
        """Test subscribing without WebSocket connection."""
        # Ensure not connected
        assert not hl_client.is_connected(), "Should not be connected initially"
        
        # Try to subscribe
        result = await hl_client.subscribe_all_mids()
        assert not result, "Subscription should fail without connection"
    
    @pytest.mark.asyncio
    async def test_reconnection_behavior(self, hl_client):
        """Test WebSocket reconnection behavior."""
        # Connect
        connected = await hl_client.connect()
        assert connected, "Should connect initially"
        
        # Disconnect
        await hl_client.disconnect()
        assert not hl_client.is_connected(), "Should be disconnected"
        
        # Reconnect
        reconnected = await hl_client.connect()
        assert reconnected, "Should be able to reconnect"
        assert hl_client.is_connected(), "Should be connected after reconnection"


@pytest.mark.asyncio
async def test_hyperliquid_integration_full_workflow():
    """Test a complete workflow using HyperLiquid integration."""
    config = Config.load_from_env()
    client = HyperLiquidIntegration(config)
    
    try:
        # 1. Health check
        health = await client.health_check()
        assert health, "API should be healthy"
        
        # 2. Get metadata
        metadata = await client.get_meta()
        assert len(metadata) > 0, "Should have symbol metadata"
        
        # 3. Get current prices
        mids = await client.get_all_mids()
        assert len(mids) > 0, "Should have current prices"
        
        # 4. Get historical data
        candles = await client.get_candles('BTC', '1h')
        assert len(candles) > 0, "Should have historical candles"
        
        # 5. Get current order book
        order_book = await client.get_order_book('BTC')
        assert len(order_book) > 0, "Should have order book data"
        
        # 6. Test WebSocket
        connected = await client.connect()
        if connected:
            # Quick subscription test
            subscribed = await client.subscribe_all_mids()
            assert subscribed, "Should be able to subscribe"
            
            # Wait briefly for potential messages
            await asyncio.sleep(1)
            
            # Test disconnection
            await client.disconnect()
            assert not client.is_connected(), "Should be disconnected"
        
        logger.info("Full workflow test completed successfully")
        
    finally:
        # Cleanup
        if client.is_connected():
            await client.disconnect()


@pytest.mark.asyncio
async def test_performance_benchmarks():
    """Test performance characteristics of HyperLiquid integration."""
    config = Config.load_from_env()
    client = HyperLiquidIntegration(config)
    
    try:
        # Benchmark API response times
        start_time = datetime.now()
        
        # Health check timing
        health_start = datetime.now()
        health = await client.health_check()
        health_time = (datetime.now() - health_start).total_seconds()
        
        assert health, "Health check should pass"
        assert health_time < 5.0, f"Health check took too long: {health_time}s"
        
        # Mids timing
        mids_start = datetime.now()
        mids = await client.get_all_mids()
        mids_time = (datetime.now() - mids_start).total_seconds()
        
        assert len(mids) > 0, "Should get mid prices"
        assert mids_time < 5.0, f"Get mids took too long: {mids_time}s"
        
        # Candles timing
        candles_start = datetime.now()
        candles = await client.get_candles('BTC', '5m')
        candles_time = (datetime.now() - candles_start).total_seconds()
        
        assert len(candles) > 0, "Should get candles"
        assert candles_time < 10.0, f"Get candles took too long: {candles_time}s"
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Performance benchmarks:")
        logger.info(f"  Health check: {health_time:.3f}s")
        logger.info(f"  Get mids: {mids_time:.3f}s")
        logger.info(f"  Get candles: {candles_time:.3f}s")
        logger.info(f"  Total time: {total_time:.3f}s")
        
        # Overall performance should be reasonable
        assert total_time < 20.0, f"Total benchmark time too long: {total_time}s"
        
    finally:
        if client.is_connected():
            await client.disconnect() 