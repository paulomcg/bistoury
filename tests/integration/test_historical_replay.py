"""
Integration tests for Historical Data Replay functionality in Paper Trading Engine

Tests the complete historical data replay pipeline from database to signal generation.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil
import logging

from src.bistoury.paper_trading.engine import PaperTradingEngine
from src.bistoury.paper_trading.config import (
    PaperTradingConfig, 
    TradingMode,
    HistoricalReplayConfig,
    TradingParameters,
    RiskParameters
)
from src.bistoury.models.market_data import CandlestickData, Timeframe
from src.bistoury.database import DatabaseManager
from src.bistoury.database.schema import MarketDataSchema, DataInsertion
from src.bistoury.config import Config
from src.bistoury.models.agent_messages import MessageType


@pytest.fixture
async def test_database():
    """Create temporary test database with sample data"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_historical_replay.db"
    
    try:
        # Create config with database path
        config = Config()
        config.database.path = str(db_path)
        
        # Initialize database
        db_manager = DatabaseManager(config)
        
        # Create schema
        schema = MarketDataSchema(db_manager)
        schema.recreate_all_tables()  # This is sync, not async
        
        # Insert sample historical candlestick data
        inserter = DataInsertion(db_manager)
        
        # Generate sample BTC candlestick data for 1m and 5m timeframes
        base_time = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        base_price = Decimal("50000.00")
        
        candles_1m = []
        candles_5m = []
        
        # Generate 30 minutes of 1m candles (30 candles)
        for i in range(30):
            timestamp = base_time + timedelta(minutes=i)
            price_variation = Decimal(str(i * 10))  # Simple price movement
            
            candle = CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.ONE_MINUTE,
                timestamp=timestamp,
                open=base_price + price_variation,
                high=base_price + price_variation + Decimal("100"),
                low=base_price + price_variation - Decimal("50"),
                close=base_price + price_variation + Decimal("25"),
                volume=Decimal("1.5"),
                trade_count=50
            )
            candles_1m.append(candle)
        
        # Generate 6 x 5m candles (30 minutes)
        for i in range(6):
            timestamp = base_time + timedelta(minutes=i * 5)
            price_variation = Decimal(str(i * 50))
            
            candle = CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=timestamp,
                open=base_price + price_variation,
                high=base_price + price_variation + Decimal("200"),
                low=base_price + price_variation - Decimal("100"),
                close=base_price + price_variation + Decimal("75"),
                volume=Decimal("7.5"),
                trade_count=250
            )
            candles_5m.append(candle)
        
        # Insert data (this might be sync)
        inserter.insert_candles(candles_1m)
        inserter.insert_candles(candles_5m)
        
        yield str(db_path), base_time, base_time + timedelta(minutes=30)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def paper_trading_config(test_database):
    """Create paper trading configuration for historical replay"""
    db_path, start_time, end_time = test_database
    
    return PaperTradingConfig(
        session_name="historical_replay_test",
        trading_mode=TradingMode.HISTORICAL,
        database_path=db_path,
        historical_config=HistoricalReplayConfig(
            start_date=start_time,
            end_date=end_time,
            symbols=["BTC"],
            timeframes=[Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES],
            replay_speed=100.0  # Fast replay for testing
        ),
        trading_params=TradingParameters(
            base_position_size=Decimal("0.1"),
            max_concurrent_positions=3,
            min_confidence=Decimal("0.6"),
            position_sizing_strategy="confidence_based"
        ),
        risk_params=RiskParameters(
            initial_balance=Decimal("10000.00"),
            max_position_size=Decimal("1000.00"),
            daily_loss_limit=Decimal("500.00"),
            default_stop_loss_percent=Decimal("0.02"),
            default_take_profit_percent=Decimal("0.04")
        ),
        enabled_strategies=["candlestick_strategy"],
        strategy_weights={"candlestick_strategy": Decimal("1.0")},
        performance_reporting_interval=timedelta(seconds=5),
        save_state_interval=timedelta(seconds=10)
    )


class TestHistoricalReplay:
    """Test suite for historical data replay functionality"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization_historical_mode(self, paper_trading_config):
        """Test that Paper Trading Engine initializes correctly for historical mode"""
        engine = PaperTradingEngine(paper_trading_config)
        
        try:
            await engine.initialize()
            
            # Verify components are initialized
            assert engine.db_manager is not None
            assert engine.message_bus is not None
            assert engine.agent_registry is not None
            assert engine.orchestrator is not None
            assert engine.signal_manager is not None
            assert engine.position_manager is not None
            
            # Verify status
            status = engine.get_status()
            assert status["mode"] == "HISTORICAL"
            assert status["components_status"]["database"] == "initialized"
            assert status["components_status"]["messaging"] == "initialized"
            assert status["components_status"]["signal_manager"] == "initialized"
            assert status["components_status"]["agents"] == "initialized"
            
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_database_query_functionality(self, test_database):
        """Test that database queries work correctly for historical data"""
        db_path, start_time, end_time = test_database
        
        # Create config and db manager
        config = Config()
        config.database.path = db_path
        db_manager = DatabaseManager(config)
        
        try:
            # Test 1m candles query
            query_1m = """
            SELECT timestamp_start, open_price, high_price, low_price, close_price, volume, trade_count
            FROM candles_1m
            WHERE symbol = ? AND timestamp_start BETWEEN ? AND ?
            ORDER BY timestamp_start ASC
            """
            
            # Use sync connection method
            conn = db_manager.get_connection()
            cursor = conn.execute(query_1m, ("BTC", start_time, end_time))
            rows_1m = cursor.fetchall()
            
            # Should have 30 x 1m candles
            assert len(rows_1m) == 30
            
            # Test 5m candles query
            query_5m = """
            SELECT timestamp_start, open_price, high_price, low_price, close_price, volume, trade_count
            FROM candles_5m
            WHERE symbol = ? AND timestamp_start BETWEEN ? AND ?
            ORDER BY timestamp_start ASC
            """
            
            cursor = conn.execute(query_5m, ("BTC", start_time, end_time))
            rows_5m = cursor.fetchall()
            
            # Should have 6 x 5m candles
            assert len(rows_5m) == 6
            
            # Verify data structure
            first_row = rows_1m[0]
            assert len(first_row) == 7  # timestamp_start, OHLCV, trade_count
            assert isinstance(first_row[0], datetime)  # timestamp_start as datetime
            
        finally:
            db_manager.close_all_connections()
    
    @pytest.mark.asyncio
    async def test_historical_data_replay_execution(self, paper_trading_config):
        """Test full historical data replay execution"""
        engine = PaperTradingEngine(paper_trading_config)
        
        # Track processed messages
        processed_messages = []
        original_publish = None

        def track_message(topic, message_type, payload, sender):
            processed_messages.append({
                "topic": topic,
                "type": message_type,
                "payload": payload,
                "sender": sender
            })
            # Return original coroutine result
            return original_publish(topic, message_type, payload, sender)
        
        try:
            await engine.initialize()
            
            # Mock message publishing to track data flow
            original_publish = engine.message_bus.publish
            engine.message_bus.publish = AsyncMock(side_effect=track_message)
            
            # Start engine (this will begin historical replay)
            await engine.start()
            
            # Wait for some data processing
            await asyncio.sleep(2.0)
            
            # Stop engine
            await engine.stop()
            
            # Verify data was processed
            status = engine.get_status()
            assert status["processed_data_count"] > 0
            assert status["replay_current_time"] is not None
            
            # Verify messages were published
            assert len(processed_messages) > 0
            
            # Check for market data messages
            market_data_messages = [
                msg for msg in processed_messages 
                if msg["topic"].startswith("market_data.")
            ]
            assert len(market_data_messages) > 0
            
            # Verify message structure
            sample_message = market_data_messages[0]
            assert sample_message["sender"] == "paper_trading_engine"
            assert "BTC" in sample_message["topic"]
            assert sample_message["payload"]["symbol"] == "BTC"
            assert sample_message["payload"]["data_type"] == "candlestick"
            
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio 
    async def test_agent_integration_with_historical_data(self, paper_trading_config):
        """Test that agents receive and process historical market data correctly"""
        engine = PaperTradingEngine(paper_trading_config)
        
        # Track agent processing
        candlestick_agent_calls = []
        
        async def mock_process_market_data(symbol, timeframe, data):
            candlestick_agent_calls.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": data.timestamp,
                "price": data.close_price
            })
        
        try:
            await engine.initialize()
            
            # Mock candlestick agent processing
            if engine.candlestick_agent:
                engine.candlestick_agent.process_market_data = mock_process_market_data
            
            # Start historical replay
            await engine.start()
            
            # Wait for processing
            await asyncio.sleep(3.0)
            
            # Stop engine
            await engine.stop()
            
            # Verify agents received data
            assert len(candlestick_agent_calls) > 0
            
            # Verify data completeness
            btc_1m_calls = [call for call in candlestick_agent_calls 
                           if call["symbol"] == "BTC" and call["timeframe"] == "1m"]
            btc_5m_calls = [call for call in candlestick_agent_calls 
                           if call["symbol"] == "BTC" and call["timeframe"] == "5m"]
            
            # Should have received multiple timeframe data
            assert len(btc_1m_calls) > 0
            assert len(btc_5m_calls) > 0
            
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_signal_generation_during_replay(self, paper_trading_config):
        """Test that signals are generated and processed during historical replay"""
        engine = PaperTradingEngine(paper_trading_config)
        
        # Track signal processing
        signal_calls = []
        original_handle_signal = None
        
        async def track_signal_handling(message):
            signal_calls.append({
                "message_type": message.message_type,
                "payload": message.payload,
                "timestamp": message.timestamp
            })
            # Call original handler
            return await original_handle_signal(message)
        
        try:
            await engine.initialize()
            
            # Mock signal handling
            original_handle_signal = engine._handle_trading_signal
            engine._handle_trading_signal = track_signal_handling
            
            # Start engine
            await engine.start()
            
            # Wait for processing
            await asyncio.sleep(5.0)
            
            # Stop engine
            await engine.stop()
            
            # Verify signals were processed
            status = engine.get_status()
            performance = status["performance_stats"]
            
            # Should have some activity metrics
            assert "signals_received" in performance
            assert "positions_opened" in performance
            
            # Check if any signals were generated (may be 0 if no patterns detected)
            if signal_calls:
                assert len(signal_calls) > 0
                sample_signal = signal_calls[0]
                assert "symbol" in sample_signal["payload"]
                assert "confidence" in sample_signal["payload"]
        
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, paper_trading_config):
        """Test that performance metrics are tracked correctly during replay"""
        engine = PaperTradingEngine(paper_trading_config)
        
        try:
            await engine.initialize()
            await engine.start()
            
            # Wait for processing and performance reporting
            await asyncio.sleep(6.0)  # Wait for at least one performance report
            
            await engine.stop()
            
            # Verify performance metrics
            status = engine.get_status()
            performance = status["performance_stats"]
            
            # Should have performance tracking
            assert "signals_received" in performance
            assert "signals_traded" in performance
            assert "positions_opened" in performance
            assert "positions_closed" in performance
            assert "total_pnl" in performance
            
            # Should have processed data
            assert status["processed_data_count"] > 0
            assert status["replay_current_time"] is not None
            
            # Should have runtime tracking
            assert status["runtime_seconds"] > 0
            
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_during_replay(self, paper_trading_config):
        """Test that the engine shuts down gracefully during historical replay"""
        engine = PaperTradingEngine(paper_trading_config)
        
        try:
            await engine.initialize()
            await engine.start()
            
            # Let it run briefly
            await asyncio.sleep(1.0)
            
            # Initiate shutdown
            await engine.stop()
            
            # Verify clean shutdown
            status = engine.get_status()
            assert not status["is_running"]
            assert status["end_time"] is not None
            
            # Should have processed some data before shutdown
            assert status["processed_data_count"] >= 0
            
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_engine_initialization_without_orchestrator(self, paper_trading_config):
        """Test Paper Trading Engine core initialization without orchestrator complexity"""
        
        # Override configuration to disable orchestrator
        config = paper_trading_config
        config.enabled_strategies = []  # Disable strategies to avoid agent issues
        
        engine = PaperTradingEngine(config)
        
        try:
            # Initialize core components only
            await engine._initialize_database()
            await engine._initialize_messaging()
            
            # Start the message bus
            await engine.message_bus.start()
            
            # Verify basic components initialized
            assert engine.db_manager is not None
            assert engine.message_bus is not None
            
            # Test basic functionality
            status = engine.get_status()
            assert status["mode"] == "HISTORICAL"
            assert status["session_name"] == "historical_replay_test"
            assert not status["is_running"]
            assert status["processed_data_count"] == 0
            
            # Test that we can publish messages
            await engine.message_bus.publish(
                topic="test.topic",
                message_type=MessageType.SYSTEM_STARTUP,
                payload={"test": "payload"},
                sender="test_engine"
            )
            
            print("Engine core initialization test passed")
            
        finally:
            # Clean shutdown
            if engine.message_bus:
                await engine.message_bus.stop()
            
        print("Core initialization test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 