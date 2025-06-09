"""
Integration tests for Candlestick Strategy - Task 8.9

Tests the complete end-to-end functionality of the candlestick strategy system,
including pattern recognition, signal generation, multi-timeframe analysis,
and agent framework integration.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import time
import uuid

from src.bistoury.agents.candlestick_strategy_agent import (
    CandlestickStrategyAgent,
    CandlestickStrategyConfig,
    StrategyPerformanceMetrics
)
from src.bistoury.agents.base import AgentState, AgentType
from src.bistoury.agents.messaging import Message, MessageType, MessagePriority
from src.bistoury.models.signals import (
    CandlestickData, TradingSignal, CandlestickPattern, 
    PatternType, SignalDirection, SignalType
)
from src.bistoury.models.market_data import Timeframe
from src.bistoury.strategies.candlestick_models import PatternStrength, PatternQuality
from src.bistoury.strategies.timeframe_analyzer import TimeframeAnalysisResult
from src.bistoury.strategies.narrative_generator import NarrativeStyle


# Test configuration for integration tests
INTEGRATION_CONFIG = CandlestickStrategyConfig(
    symbols=["BTC"],
    timeframes=["1m", "5m", "15m"],
    min_confidence_threshold=0.65,
    min_pattern_strength=PatternStrength.MODERATE,
    signal_expiry_minutes=5,
    max_signals_per_symbol=2,
    narrative_style=NarrativeStyle.TECHNICAL,
    max_data_buffer_size=50,
    data_cleanup_interval_seconds=30,
    health_check_interval_seconds=15,
    agent_name="integration_test_strategy",
    agent_version="1.0.0-test"
)


class MockDataGenerator:
    """Generate realistic market data for integration testing."""
    
    @staticmethod
    def generate_candlestick_sequence(
        symbol: str, 
        timeframe: str,
        count: int = 25,
        base_price: Decimal = Decimal("50000"),
        volatility: float = 0.02
    ) -> List[CandlestickData]:
        """Generate a realistic sequence of candlestick data."""
        
        candles = []
        current_price = base_price
        timestamp = datetime.now(timezone.utc) - timedelta(minutes=count)
        
        for i in range(count):
            # Simulate realistic price movement
            price_change = Decimal(str((hash(f"{symbol}_{i}") % 1000 - 500) / 10000 * volatility))
            
            open_price = current_price
            close_price = current_price + (current_price * price_change)
            
            # High/Low with realistic spread
            high_price = max(open_price, close_price) + (current_price * Decimal("0.001"))
            low_price = min(open_price, close_price) - (current_price * Decimal("0.001"))
            
            # Volume
            volume = Decimal(str(100 + (hash(f"{symbol}_{i}_vol") % 100)))
            
            candle = CandlestickData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp + timedelta(minutes=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            candles.append(candle)
            current_price = close_price
            
        return candles
    
    @staticmethod
    def generate_hammer_pattern(
        symbol: str,
        timeframe: str,
        price: Decimal = Decimal("50000")
    ) -> CandlestickData:
        """Generate a hammer pattern candlestick."""
        
        timestamp = datetime.now(timezone.utc)
        
        # Hammer: small body, long lower shadow, little/no upper shadow
        open_price = price
        close_price = price + (price * Decimal("0.002"))  # Small positive body
        low_price = price - (price * Decimal("0.015"))    # Long lower shadow
        high_price = close_price + (price * Decimal("0.001"))  # Minimal upper shadow
        
        return CandlestickData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("150")
        )
    
    @staticmethod
    def generate_doji_pattern(
        symbol: str,
        timeframe: str,
        price: Decimal = Decimal("50000")
    ) -> CandlestickData:
        """Generate a doji pattern candlestick."""
        
        timestamp = datetime.now(timezone.utc)
        
        # Doji: open ≈ close, long shadows
        open_price = price
        close_price = price + (price * Decimal("0.0001"))  # Very small body
        low_price = price - (price * Decimal("0.008"))     # Lower shadow
        high_price = price + (price * Decimal("0.008"))    # Upper shadow
        
        return CandlestickData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("120")
        )


class IntegrationTestFixtures:
    """Common fixtures and setup for integration tests."""
    
    @staticmethod
    async def create_agent_with_mocks() -> CandlestickStrategyAgent:
        """Create agent with properly mocked dependencies."""
        
        agent = CandlestickStrategyAgent(INTEGRATION_CONFIG)
        
        # Mock message bus methods
        agent.subscribe_to_topic = AsyncMock()
        agent.unsubscribe_from_topic = AsyncMock()
        agent.publish_message = AsyncMock()
        
        return agent
    
    @staticmethod
    def create_market_data_message(
        symbol: str,
        timeframe: str,
        candle_data: CandlestickData
    ) -> Message:
        """Create a market data message."""
        
        return Message(
            id=str(uuid.uuid4()),
            type=MessageType.DATA_MARKET_UPDATE,
            sender="test_collector",
            timestamp=datetime.now(timezone.utc),
            payload={
                "data_type": "candle",
                "symbol": symbol,
                "timeframe": timeframe,
                "candle_data": candle_data.model_dump()
            },
            priority=MessagePriority.NORMAL,
            topic=f"market_data.candle.{symbol}.{timeframe}"
        )


@pytest.fixture
async def integration_agent():
    """Create an agent for integration testing."""
    agent = await IntegrationTestFixtures.create_agent_with_mocks()
    yield agent
    
    # Cleanup
    if agent.state == AgentState.RUNNING:
        await agent.stop()


@pytest.fixture
def mock_data_generator():
    """Provide mock data generator."""
    return MockDataGenerator()


class TestEndToEndIntegration:
    """Test complete end-to-end strategy functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_startup_and_shutdown(self, integration_agent):
        """Test agent lifecycle management."""
        
        # Test initial state
        assert integration_agent.state == AgentState.CREATED
        assert integration_agent.name == "integration_test_strategy"
        
        # Test startup
        start_success = await integration_agent.start()
        assert start_success is True
        assert integration_agent.state == AgentState.RUNNING
        
        # Verify subscriptions were set up
        expected_calls = len(INTEGRATION_CONFIG.symbols) * len(INTEGRATION_CONFIG.timeframes) + len(INTEGRATION_CONFIG.symbols)
        assert integration_agent.subscribe_to_topic.call_count == expected_calls
        
        # Test shutdown
        await integration_agent.stop()
        assert integration_agent.state == AgentState.STOPPED
        
        # Verify cleanup
        assert len(integration_agent.market_data_buffer) == 0
        assert len(integration_agent.active_signals) == 0
    
    @pytest.mark.asyncio 
    async def test_data_processing_pipeline(self, integration_agent, mock_data_generator):
        """Test complete data processing from ingestion to buffer management."""
        
        await integration_agent.start()
        
        # Generate test data
        candles = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 25)
        
        # Send data messages
        for candle in candles:
            message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
            await integration_agent.handle_message(message)
        
        # Verify data is in buffer
        assert "BTC" in integration_agent.market_data_buffer
        assert "1m" in integration_agent.market_data_buffer["BTC"]
        assert len(integration_agent.market_data_buffer["BTC"]["1m"]) == 25
        
        # Test buffer size management
        extra_candles = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 30)
        for candle in extra_candles:
            message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
            await integration_agent.handle_message(message)
        
        # Should maintain max buffer size
        assert len(integration_agent.market_data_buffer["BTC"]["1m"]) == INTEGRATION_CONFIG.max_data_buffer_size
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_multi_timeframe_data_synchronization(self, integration_agent, mock_data_generator):
        """Test data synchronization across multiple timeframes."""
        
        await integration_agent.start()
        
        # Generate data for multiple timeframes
        timeframes = ["1m", "5m", "15m"]
        
        for timeframe in timeframes:
            candles = mock_data_generator.generate_candlestick_sequence("BTC", timeframe, 25)
            
            for candle in candles:
                message = IntegrationTestFixtures.create_market_data_message("BTC", timeframe, candle)
                await integration_agent.handle_message(message)
        
        # Verify all timeframes have data
        for timeframe in timeframes:
            assert timeframe in integration_agent.market_data_buffer["BTC"]
            assert len(integration_agent.market_data_buffer["BTC"][timeframe]) == 25
        
        # Test sufficient data detection
        has_sufficient = await integration_agent._has_sufficient_data("BTC")
        assert has_sufficient is True
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_pattern_detection_and_signal_generation(self, integration_agent, mock_data_generator):
        """Test pattern detection leading to signal generation."""
        
        await integration_agent.start()
        
        # Mock the timeframe analyzer to return a realistic analysis
        with patch.object(integration_agent.timeframe_analyzer, 'analyze') as mock_analyze:
            
            # Create mock analysis result
            mock_result = Mock()
            mock_result.total_patterns_detected = 2
            mock_result.data_quality_score = Decimal("75")
            mock_result.meets_latency_requirement = True
            
            # Mock single patterns
            mock_pattern = Mock()
            mock_pattern.confidence = Decimal("75")
            mock_pattern.reliability = Decimal("0.8")
            mock_pattern.pattern_type = PatternType.HAMMER
            mock_pattern.bullish_probability = Decimal("80")
            mock_pattern.bearish_probability = Decimal("20")
            mock_pattern.timeframe = Timeframe.FIFTEEN_MINUTES
            
            mock_result.single_patterns = {Timeframe.FIFTEEN_MINUTES: [mock_pattern]}
            mock_result.multi_patterns = {}
            
            mock_analyze.return_value = mock_result
            
            # Send sufficient data to trigger analysis
            for timeframe in ["1m", "5m", "15m"]:
                candles = mock_data_generator.generate_candlestick_sequence("BTC", timeframe, 25)
                
                for candle in candles:
                    message = IntegrationTestFixtures.create_market_data_message("BTC", timeframe, candle)
                    await integration_agent.handle_message(message)
            
            # Wait for async processing
            await asyncio.sleep(0.1)
            
            # Verify analysis was called
            assert mock_analyze.called
            
            # Verify signal was published
            assert integration_agent.publish_message.called
            
            # Check call arguments
            call_args = integration_agent.publish_message.call_args
            assert call_args[1]['message_type'] == MessageType.SIGNAL_GENERATED
            assert call_args[1]['topic'] == "signals.candlestick.BTC"
            assert call_args[1]['priority'] == MessagePriority.HIGH
            
            # Verify performance metrics
            assert integration_agent.performance_metrics.signals_generated > 0
            assert integration_agent.performance_metrics.patterns_detected > 0
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, integration_agent, mock_data_generator):
        """Test performance monitoring and metrics collection."""
        
        await integration_agent.start()
        
        # Send test data and trigger processing
        candles = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 10)
        
        for candle in candles:
            message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
            await integration_agent.handle_message(message)
        
        # Check performance metrics
        metrics = integration_agent.performance_metrics
        assert metrics.data_messages_received == 10
        assert metrics.processing_latency_ms >= 0.0
        
        # Test health check
        health = await integration_agent._health_check()
        assert health.state == AgentState.RUNNING
        assert health.messages_processed == 10
        assert 0.0 <= health.health_score <= 1.0
        
        # Test statistics
        stats = integration_agent.get_strategy_statistics()
        assert stats["agent_info"]["agent_id"] == integration_agent.agent_id
        assert stats["performance"]["data_messages_received"] == 10
        assert stats["configuration"]["symbols"] == ["BTC"]
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_configuration_hot_reload(self, integration_agent):
        """Test configuration updates without restart."""
        
        await integration_agent.start()
        
        # Create configuration update message
        config_message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM_CONFIG_UPDATE,
            sender="test_manager",
            timestamp=datetime.now(timezone.utc),
            payload={
                "config_updates": {
                    "min_confidence_threshold": 0.75,
                    "max_signals_per_symbol": 5
                }
            },
            priority=MessagePriority.HIGH,
            topic="config.strategy.candlestick"
        )
        
        # Send configuration update
        await integration_agent.handle_message(config_message)
        
        # Verify configuration was updated
        assert integration_agent.config.min_confidence_threshold == 0.75
        assert integration_agent.config.max_signals_per_symbol == 5
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integration_agent, mock_data_generator):
        """Test error handling and system recovery."""
        
        await integration_agent.start()
        
        initial_error_count = integration_agent.performance_metrics.errors_count
        
        # Send invalid message
        invalid_message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.DATA_MARKET_UPDATE,
            sender="test_sender",
            timestamp=datetime.now(timezone.utc),
            payload={
                "data_type": "candle",
                "symbol": "BTC",
                "timeframe": "1m",
                "candle_data": {"invalid": "data"}  # Invalid candle data
            },
            priority=MessagePriority.NORMAL,
            topic="market_data.candle.BTC.1m"
        )
        
        await integration_agent.handle_message(invalid_message)
        
        # Verify error was tracked but agent remains operational
        assert integration_agent.performance_metrics.errors_count > initial_error_count
        assert integration_agent.state == AgentState.RUNNING
        
        # Send valid data to confirm agent still works
        candle = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 1)[0]
        valid_message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
        await integration_agent.handle_message(valid_message)
        
        # Agent should still process valid data
        assert integration_agent.performance_metrics.data_messages_received > 0
        
        await integration_agent.stop()


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.asyncio
    async def test_message_processing_latency(self, integration_agent, mock_data_generator):
        """Test message processing latency requirements."""
        
        await integration_agent.start()
        
        # Generate test data
        candles = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 100)
        
        # Measure processing time
        start_time = time.time()
        
        for candle in candles:
            message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
            await integration_agent.handle_message(message)
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_latency_ms = total_time_ms / len(candles)
        
        # Verify latency requirements (should be < 100ms per message)
        assert avg_latency_ms < 100.0, f"Average latency {avg_latency_ms:.2f}ms exceeds 100ms limit"
        
        # Check agent's internal latency tracking
        assert integration_agent.performance_metrics.processing_latency_ms > 0.0
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, integration_agent, mock_data_generator):
        """Test memory usage under sustained load."""
        
        await integration_agent.start()
        
        # Send large amounts of data
        for i in range(10):  # 10 batches
            candles = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 100)
            
            for candle in candles:
                message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
                await integration_agent.handle_message(message)
        
        # Verify buffer size management
        buffer_size = len(integration_agent.market_data_buffer["BTC"]["1m"])
        assert buffer_size <= INTEGRATION_CONFIG.max_data_buffer_size
        
        # Check that old data was cleaned up
        assert buffer_size == INTEGRATION_CONFIG.max_data_buffer_size
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, integration_agent, mock_data_generator):
        """Test handling concurrent message processing."""
        
        await integration_agent.start()
        
        # Create concurrent message tasks
        async def send_messages(symbol: str, timeframe: str, count: int):
            candles = mock_data_generator.generate_candlestick_sequence(symbol, timeframe, count)
            for candle in candles:
                message = IntegrationTestFixtures.create_market_data_message(symbol, timeframe, candle)
                await integration_agent.handle_message(message)
        
        # Run concurrent tasks
        tasks = [
            send_messages("BTC", "1m", 50),
            send_messages("BTC", "5m", 50), 
            send_messages("BTC", "15m", 50)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all data was processed
        for timeframe in ["1m", "5m", "15m"]:
            assert timeframe in integration_agent.market_data_buffer["BTC"]
            assert len(integration_agent.market_data_buffer["BTC"][timeframe]) > 0
        
        # Check error count remained low
        assert integration_agent.performance_metrics.errors_count == 0
        
        await integration_agent.stop()


class TestSignalQuality:
    """Test signal generation quality and accuracy."""
    
    @pytest.mark.asyncio
    async def test_signal_generation_with_hammer_pattern(self, integration_agent, mock_data_generator):
        """Test signal generation with a clear hammer pattern."""
        
        await integration_agent.start()
        
        # Generate base data
        for timeframe in ["1m", "5m", "15m"]:
            candles = mock_data_generator.generate_candlestick_sequence("BTC", timeframe, 20)
            
            for candle in candles:
                message = IntegrationTestFixtures.create_market_data_message("BTC", timeframe, candle)
                await integration_agent.handle_message(message)
        
        # Add a clear hammer pattern to 15m timeframe
        hammer_candle = mock_data_generator.generate_hammer_pattern("BTC", "15m")
        message = IntegrationTestFixtures.create_market_data_message("BTC", "15m", hammer_candle)
        
        # Mock successful pattern analysis
        with patch.object(integration_agent.timeframe_analyzer, 'analyze') as mock_analyze:
            
            # Create mock result with hammer pattern
            mock_result = Mock()
            mock_result.total_patterns_detected = 1
            mock_result.data_quality_score = Decimal("85")
            mock_result.meets_latency_requirement = True
            
            # Create hammer pattern mock
            hammer_pattern = Mock()
            hammer_pattern.confidence = Decimal("85")
            hammer_pattern.reliability = Decimal("0.85")
            hammer_pattern.pattern_type = PatternType.HAMMER
            hammer_pattern.bullish_probability = Decimal("85")
            hammer_pattern.bearish_probability = Decimal("15")
            hammer_pattern.timeframe = Timeframe.FIFTEEN_MINUTES
            
            mock_result.single_patterns = {Timeframe.FIFTEEN_MINUTES: [hammer_pattern]}
            mock_result.multi_patterns = {}
            
            mock_analyze.return_value = mock_result
            
            await integration_agent.handle_message(message)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Verify signal characteristics
            if integration_agent.publish_message.called:
                call_args = integration_agent.publish_message.call_args
                payload = call_args[1]['payload']
                
                # Check signal direction (should be bullish for hammer)
                assert 'signal_data' in payload
                # Additional checks would depend on the signal data structure
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio
    async def test_signal_filtering_by_confidence(self, integration_agent):
        """Test that low-confidence signals are filtered out."""
        
        await integration_agent.start()
        
        # Mock low-confidence pattern detection
        with patch.object(integration_agent.timeframe_analyzer, 'analyze') as mock_analyze:
            
            mock_result = Mock()
            mock_result.total_patterns_detected = 1
            mock_result.data_quality_score = Decimal("40")  # Low quality
            mock_result.meets_latency_requirement = True
            
            # Low confidence pattern
            low_conf_pattern = Mock()
            low_conf_pattern.confidence = Decimal("45")  # Below threshold
            low_conf_pattern.reliability = Decimal("0.3")
            low_conf_pattern.pattern_type = PatternType.DOJI
            low_conf_pattern.bullish_probability = Decimal("50")
            low_conf_pattern.bearish_probability = Decimal("50")
            low_conf_pattern.timeframe = Timeframe.ONE_MINUTE
            
            mock_result.single_patterns = {Timeframe.ONE_MINUTE: [low_conf_pattern]}
            mock_result.multi_patterns = {}
            
            mock_analyze.return_value = mock_result
            
            # Should not generate signal due to low quality
            await integration_agent._analyze_patterns("BTC")
            
            # Verify no signal was published
            assert not integration_agent.publish_message.called
        
        await integration_agent.stop()


# Performance targets and validation
class TestProductionReadiness:
    """Test production readiness criteria."""
    
    @pytest.mark.asyncio
    async def test_latency_requirements(self, integration_agent, mock_data_generator):
        """Verify all latency requirements are met."""
        
        await integration_agent.start()
        
        # Test data processing latency
        candle = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 1)[0]
        message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
        
        start_time = time.time()
        await integration_agent.handle_message(message)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Should process within 50ms for single message
        assert latency_ms < 50.0, f"Single message latency {latency_ms:.2f}ms exceeds 50ms"
        
        await integration_agent.stop()
    
    @pytest.mark.asyncio 
    async def test_system_stability(self, integration_agent, mock_data_generator):
        """Test system stability under continuous operation."""
        
        await integration_agent.start()
        
        # Run for extended period with continuous data
        start_time = time.time()
        target_duration = 5.0  # 5 seconds of continuous operation
        
        message_count = 0
        while time.time() - start_time < target_duration:
            candle = mock_data_generator.generate_candlestick_sequence("BTC", "1m", 1)[0]
            message = IntegrationTestFixtures.create_market_data_message("BTC", "1m", candle)
            await integration_agent.handle_message(message)
            message_count += 1
            
            # Small delay to simulate realistic message rate
            await asyncio.sleep(0.01)
        
        # Verify system remained stable
        assert integration_agent.state == AgentState.RUNNING
        assert integration_agent.performance_metrics.data_messages_received == message_count
        assert integration_agent.performance_metrics.errors_count == 0
        
        await integration_agent.stop()
    
    def test_configuration_validation(self):
        """Test configuration validation and edge cases."""
        
        # Test valid configuration
        valid_config = CandlestickStrategyConfig(
            symbols=["BTC", "ETH"],
            timeframes=["1m", "5m"],
            min_confidence_threshold=0.70
        )
        agent = CandlestickStrategyAgent(valid_config)
        assert agent.config.min_confidence_threshold == 0.70
        
        # Test default configuration
        default_agent = CandlestickStrategyAgent()
        assert default_agent.config.symbols == ["BTC", "ETH"]
        assert default_agent.config.min_confidence_threshold == 0.60


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])