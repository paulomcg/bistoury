"""
Test suite for CandlestickStrategyAgent.

Tests the integration of pattern analysis into the multi-agent framework.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4
from typing import Dict, List

from src.bistoury.agents.candlestick_strategy_agent import (
    CandlestickStrategyAgent,
    CandlestickStrategyConfig,
    StrategyPerformanceMetrics
)
from src.bistoury.agents.base import AgentState, AgentType, AgentHealth
from src.bistoury.agents.messaging import Message, MessageType, MessagePriority
from src.bistoury.models.signals import CandlestickData, TradingSignal, CandlestickPattern, PatternType, SignalDirection, SignalType
from src.bistoury.strategies.candlestick_models import PatternStrength, PatternQuality
from src.bistoury.strategies.narrative_generator import NarrativeStyle
from src.bistoury.strategies.signal_generator import GeneratedSignal
from src.bistoury.models.market_data import Timeframe


@pytest.fixture
def strategy_config():
    """Create a test strategy configuration."""
    return CandlestickStrategyConfig(
        symbols=["BTC", "ETH"],
        timeframes=["1m", "5m", "15m"],
        min_confidence_threshold=0.60,
        min_pattern_strength=PatternStrength.MODERATE,
        signal_expiry_minutes=10,
        max_signals_per_symbol=2,
        narrative_style=NarrativeStyle.TECHNICAL,
        max_data_buffer_size=100,
        data_cleanup_interval_seconds=60,
        health_check_interval_seconds=30,
        agent_name="test_strategy",
        agent_version="1.0.0"
    )


@pytest.fixture
def sample_candle():
    """Create a sample candlestick for testing."""
    return CandlestickData(
        symbol="BTC",
        timeframe="5m",
        timestamp=datetime.now(timezone.utc),
        open=Decimal("50000.00"),
        high=Decimal("50200.00"),
        low=Decimal("49800.00"),
        close=Decimal("50100.00"),
        volume=Decimal("100.50"),
        trade_count=150
    )


@pytest.fixture
def sample_pattern(sample_candle):
    """Create a sample candlestick pattern."""
    return CandlestickPattern(
        pattern_type=PatternType.HAMMER,
        symbol="BTC",
        timeframe="5m",
        start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
        end_time=datetime.now(timezone.utc),
        candles=[sample_candle],  # Add sample candle
        pattern_id="test_pattern_123",
        completion_price=Decimal("50100.00"),
        confidence=Decimal("0.75"),
        strength=PatternStrength.STRONG,
        direction="bullish",
        significance="Bullish reversal pattern",
        reliability=Decimal("0.80"),
        bullish_probability=Decimal("0.75"),
        bearish_probability=Decimal("0.25")
    )


@pytest.fixture
def sample_generated_signal(sample_pattern):
    """Create a sample trading signal."""
    from src.bistoury.models.signals import SignalDirection, SignalType
    from src.bistoury.models.market_data import Timeframe
    
    return TradingSignal(
        signal_id="test_signal_123",
        symbol="BTC",
        direction=SignalDirection.BUY,
        signal_type=SignalType.PATTERN,
        confidence=Decimal("75.0"),
        strength=Decimal("0.75"),
        price=Decimal("50100.00"),
        timeframe=Timeframe.FIVE_MINUTES,
        source="candlestick_strategy",
        reason="Hammer pattern detected with bullish reversal potential",
        timestamp=datetime.now(timezone.utc),
        target_price=Decimal("50600.00"),
        stop_loss=Decimal("49850.00")
    )


class TestCandlestickStrategyConfig:
    """Test configuration class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = CandlestickStrategyConfig()
        
        assert config.symbols == ["BTC", "ETH"]
        assert config.timeframes == ["1m", "5m", "15m"]
        assert config.min_confidence_threshold == 0.60
        assert config.min_pattern_strength == PatternStrength.MODERATE
        assert config.signal_expiry_minutes == 15
        assert config.max_signals_per_symbol == 3
        assert config.narrative_style == NarrativeStyle.TECHNICAL
        assert config.agent_name == "candlestick_strategy"
        assert config.agent_version == "1.0.0"
        
    def test_custom_configuration(self, strategy_config):
        """Test custom configuration values."""
        assert strategy_config.symbols == ["BTC", "ETH"]
        assert strategy_config.min_confidence_threshold == 0.60
        assert strategy_config.signal_expiry_minutes == 10
        assert strategy_config.agent_name == "test_strategy"


class TestStrategyPerformanceMetrics:
    """Test performance metrics class."""
    
    def test_default_metrics(self):
        """Test default metric values."""
        metrics = StrategyPerformanceMetrics()
        
        assert metrics.signals_generated == 0
        assert metrics.patterns_detected == 0
        assert metrics.high_confidence_signals == 0
        assert metrics.processing_latency_ms == 0.0
        assert metrics.data_messages_received == 0
        assert metrics.errors_count == 0
        assert metrics.uptime_seconds == 0.0
        assert metrics.last_signal_time is None
        assert metrics.average_confidence == 0.0
        
    def test_update_processing_latency(self):
        """Test processing latency update with exponential moving average."""
        metrics = StrategyPerformanceMetrics()
        
        # First update
        metrics.update_processing_latency(100.0)
        assert metrics.processing_latency_ms == 10.0  # 0.1 * 100 + 0.9 * 0
        
        # Second update
        metrics.update_processing_latency(200.0)
        expected = 0.1 * 200.0 + 0.9 * 10.0
        assert metrics.processing_latency_ms == expected


class TestCandlestickStrategyAgent:
    """Test the main strategy agent class."""
    
    def test_agent_initialization(self, strategy_config):
        """Test agent initialization."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        assert agent.agent_type == AgentType.STRATEGY
        assert agent.name == "test_strategy"
        assert agent.metadata.version == "1.0.0"
        assert len(agent.capabilities) == 5
        
        # Check capabilities
        capability_names = [cap.name for cap in agent.capabilities]
        assert "pattern_recognition" in capability_names
        assert "signal_generation" in capability_names
        assert "narrative_generation" in capability_names
        assert "multi_timeframe" in capability_names
        assert "real_time_processing" in capability_names
        
    def test_agent_initialization_default_config(self):
        """Test agent initialization with default config."""
        agent = CandlestickStrategyAgent()
        
        assert agent.name == "candlestick_strategy"
        assert agent.metadata.version == "1.0.0"
        assert agent.config.symbols == ["BTC", "ETH"]
        assert agent.config.min_confidence_threshold == 0.60
        
    def test_strategy_components_initialization(self, strategy_config):
        """Test strategy components are properly initialized."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        assert hasattr(agent, 'single_pattern_recognizer')
        assert hasattr(agent, 'multi_pattern_recognizer')
        assert hasattr(agent, 'timeframe_analyzer')
        assert hasattr(agent, 'pattern_scoring_engine')
        assert hasattr(agent, 'signal_generator')
        assert hasattr(agent, 'narrative_generator')
        
    @pytest.mark.asyncio
    async def test_agent_start_stop(self, strategy_config):
        """Test agent start and stop lifecycle."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # Mock BaseAgent methods (legacy subscription methods not used anymore)
        agent.create_task = Mock()  # Mock the create_task method
        
        # Test start
        await agent.start()
        assert agent.state == AgentState.RUNNING
        
        # Verify agent components were initialized and background tasks started
        assert hasattr(agent, 'single_pattern_recognizer')
        assert hasattr(agent, 'multi_pattern_recognizer')
        assert hasattr(agent, 'timeframe_analyzer')
        assert agent.create_task.call_count == 2  # Data cleanup + health monitoring
        
        # Test stop
        await agent.stop()
        assert agent.state == AgentState.STOPPED
        
        # Verify cleanup
        assert len(agent.market_data_buffer) == 0
        assert len(agent.active_signals) == 0
        
    @pytest.mark.asyncio
    async def test_setup_data_subscriptions_legacy(self, strategy_config):
        """Test that legacy subscription setup is skipped (no longer used)."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # The _setup_data_subscriptions method should be commented out and not called
        # This test verifies that the agent works without legacy subscriptions
        # Modern agents use MessageBus subscriptions instead
        
        # Verify agent can be created and has subscription tracking attributes
        assert hasattr(agent, 'subscribed_topics')
        assert len(agent.subscribed_topics) == 0  # No legacy subscriptions
        
    def test_add_to_buffer(self, strategy_config, sample_candle):
        """Test adding candlestick data to buffer."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # Add first candle
        agent._add_to_buffer("BTC", "5m", sample_candle)
        
        assert "BTC" in agent.market_data_buffer
        assert "5m" in agent.market_data_buffer["BTC"]
        assert len(agent.market_data_buffer["BTC"]["5m"]) == 1
        assert agent.market_data_buffer["BTC"]["5m"][0] == sample_candle
        
        # Add more candles to test buffer size limit
        for i in range(105):  # Exceed max buffer size
            agent._add_to_buffer("BTC", "5m", sample_candle)
            
        # Should maintain max buffer size
        assert len(agent.market_data_buffer["BTC"]["5m"]) == agent.config.max_data_buffer_size
        
    @pytest.mark.asyncio
    async def test_has_sufficient_data(self, strategy_config, sample_candle):
        """Test checking if sufficient data is available."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # No data
        assert not await agent._has_sufficient_data("BTC")
        
        # Add insufficient data (less than 20 candles)
        for i in range(10):
            agent._add_to_buffer("BTC", "5m", sample_candle)
        assert not await agent._has_sufficient_data("BTC")
        
        # Add sufficient data for one timeframe but not others
        for i in range(15):
            agent._add_to_buffer("BTC", "5m", sample_candle)
        assert not await agent._has_sufficient_data("BTC")
        
        # Add sufficient data for all timeframes
        for timeframe in strategy_config.timeframes:
            for i in range(25):
                agent._add_to_buffer("BTC", timeframe, sample_candle)
        assert await agent._has_sufficient_data("BTC")
        
    @pytest.mark.asyncio
    async def test_handle_market_data_message(self, strategy_config, sample_candle):
        """Test handling market data messages."""
        agent = CandlestickStrategyAgent(strategy_config)
        agent._process_candlestick_data = AsyncMock()
        agent._process_volume_data = AsyncMock()
        
        # Test candlestick data message
        candle_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            topic="market_data.candle.BTC.5m",
            payload={
                "data_type": "candle",
                "symbol": "BTC",
                "timeframe": "5m",
                "candle_data": sample_candle.model_dump()
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        await agent._handle_market_data(candle_message)
        agent._process_candlestick_data.assert_called_once_with("BTC", "5m", candle_message.payload)
        
        # Test volume data message
        volume_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            topic="market_data.volume.BTC",
            payload={
                "data_type": "volume",
                "symbol": "BTC",
                "volume_data": {"volume": 1000}
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        await agent._handle_market_data(volume_message)
        agent._process_volume_data.assert_called_once_with("BTC", volume_message.payload)
        
    @pytest.mark.asyncio
    async def test_process_candlestick_data(self, strategy_config, sample_candle):
        """Test processing candlestick data."""
        agent = CandlestickStrategyAgent(strategy_config)
        agent._analyze_patterns = AsyncMock()
        
        # Mock having sufficient data
        agent._has_sufficient_data = AsyncMock(return_value=True)
        
        payload = {
            "candle_data": sample_candle.model_dump()
        }
        
        await agent._process_candlestick_data("BTC", "5m", payload)
        
        # Check data was added to buffer
        assert "BTC" in agent.market_data_buffer
        assert "5m" in agent.market_data_buffer["BTC"]
        assert len(agent.market_data_buffer["BTC"]["5m"]) == 1
        
        # Check pattern analysis was triggered
        agent._analyze_patterns.assert_called_once_with("BTC")
        
    @pytest.mark.asyncio
    async def test_publish_signal(self, strategy_config, sample_generated_signal):
        """Test publishing trading signals."""
        agent = CandlestickStrategyAgent(strategy_config)
        agent.publish_message = AsyncMock()
        
        await agent._publish_signal("BTC", sample_generated_signal)
        
        # Check message was published
        agent.publish_message.assert_called_once()
        call_args = agent.publish_message.call_args
        
        assert call_args[1]["message_type"] == MessageType.SIGNAL_GENERATED
        assert call_args[1]["topic"] == "signals.BTC"
        assert call_args[1]["priority"] == MessagePriority.HIGH
        
        payload = call_args[1]["payload"]
        assert payload.symbol == "BTC"
        assert payload.metadata["signal_id"] == sample_generated_signal.signal_id
        assert payload.confidence == float(sample_generated_signal.confidence) / 100.0  # Converted to 0-1 range
        
        # Check signal was tracked
        assert "BTC" in agent.active_signals
        assert len(agent.active_signals["BTC"]) == 1
        assert agent.active_signals["BTC"][0] == sample_generated_signal
        
        # Check performance metrics were updated
        assert agent.performance_metrics.last_signal_time is not None
        
    @pytest.mark.asyncio
    async def test_handle_configuration_update(self, strategy_config):
        """Test handling configuration updates."""
        agent = CandlestickStrategyAgent(strategy_config)
        original_threshold = agent.config.min_confidence_threshold
        
        # Mock re-initialization
        agent._initialize_strategy_components = Mock()
        
        config_message = Message(
            type=MessageType.SYSTEM_CONFIG_UPDATE,
            sender="orchestrator",
            topic="config.update",
            payload={
                "config_updates": {
                    "min_confidence_threshold": 0.75,
                    "max_signals_per_symbol": 5
                }
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        await agent._handle_configuration_update(config_message)
        
        # Check configuration was updated
        assert agent.config.min_confidence_threshold == 0.75
        assert agent.config.max_signals_per_symbol == 5
        
        # Check components were re-initialized
        agent._initialize_strategy_components.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_handle_health_check_request(self, strategy_config):
        """Test handling health check requests."""
        agent = CandlestickStrategyAgent(strategy_config)
        agent.publish_message = AsyncMock()
        
        # Set some performance metrics
        agent.performance_metrics.signals_generated = 10
        agent.performance_metrics.patterns_detected = 15
        agent.performance_metrics.errors_count = 2
        
        health_message = Message(
            type=MessageType.SYSTEM_HEALTH_CHECK,
            sender="orchestrator",
            topic="health.request",
            payload={},
            timestamp=datetime.now(timezone.utc)
        )
        
        await agent._handle_health_check_request(health_message)
        
        # Check response was published
        agent.publish_message.assert_called_once()
        call_args = agent.publish_message.call_args
        
        assert call_args[1]["message_type"] == MessageType.AGENT_HEALTH_UPDATE
        assert call_args[1]["topic"] == f"health.{agent.agent_id}"
        
        payload = call_args[1]["payload"]
        assert payload["agent_id"] == agent.agent_id
        assert payload["status"] == agent.state.name
        assert payload["performance_metrics"]["signals_generated"] == 10
        assert payload["performance_metrics"]["patterns_detected"] == 15
        assert payload["performance_metrics"]["errors_count"] == 2
        
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, strategy_config, sample_candle):
        """Test cleaning up old data."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # Add lots of data to trigger cleanup
        for i in range(150):
            agent._add_to_buffer("BTC", "5m", sample_candle)
            
        await agent._cleanup_old_data()
        
        # Check buffer was cleaned up
        assert len(agent.market_data_buffer["BTC"]["5m"]) == 100  # Cleanup limit
        
    @pytest.mark.asyncio
    async def test_update_health_metrics(self, strategy_config, sample_generated_signal):
        """Test updating health metrics."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # Add some signals to track
        agent.active_signals["BTC"] = [sample_generated_signal]
        agent.performance_metrics.signals_generated = 1
        
        await agent._update_health_metrics()
        
        # Should calculate average confidence from active signals
        assert agent.performance_metrics.average_confidence > 0
        
    def test_get_strategy_statistics(self, strategy_config):
        """Test getting strategy statistics."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # Set some test data
        agent.performance_metrics.signals_generated = 5
        agent.performance_metrics.patterns_detected = 8
        agent.market_data_buffer["BTC"] = {"5m": []}
        agent.active_signals["BTC"] = [Mock(), Mock()]
        agent.subscribed_topics = {"topic1", "topic2", "topic3"}
        
        stats = agent.get_strategy_statistics()
        
        # Check structure and content
        assert "agent_info" in stats
        assert "performance" in stats
        assert "configuration" in stats
        assert "active_data" in stats
        
        assert stats["agent_info"]["agent_id"] == agent.agent_id
        assert stats["agent_info"]["agent_type"] == "strategy"
        assert stats["performance"]["signals_generated"] == 5
        assert stats["performance"]["patterns_detected"] == 8
        assert stats["active_data"]["symbols_tracking"] == ["BTC"]
        assert stats["active_data"]["active_signals_count"] == 2
        assert stats["active_data"]["subscribed_topics_count"] == 3
        
    @pytest.mark.asyncio
    async def test_message_processing_performance(self, strategy_config):
        """Test message processing updates performance metrics."""
        agent = CandlestickStrategyAgent(strategy_config)
        agent._handle_market_data = AsyncMock()
        
        initial_latency = agent.performance_metrics.processing_latency_ms
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            topic="market_data.candle.BTC.5m",
            payload={"data_type": "candle", "symbol": "BTC"},
            timestamp=datetime.now(timezone.utc)
        )
        
        await agent.handle_message(message)
        
        # Check metrics were updated
        assert agent.performance_metrics.data_messages_received == 1
        # Latency should be updated (though might be very small)
        assert agent.performance_metrics.processing_latency_ms >= initial_latency
        
    @pytest.mark.asyncio
    async def test_error_handling_in_message_processing(self, strategy_config):
        """Test error handling during message processing."""
        agent = CandlestickStrategyAgent(strategy_config)
        
        # Mock method to raise exception
        agent._handle_market_data = AsyncMock(side_effect=Exception("Test error"))
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            topic="market_data.candle.BTC.5m",
            payload={},
            timestamp=datetime.now(timezone.utc)
        )
        
        # Should not raise exception
        await agent.handle_message(message)
        
        # Error count should be incremented
        assert agent.performance_metrics.errors_count == 1
        
    @pytest.mark.asyncio
    async def test_signal_limit_enforcement(self, strategy_config, sample_generated_signal):
        """Test that signal limits per symbol are enforced."""
        agent = CandlestickStrategyAgent(strategy_config)
        agent.publish_message = AsyncMock()
        
        # Mock narrative generation
        mock_narrative = Mock()
        mock_narrative.executive_summary = "Test"
        mock_narrative.pattern_analysis = "Test"
        mock_narrative.risk_assessment = "Test"
        mock_narrative.entry_strategy = "Test"
        mock_narrative.exit_strategy = "Test"
        agent.narrative_generator.generate_signal_narrative = Mock(return_value=mock_narrative)
        
        # Publish more signals than the limit
        for i in range(5):  # Limit is 2 in test config
            await agent._publish_signal("BTC", sample_generated_signal)
            
        # Should only keep the limit number of signals
        assert len(agent.active_signals["BTC"]) == strategy_config.max_signals_per_symbol


@pytest.mark.asyncio
async def test_integration_scenario(strategy_config, sample_candle):
    """Test a complete integration scenario."""
    agent = CandlestickStrategyAgent(strategy_config)
    
    # Mock external dependencies
    agent.subscribe_to_topic = AsyncMock()
    agent.unsubscribe_from_topic = AsyncMock()
    agent.publish_message = AsyncMock()
    agent.add_background_task = Mock()
    
    # Mock analysis components to return patterns
    mock_pattern = Mock()
    mock_pattern.confidence = Decimal("80.0")
    mock_pattern.reliability = Decimal("0.8")
    mock_pattern.pattern_type = Mock()
    mock_pattern.pattern_type.value = "hammer"
    mock_pattern.timeframe = Timeframe.FIVE_MINUTES
    mock_pattern.bullish_probability = Decimal("80.0")
    mock_pattern.bearish_probability = Decimal("20.0")
    
    mock_analysis_result = Mock()
    mock_analysis_result.total_patterns_detected = 2  # Integer, not Mock
    mock_analysis_result.data_quality_score = Decimal("85")  # Decimal, not Mock
    mock_analysis_result.meets_latency_requirement = True  # Boolean, not Mock
    mock_analysis_result.single_patterns = {
        Timeframe.FIVE_MINUTES: [mock_pattern]
    }
    mock_analysis_result.multi_patterns = {
        Timeframe.FIVE_MINUTES: [mock_pattern]
    }
    mock_analysis_result.context = Mock()
    
    agent.timeframe_analyzer.analyze = AsyncMock(return_value=mock_analysis_result)
    
    # Mock pattern scoring
    mock_scored_pattern = Mock()
    mock_scored_pattern.confidence = 0.80
    mock_scored_pattern.pattern_strength = PatternStrength.STRONG
    
    agent.pattern_scoring_engine.score_pattern = AsyncMock(return_value=mock_scored_pattern)
    
    # Mock signal generation
    mock_signal = Mock()
    mock_signal.signal_id = "test_signal"
    mock_signal.symbol = "BTC"
    mock_signal.direction = Mock()
    mock_signal.direction.value = "buy"
    mock_signal.signal_type = Mock()
    mock_signal.signal_type.value = "pattern"
    mock_signal.confidence = Decimal("75.0")
    mock_signal.strength = Decimal("0.75")
    mock_signal.price = Decimal("50100.00")
    mock_signal.reason = "Test signal"
    mock_signal.stop_loss = Decimal("49900.00")
    mock_signal.target_price = Decimal("50300.00")
    mock_signal.expiry = datetime.now(timezone.utc) + timedelta(minutes=15)
    mock_signal.timeframe = Timeframe.FIVE_MINUTES
    mock_signal.timestamp = datetime.now(timezone.utc)
    mock_signal.metadata = {}  # Add empty metadata dict
    
    agent._create_trading_signal = AsyncMock(return_value=mock_signal)
    
    # Mock narrative generation
    mock_narrative = Mock()
    mock_narrative.executive_summary = "Test summary"
    mock_narrative.pattern_analysis = "Test analysis"
    mock_narrative.risk_assessment = "Test risk"
    mock_narrative.entry_strategy = "Test entry"
    mock_narrative.exit_strategy = "Test exit"
    
    agent.narrative_generator.generate_signal_narrative = Mock(return_value=mock_narrative)
    
    # Start the agent
    await agent.start()
    
    # Add sufficient data
    for timeframe in strategy_config.timeframes:
        for i in range(25):
            agent._add_to_buffer("BTC", timeframe, sample_candle)
            
    # Process candlestick data (should trigger analysis and signal generation)
    payload = {"candle_data": sample_candle.model_dump()}
    await agent._process_candlestick_data("BTC", "5m", payload)
    
    # Verify the flow worked
    agent.timeframe_analyzer.analyze.assert_called_once()
    agent.pattern_scoring_engine.score_pattern.assert_called()
    agent._create_trading_signal.assert_called()
    agent.publish_message.assert_called()
    
    # Check performance metrics were updated
    assert agent.performance_metrics.signals_generated == 1
    assert agent.performance_metrics.patterns_detected == 2  # 1 single + 1 multi pattern
    assert agent.performance_metrics.high_confidence_signals == 1  # Since signal quality > 0.75
    
    # Stop the agent
    await agent.stop()
    
    assert agent.state == AgentState.STOPPED 