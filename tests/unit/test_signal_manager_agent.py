"""
Test suite for Signal Manager Agent - Task 9.5

Tests the integration of SignalManager into the multi-agent framework.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.bistoury.agents.signal_manager_agent import SignalManagerAgent
from src.bistoury.agents.base import AgentState, AgentType
from src.bistoury.agents.messaging import MessageBus, MessageFilter
from src.bistoury.models.agent_messages import (
    Message, MessageType, MessagePriority, TradingSignalPayload,
    AggregatedSignalPayload, SystemEventPayload
)
from src.bistoury.models.signals import TradingSignal, SignalDirection, RiskLevel
from src.bistoury.models.market_data import Timeframe
from src.bistoury.signal_manager.signal_manager import SignalManagerConfiguration
from src.bistoury.signal_manager.models import AggregatedSignal, SignalQuality, SignalQualityGrade
from src.bistoury.signal_manager.narrative_buffer import NarrativeBufferConfig


class TestSignalManagerAgent:
    """Test suite for SignalManagerAgent class."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic agent configuration."""
        return {
            'subscribe_to_strategies': True,
            'publish_aggregated_signals': True,
            'auto_register_strategies': True
        }
    
    @pytest.fixture
    def signal_config(self):
        """Signal Manager configuration."""
        return SignalManagerConfiguration(
            preserve_narratives=False,  # Simplified for bootstrap strategy
            confidence_threshold=60.0,
            quality_threshold=70.0
        )
    
    @pytest.fixture
    def narrative_config(self):
        """Narrative buffer configuration."""
        return NarrativeBufferConfig(
            max_timeline_length=1000,
            max_memory_narratives=100,
            archive_after_hours=24,
            enable_continuity_tracking=False,
            background_compression=False
        )
    
    @pytest.fixture
    def mock_message_bus(self):
        """Mock message bus."""
        bus = Mock(spec=MessageBus)
        bus.subscribe = AsyncMock()
        bus.unsubscribe = AsyncMock()
        bus.publish = AsyncMock(return_value=True)
        return bus
    
    @pytest.fixture
    def sample_trading_signal(self):
        """Sample trading signal."""
        return TradingSignal(
            symbol="BTC-USD",
            signal_type="bullish_engulfing",
            direction=SignalDirection.BUY,
            confidence=0.75,
            strength=0.8,
            timeframe=Timeframe.FIFTEEN_MINUTES,
            strategy="candlestick_strategy",
            reasoning="Strong bullish engulfing pattern with high volume confirmation",
            metadata={"volume_ratio": 1.5, "pattern_quality": "high"}
        )
    
    @pytest.fixture
    def sample_aggregated_signal(self):
        """Sample aggregated signal."""
        from datetime import datetime, timezone, timedelta
        
        # Create a proper SignalQuality object
        quality = SignalQuality(
            overall_score=85.0,
            grade=SignalQualityGrade.A,
            consensus_score=90.0,
            confidence_score=80.0,
            conflict_penalty=0.0,
            temporal_consistency=85.0,
            contributing_strategies=2,
            conflicts_detected=0,
            major_conflicts=0,
            is_tradeable=True
        )
        
        return AggregatedSignal(
            direction=SignalDirection.BUY,
            confidence=80.0,
            weight=0.85,
            contributing_strategies=["candlestick_strategy", "momentum_strategy"],
            strategy_weights={"candlestick_strategy": 0.6, "momentum_strategy": 0.4},
            quality=quality,
            risk_level=RiskLevel.MEDIUM,
            expiry=datetime.now(timezone.utc) + timedelta(minutes=15)
        )
    
    # Initialization Tests
    
    def test_agent_initialization(self, basic_config, signal_config, narrative_config):
        """Test basic agent initialization."""
        agent = SignalManagerAgent(
            name="test_signal_manager",
            config=basic_config,
            signal_config=signal_config,
            narrative_config=narrative_config
        )
        
        assert agent.name == "test_signal_manager"
        assert agent.agent_type == AgentType.SIGNAL_MANAGER
        assert agent.state == AgentState.CREATED
        assert agent.signal_config == signal_config
        assert agent.narrative_config == narrative_config
        assert agent.signal_manager is None
        assert len(agent.active_strategies) == 0
        assert len(agent.strategy_weights) == 0
        assert agent.signals_received == 0
        assert agent.signals_processed == 0
        assert agent.signals_published == 0
    
    def test_agent_initialization_defaults(self):
        """Test agent initialization with default configurations."""
        agent = SignalManagerAgent()
        
        assert agent.name == "signal_manager"
        assert agent.agent_type == AgentType.SIGNAL_MANAGER
        assert isinstance(agent.signal_config, SignalManagerConfiguration)
        assert isinstance(agent.narrative_config, NarrativeBufferConfig)
        assert agent.config.get('subscribe_to_strategies', True)
        assert agent.config.get('publish_aggregated_signals', True)
        assert agent.config.get('auto_register_strategies', True)
    
    # Lifecycle Tests
    
    @pytest.mark.asyncio
    async def test_agent_start_success(self, basic_config, signal_config):
        """Test successful agent startup."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        with patch.object(agent, '_setup_subscriptions', new=AsyncMock()) as mock_setup:
            result = await agent.start()
            
            assert result is True
            assert agent.state == AgentState.RUNNING
            assert agent.signal_manager is not None
            # Setup subscriptions should be called since we don't have message bus
            mock_setup.assert_not_called()  # No message bus set
    
    @pytest.mark.asyncio
    async def test_agent_start_with_message_bus(self, basic_config, signal_config, mock_message_bus):
        """Test agent startup with message bus."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        agent.set_message_bus(mock_message_bus)
        
        with patch.object(agent, '_setup_subscriptions', new=AsyncMock()) as mock_setup:
            result = await agent.start()
            
            assert result is True
            assert agent.state == AgentState.RUNNING
            assert agent.signal_manager is not None
            mock_setup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_start_failure(self, basic_config, signal_config):
        """Test agent startup failure."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Mock SignalManager to raise exception
        with patch('src.bistoury.agents.signal_manager_agent.SignalManager') as mock_sm:
            mock_sm.side_effect = Exception("Initialization failed")
            
            result = await agent.start()
            
            assert result is False
            assert agent.state == AgentState.ERROR
    
    @pytest.mark.asyncio
    async def test_agent_stop(self, basic_config, signal_config, mock_message_bus):
        """Test agent shutdown."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        agent.set_message_bus(mock_message_bus)
        
        # Start agent first
        await agent.start()
        assert agent.state == AgentState.RUNNING
        
        # Stop agent
        await agent.stop()
        assert agent.state == AgentState.STOPPED
        assert agent.signal_manager is None
    
    # Message Bus Integration Tests
    
    def test_set_message_bus(self, basic_config, mock_message_bus):
        """Test setting message bus."""
        agent = SignalManagerAgent(config=basic_config)
        agent.set_message_bus(mock_message_bus)
        
        assert agent._message_bus == mock_message_bus
    
    @pytest.mark.asyncio
    async def test_setup_subscriptions(self, basic_config, mock_message_bus):
        """Test setting up message bus subscriptions."""
        agent = SignalManagerAgent(config=basic_config)
        agent.set_message_bus(mock_message_bus)
        
        # Mock subscription objects
        signal_subscription = Mock()
        signal_subscription.id = "signal_sub_id"
        system_subscription = Mock()
        system_subscription.id = "system_sub_id"
        
        mock_message_bus.subscribe.side_effect = [signal_subscription, system_subscription]
        
        await agent._setup_subscriptions()
        
        # Verify subscriptions were created
        assert mock_message_bus.subscribe.call_count == 2
        assert len(agent._subscriptions) == 2
        assert "trading_signals" in agent._subscriptions
        assert "system_events" in agent._subscriptions
    
    @pytest.mark.asyncio
    async def test_cleanup_subscriptions(self, basic_config, mock_message_bus):
        """Test cleaning up subscriptions."""
        agent = SignalManagerAgent(config=basic_config)
        agent.set_message_bus(mock_message_bus)
        
        # Add mock subscriptions
        sub1 = Mock()
        sub1.id = "sub1"
        sub2 = Mock()
        sub2.id = "sub2"
        agent._subscriptions = {"signals": sub1, "events": sub2}
        
        await agent._cleanup_subscriptions()
        
        # Verify unsubscribe was called for each
        assert mock_message_bus.unsubscribe.call_count == 2
        assert len(agent._subscriptions) == 0
    
    # Signal Processing Tests
    
    @pytest.mark.asyncio
    async def test_handle_trading_signal(self, basic_config, signal_config, sample_trading_signal):
        """Test handling trading signal messages."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Start agent
        await agent.start()
        
        # Create signal message
        signal_payload = TradingSignalPayload(
            symbol=sample_trading_signal.symbol,
            signal_type=sample_trading_signal.signal_type,
            direction=sample_trading_signal.direction.value,
            confidence=sample_trading_signal.confidence,
            strength=sample_trading_signal.strength,
            timeframe=sample_trading_signal.timeframe,
            strategy=sample_trading_signal.strategy,
            reasoning=sample_trading_signal.reasoning,
            metadata=sample_trading_signal.metadata,
            timestamp=sample_trading_signal.created_at
        )
        
        message = Message(
            message_type=MessageType.SIGNAL_GENERATED,
            sender="candlestick_agent",
            payload=signal_payload,
            priority=MessagePriority.NORMAL
        )
        
        # Handle signal
        await agent._handle_trading_signal(message)
        
        # Verify processing
        assert agent.signals_received == 1
        assert agent.signals_processed == 1
        assert "candlestick_strategy" in agent.active_strategies
        assert agent.active_strategies["candlestick_strategy"] == "candlestick_agent"
        assert agent.last_signal_time is not None
        assert agent.aggregation_latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_handle_invalid_signal_payload(self, basic_config, signal_config):
        """Test handling invalid signal payload."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Start agent
        await agent.start()
        
        # Create invalid message
        message = Message(
            message_type=MessageType.SIGNAL_GENERATED,
            sender="test_agent",
            payload="invalid_payload",  # Wrong type
            priority=MessagePriority.NORMAL
        )
        
        # Handle signal (should not crash)
        await agent._handle_trading_signal(message)
        
        # Should not process invalid signals
        assert agent.signals_received == 1
        assert agent.signals_processed == 0
    
    @pytest.mark.asyncio
    async def test_handle_system_event(self, basic_config):
        """Test handling system event messages."""
        agent = SignalManagerAgent(config=basic_config)
        
        # Create system event
        event_payload = SystemEventPayload(
            event_type="agent_started",
            component="strategy",
            status="running",
            description={"strategy_name": "momentum_strategy"}
        )
        
        message = Message(
            message_type=MessageType.SYSTEM_EVENT,
            sender="momentum_agent",
            payload=event_payload,
            priority=MessagePriority.NORMAL
        )
        
        # Handle event
        await agent._handle_system_event(message)
        
        # Verify strategy registration
        assert "momentum_strategy" in agent.active_strategies
        assert agent.active_strategies["momentum_strategy"] == "momentum_agent"
        assert agent.strategy_weights["momentum_strategy"] == 1.0
    
    @pytest.mark.asyncio
    async def test_register_strategy(self, basic_config):
        """Test strategy registration."""
        agent = SignalManagerAgent(config=basic_config)
        
        await agent._register_strategy("test_strategy", "test_agent")
        
        assert "test_strategy" in agent.active_strategies
        assert agent.active_strategies["test_strategy"] == "test_agent"
        assert agent.strategy_weights["test_strategy"] == 1.0
        assert agent.signal_counts["test_strategy"] == 0
    
    # Signal Publishing Tests
    
    @pytest.mark.asyncio
    async def test_on_aggregated_signal(self, basic_config, mock_message_bus, sample_aggregated_signal):
        """Test publishing aggregated signals."""
        agent = SignalManagerAgent(config=basic_config)
        agent.set_message_bus(mock_message_bus)
        
        # Call the callback
        await agent._on_aggregated_signal(sample_aggregated_signal)
        
        # Verify message was published
        mock_message_bus.publish.assert_called_once()
        call_args = mock_message_bus.publish.call_args
        
        assert call_args.kwargs['topic'] == "trading.signals.aggregated"
        assert call_args.kwargs['message_type'] == MessageType.SIGNAL_AGGREGATED
        assert isinstance(call_args.kwargs['payload'], AggregatedSignalPayload)
        assert call_args.kwargs['sender'] == agent.name
        assert call_args.kwargs['priority'] == MessagePriority.HIGH
        
        # Check published signal count
        assert agent.signals_published == 1
    
    @pytest.mark.asyncio
    async def test_on_aggregated_signal_no_publish(self, basic_config, sample_aggregated_signal):
        """Test not publishing when disabled."""
        config = basic_config.copy()
        config['publish_aggregated_signals'] = False
        
        agent = SignalManagerAgent(config=config)
        
        # Call callback (should not publish)
        await agent._on_aggregated_signal(sample_aggregated_signal)
        
        # No signals should be published
        assert agent.signals_published == 0
    
    # Health Check Tests
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, basic_config, signal_config):
        """Test health check when agent is healthy."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Start agent
        await agent.start()
        
        # Simulate some activity
        agent.signals_processed = 10
        agent.signals_published = 8
        agent.last_signal_time = datetime.now(timezone.utc)
        agent.aggregation_latency_ms = 50.0
        
        health = await agent._health_check()
        
        assert health.state == AgentState.RUNNING
        assert health.health_score > 0.8  # Should be healthy
        assert health.messages_processed == 10
        assert health.tasks_completed == 8
        assert health.uptime_seconds >= 0
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, basic_config, signal_config):
        """Test health check when agent is unhealthy."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Start agent
        await agent.start()
        
        # Simulate poor performance
        agent.aggregation_latency_ms = 2000.0  # Very high latency
        agent.last_signal_time = datetime.now(timezone.utc) - timedelta(minutes=10)  # Stale
        
        # Mock signal manager status with errors
        with patch.object(agent.signal_manager, 'get_status') as mock_status:
            mock_status.return_value.is_running = True
            mock_status.return_value.error_count = 5
            mock_status.return_value.last_error = "Processing error"
            
            with patch.object(agent.signal_manager, 'get_metrics') as mock_metrics:
                mock_metrics.return_value.signals_per_minute = 10.0
                
                health = await agent._health_check()
        
        assert health.health_score < 0.5  # Should be unhealthy
        assert health.last_error == "Processing error"
        assert health.error_count == 5
    
    @pytest.mark.asyncio
    async def test_health_check_no_signal_manager(self, basic_config):
        """Test health check when signal manager is not initialized."""
        agent = SignalManagerAgent(config=basic_config)
        
        health = await agent._health_check()
        
        assert health.health_score == 0.0
        assert health.last_error == "Signal Manager not initialized"
    
    # Public API Tests
    
    @pytest.mark.asyncio
    async def test_get_strategy_weights(self, basic_config, signal_config):
        """Test getting strategy weights."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Add some strategies
        agent.strategy_weights = {
            "strategy1": 1.0,
            "strategy2": 0.8,
            "strategy3": 1.2
        }
        
        weights = await agent.get_strategy_weights()
        
        assert weights == {
            "strategy1": 1.0,
            "strategy2": 0.8,
            "strategy3": 1.2
        }
    
    @pytest.mark.asyncio
    async def test_update_strategy_weight(self, basic_config, signal_config):
        """Test updating strategy weight."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Start agent to initialize signal manager
        await agent.start()
        
        result = await agent.update_strategy_weight("test_strategy", 0.75)
        
        assert result is True
        assert agent.strategy_weights["test_strategy"] == 0.75
    
    @pytest.mark.asyncio
    async def test_get_signal_manager_status(self, basic_config, signal_config):
        """Test getting signal manager status."""
        agent = SignalManagerAgent(
            config=basic_config,
            signal_config=signal_config
        )
        
        # Start agent
        await agent.start()
        
        # Set some agent metrics
        agent.signals_received = 5
        agent.signals_processed = 4
        agent.signals_published = 3
        agent.aggregation_latency_ms = 100.0
        agent.active_strategies = {"strategy1": "agent1"}
        agent._subscriptions = {"signals": Mock(), "events": Mock()}
        
        status = await agent.get_signal_manager_status()
        
        assert "signal_manager" in status
        assert "metrics" in status
        assert "agent" in status
        
        agent_status = status["agent"]
        assert agent_status["signals_received"] == 5
        assert agent_status["signals_processed"] == 4
        assert agent_status["signals_published"] == 3
        assert agent_status["aggregation_latency_ms"] == 100.0
        assert agent_status["active_strategies"] == 1
        assert agent_status["subscriptions"] == 2
    
    @pytest.mark.asyncio
    async def test_get_signal_manager_status_not_initialized(self, basic_config):
        """Test getting status when signal manager is not initialized."""
        agent = SignalManagerAgent(config=basic_config)
        
        status = await agent.get_signal_manager_status()
        
        assert status == {"status": "not_initialized"}
    
    @pytest.mark.asyncio
    async def test_get_active_strategies(self, basic_config):
        """Test getting active strategies."""
        agent = SignalManagerAgent(config=basic_config)
        
        agent.active_strategies = {
            "candlestick": "candlestick_agent",
            "momentum": "momentum_agent"
        }
        
        strategies = await agent.get_active_strategies()
        
        assert strategies == {
            "candlestick": "candlestick_agent",
            "momentum": "momentum_agent"
        }
        
        # Verify it returns a copy
        strategies["new_strategy"] = "new_agent"
        assert "new_strategy" not in agent.active_strategies
    
    @pytest.mark.asyncio
    async def test_reload_configuration(self, basic_config):
        """Test configuration hot-reloading."""
        agent = SignalManagerAgent(config=basic_config)
        
        new_config = {
            "subscribe_to_strategies": False,
            "publish_aggregated_signals": False,
            "new_setting": "test_value"
        }
        
        result = await agent.reload_configuration(new_config)
        
        assert result is True
        assert agent.config["subscribe_to_strategies"] is False
        assert agent.config["publish_aggregated_signals"] is False
        assert agent.config["new_setting"] == "test_value"
    
    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_on_signal_error(self, basic_config):
        """Test handling signal manager errors."""
        agent = SignalManagerAgent(config=basic_config)
        
        test_error = Exception("Test error")
        
        # Should not crash
        await agent._on_signal_error(test_error)
    
    @pytest.mark.asyncio
    async def test_handle_error_with_message_bus(self, basic_config, mock_message_bus):
        """Test error handling with message bus."""
        agent = SignalManagerAgent(config=basic_config)
        agent.set_message_bus(mock_message_bus)
        
        test_error = Exception("Test error")
        
        await agent._handle_error(test_error)
        
        # Should publish error event
        mock_message_bus.publish.assert_called_once()
        call_args = mock_message_bus.publish.call_args
        
        assert call_args.kwargs['topic'] == "system.errors"
        assert call_args.kwargs['message_type'] == MessageType.SYSTEM_EVENT
        assert isinstance(call_args.kwargs['payload'], SystemEventPayload)
        assert call_args.kwargs['priority'] == MessagePriority.HIGH


if __name__ == "__main__":
    pytest.main([__file__]) 