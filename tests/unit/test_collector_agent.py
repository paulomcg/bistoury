"""
Unit tests for the CollectorAgent class.

Tests the integration of EnhancedDataCollector with the BaseAgent framework,
including lifecycle management, health monitoring, messaging, and configuration.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
import tempfile
import os

from src.bistoury.agents.collector_agent import CollectorAgent, CollectorAgentConfig
from src.bistoury.agents.base import AgentState, AgentType
from src.bistoury.models.agent_messages import Message, MessageType, MessagePriority


class MockEnhancedDataCollector:
    """Mock EnhancedDataCollector for testing."""
    
    def __init__(self, *args, **kwargs):
        self.running = False
        self.stats = {
            'candles_collected': 100,
            'trades_collected': 50,
            'orderbooks_collected': 25,
            'total_candles_stored': 1000,
            'errors': 0,
            'batches_processed': 10,
            'last_activity': datetime.now(timezone.utc).isoformat()
        }
    
    async def start(self) -> bool:
        self.running = True
        return True
    
    async def stop(self) -> None:
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()
    
    def add_symbol(self, symbol: str) -> None:
        pass
    
    def remove_symbol(self, symbol: str) -> None:
        pass
    
    async def collect_enhanced_historical_data(self, symbol: str, days_back: int, intervals: list) -> Dict[str, int]:
        return {interval: 100 for interval in intervals}


class MockHyperLiquidIntegration:
    """Mock HyperLiquid integration for testing."""
    
    def is_connected(self) -> bool:
        return True
    
    async def connect(self) -> bool:
        return True


class MockDatabaseManager:
    """Mock database manager for testing."""
    
    def __init__(self):
        self.connected = True


class MockMessageBus:
    """Mock message bus for testing."""
    
    def __init__(self):
        self.published_messages = []
    
    async def publish(self, message: Message) -> None:
        self.published_messages.append(message)


@pytest.fixture
def mock_hyperliquid():
    """Fixture for mock HyperLiquid integration."""
    return MockHyperLiquidIntegration()


@pytest.fixture
def mock_db_manager():
    """Fixture for mock database manager."""
    return MockDatabaseManager()


@pytest.fixture
def mock_message_bus():
    """Fixture for mock message bus."""
    return MockMessageBus()


@pytest.fixture
def collector_config():
    """Fixture for collector configuration."""
    return {
        'collector': {
            'symbols': ['BTC', 'ETH'],
            'intervals': ['1m', '5m', '15m'],
            'buffer_size': 500,
            'flush_interval': 15.0,
            'stats_interval': 30.0,
            'health_check_interval': 10.0,
            'collect_historical_on_start': False,
            'publish_data_updates': True,
            'publish_stats_updates': True
        }
    }


@pytest.fixture
async def collector_agent(mock_hyperliquid, mock_db_manager, collector_config):
    """Fixture for CollectorAgent instance."""
    with patch('src.bistoury.agents.collector_agent.EnhancedDataCollector', MockEnhancedDataCollector):
        agent = CollectorAgent(
            hyperliquid=mock_hyperliquid,
            db_manager=mock_db_manager,
            config=collector_config,
            name="test_collector",
            persist_state=False  # Disable state persistence for tests
        )
        
        yield agent
        
        # Clean up after test
        try:
            if agent.is_running:
                await agent.stop()
        except Exception:
            pass  # Ignore cleanup errors


class TestCollectorAgentConfig:
    """Test CollectorAgentConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CollectorAgentConfig()
        
        assert config.symbols == {'BTC', 'ETH', 'SOL'}
        assert config.intervals == {'1m', '5m', '15m', '1h', '4h', '1d'}
        assert config.buffer_size == 1000
        assert config.flush_interval == 30.0
        assert config.max_batch_size == 5000
        assert config.stats_interval == 60.0
        assert config.health_check_interval == 30.0
        assert config.collect_orderbooks is True
        assert config.collect_funding_rates is True
        assert config.collect_historical_on_start is False
        assert config.historical_days == 7
        assert config.publish_data_updates is True
        assert config.publish_stats_updates is True
        assert config.data_update_interval == 5.0
    
    def test_custom_config(self):
        """Test configuration with custom values."""
        config = CollectorAgentConfig(
            symbols={'DOGE', 'ADA'},
            buffer_size=2000,
            collect_orderbooks=False,
            historical_days=14
        )
        
        assert config.symbols == {'DOGE', 'ADA'}
        assert config.buffer_size == 2000
        assert config.collect_orderbooks is False
        assert config.historical_days == 14


class TestCollectorAgentInitialization:
    """Test CollectorAgent initialization."""
    
    def test_initialization_with_defaults(self, mock_hyperliquid, mock_db_manager):
        """Test agent initialization with default configuration."""
        with patch('src.bistoury.agents.collector_agent.EnhancedDataCollector', MockEnhancedDataCollector):
            agent = CollectorAgent(
                hyperliquid=mock_hyperliquid,
                db_manager=mock_db_manager,
                persist_state=False
            )
            
            assert agent.name == "collector"
            assert agent.agent_type == AgentType.COLLECTOR
            assert agent.state == AgentState.CREATED
            assert isinstance(agent.collector_config, CollectorAgentConfig)
            assert agent.metadata.description == "Real-time market data collection agent"
            assert "market_data_collection" in agent.metadata.capabilities
    
    def test_initialization_with_custom_config(self, mock_hyperliquid, mock_db_manager, collector_config):
        """Test agent initialization with custom configuration."""
        with patch('src.bistoury.agents.collector_agent.EnhancedDataCollector', MockEnhancedDataCollector):
            agent = CollectorAgent(
                hyperliquid=mock_hyperliquid,
                db_manager=mock_db_manager,
                config=collector_config,
                name="custom_collector",
                persist_state=False
            )
            
            assert agent.name == "custom_collector"
            assert agent.collector_config.symbols == {'BTC', 'ETH'}
            assert agent.collector_config.buffer_size == 500
            assert agent.collector_config.stats_interval == 30.0
    
    def test_metadata_setup(self, collector_agent):
        """Test agent metadata is properly set up."""
        assert collector_agent.metadata.agent_type == AgentType.COLLECTOR
        assert collector_agent.metadata.version == "1.0.0"
        assert collector_agent.metadata.dependencies == ["hyperliquid_api", "database"]
        assert "real_time_feeds" in collector_agent.metadata.capabilities
        assert "health_monitoring" in collector_agent.metadata.capabilities


class TestCollectorAgentLifecycle:
    """Test CollectorAgent lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_start_success(self, collector_agent, mock_message_bus):
        """Test successful agent start."""
        collector_agent.set_message_bus(mock_message_bus)
        
        result = await collector_agent.start()
        
        assert result is True
        assert collector_agent.state == AgentState.RUNNING
        assert collector_agent.collector.running is True
        assert collector_agent._collection_start_time is not None
        
        # Check startup message was sent
        startup_messages = [
            msg for msg in mock_message_bus.published_messages
            if msg.type == MessageType.AGENT_STARTED
        ]
        assert len(startup_messages) == 1
    
    @pytest.mark.asyncio
    async def test_start_failure(self, collector_agent):
        """Test agent start failure when collector fails."""
        # Mock collector start to fail
        collector_agent.collector.start = AsyncMock(return_value=False)
        
        result = await collector_agent.start()
        
        assert result is False
        # The agent might go to ERROR state due to initialization state loading
        # So we check that it's not RUNNING
        assert collector_agent.state != AgentState.RUNNING
    
    @pytest.mark.asyncio
    async def test_stop_success(self, collector_agent, mock_message_bus):
        """Test successful agent stop."""
        collector_agent.set_message_bus(mock_message_bus)
        
        # Start agent first
        await collector_agent.start()
        
        # Stop agent
        await collector_agent.stop()
        
        assert collector_agent.state == AgentState.STOPPED
        assert collector_agent.collector.running is False
        
        # Check stop message was sent
        stop_messages = [
            msg for msg in mock_message_bus.published_messages
            if msg.type == MessageType.AGENT_STOPPED
        ]
        assert len(stop_messages) == 1
    
    @pytest.mark.asyncio
    async def test_restart(self, collector_agent):
        """Test agent restart functionality."""
        # Start agent
        await collector_agent.start()
        assert collector_agent.state == AgentState.RUNNING
        
        # Restart agent
        result = await collector_agent.restart()
        
        assert result is True
        assert collector_agent.state == AgentState.RUNNING


class TestCollectorAgentHealthMonitoring:
    """Test CollectorAgent health monitoring."""
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, collector_agent):
        """Test health check when collector is healthy."""
        # Start agent to set uptime
        await collector_agent.start()
        
        health = await collector_agent._health_check()
        
        assert health.health_score >= 0.5
        assert health.state == AgentState.RUNNING
        assert health.uptime_seconds > 0
        assert health.tasks_completed == 10  # From mock stats
        assert health.error_count == 0
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy_collector_stopped(self, collector_agent):
        """Test health check when collector is stopped."""
        collector_agent.collector.running = False
        
        health = await collector_agent._health_check()
        
        assert health.health_score <= 0.3
    
    @pytest.mark.asyncio
    async def test_health_check_with_errors(self, collector_agent):
        """Test health check when collector has errors."""
        collector_agent.collector.stats['errors'] = 5
        collector_agent.collector.stats['batches_processed'] = 10
        
        health = await collector_agent._health_check()
        
        assert health.health_score < 1.0
        assert health.error_count == 5
    
    @pytest.mark.asyncio
    async def test_health_check_stale_activity(self, collector_agent):
        """Test health check when last activity is stale."""
        # Set last activity to 10 minutes ago
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        collector_agent.collector.stats['last_activity'] = stale_time.isoformat()
        
        health = await collector_agent._health_check()
        
        assert health.health_score <= 0.5
    
    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, collector_agent):
        """Test health check exception handling."""
        # Mock collector.get_stats to raise exception
        collector_agent.collector.get_stats = Mock(side_effect=Exception("Test error"))
        
        health = await collector_agent._health_check()
        
        assert health.health_score == 0.1
        assert health.last_error == "Test error"
        assert health.last_error_time is not None


class TestCollectorAgentMessaging:
    """Test CollectorAgent messaging functionality."""
    
    @pytest.mark.asyncio
    async def test_publish_data_updates(self, collector_agent, mock_message_bus):
        """Test publishing data update messages."""
        collector_agent.set_message_bus(mock_message_bus)
        collector_agent.collector_config.data_update_interval = 0.01  # Very fast updates for testing
        collector_agent.collector_config.publish_data_updates = True
        
        # Force first publish by setting last publish time to past
        collector_agent._last_data_publish = datetime.now(timezone.utc) - timedelta(seconds=10)
        
        # Start data update publishing
        task = asyncio.create_task(collector_agent._publish_data_updates())
        
        # Wait for at least one update
        await asyncio.sleep(0.1)
        
        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass
        
        # Check messages were published
        data_messages = [
            msg for msg in mock_message_bus.published_messages
            if msg.type == MessageType.DATA_MARKET_UPDATE and msg.topic == "data.collection"
        ]
        assert len(data_messages) >= 1
        
        # Verify message content
        message = data_messages[0]
        assert message.sender == collector_agent.name
        assert message.priority == MessagePriority.NORMAL
        assert "candles_collected" in message.payload.data
    
    @pytest.mark.asyncio
    async def test_publish_stats_updates(self, collector_agent, mock_message_bus):
        """Test publishing statistics update messages."""
        collector_agent.set_message_bus(mock_message_bus)
        collector_agent.collector_config.stats_interval = 0.1  # Fast updates for testing
        
        # Start stats update publishing
        task = asyncio.create_task(collector_agent._publish_stats_updates())
        
        # Wait for at least one update
        await asyncio.sleep(0.2)
        task.cancel()
        
        # Check messages were published
        stats_messages = [
            msg for msg in mock_message_bus.published_messages
            if msg.type == MessageType.AGENT_HEALTH_UPDATE
        ]
        assert len(stats_messages) >= 1
        
        # Verify message content
        message = stats_messages[0]
        assert message.sender == collector_agent.name
        assert message.topic == "agent.health"
        assert "collector_stats" in message.payload.metadata
    
    @pytest.mark.asyncio
    async def test_send_system_message(self, collector_agent, mock_message_bus):
        """Test sending system messages."""
        collector_agent.set_message_bus(mock_message_bus)
        
        await collector_agent._send_system_message(
            MessageType.AGENT_ERROR,
            "Test error message"
        )
        
        # Check message was sent
        assert len(mock_message_bus.published_messages) == 1
        message = mock_message_bus.published_messages[0]
        
        assert message.type == MessageType.AGENT_ERROR
        assert message.sender == collector_agent.name
        assert message.topic == "agent.lifecycle"
        assert message.priority == MessagePriority.HIGH
        assert message.payload.description == "Test error message"
    
    def test_message_bus_not_set(self, collector_agent):
        """Test behavior when message bus is not set."""
        # Should not raise exception when message bus is None
        collector_agent._message_bus = None
        
        # These should complete without error
        asyncio.run(collector_agent._send_system_message(MessageType.AGENT_STARTED, "Test"))


class TestCollectorAgentConfiguration:
    """Test CollectorAgent configuration management."""
    
    def test_get_configuration(self, collector_agent):
        """Test getting current configuration."""
        config = collector_agent.get_configuration()
        
        assert "collector" in config
        collector_config = config["collector"]
        
        assert "symbols" in collector_config
        assert "intervals" in collector_config
        assert "buffer_size" in collector_config
        assert "flush_interval" in collector_config
        assert "publish_data_updates" in collector_config
    
    def test_update_configuration(self, collector_agent):
        """Test updating configuration."""
        original_buffer_size = collector_agent.collector_config.buffer_size
        
        updates = {
            'collector': {
                'buffer_size': 2000,
                'stats_interval': 120.0
            }
        }
        
        collector_agent.update_configuration(updates)
        
        assert collector_agent.collector_config.buffer_size == 2000
        assert collector_agent.collector_config.stats_interval == 120.0
        assert collector_agent.collector_config.buffer_size != original_buffer_size
    
    def test_update_invalid_configuration(self, collector_agent):
        """Test updating with invalid configuration keys."""
        original_buffer_size = collector_agent.collector_config.buffer_size
        
        updates = {
            'collector': {
                'invalid_key': 'invalid_value',
                'buffer_size': 3000
            }
        }
        
        collector_agent.update_configuration(updates)
        
        # Valid key should be updated
        assert collector_agent.collector_config.buffer_size == 3000
        
        # Invalid key should be ignored (no exception)
        assert not hasattr(collector_agent.collector_config, 'invalid_key')


class TestCollectorAgentPublicAPI:
    """Test CollectorAgent public API methods."""
    
    def test_add_symbol(self, collector_agent):
        """Test adding a symbol to collection."""
        original_symbols = collector_agent.collector_config.symbols.copy()
        
        collector_agent.add_symbol("DOGE")
        
        assert "DOGE" in collector_agent.collector_config.symbols
        assert len(collector_agent.collector_config.symbols) == len(original_symbols) + 1
    
    def test_add_symbol_case_insensitive(self, collector_agent):
        """Test adding a symbol is case insensitive."""
        collector_agent.add_symbol("doge")
        
        assert "DOGE" in collector_agent.collector_config.symbols
        assert "doge" not in collector_agent.collector_config.symbols
    
    def test_remove_symbol(self, collector_agent):
        """Test removing a symbol from collection."""
        # Add a symbol first
        collector_agent.add_symbol("TEST")
        assert "TEST" in collector_agent.collector_config.symbols
        
        # Remove it
        collector_agent.remove_symbol("TEST")
        
        assert "TEST" not in collector_agent.collector_config.symbols
    
    def test_remove_nonexistent_symbol(self, collector_agent):
        """Test removing a non-existent symbol doesn't raise error."""
        original_symbols = collector_agent.collector_config.symbols.copy()
        
        # Should not raise exception
        collector_agent.remove_symbol("NONEXISTENT")
        
        assert collector_agent.collector_config.symbols == original_symbols
    
    @pytest.mark.asyncio
    async def test_collect_historical_data(self, collector_agent):
        """Test collecting historical data for a symbol."""
        result = await collector_agent.collect_historical_data(
            symbol="BTC",
            days_back=5,
            intervals=["1m", "5m"]
        )
        
        assert isinstance(result, dict)
        assert "1m" in result
        assert "5m" in result
        assert result["1m"] == 100  # From mock
        assert result["5m"] == 100  # From mock
    
    @pytest.mark.asyncio
    async def test_collect_historical_data_default_intervals(self, collector_agent):
        """Test collecting historical data with default intervals."""
        result = await collector_agent.collect_historical_data(symbol="ETH")
        
        # Should use agent's configured intervals
        for interval in collector_agent.collector_config.intervals:
            assert interval in result
    
    def test_get_collection_stats(self, collector_agent):
        """Test getting collection statistics."""
        stats = collector_agent.get_collection_stats()
        
        assert "agent_name" in stats
        assert "agent_state" in stats
        assert "symbols" in stats
        assert "intervals" in stats
        assert "collection_uptime" in stats
        assert "candles_collected" in stats
        assert "trades_collected" in stats
        
        assert stats["agent_name"] == collector_agent.name
        assert stats["agent_state"] == collector_agent.state.value
    
    def test_get_collection_stats_with_uptime(self, collector_agent):
        """Test getting collection statistics with uptime calculation."""
        # Set collection start time
        collector_agent._collection_start_time = datetime.now(timezone.utc) - timedelta(seconds=100)
        
        stats = collector_agent.get_collection_stats()
        
        assert stats["collection_uptime"] > 0
        assert stats["collection_uptime"] <= 200  # Should be around 100 seconds


class TestCollectorAgentIntegration:
    """Test CollectorAgent integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_messaging(self, collector_agent, mock_message_bus):
        """Test complete agent lifecycle with message bus integration."""
        collector_agent.set_message_bus(mock_message_bus)
        
        # Start agent
        start_result = await collector_agent.start()
        assert start_result is True
        assert collector_agent.state == AgentState.RUNNING
        
        # Verify startup message
        startup_messages = [
            msg for msg in mock_message_bus.published_messages
            if msg.type == MessageType.AGENT_STARTED
        ]
        assert len(startup_messages) == 1
        
        # Wait briefly for background tasks
        await asyncio.sleep(0.1)
        
        # Stop agent
        await collector_agent.stop()
        assert collector_agent.state == AgentState.STOPPED
        
        # Verify stop message
        stop_messages = [
            msg for msg in mock_message_bus.published_messages
            if msg.type == MessageType.AGENT_STOPPED
        ]
        assert len(stop_messages) == 1
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, collector_agent, mock_message_bus):
        """Test health monitoring integration with messaging."""
        collector_agent.set_message_bus(mock_message_bus)
        collector_agent.collector_config.health_check_interval = 0.1  # Fast checks
        
        # Start agent
        await collector_agent.start()
        
        # Simulate collector failure
        collector_agent.collector.running = False
        
        # Wait for health monitor to detect failure
        await asyncio.sleep(0.2)
        
        # Check that agent state changed to ERROR
        assert collector_agent.state == AgentState.ERROR
        
        # Check error message was sent
        error_messages = [
            msg for msg in mock_message_bus.published_messages
            if msg.type == MessageType.AGENT_ERROR
        ]
        assert len(error_messages) >= 1
        
        await collector_agent.stop()
    
    @pytest.mark.asyncio
    async def test_configuration_persistence(self, mock_hyperliquid, mock_db_manager):
        """Test configuration persistence across agent lifecycle."""
        config = {
            'collector': {
                'symbols': ['CUSTOM1', 'CUSTOM2'],
                'buffer_size': 1500
            }
        }
        
        with patch('src.bistoury.agents.collector_agent.EnhancedDataCollector', MockEnhancedDataCollector):
            agent = CollectorAgent(
                hyperliquid=mock_hyperliquid,
                db_manager=mock_db_manager,
                config=config,
                name="persistent_collector",
                persist_state=False
            )
        
        # Check configuration is applied
        assert agent.collector_config.symbols == {'CUSTOM1', 'CUSTOM2'}
        assert agent.collector_config.buffer_size == 1500
        
        # Lifecycle operations should preserve config
        await agent.start()
        await agent.stop()
        
        assert agent.collector_config.symbols == {'CUSTOM1', 'CUSTOM2'}
        assert agent.collector_config.buffer_size == 1500


if __name__ == "__main__":
    pytest.main([__file__]) 