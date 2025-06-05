"""
Unit tests for the BaseAgent class and agent infrastructure.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import pytest
import yaml

from src.bistoury.agents.base import (
    BaseAgent,
    AgentState,
    AgentType,
    AgentMetadata,
    AgentHealth,
    AgentCapability
)
from src.bistoury.config import Config


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""
    
    def __init__(self, name: str = "test_agent", **kwargs):
        super().__init__(name, AgentType.COLLECTOR, **kwargs)
        self.start_called = False
        self.stop_called = False
        self.health_check_called = False
    
    async def _start(self) -> bool:
        """Test implementation of start."""
        self.start_called = True
        await asyncio.sleep(0.1)  # Simulate startup time
        return True
    
    async def _stop(self) -> None:
        """Test implementation of stop."""
        self.stop_called = True
        await asyncio.sleep(0.1)  # Simulate shutdown time
    
    async def _health_check(self) -> AgentHealth:
        """Test implementation of health check."""
        self.health_check_called = True
        return AgentHealth(
            state=self.state,
            cpu_usage=25.0,
            memory_usage=1024,
            messages_processed=100,
            tasks_completed=50
        )


class TestAgentMetadata:
    """Test AgentMetadata model."""
    
    def test_agent_metadata_creation(self):
        """Test creating agent metadata."""
        metadata = AgentMetadata(
            name="test_agent",
            agent_type=AgentType.COLLECTOR,
            description="Test agent for unit testing"
        )
        
        assert metadata.name == "test_agent"
        assert metadata.agent_type == AgentType.COLLECTOR
        assert metadata.description == "Test agent for unit testing"
        assert metadata.version == "1.0.0"
        assert metadata.agent_id is not None
        assert len(metadata.agent_id) > 0
        assert metadata.created_at is not None
    
    def test_agent_metadata_with_dependencies(self):
        """Test agent metadata with dependencies."""
        metadata = AgentMetadata(
            name="test_agent",
            agent_type=AgentType.TRADER,
            dependencies=["collector", "signal_manager"],
            required_config=["api_key", "trading_params"],
            capabilities=["spot_trading", "futures_trading"]
        )
        
        assert metadata.dependencies == ["collector", "signal_manager"]
        assert metadata.required_config == ["api_key", "trading_params"]
        assert metadata.capabilities == ["spot_trading", "futures_trading"]


class TestAgentHealth:
    """Test AgentHealth model."""
    
    def test_agent_health_creation(self):
        """Test creating agent health."""
        health = AgentHealth(
            state=AgentState.RUNNING,
            cpu_usage=45.2,
            memory_usage=2048,
            error_count=0,
            messages_processed=1000
        )
        
        assert health.state == AgentState.RUNNING
        assert health.cpu_usage == 45.2
        assert health.memory_usage == 2048
        assert health.error_count == 0
        assert health.messages_processed == 1000
        assert health.is_healthy is True
        assert health.health_score == 1.0


class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    @pytest.fixture
    async def agent(self):
        """Create a test agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "test_agent_state.yaml"
            agent = TestAgent(
                name="test_agent",
                persist_state=True,
                state_file=str(state_file)
            )
            yield agent
            
            # Cleanup
            if agent.is_running:
                await agent.stop()
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = TestAgent()
        
        assert agent.name == "test_agent"
        assert agent.agent_type == AgentType.COLLECTOR
        assert agent.state == AgentState.CREATED
        assert agent.metadata.name == "test_agent"
        assert agent.metadata.agent_type == AgentType.COLLECTOR
        assert agent.health.state == AgentState.CREATED
        assert not agent.is_running
        assert agent.is_stopped is False  # CREATED is not stopped
        assert agent.uptime == 0.0
    
    def test_agent_properties(self):
        """Test agent properties."""
        agent = TestAgent(name="prop_test", config={"test_key": "test_value"})
        
        assert agent.agent_id == agent.metadata.agent_id
        assert agent.name == "prop_test"
        assert agent.agent_type == AgentType.COLLECTOR
        assert agent.config["test_key"] == "test_value"
    
    async def test_agent_lifecycle(self, agent):
        """Test complete agent lifecycle."""
        # Initial state
        assert agent.state == AgentState.CREATED
        assert not agent.is_running
        
        # Start agent
        success = await agent.start()
        assert success is True
        assert agent.state == AgentState.RUNNING
        assert agent.is_running
        assert agent.start_called
        assert agent.metadata.started_at is not None
        assert agent.uptime > 0
        
        # Stop agent
        await agent.stop()
        assert agent.state == AgentState.STOPPED
        assert not agent.is_running
        assert agent.is_stopped
        assert agent.stop_called
        assert agent.metadata.stopped_at is not None
    
    async def test_agent_restart(self, agent):
        """Test agent restart."""
        # Start agent
        await agent.start()
        assert agent.is_running
        
        # Restart agent
        success = await agent.restart()
        assert success is True
        assert agent.is_running
        
        # Both start and stop should have been called
        assert agent.start_called
        assert agent.stop_called
    
    async def test_agent_pause_resume(self, agent):
        """Test agent pause and resume."""
        # Start agent first
        await agent.start()
        assert agent.state == AgentState.RUNNING
        
        # Pause agent
        await agent.pause()
        assert agent.state == AgentState.PAUSED
        
        # Resume agent
        await agent.resume()
        assert agent.state == AgentState.RUNNING
    
    async def test_agent_health_monitoring(self, agent):
        """Test agent health monitoring."""
        await agent.start()
        
        # Get health status
        health = await agent.get_health()
        assert isinstance(health, AgentHealth)
        assert health.state == AgentState.RUNNING
        assert health.cpu_usage == 25.0
        assert health.memory_usage == 1024
        assert health.messages_processed == 100
        assert health.tasks_completed == 50
        assert health.uptime_seconds > 0
        assert agent.health_check_called
    
    def test_configuration_management(self):
        """Test configuration management."""
        config = {"key1": "value1", "key2": 42}
        agent = TestAgent(config=config)
        
        # Test getting config values
        assert agent.get_config_value("key1") == "value1"
        assert agent.get_config_value("key2") == 42
        assert agent.get_config_value("nonexistent", "default") == "default"
        
        # Test updating config
        agent.update_config({"key3": "value3"})
        assert agent.get_config_value("key3") == "value3"
    
    def test_global_config_integration(self):
        """Test integration with global configuration."""
        agent = TestAgent()
        
        # Mock global config
        mock_config = Mock()
        mock_config.database.path = "/test/path"
        agent.set_global_config(mock_config)
        
        # Test accessing global config values
        assert agent.get_config_value("database.path") == "/test/path"
    
    def test_state_persistence(self):
        """Test state persistence functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "persistence_test.yaml"
            
            # Create agent with persistence
            agent1 = TestAgent(
                name="persistence_test",
                persist_state=True,
                state_file=str(state_file),
                config={"test_config": "test_value"}
            )
            
            # Modify some state
            agent1.health.error_count = 5
            agent1._set_state(AgentState.RUNNING)
            
            # Check that state file was created
            assert state_file.exists()
            
            # Load state data manually to verify
            with open(state_file, 'r') as f:
                state_data = yaml.safe_load(f)
            
            assert state_data['state'] == AgentState.RUNNING.value
            assert state_data['config']['test_config'] == "test_value"
            assert state_data['health']['error_count'] == 5
            
            # Create new agent with same state file
            agent2 = TestAgent(
                name="persistence_test",
                persist_state=True,
                state_file=str(state_file)
            )
            
            # Verify state was loaded
            assert agent2.config['test_config'] == "test_value"
            assert agent2.health.error_count == 5
            # State should be reset to CREATED on load
            assert agent2.state == AgentState.CREATED
    
    async def test_error_handling_during_start(self):
        """Test error handling during agent startup."""
        
        class FailingAgent(BaseAgent):
            async def _start(self) -> bool:
                raise ValueError("Test startup error")
            
            async def _stop(self) -> None:
                pass
            
            async def _health_check(self) -> AgentHealth:
                return AgentHealth(state=self.state)
        
        agent = FailingAgent("failing_agent", AgentType.COLLECTOR)
        
        success = await agent.start()
        assert success is False
        assert agent.state == AgentState.CRASHED
    
    async def test_shutdown_callbacks(self, agent):
        """Test shutdown callbacks."""
        callback_called = False
        
        async def test_callback():
            nonlocal callback_called
            callback_called = True
        
        agent.add_shutdown_callback(test_callback)
        
        await agent.start()
        await agent.stop()
        
        assert callback_called is True
    
    async def test_task_management(self, agent):
        """Test background task management."""
        await agent.start()
        
        # Create a background task
        task_completed = False
        
        async def background_task():
            nonlocal task_completed
            await asyncio.sleep(0.1)
            task_completed = True
        
        task = agent.create_task(background_task())
        
        # Wait for task to complete
        await asyncio.sleep(0.2)
        assert task_completed is True
        assert task.done()
    
    async def test_health_score_calculation(self, agent):
        """Test health score calculation."""
        await agent.start()
        
        # Test normal health
        health = await agent.get_health()
        assert health.health_score == 1.0
        assert health.is_healthy is True
        
        # Simulate errors
        agent.health.error_count = 3
        health_score = agent._calculate_health_score()
        assert health_score < 1.0  # Should be penalized for errors
        
        # Simulate high CPU usage
        agent.health.cpu_usage = 85.0
        health_score = agent._calculate_health_score()
        assert health_score < 0.8  # Should be further penalized
    
    async def test_cannot_start_running_agent(self, agent):
        """Test that running agent cannot be started again."""
        await agent.start()
        assert agent.is_running
        
        # Try to start again
        success = await agent.start()
        assert success is False
    
    async def test_double_stop_handling(self, agent):
        """Test that stopping already stopped agent is handled gracefully."""
        await agent.start()
        await agent.stop()
        assert agent.is_stopped
        
        # Try to stop again - should not raise an exception
        await agent.stop()
        assert agent.is_stopped


@pytest.mark.asyncio
async def test_agent_heartbeat():
    """Test agent heartbeat functionality."""
    agent = TestAgent()
    
    await agent.start()
    
    # Wait for at least one heartbeat
    await asyncio.sleep(0.1)
    
    health = await agent.get_health()
    assert health.last_heartbeat is not None
    
    await agent.stop()


class TestAgentCapability:
    """Test AgentCapability functionality."""
    
    def test_capability_creation(self):
        """Test creating agent capability."""
        capability = AgentCapability(
            name="data_collection",
            description="Collects market data from exchanges",
            version="1.0.0",
            dependencies=["database_connection"],
            parameters={"update_interval": 1000, "max_symbols": 100}
        )
        
        assert capability.name == "data_collection"
        assert capability.description == "Collects market data from exchanges"
        assert capability.version == "1.0.0"
        assert capability.dependencies == ["database_connection"]
        assert capability.parameters["update_interval"] == 1000
        assert capability.parameters["max_symbols"] == 100 