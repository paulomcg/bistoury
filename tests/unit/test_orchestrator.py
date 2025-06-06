"""
Unit tests for the Agent Orchestrator.

Tests cover orchestrator initialization, agent management, startup/shutdown
sequencing, resource allocation, failure handling, and load balancing.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List

from src.bistoury.agents.orchestrator import AgentOrchestrator, ResourceManager, LoadBalancer
from src.bistoury.agents.base import BaseAgent, AgentType, AgentState, AgentHealth
from src.bistoury.agents.registry import AgentRegistry
from src.bistoury.agents.messaging import MessageBus
from src.bistoury.models.orchestrator_config import (
    OrchestratorConfig, StartupPolicy, ShutdownPolicy, FailurePolicy,
    ResourceType, LoadBalancingStrategy, AgentResourceRequirements,
    ResourceLimit, StartupSequenceConfig, ShutdownSequenceConfig,
    FailureHandlingConfig, LoadBalancingConfig
)
from src.bistoury.models.agent_registry import AgentRegistration, DependencyGraph


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, name: str, agent_type: AgentType):
        super().__init__(name, agent_type)
        self._start_success = True
        self._health_score = 1.0
        
    async def _start(self) -> bool:
        await asyncio.sleep(0.1)  # Simulate startup time
        return self._start_success
    
    async def _stop(self) -> None:
        await asyncio.sleep(0.1)  # Simulate shutdown time
    
    async def _health_check(self) -> AgentHealth:
        return AgentHealth(
            state=self.state,
            health_score=self._health_score,
            is_healthy=self._health_score > 0.5
        )
    
    def set_start_success(self, success: bool):
        """Set whether start should succeed."""
        self._start_success = success
    
    def set_health_score(self, score: float):
        """Set health score for testing."""
        self._health_score = score


@pytest.fixture
def orchestrator_config():
    """Create test orchestrator configuration."""
    config = OrchestratorConfig()
    
    # Set up resource limits
    config.resource_limits = [
        ResourceLimit(
            resource_type=ResourceType.CPU,
            max_allocation=100.0,
            unit="%",
            reserved=10.0
        ),
        ResourceLimit(
            resource_type=ResourceType.MEMORY,
            max_allocation=8192.0,
            unit="MB",
            reserved=1024.0
        )
    ]
    
    # Configure startup policy
    config.startup.policy = StartupPolicy.BATCH
    config.startup.batch_size = 2
    config.startup.startup_timeout = timedelta(seconds=30)
    
    # Configure failure handling
    config.failure_handling.default_policy = FailurePolicy.RESTART
    config.failure_handling.max_restart_attempts = 2
    config.failure_handling.critical_agents = [AgentType.ORCHESTRATOR]
    
    return config


@pytest.fixture
def mock_registry():
    """Create mock agent registry."""
    from src.bistoury.agents.registry import DependencyGraph
    
    registry = Mock(spec=AgentRegistry)
    
    # Create a mock dependency graph with proper startup_order
    mock_graph = Mock(spec=DependencyGraph)
    mock_graph.startup_order = []  # Empty list, will be populated by test
    
    registry.get_dependency_graph = AsyncMock(return_value=mock_graph)
    registry.get_agent = AsyncMock(return_value=None)  # No dependencies by default
    return registry


@pytest.fixture
def mock_message_bus():
    """Create mock message bus."""
    message_bus = Mock(spec=MessageBus)
    message_bus.subscribe = AsyncMock()
    return message_bus


@pytest.fixture
def orchestrator(orchestrator_config, mock_registry, mock_message_bus):
    """Create test orchestrator."""
    return AgentOrchestrator(
        config=orchestrator_config,
        registry=mock_registry,
        message_bus=mock_message_bus
    )


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    agents = {
        "collector": MockAgent("collector", AgentType.COLLECTOR),
        "signal_manager": MockAgent("signal_manager", AgentType.SIGNAL_MANAGER),
        "trader": MockAgent("trader", AgentType.TRADER),
        "risk_manager": MockAgent("risk_manager", AgentType.RISK_MANAGER)
    }
    return agents


class TestOrchestratorBasics:
    """Test basic orchestrator operations."""
    
    def test_orchestrator_initialization(self, orchestrator_config):
        """Test orchestrator initializes correctly."""
        orchestrator = AgentOrchestrator(config=orchestrator_config)
        
        assert orchestrator.config == orchestrator_config
        assert orchestrator.state.state == AgentState.CREATED
        assert orchestrator.state.agents_managed == 0
        assert not orchestrator._running
        assert len(orchestrator.managed_agents) == 0
    
    async def test_orchestrator_start_stop(self, orchestrator):
        """Test orchestrator start and stop operations."""
        # Start orchestrator
        success = await orchestrator.start()
        assert success
        assert orchestrator._running
        assert orchestrator.state.state == AgentState.RUNNING
        assert orchestrator._start_time is not None
        
        # Stop orchestrator
        await orchestrator.stop()
        assert not orchestrator._running
        assert orchestrator.state.state == AgentState.STOPPED
    
    async def test_orchestrator_start_already_running(self, orchestrator):
        """Test starting orchestrator when already running."""
        await orchestrator.start()
        
        # Try to start again
        success = await orchestrator.start()
        assert not success  # Should return False when already running
    
    async def test_orchestrator_stop_not_running(self, orchestrator):
        """Test stopping orchestrator when not running."""
        # Should not raise exception
        await orchestrator.stop()
        assert orchestrator.state.state == AgentState.CREATED


class TestAgentRegistration:
    """Test agent registration and management."""
    
    async def test_register_agent(self, orchestrator, mock_agents):
        """Test agent registration."""
        agent = mock_agents["collector"]
        
        success = await orchestrator.register_agent(agent)
        assert success
        assert agent.agent_id in orchestrator.managed_agents
        assert orchestrator.state.agents_managed == 1
    
    async def test_register_duplicate_agent(self, orchestrator, mock_agents):
        """Test registering duplicate agent."""
        agent = mock_agents["collector"]
        
        # Register first time
        success1 = await orchestrator.register_agent(agent)
        assert success1
        
        # Try to register again
        success2 = await orchestrator.register_agent(agent)
        assert not success2  # Should fail
        assert orchestrator.state.agents_managed == 1  # Count unchanged
    
    async def test_unregister_agent(self, orchestrator, mock_agents):
        """Test agent unregistration."""
        agent = mock_agents["collector"]
        
        # Register agent first
        await orchestrator.register_agent(agent)
        assert orchestrator.state.agents_managed == 1
        
        # Unregister agent
        success = await orchestrator.unregister_agent(agent.agent_id)
        assert success
        assert agent.agent_id not in orchestrator.managed_agents
        assert orchestrator.state.agents_managed == 0
    
    async def test_unregister_nonexistent_agent(self, orchestrator):
        """Test unregistering non-existent agent."""
        success = await orchestrator.unregister_agent("nonexistent")
        assert not success


class TestAgentStartStop:
    """Test individual agent start/stop operations."""
    
    async def test_start_agent(self, orchestrator, mock_agents):
        """Test starting individual agent."""
        agent = mock_agents["collector"]
        await orchestrator.register_agent(agent)
        
        success = await orchestrator.start_agent(agent.agent_id)
        assert success
        assert agent.is_running
        assert orchestrator.state.agents_running == 1
    
    async def test_start_agent_failure(self, orchestrator, mock_agents):
        """Test agent start failure."""
        agent = mock_agents["collector"]
        agent.set_start_success(False)  # Make start fail
        await orchestrator.register_agent(agent)
        
        success = await orchestrator.start_agent(agent.agent_id)
        assert not success
        assert not agent.is_running
        assert orchestrator.startup_attempts[agent.agent_id] == 1
    
    async def test_start_unregistered_agent(self, orchestrator):
        """Test starting unregistered agent."""
        success = await orchestrator.start_agent("nonexistent")
        assert not success
    
    async def test_start_already_running_agent(self, orchestrator, mock_agents):
        """Test starting already running agent."""
        agent = mock_agents["collector"]
        await orchestrator.register_agent(agent)
        await orchestrator.start_agent(agent.agent_id)
        
        # Try to start again
        success = await orchestrator.start_agent(agent.agent_id)
        assert success  # Should return True for already running
    
    async def test_stop_agent(self, orchestrator, mock_agents):
        """Test stopping individual agent."""
        agent = mock_agents["collector"]
        await orchestrator.register_agent(agent)
        await orchestrator.start_agent(agent.agent_id)
        
        success = await orchestrator.stop_agent(agent.agent_id)
        assert success
        assert agent.is_stopped
        assert orchestrator.state.agents_running == 0
    
    async def test_stop_unregistered_agent(self, orchestrator):
        """Test stopping unregistered agent."""
        success = await orchestrator.stop_agent("nonexistent")
        assert not success
    
    async def test_stop_not_running_agent(self, orchestrator, mock_agents):
        """Test stopping agent that's not running."""
        agent = mock_agents["collector"]
        await orchestrator.register_agent(agent)
        
        success = await orchestrator.stop_agent(agent.agent_id)
        assert success  # Should return True for already stopped


class TestStartupSequencing:
    """Test agent startup sequencing policies."""
    
    async def test_sequential_startup(self, orchestrator, mock_agents):
        """Test sequential startup policy."""
        orchestrator.config.startup.policy = StartupPolicy.SEQUENTIAL
        
        # Register multiple agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
        
        # Set up the mock registry to return the proper startup order
        agent_ids = list(orchestrator.managed_agents.keys())
        orchestrator.registry.get_dependency_graph.return_value.startup_order = agent_ids
        
        success = await orchestrator.start_all_agents()
        assert success
        
        # All agents should be running
        for agent in mock_agents.values():
            assert agent.is_running
    
    async def test_parallel_startup(self, orchestrator, mock_agents):
        """Test parallel startup policy."""
        orchestrator.config.startup.policy = StartupPolicy.PARALLEL
        
        # Register multiple agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
        
        # Set up the mock registry to return the proper startup order
        agent_ids = list(orchestrator.managed_agents.keys())
        orchestrator.registry.get_dependency_graph.return_value.startup_order = agent_ids
        
        success = await orchestrator.start_all_agents()
        assert success
        
        # All agents should be running
        for agent in mock_agents.values():
            assert agent.is_running
    
    async def test_batch_startup(self, orchestrator, mock_agents):
        """Test batch startup policy."""
        orchestrator.config.startup.policy = StartupPolicy.BATCH
        orchestrator.config.startup.batch_size = 2
        
        # Register multiple agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
        
        # Set up the mock registry to return the proper startup order
        agent_ids = list(orchestrator.managed_agents.keys())
        orchestrator.registry.get_dependency_graph.return_value.startup_order = agent_ids
        
        success = await orchestrator.start_all_agents()
        assert success
        
        # All agents should be running
        for agent in mock_agents.values():
            assert agent.is_running
    
    async def test_manual_startup(self, orchestrator, mock_agents):
        """Test manual startup policy."""
        orchestrator.config.startup.policy = StartupPolicy.MANUAL
        
        # Register multiple agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
        
        success = await orchestrator.start_all_agents()
        assert success  # Should succeed but not start agents
        
        # No agents should be running
        for agent in mock_agents.values():
            assert not agent.is_running


class TestShutdownSequencing:
    """Test agent shutdown sequencing policies."""
    
    async def test_graceful_shutdown(self, orchestrator, mock_agents):
        """Test graceful shutdown policy."""
        orchestrator.config.shutdown.policy = ShutdownPolicy.GRACEFUL
        
        # Register and start agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
        
        # Set up the mock registry to return the proper startup order
        agent_ids = list(orchestrator.managed_agents.keys())
        orchestrator.registry.get_dependency_graph.return_value.startup_order = agent_ids
        
        # Start all agents first
        for agent in mock_agents.values():
            await orchestrator.start_agent(agent.agent_id)
        
        success = await orchestrator.stop_all_agents()
        assert success
        
        # All agents should be stopped
        for agent in mock_agents.values():
            assert agent.is_stopped
    
    async def test_immediate_shutdown(self, orchestrator, mock_agents):
        """Test immediate shutdown policy."""
        orchestrator.config.shutdown.policy = ShutdownPolicy.IMMEDIATE
        
        # Register and start agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
        
        # Set up the mock registry to return the proper startup order
        agent_ids = list(orchestrator.managed_agents.keys())
        orchestrator.registry.get_dependency_graph.return_value.startup_order = agent_ids
        
        # Start all agents first
        for agent in mock_agents.values():
            await orchestrator.start_agent(agent.agent_id)
        
        success = await orchestrator.stop_all_agents()
        assert success
        
        # All agents should be stopped
        for agent in mock_agents.values():
            assert agent.is_stopped
    
    async def test_timeout_shutdown(self, orchestrator, mock_agents):
        """Test timeout shutdown policy."""
        orchestrator.config.shutdown.policy = ShutdownPolicy.TIMEOUT
        orchestrator.config.shutdown.graceful_timeout = timedelta(seconds=1)
        
        # Register and start agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
        
        # Set up the mock registry to return the proper startup order
        agent_ids = list(orchestrator.managed_agents.keys())
        orchestrator.registry.get_dependency_graph.return_value.startup_order = agent_ids
        
        # Start all agents first
        for agent in mock_agents.values():
            await orchestrator.start_agent(agent.agent_id)
        
        success = await orchestrator.stop_all_agents()
        assert success
        
        # All agents should be stopped
        for agent in mock_agents.values():
            assert agent.is_stopped


class TestEmergencyStop:
    """Test emergency stop functionality."""
    
    async def test_emergency_stop(self, orchestrator, mock_agents):
        """Test emergency stop of all agents."""
        # Register and start agents
        for agent in mock_agents.values():
            await orchestrator.register_agent(agent)
            await orchestrator.start_agent(agent.agent_id)
        
        await orchestrator.emergency_stop()
        
        # Orchestrator should be stopped
        assert not orchestrator._running
        assert orchestrator.state.state == AgentState.STOPPED
        
        # All agents should be stopped
        for agent in mock_agents.values():
            assert agent.is_stopped


class TestResourceManager:
    """Test resource management functionality."""
    
    def test_resource_manager_initialization(self, orchestrator_config):
        """Test resource manager initializes correctly."""
        rm = ResourceManager(orchestrator_config)
        assert rm.config == orchestrator_config
        assert len(rm.allocations) == 0
    
    def test_allocate_resources(self, orchestrator_config):
        """Test resource allocation."""
        rm = ResourceManager(orchestrator_config)
        requirements = AgentResourceRequirements(
            agent_type=AgentType.COLLECTOR,
            cpu_cores=2.0,
            memory_mb=1024.0
        )
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0
            
            success = rm.allocate_resources("test_agent", requirements)
            assert success
            assert "test_agent" in rm.allocations
    
    def test_deallocate_resources(self, orchestrator_config):
        """Test resource deallocation."""
        rm = ResourceManager(orchestrator_config)
        requirements = AgentResourceRequirements(agent_type=AgentType.COLLECTOR)
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0
            
            rm.allocate_resources("test_agent", requirements)
            rm.deallocate_resources("test_agent")
            assert "test_agent" not in rm.allocations
    
    def test_get_current_usage(self, orchestrator_config):
        """Test getting current resource usage."""
        rm = ResourceManager(orchestrator_config)
        
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 65.0
            mock_disk.return_value.percent = 80.0
            
            usage = rm.get_current_usage()
            assert usage[ResourceType.CPU] == 75.0
            assert usage[ResourceType.MEMORY] == 65.0
            assert usage[ResourceType.STORAGE] == 80.0


class TestLoadBalancer:
    """Test load balancing functionality."""
    
    def test_load_balancer_initialization(self, orchestrator_config):
        """Test load balancer initializes correctly."""
        lb = LoadBalancer(orchestrator_config)
        assert lb.config == orchestrator_config
        assert len(lb.agent_instances) == 0
    
    def test_register_instance(self, orchestrator_config):
        """Test registering agent instance."""
        lb = LoadBalancer(orchestrator_config)
        
        lb.register_instance("agent1", AgentType.COLLECTOR)
        assert "agent1" in lb.agent_instances[AgentType.COLLECTOR]
    
    def test_unregister_instance(self, orchestrator_config):
        """Test unregistering agent instance."""
        lb = LoadBalancer(orchestrator_config)
        
        lb.register_instance("agent1", AgentType.COLLECTOR)
        lb.unregister_instance("agent1", AgentType.COLLECTOR)
        assert "agent1" not in lb.agent_instances[AgentType.COLLECTOR]
    
    def test_round_robin_selection(self, orchestrator_config):
        """Test round-robin instance selection."""
        orchestrator_config.load_balancing.strategy = LoadBalancingStrategy.ROUND_ROBIN
        lb = LoadBalancer(orchestrator_config)
        
        # Register multiple instances
        lb.register_instance("agent1", AgentType.COLLECTOR)
        lb.register_instance("agent2", AgentType.COLLECTOR)
        lb.register_instance("agent3", AgentType.COLLECTOR)
        
        # Mock time to advance by 1 second each call to ensure different selections
        import time
        selections = []
        with patch('time.time') as mock_time:
            mock_time.side_effect = [i for i in range(6)]  # Return 0, 1, 2, 3, 4, 5
            
            for i in range(6):
                selected = lb.select_instance(AgentType.COLLECTOR)
                selections.append(selected)
        
        # Should have all 3 different instances selected (cycling through them)
        assert len(set(selections)) == 3  # Should select all different instances
    
    def test_least_load_selection(self, orchestrator_config):
        """Test least load instance selection."""
        orchestrator_config.load_balancing.strategy = LoadBalancingStrategy.LEAST_LOAD
        lb = LoadBalancer(orchestrator_config)
        
        # Register instances
        lb.register_instance("agent1", AgentType.COLLECTOR)
        lb.register_instance("agent2", AgentType.COLLECTOR)
        
        # Set different load metrics
        lb.update_load_metrics("agent1", {"health_score": 0.9, "avg_response_time": 0.1})
        lb.update_load_metrics("agent2", {"health_score": 0.7, "avg_response_time": 0.3})
        
        # Should select agent1 (better health, lower response time)
        selected = lb.select_instance(AgentType.COLLECTOR)
        assert selected == "agent1"
    
    def test_no_instances_available(self, orchestrator_config):
        """Test selection when no instances available."""
        lb = LoadBalancer(orchestrator_config)
        
        selected = lb.select_instance(AgentType.COLLECTOR)
        assert selected is None


class TestFailureHandling:
    """Test failure handling and recovery."""
    
    async def test_failure_recovery_restart(self, orchestrator, mock_agents):
        """Test agent restart on failure."""
        orchestrator.config.failure_handling.default_policy = FailurePolicy.RESTART
        orchestrator.config.failure_handling.max_restart_attempts = 2
        
        agent = mock_agents["collector"]
        await orchestrator.register_agent(agent)
        await orchestrator.start_agent(agent.agent_id)
        
        # Simulate agent failure
        agent.state = AgentState.ERROR
        
        # Handle failure
        await orchestrator._handle_agent_failure(agent.agent_id, agent)
        
        # Should attempt restart
        assert orchestrator.failure_counts[agent.agent_id] == 1
    
    async def test_failure_recovery_max_attempts(self, orchestrator, mock_agents):
        """Test max restart attempts exceeded."""
        orchestrator.config.failure_handling.default_policy = FailurePolicy.RESTART
        orchestrator.config.failure_handling.max_restart_attempts = 1
        
        agent = mock_agents["collector"]
        agent.set_start_success(False)  # Make restart fail
        await orchestrator.register_agent(agent)
        
        # Simulate multiple failures - set to max attempts
        orchestrator.failure_counts[agent.agent_id] = 1  # At max attempts
        
        await orchestrator._handle_agent_failure(agent.agent_id, agent)
        
        # Should increment count but not attempt restart since it exceeds max
        assert orchestrator.failure_counts[agent.agent_id] == 2
    
    async def test_critical_agent_failure_shutdown(self, orchestrator, mock_agents):
        """Test system shutdown on critical agent failure."""
        orchestrator.config.failure_handling.agent_policies[AgentType.COLLECTOR] = FailurePolicy.SHUTDOWN
        orchestrator.config.failure_handling.critical_agents = [AgentType.COLLECTOR]
        
        agent = mock_agents["collector"]
        await orchestrator.register_agent(agent)
        await orchestrator.start_agent(agent.agent_id)
        
        # Start orchestrator
        await orchestrator.start()
        
        # Simulate critical agent failure
        agent.state = AgentState.ERROR
        
        # Handle failure - should trigger emergency stop
        await orchestrator._handle_agent_failure(agent.agent_id, agent)
        
        # System should be stopped
        assert not orchestrator._running


class TestSystemStatus:
    """Test system status reporting."""
    
    async def test_get_system_status(self, orchestrator, mock_agents):
        """Test getting comprehensive system status."""
        # Register and start some agents
        for agent in list(mock_agents.values())[:2]:
            await orchestrator.register_agent(agent)
            await orchestrator.start_agent(agent.agent_id)
        
        await orchestrator.start()
        
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 65.0
            mock_disk.return_value.percent = 80.0
            
            status = await orchestrator.get_system_status()
            
            assert "orchestrator" in status
            assert "agents" in status
            assert "resources" in status
            assert "load_balancing" in status
            assert "recent_events" in status
            
            assert status["orchestrator"]["state"] == AgentState.RUNNING.value
            assert status["agents"]["managed"] == 2
            assert status["agents"]["running"] == 2
    
    async def test_system_status_error_handling(self, orchestrator):
        """Test system status with error conditions."""
        # Force an error in resource usage
        with patch.object(orchestrator.resource_manager, 'get_current_usage', side_effect=Exception("Test error")):
            status = await orchestrator.get_system_status()
            assert "error" in status


class TestEventHandling:
    """Test event emission and handling."""
    
    async def test_event_emission(self, orchestrator):
        """Test event emission."""
        events_received = []
        
        def event_callback(event):
            events_received.append(event)
        
        orchestrator.add_event_callback(event_callback)
        
        await orchestrator._emit_event("test_event", "test_agent", "INFO", "Test message")
        
        assert len(events_received) == 1
        assert events_received[0].event_type == "test_event"
        assert events_received[0].agent_id == "test_agent"
        assert events_received[0].severity == "INFO"
        assert events_received[0].message == "Test message"
    
    async def test_event_callback_error(self, orchestrator):
        """Test event callback error handling."""
        def failing_callback(event):
            raise Exception("Callback error")
        
        orchestrator.add_event_callback(failing_callback)
        
        # Should not raise exception
        await orchestrator._emit_event("test_event", None, "INFO", "Test message")


class TestDependencyHandling:
    """Test dependency checking and handling."""
    
    async def test_check_agent_dependencies_satisfied(self, orchestrator, mock_registry):
        """Test dependency checking when dependencies are satisfied."""
        # Mock registry responses
        mock_registration = Mock()
        mock_registration.dependencies = [
            Mock(required=True, depends_on="dep_agent")
        ]
        mock_registry.get_agent.side_effect = [
            mock_registration,  # For the agent itself
            Mock(state=AgentState.RUNNING)  # For the dependency
        ]
        
        result = await orchestrator._check_agent_dependencies("test_agent")
        assert result is True
    
    async def test_check_agent_dependencies_not_satisfied(self, orchestrator, mock_registry):
        """Test dependency checking when dependencies are not satisfied."""
        # Mock registry responses
        mock_registration = Mock()
        mock_registration.dependencies = [
            Mock(required=True, depends_on="dep_agent")
        ]
        mock_registry.get_agent.side_effect = [
            mock_registration,  # For the agent itself
            Mock(state=AgentState.STOPPED)  # For the dependency (not running)
        ]
        
        result = await orchestrator._check_agent_dependencies("test_agent")
        assert result is False
    
    async def test_check_agent_dependencies_no_registry(self, orchestrator_config):
        """Test dependency checking without registry."""
        orchestrator = AgentOrchestrator(config=orchestrator_config)
        
        result = await orchestrator._check_agent_dependencies("test_agent")
        assert result is True  # Should return True when no registry available


class TestMessageBusIntegration:
    """Test message bus integration."""
    
    async def test_message_subscriptions(self, orchestrator, mock_message_bus):
        """Test setting up message bus subscriptions."""
        await orchestrator.start()
        
        # Should have subscribed to agent lifecycle events
        mock_message_bus.subscribe.assert_called_once()
    
    async def test_handle_agent_message(self, orchestrator):
        """Test handling agent messages."""
        # Create mock message
        mock_message = Mock()
        mock_message.message_type.value = "AGENT_STARTED"
        mock_message.payload.agent_id = "test_agent"
        
        # Should not raise exception
        await orchestrator._handle_agent_message(mock_message)


if __name__ == "__main__":
    pytest.main([__file__]) 