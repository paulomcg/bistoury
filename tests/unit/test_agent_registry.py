"""
Tests for Agent Registry and Discovery System

Comprehensive test suite covering agent registration, discovery, dependency resolution,
health monitoring, and all registry operations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
from unittest.mock import AsyncMock, MagicMock

from src.bistoury.agents.registry import AgentRegistry
from src.bistoury.models.agent_registry import (
    AgentRegistration, AgentDiscoveryQuery, AgentCapability, 
    AgentCapabilityType, AgentDependency, AgentCompatibility
)
from src.bistoury.agents.base import AgentType, AgentState, AgentHealth


@pytest.fixture
def sample_compatibility():
    """Sample compatibility info."""
    return AgentCompatibility(
        agent_version="1.0.0",
        framework_version="1.0.0", 
        python_version="3.9.0"
    )


@pytest.fixture
def sample_capability():
    """Sample agent capability."""
    return AgentCapability(
        type=AgentCapabilityType.DATA_COLLECTION,
        description="Collects market data from exchanges"
    )


@pytest.fixture
def sample_agent_registration(sample_compatibility, sample_capability):
    """Sample agent registration."""
    return AgentRegistration(
        agent_id="test_collector_001",
        name="Test Data Collector",
        agent_type=AgentType.COLLECTOR,
        description="Test agent for data collection",
        capabilities=[sample_capability],
        provided_services=["market_data", "order_book"],
        required_services=["database"],
        host="localhost",
        port=8080,
        compatibility=sample_compatibility,
        configuration={"symbols": ["BTC", "ETH"]},
        metadata={"version": "1.0.0"}
    )


@pytest.fixture
def registry():
    """Create a fresh registry for testing."""
    return AgentRegistry()


@pytest.fixture
async def started_registry():
    """Create and start a registry for testing."""
    reg = AgentRegistry()
    await reg.start()
    yield reg
    await reg.stop()


class TestAgentRegistryBasics:
    """Test basic registry operations."""
    
    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        registry = AgentRegistry()
        
        assert not registry._running
        assert registry._start_time is None
        assert len(registry._agents) == 0
        assert len(registry._capabilities_index) == 0
        assert len(registry._services_index) == 0
    
    async def test_registry_start_stop(self, registry):
        """Test registry start and stop operations."""
        # Start registry
        await registry.start()
        assert registry._running
        assert registry._start_time is not None
        assert registry._cleanup_task is not None
        
        # Stop registry
        await registry.stop()
        assert not registry._running


class TestAgentRegistration:
    """Test agent registration operations."""
    
    async def test_register_agent_success(self, started_registry, sample_agent_registration):
        """Test successful agent registration."""
        result = await started_registry.register_agent(sample_agent_registration)
        
        assert result is True
        assert sample_agent_registration.agent_id in started_registry._agents
        
        # Check indices are updated
        capability_type = sample_agent_registration.capabilities[0].type
        assert sample_agent_registration.agent_id in started_registry._capabilities_index[capability_type]
        
        for service in sample_agent_registration.provided_services:
            assert sample_agent_registration.agent_id in started_registry._services_index[service]
    
    async def test_register_duplicate_agent(self, started_registry, sample_agent_registration):
        """Test registering duplicate agent fails."""
        # Register first time
        result1 = await started_registry.register_agent(sample_agent_registration)
        assert result1 is True
        
        # Try to register again
        result2 = await started_registry.register_agent(sample_agent_registration)
        assert result2 is False
    
    async def test_register_agent_replace_existing(self, started_registry, sample_agent_registration):
        """Test replacing existing agent registration."""
        # Register first time
        await started_registry.register_agent(sample_agent_registration)
        
        # Modify and register again with replace flag
        sample_agent_registration.description = "Updated description"
        result = await started_registry.register_agent(sample_agent_registration, replace_existing=True)
        
        assert result is True
        assert started_registry._agents[sample_agent_registration.agent_id].description == "Updated description"
    
    async def test_register_agent_validation_failure(self, started_registry, sample_compatibility):
        """Test registration fails with invalid data."""
        invalid_registration = AgentRegistration(
            agent_id="",  # Empty agent ID should fail validation
            name="Test Agent",
            agent_type=AgentType.COLLECTOR,
            description="Test agent",
            compatibility=sample_compatibility
        )
        
        result = await started_registry.register_agent(invalid_registration)
        assert result is False
    
    async def test_deregister_agent_success(self, started_registry, sample_agent_registration):
        """Test successful agent deregistration."""
        # Register first
        await started_registry.register_agent(sample_agent_registration)
        
        # Deregister
        result = await started_registry.deregister_agent(sample_agent_registration.agent_id)
        
        assert result is True
        assert sample_agent_registration.agent_id not in started_registry._agents
        
        # Check indices are cleaned up
        capability_type = sample_agent_registration.capabilities[0].type
        assert sample_agent_registration.agent_id not in started_registry._capabilities_index[capability_type]
    
    async def test_deregister_nonexistent_agent(self, started_registry):
        """Test deregistering non-existent agent fails gracefully."""
        result = await started_registry.deregister_agent("nonexistent_agent")
        assert result is False


class TestAgentHealth:
    """Test agent health monitoring."""
    
    async def test_update_agent_health(self, started_registry, sample_agent_registration):
        """Test updating agent health status."""
        # Register agent first
        await started_registry.register_agent(sample_agent_registration)
        
        # Update health
        health = AgentHealth(
            state=AgentState.RUNNING,
            health_score=0.95,
            cpu_usage=0.25,
            memory_usage=0.40,
            error_count=0,
            last_error=None
        )
        
        result = await started_registry.update_agent_health(sample_agent_registration.agent_id, health)
        
        assert result is True
        registered_agent = started_registry._agents[sample_agent_registration.agent_id]
        assert registered_agent.health.health_score == 0.95
        assert registered_agent.last_heartbeat is not None
    
    async def test_update_health_nonexistent_agent(self, started_registry):
        """Test updating health of non-existent agent fails."""
        health = AgentHealth(state=AgentState.RUNNING, health_score=0.95)
        result = await started_registry.update_agent_health("nonexistent", health)
        assert result is False


class TestAgentDiscovery:
    """Test agent discovery functionality."""
    
    async def test_discover_all_agents(self, started_registry, sample_agent_registration):
        """Test discovering all agents."""
        # Register multiple agents
        await started_registry.register_agent(sample_agent_registration)
        
        # Second agent with different type
        second_agent = sample_agent_registration.model_copy()
        second_agent.agent_id = "test_trader_001"
        second_agent.agent_type = AgentType.TRADER
        await started_registry.register_agent(second_agent)
        
        # Discover all
        query = AgentDiscoveryQuery(healthy_only=False, exclude_expired=False)
        result = await started_registry.discover_agents(query)
        
        assert result.total_count == 2
        assert len(result.agents) == 2
        assert result.query_time_ms > 0
    
    async def test_discover_by_agent_type(self, started_registry, sample_agent_registration):
        """Test discovering agents by type."""
        await started_registry.register_agent(sample_agent_registration)
        
        # Query for collectors only
        query = AgentDiscoveryQuery(
            agent_types=[AgentType.COLLECTOR],
            healthy_only=False,
            exclude_expired=False
        )
        result = await started_registry.discover_agents(query)
        
        assert result.total_count == 1
        assert result.agents[0].agent_type == AgentType.COLLECTOR
    
    async def test_discover_by_capabilities(self, started_registry, sample_agent_registration):
        """Test discovering agents by capabilities."""
        await started_registry.register_agent(sample_agent_registration)
        
        # Query for data collection capability
        query = AgentDiscoveryQuery(
            capabilities=[AgentCapabilityType.DATA_COLLECTION],
            healthy_only=False,
            exclude_expired=False
        )
        result = await started_registry.discover_agents(query)
        
        assert result.total_count == 1
        assert result.agents[0].has_capability(AgentCapabilityType.DATA_COLLECTION)
    
    async def test_discover_by_services(self, started_registry, sample_agent_registration):
        """Test discovering agents by provided services."""
        await started_registry.register_agent(sample_agent_registration)
        
        # Query for market data service
        query = AgentDiscoveryQuery(
            services=["market_data"],
            healthy_only=False,
            exclude_expired=False
        )
        result = await started_registry.discover_agents(query)
        
        assert result.total_count == 1
        assert "market_data" in result.agents[0].provided_services
    
    async def test_discover_healthy_only(self, started_registry, sample_agent_registration):
        """Test discovering only healthy agents."""
        # Register agent
        await started_registry.register_agent(sample_agent_registration)
        
        # Set unhealthy status
        unhealthy = AgentHealth(state=AgentState.ERROR, health_score=0.3)  # Below healthy threshold
        await started_registry.update_agent_health(sample_agent_registration.agent_id, unhealthy)
        
        # Query for healthy only
        query = AgentDiscoveryQuery(healthy_only=True, exclude_expired=False)
        result = await started_registry.discover_agents(query)
        
        assert result.total_count == 0  # Should filter out unhealthy agent
    
    async def test_discover_with_limit(self, started_registry):
        """Test discovery with result limits."""
        # Register multiple agents
        for i in range(5):
            agent = AgentRegistration(
                agent_id=f"agent_{i:03d}",
                name=f"Agent {i}",
                agent_type=AgentType.COLLECTOR,
                description=f"Test agent {i}",
                compatibility=AgentCompatibility(
                    agent_version="1.0.0",
                    framework_version="1.0.0",
                    python_version="3.9.0"
                )
            )
            await started_registry.register_agent(agent)
        
        # Query with limit
        query = AgentDiscoveryQuery(limit=3, healthy_only=False, exclude_expired=False)
        result = await started_registry.discover_agents(query)
        
        assert result.total_count == 5  # Total available
        assert len(result.agents) == 3  # Limited results


class TestDependencyManagement:
    """Test dependency resolution and management."""
    
    async def test_get_dependency_graph_empty(self, started_registry):
        """Test dependency graph with no agents."""
        graph = await started_registry.get_dependency_graph()
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.startup_order) == 0
    
    async def test_dependency_graph_with_agents(self, started_registry):
        """Test dependency graph with multiple agents."""
        # Create and register agents
        collector = AgentRegistration(
            agent_id="collector_001",
            name="Data Collector",
            agent_type=AgentType.COLLECTOR,
            description="Collects data",
            compatibility=AgentCompatibility(
                agent_version="1.0.0",
                framework_version="1.0.0",
                python_version="3.9.0"
            )
        )
        
        trader = AgentRegistration(
            agent_id="trader_001", 
            name="Trader Agent",
            agent_type=AgentType.TRADER,
            description="Executes trades",
            compatibility=AgentCompatibility(
                agent_version="1.0.0",
                framework_version="1.0.0",
                python_version="3.9.0"
            )
        )
        
        await started_registry.register_agent(collector)
        await started_registry.register_agent(trader)
        
        # Add dependency: trader depends on collector
        dependency = AgentDependency(
            agent_id="trader_001",
            depends_on="collector_001",
            dependency_type="startup",
            required=True
        )
        started_registry._dependencies["trader_001"].append(dependency)
        
        # Get dependency graph
        graph = await started_registry.get_dependency_graph()
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert "collector_001" in graph.startup_order
        assert "trader_001" in graph.startup_order
        assert graph.startup_order.index("collector_001") < graph.startup_order.index("trader_001")
    
    async def test_circular_dependency_detection(self, started_registry):
        """Test detection of circular dependencies."""
        # Create agents
        agent_a = AgentRegistration(
            agent_id="agent_a",
            name="Agent A",
            agent_type=AgentType.COLLECTOR,
            description="Agent A",
            compatibility=AgentCompatibility(
                agent_version="1.0.0",
                framework_version="1.0.0",
                python_version="3.9.0"
            )
        )
        
        agent_b = AgentRegistration(
            agent_id="agent_b",
            name="Agent B", 
            agent_type=AgentType.TRADER,
            description="Agent B",
            compatibility=AgentCompatibility(
                agent_version="1.0.0",
                framework_version="1.0.0",
                python_version="3.9.0"
            )
        )
        
        await started_registry.register_agent(agent_a)
        await started_registry.register_agent(agent_b)
        
        # Create circular dependency
        dep_a_to_b = AgentDependency(
            agent_id="agent_a",
            depends_on="agent_b",
            dependency_type="startup",
            required=True
        )
        dep_b_to_a = AgentDependency(
            agent_id="agent_b",
            depends_on="agent_a",
            dependency_type="startup",
            required=True
        )
        
        started_registry._dependencies["agent_a"].append(dep_a_to_b)
        started_registry._dependencies["agent_b"].append(dep_b_to_a)
        
        # Get dependency graph
        graph = await started_registry.get_dependency_graph()
        
        assert len(graph.circular_dependencies) > 0


class TestRegistryStatistics:
    """Test registry statistics and metrics."""
    
    async def test_get_statistics_empty_registry(self, started_registry):
        """Test statistics for empty registry."""
        stats = await started_registry.get_statistics()
        
        assert stats.total_agents == 0
        assert stats.active_agents == 0
        assert stats.healthy_agents == 0
        assert stats.expired_agents == 0
        assert stats.registry_uptime_seconds > 0
    
    async def test_get_statistics_with_agents(self, started_registry, sample_agent_registration):
        """Test statistics with registered agents."""
        # Register agent
        await started_registry.register_agent(sample_agent_registration)
        
        # Update to running state with health
        await started_registry.update_agent_state(
            sample_agent_registration.agent_id, 
            AgentState.RUNNING
        )
        health = AgentHealth(state=AgentState.RUNNING, health_score=0.95)
        await started_registry.update_agent_health(sample_agent_registration.agent_id, health)
        
        # Get statistics
        stats = await started_registry.get_statistics()
        
        assert stats.total_agents == 1
        assert stats.active_agents == 1
        assert stats.healthy_agents == 1
        assert stats.agents_by_type[AgentType.COLLECTOR.value] == 1
        assert stats.agents_by_state[AgentState.RUNNING.value] == 1
        assert stats.average_health_score == 0.95


class TestPersistence:
    """Test registry state persistence."""
    
    async def test_save_and_load_state(self, sample_agent_registration):
        """Test saving and loading registry state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_path = Path(temp_dir) / "registry_state.json"
            
            # Create registry with persistence
            registry = AgentRegistry(persistence_path=persistence_path)
            await registry.start()
            
            # Register an agent
            await registry.register_agent(sample_agent_registration)
            
            # Stop registry (should save state)
            await registry.stop()
            
            # Verify file was created
            assert persistence_path.exists()
            
            # Create new registry and load state
            new_registry = AgentRegistry(persistence_path=persistence_path)
            await new_registry.start()
            
            # Verify agent was restored
            assert sample_agent_registration.agent_id in new_registry._agents
            restored_agent = new_registry._agents[sample_agent_registration.agent_id]
            assert restored_agent.name == sample_agent_registration.name
            
            await new_registry.stop()


class TestRegistryEvents:
    """Test registry event system."""
    
    async def test_event_emission(self, started_registry, sample_agent_registration):
        """Test that events are emitted correctly."""
        events_received = []
        
        def event_callback(event):
            events_received.append(event)
        
        # Add callback
        started_registry.add_event_callback(event_callback)
        
        # Register agent (should emit event)
        await started_registry.register_agent(sample_agent_registration)
        
        # Check event was received
        assert len(events_received) > 0
        registration_event = next(
            (e for e in events_received if e.event_type == "agent_registered"), 
            None
        )
        assert registration_event is not None
        assert registration_event.agent_id == sample_agent_registration.agent_id
    
    async def test_event_callback_removal(self, started_registry):
        """Test removing event callbacks."""
        events_received = []
        
        def event_callback(event):
            events_received.append(event)
        
        # Add and remove callback
        started_registry.add_event_callback(event_callback)
        started_registry.remove_event_callback(event_callback)
        
        # Emit an event (using valid event type)
        await started_registry._emit_event("error", "test_agent", {})
        
        # Should not receive any events
        assert len(events_received) == 0


class TestRegistryHelperMethods:
    """Test registry helper and utility methods."""
    
    async def test_update_agent_state(self, started_registry, sample_agent_registration):
        """Test updating agent state."""
        # Register agent
        await started_registry.register_agent(sample_agent_registration)
        
        # Update state
        result = await started_registry.update_agent_state(
            sample_agent_registration.agent_id,
            AgentState.RUNNING
        )
        
        assert result is True
        assert started_registry._agents[sample_agent_registration.agent_id].state == AgentState.RUNNING
    
    async def test_get_agent(self, started_registry, sample_agent_registration):
        """Test getting specific agent."""
        # Register agent
        await started_registry.register_agent(sample_agent_registration)
        
        # Get agent
        retrieved = await started_registry.get_agent(sample_agent_registration.agent_id)
        
        assert retrieved is not None
        assert retrieved.agent_id == sample_agent_registration.agent_id
    
    async def test_get_all_agents(self, started_registry, sample_agent_registration):
        """Test getting all agents."""
        # Register agent
        await started_registry.register_agent(sample_agent_registration)
        
        # Get all agents
        all_agents = await started_registry.get_all_agents()
        
        assert len(all_agents) == 1
        assert all_agents[0].agent_id == sample_agent_registration.agent_id


@pytest.mark.asyncio
class TestRegistryCleanup:
    """Test registry cleanup operations."""
    
    async def test_cleanup_expired_registrations(self):
        """Test cleanup of expired registrations."""
        registry = AgentRegistry(cleanup_interval=0.1)  # Fast cleanup for testing
        await registry.start()
        
        # Create expired registration
        expired_registration = AgentRegistration(
            agent_id="expired_agent",
            name="Expired Agent",
            agent_type=AgentType.COLLECTOR,
            description="Expired test agent",
            ttl_seconds=60,  # Minimum allowed TTL
            compatibility=AgentCompatibility(
                agent_version="1.0.0",
                framework_version="1.0.0",
                python_version="3.9.0"
            )
        )
        
        # Set expiration in the past
        expired_registration.expires_at = datetime.utcnow() - timedelta(seconds=10)
        
        # Manually add to registry
        registry._agents[expired_registration.agent_id] = expired_registration
        
        # Wait for cleanup
        await asyncio.sleep(0.2)
        
        # Agent should be removed
        assert expired_registration.agent_id not in registry._agents
        
        await registry.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 