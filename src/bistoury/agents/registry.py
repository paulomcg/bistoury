"""
Agent Registry and Discovery System

This module provides the central registry for tracking and discovering agents in the
multi-agent system, including dependency resolution and health monitoring.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from pathlib import Path
import json

from ..models.agent_registry import (
    AgentRegistration, AgentDiscoveryQuery, AgentDiscoveryResult,
    AgentCapability, AgentCapabilityType, AgentDependency,
    RegistryStatistics, DependencyGraph, RegistryEvent,
    AgentCompatibility
)
from ..agents.base import AgentType, AgentState, AgentHealth
from ..agents.messaging import MessageBus, Message, MessageType, MessagePriority

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for tracking and discovering agents in the multi-agent system.
    
    Provides:
    - Agent registration and deregistration
    - Agent discovery and capability queries  
    - Dependency resolution and startup ordering
    - Health monitoring and status tracking
    - Event logging and statistics
    """
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        persistence_path: Optional[Path] = None,
        cleanup_interval: float = 60.0,
        heartbeat_timeout: float = 180.0
    ):
        """
        Initialize the agent registry.
        
        Args:
            message_bus: MessageBus for inter-agent communication
            persistence_path: Path for persisting registry state
            cleanup_interval: Interval for cleanup operations (seconds)
            heartbeat_timeout: Timeout for agent heartbeats (seconds)
        """
        self.message_bus = message_bus
        self.persistence_path = persistence_path
        self.cleanup_interval = cleanup_interval
        self.heartbeat_timeout = heartbeat_timeout
        
        # Registry state
        self._agents: Dict[str, AgentRegistration] = {}
        self._capabilities_index: Dict[AgentCapabilityType, Set[str]] = defaultdict(set)
        self._services_index: Dict[str, Set[str]] = defaultdict(set)
        self._dependencies: Dict[str, List[AgentDependency]] = defaultdict(list)
        
        # Events and statistics
        self._events: deque = deque(maxlen=10000)  # Keep last 10k events
        self._query_metrics: List[float] = []
        self._registration_timestamps: List[datetime] = []
        
        # Runtime state
        self._running = False
        self._start_time: Optional[datetime] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self._event_callbacks: List[Callable[[RegistryEvent], None]] = []
        
        logger.info("Agent registry initialized")
    
    async def start(self) -> None:
        """Start the agent registry."""
        if self._running:
            logger.warning("Registry is already running")
            return
        
        self._running = True
        self._start_time = datetime.utcnow()
        
        # Load persisted state if available
        if self.persistence_path and self.persistence_path.exists():
            await self._load_state()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Subscribe to message bus events if available
        if self.message_bus:
            await self._setup_message_subscriptions()
        
        await self._emit_event("registry_started", "system", {"start_time": self._start_time.isoformat()})
        logger.info("Agent registry started")
    
    async def stop(self) -> None:
        """Stop the agent registry."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save state if persistence is enabled
        if self.persistence_path:
            await self._save_state()
        
        await self._emit_event("registry_stopped", "system", {})
        logger.info("Agent registry stopped")
    
    async def register_agent(
        self,
        registration: AgentRegistration,
        replace_existing: bool = False
    ) -> bool:
        """
        Register an agent in the registry.
        
        Args:
            registration: Agent registration information
            replace_existing: Whether to replace existing registration
            
        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            agent_id = registration.agent_id
            
            # Check if agent already exists
            if agent_id in self._agents and not replace_existing:
                logger.warning(f"Agent {agent_id} already registered")
                return False
            
            # Validate registration
            validation_errors = await self._validate_registration(registration)
            if validation_errors:
                logger.error(f"Registration validation failed for {agent_id}: {validation_errors}")
                return False
            
            # Update indices if replacing existing agent
            if agent_id in self._agents:
                await self._remove_from_indices(agent_id)
            
            # Add to registry
            self._agents[agent_id] = registration
            await self._add_to_indices(registration)
            
            # Track registration timestamp
            self._registration_timestamps.append(registration.registered_at)
            
            # Emit registration event
            await self._emit_event("agent_registered", agent_id, {
                "agent_type": registration.agent_type.value,
                "capabilities": [cap.type.value for cap in registration.capabilities],
                "replace_existing": replace_existing
            })
            
            logger.info(f"Agent {agent_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {registration.agent_id}: {e}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the registry.
        
        Args:
            agent_id: ID of agent to deregister
            
        Returns:
            True if deregistration succeeded, False otherwise
        """
        try:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not found for deregistration")
                return False
            
            # Remove from indices
            await self._remove_from_indices(agent_id)
            
            # Remove agent
            registration = self._agents.pop(agent_id)
            
            # Emit deregistration event
            await self._emit_event("agent_deregistered", agent_id, {
                "agent_type": registration.agent_type.value
            })
            
            logger.info(f"Agent {agent_id} deregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {e}")
            return False
    
    async def update_agent_health(self, agent_id: str, health: AgentHealth) -> bool:
        """
        Update an agent's health status.
        
        Args:
            agent_id: ID of agent
            health: New health status
            
        Returns:
            True if update succeeded, False otherwise
        """
        try:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not found for health update")
                return False
            
            registration = self._agents[agent_id]
            old_health_score = registration.health.health_score if registration.health else 0.0
            
            # Update health and heartbeat
            registration.update_heartbeat(health)
            
            # Emit health update event
            await self._emit_event("agent_health_updated", agent_id, {
                "old_health_score": old_health_score,
                "new_health_score": health.health_score,
                "is_healthy": health.is_healthy()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update health for agent {agent_id}: {e}")
            return False
    
    async def update_agent_state(self, agent_id: str, state: AgentState) -> bool:
        """
        Update an agent's state.
        
        Args:
            agent_id: ID of agent
            state: New agent state
            
        Returns:
            True if update succeeded, False otherwise
        """
        try:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not found for state update")
                return False
            
            registration = self._agents[agent_id]
            old_state = registration.state
            registration.state = state
            
            # Emit state change event
            await self._emit_event("agent_state_changed", agent_id, {
                "old_state": old_state.value,
                "new_state": state.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update state for agent {agent_id}: {e}")
            return False
    
    async def discover_agents(self, query: AgentDiscoveryQuery) -> AgentDiscoveryResult:
        """
        Discover agents matching the given criteria.
        
        Args:
            query: Discovery query parameters
            
        Returns:
            Discovery result with matching agents
        """
        start_time = time.time()
        
        try:
            # Start with all agents
            candidates = list(self._agents.values())
            
            # Apply filters
            if query.agent_types:
                candidates = [a for a in candidates if a.agent_type in query.agent_types]
            
            if query.capabilities:
                candidates = [
                    a for a in candidates 
                    if all(a.has_capability(cap) for cap in query.capabilities)
                ]
            
            if query.services:
                candidates = [
                    a for a in candidates
                    if all(service in a.provided_services for service in query.services)
                ]
            
            if query.states:
                candidates = [a for a in candidates if a.state in query.states]
            
            if query.healthy_only:
                candidates = [a for a in candidates if a.is_healthy()]
            
            if query.exclude_expired:
                candidates = [a for a in candidates if not a.is_expired()]
            
            if query.min_health_score is not None:
                candidates = [
                    a for a in candidates 
                    if a.health and a.health.health_score >= query.min_health_score
                ]
            
            # Sort results
            candidates = await self._sort_agents(candidates, query.sort_by, query.sort_desc)
            
            # Apply limit
            total_count = len(candidates)
            if query.limit:
                candidates = candidates[:query.limit]
            
            # Calculate query time
            query_time_ms = (time.time() - start_time) * 1000
            self._query_metrics.append(query_time_ms)
            
            # Emit discovery event
            await self._emit_event("discovery_query", "system", {
                "total_results": total_count,
                "returned_results": len(candidates),
                "query_time_ms": query_time_ms
            })
            
            return AgentDiscoveryResult(
                query=query,
                agents=candidates,
                total_count=total_count,
                query_time_ms=query_time_ms
            )
            
        except Exception as e:
            logger.error(f"Discovery query failed: {e}")
            await self._emit_event("error", "system", {"error": str(e), "operation": "discovery"})
            raise
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get a specific agent by ID."""
        return self._agents.get(agent_id)
    
    async def get_all_agents(self) -> List[AgentRegistration]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    async def get_agents_by_capability(
        self, 
        capability: AgentCapabilityType
    ) -> List[AgentRegistration]:
        """Get all agents with a specific capability."""
        agent_ids = self._capabilities_index.get(capability, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    async def get_agents_by_service(self, service: str) -> List[AgentRegistration]:
        """Get all agents providing a specific service."""
        agent_ids = self._services_index.get(service, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    async def add_dependency(
        self, 
        agent_id: str, 
        depends_on: str,
        dependency_type: str = "startup",
        required: bool = True,
        timeout_seconds: int = 30
    ) -> bool:
        """Add a dependency between agents."""
        try:
            dependency = AgentDependency(
                agent_id=agent_id,
                depends_on=depends_on,
                dependency_type=dependency_type,
                required=required,
                timeout_seconds=timeout_seconds
            )
            
            # Validate that both agents exist
            if agent_id not in self._agents or depends_on not in self._agents:
                logger.error(f"Cannot add dependency: one or both agents not found")
                return False
            
            # Check for circular dependencies
            if await self._would_create_cycle(agent_id, depends_on):
                logger.error(f"Cannot add dependency: would create circular dependency")
                return False
            
            # Add dependency
            self._dependencies[agent_id].append(dependency)
            
            # Update agent registration
            if agent_id in self._agents:
                self._agents[agent_id].dependencies.append(dependency)
            
            await self._emit_event("dependency_added", agent_id, {
                "depends_on": depends_on,
                "dependency_type": dependency_type,
                "required": required
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False
    
    async def remove_dependency(self, agent_id: str, depends_on: str) -> bool:
        """Remove a dependency between agents."""
        try:
            # Remove from dependencies index
            deps = self._dependencies.get(agent_id, [])
            self._dependencies[agent_id] = [d for d in deps if d.depends_on != depends_on]
            
            # Update agent registration
            if agent_id in self._agents:
                registration = self._agents[agent_id]
                registration.dependencies = [d for d in registration.dependencies if d.depends_on != depends_on]
            
            await self._emit_event("dependency_removed", agent_id, {
                "depends_on": depends_on
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove dependency: {e}")
            return False
    
    async def update_agent_state(self, agent_id: str, state: AgentState) -> bool:
        """
        Update an agent's state.
        
        Args:
            agent_id: ID of agent
            state: New agent state
            
        Returns:
            True if update succeeded, False otherwise
        """
        try:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not found for state update")
                return False
            
            registration = self._agents[agent_id]
            old_state = registration.state
            registration.state = state
            
            # Emit state change event
            await self._emit_event("agent_state_changed", agent_id, {
                "old_state": old_state.value,
                "new_state": state.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update state for agent {agent_id}: {e}")
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get a specific agent by ID."""
        return self._agents.get(agent_id)
    
    async def get_all_agents(self) -> List[AgentRegistration]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    async def get_agents_by_capability(
        self, 
        capability: AgentCapabilityType
    ) -> List[AgentRegistration]:
        """Get all agents with a specific capability."""
        agent_ids = self._capabilities_index.get(capability, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    async def get_agents_by_service(self, service: str) -> List[AgentRegistration]:
        """Get all agents providing a specific service."""
        agent_ids = self._services_index.get(service, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    async def add_dependency(
        self, 
        agent_id: str, 
        depends_on: str,
        dependency_type: str = "startup",
        required: bool = True,
        timeout_seconds: int = 30
    ) -> bool:
        """Add a dependency between agents."""
        try:
            dependency = AgentDependency(
                agent_id=agent_id,
                depends_on=depends_on,
                dependency_type=dependency_type,
                required=required,
                timeout_seconds=timeout_seconds
            )
            
            # Validate that both agents exist
            if agent_id not in self._agents or depends_on not in self._agents:
                logger.error(f"Cannot add dependency: one or both agents not found")
                return False
            
            # Check for circular dependencies
            if await self._would_create_cycle(agent_id, depends_on):
                logger.error(f"Cannot add dependency: would create circular dependency")
                return False
            
            # Add dependency
            self._dependencies[agent_id].append(dependency)
            
            # Update agent registration
            if agent_id in self._agents:
                self._agents[agent_id].dependencies.append(dependency)
            
            await self._emit_event("dependency_added", agent_id, {
                "depends_on": depends_on,
                "dependency_type": dependency_type,
                "required": required
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False
    
    async def remove_dependency(self, agent_id: str, depends_on: str) -> bool:
        """Remove a dependency between agents."""
        try:
            # Remove from dependencies index
            deps = self._dependencies.get(agent_id, [])
            self._dependencies[agent_id] = [d for d in deps if d.depends_on != depends_on]
            
            # Update agent registration
            if agent_id in self._agents:
                registration = self._agents[agent_id]
                registration.dependencies = [d for d in registration.dependencies if d.depends_on != depends_on]
            
            await self._emit_event("dependency_removed", agent_id, {
                "depends_on": depends_on
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove dependency: {e}")
            return False
    
    async def get_dependency_graph(self) -> DependencyGraph:
        """Get the complete dependency graph."""
        # Collect all dependencies
        all_deps = []
        for deps in self._dependencies.values():
            all_deps.extend(deps)
        
        # Build startup order
        startup_order = await self._calculate_startup_order()
        
        # Find circular dependencies
        circular_deps = await self._find_circular_dependencies()
        
        # Find orphaned agents
        orphaned = await self._find_orphaned_agents()
        
        return DependencyGraph(
            nodes=self._agents.copy(),
            edges=all_deps,
            startup_order=startup_order,
            circular_dependencies=circular_deps,
            orphaned_agents=orphaned
        )
    
    async def get_statistics(self) -> RegistryStatistics:
        """Get registry statistics and metrics."""
        now = datetime.utcnow()
        uptime = (now - self._start_time).total_seconds() if self._start_time else 0.0
        
        # Basic counts
        all_agents = list(self._agents.values())
        total_agents = len(all_agents)
        active_agents = len([a for a in all_agents if a.state in [AgentState.RUNNING, AgentState.PAUSED]])
        healthy_agents = len([a for a in all_agents if a.is_healthy()])
        expired_agents = len([a for a in all_agents if a.is_expired()])
        
        # By type and state
        agents_by_type = defaultdict(int)
        agents_by_state = defaultdict(int)
        for agent in all_agents:
            agents_by_type[agent.agent_type.value] += 1
            agents_by_state[agent.state.value] += 1
        
        # Health metrics
        health_scores = [a.health.health_score for a in all_agents if a.health]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0
        
        health_distribution = defaultdict(int)
        for score in health_scores:
            if score >= 0.9:
                health_distribution["excellent"] += 1
            elif score >= 0.7:
                health_distribution["good"] += 1
            elif score >= 0.5:
                health_distribution["fair"] += 1
            else:
                health_distribution["poor"] += 1
        
        # Capability and service counts
        capabilities_count = {cap.value: len(agents) for cap, agents in self._capabilities_index.items()}
        services_count = {service: len(agents) for service, agents in self._services_index.items()}
        
        # Performance metrics
        recent_registrations = [
            ts for ts in self._registration_timestamps 
            if (now - ts).total_seconds() < 3600  # Last hour
        ]
        registration_rate = len(recent_registrations) / 60.0  # Per minute
        
        recent_queries = [t for t in self._query_metrics if t is not None][-100:]  # Last 100 queries
        avg_query_time = sum(recent_queries) / len(recent_queries) if recent_queries else 0.0
        discovery_query_rate = len(recent_queries) / 60.0 if recent_queries else 0.0
        
        return RegistryStatistics(
            total_agents=total_agents,
            active_agents=active_agents,
            healthy_agents=healthy_agents,
            expired_agents=expired_agents,
            agents_by_type=dict(agents_by_type),
            agents_by_state=dict(agents_by_state),
            average_health_score=avg_health,
            health_score_distribution=dict(health_distribution),
            capabilities_count=capabilities_count,
            services_count=services_count,
            registration_rate=registration_rate,
            discovery_query_rate=discovery_query_rate,
            average_query_time_ms=avg_query_time,
            registry_uptime_seconds=uptime
        )
    
    def add_event_callback(self, callback: Callable[[RegistryEvent], None]) -> None:
        """Add a callback for registry events."""
        self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[RegistryEvent], None]) -> None:
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    async def _validate_registration(self, registration: AgentRegistration) -> List[str]:
        """Validate agent registration."""
        errors = []
        
        # Basic validation
        if not registration.agent_id.strip():
            errors.append("Agent ID cannot be empty")
        
        if not registration.name.strip():
            errors.append("Agent name cannot be empty")
        
        # Capability validation
        for capability in registration.capabilities:
            if not capability.description.strip():
                errors.append(f"Capability {capability.type} missing description")
        
        return errors
    
    async def _add_to_indices(self, registration: AgentRegistration) -> None:
        """Add agent to search indices."""
        agent_id = registration.agent_id
        
        # Capabilities index
        for capability in registration.capabilities:
            self._capabilities_index[capability.type].add(agent_id)
        
        # Services index
        for service in registration.provided_services:
            self._services_index[service].add(agent_id)
    
    async def _remove_from_indices(self, agent_id: str) -> None:
        """Remove agent from search indices."""
        if agent_id not in self._agents:
            return
        
        registration = self._agents[agent_id]
        
        # Remove from capabilities index
        for capability in registration.capabilities:
            self._capabilities_index[capability.type].discard(agent_id)
        
        # Remove from services index
        for service in registration.provided_services:
            self._services_index[service].discard(agent_id)
    
    async def _sort_agents(
        self, 
        agents: List[AgentRegistration], 
        sort_by: str, 
        desc: bool
    ) -> List[AgentRegistration]:
        """Sort agents by specified field."""
        def get_sort_key(agent: AgentRegistration) -> Any:
            if sort_by == "registered_at":
                return agent.registered_at
            elif sort_by == "agent_id":
                return agent.agent_id
            elif sort_by == "name":
                return agent.name
            elif sort_by == "agent_type":
                return agent.agent_type.value
            elif sort_by == "health.health_score":
                return agent.health.health_score if agent.health else 0.0
            elif sort_by == "last_heartbeat":
                return agent.last_heartbeat or datetime.min
            else:
                return agent.registered_at
        
        return sorted(agents, key=get_sort_key, reverse=desc)
    
    async def _calculate_startup_order(self) -> List[str]:
        """Calculate recommended startup order based on dependencies."""
        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # Build graph
        for agent_id in self._agents:
            if agent_id not in in_degree:
                in_degree[agent_id] = 0
        
        for deps in self._dependencies.values():
            for dep in deps:
                if dep.required:  # Only consider required dependencies
                    graph[dep.depends_on].append(dep.agent_id)
                    in_degree[dep.agent_id] += 1
        
        # Kahn's algorithm
        queue = deque([agent_id for agent_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            agent_id = queue.popleft()
            result.append(agent_id)
            
            for dependent in graph[agent_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result
    
    async def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(agent_id: str, path: List[str]) -> None:
            if agent_id in rec_stack:
                # Found a cycle
                cycle_start = path.index(agent_id)
                cycles.append(path[cycle_start:] + [agent_id])
                return
            
            if agent_id in visited:
                return
            
            visited.add(agent_id)
            rec_stack.add(agent_id)
            path.append(agent_id)
            
            # Visit dependencies
            for dep in self._dependencies.get(agent_id, []):
                if dep.required:
                    dfs(dep.depends_on, path.copy())
            
            rec_stack.remove(agent_id)
        
        for agent_id in self._agents:
            if agent_id not in visited:
                dfs(agent_id, [])
        
        return cycles
    
    async def _find_orphaned_agents(self) -> List[str]:
        """Find agents with missing dependencies."""
        orphaned = []
        
        for agent_id, deps in self._dependencies.items():
            for dep in deps:
                if dep.required and dep.depends_on not in self._agents:
                    orphaned.append(agent_id)
                    break
        
        return orphaned
    
    async def _would_create_cycle(self, agent_id: str, depends_on: str) -> bool:
        """Check if adding a dependency would create a cycle."""
        # Temporarily add the dependency and check for cycles
        temp_deps = self._dependencies.copy()
        temp_dep = AgentDependency(
            agent_id=agent_id, 
            depends_on=depends_on, 
            dependency_type="startup",
            required=True
        )
        temp_deps[agent_id] = temp_deps.get(agent_id, []) + [temp_dep]
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(current: str) -> bool:
            if current in rec_stack:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            rec_stack.add(current)
            
            for dep in temp_deps.get(current, []):
                if dep.required and has_cycle(dep.depends_on):
                    return True
            
            rec_stack.remove(current)
            return False
        
        return has_cycle(agent_id)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired registrations."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_registrations()
                await self._cleanup_stale_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired_registrations(self) -> None:
        """Remove expired agent registrations."""
        expired_agents = [
            agent_id for agent_id, registration in self._agents.items()
            if registration.is_expired()
        ]
        
        for agent_id in expired_agents:
            await self.deregister_agent(agent_id)
            await self._emit_event("registration_expired", agent_id, {})
            logger.info(f"Removed expired registration for agent {agent_id}")
    
    async def _cleanup_stale_heartbeats(self) -> None:
        """Identify agents with stale heartbeats."""
        stale_threshold = datetime.utcnow() - timedelta(seconds=self.heartbeat_timeout)
        
        for agent_id, registration in self._agents.items():
            if (registration.last_heartbeat and 
                registration.last_heartbeat < stale_threshold and
                registration.state == AgentState.RUNNING):
                
                # Mark as potentially crashed
                await self.update_agent_state(agent_id, AgentState.ERROR)
                logger.warning(f"Agent {agent_id} has stale heartbeat, marked as ERROR")
    
    async def _setup_message_subscriptions(self) -> None:
        """Setup message bus subscriptions for agent communication."""
        if not self.message_bus:
            return
        
        # Subscribe to agent heartbeat messages
        from ..models.agent_messages import MessageFilter, MessageType as MsgType
        
        heartbeat_filter = MessageFilter(
            message_types=[MsgType.AGENT_HEALTH_UPDATE]
        )
        
        async def handle_heartbeat(message):
            if hasattr(message.payload, 'agent_id') and hasattr(message.payload, 'health'):
                await self.update_agent_health(message.payload.agent_id, message.payload.health)
        
        await self.message_bus.subscribe("registry", heartbeat_filter, handle_heartbeat)
    
    async def _emit_event(self, event_type: str, agent_id: str, data: Dict[str, Any]) -> None:
        """Emit a registry event."""
        event = RegistryEvent(
            event_type=event_type,
            agent_id=agent_id,
            data=data
        )
        
        self._events.append(event)
        
        # Call event callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    async def _save_state(self) -> None:
        """Save registry state to disk."""
        if not self.persistence_path:
            return
        
        try:
            state = {
                "agents": {
                    agent_id: registration.model_dump()
                    for agent_id, registration in self._agents.items()
                },
                "dependencies": {
                    agent_id: [dep.model_dump() for dep in deps]
                    for agent_id, deps in self._dependencies.items()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.debug(f"Registry state saved to {self.persistence_path}")
            
        except Exception as e:
            logger.error(f"Failed to save registry state: {e}")
    
    async def _load_state(self) -> None:
        """Load registry state from disk."""
        try:
            with open(self.persistence_path, 'r') as f:
                state = json.load(f)
            
            # Restore agents
            for agent_id, agent_data in state.get("agents", {}).items():
                try:
                    registration = AgentRegistration(**agent_data)
                    self._agents[agent_id] = registration
                    await self._add_to_indices(registration)
                except Exception as e:
                    logger.error(f"Failed to restore agent {agent_id}: {e}")
            
            # Restore dependencies
            for agent_id, deps_data in state.get("dependencies", {}).items():
                deps = []
                for dep_data in deps_data:
                    try:
                        deps.append(AgentDependency(**dep_data))
                    except Exception as e:
                        logger.error(f"Failed to restore dependency: {e}")
                self._dependencies[agent_id] = deps
            
            logger.info(f"Registry state loaded from {self.persistence_path}")
            
        except Exception as e:
            logger.error(f"Failed to load registry state: {e}") 