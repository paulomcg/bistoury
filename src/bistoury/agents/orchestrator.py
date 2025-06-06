"""
Agent Orchestrator Implementation

This module provides the central orchestrator for coordinating all agents in the
multi-agent system, including startup sequencing, resource allocation, failure
handling, and load balancing.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union, Tuple
from pathlib import Path
import json
import psutil

from ..models.orchestrator_config import (
    OrchestratorConfig, OrchestratorState, OrchestratorEvent,
    StartupPolicy, ShutdownPolicy, FailurePolicy, ResourceType,
    LoadBalancingStrategy, AgentResourceRequirements
)
from ..models.agent_registry import AgentRegistration, AgentDiscoveryQuery
from ..agents.base import BaseAgent, AgentType, AgentState, AgentHealth
from ..agents.registry import AgentRegistry
from ..agents.messaging import MessageBus, Message, MessageType, MessagePriority

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages system resource allocation and monitoring."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.allocations: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        self.usage_history: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def allocate_resources(self, agent_id: str, requirements: AgentResourceRequirements) -> bool:
        """Allocate resources for an agent."""
        try:
            # Check if resources are available
            if not self._check_resource_availability(requirements):
                return False
            
            # Allocate resources
            self.allocations[agent_id][ResourceType.CPU] = requirements.cpu_cores
            self.allocations[agent_id][ResourceType.MEMORY] = requirements.memory_mb
            self.allocations[agent_id][ResourceType.NETWORK] = requirements.network_bandwidth_mbps
            self.allocations[agent_id][ResourceType.STORAGE] = requirements.storage_mb
            self.allocations[agent_id][ResourceType.DATABASE_CONNECTIONS] = requirements.database_connections
            self.allocations[agent_id][ResourceType.API_RATE_LIMITS] = requirements.api_rate_limit
            
            logger.info(f"Resources allocated for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to allocate resources for agent {agent_id}: {e}")
            return False
    
    def deallocate_resources(self, agent_id: str) -> None:
        """Deallocate resources for an agent."""
        if agent_id in self.allocations:
            del self.allocations[agent_id]
            logger.info(f"Resources deallocated for agent {agent_id}")
    
    def get_current_usage(self) -> Dict[ResourceType, float]:
        """Get current system resource usage."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            usage = {
                ResourceType.CPU: cpu_percent,
                ResourceType.MEMORY: memory.percent,
                ResourceType.STORAGE: disk.percent,
                ResourceType.NETWORK: 0.0,  # Would need network monitoring
                ResourceType.DATABASE_CONNECTIONS: sum(
                    alloc.get(ResourceType.DATABASE_CONNECTIONS, 0) 
                    for alloc in self.allocations.values()
                ),
                ResourceType.API_RATE_LIMITS: sum(
                    alloc.get(ResourceType.API_RATE_LIMITS, 0) 
                    for alloc in self.allocations.values()
                )
            }
            
            # Update history
            for resource_type, value in usage.items():
                self.usage_history[resource_type].append((time.time(), value))
            
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    def _check_resource_availability(self, requirements: AgentResourceRequirements) -> bool:
        """Check if required resources are available."""
        current_usage = self.get_current_usage()
        
        # Check against configured limits
        for limit in self.config.resource_limits:
            if limit.resource_type == ResourceType.CPU:
                if current_usage.get(ResourceType.CPU, 0) + (requirements.cpu_cores * 10) > limit.max_allocation:
                    return False
            elif limit.resource_type == ResourceType.MEMORY:
                if current_usage.get(ResourceType.MEMORY, 0) + (requirements.memory_mb / 1024) > limit.max_allocation:
                    return False
        
        return True


class LoadBalancer:
    """Manages load balancing for multi-instance agents."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agent_instances: Dict[AgentType, List[str]] = defaultdict(list)
        self.load_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_rebalance = time.time()
        
    def register_instance(self, agent_id: str, agent_type: AgentType) -> None:
        """Register an agent instance for load balancing."""
        if agent_id not in self.agent_instances[agent_type]:
            self.agent_instances[agent_type].append(agent_id)
            logger.info(f"Registered instance {agent_id} for load balancing")
    
    def unregister_instance(self, agent_id: str, agent_type: AgentType) -> None:
        """Unregister an agent instance from load balancing."""
        if agent_id in self.agent_instances[agent_type]:
            self.agent_instances[agent_type].remove(agent_id)
            if agent_id in self.load_metrics:
                del self.load_metrics[agent_id]
            logger.info(f"Unregistered instance {agent_id} from load balancing")
    
    def select_instance(self, agent_type: AgentType) -> Optional[str]:
        """Select the best instance for handling a request."""
        instances = self.agent_instances.get(agent_type, [])
        if not instances:
            return None
        
        strategy = self.config.load_balancing.strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(instances)
        elif strategy == LoadBalancingStrategy.LEAST_LOAD:
            return self._least_load_selection(instances)
        elif strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(instances)
        else:
            return instances[0]  # Default to first instance
    
    def update_load_metrics(self, agent_id: str, metrics: Dict[str, float]) -> None:
        """Update load metrics for an agent instance."""
        self.load_metrics[agent_id].update(metrics)
    
    def _round_robin_selection(self, instances: List[str]) -> str:
        """Round-robin instance selection."""
        # Simple round-robin based on time
        index = int(time.time()) % len(instances)
        return instances[index]
    
    def _least_load_selection(self, instances: List[str]) -> str:
        """Select instance with least load."""
        best_instance = instances[0]
        best_load = float('inf')
        
        for instance in instances:
            metrics = self.load_metrics.get(instance, {})
            load_score = self._calculate_load_score(metrics)
            
            if load_score < best_load:
                best_load = load_score
                best_instance = instance
        
        return best_instance
    
    def _calculate_load_score(self, metrics: Dict[str, float]) -> float:
        """Calculate load score for an instance."""
        config = self.config.load_balancing
        
        health_score = metrics.get('health_score', 1.0)
        response_time = metrics.get('avg_response_time', 0.0)
        queue_size = metrics.get('queue_size', 0.0)
        
        # Lower is better for load score
        load_score = (
            (1.0 - health_score) * config.health_check_weight +
            response_time * config.response_time_weight +
            queue_size * config.queue_size_weight
        )
        
        return load_score


class AgentOrchestrator:
    """
    Central orchestrator for coordinating all agents in the multi-agent system.
    
    Provides:
    - Agent startup sequencing based on dependencies
    - Graceful shutdown and emergency stop functionality
    - Resource allocation and conflict resolution
    - Load balancing for multi-instance agents
    - Failure handling and recovery
    - System-wide monitoring and coordination
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        registry: Optional[AgentRegistry] = None,
        message_bus: Optional[MessageBus] = None
    ):
        """
        Initialize the agent orchestrator.
        
        Args:
            config: Orchestrator configuration
            registry: Agent registry for discovery and management
            message_bus: Message bus for inter-agent communication
        """
        self.config = config or OrchestratorConfig()
        self.registry = registry
        self.message_bus = message_bus
        
        # Core components
        self.resource_manager = ResourceManager(self.config)
        self.load_balancer = LoadBalancer(self.config)
        
        # State management
        self.state = OrchestratorState()
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.agent_startup_order: List[str] = []
        self.startup_attempts: Dict[str, int] = defaultdict(int)
        self.failure_counts: Dict[str, int] = defaultdict(int)
        
        # Events and monitoring
        self.events: deque = deque(maxlen=10000)
        self.event_callbacks: List[Callable[[OrchestratorEvent], None]] = []
        
        # Runtime state
        self._running = False
        self._start_time: Optional[datetime] = None
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        logger.info("Agent orchestrator initialized")
    
    async def start(self) -> bool:
        """Start the orchestrator."""
        if self._running:
            logger.warning("Orchestrator is already running")
            return False
        
        try:
            logger.info("Starting agent orchestrator")
            self._running = True
            self._start_time = datetime.utcnow()
            self.state.state = AgentState.STARTING
            self.state.started_at = self._start_time.isoformat()
            
            # Start background tasks
            self._background_tasks.add(
                asyncio.create_task(self._monitoring_loop())
            )
            self._background_tasks.add(
                asyncio.create_task(self._health_check_loop())
            )
            self._background_tasks.add(
                asyncio.create_task(self._failure_recovery_loop())
            )
            
            # Subscribe to message bus events if available
            if self.message_bus:
                await self._setup_message_subscriptions()
            
            self.state.state = AgentState.RUNNING
            await self._emit_event("orchestrator_started", None, "INFO", "Orchestrator started successfully")
            
            logger.info("Agent orchestrator started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}", exc_info=True)
            self.state.state = AgentState.ERROR
            await self._emit_event("orchestrator_start_failed", None, "ERROR", f"Failed to start: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        if not self._running:
            return
        
        try:
            logger.info("Stopping agent orchestrator")
            self.state.state = AgentState.STOPPING
            
            # Signal shutdown to background tasks
            self._shutdown_event.set()
            
            # Stop all managed agents
            await self.stop_all_agents()
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            self._running = False
            self.state.state = AgentState.STOPPED
            await self._emit_event("orchestrator_stopped", None, "INFO", "Orchestrator stopped")
            
            logger.info("Agent orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}", exc_info=True)
            self.state.state = AgentState.ERROR
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent with the orchestrator."""
        try:
            agent_id = agent.agent_id
            
            if agent_id in self.managed_agents:
                logger.warning(f"Agent {agent_id} is already registered")
                return False
            
            # Allocate resources
            requirements = self.config.get_resource_requirements(agent.agent_type)
            if not self.resource_manager.allocate_resources(agent_id, requirements):
                logger.error(f"Failed to allocate resources for agent {agent_id}")
                return False
            
            # Register with load balancer
            self.load_balancer.register_instance(agent_id, agent.agent_type)
            
            # Add to managed agents
            self.managed_agents[agent_id] = agent
            self.state.agents_managed += 1
            
            await self._emit_event("agent_registered", agent_id, "INFO", f"Agent {agent_id} registered")
            logger.info(f"Agent {agent_id} registered with orchestrator")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the orchestrator."""
        try:
            if agent_id not in self.managed_agents:
                logger.warning(f"Agent {agent_id} is not registered")
                return False
            
            agent = self.managed_agents[agent_id]
            
            # Stop agent if running
            if agent.is_running:
                await agent.stop()
            
            # Deallocate resources
            self.resource_manager.deallocate_resources(agent_id)
            
            # Unregister from load balancer
            self.load_balancer.unregister_instance(agent_id, agent.agent_type)
            
            # Remove from managed agents
            del self.managed_agents[agent_id]
            self.state.agents_managed -= 1
            
            await self._emit_event("agent_unregistered", agent_id, "INFO", f"Agent {agent_id} unregistered")
            logger.info(f"Agent {agent_id} unregistered from orchestrator")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def start_agent(self, agent_id: str) -> bool:
        """Start a specific agent."""
        try:
            if agent_id not in self.managed_agents:
                logger.error(f"Agent {agent_id} is not registered")
                return False
            
            agent = self.managed_agents[agent_id]
            
            if agent.is_running:
                logger.warning(f"Agent {agent_id} is already running")
                return True
            
            # Check dependencies if registry is available
            if self.registry:
                if not await self._check_agent_dependencies(agent_id):
                    logger.error(f"Dependencies not satisfied for agent {agent_id}")
                    return False
            
            # Start the agent
            success = await agent.start()
            
            if success:
                self.state.agents_running += 1
                self.startup_attempts[agent_id] = 0  # Reset attempts on success
                await self._emit_event("agent_started", agent_id, "INFO", f"Agent {agent_id} started")
                logger.info(f"Agent {agent_id} started successfully")
            else:
                self.startup_attempts[agent_id] += 1
                await self._emit_event("agent_start_failed", agent_id, "ERROR", f"Agent {agent_id} failed to start")
                logger.error(f"Agent {agent_id} failed to start")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}: {e}")
            self.startup_attempts[agent_id] += 1
            return False
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop a specific agent."""
        try:
            if agent_id not in self.managed_agents:
                logger.error(f"Agent {agent_id} is not registered")
                return False
            
            agent = self.managed_agents[agent_id]
            
            if not agent.is_running:
                logger.warning(f"Agent {agent_id} is not running")
                return True
            
            # Stop the agent
            await agent.stop()
            
            if agent.is_stopped:
                self.state.agents_running -= 1
                await self._emit_event("agent_stopped", agent_id, "INFO", f"Agent {agent_id} stopped")
                logger.info(f"Agent {agent_id} stopped successfully")
                return True
            else:
                await self._emit_event("agent_stop_failed", agent_id, "ERROR", f"Agent {agent_id} failed to stop")
                logger.error(f"Agent {agent_id} failed to stop properly")
                return False
            
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {e}")
            return False
    
    async def start_all_agents(self) -> bool:
        """Start all registered agents according to startup policy."""
        try:
            logger.info("Starting all agents")
            
            # Get startup order
            startup_order = await self._calculate_startup_order()
            
            policy = self.config.startup.policy
            
            if policy == StartupPolicy.SEQUENTIAL:
                return await self._start_agents_sequential(startup_order)
            elif policy == StartupPolicy.PARALLEL:
                return await self._start_agents_parallel(startup_order)
            elif policy == StartupPolicy.BATCH:
                return await self._start_agents_batch(startup_order)
            else:
                logger.warning(f"Manual startup policy - agents must be started individually")
                return True
            
        except Exception as e:
            logger.error(f"Failed to start all agents: {e}")
            return False
    
    async def stop_all_agents(self) -> bool:
        """Stop all running agents according to shutdown policy."""
        try:
            logger.info("Stopping all agents")
            
            # Get shutdown order (reverse of startup order)
            startup_order = await self._calculate_startup_order()
            shutdown_order = list(reversed(startup_order)) if self.config.shutdown.reverse_dependency_order else startup_order
            
            policy = self.config.shutdown.policy
            
            if policy == ShutdownPolicy.GRACEFUL:
                return await self._stop_agents_graceful(shutdown_order)
            elif policy == ShutdownPolicy.IMMEDIATE:
                return await self._stop_agents_immediate(shutdown_order)
            elif policy == ShutdownPolicy.TIMEOUT:
                return await self._stop_agents_timeout(shutdown_order)
            else:
                logger.warning(f"Unknown shutdown policy: {policy}")
                return await self._stop_agents_graceful(shutdown_order)
            
        except Exception as e:
            logger.error(f"Failed to stop all agents: {e}")
            return False
    
    async def emergency_stop(self) -> None:
        """Emergency stop of all agents and orchestrator."""
        try:
            logger.critical("Emergency stop initiated")
            await self._emit_event("emergency_stop", None, "CRITICAL", "Emergency stop initiated")
            
            # Stop all agents immediately
            stop_tasks = []
            for agent_id, agent in self.managed_agents.items():
                if agent.is_running:
                    stop_tasks.append(asyncio.create_task(agent.stop()))
            
            # Wait for emergency stop timeout
            if stop_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*stop_tasks, return_exceptions=True),
                        timeout=self.config.emergency_stop_timeout.total_seconds()
                    )
                except asyncio.TimeoutError:
                    logger.critical("Emergency stop timeout - some agents may not have stopped cleanly")
            
            # Force stop orchestrator
            self._running = False
            self.state.state = AgentState.STOPPED
            
            logger.critical("Emergency stop completed")
            
        except Exception as e:
            logger.critical(f"Error during emergency stop: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Update resource usage
            resource_usage = self.resource_manager.get_current_usage()
            self.state.resource_usage = resource_usage
            
            # Count agent states
            agent_states = defaultdict(int)
            for agent in self.managed_agents.values():
                agent_states[agent.state.value] += 1
            
            # Calculate health score
            health_score = self.state.get_health_score()
            
            return {
                "orchestrator": {
                    "state": self.state.state.value,
                    "started_at": self.state.started_at,
                    "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
                    "health_score": health_score
                },
                "agents": {
                    "managed": self.state.agents_managed,
                    "running": self.state.agents_running,
                    "failed": self.state.agents_failed,
                    "states": dict(agent_states)
                },
                "resources": resource_usage,
                "load_balancing": {
                    "instances": {
                        agent_type.value: len(instances) 
                        for agent_type, instances in self.load_balancer.agent_instances.items()
                    }
                },
                "recent_events": [
                    {
                        "type": event.event_type,
                        "timestamp": event.timestamp,
                        "agent_id": event.agent_id,
                        "severity": event.severity,
                        "message": event.message
                    }
                    for event in list(self.events)[-10:]  # Last 10 events
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    # Private methods for internal operations
    
    async def _calculate_startup_order(self) -> List[str]:
        """Calculate agent startup order based on dependencies."""
        if self.registry:
            try:
                dependency_graph = await self.registry.get_dependency_graph()
                return dependency_graph.startup_order
            except Exception as e:
                logger.error(f"Failed to get dependency graph: {e}")
        
        # Fallback to simple ordering
        return list(self.managed_agents.keys())
    
    async def _check_agent_dependencies(self, agent_id: str) -> bool:
        """Check if agent dependencies are satisfied."""
        if not self.registry:
            return True
        
        try:
            # Get agent registration
            registration = await self.registry.get_agent(agent_id)
            if not registration:
                return True
            
            # Check each dependency
            for dependency in registration.dependencies:
                if dependency.required:
                    dep_registration = await self.registry.get_agent(dependency.depends_on)
                    if not dep_registration or dep_registration.state != AgentState.RUNNING:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check dependencies for agent {agent_id}: {e}")
            return False
    
    async def _start_agents_sequential(self, startup_order: List[str]) -> bool:
        """Start agents sequentially."""
        success_count = 0
        
        for agent_id in startup_order:
            if agent_id in self.managed_agents:
                if await self.start_agent(agent_id):
                    success_count += 1
                    # Wait a bit between starts
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Failed to start agent {agent_id} in sequential startup")
        
        return success_count == len([aid for aid in startup_order if aid in self.managed_agents])
    
    async def _start_agents_parallel(self, startup_order: List[str]) -> bool:
        """Start agents in parallel (ignoring dependencies)."""
        start_tasks = []
        
        for agent_id in startup_order:
            if agent_id in self.managed_agents:
                start_tasks.append(asyncio.create_task(self.start_agent(agent_id)))
        
        if start_tasks:
            results = await asyncio.gather(*start_tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
            return success_count == len(start_tasks)
        
        return True
    
    async def _start_agents_batch(self, startup_order: List[str]) -> bool:
        """Start agents in batches based on dependency levels."""
        batch_size = self.config.startup.batch_size
        success_count = 0
        
        for i in range(0, len(startup_order), batch_size):
            batch = startup_order[i:i + batch_size]
            batch_tasks = []
            
            for agent_id in batch:
                if agent_id in self.managed_agents:
                    batch_tasks.append(asyncio.create_task(self.start_agent(agent_id)))
            
            if batch_tasks:
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                success_count += sum(1 for result in results if result is True)
                
                # Wait between batches
                if i + batch_size < len(startup_order):
                    await asyncio.sleep(2)
        
        return success_count == len([aid for aid in startup_order if aid in self.managed_agents])
    
    async def _stop_agents_graceful(self, shutdown_order: List[str]) -> bool:
        """Stop agents gracefully."""
        success_count = 0
        
        for agent_id in shutdown_order:
            if agent_id in self.managed_agents:
                if await self.stop_agent(agent_id):
                    success_count += 1
        
        return success_count == len([aid for aid in shutdown_order if aid in self.managed_agents])
    
    async def _stop_agents_immediate(self, shutdown_order: List[str]) -> bool:
        """Stop agents immediately."""
        stop_tasks = []
        
        for agent_id in shutdown_order:
            if agent_id in self.managed_agents:
                stop_tasks.append(asyncio.create_task(self.stop_agent(agent_id)))
        
        if stop_tasks:
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
            return success_count == len(stop_tasks)
        
        return True
    
    async def _stop_agents_timeout(self, shutdown_order: List[str]) -> bool:
        """Stop agents with timeout."""
        try:
            # Try graceful stop first
            graceful_success = await asyncio.wait_for(
                self._stop_agents_graceful(shutdown_order),
                timeout=self.config.shutdown.graceful_timeout.total_seconds()
            )
            
            if graceful_success:
                return True
            
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timeout, forcing stop")
        
        # Force stop remaining agents
        return await self._stop_agents_immediate(shutdown_order)
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Update system metrics
                self.state.last_health_check = datetime.utcnow().isoformat()
                
                # Update resource usage
                resource_usage = self.resource_manager.get_current_usage()
                self.state.resource_usage = resource_usage
                
                # Update agent counts
                running_count = sum(1 for agent in self.managed_agents.values() if agent.is_running)
                failed_count = sum(1 for agent in self.managed_agents.values() if agent.state == AgentState.ERROR)
                
                self.state.agents_running = running_count
                self.state.agents_failed = failed_count
                
                await asyncio.sleep(self.config.monitoring_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                for agent_id, agent in self.managed_agents.items():
                    try:
                        health = await agent.get_health()
                        
                        # Update load balancer metrics
                        metrics = {
                            'health_score': health.health_score,
                            'avg_response_time': 0.0,  # Would be tracked separately
                            'queue_size': 0.0  # Would be tracked separately
                        }
                        self.load_balancer.update_load_metrics(agent_id, metrics)
                        
                    except Exception as e:
                        logger.error(f"Health check failed for agent {agent_id}: {e}")
                
                await asyncio.sleep(self.config.failure_handling.health_check_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)
    
    async def _failure_recovery_loop(self) -> None:
        """Background failure recovery loop."""
        while not self._shutdown_event.is_set():
            try:
                for agent_id, agent in self.managed_agents.items():
                    if agent.state == AgentState.ERROR or agent.state == AgentState.CRASHED:
                        await self._handle_agent_failure(agent_id, agent)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in failure recovery loop: {e}")
                await asyncio.sleep(10)
    
    async def _handle_agent_failure(self, agent_id: str, agent: BaseAgent) -> None:
        """Handle agent failure according to policy."""
        try:
            self.failure_counts[agent_id] += 1
            failure_policy = self.config.get_failure_policy(agent.agent_type)
            
            await self._emit_event("agent_failure", agent_id, "ERROR", 
                                 f"Agent {agent_id} failed (count: {self.failure_counts[agent_id]})")
            
            if failure_policy == FailurePolicy.RESTART:
                if self.failure_counts[agent_id] <= self.config.failure_handling.max_restart_attempts:
                    # Calculate backoff delay
                    delay = min(
                        self.config.failure_handling.restart_backoff_base ** (self.failure_counts[agent_id] - 1),
                        self.config.failure_handling.restart_backoff_max.total_seconds()
                    )
                    
                    logger.info(f"Restarting agent {agent_id} after {delay} seconds")
                    await asyncio.sleep(delay)
                    
                    if await self.start_agent(agent_id):
                        self.failure_counts[agent_id] = 0  # Reset on successful restart
                        await self._emit_event("agent_restarted", agent_id, "INFO", f"Agent {agent_id} restarted successfully")
                    else:
                        await self._emit_event("agent_restart_failed", agent_id, "ERROR", f"Failed to restart agent {agent_id}")
                else:
                    await self._emit_event("agent_restart_limit", agent_id, "ERROR", 
                                         f"Agent {agent_id} exceeded restart limit")
            
            elif failure_policy == FailurePolicy.SHUTDOWN:
                if self.config.is_critical_agent(agent.agent_type):
                    logger.critical(f"Critical agent {agent_id} failed - initiating system shutdown")
                    await self.emergency_stop()
            
            # Other policies (ISOLATE, CONTINUE) are handled by not taking action
            
        except Exception as e:
            logger.error(f"Error handling failure for agent {agent_id}: {e}")
    
    async def _setup_message_subscriptions(self) -> None:
        """Setup message bus subscriptions."""
        if not self.message_bus:
            return
        
        try:
            from ..models.agent_messages import MessageFilter, MessageType as MsgType
            
            # Subscribe to agent lifecycle events
            lifecycle_filter = MessageFilter(
                message_types=[
                    MsgType.AGENT_STARTED,
                    MsgType.AGENT_STOPPED,
                    MsgType.AGENT_ERROR,
                    MsgType.AGENT_HEALTH_UPDATE
                ]
            )
            
            await self.message_bus.subscribe("orchestrator", lifecycle_filter, self._handle_agent_message)
            
        except Exception as e:
            logger.error(f"Failed to setup message subscriptions: {e}")
    
    async def _handle_agent_message(self, message: Message) -> None:
        """Handle messages from agents."""
        try:
            if hasattr(message.payload, 'agent_id'):
                agent_id = message.payload.agent_id
                
                if message.message_type.value.startswith('AGENT_'):
                    await self._emit_event(
                        f"agent_message_{message.message_type.value.lower()}",
                        agent_id,
                        "INFO",
                        f"Received {message.message_type.value} from agent {agent_id}"
                    )
        
        except Exception as e:
            logger.error(f"Error handling agent message: {e}")
    
    async def _emit_event(self, event_type: str, agent_id: Optional[str], severity: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an orchestrator event."""
        try:
            event = OrchestratorEvent(
                event_type=event_type,
                agent_id=agent_id,
                severity=severity,
                message=message,
                data=data or {}
            )
            
            self.events.append(event)
            
            # Call event callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")
    
    def add_event_callback(self, callback: Callable[[OrchestratorEvent], None]) -> None:
        """Add an event callback."""
        self.event_callbacks.append(callback) 