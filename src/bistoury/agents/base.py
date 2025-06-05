"""
Base agent class and common infrastructure for the Bistoury multi-agent system.

This module provides the foundational BaseAgent class that all agents inherit from,
along with agent state management, lifecycle control, and common functionality.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Awaitable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator
import yaml

from ..logger import get_logger
from ..config import Config


class AgentState(str, Enum):
    """Agent lifecycle states."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    CRASHED = "crashed"


class AgentType(str, Enum):
    """Types of agents in the system."""
    COLLECTOR = "collector"
    SIGNAL_MANAGER = "signal_manager"
    TRADER = "trader"
    POSITION_MANAGER = "position_manager"
    RISK_MANAGER = "risk_manager"
    MONITOR = "monitor"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentCapability:
    """Represents a capability that an agent provides."""
    name: str
    description: str
    version: str
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


class AgentMetadata(BaseModel):
    """Metadata for agent identification and management."""
    
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_type: AgentType
    version: str = "1.0.0"
    description: str = ""
    
    # Lifecycle information
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    
    # Dependencies and requirements
    dependencies: List[str] = Field(default_factory=list)
    required_config: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    
    # Resource requirements
    cpu_limit: Optional[float] = None
    memory_limit: Optional[int] = None
    
    model_config = {"arbitrary_types_allowed": True}


class AgentHealth(BaseModel):
    """Agent health and performance metrics."""
    
    state: AgentState
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_usage: float = 0.0
    memory_usage: int = 0
    error_count: int = 0
    warning_count: int = 0
    
    # Performance metrics
    messages_processed: int = 0
    tasks_completed: int = 0
    uptime_seconds: float = 0.0
    
    # Health indicators
    is_healthy: bool = True
    health_score: float = 1.0  # 0.0 to 1.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class BaseAgent(ABC):
    """
    Base class for all agents in the Bistoury trading system.
    
    Provides common functionality including:
    - Agent lifecycle management (start, stop, pause, restart)
    - State persistence and recovery
    - Health monitoring and metrics collection
    - Configuration management
    - Logging and error handling
    - Message handling infrastructure
    """
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        config: Optional[Dict[str, Any]] = None,
        persist_state: bool = True,
        state_file: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for this agent instance
            agent_type: Type classification for this agent
            config: Agent-specific configuration
            persist_state: Whether to persist agent state to disk
            state_file: Custom path for state persistence file
        """
        # Core identification
        self.metadata = AgentMetadata(
            name=name,
            agent_type=agent_type,
            description=self.__class__.__doc__ or f"{agent_type.value} agent"
        )
        
        # Agent state management
        self.state = AgentState.CREATED
        self.health = AgentHealth(state=self.state)
        self.persist_state = persist_state
        self.state_file = Path(state_file) if state_file else Path(f"data/agents/{name}_state.yaml")
        
        # Configuration
        self.config = config or {}
        self.global_config: Optional[Config] = None
        
        # Lifecycle management
        self._start_time: Optional[float] = None
        self._stop_event = asyncio.Event()
        self._running_tasks: Set[asyncio.Task] = set()
        self._shutdown_callbacks: List[Callable[[], Awaitable[None]]] = []
        
        # Logging
        self.logger = get_logger(f"agent.{name}")
        
        # Message handling (will be set by messaging system)
        self._message_bus = None
        
        # Load persisted state if available
        if self.persist_state:
            self._load_state()
        
        self.logger.info(f"Agent {name} ({agent_type.value}) initialized")
    
    @property
    def agent_id(self) -> str:
        """Get the unique agent ID."""
        return self.metadata.agent_id
    
    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.metadata.name
    
    @property
    def agent_type(self) -> AgentType:
        """Get the agent type."""
        return self.metadata.agent_type
    
    @property
    def is_running(self) -> bool:
        """Check if agent is currently running."""
        return self.state == AgentState.RUNNING
    
    @property
    def is_stopped(self) -> bool:
        """Check if agent is stopped."""
        return self.state in (AgentState.STOPPED, AgentState.CRASHED, AgentState.ERROR)
    
    @property
    def uptime(self) -> float:
        """Get agent uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def _start(self) -> bool:
        """
        Start the agent's main functionality.
        
        This method should initialize the agent's core functionality and
        start any background tasks needed for operation.
        
        Returns:
            bool: True if startup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def _stop(self) -> None:
        """
        Stop the agent's main functionality.
        
        This method should gracefully shutdown the agent's operations,
        clean up resources, and stop background tasks.
        """
        pass
    
    @abstractmethod
    async def _health_check(self) -> AgentHealth:
        """
        Perform a health check and return current health status.
        
        Returns:
            AgentHealth: Current health metrics and status
        """
        pass
    
    # Lifecycle management methods
    
    async def start(self) -> bool:
        """
        Start the agent.
        
        Returns:
            bool: True if agent started successfully
        """
        if self.state != AgentState.CREATED and self.state != AgentState.STOPPED:
            self.logger.warning(f"Cannot start agent in state {self.state}")
            return False
        
        try:
            self.logger.info(f"Starting agent {self.name}")
            self._set_state(AgentState.STARTING)
            
            # Set start time
            self._start_time = time.time()
            self.metadata.started_at = datetime.now(timezone.utc)
            
            # Call the agent's startup implementation
            if await self._start():
                self._set_state(AgentState.RUNNING)
                self.logger.info(f"Agent {self.name} started successfully")
                
                # Start background tasks
                self._running_tasks.add(
                    asyncio.create_task(self._heartbeat_loop())
                )
                
                return True
            else:
                self._set_state(AgentState.ERROR)
                self.logger.error(f"Agent {self.name} failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during agent startup: {e}", exc_info=True)
            self._set_state(AgentState.CRASHED)
            return False
    
    async def stop(self) -> None:
        """Stop the agent gracefully."""
        if self.state == AgentState.STOPPED:
            self.logger.warning(f"Agent {self.name} is already stopped")
            return
        
        try:
            self.logger.info(f"Stopping agent {self.name}")
            self._set_state(AgentState.STOPPING)
            
            # Signal stop to background tasks
            self._stop_event.set()
            
            # Call shutdown callbacks
            for callback in self._shutdown_callbacks:
                try:
                    await callback()
                except Exception as e:
                    self.logger.error(f"Error in shutdown callback: {e}")
            
            # Stop running tasks
            for task in self._running_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._running_tasks:
                await asyncio.gather(*self._running_tasks, return_exceptions=True)
            
            # Call the agent's stop implementation
            await self._stop()
            
            # Update metadata
            self.metadata.stopped_at = datetime.now(timezone.utc)
            self._set_state(AgentState.STOPPED)
            
            self.logger.info(f"Agent {self.name} stopped")
            
        except Exception as e:
            self.logger.error(f"Exception during agent shutdown: {e}", exc_info=True)
            self._set_state(AgentState.CRASHED)
    
    async def restart(self) -> bool:
        """Restart the agent."""
        self.logger.info(f"Restarting agent {self.name}")
        await self.stop()
        return await self.start()
    
    async def pause(self) -> None:
        """Pause the agent (if supported by implementation)."""
        if self.state != AgentState.RUNNING:
            self.logger.warning(f"Cannot pause agent in state {self.state}")
            return
        
        self._set_state(AgentState.PAUSED)
        self.logger.info(f"Agent {self.name} paused")
    
    async def resume(self) -> None:
        """Resume the agent from paused state."""
        if self.state != AgentState.PAUSED:
            self.logger.warning(f"Cannot resume agent in state {self.state}")
            return
        
        self._set_state(AgentState.RUNNING)
        self.logger.info(f"Agent {self.name} resumed")
    
    # Configuration management
    
    def set_global_config(self, config: Config) -> None:
        """Set the global system configuration."""
        self.global_config = config
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from agent config or global config."""
        # First check agent-specific config
        if key in self.config:
            return self.config[key]
        
        # Then check global config if available
        if self.global_config:
            try:
                # Handle nested keys like 'database.path'
                value = self.global_config
                for key_part in key.split('.'):
                    value = getattr(value, key_part)
                return value
            except AttributeError:
                pass
        
        return default
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update agent configuration."""
        self.config.update(updates)
        self.logger.info(f"Configuration updated for agent {self.name}")
    
    # Health and monitoring
    
    async def get_health(self) -> AgentHealth:
        """Get current health status."""
        try:
            # Update basic metrics
            self.health.state = self.state
            self.health.last_heartbeat = datetime.now(timezone.utc)
            self.health.uptime_seconds = self.uptime
            
            # Get agent-specific health metrics
            agent_health = await self._health_check()
            
            # Merge with current health
            self.health.cpu_usage = agent_health.cpu_usage
            self.health.memory_usage = agent_health.memory_usage
            self.health.messages_processed = agent_health.messages_processed
            self.health.tasks_completed = agent_health.tasks_completed
            
            # Calculate health score
            self.health.health_score = self._calculate_health_score()
            self.health.is_healthy = self.health.health_score > 0.5
            
            return self.health
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            self.health.last_error = str(e)
            self.health.last_error_time = datetime.now(timezone.utc)
            self.health.error_count += 1
            self.health.is_healthy = False
            return self.health
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        score = 1.0
        
        # Penalize for errors
        if self.health.error_count > 0:
            score -= min(0.3, self.health.error_count * 0.1)
        
        # Penalize for high CPU usage
        if self.health.cpu_usage > 80:
            score -= 0.2
        
        # Penalize for not running state
        if self.state != AgentState.RUNNING:
            score -= 0.4
        
        return max(0.0, score)
    
    # State persistence
    
    def _set_state(self, new_state: AgentState) -> None:
        """Set the agent state and persist if configured."""
        old_state = self.state
        self.state = new_state
        self.health.state = new_state
        
        self.logger.debug(f"Agent {self.name} state changed: {old_state} -> {new_state}")
        
        if self.persist_state:
            self._save_state()
    
    def _save_state(self) -> None:
        """Save agent state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize metadata and health with mode='json' to avoid enum serialization issues
            metadata_dict = self.metadata.model_dump(mode='json')
            health_dict = self.health.model_dump(mode='json')
            
            state_data = {
                'metadata': metadata_dict,
                'state': self.state.value,
                'config': self.config,
                'health': health_dict
            }
            
            with open(self.state_file, 'w') as f:
                yaml.dump(state_data, f, default_flow_style=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> None:
        """Load agent state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = yaml.safe_load(f)
                
                # Restore metadata
                if 'metadata' in state_data:
                    self.metadata = AgentMetadata(**state_data['metadata'])
                
                # Restore configuration
                if 'config' in state_data:
                    self.config.update(state_data['config'])
                
                # Restore health (but reset state to CREATED)
                if 'health' in state_data:
                    health_data = state_data['health']
                    health_data['state'] = AgentState.CREATED
                    self.health = AgentHealth(**health_data)
                
                self.logger.info(f"State loaded for agent {self.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    # Background tasks
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat updates."""
        while not self._stop_event.is_set():
            try:
                await self.get_health()
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    # Utility methods
    
    def add_shutdown_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Add a callback to be called during shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def create_task(self, coro: Awaitable) -> asyncio.Task:
        """Create and track a background task."""
        task = asyncio.create_task(coro)
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)
        return task
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', state={self.state.value})" 