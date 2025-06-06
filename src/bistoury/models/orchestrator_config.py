"""
Orchestrator Configuration Models

This module contains Pydantic models for configuring the agent orchestrator,
including startup policies, resource allocation, and failure handling strategies.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator

from ..agents.base import AgentType, AgentState


class StartupPolicy(str, Enum):
    """Agent startup policies."""
    SEQUENTIAL = "sequential"  # Start agents one by one in dependency order
    PARALLEL = "parallel"     # Start independent agents in parallel
    BATCH = "batch"          # Start agents in batches based on dependency levels
    MANUAL = "manual"        # Manual startup control only


class ShutdownPolicy(str, Enum):
    """Agent shutdown policies."""
    GRACEFUL = "graceful"    # Wait for agents to finish current tasks
    IMMEDIATE = "immediate"  # Stop agents immediately
    TIMEOUT = "timeout"      # Graceful with timeout, then force stop


class FailurePolicy(str, Enum):
    """Agent failure handling policies."""
    RESTART = "restart"      # Restart failed agents automatically
    ISOLATE = "isolate"      # Isolate failed agents from system
    SHUTDOWN = "shutdown"    # Shutdown system on critical agent failure
    CONTINUE = "continue"    # Continue operation without failed agent


class ResourceType(str, Enum):
    """Types of resources that can be allocated."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    DATABASE_CONNECTIONS = "database_connections"
    API_RATE_LIMITS = "api_rate_limits"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies for multi-instance agents."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LOAD = "least_load"
    RANDOM = "random"
    WEIGHTED = "weighted"


class ResourceLimit(BaseModel):
    """Resource allocation limit configuration."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    resource_type: ResourceType = Field(..., description="Type of resource")
    max_allocation: float = Field(..., ge=0, description="Maximum allocation amount")
    unit: str = Field(..., description="Unit of measurement (%, MB, connections, etc.)")
    per_agent_limit: Optional[float] = Field(None, ge=0, description="Per-agent limit")
    reserved: float = Field(0, ge=0, description="Reserved amount for system")
    
    def get_available(self, current_usage: float) -> float:
        """Get available resource amount."""
        return max(0, self.max_allocation - self.reserved - current_usage)


class AgentResourceRequirements(BaseModel):
    """Resource requirements for an agent."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    agent_type: AgentType = Field(..., description="Type of agent")
    cpu_cores: float = Field(1.0, ge=0.1, description="CPU cores required")
    memory_mb: float = Field(512.0, ge=64, description="Memory in MB")
    network_bandwidth_mbps: float = Field(10.0, ge=1, description="Network bandwidth in Mbps")
    storage_mb: float = Field(100.0, ge=10, description="Storage in MB")
    database_connections: int = Field(2, ge=1, description="Database connections needed")
    api_rate_limit: int = Field(100, ge=1, description="API calls per minute")
    priority: int = Field(5, ge=1, le=10, description="Resource allocation priority (1-10)")


class StartupSequenceConfig(BaseModel):
    """Configuration for agent startup sequencing."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    policy: StartupPolicy = Field(default=StartupPolicy.BATCH, description="Startup policy")
    batch_size: int = Field(3, ge=1, description="Number of agents to start per batch")
    startup_timeout: timedelta = Field(default=timedelta(minutes=5), description="Timeout for agent startup")
    retry_attempts: int = Field(3, ge=0, description="Number of retry attempts for failed startups")
    retry_delay: timedelta = Field(default=timedelta(seconds=30), description="Delay between retry attempts")
    dependency_wait_timeout: timedelta = Field(default=timedelta(minutes=10), description="Max wait for dependencies")
    parallel_startup_limit: int = Field(5, ge=1, description="Max agents starting in parallel")


class ShutdownSequenceConfig(BaseModel):
    """Configuration for agent shutdown sequencing."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    policy: ShutdownPolicy = Field(default=ShutdownPolicy.GRACEFUL, description="Shutdown policy")
    graceful_timeout: timedelta = Field(default=timedelta(minutes=2), description="Graceful shutdown timeout")
    force_timeout: timedelta = Field(default=timedelta(seconds=30), description="Force shutdown timeout")
    reverse_dependency_order: bool = Field(True, description="Shutdown in reverse dependency order")
    allow_parallel_shutdown: bool = Field(True, description="Allow parallel shutdown of independent agents")


class FailureHandlingConfig(BaseModel):
    """Configuration for handling agent failures."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    default_policy: FailurePolicy = Field(default=FailurePolicy.RESTART, description="Default failure policy")
    agent_policies: Dict[AgentType, FailurePolicy] = Field(default_factory=dict, description="Per-agent-type policies")
    max_restart_attempts: int = Field(3, ge=0, description="Maximum restart attempts")
    restart_backoff_base: float = Field(2.0, ge=1.0, description="Exponential backoff base for restarts")
    restart_backoff_max: timedelta = Field(default=timedelta(minutes=10), description="Maximum restart delay")
    health_check_interval: timedelta = Field(default=timedelta(seconds=30), description="Health check interval")
    failure_threshold: int = Field(3, ge=1, description="Consecutive failures before policy action")
    critical_agents: List[AgentType] = Field(default_factory=list, description="Agents critical for system operation")


class LoadBalancingConfig(BaseModel):
    """Configuration for load balancing multi-instance agents."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    strategy: LoadBalancingStrategy = Field(default=LoadBalancingStrategy.LEAST_LOAD, description="Load balancing strategy")
    health_check_weight: float = Field(0.3, ge=0, le=1, description="Weight of health in load calculation")
    response_time_weight: float = Field(0.4, ge=0, le=1, description="Weight of response time in load calculation")
    queue_size_weight: float = Field(0.3, ge=0, le=1, description="Weight of queue size in load calculation")
    rebalance_interval: timedelta = Field(default=timedelta(minutes=5), description="Load rebalancing interval")
    min_instances: int = Field(1, ge=1, description="Minimum instances per agent type")
    max_instances: int = Field(5, ge=1, description="Maximum instances per agent type")
    scale_up_threshold: float = Field(0.8, ge=0, le=1, description="Load threshold for scaling up")
    scale_down_threshold: float = Field(0.3, ge=0, le=1, description="Load threshold for scaling down")


class OrchestratorConfig(BaseModel):
    """Complete orchestrator configuration."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Basic configuration
    orchestrator_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique orchestrator ID")
    name: str = Field("bistoury-orchestrator", description="Orchestrator name")
    description: str = Field("Bistoury multi-agent system orchestrator", description="Description")
    
    # Sequencing configuration
    startup: StartupSequenceConfig = Field(default_factory=StartupSequenceConfig, description="Startup configuration")
    shutdown: ShutdownSequenceConfig = Field(default_factory=ShutdownSequenceConfig, description="Shutdown configuration")
    
    # Failure handling
    failure_handling: FailureHandlingConfig = Field(default_factory=FailureHandlingConfig, description="Failure handling config")
    
    # Resource management
    resource_limits: List[ResourceLimit] = Field(default_factory=list, description="System resource limits")
    agent_requirements: Dict[AgentType, AgentResourceRequirements] = Field(default_factory=dict, description="Agent resource requirements")
    
    # Load balancing
    load_balancing: LoadBalancingConfig = Field(default_factory=LoadBalancingConfig, description="Load balancing config")
    
    # Monitoring and logging
    monitoring_interval: timedelta = Field(default=timedelta(seconds=10), description="System monitoring interval")
    log_level: str = Field("INFO", description="Logging level")
    metrics_retention: timedelta = Field(default=timedelta(hours=24), description="Metrics retention period")
    
    # Emergency controls
    emergency_stop_timeout: timedelta = Field(default=timedelta(seconds=10), description="Emergency stop timeout")
    enable_auto_recovery: bool = Field(True, description="Enable automatic recovery from failures")
    enable_auto_scaling: bool = Field(False, description="Enable automatic scaling of agents")
    
    @field_validator('agent_requirements')
    @classmethod
    def validate_agent_requirements(cls, v):
        """Validate agent requirements configuration."""
        # Ensure all agent types have requirements
        for agent_type in AgentType:
            if agent_type not in v:
                # Set default requirements
                v[agent_type] = AgentResourceRequirements(agent_type=agent_type)
        return v
    
    def get_failure_policy(self, agent_type: AgentType) -> FailurePolicy:
        """Get failure policy for a specific agent type."""
        return self.failure_handling.agent_policies.get(agent_type, self.failure_handling.default_policy)
    
    def get_resource_requirements(self, agent_type: AgentType) -> AgentResourceRequirements:
        """Get resource requirements for a specific agent type."""
        return self.agent_requirements.get(agent_type, AgentResourceRequirements(agent_type=agent_type))
    
    def is_critical_agent(self, agent_type: AgentType) -> bool:
        """Check if an agent type is critical for system operation."""
        return agent_type in self.failure_handling.critical_agents


class OrchestratorState(BaseModel):
    """Current state of the orchestrator."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    state: AgentState = Field(default=AgentState.CREATED, description="Orchestrator state")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    agents_managed: int = Field(0, ge=0, description="Number of agents under management")
    agents_running: int = Field(0, ge=0, description="Number of running agents")
    agents_failed: int = Field(0, ge=0, description="Number of failed agents")
    last_health_check: Optional[str] = Field(None, description="Last health check timestamp")
    resource_usage: Dict[ResourceType, float] = Field(default_factory=dict, description="Current resource usage")
    active_operations: List[str] = Field(default_factory=list, description="Currently active operations")
    
    def get_health_score(self) -> float:
        """Calculate orchestrator health score."""
        if self.agents_managed == 0:
            return 1.0
        
        running_ratio = self.agents_running / self.agents_managed
        failure_ratio = self.agents_failed / self.agents_managed
        
        # Health score based on running agents and failure rate
        health_score = running_ratio - (failure_ratio * 0.5)
        return max(0.0, min(1.0, health_score))


class OrchestratorEvent(BaseModel):
    """Orchestrator event for logging and monitoring."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique event ID")
    event_type: str = Field(..., description="Type of event")
    timestamp: str = Field(default_factory=lambda: str(datetime.utcnow()), description="Event timestamp")
    agent_id: Optional[str] = Field(None, description="Related agent ID")
    severity: str = Field("INFO", description="Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    message: str = Field(..., description="Event message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    
    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v):
        """Validate severity level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Severity must be one of {valid_levels}")
        return v.upper() 