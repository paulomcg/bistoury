"""
Agent Registry Models

This module contains Pydantic models for the agent registry and discovery system,
including agent registration, capabilities, dependencies, and health tracking.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator

from ..agents.base import AgentType, AgentState, AgentHealth


class AgentCapabilityType(str, Enum):
    """Types of capabilities an agent can provide."""
    
    # Data capabilities
    DATA_COLLECTION = "data.collection"
    DATA_PROCESSING = "data.processing" 
    DATA_ANALYSIS = "data.analysis"
    DATA_STORAGE = "data.storage"
    
    # Trading capabilities
    SIGNAL_GENERATION = "signal.generation"
    SIGNAL_AGGREGATION = "signal.aggregation"
    ORDER_EXECUTION = "order.execution"
    POSITION_MANAGEMENT = "position.management"
    RISK_MANAGEMENT = "risk.management"
    
    # System capabilities
    MONITORING = "system.monitoring"
    CONFIGURATION = "system.configuration"
    ORCHESTRATION = "system.orchestration"
    LOGGING = "system.logging"
    
    # Analysis capabilities
    TECHNICAL_ANALYSIS = "analysis.technical"
    FUNDAMENTAL_ANALYSIS = "analysis.fundamental"
    SENTIMENT_ANALYSIS = "analysis.sentiment"
    PATTERN_RECOGNITION = "analysis.patterns"


class AgentCapability(BaseModel):
    """Represents a capability provided by an agent."""
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)
    
    type: AgentCapabilityType = Field(..., description="Type of capability")
    version: str = Field("1.0.0", description="Capability version")
    description: str = Field(..., description="Human-readable description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")
    dependencies: List[AgentCapabilityType] = Field(default_factory=list, description="Required capabilities")
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split('.')
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("Version must be in format 'major.minor.patch'")
        return v


class AgentDependency(BaseModel):
    """Represents a dependency between agents."""
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)
    
    agent_id: str = Field(..., description="ID of the dependent agent")
    depends_on: str = Field(..., description="ID of the required agent")
    dependency_type: str = Field("startup", description="Type of dependency")
    required: bool = Field(True, description="Whether dependency is required or optional")
    timeout_seconds: int = Field(30, ge=1, description="Maximum wait time for dependency")
    
    @field_validator('dependency_type')
    @classmethod
    def validate_dependency_type(cls, v: str) -> str:
        """Validate dependency type."""
        valid_types = ["startup", "runtime", "data", "configuration"]
        if v not in valid_types:
            raise ValueError(f"Dependency type must be one of: {valid_types}")
        return v


class AgentCompatibility(BaseModel):
    """Agent version and compatibility information."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    agent_version: str = Field(..., description="Agent version")
    framework_version: str = Field(..., description="Required framework version")
    python_version: str = Field(..., description="Required Python version")
    compatible_agents: Dict[str, str] = Field(default_factory=dict, description="Compatible agent versions")
    
    @field_validator('agent_version', 'framework_version', 'python_version')
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate version format."""
        if not v or len(v.split('.')) < 2:
            raise ValueError("Version must have at least major.minor format")
        return v


class AgentRegistration(BaseModel):
    """Complete agent registration information."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Basic Information
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    agent_type: AgentType = Field(..., description="Type of agent")
    description: str = Field(..., description="Agent description")
    
    # Capabilities
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    provided_services: List[str] = Field(default_factory=list, description="Services provided by agent")
    required_services: List[str] = Field(default_factory=list, description="Services required by agent")
    
    # Network and Communication
    host: str = Field("localhost", description="Host where agent is running")
    port: Optional[int] = Field(None, description="Port for agent communication")
    endpoint: Optional[str] = Field(None, description="Agent communication endpoint")
    
    # Registration metadata
    registered_at: datetime = Field(default_factory=datetime.utcnow, description="Registration timestamp")
    expires_at: Optional[datetime] = Field(None, description="Registration expiration")
    ttl_seconds: int = Field(300, ge=60, description="Time-to-live for registration")
    
    # Status and Health
    state: AgentState = Field(default=AgentState.CREATED, description="Current agent state")
    health: Optional[AgentHealth] = Field(None, description="Current health status")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat timestamp")
    
    # Compatibility and Dependencies
    compatibility: AgentCompatibility = Field(..., description="Version compatibility info")
    dependencies: List[AgentDependency] = Field(default_factory=list, description="Agent dependencies")
    
    # Configuration
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def __post_init__(self):
        """Set expiration time if not provided."""
        if self.expires_at is None:
            self.expires_at = self.registered_at + timedelta(seconds=self.ttl_seconds)
    
    def is_expired(self) -> bool:
        """Check if registration has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_healthy(self) -> bool:
        """Check if agent is currently healthy."""
        if self.health is None:
            return False
        return self.health.is_healthy()
    
    def time_since_heartbeat(self) -> Optional[timedelta]:
        """Get time since last heartbeat."""
        if self.last_heartbeat is None:
            return None
        return datetime.utcnow() - self.last_heartbeat
    
    def refresh_registration(self, ttl_seconds: Optional[int] = None) -> None:
        """Refresh registration expiration."""
        if ttl_seconds is not None:
            self.ttl_seconds = ttl_seconds
        self.expires_at = datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
    
    def update_heartbeat(self, health: Optional[AgentHealth] = None) -> None:
        """Update last heartbeat and optionally health status."""
        self.last_heartbeat = datetime.utcnow()
        if health is not None:
            self.health = health
    
    def has_capability(self, capability_type: AgentCapabilityType) -> bool:
        """Check if agent has a specific capability."""
        return any(cap.type == capability_type for cap in self.capabilities)
    
    def get_capability(self, capability_type: AgentCapabilityType) -> Optional[AgentCapability]:
        """Get a specific capability if available."""
        for cap in self.capabilities:
            if cap.type == capability_type:
                return cap
        return None


class AgentDiscoveryQuery(BaseModel):
    """Query parameters for agent discovery."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Filtering criteria
    agent_types: Optional[List[AgentType]] = Field(None, description="Filter by agent types")
    capabilities: Optional[List[AgentCapabilityType]] = Field(None, description="Required capabilities")
    services: Optional[List[str]] = Field(None, description="Required services")
    states: Optional[List[AgentState]] = Field(None, description="Filter by agent states")
    
    # Health and status filters
    healthy_only: bool = Field(True, description="Only return healthy agents")
    exclude_expired: bool = Field(True, description="Exclude expired registrations")
    min_health_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum health score")
    
    # Compatibility filters
    framework_version: Optional[str] = Field(None, description="Required framework version")
    agent_version: Optional[str] = Field(None, description="Required agent version")
    
    # Sorting and limits
    sort_by: str = Field("registered_at", description="Sort field")
    sort_desc: bool = Field(True, description="Sort in descending order")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum results")
    
    @field_validator('sort_by')
    @classmethod
    def validate_sort_field(cls, v: str) -> str:
        """Validate sort field."""
        valid_fields = [
            "registered_at", "agent_id", "name", "agent_type", 
            "health.health_score", "last_heartbeat"
        ]
        if v not in valid_fields:
            raise ValueError(f"Sort field must be one of: {valid_fields}")
        return v


class AgentDiscoveryResult(BaseModel):
    """Result of agent discovery query."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    query: AgentDiscoveryQuery = Field(..., description="Original query")
    agents: List[AgentRegistration] = Field(..., description="Matching agents")
    total_count: int = Field(..., description="Total matching agents")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")


class RegistryStatistics(BaseModel):
    """Agent registry statistics and metrics."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Basic counts
    total_agents: int = Field(..., description="Total registered agents")
    active_agents: int = Field(..., description="Currently active agents")
    healthy_agents: int = Field(..., description="Currently healthy agents")
    expired_agents: int = Field(..., description="Expired registrations")
    
    # By type
    agents_by_type: Dict[str, int] = Field(..., description="Agent count by type")
    agents_by_state: Dict[str, int] = Field(..., description="Agent count by state")
    
    # Health metrics
    average_health_score: float = Field(..., description="Average health score")
    health_score_distribution: Dict[str, int] = Field(..., description="Health score ranges")
    
    # Capability metrics
    capabilities_count: Dict[str, int] = Field(..., description="Capability usage count")
    services_count: Dict[str, int] = Field(..., description="Service usage count")
    
    # Performance metrics
    registration_rate: float = Field(..., description="Registrations per minute")
    discovery_query_rate: float = Field(..., description="Discovery queries per minute")
    average_query_time_ms: float = Field(..., description="Average query time in ms")
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Statistics generation time")
    registry_uptime_seconds: float = Field(..., description="Registry uptime in seconds")


class DependencyGraph(BaseModel):
    """Agent dependency graph representation."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    nodes: Dict[str, AgentRegistration] = Field(..., description="Agent nodes in graph")
    edges: List[AgentDependency] = Field(..., description="Dependency edges")
    startup_order: List[str] = Field(..., description="Recommended startup order")
    circular_dependencies: List[List[str]] = Field(default_factory=list, description="Circular dependency chains")
    orphaned_agents: List[str] = Field(default_factory=list, description="Agents with missing dependencies")
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate dependency graph and return validation errors."""
        errors = {}
        
        # Check for missing dependencies
        for edge in self.edges:
            if edge.depends_on not in self.nodes:
                agent_errors = errors.setdefault(edge.agent_id, [])
                agent_errors.append(f"Missing dependency: {edge.depends_on}")
        
        # Check for circular dependencies
        if self.circular_dependencies:
            for cycle in self.circular_dependencies:
                for agent_id in cycle:
                    agent_errors = errors.setdefault(agent_id, [])
                    agent_errors.append(f"Circular dependency: {' -> '.join(cycle)}")
        
        return errors
    
    def get_dependencies(self, agent_id: str) -> List[str]:
        """Get all dependencies for an agent."""
        return [edge.depends_on for edge in self.edges if edge.agent_id == agent_id]
    
    def get_dependents(self, agent_id: str) -> List[str]:
        """Get all agents that depend on this agent."""
        return [edge.agent_id for edge in self.edges if edge.depends_on == agent_id]


class RegistryEvent(BaseModel):
    """Event that occurred in the agent registry."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: UUID = Field(default_factory=uuid4, description="Event identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    event_type: str = Field(..., description="Type of event")
    agent_id: str = Field(..., description="Agent involved in event")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    
    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type."""
        valid_types = [
            "agent_registered", "agent_deregistered", "agent_heartbeat",
            "agent_state_changed", "agent_health_updated", "dependency_added",
            "dependency_removed", "registration_expired", "registry_started",
            "registry_stopped", "discovery_query", "error"
        ]
        if v not in valid_types:
            raise ValueError(f"Event type must be one of: {valid_types}")
        return v 