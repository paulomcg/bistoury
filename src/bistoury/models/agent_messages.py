"""
Agent messaging models for inter-agent communication.

This module defines the core message types and data structures used for 
communication between agents in the Bistoury multi-agent trading system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator
from decimal import Decimal


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""
    
    # System Messages
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_CONFIG_UPDATE = "system.config_update"
    SYSTEM_HISTORICAL_REPLAY_COMPLETE = "system.historical_replay_complete"
    
    # Agent Lifecycle Messages
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_PAUSED = "agent.paused"
    AGENT_RESUMED = "agent.resumed"
    AGENT_ERROR = "agent.error"
    AGENT_HEALTH_UPDATE = "agent.health_update"
    
    # Data Messages
    DATA_MARKET_UPDATE = "data.market_update"
    DATA_PRICE_UPDATE = "data.price_update"
    DATA_TRADE_UPDATE = "data.trade_update"
    DATA_ORDERBOOK_UPDATE = "data.orderbook_update"
    DATA_FUNDING_RATE_UPDATE = "data.funding_rate_update"
    
    # Signal Messages
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_AGGREGATED = "signal.aggregated"
    SIGNAL_ANALYSIS_COMPLETE = "signal.analysis_complete"
    
    # Trading Messages
    TRADE_ORDER_REQUEST = "trade.order_request"
    TRADE_ORDER_FILLED = "trade.order_filled"
    TRADE_ORDER_CANCELLED = "trade.order_cancelled"
    TRADE_POSITION_UPDATE = "trade.position_update"
    TRADE_PNL_UPDATE = "trade.pnl_update"
    
    # Risk Messages
    RISK_LIMIT_BREACH = "risk.limit_breach"
    RISK_WARNING = "risk.warning"
    RISK_EMERGENCY_STOP = "risk.emergency_stop"
    
    # Command Messages
    COMMAND_START_COLLECTION = "command.start_collection"
    COMMAND_STOP_COLLECTION = "command.stop_collection"
    COMMAND_PAUSE_TRADING = "command.pause_trading"
    COMMAND_RESUME_TRADING = "command.resume_trading"
    
    # Response Messages
    RESPONSE_SUCCESS = "response.success"
    RESPONSE_ERROR = "response.error"
    RESPONSE_ACKNOWLEDGMENT = "response.ack"


class MessagePriority(str, Enum):
    """Message priority levels for queue processing."""
    
    CRITICAL = "critical"    # Emergency stops, system failures
    HIGH = "high"           # Trading signals, risk warnings
    NORMAL = "normal"       # Regular data updates, status
    LOW = "low"             # Analytics, historical data


class MessageDeliveryMode(str, Enum):
    """Message delivery modes."""
    
    FIRE_AND_FORGET = "fire_and_forget"  # No delivery confirmation needed
    AT_LEAST_ONCE = "at_least_once"      # Retry until acknowledged
    EXACTLY_ONCE = "exactly_once"        # Deliver once with deduplication


class MessageStatus(str, Enum):
    """Message processing status."""
    
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


class MarketDataPayload(BaseModel):
    """Payload for market data messages."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    symbol: str = Field(..., description="Trading symbol")
    price: Optional[Decimal] = Field(None, description="Current price")
    volume: Optional[Decimal] = Field(None, description="Volume")
    timestamp: datetime = Field(..., description="Data timestamp")
    source: str = Field(..., description="Data source identifier")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")


class TradingSignalPayload(BaseModel):
    """Payload for trading signal messages."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    symbol: str = Field(..., description="Trading symbol")
    signal_type: str = Field(..., description="Type of signal")
    direction: str = Field(..., description="Buy/Sell direction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence (0-1)")
    strength: float = Field(..., ge=0.0, le=10.0, description="Signal strength (0-10)")
    timeframe: str = Field(..., description="Signal timeframe")
    strategy: str = Field(..., description="Strategy that generated signal")
    reasoning: str = Field(..., description="Human-readable reasoning")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: Optional[datetime] = Field(None, description="Signal timestamp")


class AggregatedSignalPayload(BaseModel):
    """Payload for aggregated signal messages."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="Aggregated direction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregated confidence (0-1)")
    strength: float = Field(..., ge=0.0, le=10.0, description="Aggregated strength (0-10)")
    signal_count: int = Field(..., ge=1, description="Number of contributing signals")
    contributing_strategies: List[str] = Field(..., description="List of contributing strategies")
    timestamp: datetime = Field(..., description="Aggregation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Aggregation metadata")


class TradingOrderPayload(BaseModel):
    """Payload for trading order messages."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Buy/Sell side")
    order_type: str = Field(..., description="Order type (market, limit, etc.)")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    price: Optional[Decimal] = Field(None, description="Order price (for limit orders)")
    stop_price: Optional[Decimal] = Field(None, description="Stop price")
    time_in_force: str = Field(default="GTC", description="Time in force")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional order data")


class RiskEventPayload(BaseModel):
    """Payload for risk management messages."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    risk_type: str = Field(..., description="Type of risk event")
    severity: str = Field(..., description="Risk severity level")
    description: str = Field(..., description="Risk event description")
    affected_symbols: List[str] = Field(default_factory=list, description="Affected trading symbols")
    current_exposure: Optional[Decimal] = Field(None, description="Current exposure amount")
    limit_threshold: Optional[Decimal] = Field(None, description="Risk limit threshold")
    recommended_action: str = Field(..., description="Recommended action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional risk data")


class SystemEventPayload(BaseModel):
    """Payload for system event messages."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    event_type: str = Field(..., description="Type of system event")
    component: str = Field(..., description="System component involved")
    status: str = Field(..., description="Event status")
    description: str = Field(..., description="Event description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional system data")


# Union type for all possible message payloads
MessagePayload = Union[
    MarketDataPayload,
    TradingSignalPayload,
    AggregatedSignalPayload,
    TradingOrderPayload,
    RiskEventPayload,
    SystemEventPayload,
    Dict[str, Any]  # Fallback for custom payloads
]


class Message(BaseModel):
    """Core message model for inter-agent communication."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Message Identity
    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    correlation_id: Optional[UUID] = Field(None, description="Correlation ID for request/response")
    
    # Message Metadata
    type: MessageType = Field(..., description="Message type")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
    delivery_mode: MessageDeliveryMode = Field(
        default=MessageDeliveryMode.FIRE_AND_FORGET, 
        description="Delivery mode"
    )
    
    # Routing Information
    sender: str = Field(..., description="Sender agent identifier")
    receiver: Optional[str] = Field(None, description="Specific receiver (None for broadcast)")
    topic: Optional[str] = Field(None, description="Pub/sub topic")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message creation time")
    expires_at: Optional[datetime] = Field(None, description="Message expiration time")
    
    # Content
    payload: MessagePayload = Field(..., description="Message payload")
    
    # Processing Status
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Processing status")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    # Metadata
    headers: Dict[str, str] = Field(default_factory=dict, description="Message headers")
    tags: List[str] = Field(default_factory=list, description="Message tags for filtering")
    
    @field_validator('receiver')
    @classmethod
    def validate_receiver(cls, v: Optional[str]) -> Optional[str]:
        """Validate receiver format."""
        if v is not None and not v.strip():
            raise ValueError("Receiver cannot be empty string")
        return v
    
    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v: Optional[str]) -> Optional[str]:
        """Validate topic format."""
        if v is not None:
            if not v.strip():
                raise ValueError("Topic cannot be empty string")
            # Topic should follow a hierarchical format like "data.market.BTC"
            if not all(part.isalnum() or part in ['_', '-'] for part in v.replace('.', '')):
                raise ValueError("Topic can only contain alphanumeric characters, dots, underscores, and hyphens")
        return v
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired()
    
    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1
    
    def add_header(self, key: str, value: str) -> None:
        """Add a header to the message."""
        self.headers[key] = value
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the message."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if message has a specific tag."""
        return tag in self.tags
    
    def create_response(
        self,
        response_type: MessageType,
        payload: MessagePayload,
        sender: str
    ) -> "Message":
        """Create a response message with correlation ID."""
        return Message(
            type=response_type,
            correlation_id=self.id,
            sender=sender,
            receiver=self.sender,
            payload=payload,
            priority=self.priority,
            headers={"in_response_to": str(self.id)}
        )


class MessageFilter(BaseModel):
    """Filter criteria for message subscription and routing."""
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)
    
    message_types: Optional[List[MessageType]] = Field(None, description="Filter by message types")
    senders: Optional[List[str]] = Field(None, description="Filter by sender agents")
    topics: Optional[List[str]] = Field(None, description="Filter by topics")
    priorities: Optional[List[MessagePriority]] = Field(None, description="Filter by priorities")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (any tag matches)")
    require_all_tags: bool = Field(default=False, description="Require all tags to match")
    
    def matches(self, message: Message) -> bool:
        """Check if a message matches this filter."""
        # Check message types
        if self.message_types and message.type not in self.message_types:
            return False
        
        # Check senders
        if self.senders and message.sender not in self.senders:
            return False
        
        # Check topics
        if self.topics and message.topic:
            topic_match = any(
                message.topic.startswith(topic) or topic.startswith(message.topic)
                for topic in self.topics
            )
            if not topic_match:
                return False
        elif self.topics and not message.topic:
            return False
        
        # Check priorities
        if self.priorities and message.priority not in self.priorities:
            return False
        
        # Check tags
        if self.tags:
            if self.require_all_tags:
                if not all(tag in message.tags for tag in self.tags):
                    return False
            else:
                if not any(tag in message.tags for tag in self.tags):
                    return False
        
        return True


class Subscription(BaseModel):
    """Subscription model for pub/sub messaging."""
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)
    
    id: UUID = Field(default_factory=uuid4, description="Subscription identifier")
    subscriber: str = Field(..., description="Subscriber agent identifier")
    filter: MessageFilter = Field(..., description="Message filter criteria")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Subscription creation time")
    last_message_at: Optional[datetime] = Field(None, description="Last message received time")
    message_count: int = Field(default=0, ge=0, description="Number of messages received")
    is_active: bool = Field(default=True, description="Subscription status")
    
    def __hash__(self) -> int:
        """Make subscription hashable for use in sets."""
        return hash((self.id, self.subscriber))
    
    def update_activity(self) -> "Subscription":
        """Update subscription activity metrics and return new instance."""
        return self.model_copy(update={
            'last_message_at': datetime.utcnow(),
            'message_count': self.message_count + 1
        })


class MessageBatch(BaseModel):
    """Batch of messages for efficient processing."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: UUID = Field(default_factory=uuid4, description="Batch identifier")
    messages: List[Message] = Field(..., description="Messages in batch")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation time")
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        """Validate batch contains messages."""
        if not v:
            raise ValueError("Message batch cannot be empty")
        return v
    
    def size(self) -> int:
        """Get batch size."""
        return len(self.messages)
    
    def priorities(self) -> List[MessagePriority]:
        """Get unique priorities in batch."""
        return list(set(msg.priority for msg in self.messages))
    
    def filter_by_priority(self, priority: MessagePriority) -> List[Message]:
        """Filter messages by priority."""
        return [msg for msg in self.messages if msg.priority == priority]
    
    def filter_by_type(self, message_type: MessageType) -> List[Message]:
        """Filter messages by type."""
        return [msg for msg in self.messages if msg.type == message_type]


class DeliveryReceipt(BaseModel):
    """Receipt confirming message delivery."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    message_id: UUID = Field(..., description="Original message ID")
    receiver: str = Field(..., description="Receiver agent identifier")
    status: MessageStatus = Field(..., description="Delivery status")
    delivered_at: datetime = Field(default_factory=datetime.utcnow, description="Delivery timestamp")
    error_message: Optional[str] = Field(None, description="Error message if delivery failed")
    processing_time_ms: Optional[float] = Field(None, description="Message processing time in milliseconds") 