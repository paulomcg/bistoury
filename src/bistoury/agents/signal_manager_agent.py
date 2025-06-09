"""
Signal Manager Agent - Task 9.5

Integrates SignalManager into the multi-agent framework with messaging capabilities.
Provides the bridge between strategy agents and the Position Manager for signal flow.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID

from .base import BaseAgent, AgentType, AgentHealth, AgentState
from .messaging import MessageBus, MessageFilter, Subscription
from ..models.agent_messages import (
    Message, MessageType, MessagePriority, TradingSignalPayload, 
    AggregatedSignalPayload, SystemEventPayload
)
from ..models.signals import TradingSignal, SignalDirection
from ..signal_manager.signal_manager import SignalManager, SignalManagerConfiguration
from ..signal_manager.models import AggregatedSignal
from ..signal_manager.narrative_buffer import NarrativeBufferConfig
from ..strategies.narrative_generator import TradingNarrative


class SignalManagerAgent(BaseAgent):
    """
    Agent wrapper for SignalManager providing message bus integration.
    
    Key responsibilities:
    - Subscribe to trading signals from strategy agents
    - Aggregate and filter signals using SignalManager
    - Publish aggregated signals to Position Manager and other subscribers
    - Manage signal quality, conflicts, and strategy weights
    - Provide health monitoring and performance reporting
    """
    
    def __init__(
        self,
        name: str = "signal_manager",
        config: Optional[Dict[str, Any]] = None,
        signal_config: Optional[SignalManagerConfiguration] = None,
        narrative_config: Optional[NarrativeBufferConfig] = None,
        **kwargs
    ):
        """
        Initialize Signal Manager Agent.
        
        Args:
            name: Agent name (default: "signal_manager")
            config: Agent configuration
            signal_config: SignalManager-specific configuration
            narrative_config: Narrative buffer configuration
        """
        super().__init__(
            name=name,
            agent_type=AgentType.SIGNAL_MANAGER,
            config=config,
            **kwargs
        )
        
        # Signal Manager components
        self.signal_config = signal_config or SignalManagerConfiguration()
        self.narrative_config = narrative_config or NarrativeBufferConfig()
        self.signal_manager: Optional[SignalManager] = None
        
        # Message bus integration
        self._message_bus: Optional[MessageBus] = None
        self._subscriptions: Dict[str, Subscription] = {}
        
        # Strategy management
        self.active_strategies: Dict[str, str] = {}  # strategy_id -> agent_id
        self.strategy_weights: Dict[str, float] = {}
        self.signal_counts: Dict[str, int] = {}
        
        # Performance tracking
        self.signals_received = 0
        self.signals_processed = 0
        self.signals_published = 0
        self.aggregation_latency_ms = 0.0
        self.last_signal_time: Optional[datetime] = None
        
        # Configuration
        config = config or {}
        self.subscribe_to_strategies = config.get('subscribe_to_strategies', True)
        self.publish_aggregated_signals = config.get('publish_aggregated_signals', True)
        self.auto_register_strategies = config.get('auto_register_strategies', True)
        
        self.logger.info(f"Signal Manager Agent initialized with config: {self.signal_config}")
    
    async def _start(self) -> bool:
        """Start the Signal Manager Agent."""
        try:
            self.logger.info("Starting Signal Manager Agent...")
            
            # Initialize Signal Manager
            self.signal_manager = SignalManager(
                config=self.signal_config,
                narrative_config=self.narrative_config
            )
            
            # Set up signal callbacks
            self.signal_manager.add_signal_callback(self._on_aggregated_signal)
            self.signal_manager.add_error_callback(self._on_signal_error)
            
            # Start Signal Manager
            await self.signal_manager.start()
            
            # Set up message bus subscriptions if available
            if self._message_bus and self.subscribe_to_strategies:
                await self._setup_subscriptions()
            
            self.logger.info("Signal Manager Agent started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Signal Manager Agent: {e}", exc_info=True)
            self._set_state(AgentState.ERROR)
            return False
    
    async def _stop(self) -> None:
        """Stop the Signal Manager Agent."""
        try:
            self.logger.info("Stopping Signal Manager Agent...")
            
            # Clean up subscriptions
            if self._message_bus:
                await self._cleanup_subscriptions()
            
            # Stop Signal Manager
            if self.signal_manager:
                await self.signal_manager.stop()
                self.signal_manager = None
            
            self.logger.info("Signal Manager Agent stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Signal Manager Agent: {e}", exc_info=True)
    
    async def _health_check(self) -> AgentHealth:
        """Perform health check and return agent health status."""
        health = AgentHealth(state=self.state)
        
        try:
            if self.signal_manager:
                # Get Signal Manager status
                status = self.signal_manager.get_status()
                metrics = self.signal_manager.get_metrics()
                
                # Update health metrics
                health.messages_processed = self.signals_processed
                health.tasks_completed = self.signals_published
                health.uptime_seconds = self.uptime
                
                # Calculate health score based on performance
                health_score = 1.0
                
                # Check if Signal Manager is running
                if not status.is_running:
                    health_score -= 0.5
                
                # Check for recent errors
                if status.error_count > 0:
                    health_score -= min(0.3, status.error_count * 0.1)
                
                # Check processing latency
                if metrics.signals_per_minute > 0:
                    if self.aggregation_latency_ms > 1000:  # > 1 second
                        health_score -= 0.2
                    elif self.aggregation_latency_ms > 500:  # > 500ms
                        health_score -= 0.1
                
                # Check for stale signals
                if self.last_signal_time:
                    time_since_signal = (datetime.now(timezone.utc) - self.last_signal_time).total_seconds()
                    if time_since_signal > 300:  # 5 minutes
                        health_score -= 0.2
                
                health.health_score = max(0.0, health_score)
                
                # Update error information
                if status.last_error:
                    health.last_error = status.last_error
                    health.error_count = status.error_count
            
            else:
                health.health_score = 0.0
                health.last_error = "Signal Manager not initialized"
        
        except Exception as e:
            health.health_score = 0.0
            health.last_error = str(e)
            health.error_count += 1
            self.logger.error(f"Health check failed: {e}", exc_info=True)
        
        return health
    
    def set_message_bus(self, message_bus: MessageBus) -> None:
        """Set the message bus for communication."""
        self._message_bus = message_bus
        self.logger.info("Message bus connected")
    
    async def _setup_subscriptions(self) -> None:
        """Set up message bus subscriptions for trading signals."""
        if not self._message_bus:
            self.logger.warning("No message bus available for subscriptions")
            return
        
        try:
            # Subscribe to trading signals from strategy agents
            signal_filter = MessageFilter(
                message_types=[MessageType.SIGNAL_GENERATED],
                sender_types=["strategy"]
            )
            
            subscription = await self._message_bus.subscribe(
                agent_id=self.agent_id,
                filter=signal_filter,
                handler=self._handle_trading_signal,
                is_async=True
            )
            
            self._subscriptions["trading_signals"] = subscription
            self.logger.info("Subscribed to trading signals")
            
            # Subscribe to system events for strategy registration
            if self.auto_register_strategies:
                system_filter = MessageFilter(
                    message_types=[MessageType.AGENT_STARTED],
                    topics=["agent_registered", "agent_started"]
                )
                
                system_subscription = await self._message_bus.subscribe(
                    agent_id=self.agent_id,
                    filter=system_filter,
                    handler=self._handle_system_event,
                    is_async=True
                )
                
                self._subscriptions["system_events"] = system_subscription
                self.logger.info("Subscribed to system events")
        
        except Exception as e:
            self.logger.error(f"Failed to setup subscriptions: {e}", exc_info=True)
    
    async def _cleanup_subscriptions(self) -> None:
        """Clean up message bus subscriptions."""
        for sub_name, subscription in self._subscriptions.items():
            try:
                await self._message_bus.unsubscribe(self.agent_id, subscription.id)
                self.logger.info(f"Unsubscribed from {sub_name}")
            except Exception as e:
                self.logger.error(f"Failed to unsubscribe from {sub_name}: {e}")
        
        self._subscriptions.clear()
    
    async def _handle_trading_signal(self, message: Message) -> None:
        """Handle incoming trading signal messages."""
        try:
            start_time = datetime.now(timezone.utc)
            self.signals_received += 1
            
            # Extract signal payload
            if not isinstance(message.payload, TradingSignalPayload):
                self.logger.warning(f"Invalid signal payload type: {type(message.payload)}")
                return
            
            signal_payload = message.payload
            
            # Convert to TradingSignal
            trading_signal = TradingSignal(
                symbol=signal_payload.symbol,
                signal_type=signal_payload.signal_type,
                direction=SignalDirection(signal_payload.direction),
                confidence=signal_payload.confidence,
                strength=signal_payload.strength,
                timeframe=signal_payload.timeframe,
                strategy=signal_payload.strategy,
                reasoning=signal_payload.reasoning,
                metadata=signal_payload.metadata or {},
                created_at=signal_payload.timestamp or datetime.now(timezone.utc)
            )
            
            # Register strategy if new
            if signal_payload.strategy not in self.active_strategies:
                await self._register_strategy(signal_payload.strategy, message.sender)
            
            # Process through Signal Manager
            if self.signal_manager:
                narrative = None  # Could extract from metadata if available
                
                aggregated = await self.signal_manager.process_signal(
                    signal=trading_signal,
                    narrative=narrative,
                    strategy_id=signal_payload.strategy
                )
                
                self.signals_processed += 1
                self.last_signal_time = datetime.now(timezone.utc)
                
                # Calculate latency
                latency = (self.last_signal_time - start_time).total_seconds() * 1000
                self.aggregation_latency_ms = (self.aggregation_latency_ms + latency) / 2
                
                self.logger.debug(
                    f"Processed signal from {signal_payload.strategy}: "
                    f"{signal_payload.symbol} {signal_payload.direction} "
                    f"(confidence: {signal_payload.confidence:.2f}, latency: {latency:.1f}ms)"
                )
        
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {e}", exc_info=True)
            await self._handle_error(e)
    
    async def _handle_system_event(self, message: Message) -> None:
        """Handle system event messages for strategy registration."""
        try:
            if not isinstance(message.payload, SystemEventPayload):
                return
            
            event = message.payload
            
            # Auto-register strategy agents
            if (event.event_type in ["agent_registered", "agent_started"] and 
                event.component == "strategy"):
                
                strategy_name = event.description.get('strategy_name') if event.description else None
                if strategy_name:
                    await self._register_strategy(strategy_name, message.sender)
        
        except Exception as e:
            self.logger.error(f"Error handling system event: {e}", exc_info=True)
    
    async def _register_strategy(self, strategy_id: str, agent_id: str) -> None:
        """Register a new strategy for signal processing."""
        if strategy_id not in self.active_strategies:
            self.active_strategies[strategy_id] = agent_id
            self.strategy_weights[strategy_id] = 1.0  # Default weight
            self.signal_counts[strategy_id] = 0
            
            self.logger.info(f"Registered strategy: {strategy_id} (agent: {agent_id})")
    
    async def _on_aggregated_signal(self, aggregated_signal: AggregatedSignal) -> None:
        """Callback for when Signal Manager produces an aggregated signal."""
        try:
            if not self.publish_aggregated_signals or not self._message_bus:
                return
            
            # Create aggregated signal payload
            payload = AggregatedSignalPayload(
                symbol=aggregated_signal.symbol,
                direction=aggregated_signal.direction.value,
                confidence=aggregated_signal.confidence,
                strength=aggregated_signal.strength,
                signal_count=aggregated_signal.signal_count,
                contributing_strategies=list(aggregated_signal.contributing_strategies),
                timestamp=aggregated_signal.created_at,
                metadata={
                    'quality_score': aggregated_signal.quality.value if aggregated_signal.quality else None,
                    'conflicts_resolved': len(aggregated_signal.conflicts) if aggregated_signal.conflicts else 0,
                    'timeframe': aggregated_signal.timeframe,
                    'reasoning': aggregated_signal.reasoning
                }
            )
            
            # Publish to message bus
            success = await self._message_bus.publish(
                topic="trading.signals.aggregated",
                message_type=MessageType.SIGNAL_AGGREGATED,
                payload=payload,
                sender=self.name,
                priority=MessagePriority.HIGH
            )
            
            if success:
                self.signals_published += 1
                self.logger.debug(
                    f"Published aggregated signal: {aggregated_signal.symbol} "
                    f"{aggregated_signal.direction.value} (confidence: {aggregated_signal.confidence:.2f})"
                )
            else:
                self.logger.warning(f"Failed to publish aggregated signal for {aggregated_signal.symbol}")
        
        except Exception as e:
            self.logger.error(f"Error publishing aggregated signal: {e}", exc_info=True)
    
    async def _on_signal_error(self, error: Exception) -> None:
        """Callback for Signal Manager errors."""
        self.logger.error(f"Signal Manager error: {error}")
        await self._handle_error(error)
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors and update agent health."""
        # Could publish error events to message bus for monitoring
        if self._message_bus:
            try:
                error_payload = SystemEventPayload(
                    event_type="agent_error",
                    component="signal_manager",
                    status="error",
                    description={"error": str(error), "agent": self.name}
                )
                
                await self._message_bus.publish(
                    topic="system.errors",
                    message_type=MessageType.SYSTEM_EVENT,
                    payload=error_payload,
                    sender=self.name,
                    priority=MessagePriority.HIGH
                )
            except Exception as e:
                self.logger.error(f"Failed to publish error event: {e}")
    
    # Public API methods
    
    async def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        if self.signal_manager:
            return self.signal_manager.get_strategy_weights()
        return self.strategy_weights.copy()
    
    async def update_strategy_weight(self, strategy_id: str, weight: float) -> bool:
        """Update a strategy's weight."""
        try:
            if self.signal_manager:
                await self.signal_manager.update_strategy_weight(strategy_id, weight)
            
            self.strategy_weights[strategy_id] = weight
            self.logger.info(f"Updated weight for {strategy_id}: {weight}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to update strategy weight: {e}")
            return False
    
    async def get_signal_manager_status(self) -> Dict[str, Any]:
        """Get detailed Signal Manager status."""
        if not self.signal_manager:
            return {"status": "not_initialized"}
        
        status = self.signal_manager.get_status()
        metrics = self.signal_manager.get_metrics()
        
        return {
            "signal_manager": {
                "is_running": status.is_running,
                "start_time": status.start_time.isoformat() if status.start_time else None,
                "signals_processed": status.signals_processed,
                "signals_published": status.signals_published,
                "error_count": status.error_count,
                "last_error": status.last_error,
                "processing_latency_ms": status.processing_latency_ms
            },
            "metrics": {
                "signals_per_minute": metrics.signals_per_minute,
                "average_confidence": metrics.average_confidence,
                "quality_distribution": metrics.quality_distribution,
                "strategy_performance": metrics.strategy_performance,
                "conflict_rate": metrics.conflict_rate
            },
            "agent": {
                "signals_received": self.signals_received,
                "signals_processed": self.signals_processed,
                "signals_published": self.signals_published,
                "aggregation_latency_ms": self.aggregation_latency_ms,
                "active_strategies": len(self.active_strategies),
                "subscriptions": len(self._subscriptions)
            }
        }
    
    async def get_active_strategies(self) -> Dict[str, str]:
        """Get currently active strategies."""
        return self.active_strategies.copy()
    
    async def reload_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Hot-reload agent configuration."""
        try:
            # Update agent config
            self.config.update(new_config)
            
            # Update Signal Manager config if provided
            if 'signal_manager' in new_config and self.signal_manager:
                # Could implement config hot-reloading in SignalManager
                pass
            
            self.logger.info("Configuration reloaded successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False 