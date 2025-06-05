"""
Agent messaging system for inter-agent communication.

This module provides the MessageBus implementation that handles routing,
queuing, persistence, and delivery of messages between agents in the
Bistoury multi-agent trading system.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from ..models.agent_messages import (
    Message, MessageType, MessagePriority, MessageStatus, MessageDeliveryMode,
    Subscription, MessageFilter, MessageBatch, DeliveryReceipt, MessagePayload
)

logger = logging.getLogger(__name__)


class MessageHandler:
    """Handler for processing received messages."""
    
    def __init__(
        self,
        handler_func: Callable[[Message], Any],
        filter: Optional[MessageFilter] = None,
        is_async: bool = True
    ):
        self.handler_func = handler_func
        self.filter = filter or MessageFilter()
        self.is_async = is_async
        self.call_count = 0
        self.last_called = None
        self.errors = 0
    
    async def handle(self, message: Message) -> bool:
        """Handle a message if it matches the filter."""
        if not self.filter.matches(message):
            return False
        
        try:
            self.call_count += 1
            self.last_called = datetime.utcnow()
            
            if self.is_async:
                if asyncio.iscoroutinefunction(self.handler_func):
                    await self.handler_func(message)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.handler_func, message)
            else:
                self.handler_func(message)
            
            return True
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Error in message handler: {e}", exc_info=True)
            return False


class MessageQueue:
    """Priority-based message queue with persistence."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues = {
            MessagePriority.CRITICAL: deque(),
            MessagePriority.HIGH: deque(),
            MessagePriority.NORMAL: deque(),
            MessagePriority.LOW: deque()
        }
        self._size = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._dropped_count = 0
    
    async def put(self, message: Message) -> bool:
        """Add message to appropriate priority queue."""
        async with self._lock:
            if self._size >= self.max_size:
                # Drop lowest priority messages first
                if self._drop_lowest_priority():
                    self._dropped_count += 1
                    logger.warning(f"Dropped message due to queue full: {message.id}")
                else:
                    logger.error(f"Queue full, cannot add message: {message.id}")
                    return False
            
            self._queues[message.priority].append(message)
            self._size += 1
            self._not_empty.notify()
            return True
    
    async def get(self) -> Optional[Message]:
        """Get highest priority message from queue."""
        async with self._not_empty:
            while self._size == 0:
                await self._not_empty.wait()
            
            # Get from highest priority queue first
            for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                queue = self._queues[priority]
                if queue:
                    message = queue.popleft()
                    self._size -= 1
                    return message
            
            return None
    
    async def peek(self) -> Optional[Message]:
        """Peek at next message without removing it."""
        async with self._lock:
            for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH,
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                queue = self._queues[priority]
                if queue:
                    return queue[0]
            return None
    
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return self._size
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        async with self._lock:
            return {
                'total_size': self._size,
                'max_size': self.max_size,
                'dropped_count': self._dropped_count,
                'priority_sizes': {
                    priority.value: len(queue) 
                    for priority, queue in self._queues.items()
                }
            }
    
    def _drop_lowest_priority(self) -> bool:
        """Drop a message from the lowest priority non-empty queue."""
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL,
                        MessagePriority.HIGH, MessagePriority.CRITICAL]:
            queue = self._queues[priority]
            if queue:
                queue.popleft()
                self._size -= 1
                return True
        return False


class MessagePersistence:
    """Handles message persistence to disk."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/messages")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def save_message(self, message: Message) -> bool:
        """Save message to persistent storage."""
        try:
            async with self._lock:
                file_path = self.storage_path / f"{message.id}.json"
                data = message.model_dump(mode='json')
                
                # Convert datetime objects to ISO strings
                def serialize_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return obj
                
                # Write to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, default=serialize_datetime, indent=2)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to save message {message.id}: {e}")
            return False
    
    async def load_message(self, message_id: UUID) -> Optional[Message]:
        """Load message from persistent storage."""
        try:
            async with self._lock:
                file_path = self.storage_path / f"{message_id}.json"
                if not file_path.exists():
                    return None
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                return Message.model_validate(data)
                
        except Exception as e:
            logger.error(f"Failed to load message {message_id}: {e}")
            return None
    
    async def delete_message(self, message_id: UUID) -> bool:
        """Delete message from persistent storage."""
        try:
            async with self._lock:
                file_path = self.storage_path / f"{message_id}.json"
                if file_path.exists():
                    file_path.unlink()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete message {message_id}: {e}")
            return False
    
    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Clean up expired message files."""
        try:
            async with self._lock:
                cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
                deleted_count = 0
                
                for file_path in self.storage_path.glob("*.json"):
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired messages: {e}")
            return 0


class MessageBus:
    """Central message bus for inter-agent communication."""
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        enable_persistence: bool = True,
        storage_path: Optional[Path] = None,
        retry_interval: float = 5.0,
        max_retry_attempts: int = 3
    ):
        self.max_queue_size = max_queue_size
        self.retry_interval = retry_interval
        self.max_retry_attempts = max_retry_attempts
        
        # Core components
        self._queue = MessageQueue(max_queue_size)
        self._persistence = MessagePersistence(storage_path) if enable_persistence else None
        
        # Subscriptions and routing
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._topic_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Delivery tracking
        self._pending_confirmations: Dict[UUID, Message] = {}
        self._delivery_receipts: Dict[UUID, DeliveryReceipt] = {}
        
        # Statistics
        self._stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'messages_retried': 0,
            'active_subscriptions': 0,
            'uptime_start': datetime.utcnow()
        }
        
        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        logger.info("MessageBus initialized")
    
    async def start(self) -> None:
        """Start the message bus."""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._retry_processor()),
            asyncio.create_task(self._cleanup_processor())
        ]
        
        logger.info("MessageBus started")
    
    async def stop(self) -> None:
        """Stop the message bus gracefully."""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("MessageBus stopped")
    
    async def send_message(
        self,
        message: Message,
        wait_for_confirmation: bool = False
    ) -> bool:
        """Send a message through the bus."""
        if not self._running:
            logger.warning("Cannot send message: MessageBus not running")
            return False
        
        try:
            # Persist message if enabled
            if self._persistence and message.delivery_mode != MessageDeliveryMode.FIRE_AND_FORGET:
                await self._persistence.save_message(message)
            
            # Add to queue
            success = await self._queue.put(message)
            if success:
                self._stats['messages_sent'] += 1
                
                # Track delivery confirmation if required
                if message.delivery_mode != MessageDeliveryMode.FIRE_AND_FORGET:
                    self._pending_confirmations[message.id] = message
                
                # Wait for confirmation if requested
                if wait_for_confirmation and message.delivery_mode != MessageDeliveryMode.FIRE_AND_FORGET:
                    return await self._wait_for_confirmation(message.id)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {e}")
            return False
    
    async def publish(
        self,
        topic: str,
        message_type: MessageType,
        payload: MessagePayload,
        sender: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> bool:
        """Publish a message to a topic."""
        message = Message(
            type=message_type,
            sender=sender,
            topic=topic,
            payload=payload,
            priority=priority,
            **kwargs
        )
        
        return await self.send_message(message)
    
    async def subscribe(
        self,
        agent_id: str,
        filter: MessageFilter,
        handler: Callable[[Message], Any],
        is_async: bool = True
    ) -> Subscription:
        """Subscribe to messages matching a filter."""
        subscription = Subscription(
            subscriber=agent_id,
            filter=filter
        )
        
        # Add subscription
        self._subscriptions[agent_id].append(subscription)
        
        # Add handler
        message_handler = MessageHandler(handler, filter, is_async)
        self._handlers[agent_id].append(message_handler)
        
        # Track topic subscriptions
        if filter.topics:
            for topic in filter.topics:
                self._topic_subscriptions[topic].add(agent_id)
        
        self._stats['active_subscriptions'] += 1
        
        logger.info(f"Agent {agent_id} subscribed to messages")
        return subscription
    
    async def unsubscribe(self, agent_id: str, subscription_id: UUID) -> bool:
        """Unsubscribe from messages."""
        try:
            # Find and remove subscription
            subscriptions = self._subscriptions[agent_id]
            for i, subscription in enumerate(subscriptions):
                if subscription.id == subscription_id:
                    # Remove subscription
                    del subscriptions[i]
                    
                    # Remove from topic subscriptions
                    if subscription.filter.topics:
                        for topic in subscription.filter.topics:
                            self._topic_subscriptions[topic].discard(agent_id)
                    
                    # Remove corresponding handler (simplified - remove all handlers)
                    self._handlers[agent_id].clear()
                    
                    self._stats['active_subscriptions'] -= 1
                    
                    logger.info(f"Agent {agent_id} unsubscribed from messages")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe agent {agent_id}: {e}")
            return False
    
    async def send_response(
        self,
        original_message: Message,
        response_payload: MessagePayload,
        sender: str,
        success: bool = True
    ) -> bool:
        """Send a response to a message."""
        response_type = MessageType.RESPONSE_SUCCESS if success else MessageType.RESPONSE_ERROR
        response = original_message.create_response(response_type, response_payload, sender)
        
        return await self.send_message(response)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        queue_stats = await self._queue.get_stats()
        
        uptime = datetime.utcnow() - self._stats['uptime_start']
        
        return {
            **self._stats,
            'uptime_seconds': uptime.total_seconds(),
            'queue_stats': queue_stats,
            'pending_confirmations': len(self._pending_confirmations),
            'delivery_receipts': len(self._delivery_receipts),
            'running': self._running
        }
    
    async def _message_processor(self) -> None:
        """Background task to process messages from queue."""
        logger.info("Message processor started")
        
        try:
            while self._running:
                try:
                    # Get next message from queue
                    message = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                    
                    if message:
                        await self._deliver_message(message)
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in message processor: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            logger.info("Message processor cancelled")
        except Exception as e:
            logger.error(f"Message processor error: {e}")
    
    async def _deliver_message(self, message: Message) -> None:
        """Deliver a message to appropriate handlers."""
        try:
            delivered_count = 0
            
            # Direct delivery to specific receiver
            if message.receiver:
                if message.receiver in self._handlers:
                    for handler in self._handlers[message.receiver]:
                        if await handler.handle(message):
                            delivered_count += 1
            
            # Topic-based delivery
            elif message.topic:
                # Check all subscriptions for topic matches (not just exact matches)
                for agent_id, handlers in self._handlers.items():
                    for handler in handlers:
                        if await handler.handle(message):
                            delivered_count += 1
            
            # Broadcast delivery
            else:
                for agent_id, handlers in self._handlers.items():
                    for handler in handlers:
                        if await handler.handle(message):
                            delivered_count += 1
            
            # Update message status and statistics
            if delivered_count > 0:
                message.status = MessageStatus.DELIVERED
                self._stats['messages_delivered'] += 1
                
                # Create delivery receipt
                if message.delivery_mode != MessageDeliveryMode.FIRE_AND_FORGET:
                    receipt = DeliveryReceipt(
                        message_id=message.id,
                        receiver=message.receiver or "broadcast",
                        status=MessageStatus.DELIVERED
                    )
                    self._delivery_receipts[message.id] = receipt
                    
                    # Remove from pending confirmations
                    self._pending_confirmations.pop(message.id, None)
                    
                    # Clean up persistence
                    if self._persistence:
                        await self._persistence.delete_message(message.id)
            else:
                message.status = MessageStatus.FAILED
                self._stats['messages_failed'] += 1
                
                # Queue for retry if applicable
                if message.can_retry():
                    message.increment_retry()
                    await self._queue.put(message)
                    self._stats['messages_retried'] += 1
                    logger.warning(f"Message {message.id} queued for retry {message.retry_count}")
                else:
                    logger.error(f"Message {message.id} delivery failed permanently")
                    
                    # Create failed delivery receipt
                    if message.delivery_mode != MessageDeliveryMode.FIRE_AND_FORGET:
                        receipt = DeliveryReceipt(
                            message_id=message.id,
                            receiver=message.receiver or "broadcast",
                            status=MessageStatus.FAILED,
                            error_message="No matching handlers found"
                        )
                        self._delivery_receipts[message.id] = receipt
                        self._pending_confirmations.pop(message.id, None)
                        
        except Exception as e:
            logger.error(f"Error delivering message {message.id}: {e}")
            message.status = MessageStatus.FAILED
            self._stats['messages_failed'] += 1
    
    async def _retry_processor(self) -> None:
        """Background task to handle message retries."""
        logger.info("Retry processor started")
        
        try:
            while self._running:
                try:
                    await asyncio.sleep(self.retry_interval)
                    
                    # Check for messages that need retry
                    expired_messages = []
                    for message_id, message in self._pending_confirmations.items():
                        if message.is_expired():
                            expired_messages.append(message_id)
                    
                    # Clean up expired messages
                    for message_id in expired_messages:
                        message = self._pending_confirmations.pop(message_id, None)
                        if message and self._persistence:
                            await self._persistence.delete_message(message_id)
                        
                        logger.warning(f"Message {message_id} expired")
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in retry processor: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Retry processor cancelled")
    
    async def _cleanup_processor(self) -> None:
        """Background task for cleanup operations."""
        logger.info("Cleanup processor started")
        
        try:
            while self._running:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    
                    # Clean up old delivery receipts (keep for 24 hours)
                    cutoff_time = datetime.utcnow() - timedelta(hours=24)
                    expired_receipts = [
                        receipt_id for receipt_id, receipt in self._delivery_receipts.items()
                        if receipt.delivered_at < cutoff_time
                    ]
                    
                    for receipt_id in expired_receipts:
                        self._delivery_receipts.pop(receipt_id, None)
                    
                    # Clean up persistent storage
                    if self._persistence:
                        deleted_count = await self._persistence.cleanup_expired()
                        if deleted_count > 0:
                            logger.info(f"Cleaned up {deleted_count} expired message files")
                            
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup processor: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Cleanup processor cancelled")
    
    async def _wait_for_confirmation(self, message_id: UUID, timeout: float = 30.0) -> bool:
        """Wait for message delivery confirmation."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if message_id in self._delivery_receipts:
                receipt = self._delivery_receipts[message_id]
                return receipt.status == MessageStatus.DELIVERED
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Timeout waiting for confirmation of message {message_id}")
        return False


# Convenience functions for creating common message types

async def create_market_data_message(
    symbol: str,
    price: float,
    volume: float,
    sender: str,
    source: str = "hyperliquid"
) -> Message:
    """Create a market data update message."""
    from ..models.agent_messages import MarketDataPayload
    from decimal import Decimal
    
    payload = MarketDataPayload(
        symbol=symbol,
        price=Decimal(str(price)),
        volume=Decimal(str(volume)),
        timestamp=datetime.utcnow(),
        source=source
    )
    
    return Message(
        type=MessageType.DATA_MARKET_UPDATE,
        sender=sender,
        topic=f"data.market.{symbol}",
        payload=payload,
        priority=MessagePriority.HIGH
    )


async def create_trading_signal_message(
    symbol: str,
    signal_type: str,
    direction: str,
    confidence: float,
    strategy: str,
    reasoning: str,
    sender: str
) -> Message:
    """Create a trading signal message."""
    from ..models.agent_messages import TradingSignalPayload
    
    payload = TradingSignalPayload(
        symbol=symbol,
        signal_type=signal_type,
        direction=direction,
        confidence=confidence,
        strength=confidence * 10,  # Convert to 0-10 scale
        timeframe="1m",
        strategy=strategy,
        reasoning=reasoning
    )
    
    return Message(
        type=MessageType.SIGNAL_GENERATED,
        sender=sender,
        topic=f"signals.{symbol}",
        payload=payload,
        priority=MessagePriority.HIGH
    )


async def create_system_event_message(
    event_type: str,
    component: str,
    status: str,
    description: str,
    sender: str
) -> Message:
    """Create a system event message."""
    from ..models.agent_messages import SystemEventPayload
    
    payload = SystemEventPayload(
        event_type=event_type,
        component=component,
        status=status,
        description=description
    )
    
    return Message(
        type=MessageType.SYSTEM_HEALTH_CHECK,
        sender=sender,
        topic="system.events",
        payload=payload,
        priority=MessagePriority.NORMAL
    ) 