"""
Unit tests for the agent messaging system.

Tests cover message models, MessageBus functionality, pub/sub system,
persistence, delivery confirmation, and retry logic.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from bistoury.models.agent_messages import (
    Message, MessageType, MessagePriority, MessageStatus, MessageDeliveryMode,
    MessageFilter, Subscription, MessageBatch, DeliveryReceipt,
    MarketDataPayload, TradingSignalPayload, SystemEventPayload
)
from bistoury.agents.messaging import (
    MessageBus, MessageHandler, MessageQueue, MessagePersistence,
    create_market_data_message, create_trading_signal_message, create_system_event_message
)


class TestMessageModels:
    """Test message model validation and functionality."""
    
    def test_market_data_payload_validation(self):
        """Test market data payload validation."""
        from decimal import Decimal
        
        payload = MarketDataPayload(
            symbol="BTC",
            price=Decimal("50000.00"),
            volume=Decimal("1.5"),
            timestamp=datetime.utcnow(),
            source="hyperliquid"
        )
        
        assert payload.symbol == "BTC"
        assert payload.price == Decimal("50000.00")
        assert payload.volume == Decimal("1.5")
        assert payload.source == "hyperliquid"
    
    def test_trading_signal_payload_validation(self):
        """Test trading signal payload validation."""
        payload = TradingSignalPayload(
            symbol="BTC",
            signal_type="bullish_engulfing",
            direction="BUY",
            confidence=0.85,
            strength=8.5,
            timeframe="5m",
            strategy="candlestick_patterns",
            reasoning="Strong bullish engulfing pattern with high volume"
        )
        
        assert payload.symbol == "BTC"
        assert payload.confidence == 0.85
        assert payload.strength == 8.5
        assert payload.direction == "BUY"
    
    def test_message_creation_and_validation(self):
        """Test message creation with proper validation."""
        payload = SystemEventPayload(
            event_type="startup",
            component="collector",
            status="running",
            description="Data collector started successfully"
        )
        
        message = Message(
            type=MessageType.SYSTEM_STARTUP,
            sender="collector_agent",
            payload=payload,
            priority=MessagePriority.HIGH
        )
        
        assert message.type == MessageType.SYSTEM_STARTUP
        assert message.sender == "collector_agent"
        assert message.priority == MessagePriority.HIGH
        assert message.status == MessageStatus.PENDING
        assert message.retry_count == 0
        assert not message.is_expired()
    
    def test_message_with_expiration(self):
        """Test message expiration functionality."""
        payload = {"test": "data"}
        
        # Create message that expires in 1 second
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="test_agent",
            payload=payload,
            expires_at=datetime.utcnow() + timedelta(seconds=1)
        )
        
        assert not message.is_expired()
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        assert message.is_expired()
    
    def test_message_retry_logic(self):
        """Test message retry logic."""
        payload = {"test": "data"}
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="test_agent",
            payload=payload,
            max_retries=2
        )
        
        assert message.can_retry()
        
        message.increment_retry()
        assert message.retry_count == 1
        assert message.can_retry()
        
        message.increment_retry()
        assert message.retry_count == 2
        assert not message.can_retry()  # Reached max retries
    
    def test_message_tags_and_headers(self):
        """Test message tags and headers functionality."""
        payload = {"test": "data"}
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="test_agent",
            payload=payload
        )
        
        # Test tags
        message.add_tag("urgent")
        message.add_tag("btc")
        assert message.has_tag("urgent")
        assert message.has_tag("btc")
        assert not message.has_tag("eth")
        
        # Test headers
        message.add_header("source", "hyperliquid")
        message.add_header("version", "1.0")
        assert message.headers["source"] == "hyperliquid"
        assert message.headers["version"] == "1.0"
    
    def test_message_response_creation(self):
        """Test creating response messages."""
        original_payload = {"request": "data"}
        response_payload = {"response": "success"}
        
        original = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            receiver="trader",
            payload=original_payload
        )
        
        response = original.create_response(
            MessageType.RESPONSE_SUCCESS,
            response_payload,
            "trader"
        )
        
        assert response.correlation_id == original.id
        assert response.sender == "trader"
        assert response.receiver == "collector"
        assert response.type == MessageType.RESPONSE_SUCCESS
        assert response.headers["in_response_to"] == str(original.id)


class TestMessageFilter:
    """Test message filtering functionality."""
    
    def test_message_type_filtering(self):
        """Test filtering by message type."""
        filter = MessageFilter(
            message_types=[MessageType.DATA_MARKET_UPDATE, MessageType.SIGNAL_GENERATED]
        )
        
        market_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        signal_message = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="strategy",
            payload={"test": "signal"}
        )
        
        system_message = Message(
            type=MessageType.SYSTEM_STARTUP,
            sender="orchestrator",
            payload={"test": "system"}
        )
        
        assert filter.matches(market_message)
        assert filter.matches(signal_message)
        assert not filter.matches(system_message)
    
    def test_sender_filtering(self):
        """Test filtering by sender."""
        filter = MessageFilter(senders=["collector", "trader"])
        
        collector_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        strategy_message = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="strategy",
            payload={"test": "signal"}
        )
        
        assert filter.matches(collector_message)
        assert not filter.matches(strategy_message)
    
    def test_topic_filtering(self):
        """Test filtering by topic."""
        filter = MessageFilter(topics=["data.market", "signals"])
        
        market_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            topic="data.market.BTC",
            payload={"test": "data"}
        )
        
        signal_message = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="strategy",
            topic="signals.BTC",
            payload={"test": "signal"}
        )
        
        system_message = Message(
            type=MessageType.SYSTEM_STARTUP,
            sender="orchestrator",
            topic="system.events",
            payload={"test": "system"}
        )
        
        assert filter.matches(market_message)
        assert filter.matches(signal_message)
        assert not filter.matches(system_message)
    
    def test_priority_filtering(self):
        """Test filtering by priority."""
        filter = MessageFilter(priorities=[MessagePriority.HIGH, MessagePriority.CRITICAL])
        
        high_message = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="strategy",
            payload={"test": "signal"},
            priority=MessagePriority.HIGH
        )
        
        normal_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"},
            priority=MessagePriority.NORMAL
        )
        
        assert filter.matches(high_message)
        assert not filter.matches(normal_message)
    
    def test_tag_filtering(self):
        """Test filtering by tags."""
        filter = MessageFilter(tags=["urgent", "btc"])
        
        urgent_message = Message(
            type=MessageType.RISK_LIMIT_BREACH,
            sender="risk_manager",
            payload={"test": "risk"},
            tags=["urgent", "position"]
        )
        
        btc_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"},
            tags=["btc", "price"]
        )
        
        normal_message = Message(
            type=MessageType.SYSTEM_STARTUP,
            sender="orchestrator",
            payload={"test": "system"},
            tags=["system", "startup"]
        )
        
        assert filter.matches(urgent_message)  # Has "urgent" tag
        assert filter.matches(btc_message)     # Has "btc" tag
        assert not filter.matches(normal_message)  # No matching tags
    
    def test_require_all_tags_filtering(self):
        """Test filtering requiring all tags to match."""
        filter = MessageFilter(
            tags=["urgent", "btc"],
            require_all_tags=True
        )
        
        full_match_message = Message(
            type=MessageType.RISK_LIMIT_BREACH,
            sender="risk_manager",
            payload={"test": "risk"},
            tags=["urgent", "btc", "position"]
        )
        
        partial_match_message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"},
            tags=["btc", "price"]
        )
        
        assert filter.matches(full_match_message)     # Has both tags
        assert not filter.matches(partial_match_message)  # Missing "urgent" tag


@pytest.mark.asyncio
class TestMessageQueue:
    """Test message queue functionality."""
    
    async def test_basic_queue_operations(self):
        """Test basic queue put/get operations."""
        queue = MessageQueue(max_size=100)
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        # Test put
        success = await queue.put(message)
        assert success
        assert await queue.size() == 1
        
        # Test get
        retrieved = await queue.get()
        assert retrieved.id == message.id
        assert await queue.size() == 0
    
    async def test_priority_ordering(self):
        """Test that messages are retrieved by priority."""
        queue = MessageQueue(max_size=100)
        
        # Add messages in reverse priority order
        low_msg = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"priority": "low"},
            priority=MessagePriority.LOW
        )
        
        normal_msg = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"priority": "normal"},
            priority=MessagePriority.NORMAL
        )
        
        high_msg = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="strategy",
            payload={"priority": "high"},
            priority=MessagePriority.HIGH
        )
        
        critical_msg = Message(
            type=MessageType.RISK_EMERGENCY_STOP,
            sender="risk_manager",
            payload={"priority": "critical"},
            priority=MessagePriority.CRITICAL
        )
        
        # Add in random order
        await queue.put(normal_msg)
        await queue.put(critical_msg)
        await queue.put(low_msg)
        await queue.put(high_msg)
        
        # Should get back in priority order
        first = await queue.get()
        assert first.priority == MessagePriority.CRITICAL
        
        second = await queue.get()
        assert second.priority == MessagePriority.HIGH
        
        third = await queue.get()
        assert third.priority == MessagePriority.NORMAL
        
        fourth = await queue.get()
        assert fourth.priority == MessagePriority.LOW
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test queue behavior when max size is reached."""
        queue = MessageQueue(max_size=2)
        
        # Fill queue to capacity
        msg1 = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"order": 1},
            priority=MessagePriority.NORMAL
        )
        
        msg2 = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"order": 2},
            priority=MessagePriority.HIGH
        )
        
        msg3 = Message(
            type=MessageType.RISK_EMERGENCY_STOP,
            sender="risk_manager",
            payload={"order": 3},
            priority=MessagePriority.CRITICAL
        )
        
        # Add first two messages
        assert await queue.put(msg1)
        assert await queue.put(msg2)
        assert await queue.size() == 2
        
        # Third message should cause overflow and drop lowest priority
        assert await queue.put(msg3)
        assert await queue.size() == 2
        
        # Should get high priority and critical messages
        first = await queue.get()
        assert first.priority == MessagePriority.CRITICAL
        
        second = await queue.get()
        assert second.priority == MessagePriority.HIGH
        
        # The normal priority message should have been dropped
        stats = await queue.get_stats()
        assert stats['dropped_count'] == 1


class TestMessagePersistence:
    """Test message persistence functionality."""
    
    @pytest.mark.asyncio
    async def test_save_and_load_message(self, tmp_path):
        """Test saving and loading messages."""
        persistence = MessagePersistence(tmp_path)
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"},
            priority=MessagePriority.HIGH
        )
        
        # Save message
        success = await persistence.save_message(message)
        assert success
        
        # Check file exists
        file_path = tmp_path / f"{message.id}.json"
        assert file_path.exists()
        
        # Load message
        loaded = await persistence.load_message(message.id)
        assert loaded is not None
        assert loaded.id == message.id
        assert loaded.type == message.type
        assert loaded.sender == message.sender
        assert loaded.priority == message.priority
    
    @pytest.mark.asyncio
    async def test_delete_message(self, tmp_path):
        """Test deleting persisted messages."""
        persistence = MessagePersistence(tmp_path)
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        # Save and delete
        await persistence.save_message(message)
        file_path = tmp_path / f"{message.id}.json"
        assert file_path.exists()
        
        success = await persistence.delete_message(message.id)
        assert success
        assert not file_path.exists()
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_messages(self, tmp_path):
        """Test cleanup of expired message files."""
        persistence = MessagePersistence(tmp_path)
        
        # Create some test files with different ages
        old_file = tmp_path / "old_message.json"
        recent_file = tmp_path / "recent_message.json"
        
        # Create old file (simulate 25 hours ago)
        old_file.write_text('{"test": "old"}')
        old_time = datetime.utcnow() - timedelta(hours=25)
        
        # Use os.utime for setting file times
        import os
        os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
        
        # Create recent file
        recent_file.write_text('{"test": "recent"}')
        
        assert old_file.exists()
        assert recent_file.exists()
        
        # Cleanup with 24 hour threshold
        deleted_count = await persistence.cleanup_expired(max_age_hours=24)
        
        assert deleted_count == 1
        assert not old_file.exists()
        assert recent_file.exists()


class TestMessageHandler:
    """Test message handler functionality."""
    
    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Test async message handler."""
        received_messages = []
        
        async def async_handler(message: Message):
            received_messages.append(message)
        
        filter = MessageFilter(message_types=[MessageType.DATA_MARKET_UPDATE])
        handler = MessageHandler(async_handler, filter, is_async=True)
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        handled = await handler.handle(message)
        assert handled
        assert len(received_messages) == 1
        assert received_messages[0].id == message.id
        assert handler.call_count == 1
    
    @pytest.mark.asyncio
    async def test_sync_handler(self):
        """Test sync message handler."""
        received_messages = []
        
        def sync_handler(message: Message):
            received_messages.append(message)
        
        filter = MessageFilter()
        handler = MessageHandler(sync_handler, filter, is_async=True)
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        handled = await handler.handle(message)
        assert handled
        assert len(received_messages) == 1
        assert handler.call_count == 1
    
    @pytest.mark.asyncio
    async def test_handler_with_filter(self):
        """Test handler that filters messages."""
        received_messages = []
        
        async def handler_func(message: Message):
            received_messages.append(message)
        
        filter = MessageFilter(message_types=[MessageType.SIGNAL_GENERATED])
        handler = MessageHandler(handler_func, filter)
        
        # This should be handled
        signal_msg = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="strategy",
            payload={"test": "signal"}
        )
        
        # This should be filtered out
        data_msg = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        handled1 = await handler.handle(signal_msg)
        handled2 = await handler.handle(data_msg)
        
        assert handled1
        assert not handled2
        assert len(received_messages) == 1
        assert handler.call_count == 1
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """Test handler error handling."""
        async def failing_handler(message: Message):
            raise ValueError("Handler error")
        
        handler = MessageHandler(failing_handler)
        
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        handled = await handler.handle(message)
        assert not handled  # Should return False on error
        assert handler.errors == 1


@pytest.mark.asyncio
class TestMessageBus:
    """Test MessageBus functionality."""
    
    async def test_message_bus_lifecycle(self, tmp_path):
        """Test MessageBus start/stop lifecycle."""
        bus = MessageBus(storage_path=tmp_path, enable_persistence=False)
        
        assert not bus._running
        
        await bus.start()
        assert bus._running
        
        await bus.stop()
        assert not bus._running
    
    async def test_send_and_deliver_message(self, tmp_path):
        """Test sending and delivering messages."""
        bus = MessageBus(storage_path=tmp_path, enable_persistence=False)
        received_messages = []
        
        async def message_handler(message: Message):
            received_messages.append(message)
        
        # Subscribe to messages
        filter = MessageFilter(message_types=[MessageType.DATA_MARKET_UPDATE])
        await bus.subscribe("test_agent", filter, message_handler)
        
        await bus.start()
        
        # Send message
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            receiver="test_agent",
            payload={"test": "data"}
        )
        
        success = await bus.send_message(message)
        assert success
        
        # Wait for delivery
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
        assert received_messages[0].id == message.id
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_topic_based_publishing(self, tmp_path):
        """Test topic-based message publishing."""
        bus = MessageBus(storage_path=tmp_path, enable_persistence=False)
        btc_messages = []
        eth_messages = []
        
        async def btc_handler(message: Message):
            btc_messages.append(message)
        
        async def eth_handler(message: Message):
            eth_messages.append(message)
        
        # Subscribe to different topics
        btc_filter = MessageFilter(topics=["data.market.BTC"])
        eth_filter = MessageFilter(topics=["data.market.ETH"])
        
        await bus.subscribe("btc_agent", btc_filter, btc_handler)
        await bus.subscribe("eth_agent", eth_filter, eth_handler)
        
        await bus.start()
        
        # Publish messages to different topics
        await bus.publish(
            "data.market.BTC",
            MessageType.DATA_MARKET_UPDATE,
            {"price": 50000},
            "collector"
        )
        
        await bus.publish(
            "data.market.ETH",
            MessageType.DATA_MARKET_UPDATE,
            {"price": 3000},
            "collector"
        )
        
        # Wait for delivery
        await asyncio.sleep(0.1)
        
        assert len(btc_messages) == 1
        assert len(eth_messages) == 1
        assert btc_messages[0].topic == "data.market.BTC"
        assert eth_messages[0].topic == "data.market.ETH"
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_message_response(self, tmp_path):
        """Test sending message responses."""
        bus = MessageBus(storage_path=tmp_path, enable_persistence=False)
        
        await bus.start()
        
        # Original message
        request = Message(
            type=MessageType.COMMAND_START_COLLECTION,
            sender="orchestrator",
            receiver="collector",
            payload={"command": "start"}
        )
        
        # Send response
        response_payload = {"status": "started"}
        success = await bus.send_response(request, response_payload, "collector", True)
        assert success
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_message_bus_statistics(self, tmp_path):
        """Test MessageBus statistics tracking."""
        bus = MessageBus(storage_path=tmp_path, enable_persistence=False)
        
        await bus.start()
        
        # Initial stats
        stats = await bus.get_stats()
        assert stats['messages_sent'] == 0
        assert stats['running'] is True
        
        # Send a message
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            payload={"test": "data"}
        )
        
        await bus.send_message(message)
        
        # Check updated stats
        stats = await bus.get_stats()
        assert stats['messages_sent'] == 1
        
        await bus.stop()


class TestConvenienceFunctions:
    """Test convenience functions for creating messages."""
    
    @pytest.mark.asyncio
    async def test_create_market_data_message(self):
        """Test creating market data messages."""
        message = await create_market_data_message(
            symbol="BTC",
            price=50000.0,
            volume=1.5,
            sender="collector"
        )
        
        assert message.type == MessageType.DATA_MARKET_UPDATE
        assert message.sender == "collector"
        assert message.topic == "data.market.BTC"
        assert message.priority == MessagePriority.HIGH
        
        # Check payload
        payload = message.payload
        assert isinstance(payload, MarketDataPayload)
        assert payload.symbol == "BTC"
        assert payload.source == "hyperliquid"
    
    @pytest.mark.asyncio
    async def test_create_trading_signal_message(self):
        """Test creating trading signal messages."""
        message = await create_trading_signal_message(
            symbol="BTC",
            signal_type="bullish_engulfing",
            direction="BUY",
            confidence=0.85,
            strategy="candlestick_patterns",
            reasoning="Strong bullish pattern",
            sender="strategy_agent"
        )
        
        assert message.type == MessageType.SIGNAL_GENERATED
        assert message.sender == "strategy_agent"
        assert message.topic == "signals.BTC"
        assert message.priority == MessagePriority.HIGH
        
        # Check payload
        payload = message.payload
        assert isinstance(payload, TradingSignalPayload)
        assert payload.symbol == "BTC"
        assert payload.confidence == 0.85
        assert payload.direction == "BUY"
    
    @pytest.mark.asyncio
    async def test_create_system_event_message(self):
        """Test creating system event messages."""
        message = await create_system_event_message(
            event_type="startup",
            component="collector",
            status="running",
            description="Collector started successfully",
            sender="orchestrator"
        )
        
        assert message.type == MessageType.SYSTEM_HEALTH_CHECK
        assert message.sender == "orchestrator"
        assert message.topic == "system.events"
        assert message.priority == MessagePriority.NORMAL
        
        # Check payload
        payload = message.payload
        assert isinstance(payload, SystemEventPayload)
        assert payload.event_type == "startup"
        assert payload.component == "collector"


# Integration tests

class TestMessagingIntegration:
    """Integration tests for the complete messaging system."""
    
    @pytest.mark.asyncio
    async def test_full_messaging_workflow(self, tmp_path):
        """Test complete messaging workflow from creation to delivery."""
        bus = MessageBus(storage_path=tmp_path)
        
        # Track received messages
        collector_messages = []
        trader_messages = []
        all_messages = []
        
        async def collector_handler(message: Message):
            collector_messages.append(message)
        
        async def trader_handler(message: Message):
            trader_messages.append(message)
        
        async def broadcast_handler(message: Message):
            all_messages.append(message)
        
        # Set up subscriptions
        collector_filter = MessageFilter(
            message_types=[MessageType.COMMAND_START_COLLECTION],
            senders=["orchestrator"]
        )
        
        trader_filter = MessageFilter(
            topics=["signals"],
            priorities=[MessagePriority.HIGH]
        )
        
        broadcast_filter = MessageFilter(
            message_types=[MessageType.SYSTEM_STARTUP]
        )
        
        await bus.subscribe("collector", collector_filter, collector_handler)
        await bus.subscribe("trader", trader_filter, trader_handler)
        await bus.subscribe("monitor", broadcast_filter, broadcast_handler)
        
        await bus.start()
        
        # 1. Send command to collector
        command_msg = Message(
            type=MessageType.COMMAND_START_COLLECTION,
            sender="orchestrator",
            receiver="collector",
            payload={"symbols": ["BTC", "ETH"]}
        )
        
        await bus.send_message(command_msg)
        
        # 2. Publish trading signal
        signal_msg = await create_trading_signal_message(
            symbol="BTC",
            signal_type="bullish",
            direction="BUY",
            confidence=0.9,
            strategy="ml_model",
            reasoning="Strong buy signal detected",
            sender="strategy_agent"
        )
        
        await bus.send_message(signal_msg)
        
        # 3. Broadcast system startup
        system_msg = Message(
            type=MessageType.SYSTEM_STARTUP,
            sender="orchestrator",
            payload={"status": "all_systems_operational"}
        )
        
        await bus.send_message(system_msg)
        
        # Wait for message processing
        await asyncio.sleep(0.2)
        
        # Verify deliveries
        assert len(collector_messages) == 1
        assert collector_messages[0].type == MessageType.COMMAND_START_COLLECTION
        
        assert len(trader_messages) == 1
        assert trader_messages[0].type == MessageType.SIGNAL_GENERATED
        
        assert len(all_messages) == 1
        assert all_messages[0].type == MessageType.SYSTEM_STARTUP
        
        # Check statistics
        stats = await bus.get_stats()
        assert stats['messages_sent'] == 3
        assert stats['messages_delivered'] >= 3  # Could be more due to multiple handlers
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_message_retry_and_persistence(self, tmp_path):
        """Test message retry logic and persistence."""
        bus = MessageBus(
            storage_path=tmp_path,
            enable_persistence=True,
            retry_interval=0.1
        )
        
        failed_messages = []
        
        async def failing_handler(message: Message):
            failed_messages.append(message)
            raise RuntimeError("Handler failure")
        
        filter = MessageFilter()
        await bus.subscribe("failing_agent", filter, failing_handler)
        
        await bus.start()
        
        # Send message with retry enabled
        message = Message(
            type=MessageType.DATA_MARKET_UPDATE,
            sender="collector",
            receiver="failing_agent",
            payload={"test": "retry"},
            delivery_mode=MessageDeliveryMode.AT_LEAST_ONCE,
            max_retries=2
        )
        
        await bus.send_message(message)
        
        # Wait for retries
        await asyncio.sleep(0.5)
        
        # Should have attempted delivery multiple times
        assert len(failed_messages) >= 2
        
        # Check that message was persisted and cleaned up
        stats = await bus.get_stats()
        assert stats['messages_failed'] > 0
        assert stats['messages_retried'] > 0
        
        await bus.stop()


# Fixtures

@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        type=MessageType.DATA_MARKET_UPDATE,
        sender="test_agent",
        payload={"test": "data"},
        priority=MessagePriority.NORMAL
    )


@pytest.fixture
def sample_filter():
    """Create a sample message filter for testing."""
    return MessageFilter(
        message_types=[MessageType.DATA_MARKET_UPDATE, MessageType.SIGNAL_GENERATED],
        priorities=[MessagePriority.HIGH, MessagePriority.CRITICAL]
    ) 