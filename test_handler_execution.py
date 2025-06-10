#!/usr/bin/env python3
"""
Test handler execution to see what happens when agents actually process messages
"""

import asyncio
import sys
import traceback
from datetime import datetime, timezone
from decimal import Decimal

# Add project root to path
sys.path.append('.')

from src.bistoury.agents.messaging import MessageBus
from src.bistoury.models.agent_messages import MessageFilter, MessageType, Message, MarketDataPayload
from src.bistoury.agents.candlestick_strategy_agent import CandlestickStrategyAgent, CandlestickStrategyConfig
from src.bistoury.agents.position_manager_agent import PositionManagerAgent, PositionManagerConfig


async def test_handler_execution():
    """Test what happens when agents actually execute their handle_message methods."""
    
    print("🔍 TESTING AGENT HANDLER EXECUTION")
    print("=" * 50)
    
    # Create message bus
    message_bus = MessageBus(enable_persistence=False)
    await message_bus.start()
    
    # Create agents like in paper trading
    candlestick_config = CandlestickStrategyConfig(
        symbols=["BTC"],
        agent_name="candlestick_strategy"
    )
    candlestick_agent = CandlestickStrategyAgent(config=candlestick_config)
    candlestick_agent._message_bus = message_bus
    
    position_config = PositionManagerConfig(initial_balance=Decimal("10000"))
    position_agent = PositionManagerAgent(name="position_manager", config=position_config)
    position_agent._message_bus = message_bus
    
    # Track success/failures
    candlestick_handled = False
    candlestick_error = None
    position_handled = False
    position_error = None
    
    # Wrap agent handlers to track execution
    original_candlestick_handler = candlestick_agent.handle_message
    original_position_handler = position_agent.handle_message
    
    async def wrapped_candlestick_handler(message):
        nonlocal candlestick_handled, candlestick_error
        try:
            print(f"📨 Candlestick handler called with: {message.id}")
            await original_candlestick_handler(message)
            candlestick_handled = True
            print(f"✅ Candlestick handler completed successfully")
        except Exception as e:
            candlestick_error = e
            print(f"❌ Candlestick handler failed: {e}")
            traceback.print_exc()
            raise
            
    async def wrapped_position_handler(message):
        nonlocal position_handled, position_error
        try:
            print(f"📨 Position handler called with: {message.id}")
            await original_position_handler(message)
            position_handled = True
            print(f"✅ Position handler completed successfully")
        except Exception as e:
            position_error = e
            print(f"❌ Position handler failed: {e}")
            traceback.print_exc()
            raise
    
    # Start agents
    print("\n🚀 Starting agents...")
    candlestick_started = await candlestick_agent.start()
    position_started = await position_agent.start()
    print(f"Candlestick agent started: {candlestick_started}")
    print(f"Position agent started: {position_started}")
    
    # Subscribe with wrapped handlers
    print("\n📡 Setting up subscriptions...")
    
    # Candlestick agent subscription
    market_filter = MessageFilter(
        message_types=[MessageType.DATA_MARKET_UPDATE],
        topics=["market_data.BTC.15m"]
    )
    await message_bus.subscribe(
        agent_id=candlestick_agent.agent_id,  # Use actual UUID
        filter=market_filter,
        handler=wrapped_candlestick_handler,
        is_async=True
    )
    print(f"✅ Candlestick agent subscribed with ID: {candlestick_agent.agent_id}")
    
    # Position manager subscription
    signal_filter = MessageFilter(
        message_types=[MessageType.SIGNAL_GENERATED],
        topics=["signals.*"]
    )
    await message_bus.subscribe(
        agent_id=position_agent.agent_id,  # Use actual UUID
        filter=signal_filter,
        handler=wrapped_position_handler,
        is_async=True
    )
    print(f"✅ Position agent subscribed with ID: {position_agent.agent_id}")
    
    # Create market data message like CollectorAgent would
    print("\n📊 Publishing market data message...")
    
    market_payload = MarketDataPayload(
        symbol="BTC",
        price=Decimal("50000"),
        volume=Decimal("100"),
        timestamp=datetime.now(timezone.utc),
        source="collector_agent_historical",
        data={
            "timeframe": "15m",
            "data_type": "candlestick",
            "candle_data": {
                "symbol": "BTC",  # Add required fields
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timeframe": "15m",
                "open": 49000,
                "high": 51000,
                "low": 48000,
                "close": 50000,
                "volume": 100,
                "trade_count": 50
            }
        }
    )
    
    market_success = await message_bus.publish(
        topic="market_data.BTC.15m",
        message_type=MessageType.DATA_MARKET_UPDATE,
        payload=market_payload,
        sender="collector_historical"
    )
    
    print(f"Market data published: {market_success}")
    
    # Wait for processing
    print("\n⏱️  Waiting for message processing...")
    await asyncio.sleep(2.0)
    
    # Check results
    print(f"\n📋 EXECUTION RESULTS:")
    print(f"Candlestick handler called: {candlestick_handled}")
    if candlestick_error:
        print(f"Candlestick error: {candlestick_error}")
    
    print(f"Position handler called: {position_handled}")
    if position_error:
        print(f"Position error: {position_error}")
    
    # Cleanup
    await candlestick_agent.stop()
    await position_agent.stop()
    await message_bus.stop()
    
    # Summary
    if candlestick_handled and not candlestick_error:
        print("\n🎉 Message delivery and handler execution working!")
    else:
        print("\n🔧 Handler execution issues identified.")


if __name__ == "__main__":
    asyncio.run(test_handler_execution()) 