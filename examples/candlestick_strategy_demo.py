#!/usr/bin/env python3
"""
Candlestick Strategy Demo - Task 8.9

Comprehensive demonstration of the candlestick strategy system showing:
- Agent initialization and configuration
- Real-time pattern recognition
- Signal generation and monitoring
- Performance tracking
- Integration with multi-agent framework

This demo can run in two modes:
1. Basic Mode: Demonstrates core functionality with mock data
2. Full Mode: Shows complete integration with message bus and real-time processing
"""

import asyncio
import logging
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bistoury.agents.candlestick_strategy_agent import (
    CandlestickStrategyAgent,
    CandlestickStrategyConfig,
    StrategyPerformanceMetrics
)
from bistoury.agents.base import AgentState, AgentType
from bistoury.agents.messaging import Message, MessageType, MessagePriority, MessageBus
from bistoury.models.signals import CandlestickData, SignalDirection, SignalType
from bistoury.strategies.candlestick_models import PatternStrength
from bistoury.strategies.narrative_generator import NarrativeStyle


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('candlestick_strategy_demo.log')
    ]
)
logger = logging.getLogger(__name__)


class DemoDataGenerator:
    """Generate realistic demo data for candlestick strategy testing."""
    
    @staticmethod
    def generate_trending_market(
        symbol: str = "BTC",
        timeframe: str = "1m",
        count: int = 50,
        trend: str = "bullish"
    ) -> List[CandlestickData]:
        """Generate trending market data with realistic patterns."""
        
        candles = []
        base_price = Decimal("50000")
        current_price = base_price
        timestamp = datetime.now(timezone.utc) - timedelta(minutes=count)
        
        # Trend parameters
        trend_strength = Decimal("0.001") if trend == "bullish" else Decimal("-0.001")
        volatility = Decimal("0.008")
        
        for i in range(count):
            # Add trend component
            price_change = trend_strength + (
                Decimal(str((hash(f"{symbol}_{i}") % 1000 - 500) / 100000)) * volatility
            )
            
            open_price = current_price
            close_price = current_price + (current_price * price_change)
            
            # Realistic high/low
            if close_price > open_price:  # Bullish candle
                high_price = close_price + (current_price * Decimal("0.002"))
                low_price = open_price - (current_price * Decimal("0.001"))
            else:  # Bearish candle
                high_price = open_price + (current_price * Decimal("0.001"))
                low_price = close_price - (current_price * Decimal("0.002"))
            
            # Volume with some correlation to price movement
            volume_base = 100 + abs(hash(f"{symbol}_vol_{i}") % 50)
            volume_multiplier = 1.5 if abs(price_change) > volatility * Decimal("0.5") else 1.0
            volume = Decimal(str(volume_base * volume_multiplier))
            
            candle = CandlestickData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp + timedelta(minutes=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            candles.append(candle)
            current_price = close_price
            
        return candles
    
    @staticmethod
    def generate_reversal_setup(
        symbol: str = "BTC",
        timeframe: str = "15m",
        pattern_type: str = "hammer"
    ) -> List[CandlestickData]:
        """Generate data showing a reversal pattern setup."""
        
        candles = []
        base_price = Decimal("49000")  # Start lower for reversal
        timestamp = datetime.now(timezone.utc) - timedelta(minutes=20)
        
        # Generate downtrend leading to reversal
        for i in range(18):
            open_price = base_price + Decimal(str(i * -50))
            close_price = open_price - Decimal(str(100 + i * 20))
            high_price = open_price + Decimal("50")
            low_price = close_price - Decimal("30")
            
            candle = CandlestickData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp + timedelta(minutes=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=Decimal("120")
            )
            candles.append(candle)
        
        # Add reversal pattern
        reversal_price = candles[-1].close
        
        if pattern_type == "hammer":
            # Hammer pattern: long lower shadow, small body, little upper shadow
            open_price = reversal_price
            close_price = reversal_price + Decimal("100")  # Small bullish body
            low_price = reversal_price - Decimal("400")    # Long lower shadow
            high_price = close_price + Decimal("50")       # Small upper shadow
        
        elif pattern_type == "doji":
            # Doji pattern: open ≈ close, long shadows
            open_price = reversal_price
            close_price = reversal_price + Decimal("20")   # Very small body
            low_price = reversal_price - Decimal("200")    # Lower shadow
            high_price = reversal_price + Decimal("200")   # Upper shadow
        
        else:
            # Default to hammer
            open_price = reversal_price
            close_price = reversal_price + Decimal("100")
            low_price = reversal_price - Decimal("400")
            high_price = close_price + Decimal("50")
        
        reversal_candle = CandlestickData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp + timedelta(minutes=18),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("200")  # Higher volume for reversal
        )
        candles.append(reversal_candle)
        
        return candles


class CandlestickStrategyDemo:
    """Main demo class for candlestick strategy system."""
    
    def __init__(self, mode: str = "basic"):
        """Initialize the demo.
        
        Args:
            mode: "basic" for simple demo, "full" for complete integration
        """
        self.mode = mode
        self.agent: Optional[CandlestickStrategyAgent] = None
        self.message_bus: Optional[MessageBus] = None
        self.data_generator = DemoDataGenerator()
        
    async def setup(self):
        """Set up the demo environment."""
        
        logger.info(f"Setting up Candlestick Strategy Demo in {self.mode} mode...")
        
        # Create strategy configuration
        config = CandlestickStrategyConfig(
            symbols=["BTC", "ETH"],
            timeframes=["1m", "5m", "15m"],
            min_confidence_threshold=0.65,
            min_pattern_strength=PatternStrength.MODERATE,
            signal_expiry_minutes=10,
            max_signals_per_symbol=3,
            narrative_style=NarrativeStyle.TECHNICAL,
            max_data_buffer_size=100,
            data_cleanup_interval_seconds=60,
            health_check_interval_seconds=30,
            agent_name="candlestick_demo_agent",
            agent_version="1.0.0-demo"
        )
        
        # Create strategy agent
        self.agent = CandlestickStrategyAgent(config)
        
        if self.mode == "full":
            # Set up message bus for full integration
            self.message_bus = MessageBus()
            await self.message_bus.start()
            
            # Connect agent to message bus (in a real system)
            # For demo, we'll mock the message bus methods
            self.agent.subscribe_to_topic = self._mock_subscribe
            self.agent.unsubscribe_from_topic = self._mock_unsubscribe
            self.agent.publish_message = self._mock_publish_message
        else:
            # Basic mode - mock message bus methods
            self.agent.subscribe_to_topic = self._mock_subscribe
            self.agent.unsubscribe_from_topic = self._mock_unsubscribe
            self.agent.publish_message = self._mock_publish_message
        
        logger.info("Demo setup complete!")
    
    async def _mock_subscribe(self, topic: str):
        """Mock topic subscription."""
        logger.debug(f"Subscribed to topic: {topic}")
    
    async def _mock_unsubscribe(self, topic: str):
        """Mock topic unsubscription."""
        logger.debug(f"Unsubscribed from topic: {topic}")
    
    async def _mock_publish_message(self, **kwargs):
        """Mock message publishing with logging."""
        message_type = kwargs.get('message_type', 'UNKNOWN')
        topic = kwargs.get('topic', 'unknown')
        payload = kwargs.get('payload', {})
        
        if message_type == MessageType.SIGNAL_GENERATED:
            signal_data = payload.get('signal_data', {})
            logger.info(f"🚨 SIGNAL GENERATED: {signal_data.get('direction', 'UNKNOWN')} "
                       f"{payload.get('symbol', 'UNKNOWN')} "
                       f"(Confidence: {payload.get('confidence', 0):.1f}%)")
            
            # Log narrative if available
            narrative = payload.get('narrative', {})
            if narrative.get('executive_summary'):
                logger.info(f"📊 Analysis: {narrative['executive_summary']}")
        
        elif message_type == MessageType.AGENT_HEALTH_UPDATE:
            health_data = payload.get('performance_metrics', {})
            logger.info(f"💓 Health Update: {health_data.get('signals_generated', 0)} signals, "
                       f"{health_data.get('processing_latency_ms', 0):.1f}ms latency")
    
    async def run_basic_demo(self):
        """Run basic demonstration showing core functionality."""
        
        logger.info("=" * 60)
        logger.info("BASIC DEMO: Core Candlestick Strategy Functionality")
        logger.info("=" * 60)
        
        # Start the agent
        logger.info("1. Starting Candlestick Strategy Agent...")
        await self.agent.start()
        
        assert self.agent.state == AgentState.RUNNING
        logger.info(f"✅ Agent started successfully (ID: {self.agent.agent_id})")
        
        # Show agent capabilities
        logger.info(f"🎯 Agent Capabilities:")
        for capability in self.agent.capabilities:
            logger.info(f"   - {capability.name}: {capability.description}")
        
        # Demonstrate data processing
        logger.info("\n2. Processing Market Data...")
        
        # Generate realistic market data
        timeframes = ["1m", "5m", "15m"]
        for timeframe in timeframes:
            logger.info(f"   📈 Generating {timeframe} data...")
            candles = self.data_generator.generate_trending_market("BTC", timeframe, 25, "bullish")
            
            # Send data to agent
            for i, candle in enumerate(candles):
                message = self._create_market_data_message("BTC", timeframe, candle)
                await self.agent.handle_message(message)
                
                if i % 10 == 0:
                    logger.info(f"      Processed {i+1}/{len(candles)} candles...")
            
            logger.info(f"   ✅ {timeframe} data processing complete")
        
        # Check data buffer
        logger.info(f"\n📊 Data Buffer Status:")
        for symbol, timeframes in self.agent.market_data_buffer.items():
            for tf, candles in timeframes.items():
                logger.info(f"   {symbol} {tf}: {len(candles)} candles")
        
        # Show performance metrics
        await self._show_performance_metrics()
        
        # Demonstrate pattern recognition
        logger.info("\n3. Testing Pattern Recognition...")
        
        # Generate data with reversal pattern
        reversal_data = self.data_generator.generate_reversal_setup("BTC", "15m", "hammer")
        logger.info("   🔨 Injecting Hammer pattern...")
        
        for candle in reversal_data[-5:]:  # Send last 5 candles including hammer
            message = self._create_market_data_message("BTC", "15m", candle)
            await self.agent.handle_message(message)
            await asyncio.sleep(0.1)  # Small delay for processing
        
        # Wait for potential signal generation
        await asyncio.sleep(0.5)
        
        # Show active signals
        logger.info(f"\n🔍 Active Signals:")
        for symbol, signals in self.agent.active_signals.items():
            if signals:
                logger.info(f"   {symbol}: {len(signals)} active signals")
                for signal in signals:
                    logger.info(f"      - {signal.direction.value} @ {signal.price} "
                               f"(Confidence: {signal.confidence}%)")
            else:
                logger.info(f"   {symbol}: No active signals")
        
        # Test configuration update
        logger.info("\n4. Testing Configuration Hot-Reload...")
        config_message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM_CONFIG_UPDATE,
            sender="demo_manager",
            timestamp=datetime.now(timezone.utc),
            payload={
                "config_updates": {
                    "min_confidence_threshold": 0.70,
                    "max_signals_per_symbol": 5
                }
            },
            priority=MessagePriority.HIGH,
            topic="config.strategy.candlestick"
        )
        
        await self.agent.handle_message(config_message)
        logger.info(f"   ✅ Configuration updated: threshold={self.agent.config.min_confidence_threshold}")
        
        # Stop the agent
        logger.info("\n5. Stopping Agent...")
        await self.agent.stop()
        logger.info("   ✅ Agent stopped successfully")
        
        logger.info("\n" + "=" * 60)
        logger.info("BASIC DEMO COMPLETE")
        logger.info("=" * 60)
    
    async def run_full_demo(self):
        """Run full demonstration with complete integration."""
        
        logger.info("=" * 60)
        logger.info("FULL DEMO: Complete Strategy Integration")
        logger.info("=" * 60)
        
        # Start the agent
        logger.info("1. Starting Strategy Agent with Message Bus...")
        await self.agent.start()
        logger.info(f"✅ Agent started (ID: {self.agent.agent_id})")
        
        # Simulate real-time data streaming
        logger.info("\n2. Simulating Real-Time Data Stream...")
        
        async def data_stream_task():
            """Simulate continuous data streaming."""
            symbols = ["BTC", "ETH"]
            timeframes = ["1m", "5m", "15m"]
            
            for round_num in range(5):
                logger.info(f"   📡 Data Round {round_num + 1}/5...")
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        # Generate new candle
                        candles = self.data_generator.generate_trending_market(
                            symbol, timeframe, 1, "bullish" if round_num % 2 == 0 else "bearish"
                        )
                        
                        message = self._create_market_data_message(symbol, timeframe, candles[0])
                        await self.agent.handle_message(message)
                        
                        await asyncio.sleep(0.1)  # Realistic delay
                
                await asyncio.sleep(1.0)  # Pause between rounds
        
        # Performance monitoring task
        async def monitoring_task():
            """Monitor agent performance."""
            for i in range(10):
                await asyncio.sleep(2.0)
                
                health = await self.agent._health_check()
                logger.info(f"   💓 Health Check {i+1}: Score={health.health_score:.2f}, "
                           f"Messages={health.messages_processed}")
        
        # Run concurrent tasks
        logger.info("   🚀 Starting concurrent data streaming and monitoring...")
        await asyncio.gather(
            data_stream_task(),
            monitoring_task()
        )
        
        # Show comprehensive statistics
        await self._show_comprehensive_statistics()
        
        # Test error handling
        logger.info("\n3. Testing Error Handling...")
        
        # Send invalid data
        invalid_message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.DATA_MARKET_UPDATE,
            sender="demo_sender",
            timestamp=datetime.now(timezone.utc),
            payload={
                "data_type": "candle",
                "symbol": "BTC",
                "timeframe": "1m",
                "candle_data": {"invalid": "data"}
            },
            priority=MessagePriority.NORMAL,
            topic="market_data.candle.BTC.1m"
        )
        
        initial_errors = self.agent.performance_metrics.errors_count
        await self.agent.handle_message(invalid_message)
        
        if self.agent.performance_metrics.errors_count > initial_errors:
            logger.info("   ✅ Error handling working correctly")
        
        # Performance benchmarking
        logger.info("\n4. Performance Benchmarking...")
        await self._run_performance_benchmark()
        
        # Stop the agent
        logger.info("\n5. Graceful Shutdown...")
        await self.agent.stop()
        
        if self.message_bus:
            await self.message_bus.stop()
        
        logger.info("   ✅ All components stopped successfully")
        
        logger.info("\n" + "=" * 60)
        logger.info("FULL DEMO COMPLETE")
        logger.info("=" * 60)
    
    async def _show_performance_metrics(self):
        """Display current performance metrics."""
        
        metrics = self.agent.performance_metrics
        logger.info(f"\n📈 Performance Metrics:")
        logger.info(f"   Messages Received: {metrics.data_messages_received}")
        logger.info(f"   Signals Generated: {metrics.signals_generated}")
        logger.info(f"   Patterns Detected: {metrics.patterns_detected}")
        logger.info(f"   High Confidence Signals: {metrics.high_confidence_signals}")
        logger.info(f"   Processing Latency: {metrics.processing_latency_ms:.2f}ms")
        logger.info(f"   Error Count: {metrics.errors_count}")
        logger.info(f"   Average Confidence: {metrics.average_confidence:.1f}%")
    
    async def _show_comprehensive_statistics(self):
        """Show comprehensive agent statistics."""
        
        stats = self.agent.get_strategy_statistics()
        
        logger.info(f"\n📊 Comprehensive Statistics:")
        logger.info(f"   Agent Info:")
        logger.info(f"      ID: {stats['agent_info']['agent_id']}")
        logger.info(f"      Type: {stats['agent_info']['agent_type']}")
        logger.info(f"      Status: {stats['agent_info']['status']}")
        logger.info(f"      Version: {stats['agent_info']['version']}")
        
        logger.info(f"   Performance:")
        for key, value in stats['performance'].items():
            logger.info(f"      {key}: {value}")
        
        logger.info(f"   Configuration:")
        for key, value in stats['configuration'].items():
            logger.info(f"      {key}: {value}")
        
        logger.info(f"   Active Data:")
        for key, value in stats['active_data'].items():
            logger.info(f"      {key}: {value}")
    
    async def _run_performance_benchmark(self):
        """Run performance benchmarking tests."""
        
        logger.info("   🏃 Running latency benchmark...")
        
        # Generate test data
        candles = self.data_generator.generate_trending_market("BTC", "1m", 100)
        
        # Measure processing time
        start_time = time.time()
        
        for candle in candles:
            message = self._create_market_data_message("BTC", "1m", candle)
            await self.agent.handle_message(message)
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_latency_ms = total_time_ms / len(candles)
        
        logger.info(f"   📏 Benchmark Results:")
        logger.info(f"      Total Time: {total_time_ms:.2f}ms")
        logger.info(f"      Average Latency: {avg_latency_ms:.2f}ms per message")
        logger.info(f"      Throughput: {len(candles) / (total_time_ms / 1000):.1f} messages/sec")
        
        # Verify performance targets
        if avg_latency_ms < 50.0:
            logger.info("   ✅ Latency target met (< 50ms)")
        else:
            logger.warning(f"   ⚠️  Latency target missed ({avg_latency_ms:.2f}ms)")
    
    def _create_market_data_message(
        self, 
        symbol: str, 
        timeframe: str, 
        candle_data: CandlestickData
    ) -> Message:
        """Create a market data message."""
        
        return Message(
            id=str(uuid.uuid4()),
            type=MessageType.DATA_MARKET_UPDATE,
            sender="demo_collector",
            timestamp=datetime.now(timezone.utc),
            payload={
                "data_type": "candle",
                "symbol": symbol,
                "timeframe": timeframe,
                "candle_data": candle_data.model_dump()
            },
            priority=MessagePriority.NORMAL,
            topic=f"market_data.candle.{symbol}.{timeframe}"
        )
    
    async def cleanup(self):
        """Clean up demo resources."""
        
        if self.agent and self.agent.state == AgentState.RUNNING:
            await self.agent.stop()
        
        if self.message_bus:
            await self.message_bus.stop()


async def main():
    """Main demo function."""
    
    print("\n🎯 Candlestick Strategy Demo - Task 8.9")
    print("=" * 50)
    
    # Parse command line arguments
    mode = "basic"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode not in ["basic", "full"]:
        print("Usage: python candlestick_strategy_demo.py [basic|full]")
        print("  basic: Basic functionality demo (default)")
        print("  full:  Complete integration demo")
        return
    
    demo = CandlestickStrategyDemo(mode)
    
    try:
        # Set up demo
        await demo.setup()
        
        # Run appropriate demo
        if mode == "basic":
            await demo.run_basic_demo()
        else:
            await demo.run_full_demo()
        
        print("\n🎉 Demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        raise
    finally:
        # Clean up
        await demo.cleanup()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())