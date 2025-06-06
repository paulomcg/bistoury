#!/usr/bin/env python3
"""
Collector Agent Demo Script

This script demonstrates the CollectorAgent integration with the BaseAgent framework.
It shows how to:
- Initialize a CollectorAgent with custom configuration
- Start and stop the agent
- Monitor health and statistics
- Integrate with the messaging system
- Handle lifecycle events
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bistoury.agents.collector_agent import CollectorAgent, CollectorAgentConfig
from bistoury.agents.messaging import MessageBus
from bistoury.hyperliquid.client import HyperLiquidIntegration
from bistoury.database import DatabaseManager
from bistoury.config import Config
from bistoury.logger import get_logger
from bistoury.models.agent_messages import MessageType, MessageFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class DemoMessageHandler:
    """Demo message handler to show message bus integration."""
    
    def __init__(self, name: str):
        self.name = name
        self.messages_received = []
    
    async def handle_message(self, message):
        """Handle incoming messages."""
        self.messages_received.append(message)
        logger.info(f"{self.name} received message: {message.type} from {message.sender}")
        
        # Log specific message types
        if message.type == MessageType.AGENT_STARTED:
            logger.info(f"  Agent started: {message.payload.description}")
        elif message.type == MessageType.DATA_MARKET_UPDATE:
            data = message.payload.data
            logger.info(f"  Market data: {data.get('candles_collected', 0)} candles, "
                       f"{data.get('trades_collected', 0)} trades")
        elif message.type == MessageType.AGENT_HEALTH_UPDATE:
            metadata = message.payload.metadata
            health_score = metadata.get('health_score', 0)
            logger.info(f"  Health update: score={health_score:.2f}, "
                       f"uptime={metadata.get('uptime_seconds', 0):.1f}s")


async def demo_collector_agent():
    """Demonstrate CollectorAgent functionality."""
    
    logger.info("🚀 Starting CollectorAgent Demo")
    
    # 1. Load configuration
    try:
        config = Config()
        logger.info("✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        logger.info("ℹ️  Using demo configuration instead")
        config = None
    
    # 2. Initialize dependencies
    try:
        # Database manager
        db_manager = DatabaseManager()
        logger.info("✅ Database manager initialized")
        
        # HyperLiquid integration (demo mode - may not connect)
        hyperliquid = HyperLiquidIntegration(testnet=True)
        logger.info("✅ HyperLiquid integration initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize dependencies: {e}")
        return
    
    # 3. Initialize message bus
    message_bus = MessageBus()
    await message_bus.start()
    logger.info("✅ Message bus started")
    
    # 4. Set up message handlers
    data_handler = DemoMessageHandler("DataHandler")
    health_handler = DemoMessageHandler("HealthHandler")
    
    # Subscribe to different message types
    await message_bus.subscribe(
        subscriber="demo_data_handler",
        message_filter=MessageFilter(
            message_types=[MessageType.DATA_MARKET_UPDATE],
            topics=["data.collection"]
        ),
        handler=data_handler.handle_message
    )
    
    await message_bus.subscribe(
        subscriber="demo_health_handler", 
        message_filter=MessageFilter(
            message_types=[MessageType.AGENT_HEALTH_UPDATE, MessageType.AGENT_STARTED, MessageType.AGENT_STOPPED],
            topics=["agent.health", "agent.lifecycle"]
        ),
        handler=health_handler.handle_message
    )
    
    logger.info("✅ Message handlers subscribed")
    
    # 5. Configure the collector agent
    collector_config = {
        'collector': {
            'symbols': ['BTC', 'ETH', 'SOL'],
            'intervals': ['1m', '5m'],
            'buffer_size': 100,
            'flush_interval': 10.0,
            'stats_interval': 15.0,
            'health_check_interval': 5.0,
            'collect_historical_on_start': False,
            'publish_data_updates': True,
            'publish_stats_updates': True,
            'data_update_interval': 3.0
        }
    }
    
    # 6. Initialize the collector agent
    try:
        collector_agent = CollectorAgent(
            hyperliquid=hyperliquid,
            db_manager=db_manager,
            config=collector_config,
            name="demo_collector",
            persist_state=False
        )
        
        # Connect to message bus
        collector_agent.set_message_bus(message_bus)
        
        logger.info("✅ CollectorAgent initialized")
        logger.info(f"   Agent ID: {collector_agent.agent_id}")
        logger.info(f"   Agent Type: {collector_agent.agent_type.value}")
        logger.info(f"   Symbols: {list(collector_agent.collector_config.symbols)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize CollectorAgent: {e}")
        await message_bus.stop()
        return
    
    # 7. Demonstrate agent lifecycle
    try:
        logger.info("\n📊 Agent Configuration:")
        config_info = collector_agent.get_configuration()
        for key, value in config_info['collector'].items():
            logger.info(f"   {key}: {value}")
        
        logger.info("\n🔄 Starting CollectorAgent...")
        start_success = await collector_agent.start()
        
        if start_success:
            logger.info("✅ CollectorAgent started successfully")
            
            # Monitor for a short period
            logger.info("\n👀 Monitoring agent for 20 seconds...")
            
            for i in range(4):
                await asyncio.sleep(5)
                
                # Get health status
                health = await collector_agent.get_health()
                logger.info(f"🏥 Health Check {i+1}: score={health.health_score:.2f}, "
                           f"state={health.state.value}, uptime={health.uptime_seconds:.1f}s")
                
                # Get statistics
                stats = collector_agent.get_collection_stats()
                logger.info(f"📈 Stats: {stats.get('candles_collected', 0)} candles, "
                           f"{stats.get('trades_collected', 0)} trades, "
                           f"{stats.get('errors', 0)} errors")
            
            # 8. Demonstrate configuration updates
            logger.info("\n⚙️  Testing configuration update...")
            collector_agent.update_configuration({
                'collector': {
                    'buffer_size': 200,
                    'stats_interval': 30.0
                }
            })
            logger.info("✅ Configuration updated")
            
            # 9. Demonstrate symbol management
            logger.info("\n🔧 Testing symbol management...")
            collector_agent.add_symbol("DOGE")
            logger.info(f"   Added DOGE. Symbols: {list(collector_agent.collector_config.symbols)}")
            
            collector_agent.remove_symbol("SOL")
            logger.info(f"   Removed SOL. Symbols: {list(collector_agent.collector_config.symbols)}")
            
            # 10. Stop the agent
            logger.info("\n🛑 Stopping CollectorAgent...")
            await collector_agent.stop()
            logger.info("✅ CollectorAgent stopped successfully")
            
        else:
            logger.error("❌ Failed to start CollectorAgent")
    
    except Exception as e:
        logger.error(f"❌ Error during agent lifecycle demo: {e}")
        
        # Ensure agent is stopped
        try:
            if collector_agent.is_running:
                await collector_agent.stop()
        except:
            pass
    
    # 11. Display message statistics
    logger.info("\n📨 Message Statistics:")
    logger.info(f"   Data messages received: {len(data_handler.messages_received)}")
    logger.info(f"   Health messages received: {len(health_handler.messages_received)}")
    
    # Stop message bus
    await message_bus.stop()
    logger.info("✅ Message bus stopped")
    
    logger.info("\n🎉 CollectorAgent Demo Complete!")


async def demo_basic_functionality():
    """Demonstrate basic CollectorAgent functionality without external dependencies."""
    
    logger.info("🚀 Starting Basic CollectorAgent Demo (Mock Mode)")
    
    # Mock dependencies for demonstration
    class MockHyperLiquid:
        def is_connected(self): return True
        async def connect(self): return True
    
    class MockDatabase:
        def __init__(self): self.connected = True
    
    # Initialize with mocks
    hyperliquid = MockHyperLiquid()
    db_manager = MockDatabase()
    
    # Configure the agent
    config = {
        'collector': {
            'symbols': ['BTC', 'ETH'],
            'intervals': ['1m', '5m'],
            'buffer_size': 50,
            'collect_historical_on_start': False,
            'publish_data_updates': False,  # Disable for basic demo
            'publish_stats_updates': False
        }
    }
    
    # Initialize agent
    agent = CollectorAgent(
        hyperliquid=hyperliquid,
        db_manager=db_manager,
        config=config,
        name="basic_demo_collector",
        persist_state=False
    )
    
    logger.info("✅ Basic CollectorAgent created")
    logger.info(f"   Name: {agent.name}")
    logger.info(f"   Type: {agent.agent_type.value}")
    logger.info(f"   State: {agent.state.value}")
    logger.info(f"   Symbols: {list(agent.collector_config.symbols)}")
    
    # Test configuration
    logger.info("\n⚙️  Testing configuration methods...")
    original_config = agent.get_configuration()
    logger.info(f"   Buffer size: {original_config['collector']['buffer_size']}")
    
    agent.update_configuration({'collector': {'buffer_size': 100}})
    updated_config = agent.get_configuration()
    logger.info(f"   Updated buffer size: {updated_config['collector']['buffer_size']}")
    
    # Test symbol management
    logger.info("\n🔧 Testing symbol management...")
    agent.add_symbol("SOL")
    logger.info(f"   Added SOL: {list(agent.collector_config.symbols)}")
    
    agent.remove_symbol("ETH")
    logger.info(f"   Removed ETH: {list(agent.collector_config.symbols)}")
    
    logger.info("\n🎉 Basic Demo Complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CollectorAgent Demo Script")
    parser.add_argument("--basic", action="store_true", 
                       help="Run basic demo with mocked dependencies")
    args = parser.parse_args()
    
    if args.basic:
        asyncio.run(demo_basic_functionality())
    else:
        asyncio.run(demo_collector_agent()) 