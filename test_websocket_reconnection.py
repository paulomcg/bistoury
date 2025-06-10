#!/usr/bin/env python3
"""
Test WebSocket Reconnection Functionality

This script demonstrates the automatic reconnection feature for the HyperLiquid WebSocket connection.
It simulates connection drops and shows how the system automatically reconnects.
"""

import asyncio
import logging
import sys
from datetime import datetime
import signal

sys.path.append('.')

from src.bistoury.hyperliquid.client import HyperLiquidIntegration
from src.bistoury.config import Config

# Set up detailed logging to see reconnection events
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class ReconnectionTest:
    def __init__(self):
        self.client = None
        self.test_running = True
        self.message_count = 0
        
    async def message_handler(self, message):
        """Handle incoming WebSocket messages."""
        self.message_count += 1
        if self.message_count % 10 == 0:  # Log every 10th message
            logger.info(f"📊 Received {self.message_count} messages (latest: {datetime.now().strftime('%H:%M:%S')})")
    
    async def run_test(self):
        """Run the reconnection test."""
        logger.info("🚀 Starting HyperLiquid WebSocket Reconnection Test")
        logger.info("=" * 60)
        logger.info("This test will:")
        logger.info("  1. Connect to HyperLiquid WebSocket")
        logger.info("  2. Subscribe to price feeds")
        logger.info("  3. Show automatic reconnection when connection drops")
        logger.info("  4. Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        try:
            # Initialize client
            config = Config.load_from_env()
            self.client = HyperLiquidIntegration(config)
            
            # Connect to WebSocket
            logger.info("🔌 Connecting to HyperLiquid WebSocket...")
            connected = await self.client.connect()
            
            if not connected:
                logger.error("❌ Failed to connect to WebSocket")
                return
            
            logger.info("✅ Connected to HyperLiquid WebSocket")
            
            # Subscribe to price feeds
            logger.info("📡 Subscribing to price feeds...")
            await self.client.subscribe_all_mids(self.message_handler)
            logger.info("✅ Subscribed to all mid prices")
            
            # Show initial connection stats
            stats = self.client.get_reconnection_stats()
            logger.info(f"🔧 Reconnection enabled: {stats['auto_reconnect_enabled']}")
            logger.info(f"🔗 Currently connected: {stats['currently_connected']}")
            logger.info(f"👁️ Connection monitoring: {stats['connection_monitor_active']}")
            logger.info(f"⏱️ Message timeout: {stats['message_timeout_seconds']}s")
            
            # Set shorter timeout for testing (default is 30s)
            self.client.set_message_timeout(10.0)
            logger.info("⏱️ Set message timeout to 10s for testing")
            
            logger.info("")
            logger.info("📊 Monitoring WebSocket messages...")
            logger.info("💡 To test reconnection: disconnect your network for 10+ seconds")
            logger.info("   You should see reconnection attempts (without spam messages)")
            logger.info("   Only successful and stable reconnections will be reported")
            logger.info("")
            
            # Monitor messages and connection
            while self.test_running:
                await asyncio.sleep(5)
                
                # Check connection status
                if self.client.is_connected():
                    logger.debug(f"✅ Connection healthy, {self.message_count} messages received")
                else:
                    logger.warning("⚠️ Connection appears to be down")
                
                # Show reconnection stats periodically
                if self.message_count > 0 and self.message_count % 50 == 0:
                    stats = self.client.get_reconnection_stats()
                    logger.info(f"📈 Stats: Connected={stats['currently_connected']}, "
                              f"Messages={self.message_count}, "
                              f"Monitoring={stats['connection_monitor_active']}, "
                              f"Last_msg={stats['last_message_time'][-8:] if stats['last_message_time'] else 'None'}")
                
                # Show connection health every 30 seconds
                if self.message_count > 0 and self.message_count % 30 == 0:
                    stats = self.client.get_reconnection_stats()
                    time_since_last = "Never"
                    if stats['last_message_time']:
                        from datetime import datetime
                        last_time = datetime.fromisoformat(stats['last_message_time'].replace('Z', '+00:00'))
                        time_since_last = f"{(datetime.now().replace(tzinfo=last_time.tzinfo) - last_time).total_seconds():.1f}s ago"
                    
                    logger.info(f"🔍 Health Check: Messages={self.message_count}, Last_message={time_since_last}")
                
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("🛑 Stopping WebSocket test...")
        
        if self.client:
            try:
                # Disable auto-reconnect for clean shutdown
                self.client.disable_auto_reconnect()
                
                # Disconnect
                await self.client.disconnect(disable_auto_reconnect=True)
                logger.info("✅ WebSocket disconnected cleanly")
                
            except Exception as e:
                logger.warning(f"⚠️ Error during cleanup: {e}")
        
        logger.info(f"📊 Test completed. Total messages received: {self.message_count}")
    
    def stop_test(self):
        """Stop the test (called by signal handler)."""
        self.test_running = False

async def main():
    """Main test function."""
    test = ReconnectionTest()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("🛑 Received interrupt signal, stopping test...")
        test.stop_test()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await test.run_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc() 