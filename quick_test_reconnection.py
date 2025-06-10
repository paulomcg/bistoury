#!/usr/bin/env python3
"""Quick test of enhanced WebSocket reconnection system."""

import asyncio
import sys
sys.path.append('.')

from src.bistoury.hyperliquid.client import HyperLiquidIntegration
from src.bistoury.config import Config

async def main():
    print('🚀 Testing enhanced WebSocket reconnection system...')
    
    config = Config.load_from_env()
    client = HyperLiquidIntegration(config)
    
    # Show monitoring capabilities
    stats = client.get_reconnection_stats()
    print(f'✅ Auto-reconnect: {stats["auto_reconnect_enabled"]}')
    print(f'✅ Message timeout: {stats["message_timeout_seconds"]}s')
    print(f'✅ Connection monitor: {stats["connection_monitor_active"]}')
    
    # Test timeout configuration
    client.set_message_timeout(10.0)
    stats = client.get_reconnection_stats()
    print(f'✅ Updated timeout: {stats["message_timeout_seconds"]}s')
    
    print()
    print('✅ Enhanced reconnection system ready!')
    print('💡 Key improvements:')
    print('   - Monitors message flow to detect silent failures')
    print('   - Automatically reconnects when no messages for 10s+')
    print('   - Restores all subscriptions after reconnection')
    print('   - Works even when WebSocket "appears" connected')
    print()
    print('🔧 To test: Run test_websocket_reconnection.py and disconnect network')

if __name__ == "__main__":
    asyncio.run(main()) 