#!/usr/bin/env python3
"""
Test script for database switcher functionality.
Demonstrates switching between production and test databases with compatibility layer.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bistoury.database import (
    get_database_switcher, 
    switch_database, 
    get_compatible_query,
    DatabaseSwitcher,
    CompatibleDataQuery
)


def test_database_switcher():
    """Test database switching functionality."""
    print("🧪 Testing Database Switcher Functionality")
    print("=" * 50)
    
    switcher = get_database_switcher()
    
    # List available databases
    print("📋 Available Databases:")
    databases = switcher.list_available_databases()
    
    for name, info in databases.items():
        status = "✅" if info['exists'] else "❌"
        schema_type = info.get('schema_type', 'unknown')
        tables = info.get('tables', 'N/A')
        
        print(f"   {status} {name}:")
        print(f"      Path: {info['path']}")
        print(f"      Size: {info['size']}")
        print(f"      Schema: {schema_type}")
        print(f"      Tables: {tables}")
        
        if 'error' in info:
            print(f"      Error: {info['error']}")
        print()
    
    return databases


def test_production_database():
    """Test production database functionality."""
    print("🏭 Testing Production Database")
    print("=" * 30)
    
    try:
        # Switch to production database
        db_manager = switch_database('production')
        query = get_compatible_query(db_manager, 'production')
        
        print("✅ Connected to production database")
        
        # Test symbols
        symbols = query.get_symbols()
        print(f"📊 Found {len(symbols)} symbols")
        if symbols:
            for symbol in symbols[:3]:  # Show first 3
                print(f"   • {symbol['symbol']}: {symbol['max_leverage']}x leverage")
        
        # Test trades if we have symbols
        if symbols:
            symbol = symbols[0]['symbol']
            trades = query.get_latest_trades(symbol, limit=5)
            print(f"\n💹 Latest {len(trades)} trades for {symbol}:")
            for trade in trades[:3]:  # Show first 3
                side = "SELL" if trade['side'] == 'A' else "BUY"
                print(f"   • {trade['timestamp']}: {side} {trade['size']} @ {trade['price']}")
        
        # Test orderbook
        if symbols:
            symbol = symbols[0]['symbol']
            orderbook = query.get_latest_orderbook(symbol)
            if orderbook:
                print(f"\n📚 Latest orderbook for {symbol}:")
                print(f"   • {len(orderbook['levels'][0])} bids, {len(orderbook['levels'][1])} asks")
            else:
                print(f"\n📚 No orderbook data for {symbol}")
        
        return True
        
    except Exception as e:
        print(f"❌ Production database test failed: {e}")
        return False


def test_test_database():
    """Test test database functionality with compatibility layer."""
    print("\n🧪 Testing Test Database (Legacy Schema)")
    print("=" * 40)
    
    try:
        # Switch to test database
        db_manager = switch_database('test')
        query = get_compatible_query(db_manager, 'test_legacy')
        
        print("✅ Connected to test database with compatibility layer")
        
        # Test symbols (extracted from trades)
        symbols = query.get_symbols()
        print(f"📊 Found {len(symbols)} symbols (extracted from trades)")
        if symbols:
            for symbol in symbols[:5]:  # Show first 5
                print(f"   • {symbol['symbol']}: {symbol['max_leverage']}x leverage")
        
        # Test trades if we have symbols
        if symbols:
            symbol = symbols[0]['symbol']
            trades = query.get_latest_trades(symbol, limit=5)
            print(f"\n💹 Latest {len(trades)} trades for {symbol}:")
            for trade in trades[:3]:  # Show first 3
                side = "SELL" if trade['side'] == 'A' else "BUY"
                print(f"   • {trade['timestamp']}: {side} {trade['size']} @ {trade['price']}")
        
        # Test orderbook
        if symbols:
            symbol = symbols[0]['symbol']
            orderbook = query.get_latest_orderbook(symbol)
            if orderbook:
                print(f"\n📚 Latest orderbook for {symbol}:")
                bids = len(orderbook['levels'][0]) if orderbook['levels'] and len(orderbook['levels']) > 0 else 0
                asks = len(orderbook['levels'][1]) if orderbook['levels'] and len(orderbook['levels']) > 1 else 0
                print(f"   • {bids} bids, {asks} asks")
                print(f"   • Time: {orderbook['time']}")
            else:
                print(f"\n📚 No orderbook data for {symbol}")
        
        # Test candles (should warn about unavailability)
        if symbols:
            symbol = symbols[0]['symbol']
            candles = query.get_candles('1m', symbol, limit=5)
            print(f"\n📈 Candles for {symbol}: {len(candles)} (expected 0 for test_legacy)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_switching():
    """Test switching between databases."""
    print("\n🔄 Testing Database Switching")
    print("=" * 30)
    
    try:
        switcher = get_database_switcher()
        
        # Switch to production
        print("Switching to production...")
        db_manager = switcher.switch_to_database('production')
        current_db = switcher.get_current_database()
        print(f"✅ Current database: {current_db}")
        
        # Switch to test
        print("Switching to test...")
        db_manager = switcher.switch_to_database('test')
        current_db = switcher.get_current_database()
        print(f"✅ Current database: {current_db}")
        
        # Switch back to production
        print("Switching back to production...")
        db_manager = switcher.switch_to_database('production')
        current_db = switcher.get_current_database()
        print(f"✅ Current database: {current_db}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database switching test failed: {e}")
        return False


def get_test_database_stats():
    """Get detailed stats about the test database."""
    print("\n📊 Test Database Statistics")
    print("=" * 30)
    
    try:
        # Switch to test database
        db_manager = switch_database('test')
        
        # Get raw stats using compatibility layer
        from bistoury.database.database_switcher import TestDatabaseCompatibilityLayer
        compat_layer = TestDatabaseCompatibilityLayer(db_manager)
        
        stats = compat_layer.get_raw_data_stats()
        
        if 'date_range' in stats:
            print(f"📅 Date Range:")
            print(f"   Start: {stats['date_range']['start']}")
            print(f"   End: {stats['date_range']['end']}")
        
        if 'channels' in stats:
            print(f"\n📡 Message Channels:")
            for channel, count in stats['channels'].items():
                print(f"   • {channel}: {count:,} messages")
        
        if 'symbols' in stats:
            print(f"\n🪙 Trading Symbols:")
            for symbol, count in list(stats['symbols'].items())[:10]:  # Top 10
                print(f"   • {symbol}: {count:,} trades")
        
        return True
        
    except Exception as e:
        print(f"❌ Test database stats failed: {e}")
        return False


def main():
    """Run all database switcher tests."""
    print("🧪 Bistoury Database Switcher Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test database listing
    print("Step 1: Listing available databases...")
    databases = test_database_switcher()
    results.append(len(databases) > 0)
    
    # Test production database if available
    if databases.get('production', {}).get('exists', False):
        print("\nStep 2: Testing production database...")
        results.append(test_production_database())
    else:
        print("\nStep 2: Skipping production database (not available)")
        results.append(True)  # Don't fail if production DB doesn't exist
    
    # Test test database if available
    if databases.get('test', {}).get('exists', False):
        print("\nStep 3: Testing test database...")
        results.append(test_test_database())
        
        print("\nStep 4: Getting test database statistics...")
        results.append(get_test_database_stats())
    else:
        print("\nStep 3: Skipping test database (not available)")
        results.append(False)  # This should exist for this test
    
    # Test database switching
    print("\nStep 5: Testing database switching...")
    results.append(test_database_switching())
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 20)
    
    if all(results):
        print("✅ All tests passed!")
        print("✅ Database switcher is working correctly")
        print("\n💡 You can now:")
        print("   • Switch between production and test databases")
        print("   • Use the compatibility layer for test database")
        print("   • Access 9.4GB of historical trading data for backtesting")
        print("   • Set up environment variables for database selection")
    else:
        print("⚠️ Some tests failed")
        print("   • Check the output above for specific issues")
        
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main()) 