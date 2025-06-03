#!/usr/bin/env python3
"""
Test script for database schema implementation.
Tests all schema operations with sample HyperLiquid data.
"""

import json
import sys
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bistoury.config import Config
from bistoury.database import DatabaseManager, MarketDataSchema, DataInsertion, DataQuery

def main():
    """Run all schema implementation tests."""
    try:
        print("🧪 Starting Comprehensive Database Schema Tests")
        print("=" * 55)
        
        global db_manager
        db_manager = DatabaseManager()
        
        # Run all tests
        test_schema_creation()
        test_symbol_operations()
        test_candle_operations()
        test_trade_operations()
        test_orderbook_operations()
        test_funding_rate_operations()
        test_bulk_operations()
        test_query_operations()
        test_enhanced_orderbook_operations()
        test_enhanced_funding_rate_operations()
        test_precision_handling()
        test_hyperliquid_data_compatibility()
        
        print("\n🎉 ALL TESTS PASSED! Database schema is production-ready.")
        print("=" * 55)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return 1
    finally:
        if db_manager:
            db_manager.close_all_connections()
            
    return 0

def test_schema_creation():
    """Test schema creation."""
    print("\n🧪 Testing Schema Creation...")
    
    schema = MarketDataSchema(db_manager)
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Create schema (drop existing first for clean test)
    schema.recreate_all_tables()
    print("✅ Schema recreated with enhanced HyperLiquid format")
    
    # Validate schema
    if schema.validate_schema():
        print("✅ Schema validation passed")
    else:
        raise Exception("Schema validation failed")

def test_symbol_operations():
    """Test symbol operations."""
    print("\n🧪 Testing Symbol Operations...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Insert symbols
    sample_symbols = [
        {
            'name': 'BTC',
            'szDecimals': 5,
            'maxLeverage': 40,
            'marginTableId': 56,
            'isDelisted': False
        },
        {
            'name': 'ETH',
            'szDecimals': 4,
            'maxLeverage': 25,
            'marginTableId': 45,
            'isDelisted': False
        },
        {
            'name': 'MATIC',
            'szDecimals': 1,
            'maxLeverage': 20,
            'marginTableId': 20,
            'isDelisted': True
        }
    ]
    
    for symbol in sample_symbols:
        symbol_id = insertion.insert_symbol(symbol)
        print(f"✅ Inserted symbol {symbol['name']} (ID: {symbol_id})")

def test_candle_operations():
    """Test candle operations."""
    print("\n🧪 Testing Candle Operations...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Insert candles
    sample_candles = [
        {
            't': 1748980440000,  # start timestamp
            'T': 1748980499999,  # end timestamp
            's': 'BTC',          # symbol
            'i': '1m',           # interval
            'o': '106115.0',     # open
            'c': '106075.0',     # close
            'h': '106115.0',     # high
            'l': '106075.0',     # low
            'v': '2.8626',       # volume
            'n': 133             # trade count
        },
        {
            't': 1748980500000,
            'T': 1748980559999,
            's': 'BTC',
            'i': '1m',
            'o': '106076.0',
            'c': '106091.0',
            'h': '106096.0',
            'l': '106045.0',
            'v': '7.54368',
            'n': 155
        }
    ]
    
    for candle in sample_candles:
        candle_id = insertion.insert_candle('1m', candle)
        print(f"✅ Inserted candle for {candle['s']} (ID: {candle_id})")

def test_trade_operations():
    """Test trade operations."""
    print("\n🧪 Testing Trade Operations...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Insert trades
    sample_trades = [
        {
            'coin': 'BTC',
            'side': 'A',  # Ask/Sell
            'px': '105748.0',
            'sz': '0.00157',
            'time': 1748984080014,
            'hash': '0x0000000000000000000000000000000000000000000000000000000000000000',
            'tid': 956057189466947,
            'users': [
                '0xaca2d79168f9e9072c089e5f7d6eb0daefc13915',
                '0x6e2c5323888a98cbbe4d29cfbf1113841ffcf48e'
            ]
        },
        {
            'coin': 'BTC',
            'side': 'B',  # Bid/Buy
            'px': '105749.0',
            'sz': '0.47253',
            'time': 1748984077682,
            'hash': '0xa7bcd83806a238396d870424c8ec5d020195009b212d1831f14d3c14e5602906',
            'tid': 875568394559775,
            'users': [
                '0x5b5d51203a0f9079f8aeb098a6523a13f298c060',
                '0xe13c095580cf6fcb9c087308668062b76b4939e2'
            ]
        }
    ]
    
    for trade in sample_trades:
        trade_id = insertion.insert_trade(trade)
        print(f"✅ Inserted trade {trade['side']} {trade['sz']} {trade['coin']} @ {trade['px']} (ID: {trade_id})")

def test_orderbook_operations():
    """Test orderbook operations."""
    print("\n🧪 Testing Orderbook Operations...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Insert orderbook
    sample_orderbook = {
        'coin': 'BTC',
        'time': 1748984080374,
        'levels': [
            [  # Bids
                {'px': '105748.0', 'sz': '5.73048', 'n': 11},
                {'px': '105747.0', 'sz': '0.33116', 'n': 2},
                {'px': '105746.0', 'sz': '0.04728', 'n': 1}
            ],
            [  # Asks
                {'px': '105749.0', 'sz': '2.1234', 'n': 8},
                {'px': '105750.0', 'sz': '1.5678', 'n': 5},
                {'px': '105751.0', 'sz': '0.9876', 'n': 3}
            ]
        ]
    }
    
    orderbook_id = insertion.insert_orderbook_snapshot(sample_orderbook)
    print(f"✅ Inserted orderbook snapshot for {sample_orderbook['coin']} (ID: {orderbook_id})")

def test_funding_rate_operations():
    """Test funding rate operations."""
    print("\n🧪 Testing Funding Rate Operations...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Insert funding rate (HyperLiquid format)
    sample_funding = {
        "coin": "BTC",
        "fundingRate": "0.0001",
        "premium": "0.00008",
        "time": 1748984080000
    }
    
    insertion.insert_funding_rate(sample_funding)
    print(f"✅ Inserted funding rate for {sample_funding['coin']}")

def test_bulk_operations():
    """Test bulk insertion operations."""
    print("\n🧪 Testing Bulk Insertion Operations...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Insert bulk candles
    bulk_candles = [
        {
            't': 1748980560000 + i * 60000,  # Each minute
            'T': 1748980619999 + i * 60000,
            's': 'ETH',
            'i': '1m',
            'o': f'{2600 + i}',
            'c': f'{2605 + i}', 
            'h': f'{2610 + i}',
            'l': f'{2595 + i}',
            'v': f'{100.5 + i}',
            'n': 50 + i
        }
        for i in range(10)  # 10 candles
    ]
    
    inserted_count = insertion.bulk_insert_candles('1m', bulk_candles)
    print(f"✅ Bulk inserted {inserted_count} ETH candles")

def test_query_operations():
    """Test data querying operations."""
    print("\n🧪 Testing Query Operations...")
    
    query = DataQuery(db_manager)
    
    # Query symbols
    symbols = query.get_symbols()
    print(f"✅ Found {len(symbols)} symbols:")
    for symbol in symbols:
        status = "delisted" if symbol['is_delisted'] else "active"
        print(f"   • {symbol['symbol']}: {symbol['max_leverage']}x leverage ({status})")
        
    # Query candles
    candles = query.get_candles('1m', 'BTC', limit=5)
    print(f"✅ Found {len(candles)} BTC 1m candles:")
    for candle in candles:
        print(f"   • {candle['timestamp_start']}: O:{candle['open_price']} H:{candle['high_price']} L:{candle['low_price']} C:{candle['close_price']} V:{candle['volume']}")
        
    # Query trades
    trades = query.get_latest_trades('BTC', limit=5)
    print(f"✅ Found {len(trades)} BTC trades:")
    for trade in trades:
        side_name = "SELL" if trade['side'] == 'A' else "BUY"
        print(f"   • {trade['timestamp']}: {side_name} {trade['size']} @ {trade['price']}")
        
    # Query orderbook
    orderbook = query.get_latest_orderbook('BTC')
    if orderbook:
        print(f"✅ Found latest BTC orderbook from {orderbook['timestamp']}")
        print(f"   • Bids: {len(orderbook['levels'][0])} levels")
        print(f"   • Asks: {len(orderbook['levels'][1])} levels")

def test_enhanced_orderbook_operations():
    """Test enhanced orderbook operations with HyperLiquid format."""
    print("\n🧪 Testing Enhanced Order Book Operations...")
    
    schema = MarketDataSchema(db_manager)
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Sample HyperLiquid L2Book data
    l2book_data = {
        "coin": "BTC",
        "time": 1748989278468,
        "levels": [
            [  # bids
                {"px": "105699.0", "sz": "12.97221", "n": 30},
                {"px": "105698.0", "sz": "12.75723", "n": 1},
                {"px": "105697.0", "sz": "6.26251", "n": 10}
            ],
            [  # asks
                {"px": "105700.0", "sz": "5.5", "n": 15},
                {"px": "105701.0", "sz": "3.2", "n": 8},
                {"px": "105702.0", "sz": "7.1", "n": 22}
            ]
        ]
    }
    
    # Test insertion
    insertion.insert_orderbook_snapshot(l2book_data)
    print("✅ Enhanced orderbook snapshot inserted successfully")
    
    # Test retrieval
    latest_book = query.get_latest_orderbook("BTC")
    assert latest_book is not None
    assert latest_book['coin'] == "BTC"
    assert latest_book['time'] == 1748989278468
    assert len(latest_book['levels']) == 2
    assert len(latest_book['levels'][0]) == 3  # 3 bids
    assert len(latest_book['levels'][1]) == 3  # 3 asks
    print("✅ Enhanced orderbook retrieval working correctly")
    
    # Test conflict resolution (upsert)
    updated_data = l2book_data.copy()
    updated_data['levels'][0][0]['sz'] = "15.5"  # Update first bid size
    insertion.insert_orderbook_snapshot(updated_data)
    
    latest_book = query.get_latest_orderbook("BTC")
    assert latest_book['levels'][0][0]['sz'] == "15.5"
    print("✅ Enhanced orderbook upsert working correctly")

def test_enhanced_funding_rate_operations():
    """Test enhanced funding rate operations with HyperLiquid format."""
    print("\n🧪 Testing Enhanced Funding Rate Operations...")
    
    schema = MarketDataSchema(db_manager)
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Sample HyperLiquid fundingHistory data for ETH
    funding_data = {
        "coin": "ETH",
        "fundingRate": "0.0000125",
        "premium": "0.0004683623",
        "time": 1748905200002
    }
    
    # Test insertion
    insertion.insert_funding_rate(funding_data)
    print("✅ Enhanced funding rate inserted successfully")
    
    # Test retrieval
    latest_rate = query.get_latest_funding_rate("ETH")
    assert latest_rate is not None
    assert latest_rate['coin'] == "ETH"
    assert latest_rate['fundingRate'] == "0.0000125"
    assert latest_rate['premium'] == "0.0004683623"
    assert latest_rate['time'] == 1748905200002
    assert latest_rate['fundingRateDecimal'] is not None
    assert abs(latest_rate['fundingRateDecimal'] - 0.0000125) < 1e-10
    print("✅ Enhanced funding rate retrieval working correctly")
    
    # Test with missing premium
    funding_data_no_premium = {
        "coin": "SOL",
        "fundingRate": "0.0000075",
        "time": 1748905260002
    }
    
    insertion.insert_funding_rate(funding_data_no_premium)
    latest_sol_rate = query.get_latest_funding_rate("SOL")
    assert latest_sol_rate['fundingRate'] == "0.0000075"
    assert latest_sol_rate['premium'] == ""
    assert latest_sol_rate['premiumDecimal'] is None
    print("✅ Enhanced funding rate with missing fields working correctly")

def test_precision_handling():
    """Test precision handling for financial data."""
    print("\n🧪 Testing Precision Handling...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Test very small funding rate
    precise_funding_data = {
        "coin": "TEST",
        "fundingRate": "0.0000000125",
        "premium": "0.0000004683623",
        "time": 1748905300002
    }
    
    insertion.insert_funding_rate(precise_funding_data)
    result = query.get_latest_funding_rate("TEST")
    
    # Verify string precision is maintained
    assert result['fundingRate'] == "0.0000000125"
    assert result['premium'] == "0.0000004683623"
    
    # Verify decimal conversion is accurate
    assert result['fundingRateDecimal'] is not None
    assert abs(result['fundingRateDecimal'] - 0.0000000125) < 1e-12
    print("✅ High precision handling working correctly")

def test_hyperliquid_data_compatibility():
    """Test compatibility with actual HyperLiquid API response format."""
    print("\n🧪 Testing HyperLiquid Data Compatibility...")
    
    insertion = DataInsertion(db_manager)
    query = DataQuery(db_manager)
    
    # Simulate WebSocket l2Book message
    ws_l2book = {
        "channel": "l2Book",
        "data": {
            "coin": "SOL",
            "time": 1748989280126,
            "levels": [
                [
                    {"px": "155.515", "sz": "11.02066", "n": 16},
                    {"px": "155.514", "sz": "0.00011", "n": 1},
                    {"px": "155.513", "sz": "0.11836", "n": 2}
                ],
                [
                    {"px": "155.516", "sz": "8.5", "n": 12},
                    {"px": "155.517", "sz": "2.3", "n": 5}
                ]
            ]
        }
    }
    
    # Extract data part for insertion
    l2book_data = ws_l2book['data']
    insertion.insert_orderbook_snapshot(l2book_data)
    
    # Verify it can be retrieved in compatible format
    latest_book = query.get_latest_orderbook("SOL")
    assert latest_book['coin'] == "SOL"
    assert latest_book['time'] == 1748989280126
    print("✅ WebSocket l2Book format compatibility verified")
    
    # Test range queries
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    past_hour = now - timedelta(hours=1)
    
    snapshots = query.get_orderbook_snapshots("SOL", start_time=past_hour, limit=10)
    assert len(snapshots) > 0
    assert snapshots[0]['coin'] == "SOL"
    print("✅ Range query functionality working correctly")

if __name__ == "__main__":
    sys.exit(main()) 