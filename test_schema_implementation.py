#!/usr/bin/env python3
"""
Test script for database schema implementation.
Tests all schema operations with sample HyperLiquid data.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bistoury.config import Config
from bistoury.database import DatabaseManager, MarketDataSchema, DataInsertion, DataQuery

def main():
    """Test schema implementation."""
    print("🧪 Testing Database Schema Implementation")
    print("=" * 50)
    
    try:
        # 1. Initialize components
        print("\n1️⃣ Initializing components...")
        cfg = Config()
        db_manager = DatabaseManager(cfg)
        schema = MarketDataSchema(db_manager)
        insertion = DataInsertion(db_manager)
        query = DataQuery(db_manager)
        print("✅ Components initialized")
        
        # 2. Create schema
        print("\n2️⃣ Creating database schema...")
        schema.create_all_tables()
        print("✅ Schema created")
        
        # 3. Validate schema
        print("\n3️⃣ Validating schema...")
        if schema.validate_schema():
            print("✅ Schema validation passed")
        else:
            print("❌ Schema validation failed")
            return
            
        # 4. Test symbol insertion
        print("\n4️⃣ Testing symbol insertion...")
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
            
        # 5. Test candlestick insertion
        print("\n5️⃣ Testing candlestick insertion...")
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
            
        # 6. Test trade insertion
        print("\n6️⃣ Testing trade insertion...")
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
            
        # 7. Test orderbook insertion
        print("\n7️⃣ Testing orderbook insertion...")
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
        
        # 8. Test funding rate insertion
        print("\n8️⃣ Testing funding rate insertion...")
        sample_funding = {
            'symbol': 'BTC',
            'timestamp': 1748984080000,
            'funding_rate': 0.0001,
            'predicted_rate': 0.00008,
            'open_interest': 1500000.0
        }
        
        funding_id = insertion.insert_funding_rate(sample_funding)
        print(f"✅ Inserted funding rate for {sample_funding['symbol']} (ID: {funding_id})")
        
        # 9. Test data querying
        print("\n9️⃣ Testing data querying...")
        
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
            print(f"   • Bids: {len(orderbook['bids'])} levels")
            print(f"   • Asks: {len(orderbook['asks'])} levels")
        
        # 10. Test bulk insertion
        print("\n🔟 Testing bulk insertion...")
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
        
        # 11. Get schema information
        print("\n1️⃣1️⃣ Schema information...")
        schema_info = schema.get_schema_info()
        for table_name, info in schema_info.items():
            row_count = info.get('row_count', 0)
            col_count = len(info.get('columns', []))
            print(f"✅ {table_name}: {row_count} rows, {col_count} columns")
            
        print("\n🎉 All tests passed successfully!")
        print("\n📈 Database schema is ready for market data storage")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()
            print("\n🧹 Database connections cleaned up")
            
    return 0

if __name__ == "__main__":
    sys.exit(main()) 