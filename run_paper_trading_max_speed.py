#!/usr/bin/env python3
"""
Maximum Speed Historical Paper Trading with Test Database

This script runs paper trading at maximum speed using the test database
which contains 436,138 trades and 587,661 orderbook snapshots.
Since there's no candle data, we'll use recent date ranges where trades exist.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Use test database which has the actual data
os.environ['BISTOURY_DATABASE'] = 'test'

from src.bistoury.paper_trading import PaperTradingEngine
from src.bistoury.paper_trading.config import (
    PaperTradingConfig,
    TradingParameters, 
    RiskParameters,
    HistoricalReplayConfig,
    TradingMode,
    PositionSizing,
    SignalFiltering
)
from src.bistoury.models.market_data import Timeframe
from src.bistoury.database import get_database_switcher, get_compatible_query

# Set up logging for minimal output
logging.getLogger().setLevel(logging.WARNING)  # Only show warnings and errors

async def main():
    """Run maximum speed paper trading with test database."""
    
    print("🚀 MAXIMUM SPEED PAPER TRADING (TEST DATABASE)")
    print("=" * 50)
    print("📊 Using test database with 436k trades + 587k orderbook")
    print("⚡ 100x replay speed optimization")
    print("🎯 Trade data from May 2025")
    print("=" * 50)
    
    # Check available data first
    switcher = get_database_switcher()
    db = switcher.switch_to_database('test')
    query = get_compatible_query(db, 'test_legacy')
    trades = query.get_latest_trades('BTC', limit=10)
    
    if trades:
        latest_trade = trades[0]
        print(f"📅 Latest trade: {latest_trade['timestamp']}")
        print(f"💰 Latest price: ${latest_trade['price']:,.2f}")
        
        # Use a date range around the latest trades (last few days)
        end_date = latest_trade['timestamp']
        start_date = end_date - timedelta(days=3)  # 3 days of data
        
        print(f"📅 Date range: {start_date} to {end_date}")
    else:
        print("❌ No trades found in database!")
        return
    
    # Configure for maximum speed paper trading
    config = PaperTradingConfig(
        trading_mode=TradingMode.HISTORICAL,
        session_name="max_speed_test_db",
        
        # Speed-optimized trading parameters
        trading_params=TradingParameters(
            base_position_size=Decimal("100.0"),
            max_position_size=Decimal("500.0"),
            min_confidence=Decimal("0.50"),  # Lower threshold for more signals
            max_concurrent_positions=5,  # More positions = more activity
            position_sizing=PositionSizing.CONFIDENCE_BASED,
            signal_filtering=SignalFiltering.CONFIDENCE
        ),
        
        # Risk parameters
        risk_params=RiskParameters(
            initial_balance=Decimal("10000.0"),
            max_drawdown_percent=Decimal("20.0"),
            default_stop_loss_percent=Decimal("2.0"),
            default_take_profit_percent=Decimal("4.0")
        ),
        
        # MAXIMUM SPEED historical replay configuration
        historical_config=HistoricalReplayConfig(
            symbols=["BTC"],
            timeframes=[Timeframe.FIFTEEN_MINUTES],  # Single timeframe only
            start_date=start_date,
            end_date=end_date,
            
            # SPEED OPTIMIZATIONS:
            replay_speed=100.0,          # 100x speed (maximum)
            include_orderbook=False,     # Skip orderbook for speed
            include_trades=True,         # Keep trades for signal generation
            simulate_latency=False,      # No artificial delays
            base_latency_ms=0,          # Zero latency
            latency_variance_ms=0       # No latency variance
        ),
        
        # Strategy configuration
        enabled_strategies={"candlestick_strategy"},
        strategy_weights={"candlestick_strategy": Decimal("1.0")},
        
        # Performance monitoring (minimal for speed)
        performance_reporting_interval=timedelta(minutes=30),
        enable_detailed_logging=False,  # Disable for speed
        save_state_interval=timedelta(minutes=10)
    )
    
    print(f"⚡ Replay speed: {config.historical_config.replay_speed}x")
    print(f"🎯 Timeframes: {[tf.value for tf in config.historical_config.timeframes]}")
    print(f"💰 Initial balance: ${config.risk_params.initial_balance}")
    print(f"📊 Min confidence: {config.trading_params.min_confidence}")
    print(f"")
    
    # Initialize and run paper trading engine
    print("🔧 Initializing paper trading engine...")
    engine = PaperTradingEngine(config)
    
    try:
        # Start the engine
        print("🚀 Starting maximum speed paper trading...")
        print("   (This will process 3 days of trade data at 100x speed)")
        print("")
        
        await engine.start()
        
        # Let it run for a reasonable time
        duration = 60  # 60 seconds of real time
        print(f"⏱️  Running for {duration} seconds (processing ~{duration * 100 / 60:.1f} hours of market data)...")
        
        # Monitor progress
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration:
            await asyncio.sleep(5.0)  # Check every 5 seconds
            
            runtime = asyncio.get_event_loop().time() - start_time
            
            # Get stats (simplified)
            stats = getattr(engine, 'stats', {})
            data_points = stats.get('data_points_processed', 0)
            signals = stats.get('signals_generated', 0) 
            trades = stats.get('trades_executed', 0)
            
            print(f"⏱️  Runtime: {runtime:.1f}s | 📊 Data: {data_points} | 📈 Signals: {signals} | 💱 Trades: {trades}")
        
        print("")
        print("✅ Paper trading session completed!")
        
        # Show final results
        if hasattr(engine, 'get_performance_summary'):
            summary = await engine.get_performance_summary()
            print("📊 Final Performance Summary:")
            print(f"   💰 Final Balance: ${summary.get('final_balance', 'N/A')}")
            print(f"   📈 Total Return: {summary.get('total_return_pct', 'N/A')}%")
            print(f"   💱 Total Trades: {summary.get('total_trades', 'N/A')}")
            print(f"   📊 Data Processed: {summary.get('data_points', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error during paper trading: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Stop the engine
        print("🛑 Stopping paper trading engine...")
        try:
            await engine.stop()
        except Exception as e:
            print(f"⚠️  Warning during shutdown: {e}")
        
        print("✅ Paper trading engine stopped.")

if __name__ == "__main__":
    asyncio.run(main()) 