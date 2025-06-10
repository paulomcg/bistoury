#!/usr/bin/env python3
"""
Historical Paper Trading with Test Database

This script demonstrates how to run paper trading using the test.duckdb database
which contains 9.4GB of historical trading data.
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


async def setup_test_database():
    """Switch to test database and analyze available data."""
    
    print("🔄 Setting up test database...")
    
    # Set environment variable to use test database
    os.environ['BISTOURY_DATABASE'] = 'test'
    
    # Get database switcher and switch to test database
    switcher = get_database_switcher()
    
    # List available databases
    databases = switcher.list_available_databases()
    print("📋 Available databases:")
    for name, info in databases.items():
        status = "✅" if info['exists'] else "❌"
        print(f"  {status} {name}: {info['size']} ({info.get('schema_type', 'unknown')} schema)")
    
    # Switch to test database
    db_manager = switcher.switch_to_database('test')
    print(f"✅ Switched to test database: {databases['test']['path']}")
    
    # Get compatible query interface for test schema
    query = get_compatible_query(db_manager, 'test_legacy')
    
    # Analyze available data
    print("\n📊 Analyzing test database contents...")
    
    # Get symbols
    symbols = query.get_symbols()
    print(f"  📈 Symbols available: {len(symbols)}")
    if symbols:
        symbol_names = [s['symbol'] for s in symbols[:5]]
        print(f"     First 5: {', '.join(symbol_names)}")
    
    # Get trades for first symbol
    if symbols:
        first_symbol = symbols[0]['symbol']
        trades = query.get_latest_trades(first_symbol, limit=10)
        print(f"  💱 Recent trades for {first_symbol}: {len(trades)}")
        
        if trades:
            # Get date range
            timestamps = [t['timestamp'] for t in trades if t['timestamp']]
            if timestamps:
                min_time = min(timestamps)
                max_time = max(timestamps)
                print(f"     Date range: {min_time} to {max_time}")
    
    # Get orderbook sample
    if symbols:
        orderbook = query.get_latest_orderbook(symbols[0]['symbol'])
        if orderbook:
            levels = orderbook.get('levels', [[], []])
            bids = len(levels[0]) if len(levels) > 0 else 0
            asks = len(levels[1]) if len(levels) > 1 else 0
            print(f"  📖 Latest orderbook levels: {bids} bids, {asks} asks")
    
    return symbols, db_manager


async def create_test_config(symbols) -> PaperTradingConfig:
    """Create configuration for test database paper trading."""
    
    print("\n⚙️ Creating paper trading configuration...")
    
    # Use a reasonable date range from test data
    # For demo purposes, we'll use a 2-hour window
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=2)
    
    # Focus on BTC if available, otherwise use first symbol
    target_symbols = ["BTC"]
    if symbols:
        available_symbols = [s['symbol'] for s in symbols]
        target_symbols = [s for s in target_symbols if s in available_symbols]
        if not target_symbols:
            target_symbols = [available_symbols[0]]  # Use first available
    
    print(f"  🎯 Target symbols: {target_symbols}")
    print(f"  📅 Date range: {start_date} to {end_date}")
    
    # Trading parameters optimized for test data
    trading_params = TradingParameters(
        base_position_size=Decimal("50.0"),     # $50 base position (smaller for testing)
        max_position_size=Decimal("200.0"),    # $200 max position  
        position_sizing=PositionSizing.CONFIDENCE_BASED,
        signal_filtering=SignalFiltering.CONFIDENCE,
        min_confidence=Decimal("0.60"),        # 60% minimum confidence (lower for more signals)
        max_concurrent_positions=2,            # Fewer concurrent positions for testing
        allow_short_positions=True,
        min_signal_age=timedelta(seconds=1),
        max_signal_age=timedelta(minutes=5)    # Longer signal age for historical data
    )
    
    # Risk parameters
    risk_params = RiskParameters(
        initial_balance=Decimal("5000.0"),     # $5,000 starting balance
        max_drawdown_percent=Decimal("20.0"),  # 20% max drawdown
        max_daily_loss_percent=Decimal("10.0"), # 10% daily loss limit
        max_position_percent=Decimal("25.0"),   # 25% max position size
        default_stop_loss_percent=Decimal("3.0"),    # 3% stop loss
        default_take_profit_percent=Decimal("6.0"),  # 6% take profit
        circuit_breaker_enabled=True,
        circuit_breaker_loss_percent=Decimal("15.0")
    )
    
    # Historical replay configuration
    historical_config = HistoricalReplayConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=target_symbols,
        timeframes=[
            Timeframe.ONE_MINUTE,
            Timeframe.FIVE_MINUTES,
            Timeframe.FIFTEEN_MINUTES
        ],
        replay_speed=5.0,  # 5x speed for reasonable demo time
        include_orderbook=True,   # Include orderbook for test
        include_trades=True,      # Include trades for test
        simulate_latency=True,
        base_latency_ms=25,
        latency_variance_ms=15
    )
    
    # Main configuration
    config = PaperTradingConfig(
        trading_mode=TradingMode.HISTORICAL,
        session_name=f"test_db_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        trading_params=trading_params,
        risk_params=risk_params,
        historical_config=historical_config,
        enabled_strategies={"candlestick_strategy"},
        strategy_weights={"candlestick_strategy": Decimal("1.0")},
        performance_reporting_interval=timedelta(minutes=2),
        enable_detailed_logging=True,
        save_state_interval=timedelta(minutes=1)
    )
    
    print(f"  ✅ Configuration created: {config.session_name}")
    return config


async def run_paper_trading_session(config: PaperTradingConfig):
    """Run the paper trading session with test database."""
    
    print(f"\n🚀 Starting paper trading session: {config.session_name}")
    print("=" * 60)
    
    try:
        # Create and initialize engine
        engine = PaperTradingEngine(config)
        
        print("🔧 Initializing Paper Trading Engine...")
        await engine.initialize()
        
        print("▶️  Starting historical replay...")
        print(f"   📊 Symbols: {config.historical_config.symbols}")
        print(f"   ⏱️  Speed: {config.historical_config.replay_speed}x")
        print(f"   💰 Initial Balance: ${config.risk_params.initial_balance}")
        print(f"   📈 Min Signal Confidence: {config.trading_params.min_confidence}")
        
        # Start the engine
        await engine.start()
        
        # Run session and report progress
        print(f"\n⏳ Running session for 60 seconds...")
        print("   (Press Ctrl+C to stop early)")
        
        start_time = asyncio.get_event_loop().time()
        report_interval = 10  # Report every 10 seconds
        last_report = start_time
        
        try:
            while True:
                await asyncio.sleep(1)
                current_time = asyncio.get_event_loop().time()
                
                # Report progress every interval
                if current_time - last_report >= report_interval:
                    await report_progress(engine)
                    last_report = current_time
                
                # Stop after 60 seconds
                if current_time - start_time >= 60:
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⚠️  Session interrupted by user")
        
        # Final report
        print(f"\n📋 Final Session Report:")
        print("=" * 40)
        await report_final_results(engine)
        
        # Stop the engine
        print(f"\n⏹️  Stopping Paper Trading Engine...")
        await engine.stop()
        
        print(f"✅ Paper trading session completed successfully!")
        
    except Exception as e:
        print(f"❌ Paper trading session failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def report_progress(engine):
    """Report current progress of the trading session."""
    
    status = engine.get_status()
    
    print(f"\n📊 Progress Report ({datetime.now().strftime('%H:%M:%S')}):")
    print(f"   Status: {'🟢 Running' if status['is_running'] else '🔴 Stopped'}")
    print(f"   Data Processed: {status.get('processed_data_count', 0):,} points")
    print(f"   Signals Processed: {status['signals_processed']}")
    print(f"   Trades Executed: {status['trades_executed']}")
    print(f"   Errors: {status['errors']}")
    
    performance = status.get('performance_stats', {})
    if performance:
        print(f"   💰 Current P&L: ${performance.get('total_pnl', 0)}")
        print(f"   📈 Positions: {performance.get('positions_opened', 0)} opened, {performance.get('positions_closed', 0)} closed")


async def report_final_results(engine):
    """Report final trading results."""
    
    status = engine.get_status()
    performance = status.get('performance_stats', {})
    
    print(f"📈 Trading Summary:")
    print(f"   Signals Received: {performance.get('signals_received', 0)}")
    print(f"   Signals Traded: {performance.get('signals_traded', 0)}")
    print(f"   Trade Success Rate: {(performance.get('signals_traded', 0) / max(1, performance.get('signals_received', 1)) * 100):.1f}%")
    
    print(f"\n💼 Portfolio Results:")
    print(f"   Positions Opened: {performance.get('positions_opened', 0)}")
    print(f"   Positions Closed: {performance.get('positions_closed', 0)}")
    print(f"   Total P&L: ${performance.get('total_pnl', 0)}")
    
    print(f"\n📊 System Performance:")
    print(f"   Data Points Processed: {status.get('processed_data_count', 0):,}")
    print(f"   Total Errors: {status['errors']}")
    print(f"   Session Duration: {status.get('uptime_seconds', 0):.1f} seconds")


async def main():
    """Main function to run paper trading with test database."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("🎯 Bistoury Paper Trading with Test Database")
    print("=" * 50)
    
    try:
        # Setup test database
        symbols, db_manager = await setup_test_database()
        
        if not symbols:
            print("❌ No symbols found in test database. Cannot proceed.")
            return
        
        # Create configuration
        config = await create_test_config(symbols)
        
        # Run paper trading session
        await run_paper_trading_session(config)
        
    except Exception as e:
        print(f"❌ Failed to run paper trading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup database connections
        try:
            switcher = get_database_switcher()
            if hasattr(switcher, 'current_manager') and switcher.current_manager:
                switcher.current_manager.close_all_connections()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main()) 