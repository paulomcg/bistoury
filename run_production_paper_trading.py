#!/usr/bin/env python3
"""
Maximum Speed Paper Trading with Production Database

This script runs paper trading at maximum speed using the production database
which should have candlestick data available.
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

# Use test database which has the actual data (9.4GB)
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

async def main():
    """Run maximum speed paper trading with production database."""
    
    print("🚀 MAXIMUM SPEED PAPER TRADING")
    print("=" * 50)
    print("📊 Using production database with candle data")
    print("⚡ 100x replay speed optimization")
    print("🎯 Single timeframe focus (15m)")
    print("=" * 50)
    
    # Ultra-optimized configuration for maximum speed
    config = PaperTradingConfig(
        session_name="max_speed_production",
        trading_mode=TradingMode.HISTORICAL,
        
        # Trading parameters optimized for speed
        trading_params=TradingParameters(
            base_position_size=Decimal("100"),
            min_confidence=Decimal("0.50"),  # Lower threshold for more signals
            max_concurrent_positions=5,  # More positions = more activity
            position_sizing=PositionSizing.CONFIDENCE_BASED,
            signal_filtering=SignalFiltering.CONFIDENCE
        ),
        
        # Risk parameters
        risk_params=RiskParameters(
            initial_balance=Decimal("10000"),
            max_position_size=Decimal("500"),
            default_stop_loss_percent=Decimal("0.02"),
            default_take_profit_percent=Decimal("0.04"),
            max_total_exposure=Decimal("2000"),
            max_drawdown_percent=Decimal("0.10"),
            enable_circuit_breaker=True
        ),
        
        # Historical replay - maximum speed settings
        historical_config=HistoricalReplayConfig(
            symbols=["BTC"],
            timeframes=[Timeframe.FIFTEEN_MINUTES],  # Single timeframe only
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 7, tzinfo=timezone.utc),  # 1 week of data
            replay_speed=100.0,  # Maximum speed (100x)
            include_orderbook=False,  # Speed optimization
            include_trades=False,    # Speed optimization
            enable_latency_simulation=False  # Speed optimization
        ),
        
        # Strategy configuration
        enabled_strategies=["candlestick_strategy"],
        strategy_weights={"candlestick_strategy": Decimal("1.0")},
        
        # Performance monitoring
        performance_reporting_interval=timedelta(seconds=10),
        save_state_interval=timedelta(seconds=30)
    )
    
    print(f"📅 Date range: {config.historical_config.start_date.date()} to {config.historical_config.end_date.date()}")
    print(f"⚡ Replay speed: {config.historical_config.replay_speed}x")
    print(f"🎯 Timeframes: {[tf.value for tf in config.historical_config.timeframes]}")
    print(f"💰 Initial balance: ${config.risk_params.initial_balance}")
    print(f"📊 Min confidence: {config.trading_params.min_confidence}")
    print()
    
    # Initialize and run the paper trading engine
    engine = PaperTradingEngine(config)
    
    try:
        print("🔧 Initializing paper trading engine...")
        await engine.initialize()
        
        print("🚀 Starting maximum speed paper trading...")
        print("   (This will process 1 week of data at 100x speed)")
        print()
        
        start_time = datetime.now()
        await engine.start()
        
        # Monitor progress
        while engine.status.is_running:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            status = engine.get_status()
            runtime = (datetime.now() - start_time).total_seconds()
            
            print(f"⏱️  Runtime: {runtime:.1f}s | "
                  f"📊 Data: {status['processed_data_count']:,} | "
                  f"📈 Signals: {status['signals_processed']} | "
                  f"💱 Trades: {status['trades_executed']}")
            
            # Auto-stop after reasonable time (safety measure)
            if runtime > 300:  # 5 minutes max
                print("⏰ Maximum runtime reached, stopping...")
                break
        
        print()
        print("🏁 Paper trading completed!")
        
        # Final performance report
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()
        final_status = engine.get_status()
        
        print("=" * 50)
        print("📊 FINAL PERFORMANCE REPORT")
        print("=" * 50)
        print(f"⏱️  Total runtime: {total_runtime:.2f} seconds")
        print(f"📊 Data points processed: {final_status['processed_data_count']:,}")
        print(f"📈 Signals generated: {final_status['signals_processed']}")
        print(f"💱 Trades executed: {final_status['trades_executed']}")
        print(f"❌ Errors: {final_status['errors']}")
        
        # Calculate processing speed
        if total_runtime > 0:
            processing_speed = final_status['processed_data_count'] / total_runtime
            print(f"🚀 Processing speed: {processing_speed:.1f} data points/second")
            
            if processing_speed > 100:
                print("🟢 EXCELLENT: >100 points/second achieved!")
            elif processing_speed > 50:
                print("🟡 GOOD: >50 points/second achieved")
            else:
                print("🔴 SLOW: <50 points/second - needs optimization")
        
        # Performance statistics
        perf_stats = final_status.get('performance_stats', {})
        if perf_stats:
            print(f"💰 P&L: {perf_stats.get('total_pnl', 0)}")
            print(f"📊 Positions opened: {perf_stats.get('positions_opened', 0)}")
            print(f"📊 Positions closed: {perf_stats.get('positions_closed', 0)}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user...")
    except Exception as e:
        print(f"\n❌ Error during paper trading: {e}")
    finally:
        print("\n⏹️  Stopping engine...")
        await engine.stop()

if __name__ == "__main__":
    # Set up logging for minimal output
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s | %(message)s'
    )
    
    asyncio.run(main()) 