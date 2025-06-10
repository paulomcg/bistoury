#!/usr/bin/env python3
"""
Paper Trading Engine Demo

Demonstrates the Paper Trading Engine with historical data replay,
showing mathematical signal processing and trade execution.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import sys
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


async def create_demo_config() -> PaperTradingConfig:
    """Create demo configuration for paper trading"""
    
    # Historical data range (using recent data from our database)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=1)  # Last 24 hours
    
    # Trading parameters
    trading_params = TradingParameters(
        base_position_size=Decimal("100.0"),  # $100 base position
        max_position_size=Decimal("500.0"),   # $500 max position
        position_sizing=PositionSizing.CONFIDENCE_BASED,
        signal_filtering=SignalFiltering.CONFIDENCE,
        min_confidence=Decimal("0.65"),       # 65% minimum confidence
        max_concurrent_positions=3,
        allow_short_positions=True,
        min_signal_age=timedelta(seconds=1),
        max_signal_age=timedelta(minutes=3)
    )
    
    # Risk parameters
    risk_params = RiskParameters(
        initial_balance=Decimal("10000.0"),   # $10,000 starting balance
        max_drawdown_percent=Decimal("15.0"), # 15% max drawdown
        max_daily_loss_percent=Decimal("5.0"), # 5% daily loss limit
        max_position_percent=Decimal("20.0"),  # 20% max position size
        default_stop_loss_percent=Decimal("2.0"),    # 2% stop loss
        default_take_profit_percent=Decimal("4.0"),  # 4% take profit
        circuit_breaker_enabled=True,
        circuit_breaker_loss_percent=Decimal("10.0")
    )
    
    # Historical replay configuration
    historical_config = HistoricalReplayConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=["BTC"],  # Focus on BTC for demo
        timeframes=[
            Timeframe.ONE_MINUTE,
            Timeframe.FIVE_MINUTES, 
            Timeframe.FIFTEEN_MINUTES
        ],
        replay_speed=10.0,  # 10x speed for demo
        include_orderbook=False,  # Disable for demo speed
        include_trades=False,
        simulate_latency=True,
        base_latency_ms=50,
        latency_variance_ms=25
    )
    
    # Main configuration
    config = PaperTradingConfig(
        trading_mode=TradingMode.HISTORICAL,
        session_name=f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        trading_params=trading_params,
        risk_params=risk_params,
        historical_config=historical_config,
        enabled_strategies={"candlestick_strategy"},
        strategy_weights={"candlestick_strategy": Decimal("1.0")},
        performance_reporting_interval=timedelta(minutes=5),
        enable_detailed_logging=True,
        save_state_interval=timedelta(minutes=2)
    )
    
    return config


async def run_demo():
    """Run the paper trading demo"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger("demo")
    logger.info("🚀 Starting Paper Trading Engine Demo")
    
    try:
        # Create configuration
        config = await create_demo_config()
        logger.info(f"📋 Demo Configuration:")
        logger.info(f"   Session: {config.session_name}")
        logger.info(f"   Mode: {config.trading_mode}")
        logger.info(f"   Date Range: {config.historical_config.start_date} to {config.historical_config.end_date}")
        logger.info(f"   Symbols: {config.historical_config.symbols}")
        logger.info(f"   Initial Balance: ${config.risk_params.initial_balance}")
        logger.info(f"   Min Confidence: {config.trading_params.min_confidence}")
        
        # Create and initialize engine
        engine = PaperTradingEngine(config)
        
        logger.info("🔧 Initializing Paper Trading Engine...")
        await engine.initialize()
        
        logger.info("▶️  Starting paper trading session...")
        
        # Start the engine (this will run the historical replay)
        await engine.start()
        
        # Run for a while to let it process data
        logger.info("⏱️  Running demo for 30 seconds...")
        await asyncio.sleep(30)
        
        # Get status report
        status = engine.get_status()
        logger.info("📊 Demo Status Report:")
        logger.info(f"   Running: {status['is_running']}")
        logger.info(f"   Signals Processed: {status['signals_processed']}")
        logger.info(f"   Trades Executed: {status['trades_executed']}")
        logger.info(f"   Errors: {status['errors']}")
        logger.info(f"   Data Points Processed: {status['replay_progress']['processed_count']}")
        
        performance = status['performance_stats']
        logger.info("💰 Performance Stats:")
        logger.info(f"   Signals Received: {performance['signals_received']}")
        logger.info(f"   Signals Traded: {performance['signals_traded']}")
        logger.info(f"   Positions Opened: {performance['positions_opened']}")
        logger.info(f"   Positions Closed: {performance['positions_closed']}")
        logger.info(f"   Total P&L: ${performance['total_pnl']}")
        
        # Stop the engine
        logger.info("⏹️  Stopping Paper Trading Engine...")
        await engine.stop()
        
        logger.info("✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


async def run_quick_test():
    """Run a quick configuration test"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("quick_test")
    
    try:
        logger.info("🧪 Running quick configuration test...")
        
        # Test configuration creation
        config = await create_demo_config()
        logger.info(f"✅ Configuration created: {config.session_name}")
        
        # Test engine creation
        engine = PaperTradingEngine(config)
        logger.info("✅ Engine created successfully")
        
        # Test status
        status = engine.get_status()
        logger.info(f"✅ Status retrieved: {len(status)} fields")
        
        logger.info("✅ Quick test passed!")
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test mode
        asyncio.run(run_quick_test())
    else:
        # Full demo mode
        asyncio.run(run_demo()) 