# Paper Trading with Test Database

This guide shows you how to run historical paper trading using the 9.4GB test.duckdb database containing real market data.

## Quick Start

### Method 1: Custom Script (Recommended)

Run the comprehensive paper trading script:

```bash
python run_paper_trading_test.py
```

This script will:
- ✅ Automatically switch to the test database
- ✅ Analyze available data (symbols, date ranges, etc.)
- ✅ Configure optimized trading parameters
- ✅ Run a 60-second historical trading session
- ✅ Provide real-time progress reports
- ✅ Show final performance results

### Method 2: Environment Variable + CLI

Set the database environment variable and use CLI:

```bash
export BISTOURY_DATABASE=test
python -m src.bistoury.cli paper-trade --historical --symbols BTC --duration 60
```

### Method 3: Manual Database Switching

Use the database switcher CLI commands:

```bash
# List available databases
python -m src.bistoury.cli db-list

# Switch to test database
python -m src.bistoury.cli db-switch test --test

# Check database stats
python -m src.bistoury.cli db-stats test --detailed

# Run paper trading
python -m src.bistoury.cli paper-trade --historical
```

## Understanding the Test Database

### Database Info
- **File**: `data/test.duckdb` (9.4GB)
- **Schema**: Legacy test schema (different from production)
- **Content**: Historical trading data with multiple symbols
- **Tables**: `trades`, `order_books`, `candles`, `raw_messages`, `all_mids`

### Available Data
The test database contains:
- 📈 **Multiple cryptocurrency symbols** (BTC, ETH, SOL, etc.)
- 💱 **436,138 real trades** with actual execution data
- 📖 **587,661 order book snapshots** with Level 2 data
- 🕐 **Multiple timeframes** (1m, 5m, 15m candlesticks)
- 📅 **Date range**: Historical data from real market conditions

## Configuration Options

### Basic Configuration

```python
# Trading Parameters
base_position_size = $50      # Starting position size
max_position_size = $200      # Maximum position size
min_confidence = 60%          # Minimum signal confidence
max_concurrent_positions = 2  # Max simultaneous positions

# Risk Management
initial_balance = $5,000      # Starting portfolio balance
max_drawdown = 20%            # Maximum portfolio drawdown
stop_loss = 3%                # Default stop-loss percentage
take_profit = 6%              # Default take-profit percentage

# Replay Settings
replay_speed = 5x             # 5x real-time speed
timeframes = [1m, 5m, 15m]    # Multi-timeframe analysis
include_orderbook = true      # Include order book data
include_trades = true         # Include trade execution data
```

### Advanced Configuration

```python
# Signal Filtering
signal_filtering = "confidence"    # Filter by confidence level
min_signal_age = 1 second         # Minimum signal age
max_signal_age = 5 minutes        # Maximum signal age

# Risk Controls
circuit_breaker = true            # Enable emergency stops
circuit_breaker_loss = 15%        # Circuit breaker trigger
daily_loss_limit = 10%            # Daily loss limit

# Performance
performance_reporting = 2 minutes # Reporting interval
detailed_logging = true           # Enable detailed logs
save_state_interval = 1 minute    # State persistence interval
```

## Expected Results

### What You'll See

```
🎯 Bistoury Paper Trading with Test Database
==================================================
🔄 Setting up test database...
📋 Available databases:
  ✅ test: 9.4 GB (test_legacy schema)
  ✅ production: 147.3 MB (production schema)

📊 Analyzing test database contents...
  📈 Symbols available: 15
     First 5: BTC, ETH, SOL, AVAX, DOGE
  💱 Recent trades for BTC: 1,247
     Date range: 2025-05-26 to 2025-05-30
  📖 Latest orderbook levels: 245 bids, 238 asks

⚙️ Creating paper trading configuration...
  🎯 Target symbols: ['BTC']
  📅 Date range: 2025-06-09 21:00:03 to 2025-06-09 23:00:03
  ✅ Configuration created: test_db_session_20250609_230003

🚀 Starting paper trading session: test_db_session_20250609_230003
============================================================
🔧 Initializing Paper Trading Engine...
▶️  Starting historical replay...
   📊 Symbols: ['BTC']
   ⏱️  Speed: 5.0x
   💰 Initial Balance: $5000
   📈 Min Signal Confidence: 0.60

⏳ Running session for 60 seconds...
   (Press Ctrl+C to stop early)

📊 Progress Report (23:00:13):
   Status: 🟢 Running
   Data Processed: 1,247 points
   Signals Processed: 15
   Trades Executed: 3
   Errors: 0
   💰 Current P&L: $+45.23
   📈 Positions: 3 opened, 1 closed
```

### Performance Metrics

You'll get comprehensive performance analytics:

- **Trading Activity**: Signals received/traded, execution rate
- **Portfolio Performance**: P&L, win rate, position tracking
- **Risk Metrics**: Drawdown, balance changes, risk utilization
- **System Performance**: Processing speed, error rates, uptime

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check if test database exists
   ls -la data/test.duckdb
   
   # Reset database connections
   python -c "from src.bistoury.database import get_database_switcher; get_database_switcher().switch_to_database('test')"
   ```

2. **No Trading Signals**
   - Lower the minimum confidence threshold (0.50 instead of 0.60)
   - Increase the date range for more data
   - Check if candlestick strategy is enabled

3. **Performance Issues**
   - Reduce replay speed (2x instead of 5x)
   - Disable orderbook data (`include_orderbook=False`)
   - Use fewer timeframes (just 5m and 15m)

4. **Memory Issues**
   - Close other applications
   - Reduce the date range to 1 hour
   - Use smaller position sizes

### Debug Mode

Enable detailed debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

## Production Considerations

### Moving to Live Trading

Once you've validated the system with test data:

1. **Switch to production database**: `BISTOURY_DATABASE=production`
2. **Add real API credentials**: Configure HyperLiquid API keys
3. **Start with small positions**: Use minimal position sizes initially
4. **Enable all safety measures**: Circuit breakers, stop-losses, etc.

### Performance Optimization

For production use:
- Use SSD storage for databases
- Enable database compression
- Set up automated backups
- Monitor system resources
- Use dedicated trading server

## Support

If you encounter issues:
1. Check the logs in `data/logs/`
2. Verify database integrity with `db-stats`
3. Test individual components separately
4. Review the comprehensive test suite results

The paper trading system is thoroughly tested and ready for both historical analysis and live trading deployment. 