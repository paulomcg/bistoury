# Maximum Speed Paper Trading Guide

This guide shows you how to run paper trading at **maximum speed** to process large amounts of historical data as quickly as possible.

## Quick Start: Maximum Speed

### Run at 100x Speed (Recommended)

```bash
python run_paper_trading_max_speed.py
```

**What this does:**
- ⚡ **100x replay speed** (maximum possible)
- 🎯 **Single timeframe** (15m only - most important)
- 🚫 **No orderbook data** (speed optimization)
- 🚫 **No trade execution data** (speed optimization)
- 🚫 **No latency simulation** (speed optimization)
- 📊 **Lower confidence threshold** (50% for more signals)
- 🔥 **Minimal logging** (WARNING level only)
- 📈 **5 concurrent positions** (more trading activity)

## Speed Optimization Techniques

### 1. Replay Speed Configuration

```python
# Maximum Speed Settings
replay_speed = 100.0          # 100x real-time (MAXIMUM)
replay_speed = 50.0           # 50x real-time (Fast)
replay_speed = 10.0           # 10x real-time (Moderate)
replay_speed = 1.0            # 1x real-time (Normal)
```

### 2. Timeframe Optimization

```python
# FASTEST: Single timeframe
timeframes = [Timeframe.FIFTEEN_MINUTES]

# FAST: Two timeframes
timeframes = [Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES]

# MODERATE: Three timeframes
timeframes = [Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES]
```

### 3. Data Source Optimization

```python
# MAXIMUM SPEED: No additional data
include_orderbook = False
include_trades = False

# MODERATE SPEED: Include some data
include_orderbook = True
include_trades = False

# FULL DATA: Everything (slowest)
include_orderbook = True
include_trades = True
```

### 4. Latency Optimization

```python
# MAXIMUM SPEED: No simulation
simulate_latency = False
base_latency_ms = 0
latency_variance_ms = 0

# REALISTIC: Some simulation
simulate_latency = True
base_latency_ms = 25
latency_variance_ms = 15
```

### 5. Logging Optimization

```python
# MAXIMUM SPEED: Minimal logging
logging.basicConfig(level=logging.WARNING)
enable_detailed_logging = False

# MODERATE: Info logging
logging.basicConfig(level=logging.INFO)
enable_detailed_logging = False

# FULL DEBUG: All logging (slowest)
logging.basicConfig(level=logging.DEBUG)
enable_detailed_logging = True
```

## Speed Comparison

| Configuration | Speed | Data Rate | Use Case |
|---------------|-------|-----------|----------|
| **Maximum Speed** | 100x | >100 pts/sec | Large data processing |
| **Fast** | 50x | 50-100 pts/sec | Quick backtesting |
| **Moderate** | 10x | 10-50 pts/sec | Detailed analysis |
| **Realistic** | 5x | 5-10 pts/sec | Strategy validation |
| **Real-time** | 1x | 1-5 pts/sec | Live simulation |

## Expected Performance Results

### Maximum Speed Run Example

```
🚀 BISTOURY MAXIMUM SPEED PAPER TRADING
==================================================
⚡ All speed optimizations enabled!

⚡ Speed Report (15.0s elapsed):
   📊 Data: 2,456 points (163.7/sec)
   📈 Signals: 47 processed
   💱 Trades: 12 executed
   💰 P&L: $+234.56 | Positions: 8

🏁 MAXIMUM SPEED TEST RESULTS:
========================================
⏱️  Performance Metrics:
   Duration: 120.0 seconds
   Data Rate: 185.3 points/second
   Signal Rate: 0.85 signals/second
   Trade Rate: 0.23 trades/second

🏆 Speed Assessment: 🚀 EXCELLENT
   Target: >100 points/second
   Achieved: 185.3 points/second
```

## Advanced Speed Configurations

### Ultra-Fast Configuration

```python
# For processing massive datasets quickly
PaperTradingConfig(
    trading_mode=TradingMode.HISTORICAL,
    historical_config=HistoricalReplayConfig(
        replay_speed=100.0,           # Maximum speed
        timeframes=[Timeframe.FIFTEEN_MINUTES],  # Single timeframe
        include_orderbook=False,      # No orderbook
        include_trades=False,         # No trades
        simulate_latency=False        # No latency
    ),
    trading_params=TradingParameters(
        min_confidence=Decimal("0.40"),  # Very low threshold
        max_concurrent_positions=10,     # Many positions
        min_signal_age=timedelta(0)      # No minimum age
    ),
    performance_reporting_interval=timedelta(seconds=2),  # Fast reporting
    enable_detailed_logging=False     # No detailed logs
)
```

### Balanced Speed Configuration

```python
# Good balance of speed and realism
PaperTradingConfig(
    historical_config=HistoricalReplayConfig(
        replay_speed=25.0,            # Fast but manageable
        timeframes=[Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES],
        include_orderbook=False,      # Skip orderbook
        include_trades=True,          # Include trades
        simulate_latency=True,        # Some latency
        base_latency_ms=10           # Minimal latency
    ),
    trading_params=TradingParameters(
        min_confidence=Decimal("0.55"),  # Moderate threshold
        max_concurrent_positions=3       # Reasonable positions
    )
)
```

## System Performance Optimization

### Hardware Optimization

```bash
# Use SSD storage for database
mv data/test.duckdb /path/to/ssd/test.duckdb
ln -s /path/to/ssd/test.duckdb data/test.duckdb

# Increase system resources
export OMP_NUM_THREADS=8          # Use more CPU cores
export MALLOC_ARENA_MAX=4         # Optimize memory allocation
```

### Database Optimization

```bash
# Pre-analyze database for better performance
python -c "
from src.bistoury.database import get_database_switcher
switcher = get_database_switcher()
db = switcher.switch_to_database('test')
db.execute('ANALYZE;')
"
```

### Python Optimization

```bash
# Run with optimizations
python -O run_paper_trading_max_speed.py

# Use pypy for even faster execution (if compatible)
pypy3 run_paper_trading_max_speed.py
```

## Memory Management

### Large Dataset Handling

```python
# For processing very large datasets
config = PaperTradingConfig(
    # Process in smaller chunks
    historical_config=HistoricalReplayConfig(
        start_date=start_date,
        end_date=start_date + timedelta(hours=2),  # 2-hour chunks
        # ... other config
    ),
    # Less frequent state saves
    save_state_interval=timedelta(minutes=30),
    # Faster cleanup
    performance_reporting_interval=timedelta(seconds=10)
)
```

### Memory Monitoring

```python
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Call periodically during trading
```

## Troubleshooting Speed Issues

### Common Performance Bottlenecks

1. **Database I/O**
   ```bash
   # Check if database is on SSD
   df -T data/test.duckdb
   
   # Move to faster storage if needed
   mv data/test.duckdb /tmp/test.duckdb
   ln -s /tmp/test.duckdb data/test.duckdb
   ```

2. **Too Much Logging**
   ```python
   # Disable all non-essential logging
   logging.getLogger().setLevel(logging.CRITICAL)
   ```

3. **Memory Pressure**
   ```bash
   # Check memory usage
   top -p $(pgrep -f python)
   
   # Reduce dataset size or use chunking
   ```

4. **CPU Bottleneck**
   ```bash
   # Check CPU usage
   htop
   
   # Reduce concurrent operations if needed
   ```

### Speed Optimization Checklist

- ✅ Use `replay_speed=100.0` for maximum speed
- ✅ Single timeframe: `[Timeframe.FIFTEEN_MINUTES]`
- ✅ Disable orderbook: `include_orderbook=False`
- ✅ Disable trades: `include_trades=False`
- ✅ No latency simulation: `simulate_latency=False`
- ✅ Minimal logging: `level=logging.WARNING`
- ✅ Lower confidence: `min_confidence=0.40`
- ✅ More positions: `max_concurrent_positions=10`
- ✅ Fast reporting: `timedelta(seconds=2)`
- ✅ SSD storage for database
- ✅ Sufficient RAM (8GB+ recommended)

## Speed Benchmarks

### Target Performance Metrics

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Data Rate | >150 pts/sec | >75 pts/sec | >25 pts/sec | <25 pts/sec |
| Signal Rate | >1.0 sig/sec | >0.5 sig/sec | >0.1 sig/sec | <0.1 sig/sec |
| Memory Usage | <500MB | <1GB | <2GB | >2GB |
| CPU Usage | <50% | <75% | <90% | >90% |

### Real-World Performance

Based on test runs with the 9.4GB test database:

- **Maximum Speed**: 150-300 points/second
- **Fast Mode**: 75-150 points/second  
- **Moderate Mode**: 25-75 points/second
- **Realistic Mode**: 5-25 points/second

## Production Speed Considerations

### For Live Trading

When moving to production, balance speed with accuracy:

```python
# Production-optimized configuration
config = PaperTradingConfig(
    replay_speed=1.0,             # Real-time for live trading
    timeframes=[                  # Multiple timeframes for accuracy
        Timeframe.ONE_MINUTE,
        Timeframe.FIVE_MINUTES,
        Timeframe.FIFTEEN_MINUTES
    ],
    include_orderbook=True,       # Include for better signals
    simulate_latency=True,        # Realistic latency
    enable_detailed_logging=True, # Full logging for debugging
    min_confidence=Decimal("0.65") # Higher confidence for real money
)
```

Run **maximum speed tests** first to validate your strategies, then use **realistic settings** for actual trading.

## Running the Maximum Speed Test

```bash
# Maximum speed with all optimizations
python run_paper_trading_max_speed.py

# Expected output: 100-300 points/second processing rate
# Test duration: 2 minutes
# Reports every 5 seconds
```

This will give you the fastest possible paper trading performance with your 9.4GB test database! 