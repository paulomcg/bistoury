# Database Compatibility Setup for Bistoury

## Overview

Your Bistoury system now supports multiple database files with automatic schema detection and compatibility layers. This allows you to seamlessly switch between production and test databases for development, testing, and backtesting purposes.

## Available Databases

| Database Name | Path | Size | Schema Type | Purpose |
|---------------|------|------|-------------|---------|
| `production` | `data/bistoury.db` | 14.5 MB | production | Live production data with full schema |
| `test` | `data/test.duckdb` | 9.4 GB | test_legacy | Historical data for backtesting (4+ days) |
| `memory` | `:memory:` | N/A | unknown | In-memory database for testing |

## Test Database Contents

The test database contains real historical trading data:

- **Date Range**: 2025-05-26 to 2025-05-30 (4+ days of data)
- **Trading Data**: 436,138 BTC trades
- **Order Book Data**: 587,661 level 2 snapshots  
- **Message Data**: 1.3M+ raw market messages
- **Total Size**: 9.4 GB of high-quality market data

## Database Switching

### 1. Using CLI Commands

```bash
# List all available databases
poetry run python -c "from src.bistoury.cli.commands import cli; cli()" db-list

# Switch to test database with connection test
poetry run python -c "from src.bistoury.cli.commands import cli; cli()" db-switch test --test

# Get detailed statistics
poetry run python -c "from src.bistoury.cli.commands import cli; cli()" db-stats test --detailed

# Switch back to production
poetry run python -c "from src.bistoury.cli.commands import cli; cli()" db-switch production
```

### 2. Using Environment Variables

```bash
# Run with test database (for backtesting)
BISTOURY_DATABASE=test poetry run python your_script.py

# Run with production database (default)
BISTOURY_DATABASE=production poetry run python your_script.py

# Run with in-memory database (for testing)
BISTOURY_DATABASE=memory poetry run python your_script.py
```

### 3. Programmatic Usage

```python
from bistoury.database import get_database_switcher, switch_database, get_compatible_query

# Get database switcher
switcher = get_database_switcher()

# List available databases
databases = switcher.list_available_databases()

# Switch to test database
db_manager = switch_database('test')

# Get compatible query interface
query = get_compatible_query(db_manager, 'test_legacy')

# Query data using unified interface
symbols = query.get_symbols()
trades = query.get_latest_trades('BTC', limit=100)
orderbook = query.get_latest_orderbook('BTC')
```

## Compatibility Layer

The system automatically detects schema types and provides compatibility layers:

### Production Schema
- Full schema with symbols, trades, orderbook_snapshots, funding_rates, candles tables
- HyperLiquid API compatible format
- Decimal precision for cryptocurrency trading

### Test Legacy Schema  
- Original format: trades, order_books, raw_messages, all_mids, candles tables
- Automatic symbol extraction from trades
- JSON level storage for order books
- Timestamp compatibility

### Unified Query Interface

The `CompatibleDataQuery` class provides a unified interface that works with both schemas:

```python
# Works with both production and test databases
symbols = query.get_symbols()           # Extracted from trades in test DB
trades = query.get_latest_trades(...)   # Compatible field mapping
orderbook = query.get_latest_orderbook(...)  # JSON level parsing
candles = query.get_candles(...)        # Only available in production
```

## Configuration

Add to your `.env` file:

```bash
# Database selection (production, test, memory, or custom path)
BISTOURY_DATABASE=production

# Custom database paths
DATABASE_PATH=./data/bistoury.db
TEST_DATABASE_PATH=./data/test.duckdb
```

## Use Cases

### 1. Development & Testing
- Use `memory` database for unit tests
- Use `test` database for integration testing with real data

### 2. Backtesting
- Use `test` database for historical analysis
- 4+ days of high-frequency BTC trading data
- Perfect for strategy validation

### 3. Production
- Use `production` database for live trading
- HyperLiquid API compatible schema
- Real-time data storage

## Data Quality

The test database contains production-quality data:
- **High Frequency**: Level 2 order book updates every few milliseconds  
- **Complete Coverage**: All message types (trades, l2Book, allMids)
- **Real Market Conditions**: Actual BTC trading from HyperLiquid
- **Large Dataset**: 9.4 GB of compressed data

## Next Steps

1. **Environment Setup**: Configure `.env` file for your preferred default database
2. **Backtesting**: Use the test database to validate trading strategies
3. **Development**: Switch between databases as needed for development workflows
4. **Production**: Deploy with production database for live trading

The system is now fully compatible and ready for both development and production use! 