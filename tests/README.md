# Bistoury Test Suite

This directory contains the complete test suite for the Bistoury cryptocurrency trading system.

## Test Structure

```
tests/
├── unit/                    # Unit tests (no external dependencies)
│   ├── test_config.py      # Configuration loading tests
│   ├── test_database.py    # Database functionality tests
│   └── test_database_schema_compression_backup.py
├── integration/            # Integration tests (require API access)
│   └── test_hyperliquid_integration.py  # ✅ 22 tests - ALL PASSING
├── e2e/                   # End-to-end tests
│   └── test_hyperliquid_complete.py  # Comprehensive system test
├── conftest.py            # Pytest fixtures and configuration
├── run_tests.py           # Simple test runner (no pytest needed)
└── README.md              # This file
```

## ✅ Test Status

**Integration Tests**: 22/22 PASSING  
- All HyperLiquid API integrations working
- Database schema properly aligned
- WebSocket connections functional
- Historical data collection working
- Rate limiting and optimization verified

## Running Tests

### Option 1: Using Poetry (Recommended)

If you have Poetry installed:

```bash
# Install dependencies
poetry install

# Run all tests with coverage
poetry run pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
poetry run pytest tests/unit/ -v              # Unit tests only
poetry run pytest tests/integration/ -v       # Integration tests only
poetry run pytest tests/e2e/ -v              # End-to-end tests only

# Run comprehensive HyperLiquid test
poetry run python tests/e2e/test_hyperliquid_complete.py
```

### Option 2: Using pip/python directly

If Poetry is not available:

```bash
# Install pytest first
pip install pytest pytest-asyncio pytest-cov

# Run tests
python -m pytest tests/ -v

# Or use the simple test runner (no pytest needed)
python tests/run_tests.py
```

### Option 3: Simple Test Runner (No Dependencies)

For basic integration testing without installing pytest:

```bash
python tests/run_tests.py
```

This runs core integration tests to verify:
- ✅ HyperLiquid API connectivity
- ✅ Market data retrieval
- ✅ WebSocket connections
- ✅ Basic functionality

## Test Categories Explained

### Unit Tests (`tests/unit/`)
- **No external dependencies** (mocked APIs)
- Test individual components in isolation
- Fast execution (~10 seconds)
- Always should pass regardless of network/API status

### Integration Tests (`tests/integration/`)
- **Requires HyperLiquid API access**
- Test real API interactions
- Moderate execution time (~30 seconds)
- May fail if API is down or rate limits hit

### End-to-End Tests (`tests/e2e/`)
- **Full system validation**
- Tests complete workflows
- Comprehensive performance benchmarks
- Longer execution time (~2-5 minutes)

## Prerequisites

### Environment Setup

1. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Configure environment variables**:
   ```bash
   # Required for integration tests
   HYPERLIQUID_TESTNET=true
   
   # Optional: Add API keys for full functionality
   # (Tests will use testnet by default for safety)
   ```

### Database Setup

Tests use temporary databases by default, but you can also test against your main database:

```bash
# Tests will create temporary databases automatically
# No manual database setup required
```

## Test Results Interpretation

### Expected Results:

#### ✅ **Unit Tests** (should always pass):
- Configuration loading and validation
- Database schema management  
- Data compression and backup systems
- Core business logic components

#### ✅ **Integration Tests** (network dependent):
- HyperLiquid API connectivity
- Real-time data collection
- WebSocket subscriptions
- Historical data retrieval
- Error handling and resilience

#### ✅ **End-to-End Tests** (full validation):
- 9 comprehensive test categories
- Performance benchmarks (<5s API calls)
- Rate limiting and optimization
- Connection health monitoring
- Bulk data collection capabilities

### Common Issues & Solutions:

#### 🚨 **"ModuleNotFoundError: No module named 'pytest'"**
```bash
# Solution 1: Install pytest
pip install pytest pytest-asyncio

# Solution 2: Use simple runner
python tests/run_tests.py
```

#### 🚨 **"Failed to connect to HyperLiquid"**
```bash
# Check internet connection and API status
# Verify .env configuration
# Try running unit tests first: pytest tests/unit/ -v
```

#### 🚨 **"Missing test_database fixture"**
This was fixed in the recent updates. Make sure you have the latest `conftest.py`.

#### 🚨 **Import path errors**
All import paths have been updated to use `src.bistoury.*` format.

## Recent Fixes Applied

- ✅ **Fixed missing `test_database` fixture** in `conftest.py`
- ✅ **Corrected import paths** to use `src.` prefix
- ✅ **Fixed `test_connection_info`** to match actual API response fields
- ✅ **Moved comprehensive test** from `scripts/` to `tests/e2e/`
- ✅ **Added simple test runner** for environments without pytest
- ✅ **Enhanced database fixture** with proper cleanup

## Performance Expectations

| Test Type | Duration | Requirements |
|-----------|----------|--------------|
| Unit Tests | ~10 seconds | None |
| Integration Tests | ~30 seconds | API access |
| End-to-End Tests | ~2-5 minutes | API + Database |

## CI/CD Integration

For continuous integration, run:

```bash
# Full test suite with linting
poetry run black --check src/ tests/
poetry run ruff check src/ tests/  
poetry run mypy src/
poetry run pytest tests/ --cov=src --cov-report=xml
```

## Getting Help

If tests are failing:

1. **Start with unit tests**: `pytest tests/unit/ -v`
2. **Check environment setup**: Verify `.env` file exists
3. **Test basic connectivity**: `python tests/run_tests.py`
4. **Review logs**: Check `logs/` directory for detailed error information
5. **Check network**: Ensure internet access to HyperLiquid APIs

For more help, check the main project README or documentation. 