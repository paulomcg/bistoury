# Bistoury: LLM-Driven Cryptocurrency Trading System

An autonomous cryptocurrency trading system that leverages Large Language Models (LLMs) to execute intelligent trading strategies on the HyperLiquid exchange.

## Overview

Bistoury combines advanced pattern recognition, multi-dimensional market analysis, and natural language processing to achieve consistent profitability through automated trading. The system transforms raw market data into narrative-rich contexts that LLMs can interpret naturally, enabling nuanced trading decisions that adapt to changing market conditions.

## Key Features

- **Multi-Agent Architecture**: Specialized components for data collection, signal analysis, position management, and trade execution
- **LLM-Driven Decision Making**: Uses language models for intelligent trading decisions based on market narratives
- **Multiple Trading Strategies**: Candlestick patterns, funding rates, order flow analysis, and volume profiles
- **Comprehensive Risk Management**: Position sizing, stop-loss, take-profit, and emergency controls
- **Paper Trading**: Test strategies safely before deploying real capital
- **HyperLiquid Integration**: Native support for HyperLiquid exchange APIs and websockets

## Architecture

The system operates through four main modes:
1. **Data Collection**: Capture and store market data in original format
2. **Paper Trading Historical**: Test strategies on historical data
3. **Paper Trading Live**: Validate strategies on real-time data without risk
4. **Live Trading**: Execute actual trades with real capital

## Installation

### Method 1: Using Virtual Environment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd bistoury

# Quick setup with automatic virtual environment
./activate_venv.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install hyperliquid-python-sdk duckdb python-dotenv pydantic schedule pytest pytest-asyncio

# Copy environment template and configure API keys
cp .env.example .env
# Edit .env with your API credentials
```

### Method 2: Using Poetry

```bash
# Install dependencies using Poetry
poetry install

# Copy environment template and configure API keys
cp .env.example .env
# Edit .env with your API credentials
```

## Testing

The project includes comprehensive test suites to validate all functionality:

### Quick Test (Basic Integration)
```bash
python tests/run_tests.py
```

### Full Integration Tests
```bash
pytest tests/integration/ -v
```

### End-to-End Tests (Comprehensive)
```bash
PYTHONPATH=. python tests/e2e/test_hyperliquid_complete.py
```

### With Poetry
```bash
poetry run pytest tests/integration/ -v
```

The test suite validates:
- ✅ **API Connectivity** - HyperLiquid integration and health checks
- ✅ **Rate Limiting** - Automatic API rate management
- ✅ **WebSocket Functionality** - Real-time data subscriptions
- ✅ **Historical Data Collection** - Bulk data retrieval and storage
- ✅ **Data Collector Integration** - Live data capture and buffering
- ✅ **Error Handling & Resilience** - Automatic reconnection and recovery
- ✅ **Performance & Optimization** - Response times and throughput

## Configuration

Configure your API keys in the `.env` file:
- HyperLiquid API credentials
- LLM provider API keys (OpenAI, Claude, etc.)

## Usage

```bash
# Start data collection
bistoury collect --pairs BTC,ETH --mode continuous

# Run paper trading
bistoury paper-trade --mode live --duration 24h

# Start live trading (requires confirmation)
bistoury trade --confirm --risk-limit 1000

# Monitor system status
bistoury status
```

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run linting and formatting
poetry run black .
poetry run ruff check .
poetry run mypy src/

# Run all checks
poetry run pre-commit run --all-files
```

## Current Status

- ✅ **Task 1**: Project structure and initial database design
- ✅ **Task 2**: Core market data schema for HyperLiquid
- ✅ **Task 3**: Complete HyperLiquid API integration with enterprise features
- ⏳ **Task 4-25**: [View complete roadmap in PRD-LLM.txt]

**Task 3 Achievement**: 100% complete with 22/22 integration tests passing, enterprise-grade rate limiting, WebSocket support, comprehensive error handling, and production-ready database integration.

## Performance Target

- **Profitability**: Target 1% portfolio growth per 24 hours
- **Uptime**: 99.9% availability excluding maintenance
- **Latency**: <100ms from signal to order execution
- **Resource Efficiency**: Runs on modest hardware (4 cores, 8GB RAM)

## Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Use at your own risk and never trade with money you cannot afford to lose.

## License

[Add appropriate license information] 