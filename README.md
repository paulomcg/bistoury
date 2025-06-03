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

```bash
# Clone the repository
git clone <repository-url>
cd bistoury

# Install dependencies using Poetry
poetry install

# Copy environment template and configure API keys
cp .env.example .env
# Edit .env with your API credentials
```

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

## Performance Target

- **Profitability**: Target 1% portfolio growth per 24 hours
- **Uptime**: 99.9% availability excluding maintenance
- **Latency**: <100ms from signal to order execution
- **Resource Efficiency**: Runs on modest hardware (4 cores, 8GB RAM)

## Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Use at your own risk and never trade with money you cannot afford to lose.

## License

[Add appropriate license information] 