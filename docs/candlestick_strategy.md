# Candlestick Strategy Documentation

## Overview

The Candlestick Strategy is a comprehensive trading strategy implementation that performs real-time candlestick pattern recognition across multiple timeframes (1m, 5m, 15m) to generate actionable trading signals. This document provides complete documentation for understanding, configuring, and deploying the strategy.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Pattern Recognition](#pattern-recognition)
4. [Signal Generation](#signal-generation)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [Performance Monitoring](#performance-monitoring)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Architecture Overview

The Candlestick Strategy follows a modular, multi-agent architecture designed for scalability, maintainability, and real-time performance.

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Feed     │───▶│ Candlestick     │───▶│ Signal Manager  │
│   (Collector)   │    │ Strategy Agent  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Market Data     │    │ Pattern         │    │ Trading         │
│ Buffer          │    │ Analysis        │    │ Signals         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Design Principles

- **Real-time Processing**: Sub-second pattern recognition and signal generation
- **Multi-timeframe Analysis**: Simultaneous analysis across 1m, 5m, and 15m timeframes
- **Modular Components**: Loosely coupled modules for easy testing and maintenance
- **Scalable Architecture**: Agent-based framework supporting horizontal scaling
- **Performance Monitoring**: Comprehensive metrics and health tracking

## Core Components

### 1. CandlestickStrategyAgent

The main agent class that orchestrates the entire strategy workflow.

**Key Responsibilities:**
- Market data ingestion and buffering
- Pattern recognition coordination
- Signal generation and publishing
- Performance monitoring and health checks
- Configuration management

**Integration Points:**
- Inherits from `BaseAgent` for agent framework integration
- Subscribes to market data topics via message bus
- Publishes trading signals for downstream consumption

### 2. Pattern Recognition System

Multi-layered pattern recognition system supporting both single and multi-candlestick patterns.

#### Single Candlestick Patterns
- **Doji**: Market indecision patterns (4 subtypes)
- **Hammer**: Bullish reversal with long lower shadow
- **Shooting Star**: Bearish reversal with long upper shadow
- **Spinning Top**: Market uncertainty with balanced shadows
- **Marubozu**: Strong directional patterns with minimal shadows

#### Multi-Candlestick Patterns
- **Engulfing**: Bullish/bearish trend reversal patterns
- **Harami**: Inside bar patterns indicating consolidation
- **Piercing Line**: Bullish reversal with gap penetration
- **Dark Cloud Cover**: Bearish reversal with gap coverage
- **Morning/Evening Star**: Three-candle reversal formations

### 3. Timeframe Analysis Engine

Sophisticated multi-timeframe analysis providing pattern confluence and trend alignment.

**Features:**
- Synchronizes data across multiple timeframes
- Calculates pattern confluence scores
- Identifies trend alignment and conflicts
- Generates trading recommendations based on confluence

### 4. Pattern Scoring System

Advanced scoring system combining multiple factors for pattern reliability assessment.

**Scoring Components:**
- **Technical Scoring**: Pattern geometry and textbook compliance
- **Volume Scoring**: Volume confirmation and spike analysis
- **Market Context**: Session timing, volatility, and trend alignment
- **Historical Performance**: Pattern success rates and reliability

### 5. Signal Generation Engine

Converts pattern analysis into actionable trading signals with risk management.

**Signal Features:**
- Entry point calculation with timing strategies
- Dynamic stop-loss and take-profit levels
- Confidence scoring and quality grading
- Signal expiration and lifecycle management

### 6. Narrative Generation

Human-readable analysis generation for LLM integration and decision support.

**Narrative Types:**
- Executive summaries for quick decision making
- Technical analysis explanations
- Risk assessment and position sizing guidance
- Entry and exit strategy recommendations

## Pattern Recognition

### Pattern Detection Methodology

The strategy employs a multi-step pattern recognition process:

1. **Data Preprocessing**: Candlestick data validation and normalization
2. **Pattern Scanning**: Sequential analysis using sliding window approach
3. **Confidence Calculation**: Mathematical scoring based on pattern geometry
4. **Volume Confirmation**: Volume profile analysis for pattern validation
5. **Timeframe Confluence**: Cross-timeframe pattern alignment assessment

### Pattern Confidence Scoring

Each detected pattern receives a confidence score (0-100) based on:

- **Geometric Accuracy**: How closely the pattern matches textbook definitions
- **Volume Confirmation**: Volume spikes and trends supporting the pattern
- **Market Context**: Session timing, volatility, and trend environment
- **Historical Reliability**: Pattern's historical success rate

### Quality Grading System

Patterns are assigned letter grades (A+ to F) for tradability assessment:

- **A+ (90-100)**: Exceptional patterns with high probability of success
- **A (80-89)**: Strong patterns suitable for primary signals
- **B (70-79)**: Good patterns suitable for confirmation
- **C (60-69)**: Acceptable patterns requiring additional validation
- **D (50-59)**: Weak patterns suitable only for context
- **F (0-49)**: Poor patterns filtered out from trading

## Signal Generation

### Signal Creation Process

1. **Pattern Analysis**: Multi-timeframe pattern detection and scoring
2. **Confluence Assessment**: Pattern alignment across timeframes
3. **Entry Point Calculation**: Optimal entry prices and timing
4. **Risk Management**: Stop-loss and take-profit level calculation
5. **Signal Validation**: Final quality checks and filtering
6. **Publication**: Signal broadcasting to downstream systems

### Signal Types and Timing

#### Signal Directions
- **BUY**: Bullish patterns indicating upward momentum
- **SELL**: Bearish patterns indicating downward momentum
- **STRONG_BUY**: High-confidence bullish signals
- **STRONG_SELL**: High-confidence bearish signals
- **HOLD**: Neutral or conflicting signals

#### Entry Timing Strategies
- **IMMEDIATE**: Enter at current market price
- **CONFIRMATION**: Wait for pattern confirmation
- **BREAKOUT**: Enter on breakout above/below pattern levels

### Risk Management

Automated risk management with dynamic position sizing:

- **Position Sizing**: Based on account risk tolerance and pattern confidence
- **Stop Loss**: Calculated from pattern structure and volatility
- **Take Profit**: Risk/reward ratio optimization (typically 1.5:1 to 3:1)
- **Signal Expiry**: Automatic signal invalidation after time threshold

## Configuration

### CandlestickStrategyConfig

Comprehensive configuration management for all strategy parameters.

```python
config = CandlestickStrategyConfig(
    # Market Configuration
    symbols=["BTC", "ETH", "SOL"],
    timeframes=["1m", "5m", "15m"],
    
    # Pattern Analysis
    min_confidence_threshold=0.65,
    min_pattern_strength=PatternStrength.MODERATE,
    enable_single_patterns=True,
    enable_multi_patterns=True,
    
    # Signal Generation
    signal_expiry_minutes=15,
    max_signals_per_symbol=3,
    
    # Narrative Generation
    narrative_style=NarrativeStyle.TECHNICAL,
    include_technical_details=True,
    include_risk_metrics=True,
    
    # Performance
    max_data_buffer_size=1000,
    data_cleanup_interval_seconds=300,
    health_check_interval_seconds=30
)
```

### Configuration Parameters

#### Market Data Configuration
- `symbols`: List of trading symbols to analyze
- `timeframes`: Timeframes for analysis (1m, 5m, 15m, 1h, 4h, 1d)
- `max_data_buffer_size`: Maximum candles to keep in memory

#### Pattern Analysis Configuration
- `min_confidence_threshold`: Minimum pattern confidence (0.0-1.0)
- `min_pattern_strength`: Minimum pattern strength requirement
- `enable_single_patterns`: Enable single candlestick pattern detection
- `enable_multi_patterns`: Enable multi-candlestick pattern detection

#### Signal Generation Configuration
- `signal_expiry_minutes`: Signal validity duration
- `max_signals_per_symbol`: Maximum concurrent signals per symbol
- `entry_timing_strategy`: Signal entry timing approach

#### Performance Configuration
- `data_cleanup_interval_seconds`: Buffer cleanup frequency
- `health_check_interval_seconds`: Health monitoring frequency
- `processing_timeout_seconds`: Maximum processing time per analysis

## Deployment

### Prerequisites

- Python 3.9+
- Required dependencies (see requirements.txt)
- Market data feed (HyperLiquid or compatible)
- Message bus infrastructure (Redis/RabbitMQ recommended)

### Basic Deployment

```python
import asyncio
from bistoury.agents.candlestick_strategy_agent import CandlestickStrategyAgent

async def deploy_strategy():
    # Create and configure agent
    agent = CandlestickStrategyAgent()
    
    # Start the agent
    await agent.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()

asyncio.run(deploy_strategy())
```

### Production Deployment

For production deployment, consider the following:

#### Infrastructure Requirements
- **CPU**: Minimum 2 cores, recommended 4+ cores
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: SSD recommended for database operations
- **Network**: Low-latency connection to data feed

#### Monitoring and Alerting
- Health check endpoints for external monitoring
- Performance metrics collection and visualization
- Error tracking and alerting systems
- Log aggregation and analysis

#### High Availability
- Multiple agent instances for redundancy
- Load balancing for signal distribution
- Database replication for data persistence
- Graceful failover and recovery mechanisms

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["python", "-m", "src.bistoury.agents.candlestick_strategy_agent"]
```

## Performance Monitoring

### Key Performance Metrics

#### Processing Metrics
- **Message Processing Latency**: Average time to process market data
- **Pattern Detection Rate**: Patterns detected per minute
- **Signal Generation Rate**: Trading signals generated per hour
- **Memory Usage**: Current memory consumption and trends

#### Quality Metrics
- **Pattern Confidence Distribution**: Histogram of pattern confidence scores
- **Signal Quality Grades**: Distribution of signal grades (A+ to F)
- **False Positive Rate**: Percentage of signals that don't perform as expected
- **Success Rate**: Percentage of profitable signals

#### System Health Metrics
- **Agent Uptime**: Continuous operation duration
- **Error Rate**: Errors per thousand processed messages
- **Data Quality Score**: Completeness and accuracy of market data
- **Throughput**: Messages processed per second

### Performance Targets

- **Latency**: < 100ms average message processing
- **Throughput**: > 1000 messages/second sustained
- **Accuracy**: > 65% signal success rate
- **Uptime**: > 99.9% availability
- **Memory**: < 2GB steady-state consumption

### Monitoring Tools

#### Built-in Monitoring
```python
# Get comprehensive statistics
stats = agent.get_strategy_statistics()

# Performance metrics
metrics = agent.performance_metrics
print(f"Latency: {metrics.processing_latency_ms:.2f}ms")
print(f"Signals: {metrics.signals_generated}")
print(f"Errors: {metrics.errors_count}")
```

#### Health Check API
```python
# Health check endpoint
health = await agent._health_check()
print(f"Health Score: {health.health_score:.2f}")
print(f"Status: {health.state}")
```

## Troubleshooting

### Common Issues

#### High Latency
**Symptoms**: Processing latency > 100ms
**Causes**: 
- Insufficient CPU resources
- Memory pressure
- Network latency
- Large data buffers

**Solutions**:
- Increase CPU allocation
- Reduce buffer sizes
- Optimize network connection
- Enable data cleanup

#### Low Signal Quality
**Symptoms**: Low confidence scores, poor signal performance
**Causes**:
- Incorrect configuration
- Poor market conditions
- Insufficient data quality

**Solutions**:
- Adjust confidence thresholds
- Review pattern parameters
- Validate data feed quality
- Consider market regime changes

#### Memory Leaks
**Symptoms**: Continuously increasing memory usage
**Causes**:
- Unbounded buffer growth
- Incomplete cleanup
- Reference cycles

**Solutions**:
- Enable automatic cleanup
- Reduce buffer sizes
- Monitor memory usage
- Restart agent periodically

### Debugging Tools

#### Logging Configuration
```python
import logging

# Enable debug logging
logging.getLogger('bistoury.agents.candlestick_strategy_agent').setLevel(logging.DEBUG)
```

#### Performance Profiling
```python
import cProfile

# Profile agent performance
cProfile.run('await agent.handle_message(message)')
```

## API Reference

### CandlestickStrategyAgent

Main agent class for candlestick strategy implementation.

#### Methods

##### `__init__(config: Optional[CandlestickStrategyConfig] = None)`
Initialize the candlestick strategy agent.

**Parameters:**
- `config`: Strategy configuration (optional, uses defaults if not provided)

##### `async start() -> bool`
Start the strategy agent and begin processing.

**Returns:**
- `bool`: True if started successfully, False otherwise

##### `async stop()`
Stop the strategy agent and clean up resources.

##### `async handle_message(message: Message)`
Process incoming market data or control messages.

**Parameters:**
- `message`: Message containing market data or control information

##### `get_strategy_statistics() -> Dict[str, Any]`
Get comprehensive strategy statistics and metrics.

**Returns:**
- `Dict`: Statistics including performance, configuration, and status

### CandlestickStrategyConfig

Configuration class for strategy parameters.

#### Attributes

##### Core Configuration
- `symbols: List[str]`: Trading symbols to analyze
- `timeframes: List[str]`: Timeframes for pattern analysis
- `min_confidence_threshold: float`: Minimum pattern confidence (0.0-1.0)
- `min_pattern_strength: PatternStrength`: Minimum pattern strength requirement

##### Signal Configuration
- `signal_expiry_minutes: int`: Signal validity duration
- `max_signals_per_symbol: int`: Maximum concurrent signals per symbol
- `narrative_style: NarrativeStyle`: Style for narrative generation

##### Performance Configuration
- `max_data_buffer_size: int`: Maximum candles in memory buffer
- `data_cleanup_interval_seconds: int`: Cleanup frequency
- `health_check_interval_seconds: int`: Health check frequency

### StrategyPerformanceMetrics

Performance metrics tracking class.

#### Attributes
- `signals_generated: int`: Total signals generated
- `patterns_detected: int`: Total patterns detected
- `processing_latency_ms: float`: Average processing latency
- `data_messages_received: int`: Total data messages processed
- `errors_count: int`: Total error count
- `average_confidence: float`: Average pattern confidence

---

## Examples

### Basic Usage

```python
from bistoury.agents.candlestick_strategy_agent import CandlestickStrategyAgent

# Create agent with default configuration
agent = CandlestickStrategyAgent()

# Start processing
await agent.start()

# Agent will now process market data and generate signals
# Check status
stats = agent.get_strategy_statistics()
print(f"Signals generated: {stats['performance']['signals_generated']}")

# Stop when done
await agent.stop()
```

### Custom Configuration

```python
from bistoury.agents.candlestick_strategy_agent import (
    CandlestickStrategyAgent, 
    CandlestickStrategyConfig
)
from bistoury.strategies.candlestick_models import PatternStrength
from bistoury.strategies.narrative_generator import NarrativeStyle

# Custom configuration
config = CandlestickStrategyConfig(
    symbols=["BTC", "ETH"],
    timeframes=["5m", "15m"],
    min_confidence_threshold=0.75,
    min_pattern_strength=PatternStrength.STRONG,
    narrative_style=NarrativeStyle.COMPREHENSIVE,
    max_signals_per_symbol=2
)

# Create agent with custom config
agent = CandlestickStrategyAgent(config)
await agent.start()
```

### Integration with Message Bus

```python
from bistoury.agents.messaging import MessageBus

# Set up message bus
message_bus = MessageBus()
await message_bus.start()

# Connect agent to message bus
agent = CandlestickStrategyAgent()

# Subscribe to signals
async def signal_handler(message):
    if message.type == MessageType.SIGNAL_GENERATED:
        signal_data = message.payload['signal_data']
        print(f"New signal: {signal_data['direction']} {signal_data['symbol']}")

await message_bus.subscribe("signals.candlestick.*", signal_handler)

# Start agent
await agent.start()
```

---

This documentation provides comprehensive coverage of the candlestick strategy implementation. For additional support or questions, please refer to the code repository or contact the development team. 