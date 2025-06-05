"""
Bistoury Models Package

Comprehensive data models for the Bistoury cryptocurrency trading system.
Includes market data, trading operations, signals, and database optimization.
"""

from .market_data import (
    CandlestickData,
    Ticker,
    SymbolInfo,
    MarketData,
    PriceLevel,
    Timeframe
)

from .signals import (
    TradingSignal,
    CandlestickPattern,
    AnalysisContext,
    SignalAggregation,
    SignalDirection,
    SignalType,
    ConfidenceLevel,
    PatternType
)

from .strategies import (
    StrategyOutput,
    StrategyPerformance,
    SignalPerformance,
    StrategyMetadata,
    BacktestResult
)

from .trading import (
    Position,
    Order,
    TradeExecution,
    PortfolioState,
    RiskParameters,
    PositionSide,
    OrderType,
    OrderStatus,
    OrderSide,
    TimeInForce,
)

from .database import (
    DatabaseModel,
    DBCandlestickData,
    DBTradeData,
    DBOrderBookSnapshot,
    DBFundingRateData,
    DBTradingSignal,
    DBPosition,
    DBBatchOperation,
    DBArchiveRecord
)

from .serialization import (
    SerializationFormat,
    CompressionLevel,
    SerializationMetrics,
    DatabaseSerializer,
    ModelConverter,
    BatchProcessor,
    DataIntegrityValidator,
)

__all__ = [
    # Market Data Models
    "CandlestickData",
    "Ticker",
    "SymbolInfo",
    "MarketData",
    "PriceLevel",
    "Timeframe",
    
    # Signal and Strategy Models
    "TradingSignal",
    "CandlestickPattern",
    "AnalysisContext",
    "SignalAggregation",
    "SignalDirection",
    "SignalType",
    "ConfidenceLevel",
    "PatternType",
    "StrategyOutput",
    "StrategyPerformance",
    "SignalPerformance",
    "StrategyMetadata",
    "BacktestResult",
    
    # Trading Models
    "Position",
    "Order",
    "TradeExecution",
    "PortfolioState",
    "RiskParameters",
    "PositionSide",
    "OrderType",
    "OrderStatus",
    "OrderSide",
    "TimeInForce",
    
    # Database Models
    "DatabaseModel",
    "DBCandlestickData",
    "DBTradeData",
    "DBOrderBookSnapshot",
    "DBFundingRateData",
    "DBTradingSignal",
    "DBPosition",
    "DBBatchOperation",
    "DBArchiveRecord",
    
    # Serialization and Database Utilities
    "SerializationFormat",
    "CompressionLevel",
    "SerializationMetrics",
    "DatabaseSerializer",
    "ModelConverter",
    "BatchProcessor",
    "DataIntegrityValidator",
]

__version__ = "1.0.0" 