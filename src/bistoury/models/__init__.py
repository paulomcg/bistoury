"""
Bistoury Data Models Package

This package contains all Pydantic models for market data validation, 
API responses, trading operations, and database entities.

Core model categories:
- market_data: Candlesticks, tickers, symbol information, timeframes
- orderbook: Order book levels, snapshots, and analytics
- trades: Individual trade executions and aggregated analytics
- api_responses: HyperLiquid API response models
- websocket: WebSocket message models for real-time data
- trading: Positions, orders, portfolio state (coming soon)
- signals: Trading signals and strategy outputs (coming soon)
- database: Database-optimized models (coming soon)
"""

from .market_data import (
    CandlestickData,
    Ticker,
    SymbolInfo,
    MarketData,
    Timeframe,
    PriceLevel,
)

from .orderbook import (
    OrderBookLevel,
    OrderBook,
    OrderBookSnapshot,
    OrderBookDelta,
)

from .trades import (
    Trade,
    TradeAggregation,
    TradeAnalytics,
)

from .api_responses import (
    ErrorResponse,
    ResponseMetadata,
    MetadataResponse,
    AllMidsResponse,
    PositionInfo,
    UserInfoResponse,
    CandleHistoryResponse,
    TradeHistoryResponse,
    OrderBookResponse,
    ResponseWrapper,
)

from .websocket import (
    MessageType,
    SubscriptionChannel,
    WSMessage,
    PriceUpdateMessage,
    TradeUpdateMessage,
    OrderBookUpdateMessage,
    CandleUpdateMessage,
    SubscriptionMessage,
    MessageRouter,
)

__all__ = [
    # Market Data Models
    "CandlestickData",
    "Ticker", 
    "SymbolInfo",
    "MarketData",
    "Timeframe",
    "PriceLevel",
    
    # Order Book Models
    "OrderBookLevel",
    "OrderBook", 
    "OrderBookSnapshot",
    "OrderBookDelta",
    
    # Trade Models
    "Trade",
    "TradeAggregation",
    "TradeAnalytics",
    
    # API Response Models
    "ErrorResponse",
    "ResponseMetadata",
    "MetadataResponse",
    "AllMidsResponse",
    "PositionInfo",
    "UserInfoResponse",
    "CandleHistoryResponse",
    "TradeHistoryResponse",
    "OrderBookResponse",
    "ResponseWrapper",
    
    # WebSocket Models
    "MessageType",
    "SubscriptionChannel",
    "WSMessage",
    "PriceUpdateMessage",
    "TradeUpdateMessage",
    "OrderBookUpdateMessage",
    "CandleUpdateMessage",
    "SubscriptionMessage",
    "MessageRouter",
]

__version__ = "1.0.0" 