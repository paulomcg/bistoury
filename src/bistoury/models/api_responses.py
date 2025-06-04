"""
HyperLiquid API Response Models

This module contains Pydantic models for all HyperLiquid API responses:
- MetadataResponse: Exchange information and metadata
- AllMidsResponse: Real-time price snapshot responses
- UserInfoResponse: Account and user data responses
- HistoricalResponse: Historical data bulk responses
- ErrorResponse: Error handling and status codes
- ResponseWrapper: Pagination and response metadata

All models are designed for direct API compatibility and include
comprehensive validation for HyperLiquid's specific response formats.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .market_data import SymbolInfo, CandlestickData, Timeframe
from .orderbook import OrderBook
from .trades import Trade


class ErrorResponse(BaseModel):
    """
    Standard error response from HyperLiquid API.
    
    Handles all error scenarios including rate limiting, authentication
    failures, invalid parameters, and internal server errors.
    """
    
    error: str = Field(
        ...,
        description="Error message or error type"
    )
    message: Optional[str] = Field(
        None,
        description="Detailed error description"
    )
    code: Optional[int] = Field(
        None,
        description="HTTP status code or error code",
        ge=100,
        le=599
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for error tracking"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )
    
    @property
    def is_rate_limit_error(self) -> bool:
        """Check if error is due to rate limiting."""
        return (
            self.code == 429 or
            "rate limit" in self.error.lower() or
            "too many requests" in (self.message or "").lower()
        )
    
    @property
    def is_auth_error(self) -> bool:
        """Check if error is due to authentication."""
        return (
            self.code in [401, 403] or
            "auth" in self.error.lower() or
            "unauthorized" in (self.message or "").lower()
        )
    
    @property
    def is_server_error(self) -> bool:
        """Check if error is a server-side issue."""
        return self.code is not None and 500 <= self.code < 600


class ResponseMetadata(BaseModel):
    """
    Common metadata for API responses.
    
    Provides pagination, timing, and request tracking information
    that's common across different API endpoints.
    """
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Unique request identifier"
    )
    total_count: Optional[int] = Field(
        None,
        description="Total number of items available",
        ge=0
    )
    page: Optional[int] = Field(
        None,
        description="Current page number",
        ge=1
    )
    page_size: Optional[int] = Field(
        None,
        description="Number of items per page",
        ge=1,
        le=10000
    )
    has_next: Optional[bool] = Field(
        None,
        description="Whether more pages are available"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )


class MetadataResponse(BaseModel):
    """
    Response from HyperLiquid's metadata endpoint.
    
    Contains exchange information including available symbols,
    trading specifications, and system status.
    """
    
    universe: List[SymbolInfo] = Field(
        ...,
        description="List of all available trading symbols"
    )
    exchange_status: Optional[str] = Field(
        None,
        description="Current exchange operational status"
    )
    server_time: Optional[datetime] = Field(
        None,
        description="Exchange server time"
    )
    maintenance_mode: bool = Field(
        default=False,
        description="Whether exchange is in maintenance mode"
    )
    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'MetadataResponse':
        """
        Create MetadataResponse from HyperLiquid metadata API response.
        
        Expected format: {
            'universe': [
                {
                    'name': 'BTC',
                    'szDecimals': 3,
                    'maxLeverage': 50,
                    'onlyIsolated': false
                },
                ...
            ]
        }
        """
        universe = []
        
        if 'universe' in data:
            for symbol_data in data['universe']:
                symbol_info = SymbolInfo.from_hyperliquid(symbol_data)
                universe.append(symbol_info)
        
        return cls(
            universe=universe,
            exchange_status=data.get('status'),
            server_time=data.get('time'),
            maintenance_mode=data.get('maintenanceMode', False)
        )
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information by symbol name."""
        symbol = symbol.upper()
        for symbol_info in self.universe:
            if symbol_info.symbol == symbol:
                return symbol_info
        return None
    
    @property
    def active_symbols(self) -> List[SymbolInfo]:
        """Get only actively trading symbols."""
        return [s for s in self.universe if s.is_active]


class AllMidsResponse(BaseModel):
    """
    Response from HyperLiquid's all mids endpoint.
    
    Contains current mid prices for all trading symbols,
    providing a snapshot of the entire market.
    """
    
    mids: Dict[str, Decimal] = Field(
        ...,
        description="Symbol to mid price mapping"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Snapshot timestamp"
    )
    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @field_validator('mids', mode='before')
    @classmethod
    def validate_mids(cls, v) -> Dict[str, Decimal]:
        """Convert price strings to Decimal values."""
        if isinstance(v, dict):
            return {str(k): Decimal(str(price)) for k, price in v.items()}
        return v
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'AllMidsResponse':
        """
        Create AllMidsResponse from HyperLiquid allMids API response.
        
        Expected format: {
            'BTC': '50000.5',
            'ETH': '3000.25',
            ...
        }
        """
        return cls(mids=data)
    
    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get mid price for a specific symbol."""
        return self.mids.get(symbol.upper())
    
    @property
    def symbol_count(self) -> int:
        """Get number of symbols in response."""
        return len(self.mids)


class PositionInfo(BaseModel):
    """Individual position information from user data."""
    
    coin: str = Field(
        ...,
        description="Position symbol"
    )
    szi: Decimal = Field(
        ...,
        description="Position size"
    )
    entryPx: Optional[Decimal] = Field(
        None,
        description="Entry price"
    )
    positionValue: Optional[Decimal] = Field(
        None,
        description="Current position value"
    )
    unrealizedPnl: Optional[Decimal] = Field(
        None,
        description="Unrealized PnL"
    )
    leverage: Optional[Decimal] = Field(
        None,
        description="Position leverage"
    )
    liquidationPx: Optional[Decimal] = Field(
        None,
        description="Liquidation price"
    )
    
    @field_validator('szi', 'entryPx', 'positionValue', 'unrealizedPnl', 'leverage', 'liquidationPx', mode='before')
    @classmethod
    def validate_decimal_fields(cls, v) -> Optional[Decimal]:
        """Convert string/float to Decimal."""
        if v is None or v == "":
            return None
        return Decimal(str(v))


class UserInfoResponse(BaseModel):
    """
    Response from HyperLiquid's user info endpoint.
    
    Contains account information including positions, balances,
    and margin requirements.
    """
    
    marginSummary: Dict[str, Any] = Field(
        ...,
        description="Margin and balance summary"
    )
    assetPositions: List[PositionInfo] = Field(
        default_factory=list,
        description="List of open positions"
    )
    crossMarginSummary: Optional[Dict[str, Any]] = Field(
        None,
        description="Cross margin summary"
    )
    crossPositions: Optional[List[PositionInfo]] = Field(
        None,
        description="Cross margin positions"
    )
    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'UserInfoResponse':
        """
        Create UserInfoResponse from HyperLiquid user info API response.
        
        Expected format: {
            'marginSummary': {
                'accountValue': '1000.0',
                'totalNtlPos': '500.0',
                ...
            },
            'assetPositions': [
                {
                    'position': {
                        'coin': 'BTC',
                        'szi': '0.1',
                        'entryPx': '50000',
                        ...
                    },
                    ...
                },
                ...
            ]
        }
        """
        asset_positions = []
        
        if 'assetPositions' in data:
            for pos_data in data['assetPositions']:
                if 'position' in pos_data:
                    position = PositionInfo(**pos_data['position'])
                    asset_positions.append(position)
        
        cross_positions = None
        if 'crossPositions' in data:
            cross_positions = []
            for pos_data in data['crossPositions']:
                if 'position' in pos_data:
                    position = PositionInfo(**pos_data['position'])
                    cross_positions.append(position)
        
        return cls(
            marginSummary=data.get('marginSummary', {}),
            assetPositions=asset_positions,
            crossMarginSummary=data.get('crossMarginSummary'),
            crossPositions=cross_positions
        )
    
    @property
    def account_value(self) -> Optional[Decimal]:
        """Get total account value."""
        if 'accountValue' in self.marginSummary:
            return Decimal(str(self.marginSummary['accountValue']))
        return None
    
    @property
    def total_margin_used(self) -> Optional[Decimal]:
        """Get total margin used."""
        if 'totalMarginUsed' in self.marginSummary:
            return Decimal(str(self.marginSummary['totalMarginUsed']))
        return None
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for a specific symbol."""
        symbol = symbol.upper()
        for position in self.assetPositions:
            if position.coin == symbol:
                return position
        return None


class CandleHistoryResponse(BaseModel):
    """
    Response from HyperLiquid's candle history endpoint.
    
    Contains historical candlestick data for a specific symbol
    and timeframe with pagination support.
    """
    
    candles: List[CandlestickData] = Field(
        ...,
        description="List of candlestick data"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    timeframe: Timeframe = Field(
        ...,
        description="Candlestick timeframe"
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Query start time"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="Query end time"
    )
    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: List[Dict[str, Any]], symbol: str, timeframe: Timeframe) -> 'CandleHistoryResponse':
        """
        Create CandleHistoryResponse from HyperLiquid candles API response.
        
        Expected format: [
            {
                't': 1705316200000,
                's': 'BTC',
                'o': '50000.0',
                'h': '50100.0',
                'l': '49900.0',
                'c': '50050.0',
                'v': '123.45',
                'n': 1500
            },
            ...
        ]
        """
        candles = []
        
        for candle_data in data:
            candle = CandlestickData.from_hyperliquid(candle_data, timeframe)
            candles.append(candle)
        
        # Sort by timestamp
        candles.sort(key=lambda c: c.timestamp)
        
        start_time = candles[0].timestamp if candles else None
        end_time = candles[-1].timestamp if candles else None
        
        return cls(
            candles=candles,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
    
    @property
    def candle_count(self) -> int:
        """Get number of candles in response."""
        return len(self.candles)
    
    @property
    def time_range_hours(self) -> Optional[float]:
        """Get time range covered in hours."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() / 3600
        return None


class TradeHistoryResponse(BaseModel):
    """
    Response from HyperLiquid's trade history endpoint.
    
    Contains historical trade data for a specific symbol
    with pagination and filtering support.
    """
    
    trades: List[Trade] = Field(
        ...,
        description="List of trade executions"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Query start time"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="Query end time"
    )
    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: List[Dict[str, Any]], symbol: str) -> 'TradeHistoryResponse':
        """
        Create TradeHistoryResponse from HyperLiquid trades API response.
        
        Expected format: [
            {
                'time': 1705316200000,
                'px': '50000.5',
                'sz': '1.5',
                'side': 'B',
                'tid': 'trade_123'
            },
            ...
        ]
        """
        trades = []
        
        for trade_data in data:
            trade = Trade.from_hyperliquid(trade_data, symbol)
            trades.append(trade)
        
        # Sort by timestamp
        trades.sort(key=lambda t: t.timestamp)
        
        start_time = trades[0].timestamp if trades else None
        end_time = trades[-1].timestamp if trades else None
        
        return cls(
            trades=trades,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
    
    @property
    def trade_count(self) -> int:
        """Get number of trades in response."""
        return len(self.trades)
    
    @property
    def total_volume(self) -> Decimal:
        """Get total trading volume."""
        return sum(trade.quantity for trade in self.trades)
    
    @property
    def buy_trades(self) -> List[Trade]:
        """Get only buy trades."""
        return [trade for trade in self.trades if trade.side == "buy"]
    
    @property
    def sell_trades(self) -> List[Trade]:
        """Get only sell trades."""
        return [trade for trade in self.trades if trade.side == "sell"]


class OrderBookResponse(BaseModel):
    """
    Response from HyperLiquid's order book endpoint.
    
    Contains Level 2 order book data with bid/ask levels
    and market depth information.
    """
    
    order_book: OrderBook = Field(
        ...,
        description="Complete order book with bids and asks"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Order book timestamp"
    )
    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any], symbol: str) -> 'OrderBookResponse':
        """
        Create OrderBookResponse from HyperLiquid l2Book API response.
        
        Expected format: {
            'levels': [
                [
                    ['50000.0', '1.5', 3],  # [price, size, count]
                    ['49995.0', '2.0', 5],
                    ...
                ],
                [
                    ['50005.0', '1.2', 2],
                    ['50010.0', '0.8', 1],
                    ...
                ]
            ]
        }
        """
        order_book = OrderBook.from_hyperliquid(data)
        
        return cls(
            order_book=order_book,
            symbol=symbol
        )
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Get bid-ask spread."""
        return self.order_book.spread
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Get mid price."""
        return self.order_book.mid_price


class ResponseWrapper(BaseModel):
    """
    Generic response wrapper for HyperLiquid API responses.
    
    Provides unified structure for handling successful responses,
    errors, and pagination across all API endpoints.
    """
    
    success: bool = Field(
        ...,
        description="Whether the request was successful"
    )
    data: Optional[Union[
        MetadataResponse,
        AllMidsResponse,
        UserInfoResponse,
        CandleHistoryResponse,
        TradeHistoryResponse,
        OrderBookResponse,
        Dict[str, Any]
    ]] = Field(
        None,
        description="Response data"
    )
    error: Optional[ErrorResponse] = Field(
        None,
        description="Error information if request failed"
    )
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Response metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @model_validator(mode='after')
    def validate_response_data(self):
        """Validate that successful responses have data and failed responses have errors."""
        if self.success and self.data is None:
            raise ValueError("Successful responses must include data")
        
        if not self.success and self.error is None:
            raise ValueError("Failed responses must include error information")
        
        return self
    
    @classmethod
    def success_response(cls, data: Any, metadata: Optional[ResponseMetadata] = None) -> 'ResponseWrapper':
        """Create a successful response wrapper."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or ResponseMetadata()
        )
    
    @classmethod
    def error_response(cls, error: Union[str, ErrorResponse], metadata: Optional[ResponseMetadata] = None) -> 'ResponseWrapper':
        """Create an error response wrapper."""
        if isinstance(error, str):
            error = ErrorResponse(error=error)
        
        return cls(
            success=False,
            error=error,
            metadata=metadata or ResponseMetadata()
        )
    
    def raise_for_status(self):
        """Raise exception if response indicates an error."""
        if not self.success and self.error:
            raise Exception(f"API Error: {self.error.error} - {self.error.message}") 