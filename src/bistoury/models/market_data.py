"""
Core Market Data Models

This module contains Pydantic models for fundamental market data types:
- CandlestickData: OHLCV data with comprehensive validation
- Ticker: Current price and 24h statistics
- SymbolInfo: Trading pair metadata and specifications
- MarketData: Base class with common fields
- Timeframe: Enumeration of supported timeframes
- PriceLevel: Individual price/quantity levels

All models are designed for HyperLiquid API compatibility and include
robust validation for cryptocurrency trading requirements.
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import re


class Timeframe(str, Enum):
    """Supported candlestick timeframes matching HyperLiquid API."""
    
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    
    @property
    def seconds(self) -> int:
        """Convert timeframe to seconds."""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return mapping[self.value]
    
    @property
    def milliseconds(self) -> int:
        """Convert timeframe to milliseconds."""
        return self.seconds * 1000


class MarketData(BaseModel):
    """
    Base class for all market data models with common fields and validation.
    
    Provides consistent timestamp handling, symbol validation, and
    serialization helpers for database storage.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol (e.g., 'BTC', 'ETH')",
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9]+$"
    )
    timestamp: datetime = Field(
        ...,
        description="Data timestamp in UTC timezone"
    )
    time_ms: Optional[int] = Field(
        None,
        description="Timestamp in milliseconds (HyperLiquid format)",
        ge=0
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def validate_timestamp(cls, v) -> datetime:
        """Ensure timestamp is timezone-aware UTC."""
        if isinstance(v, str):
            # Parse ISO format string
            if v.endswith('Z'):
                v = v[:-1] + '+00:00'
            dt = datetime.fromisoformat(v)
        elif isinstance(v, (int, float)):
            # Convert from milliseconds timestamp
            dt = datetime.fromtimestamp(v / 1000, tz=timezone.utc)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Invalid timestamp format: {type(v)}")
            
        # Ensure UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)
            
        return dt
    
    @model_validator(mode='after')
    def validate_time_ms(self):
        """Auto-generate time_ms from timestamp if not provided."""
        if self.time_ms is None:
            if self.timestamp:
                self.time_ms = int(self.timestamp.timestamp() * 1000)
            else:
                self.time_ms = 0
        return self
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v) -> str:
        """Validate and normalize symbol format."""
        v = v.upper().strip()
        
        # Check for valid characters
        if not re.match(r'^[A-Z0-9]+$', v):
            raise ValueError(f"Symbol must contain only uppercase letters and numbers: {v}")
        
        # Common symbol validation
        if len(v) < 1 or len(v) > 20:
            raise ValueError(f"Symbol length must be 1-20 characters: {v}")
            
        return v


class PriceLevel(BaseModel):
    """
    Individual price level with quantity information.
    
    Used for order book levels, price points, and volume analysis.
    Includes validation for positive prices and quantities.
    """
    
    price: Decimal = Field(
        ...,
        description="Price level as decimal",
        gt=0,
        decimal_places=8
    )
    quantity: Decimal = Field(
        ...,
        description="Quantity at this price level",
        ge=0,
        decimal_places=8
    )
    count: Optional[int] = Field(
        None,
        description="Number of orders at this level",
        ge=0
    )
    
    @field_validator('price', 'quantity', mode='before')
    @classmethod
    def validate_decimal_fields(cls, v) -> Decimal:
        """Convert string/float to Decimal with proper precision."""
        if isinstance(v, str):
            # Handle scientific notation and remove extra whitespace
            v = v.strip()
            if 'e' in v.lower() or 'E' in v:
                v = f"{float(v):.8f}"
                
        decimal_val = Decimal(str(v))
        
        # Round to 8 decimal places for cryptocurrency precision
        return decimal_val.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
    
    def __lt__(self, other: 'PriceLevel') -> bool:
        """Enable sorting by price."""
        return self.price < other.price
    
    def __eq__(self, other: 'PriceLevel') -> bool:
        """Enable equality comparison by price."""
        return self.price == other.price


class CandlestickData(MarketData):
    """
    OHLCV candlestick data model with comprehensive validation.
    
    Matches HyperLiquid API candlestick format:
    {'t': timestamp_ms, 's': symbol, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume}
    
    Includes validation for:
    - Price relationships (high >= low, high >= open/close, low <= open/close)
    - Positive prices and volumes
    - Proper decimal precision for cryptocurrency trading
    """
    
    timeframe: Timeframe = Field(
        ...,
        description="Candlestick timeframe"
    )
    open: Decimal = Field(
        ...,
        description="Opening price",
        gt=0,
        decimal_places=8
    )
    high: Decimal = Field(
        ...,
        description="Highest price",
        gt=0,
        decimal_places=8
    )
    low: Decimal = Field(
        ...,
        description="Lowest price", 
        gt=0,
        decimal_places=8
    )
    close: Decimal = Field(
        ...,
        description="Closing price",
        gt=0,
        decimal_places=8
    )
    volume: Decimal = Field(
        ...,
        description="Trading volume",
        ge=0,
        decimal_places=8
    )
    trade_count: Optional[int] = Field(
        None,
        description="Number of trades in this candle",
        ge=0
    )
    vwap: Optional[Decimal] = Field(
        None,
        description="Volume weighted average price",
        gt=0,
        decimal_places=8
    )
    
    @field_validator('open', 'high', 'low', 'close', 'volume', 'vwap', mode='before')
    @classmethod
    def validate_price_fields(cls, v) -> Decimal:
        """Convert and validate price/volume fields."""
        if v is None:
            return v
            
        if isinstance(v, str):
            v = v.strip()
            if 'e' in v.lower() or 'E' in v:
                v = f"{float(v):.8f}"
                
        decimal_val = Decimal(str(v))
        return decimal_val.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
    
    @model_validator(mode='after')
    def validate_ohlc_relationships(self):
        """Validate OHLC price relationships."""
        if not all([self.open, self.high, self.low, self.close]):
            return self
            
        # High must be highest price
        if self.high < max(self.open, self.close):
            raise ValueError(f"High price {self.high} must be >= max(open, close)")
        if self.high < self.low:
            raise ValueError(f"High price {self.high} must be >= low price {self.low}")
            
        # Low must be lowest price
        if self.low > min(self.open, self.close):
            raise ValueError(f"Low price {self.low} must be <= min(open, close)")
            
        return self
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any], timeframe: Timeframe) -> 'CandlestickData':
        """
        Create CandlestickData from HyperLiquid API response.
        
        Expected format: {'t': timestamp_ms, 's': symbol, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume}
        """
        return cls(
            symbol=data['s'],
            timestamp=datetime.fromtimestamp(int(data['t']) / 1000, tz=timezone.utc),
            time_ms=int(data['t']),
            timeframe=timeframe,
            open=data['o'],
            high=data['h'],
            low=data['l'],
            close=data['c'],
            volume=data['v'],
            trade_count=data.get('n'),  # Trade count if available
        )
    
    def to_hyperliquid(self) -> Dict[str, Any]:
        """Convert to HyperLiquid API format."""
        result = {
            't': self.time_ms,
            's': self.symbol,
            'o': str(self.open),
            'h': str(self.high),
            'l': str(self.low),
            'c': str(self.close),
            'v': str(self.volume),
        }
        
        if self.trade_count is not None:
            result['n'] = self.trade_count
            
        return result
    
    @property
    def body_size(self) -> Decimal:
        """Calculate candlestick body size (absolute difference between open and close)."""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> Decimal:
        """Calculate upper shadow length."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> Decimal:
        """Calculate lower shadow length."""
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open)."""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """Check if candle is a doji (open ≈ close)."""
        return abs(self.close - self.open) <= (self.high - self.low) * Decimal('0.01')


class Ticker(MarketData):
    """
    Current price and 24-hour statistics for a trading symbol.
    
    Provides real-time price information with 24-hour statistics
    including volume, price changes, and trading activity.
    """
    
    last_price: Decimal = Field(
        ...,
        description="Last traded price",
        gt=0,
        decimal_places=8
    )
    bid_price: Optional[Decimal] = Field(
        None,
        description="Current best bid price",
        gt=0,
        decimal_places=8
    )
    ask_price: Optional[Decimal] = Field(
        None,
        description="Current best ask price", 
        gt=0,
        decimal_places=8
    )
    volume_24h: Decimal = Field(
        ...,
        description="24-hour trading volume",
        ge=0,
        decimal_places=8
    )
    price_change_24h: Decimal = Field(
        ...,
        description="24-hour price change",
        decimal_places=8
    )
    price_change_pct_24h: Decimal = Field(
        ...,
        description="24-hour price change percentage",
        decimal_places=4
    )
    high_24h: Decimal = Field(
        ...,
        description="24-hour highest price",
        gt=0,
        decimal_places=8
    )
    low_24h: Decimal = Field(
        ...,
        description="24-hour lowest price",
        gt=0,
        decimal_places=8
    )
    open_24h: Optional[Decimal] = Field(
        None,
        description="24-hour opening price",
        gt=0,
        decimal_places=8
    )
    trade_count_24h: Optional[int] = Field(
        None,
        description="24-hour trade count",
        ge=0
    )
    
    @field_validator('last_price', 'bid_price', 'ask_price', 'volume_24h', 'price_change_24h', 
                     'price_change_pct_24h', 'high_24h', 'low_24h', 'open_24h', mode='before')
    @classmethod
    def validate_price_fields(cls, v) -> Decimal:
        """Convert and validate price/volume fields."""
        if v is None:
            return v
            
        if isinstance(v, str):
            v = v.strip()
            if 'e' in v.lower() or 'E' in v:
                v = f"{float(v):.8f}"
                
        decimal_val = Decimal(str(v))
        return decimal_val.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
    
    @model_validator(mode='after')
    def validate_bid_ask_spread(self):
        """Validate bid/ask price relationships."""
        if self.bid_price is not None and self.ask_price is not None:
            if self.bid_price >= self.ask_price:
                raise ValueError(f"Bid price {self.bid_price} must be less than ask price {self.ask_price}")
                
        return self
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask."""
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid/ask spread."""
        if self.bid_price is not None and self.ask_price is not None:
            return self.ask_price - self.bid_price
        return None
    
    @property
    def spread_pct(self) -> Optional[Decimal]:
        """Calculate bid/ask spread as percentage of mid price."""
        spread = self.spread
        mid = self.mid_price
        
        if spread is not None and mid is not None and mid > 0:
            return (spread / mid) * 100
        return None


class SymbolInfo(BaseModel):
    """
    Trading symbol metadata and specifications.
    
    Contains comprehensive symbol information including trading rules,
    precision requirements, and exchange-specific metadata.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol",
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9]+$"
    )
    name: Optional[str] = Field(
        None,
        description="Full name of the symbol",
        max_length=100
    )
    max_leverage: Optional[Decimal] = Field(
        None,
        description="Maximum leverage allowed",
        ge=1,
        decimal_places=2
    )
    only_cross: Optional[bool] = Field(
        None,
        description="Whether only cross margin is allowed"
    )
    sz_decimals: Optional[int] = Field(
        None,
        description="Decimal places for size/quantity",
        ge=0,
        le=8
    )
    price_decimals: Optional[int] = Field(
        None,
        description="Decimal places for price",
        ge=0,
        le=8
    )
    min_order_size: Optional[Decimal] = Field(
        None,
        description="Minimum order size",
        gt=0,
        decimal_places=8
    )
    max_order_size: Optional[Decimal] = Field(
        None,
        description="Maximum order size", 
        gt=0,
        decimal_places=8
    )
    tick_size: Optional[Decimal] = Field(
        None,
        description="Minimum price increment",
        gt=0,
        decimal_places=8
    )
    step_size: Optional[Decimal] = Field(
        None,
        description="Minimum quantity increment",
        gt=0,
        decimal_places=8
    )
    is_active: bool = Field(
        True,
        description="Whether symbol is actively trading"
    )
    base_asset: Optional[str] = Field(
        None,
        description="Base asset (e.g., 'BTC' for BTCUSDT)",
        max_length=20
    )
    quote_asset: Optional[str] = Field(
        None,
        description="Quote asset (e.g., 'USDT' for BTCUSDT)",
        max_length=20
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @field_validator('symbol', 'base_asset', 'quote_asset')
    @classmethod
    def validate_asset_symbols(cls, v) -> Optional[str]:
        """Validate and normalize asset symbols."""
        if v is None:
            return v
            
        v = v.upper().strip()
        
        if not re.match(r'^[A-Z0-9]+$', v):
            raise ValueError(f"Asset symbol must contain only uppercase letters and numbers: {v}")
            
        return v
    
    @field_validator('min_order_size', 'max_order_size', 'tick_size', 'step_size', mode='before')
    @classmethod
    def validate_size_fields(cls, v) -> Optional[Decimal]:
        """Convert and validate size/precision fields."""
        if v is None:
            return v
            
        if isinstance(v, str):
            v = v.strip()
            if 'e' in v.lower() or 'E' in v:
                v = f"{float(v):.8f}"
                
        decimal_val = Decimal(str(v))
        return decimal_val.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
    
    @model_validator(mode='after')
    def validate_order_size_range(self):
        """Validate min/max order size relationship."""
        if self.min_order_size is not None and self.max_order_size is not None:
            if self.min_order_size >= self.max_order_size:
                raise ValueError(f"Min order size {self.min_order_size} must be less than max order size {self.max_order_size}")
                
        return self
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'SymbolInfo':
        """
        Create SymbolInfo from HyperLiquid metadata response.
        
        Expected format from universe array in metadata response.
        """
        return cls(
            symbol=data['name'],
            max_leverage=data.get('maxLeverage'),
            only_cross=data.get('onlyCross', False),
            sz_decimals=data.get('szDecimals'),
        )
    
    def validate_price(self, price: Union[Decimal, str, float]) -> Decimal:
        """Validate price according to symbol specifications."""
        price_decimal = Decimal(str(price))
        
        if self.tick_size:
            # Round to nearest tick size
            remainder = price_decimal % self.tick_size
            if remainder != 0:
                price_decimal = price_decimal - remainder + self.tick_size
                
        if self.price_decimals is not None:
            # Round to specified decimal places
            quantize_str = '0.' + '0' * self.price_decimals
            price_decimal = price_decimal.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
            
        return price_decimal
    
    def validate_quantity(self, quantity: Union[Decimal, str, float]) -> Decimal:
        """Validate quantity according to symbol specifications."""
        qty_decimal = Decimal(str(quantity))
        
        if self.min_order_size and qty_decimal < self.min_order_size:
            raise ValueError(f"Quantity {qty_decimal} below minimum {self.min_order_size}")
            
        if self.max_order_size and qty_decimal > self.max_order_size:
            raise ValueError(f"Quantity {qty_decimal} above maximum {self.max_order_size}")
        
        if self.step_size:
            # Round to nearest step size
            remainder = qty_decimal % self.step_size
            if remainder != 0:
                qty_decimal = qty_decimal - remainder + self.step_size
                
        if self.sz_decimals is not None:
            # Round to specified decimal places
            quantize_str = '0.' + '0' * self.sz_decimals
            qty_decimal = qty_decimal.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
            
        return qty_decimal 