"""
Trade Execution Data Models

This module contains Pydantic models for individual trade executions:
- Trade: Individual trade execution with comprehensive validation
- TradeAggregation: Aggregated trade statistics over time periods
- TradeAnalytics: Advanced analytics and metrics for trade data

All models are designed for HyperLiquid API compatibility and include
robust validation for cryptocurrency trading requirements.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .market_data import MarketData


class Trade(MarketData):
    """
    Individual trade execution model with comprehensive validation.
    
    Represents a single trade execution from exchange feeds with:
    - Price and quantity validation for cryptocurrency precision
    - Trade direction (buy/sell) validation
    - HyperLiquid API format compatibility
    - Advanced trade analytics properties
    """
    
    trade_id: Optional[str] = Field(
        None,
        description="Unique trade identifier from exchange",
        max_length=50
    )
    price: Decimal = Field(
        ...,
        description="Trade execution price",
        gt=0,
        decimal_places=8
    )
    quantity: Decimal = Field(
        ...,
        description="Trade quantity",
        gt=0,
        decimal_places=8
    )
    side: Literal["buy", "sell"] = Field(
        ...,
        description="Trade direction (buy/sell)"
    )
    is_buyer_maker: Optional[bool] = Field(
        None,
        description="Whether the buyer was the maker (true) or taker (false)"
    )
    exchange: str = Field(
        "hyperliquid",
        description="Exchange identifier",
        max_length=50
    )
    user_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional user/exchange specific data"
    )
    
    @field_validator('price', 'quantity', mode='before')
    @classmethod
    def validate_decimal_fields(cls, v) -> Decimal:
        """Convert string/float to Decimal with proper precision."""
        if isinstance(v, str):
            v = v.strip()
            if 'e' in v.lower() or 'E' in v:
                v = f"{float(v):.8f}"
                
        decimal_val = Decimal(str(v))
        return decimal_val.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any], symbol: str) -> 'Trade':
        """
        Create Trade from HyperLiquid trades API response.
        
        Expected format: {
            'time': timestamp_ms,
            'px': price,
            'sz': size,
            'side': 'A' (ask/sell) or 'B' (bid/buy),
            'user': user_data
        }
        """
        # Convert HyperLiquid side notation
        side = "buy" if data.get('side') == 'B' else "sell"
        
        return cls(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(int(data['time']) / 1000, tz=timezone.utc),
            time_ms=int(data['time']),
            trade_id=data.get('tid'),  # Trade ID if available
            price=data['px'],
            quantity=data['sz'],
            side=side,
            user_data=data.get('user'),
            exchange="hyperliquid"
        )
    
    def to_hyperliquid(self) -> Dict[str, Any]:
        """Convert to HyperLiquid trades format."""
        # Convert side to HyperLiquid notation
        hl_side = 'B' if self.side == 'buy' else 'A'
        
        result = {
            'time': self.time_ms,
            'px': str(self.price),
            'sz': str(self.quantity),
            'side': hl_side,
        }
        
        if self.trade_id:
            result['tid'] = self.trade_id
        
        if self.user_data:
            result['user'] = self.user_data
            
        return result
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value (price × quantity)."""
        return self.price * self.quantity
    
    @property
    def is_buy(self) -> bool:
        """Check if trade is a buy."""
        return self.side == "buy"
    
    @property
    def is_sell(self) -> bool:
        """Check if trade is a sell."""
        return self.side == "sell"
    
    @property
    def taker_side(self) -> str:
        """Get the taker side based on is_buyer_maker flag."""
        if self.is_buyer_maker is None:
            return self.side  # Default to trade side
        
        # If buyer was maker, then seller was taker (and vice versa)
        return "sell" if self.is_buyer_maker else "buy"
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Trade({self.symbol}): {self.side.upper()} {self.quantity} @ ${self.price} = ${self.notional_value:.2f}"


class TradeAggregation(BaseModel):
    """
    Aggregated trade statistics over a time period.
    
    Provides comprehensive trade analytics including volume, VWAP,
    trade count, and directional flow analysis.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol",
        min_length=1,
        max_length=20
    )
    start_time: datetime = Field(
        ...,
        description="Aggregation period start time"
    )
    end_time: datetime = Field(
        ...,
        description="Aggregation period end time"
    )
    trade_count: int = Field(
        ...,
        description="Total number of trades",
        ge=0
    )
    total_volume: Decimal = Field(
        ...,
        description="Total traded volume",
        ge=0,
        decimal_places=8
    )
    total_notional: Decimal = Field(
        ...,
        description="Total notional value",
        ge=0,
        decimal_places=8
    )
    vwap: Optional[Decimal] = Field(
        None,
        description="Volume weighted average price",
        gt=0,
        decimal_places=8
    )
    first_price: Optional[Decimal] = Field(
        None,
        description="First trade price in period",
        gt=0,
        decimal_places=8
    )
    last_price: Optional[Decimal] = Field(
        None,
        description="Last trade price in period",
        gt=0,
        decimal_places=8
    )
    min_price: Optional[Decimal] = Field(
        None,
        description="Minimum trade price in period",
        gt=0,
        decimal_places=8
    )
    max_price: Optional[Decimal] = Field(
        None,
        description="Maximum trade price in period",
        gt=0,
        decimal_places=8
    )
    buy_volume: Decimal = Field(
        default=Decimal('0'),
        description="Total buy volume",
        ge=0,
        decimal_places=8
    )
    sell_volume: Decimal = Field(
        default=Decimal('0'),
        description="Total sell volume",
        ge=0,
        decimal_places=8
    )
    buy_count: int = Field(
        default=0,
        description="Number of buy trades",
        ge=0
    )
    sell_count: int = Field(
        default=0,
        description="Number of sell trades",
        ge=0
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @field_validator('start_time', 'end_time', mode='before')
    @classmethod
    def validate_timestamp(cls, v) -> datetime:
        """Ensure timestamps are timezone-aware UTC."""
        if isinstance(v, str):
            if v.endswith('Z'):
                v = v[:-1] + '+00:00'
            dt = datetime.fromisoformat(v)
        elif isinstance(v, (int, float)):
            dt = datetime.fromtimestamp(v / 1000, tz=timezone.utc)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Invalid timestamp format: {type(v)}")
            
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)
            
        return dt
    
    @model_validator(mode='after')
    def validate_aggregation(self) -> 'TradeAggregation':
        """Validate trade aggregation data consistency."""
        # Allow equal times for edge cases (instantaneous buckets)
        if self.end_time < self.start_time:
            raise ValueError(f"End time {self.end_time} must be after or equal to start time {self.start_time}")
        
        # Validate trade counts match
        if self.buy_count is not None and self.sell_count is not None:
            if self.buy_count + self.sell_count != self.trade_count:
                raise ValueError(f"Buy count ({self.buy_count}) + sell count ({self.sell_count}) must equal trade count ({self.trade_count})")
        
        # Validate VWAP is provided when there's volume
        if self.total_volume > 0 and self.vwap is None:
            raise ValueError("VWAP must be provided when total volume > 0")
        
        return self
    
    @classmethod
    def from_trades(cls, trades: List[Trade], start_time: datetime, end_time: datetime) -> 'TradeAggregation':
        """
        Create TradeAggregation from a list of trades.
        
        Args:
            trades: List of Trade objects to aggregate
            start_time: Period start time
            end_time: Period end time
            
        Returns:
            TradeAggregation with calculated statistics
        """
        if not trades:
            return cls(
                symbol="UNKNOWN",
                start_time=start_time,
                end_time=end_time,
                trade_count=0,
                total_volume=Decimal('0'),
                total_notional=Decimal('0')
            )
        
        symbol = trades[0].symbol
        trade_count = len(trades)
        
        # Calculate volumes and notional
        total_volume = sum(trade.quantity for trade in trades)
        total_notional = sum(trade.notional_value for trade in trades)
        vwap = (total_notional / total_volume).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP) if total_volume > 0 else None
        
        # Price statistics
        prices = [trade.price for trade in trades]
        first_price = trades[0].price
        last_price = trades[-1].price
        min_price = min(prices)
        max_price = max(prices)
        
        # Side statistics
        buy_trades = [t for t in trades if t.side == "buy"]
        sell_trades = [t for t in trades if t.side == "sell"]
        
        buy_volume = sum(t.quantity for t in buy_trades)
        sell_volume = sum(t.quantity for t in sell_trades)
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)
        
        return cls(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            trade_count=trade_count,
            total_volume=total_volume,
            total_notional=total_notional,
            vwap=vwap,
            first_price=first_price,
            last_price=last_price,
            min_price=min_price,
            max_price=max_price,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_count=buy_count,
            sell_count=sell_count
        )
    
    @property
    def duration_seconds(self) -> float:
        """Get aggregation period duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def volume_per_second(self) -> Decimal:
        """Calculate average volume per second."""
        duration = self.duration_seconds
        return self.total_volume / Decimal(str(duration)) if duration > 0 else Decimal('0')
    
    @property
    def trades_per_second(self) -> Decimal:
        """Calculate average trades per second."""
        duration = self.duration_seconds
        return Decimal(str(self.trade_count)) / Decimal(str(duration)) if duration > 0 else Decimal('0')
    
    @property
    def price_change(self) -> Optional[Decimal]:
        """Calculate price change (last - first)."""
        if self.first_price and self.last_price:
            return self.last_price - self.first_price
        return None
    
    @property
    def price_change_pct(self) -> Optional[Decimal]:
        """Calculate price change percentage."""
        change = self.price_change
        if change and self.first_price and self.first_price > 0:
            return (change / self.first_price) * 100
        return None
    
    @property
    def buy_sell_ratio(self) -> Optional[Decimal]:
        """Calculate buy/sell volume ratio."""
        if self.sell_volume > 0:
            return self.buy_volume / self.sell_volume
        elif self.buy_volume > 0:
            return Decimal('999999')  # Infinite (all buys)
        else:
            return None
    
    @property
    def buy_pressure(self) -> Optional[Decimal]:
        """
        Calculate buy pressure as percentage of total volume.
        
        Returns:
            Percentage (0-100) representing buy volume / total volume
        """
        if self.total_volume > 0:
            return (self.buy_volume / self.total_volume) * 100
        return None
    
    @property
    def average_trade_size(self) -> Decimal:
        """Calculate average trade size."""
        return self.total_volume / Decimal(str(self.trade_count)) if self.trade_count > 0 else Decimal('0')
    
    @property
    def average_trade_value(self) -> Decimal:
        """Calculate average trade notional value."""
        return self.total_notional / Decimal(str(self.trade_count)) if self.trade_count > 0 else Decimal('0')


class TradeAnalytics(BaseModel):
    """
    Advanced trade analytics and metrics.
    
    Provides sophisticated analysis of trade patterns including:
    - Volume profile analysis
    - Trade size distribution
    - Temporal patterns
    - Market microstructure metrics
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol",
        min_length=1,
        max_length=20
    )
    analysis_period: timedelta = Field(
        ...,
        description="Analysis time period"
    )
    trade_aggregations: List[TradeAggregation] = Field(
        ...,
        description="Time-bucketed trade aggregations",
        min_length=1
    )
    volume_profile: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Volume at price levels"
    )
    large_trade_threshold: Decimal = Field(
        default=Decimal('1.0'),
        description="Threshold for large trade classification",
        gt=0
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
            timedelta: lambda v: v.total_seconds(),
        }
    )
    
    @classmethod
    def from_trades(cls, trades: List[Trade], bucket_size: timedelta = timedelta(minutes=1),
                   large_trade_threshold: Decimal = Decimal('1.0')) -> 'TradeAnalytics':
        """
        Create TradeAnalytics from a list of trades.
        
        Args:
            trades: List of Trade objects to analyze
            bucket_size: Time bucket size for aggregations
            large_trade_threshold: Minimum size for large trade classification
            
        Returns:
            TradeAnalytics with comprehensive analysis
        """
        if not trades:
            raise ValueError("Cannot create analytics from empty trade list")
        
        symbol = trades[0].symbol
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        # Calculate analysis period
        start_time = sorted_trades[0].timestamp
        end_time = sorted_trades[-1].timestamp
        analysis_period = end_time - start_time
        
        # For single trade or same timestamp, ensure minimum period
        if analysis_period.total_seconds() == 0:
            end_time = start_time + bucket_size
            analysis_period = bucket_size
        
        # Create time buckets
        aggregations = []
        current_bucket_start = start_time
        
        while current_bucket_start < end_time:
            current_bucket_end = current_bucket_start + bucket_size
            
            # Filter trades for this bucket
            bucket_trades = [
                t for t in sorted_trades 
                if current_bucket_start <= t.timestamp < current_bucket_end
            ]
            
            if bucket_trades:
                agg = TradeAggregation.from_trades(bucket_trades, current_bucket_start, current_bucket_end)
                aggregations.append(agg)
            
            current_bucket_start = current_bucket_end
        
        # Ensure we have at least one aggregation for edge cases
        if not aggregations and trades:
            # Create a single aggregation for all trades
            agg = TradeAggregation.from_trades(sorted_trades, start_time, start_time + bucket_size)
            aggregations.append(agg)
        
        # Calculate volume profile (volume at each price level)
        volume_profile = {}
        for trade in trades:
            price_key = str(trade.price.quantize(Decimal('0.01')))  # Round to cents
            volume_profile[price_key] = volume_profile.get(price_key, Decimal('0')) + trade.quantity
        
        return cls(
            symbol=symbol,
            analysis_period=analysis_period,
            trade_aggregations=aggregations,
            volume_profile=volume_profile,
            large_trade_threshold=large_trade_threshold
        )
    
    @property
    def total_trades(self) -> int:
        """Get total number of trades across all aggregations."""
        return sum(agg.trade_count for agg in self.trade_aggregations)
    
    @property
    def total_volume(self) -> Decimal:
        """Get total volume across all aggregations."""
        return sum(agg.total_volume for agg in self.trade_aggregations)
    
    @property
    def total_notional(self) -> Decimal:
        """Get total notional value across all aggregations."""
        return sum(agg.total_notional for agg in self.trade_aggregations)
    
    @property
    def overall_vwap(self) -> Decimal:
        """Calculate overall VWAP across all periods."""
        return self.total_notional / self.total_volume if self.total_volume > 0 else Decimal('0')
    
    @property
    def peak_volume_price(self) -> Optional[Decimal]:
        """Get price level with highest volume."""
        if not self.volume_profile:
            return None
        
        max_volume_price = max(self.volume_profile.items(), key=lambda x: x[1])
        return Decimal(max_volume_price[0])
    
    @property
    def volume_weighted_std(self) -> Decimal:
        """Calculate volume-weighted standard deviation of prices."""
        if not self.volume_profile or self.total_volume == 0:
            return Decimal('0')
        
        # Calculate volume-weighted mean (VWAP)
        vwap = self.overall_vwap
        
        # Calculate volume-weighted variance
        variance = Decimal('0')
        for price_str, volume in self.volume_profile.items():
            price = Decimal(price_str)
            weight = volume / self.total_volume
            variance += weight * ((price - vwap) ** 2)
        
        # Return standard deviation
        return variance.sqrt()
    
    def get_large_trades_analysis(self) -> Dict[str, Any]:
        """
        Analyze large trades (above threshold).
        
        Returns:
            Dict with large trade statistics and patterns
        """
        large_trades = []
        total_large_volume = Decimal('0')
        total_large_notional = Decimal('0')
        
        for agg in self.trade_aggregations:
            # This is a simplified analysis - in practice, you'd need access to individual trades
            # For now, we'll estimate based on average trade size
            avg_trade_size = agg.average_trade_size
            if avg_trade_size >= self.large_trade_threshold:
                large_trades.append(agg)
                total_large_volume += agg.total_volume
                total_large_notional += agg.total_notional
        
        return {
            'large_trade_periods': len(large_trades),
            'large_trade_volume': total_large_volume,
            'large_trade_notional': total_large_notional,
            'large_trade_volume_pct': (total_large_volume / self.total_volume * 100) if self.total_volume > 0 else Decimal('0'),
            'average_large_trade_size': total_large_volume / Decimal(str(len(large_trades))) if large_trades else Decimal('0')
        } 