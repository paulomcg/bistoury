"""
Order Book Data Models

This module contains Pydantic models for order book data structures:
- OrderBookLevel: Individual bid/ask price levels with validation
- OrderBook: Complete order book with bids and asks
- OrderBookSnapshot: Timestamped order book data from exchanges
- OrderBookDelta: Real-time order book updates

All models are designed for HyperLiquid API compatibility and include
robust validation for cryptocurrency trading requirements.
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .market_data import MarketData, PriceLevel


class OrderBookLevel(PriceLevel):
    """
    Individual order book level extending PriceLevel with order book specific functionality.
    
    Represents a single bid or ask level in the order book with price, quantity,
    and optional order count information.
    """
    
    side: Optional[Literal["bid", "ask"]] = Field(
        None,
        description="Side of the order book (bid/ask)"
    )
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value (price * quantity)."""
        return self.price * self.quantity
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        side_str = f" ({self.side})" if self.side else ""
        count_str = f" [{self.count} orders]" if self.count else ""
        return f"OrderBookLevel{side_str}: ${self.price} x {self.quantity}{count_str}"


class OrderBook(BaseModel):
    """
    Complete order book with bid and ask levels.
    
    Represents a full Level 2 order book with validation for:
    - Proper bid/ask ordering (bids descending, asks ascending)
    - No price level overlaps (best bid < best ask)
    - Positive quantities and prices
    """
    
    bids: List[OrderBookLevel] = Field(
        ...,
        description="Bid levels sorted by price (highest first)",
        min_length=0
    )
    asks: List[OrderBookLevel] = Field(
        ...,
        description="Ask levels sorted by price (lowest first)",
        min_length=0
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @field_validator('bids', 'asks')
    @classmethod
    def validate_levels(cls, v: List[OrderBookLevel]) -> List[OrderBookLevel]:
        """Validate that all levels have positive prices and quantities."""
        for level in v:
            if level.price <= 0:
                raise ValueError(f"Order book level must have positive price: {level.price}")
            if level.quantity < 0:
                raise ValueError(f"Order book level must have non-negative quantity: {level.quantity}")
        return v
    
    @model_validator(mode='after')
    def validate_bid_ask_ordering(self):
        """Validate proper bid/ask ordering and no overlap."""
        # Validate bid ordering (highest to lowest)
        if len(self.bids) > 1:
            for i in range(len(self.bids) - 1):
                if self.bids[i].price <= self.bids[i + 1].price:
                    raise ValueError(f"Bids must be ordered from highest to lowest price: {self.bids[i].price} <= {self.bids[i + 1].price}")
        
        # Validate ask ordering (lowest to highest)
        if len(self.asks) > 1:
            for i in range(len(self.asks) - 1):
                if self.asks[i].price >= self.asks[i + 1].price:
                    raise ValueError(f"Asks must be ordered from lowest to highest price: {self.asks[i].price} >= {self.asks[i + 1].price}")
        
        # Validate no bid/ask overlap
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            if best_bid >= best_ask:
                raise ValueError(f"Best bid {best_bid} must be less than best ask {best_ask}")
        
        return self
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'OrderBook':
        """
        Create OrderBook from HyperLiquid l2Book format.
        
        Expected format: {'levels': [['px', 'sz', 'n'], ...], 'type': 'bid'/'ask'}
        or {'bids': [...], 'asks': [...]}
        """
        if 'levels' in data and 'type' in data:
            # Single side format
            levels = []
            side = data['type']
            
            for level_data in data['levels']:
                level = OrderBookLevel(
                    price=level_data[0],
                    quantity=level_data[1],
                    count=level_data[2] if len(level_data) > 2 else None,
                    side=side
                )
                levels.append(level)
            
            if side == 'bid':
                return cls(bids=levels, asks=[])
            else:
                return cls(bids=[], asks=levels)
        
        elif 'bids' in data and 'asks' in data:
            # Full order book format
            bids = []
            for bid_data in data['bids']:
                bid = OrderBookLevel(
                    price=bid_data[0],
                    quantity=bid_data[1],
                    count=bid_data[2] if len(bid_data) > 2 else None,
                    side='bid'
                )
                bids.append(bid)
            
            asks = []
            for ask_data in data['asks']:
                ask = OrderBookLevel(
                    price=ask_data[0],
                    quantity=ask_data[1],
                    count=ask_data[2] if len(ask_data) > 2 else None,
                    side='ask'
                )
                asks.append(ask)
            
            return cls(bids=bids, asks=asks)
        
        else:
            raise ValueError(f"Invalid HyperLiquid order book format: {data}")
    
    def to_hyperliquid(self) -> Dict[str, Any]:
        """Convert to HyperLiquid l2Book format."""
        bids_data = []
        for bid in self.bids:
            level_data = [str(bid.price), str(bid.quantity)]
            if bid.count is not None:
                level_data.append(bid.count)
            bids_data.append(level_data)
        
        asks_data = []
        for ask in self.asks:
            level_data = [str(ask.price), str(ask.quantity)]
            if ask.count is not None:
                level_data.append(ask.count)
            asks_data.append(level_data)
        
        return {
            'bids': bids_data,
            'asks': asks_data
        }
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get the best (highest) bid level."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get the best (lowest) ask level."""
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price (average of best bid and ask)."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None
    
    @property
    def spread_bps(self) -> Optional[Decimal]:
        """Calculate spread in basis points (1/100 of 1%)."""
        spread = self.spread
        mid = self.mid_price
        
        if spread and mid and mid > 0:
            return (spread / mid) * 10000
        return None
    
    @property
    def total_bid_quantity(self) -> Decimal:
        """Calculate total quantity on bid side."""
        return sum(bid.quantity for bid in self.bids)
    
    @property
    def total_ask_quantity(self) -> Decimal:
        """Calculate total quantity on ask side."""
        return sum(ask.quantity for ask in self.asks)
    
    @property
    def depth_imbalance(self) -> Optional[Decimal]:
        """
        Calculate order book imbalance (bid_qty - ask_qty) / (bid_qty + ask_qty).
        
        Returns:
        - Positive values indicate more buy pressure
        - Negative values indicate more sell pressure
        - Range: [-1, 1]
        """
        bid_qty = self.total_bid_quantity
        ask_qty = self.total_ask_quantity
        total_qty = bid_qty + ask_qty
        
        if total_qty > 0:
            return (bid_qty - ask_qty) / total_qty
        return None
    
    def get_levels_within_spread_pct(self, percentage: Decimal) -> Dict[str, List[OrderBookLevel]]:
        """
        Get all levels within a percentage of the best bid/ask.
        
        Args:
            percentage: Percentage range from best price (e.g., 1.0 for 1%)
            
        Returns:
            Dict with 'bids' and 'asks' lists of levels within range
        """
        if not self.best_bid or not self.best_ask:
            return {'bids': [], 'asks': []}
        
        percentage_decimal = percentage / 100
        
        # Calculate price thresholds
        bid_threshold = self.best_bid.price * (1 - percentage_decimal)
        ask_threshold = self.best_ask.price * (1 + percentage_decimal)
        
        # Filter levels within range
        filtered_bids = [bid for bid in self.bids if bid.price >= bid_threshold]
        filtered_asks = [ask for ask in self.asks if ask.price <= ask_threshold]
        
        return {
            'bids': filtered_bids,
            'asks': filtered_asks
        }
    
    def calculate_market_impact(self, quantity: Decimal, side: Literal["buy", "sell"]) -> Dict[str, Decimal]:
        """
        Calculate market impact for a given trade size.
        
        Args:
            quantity: Trade quantity
            side: Trade direction ('buy' or 'sell')
            
        Returns:
            Dict with average_price, total_cost, price_impact_pct
        """
        if side == "buy":
            levels = self.asks
        else:
            levels = self.bids
        
        if not levels:
            raise ValueError(f"No {side} levels available for market impact calculation")
        
        remaining_qty = quantity
        total_cost = Decimal('0')
        levels_used = []
        
        for level in levels:
            if remaining_qty <= 0:
                break
                
            level_qty = min(remaining_qty, level.quantity)
            total_cost += level_qty * level.price
            levels_used.append((level.price, level_qty))
            remaining_qty -= level_qty
        
        if remaining_qty > 0:
            raise ValueError(f"Insufficient liquidity: {remaining_qty} remaining after consuming all levels")
        
        average_price = total_cost / quantity
        reference_price = self.best_ask.price if side == "buy" else self.best_bid.price
        price_impact_pct = abs((average_price - reference_price) / reference_price) * 100
        
        return {
            'average_price': average_price,
            'total_cost': total_cost,
            'price_impact_pct': price_impact_pct,
            'levels_consumed': len(levels_used)
        }


class OrderBookSnapshot(MarketData):
    """
    Timestamped order book snapshot from exchange.
    
    Combines order book data with timestamp and symbol information
    for database storage and historical analysis.
    """
    
    order_book: OrderBook = Field(
        ...,
        description="Complete order book with bids and asks"
    )
    sequence_id: Optional[int] = Field(
        None,
        description="Exchange sequence number for ordering",
        ge=0
    )
    exchange: str = Field(
        "hyperliquid",
        description="Exchange identifier",
        max_length=50
    )
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any], symbol: str, timestamp: datetime) -> 'OrderBookSnapshot':
        """Create OrderBookSnapshot from HyperLiquid l2Book WebSocket message."""
        order_book = OrderBook.from_hyperliquid(data)
        
        return cls(
            symbol=symbol,
            timestamp=timestamp,
            order_book=order_book,
            exchange="hyperliquid"
        )
    
    @property
    def best_bid_price(self) -> Optional[Decimal]:
        """Convenience property for best bid price."""
        return self.order_book.best_bid.price if self.order_book.best_bid else None
    
    @property
    def best_ask_price(self) -> Optional[Decimal]:
        """Convenience property for best ask price."""
        return self.order_book.best_ask.price if self.order_book.best_ask else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Convenience property for bid-ask spread."""
        return self.order_book.spread
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Convenience property for mid price."""
        return self.order_book.mid_price


class OrderBookDelta(BaseModel):
    """
    Real-time order book update/delta message.
    
    Represents incremental changes to the order book from WebSocket feeds,
    used for maintaining real-time order book state.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol",
        min_length=1,
        max_length=20
    )
    timestamp: datetime = Field(
        ...,
        description="Update timestamp in UTC"
    )
    time_ms: Optional[int] = Field(
        None,
        description="Timestamp in milliseconds",
        ge=0
    )
    sequence_id: Optional[int] = Field(
        None,
        description="Sequence number for ordering updates",
        ge=0
    )
    changes: List[Dict[str, Any]] = Field(
        ...,
        description="List of order book changes",
        min_length=1
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
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
    def validate_time_ms(self):
        """Auto-generate time_ms from timestamp if not provided."""
        if self.time_ms is None:
            self.time_ms = int(self.timestamp.timestamp() * 1000)
        return self 