"""
HyperLiquid WebSocket Message Models

This module contains Pydantic models for all HyperLiquid WebSocket messages:
- WSMessage: Base model for all WebSocket messages
- PriceUpdateMessage: Real-time price update messages
- TradeUpdateMessage: Live trade feed messages  
- OrderBookUpdateMessage: Level 2 order book update messages
- SubscriptionMessage: Subscription management messages
- MessageRouter: Message type discrimination and routing

All models are designed for direct compatibility with HyperLiquid's WebSocket API
and include comprehensive validation for real-time data streams.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Literal, Type
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum

from .market_data import CandlestickData, Timeframe
from .orderbook import OrderBookLevel, OrderBookDelta
from .trades import Trade


class MessageType(str, Enum):
    """WebSocket message types from HyperLiquid."""
    
    # Market data messages
    PRICE_UPDATE = "allMids"
    TRADE_UPDATE = "trades"
    ORDER_BOOK_UPDATE = "l2Book"
    CANDLE_UPDATE = "candle"
    
    # User data messages
    USER_UPDATE = "user"
    ORDER_UPDATE = "order"
    FILL_UPDATE = "fill"
    
    # System messages
    SUBSCRIPTION_ACK = "subscriptionAck"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    
    # Connection messages
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    RECONNECT = "reconnect"


class SubscriptionChannel(str, Enum):
    """WebSocket subscription channels."""
    
    ALL_MIDS = "allMids"
    TRADES = "trades"
    L2_BOOK = "l2Book"
    CANDLE = "candle"
    USER = "user"
    ORDER_UPDATES = "orderUpdates"
    USER_EVENTS = "userEvents"


class WSMessage(BaseModel):
    """
    Base WebSocket message model for all HyperLiquid WebSocket messages.
    
    Provides common structure and validation for all message types
    including timestamp handling, message identification, and routing.
    """
    
    channel: str = Field(
        ...,
        description="WebSocket channel/subscription name"
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Message payload data"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Message timestamp (auto-generated if not provided)"
    )
    message_id: Optional[str] = Field(
        None,
        description="Unique message identifier"
    )
    sequence: Optional[int] = Field(
        None,
        description="Message sequence number for ordering",
        ge=0
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    def __init__(self, **data):
        """Initialize with auto-generated timestamp if not provided."""
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now(timezone.utc)
        super().__init__(**data)
    
    @classmethod
    def from_hyperliquid(cls, raw_message: Dict[str, Any]) -> 'WSMessage':
        """
        Create WSMessage from raw HyperLiquid WebSocket message.
        
        Expected format: {
            'channel': 'allMids',
            'data': {...}
        }
        """
        return cls(
            channel=raw_message.get('channel', 'unknown'),
            data=raw_message.get('data', {}),
            timestamp=raw_message.get('timestamp'),
            message_id=raw_message.get('id'),
            sequence=raw_message.get('sequence')
        )
    
    def get_message_type(self) -> MessageType:
        """Determine message type from channel."""
        try:
            return MessageType(self.channel)
        except ValueError:
            # Handle unknown message types gracefully
            if "trade" in self.channel.lower():
                return MessageType.TRADE_UPDATE
            elif "book" in self.channel.lower() or "l2" in self.channel.lower():
                return MessageType.ORDER_BOOK_UPDATE
            elif "price" in self.channel.lower() or "mid" in self.channel.lower():
                return MessageType.PRICE_UPDATE
            else:
                return MessageType.ERROR
    
    @property
    def is_market_data(self) -> bool:
        """Check if message is market data related."""
        market_types = {
            MessageType.PRICE_UPDATE,
            MessageType.TRADE_UPDATE,
            MessageType.ORDER_BOOK_UPDATE,
            MessageType.CANDLE_UPDATE
        }
        return self.get_message_type() in market_types
    
    @property
    def is_user_data(self) -> bool:
        """Check if message is user data related."""
        user_types = {
            MessageType.USER_UPDATE,
            MessageType.ORDER_UPDATE,
            MessageType.FILL_UPDATE
        }
        return self.get_message_type() in user_types


class PriceUpdateMessage(BaseModel):
    """
    Real-time price update message from HyperLiquid's allMids channel.
    
    Contains current mid prices for all or specific trading symbols,
    providing high-frequency price snapshots.
    """
    
    prices: Dict[str, Decimal] = Field(
        ...,
        description="Symbol to mid price mapping"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Price update timestamp"
    )
    channel: str = Field(
        default="allMids",
        description="WebSocket channel name"
    )
    sequence: Optional[int] = Field(
        None,
        description="Update sequence number",
        ge=0
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @field_validator('prices', mode='before')
    @classmethod
    def validate_prices(cls, v) -> Dict[str, Decimal]:
        """Convert price strings to Decimal values."""
        if isinstance(v, dict):
            return {str(k): Decimal(str(price)) for k, price in v.items()}
        return v
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'PriceUpdateMessage':
        """
        Create PriceUpdateMessage from HyperLiquid WebSocket data.
        
        Expected format: {
            'BTC': '50000.5',
            'ETH': '3000.25',
            ...
        }
        """
        return cls(
            prices=data,
            timestamp=datetime.now(timezone.utc)
        )
    
    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get price for a specific symbol."""
        return self.prices.get(symbol.upper())
    
    @property
    def symbol_count(self) -> int:
        """Get number of symbols in update."""
        return len(self.prices)
    
    @property
    def symbols(self) -> List[str]:
        """Get list of symbols in update."""
        return list(self.prices.keys())


class TradeUpdateMessage(BaseModel):
    """
    Live trade update message from HyperLiquid's trades channel.
    
    Contains real-time trade executions with price, quantity, side,
    and timing information for immediate market analysis.
    """
    
    trades: List[Trade] = Field(
        ...,
        description="List of new trade executions"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol for these trades"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp"
    )
    channel: str = Field(
        default="trades",
        description="WebSocket channel name"
    )
    sequence: Optional[int] = Field(
        None,
        description="Update sequence number",
        ge=0
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: List[Dict[str, Any]], symbol: str) -> 'TradeUpdateMessage':
        """
        Create TradeUpdateMessage from HyperLiquid WebSocket data.
        
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
        
        return cls(
            trades=trades,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc)
        )
    
    @property
    def trade_count(self) -> int:
        """Get number of trades in update."""
        return len(self.trades)
    
    @property
    def total_volume(self) -> Decimal:
        """Get total volume of trades in update."""
        return sum(trade.quantity for trade in self.trades)
    
    @property
    def volume_weighted_price(self) -> Optional[Decimal]:
        """Calculate volume weighted average price of trades."""
        if not self.trades:
            return None
        
        total_value = sum(trade.price * trade.quantity for trade in self.trades)
        total_volume = self.total_volume
        
        if total_volume == 0:
            return None
        
        return total_value / total_volume
    
    @property
    def buy_trades(self) -> List[Trade]:
        """Get only buy trades from update."""
        return [trade for trade in self.trades if trade.side == "buy"]
    
    @property
    def sell_trades(self) -> List[Trade]:
        """Get only sell trades from update."""
        return [trade for trade in self.trades if trade.side == "sell"]
    
    @property
    def latest_trade(self) -> Optional[Trade]:
        """Get the most recent trade by timestamp."""
        if not self.trades:
            return None
        return max(self.trades, key=lambda t: t.timestamp)


class OrderBookUpdateMessage(BaseModel):
    """
    Level 2 order book update message from HyperLiquid's l2Book channel.
    
    Contains incremental order book changes with bid/ask level updates
    for real-time market depth analysis.
    """
    
    bids: List[OrderBookLevel] = Field(
        default_factory=list,
        description="Updated bid levels"
    )
    asks: List[OrderBookLevel] = Field(
        default_factory=list,
        description="Updated ask levels"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Update timestamp"
    )
    channel: str = Field(
        default="l2Book",
        description="WebSocket channel name"
    )
    sequence: Optional[int] = Field(
        None,
        description="Update sequence number",
        ge=0
    )
    is_snapshot: bool = Field(
        default=False,
        description="Whether this is a full snapshot or incremental update"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any], symbol: str, is_snapshot: bool = False) -> 'OrderBookUpdateMessage':
        """
        Create OrderBookUpdateMessage from HyperLiquid WebSocket data.
        
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
        bids = []
        asks = []
        
        if 'levels' in data and len(data['levels']) >= 2:
            # First array is bids, second is asks
            bid_data = data['levels'][0]
            ask_data = data['levels'][1]
            
            for bid_level in bid_data:
                if len(bid_level) >= 2:
                    level = OrderBookLevel(
                        price=str(bid_level[0]),
                        quantity=str(bid_level[1]),
                        order_count=bid_level[2] if len(bid_level) > 2 else None,
                        side="bid"
                    )
                    bids.append(level)
            
            for ask_level in ask_data:
                if len(ask_level) >= 2:
                    level = OrderBookLevel(
                        price=str(ask_level[0]),
                        quantity=str(ask_level[1]),
                        order_count=ask_level[2] if len(ask_level) > 2 else None,
                        side="ask"
                    )
                    asks.append(level)
        
        # Alternative format with direct bids/asks arrays
        elif 'bids' in data and 'asks' in data:
            for bid_level in data['bids']:
                if len(bid_level) >= 2:
                    level = OrderBookLevel(
                        price=str(bid_level[0]),
                        quantity=str(bid_level[1]),
                        order_count=bid_level[2] if len(bid_level) > 2 else None,
                        side="bid"
                    )
                    bids.append(level)
            
            for ask_level in data['asks']:
                if len(ask_level) >= 2:
                    level = OrderBookLevel(
                        price=str(ask_level[0]),
                        quantity=str(ask_level[1]),
                        order_count=ask_level[2] if len(ask_level) > 2 else None,
                        side="ask"
                    )
                    asks.append(level)
        
        return cls(
            bids=bids,
            asks=asks,
            symbol=symbol,
            is_snapshot=is_snapshot,
            timestamp=datetime.now(timezone.utc)
        )
    
    @property
    def has_updates(self) -> bool:
        """Check if update contains any changes."""
        return len(self.bids) > 0 or len(self.asks) > 0
    
    @property
    def update_count(self) -> int:
        """Get total number of level updates."""
        return len(self.bids) + len(self.asks)
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best (highest) bid level."""
        if not self.bids:
            return None
        return max(self.bids, key=lambda level: level.price)
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best (lowest) ask level."""
        if not self.asks:
            return None
        return min(self.asks, key=lambda level: level.price)
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate spread from best levels."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None


class CandleUpdateMessage(BaseModel):
    """
    Real-time candlestick update message from HyperLiquid's candle channel.
    
    Contains live candlestick data updates for various timeframes,
    providing real-time OHLCV data for trading analysis.
    """
    
    candle: CandlestickData = Field(
        ...,
        description="Updated candlestick data"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    timeframe: Timeframe = Field(
        ...,
        description="Candlestick timeframe"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Update timestamp"
    )
    channel: str = Field(
        default="candle",
        description="WebSocket channel name"
    )
    sequence: Optional[int] = Field(
        None,
        description="Update sequence number",
        ge=0
    )
    is_closed: bool = Field(
        default=False,
        description="Whether this candle is closed/final"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any], symbol: str, timeframe: Timeframe) -> 'CandleUpdateMessage':
        """
        Create CandleUpdateMessage from HyperLiquid WebSocket data.
        
        Expected format: {
            't': 1705316200000,  # open time millis
            'T': 1705316260000,  # close time millis  
            's': 'BTC',          # symbol
            'i': '1m',           # interval
            'o': 50000.0,        # open price
            'h': 50100.0,        # high price
            'l': 49900.0,        # low price
            'c': 50050.0,        # close price
            'v': 123.45,         # volume
            'n': 1500            # number of trades
        }
        
        Note: HyperLiquid does NOT send a 'closed' field.
        """
        candle = CandlestickData.from_hyperliquid(data, timeframe)
        
        # Since HyperLiquid doesn't provide a 'closed' field, we assume candles are updates
        # The collector will handle filtering based on timing logic
        return cls(
            candle=candle,
            symbol=symbol,
            timeframe=timeframe,
            is_closed=False,  # Always False since HyperLiquid doesn't indicate closure
            timestamp=datetime.now(timezone.utc)
        )
    
    @property
    def price_change(self) -> Decimal:
        """Calculate price change from open to close."""
        return self.candle.close - self.candle.open
    
    @property
    def price_change_percent(self) -> Decimal:
        """Calculate percentage price change."""
        if self.candle.open == 0:
            return Decimal('0')
        return (self.price_change / self.candle.open) * Decimal('100')


class SubscriptionMessage(BaseModel):
    """
    WebSocket subscription management message.
    
    Handles subscription requests, acknowledgments, and channel management
    for HyperLiquid WebSocket connections.
    """
    
    action: Literal["subscribe", "unsubscribe", "ack", "error"] = Field(
        ...,
        description="Subscription action type"
    )
    channel: SubscriptionChannel = Field(
        ...,
        description="Subscription channel"
    )
    symbol: Optional[str] = Field(
        None,
        description="Symbol for symbol-specific subscriptions"
    )
    timeframe: Optional[Timeframe] = Field(
        None,
        description="Timeframe for candle subscriptions"
    )
    success: bool = Field(
        default=True,
        description="Whether subscription action was successful"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if subscription failed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Subscription message timestamp"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )
    
    @classmethod
    def subscribe_request(cls, channel: SubscriptionChannel, symbol: Optional[str] = None, timeframe: Optional[Timeframe] = None) -> 'SubscriptionMessage':
        """Create a subscription request message."""
        return cls(
            action="subscribe",
            channel=channel,
            symbol=symbol,
            timeframe=timeframe
        )
    
    @classmethod
    def unsubscribe_request(cls, channel: SubscriptionChannel, symbol: Optional[str] = None) -> 'SubscriptionMessage':
        """Create an unsubscription request message."""
        return cls(
            action="unsubscribe",
            channel=channel,
            symbol=symbol
        )
    
    @classmethod
    def subscription_ack(cls, channel: SubscriptionChannel, success: bool = True, error_message: Optional[str] = None) -> 'SubscriptionMessage':
        """Create a subscription acknowledgment message."""
        return cls(
            action="ack",
            channel=channel,
            success=success,
            error_message=error_message
        )
    
    def to_hyperliquid_format(self) -> Dict[str, Any]:
        """Convert to HyperLiquid subscription format."""
        data = {
            "method": self.action,
            "subscription": {
                "type": self.channel.value
            }
        }
        
        if self.symbol:
            data["subscription"]["coin"] = self.symbol
        
        if self.timeframe:
            data["subscription"]["interval"] = self.timeframe.value
        
        return data
    
    @property
    def subscription_key(self) -> str:
        """Generate unique subscription key."""
        key = f"{self.channel.value}"
        if self.symbol:
            key += f":{self.symbol}"
        if self.timeframe:
            key += f":{self.timeframe.value}"
        return key


class MessageRouter:
    """
    WebSocket message router for parsing and routing incoming messages.
    
    Provides message type discrimination, parsing, and routing to
    appropriate message classes based on channel and content.
    """
    
    @staticmethod
    def parse_message(raw_message: Dict[str, Any]) -> Union[
        PriceUpdateMessage,
        TradeUpdateMessage,
        OrderBookUpdateMessage,
        CandleUpdateMessage,
        SubscriptionMessage,
        WSMessage
    ]:
        """
        Parse raw WebSocket message and return appropriate typed message.
        
        Args:
            raw_message: Raw message dictionary from WebSocket
            
        Returns:
            Typed message object based on channel/content
        """
        channel = raw_message.get('channel', 'unknown')
        data = raw_message.get('data', {})
        
        try:
            # Route based on channel
            if channel == "allMids" or channel == SubscriptionChannel.ALL_MIDS:
                # Only parse as PriceUpdateMessage if we have valid price data
                if data and isinstance(data, dict):
                    return PriceUpdateMessage.from_hyperliquid(data)
                else:
                    # Fall back to WSMessage for empty/invalid data
                    return WSMessage.from_hyperliquid(raw_message)
                
            elif channel == "trades" or channel == SubscriptionChannel.TRADES:
                symbol = raw_message.get('symbol', 'UNKNOWN')
                if isinstance(data, list) and data:
                    return TradeUpdateMessage.from_hyperliquid(data, symbol)
                else:
                    # Invalid trade data structure or empty
                    raise ValueError("Trade data must be a non-empty list")
                    
            elif channel == "l2Book" or channel == SubscriptionChannel.L2_BOOK:
                symbol = raw_message.get('symbol', 'UNKNOWN')
                is_snapshot = raw_message.get('snapshot', False)
                # Only parse if we have valid orderbook data
                if data and ('levels' in data or ('bids' in data and 'asks' in data)):
                    return OrderBookUpdateMessage.from_hyperliquid(data, symbol, is_snapshot)
                else:
                    # Fall back to WSMessage for invalid/empty orderbook data
                    return WSMessage.from_hyperliquid(raw_message)
                
            elif channel == "candle" or channel == SubscriptionChannel.CANDLE:
                symbol = raw_message.get('symbol', 'UNKNOWN')
                # Handle both 'interval' and 'timeframe' keys
                interval = raw_message.get('interval') or raw_message.get('timeframe', '1m')
                try:
                    timeframe = Timeframe.from_string(interval)
                except Exception:
                    # Fallback to 1m if parsing fails
                    timeframe = Timeframe.ONE_MINUTE
                
                # Only parse if we have valid candle data
                if data and isinstance(data, dict) and any(key in data for key in ['o', 'h', 'l', 'c', 'open', 'high', 'low', 'close']):
                    return CandleUpdateMessage.from_hyperliquid(data, symbol, timeframe)
                else:
                    # Fall back to WSMessage for invalid/empty candle data
                    return WSMessage.from_hyperliquid(raw_message)
                
            elif channel == "subscriptionAck" or "subscription" in channel.lower():
                # Parse subscription acknowledgment
                return SubscriptionMessage.subscription_ack(
                    channel=SubscriptionChannel.ALL_MIDS,  # Default
                    success=data.get('success', True),
                    error_message=data.get('error')
                )
            
            # Fallback to generic WSMessage for unknown channels
            return WSMessage.from_hyperliquid(raw_message)
                
        except Exception as e:
            # If parsing fails, return generic message with error info
            return WSMessage(
                channel=channel,
                data={
                    "error": f"Failed to parse message: {str(e)}",
                    "original_data": data
                }
            )
    
    @staticmethod
    def get_message_type(message: Union[WSMessage, Dict[str, Any]]) -> MessageType:
        """Determine message type from message or raw data."""
        if isinstance(message, dict):
            channel = message.get('channel', '')
        else:
            channel = message.channel
        
        try:
            return MessageType(channel)
        except ValueError:
            # Handle unknown channels
            channel_lower = channel.lower()
            if "trade" in channel_lower:
                return MessageType.TRADE_UPDATE
            elif "book" in channel_lower or "l2" in channel_lower:
                return MessageType.ORDER_BOOK_UPDATE
            elif "price" in channel_lower or "mid" in channel_lower:
                return MessageType.PRICE_UPDATE
            elif "candle" in channel_lower:
                return MessageType.CANDLE_UPDATE
            elif "subscription" in channel_lower:
                return MessageType.SUBSCRIPTION_ACK
            else:
                return MessageType.ERROR
    
    @staticmethod
    def create_subscription_requests(symbols: List[str], channels: List[SubscriptionChannel]) -> List[SubscriptionMessage]:
        """Create subscription requests for multiple symbols and channels."""
        requests = []
        
        for channel in channels:
            if channel in [SubscriptionChannel.ALL_MIDS]:
                # Global subscriptions (no symbol needed)
                requests.append(SubscriptionMessage.subscribe_request(channel))
            else:
                # Symbol-specific subscriptions
                for symbol in symbols:
                    requests.append(SubscriptionMessage.subscribe_request(channel, symbol))
        
        return requests
    
    @staticmethod
    def is_market_data_message(message: Union[WSMessage, Dict[str, Any]]) -> bool:
        """Check if message contains market data."""
        msg_type = MessageRouter.get_message_type(message)
        market_types = {
            MessageType.PRICE_UPDATE,
            MessageType.TRADE_UPDATE,
            MessageType.ORDER_BOOK_UPDATE,
            MessageType.CANDLE_UPDATE
        }
        return msg_type in market_types 