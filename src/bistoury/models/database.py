"""
Database Entity Models

This module contains database-optimized Pydantic models for efficient storage
and retrieval in DuckDB. These models are designed for:
- Optimized serialization/deserialization
- Efficient batch operations
- Compression-aware storage
- Fast query performance
- Data archival and retention

These models complement the core business models by providing
database-specific optimizations while maintaining data integrity.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict
from enum import Enum
import json

from .market_data import Timeframe
from .signals import SignalDirection, SignalType
from .trading import PositionSide, OrderType, OrderStatus


class DatabaseModel(BaseModel):
    """
    Base class for all database-optimized models.
    
    Provides common functionality for database storage including
    optimized serialization, batch operations, and compression support.
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: str,
        },
        # Optimize for database storage
        extra='forbid',
        str_strip_whitespace=True,
        use_enum_values=True
    )
    
    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary optimized for database storage."""
        data = self.model_dump()
        
        # Convert datetime objects to ISO strings for storage
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Decimal):
                data[key] = str(value)
            elif isinstance(value, (list, dict)) and value:
                # Serialize complex objects as JSON strings
                data[key] = json.dumps(value, default=str)
        
        return data
    
    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'DatabaseModel':
        """Create instance from database dictionary."""
        # Convert ISO strings back to datetime objects
        for key, value in data.items():
            if isinstance(value, str) and 'timestamp' in key.lower():
                try:
                    data[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    pass
            elif isinstance(value, str) and key.endswith('_json'):
                try:
                    data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return cls(**data)
    
    def get_batch_key(self) -> str:
        """Get key for batch operations grouping."""
        return f"{self.__class__.__name__}_{getattr(self, 'symbol', 'default')}"


class DBCandlestickData(DatabaseModel):
    """
    Database-optimized candlestick data model.
    
    Optimized for high-frequency storage and time-series queries
    with minimal overhead and efficient indexing support.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol",
        max_length=20
    )
    timestamp_start: datetime = Field(
        ...,
        description="Candle start timestamp (UTC)"
    )
    timestamp_end: datetime = Field(
        ...,
        description="Candle end timestamp (UTC)"
    )
    open_price: str = Field(
        ...,
        description="Opening price as string for precision"
    )
    high_price: str = Field(
        ...,
        description="High price as string for precision"
    )
    low_price: str = Field(
        ...,
        description="Low price as string for precision"
    )
    close_price: str = Field(
        ...,
        description="Closing price as string for precision"
    )
    volume: str = Field(
        ...,
        description="Volume as string for precision"
    )
    trade_count: Optional[int] = Field(
        None,
        description="Number of trades in this candle"
    )
    
    @computed_field
    @property
    def table_name(self) -> str:
        """Get appropriate table name for this timeframe."""
        return f"candles_{self.timestamp_start.strftime('%Y-%m-%d')}"
    
    @computed_field
    @property
    def price_range_pct(self) -> Decimal:
        """Calculate price range as percentage."""
        high_val = Decimal(self.high_price)
        low_val = Decimal(self.low_price)
        if low_val > 0:
            return ((high_val - low_val) / low_val) * Decimal('100')
        return Decimal('0')
    
    def get_compression_data(self) -> Dict[str, Any]:
        """Get data optimized for compression."""
        return {
            'symbol': self.symbol,
            'timestamp_start': self.timestamp_start.isoformat(),
            'timestamp_end': self.timestamp_end.isoformat(),
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'trade_count': self.trade_count
        }


class DBTradeData(DatabaseModel):
    """
    Database-optimized trade data model.
    
    Designed for high-frequency trade storage with efficient
    indexing and compression for historical analysis.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol",
        max_length=20
    )
    timestamp: datetime = Field(
        ...,
        description="Trade timestamp (UTC)"
    )
    price: str = Field(
        ...,
        description="Trade price as string for precision"
    )
    size: str = Field(
        ...,
        description="Trade size as string for precision"
    )
    side: str = Field(
        ...,
        description="Trade side (A=Ask/Sell, B=Bid/Buy)",
        max_length=10
    )
    trade_id: Optional[int] = Field(
        None,
        description="HyperLiquid trade ID"
    )
    hash: Optional[str] = Field(
        None,
        description="Trade hash from HyperLiquid"
    )
    user1: Optional[str] = Field(
        None,
        description="User 1 from HyperLiquid trade data"
    )
    user2: Optional[str] = Field(
        None,
        description="User 2 from HyperLiquid trade data"
    )
    
    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade."""
        return Decimal(self.price) * Decimal(self.size)
    
    @computed_field
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side.lower() == 'buy'
    
    def get_compression_data(self) -> Dict[str, Any]:
        """Get data optimized for compression."""
        return {
            'symbol': self.symbol,
            'time_ms': self.timestamp.timestamp(),
            'price': self.price,
            'size': self.size,
            'side': self.side[0],  # Just 'b' or 's'
            'trade_id': self.trade_id,
            'hash': self.hash,
            'user1': self.user1,
            'user2': self.user2
        }


class DBOrderBookSnapshot(DatabaseModel):
    """
    Database-optimized order book snapshot model.
    
    Optimized for Level 2 order book storage with efficient
    JSON compression and fast retrieval capabilities.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol",
        max_length=20
    )
    timestamp: datetime = Field(
        ...,
        description="Snapshot timestamp (UTC)"
    )
    time_ms: int = Field(
        ...,
        description="Timestamp in milliseconds"
    )
    bids: str = Field(
        ...,
        description="Bid levels as JSON string"
    )
    asks: str = Field(
        ...,
        description="Ask levels as JSON string"
    )
    
    @computed_field
    @property
    def levels(self) -> Dict[str, Any]:
        """Get parsed levels from bids/asks JSON."""
        try:
            return {
                'bids': json.loads(self.bids) if self.bids else [],
                'asks': json.loads(self.asks) if self.asks else []
            }
        except json.JSONDecodeError:
            return {'bids': [], 'asks': []}
    
    @computed_field
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        try:
            bids = json.loads(self.bids) if self.bids else []
            if bids and len(bids) > 0:
                return Decimal(bids[0]['px'])
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            pass
        return None
    
    @computed_field
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        try:
            asks = json.loads(self.asks) if self.asks else []
            if asks and len(asks) > 0:
                return Decimal(asks[0]['px'])
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            pass
        return None
    
    @computed_field
    @property
    def spread_bps(self) -> Optional[Decimal]:
        """Calculate spread in basis points."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        if best_bid and best_ask and best_bid > 0:
            spread = (best_ask - best_bid) / best_bid * Decimal('10000')
            return spread
        return None
    
    def get_compression_data(self) -> Dict[str, Any]:
        """Get data optimized for compression."""
        return {
            'symbol': self.symbol,
            'time_ms': self.time_ms,
            'bids': self.bids,
            'asks': self.asks
        }


class DBFundingRateData(DatabaseModel):
    """
    Database-optimized funding rate data model.
    
    Efficient storage for funding rate history with
    decimal precision and time-series optimization.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol",
        max_length=20
    )
    timestamp: datetime = Field(
        ...,
        description="Funding rate timestamp (UTC)"
    )
    time_ms: int = Field(
        ...,
        description="Timestamp in milliseconds"
    )
    funding_rate: str = Field(
        ...,
        description="Funding rate as string for precision"
    )
    premium: Optional[str] = Field(
        None,
        description="Funding premium as string for precision"
    )
    
    @computed_field
    @property
    def annual_rate_pct(self) -> Decimal:
        """Calculate annualized funding rate percentage."""
        rate = Decimal(self.funding_rate)
        # Funding is typically 8-hourly, so 3 times per day
        return rate * Decimal('365') * Decimal('3') * Decimal('100')
    
    @computed_field
    @property
    def is_positive(self) -> bool:
        """Check if funding rate is positive."""
        return Decimal(self.funding_rate) > 0
    
    def get_compression_data(self) -> Dict[str, Any]:
        """Get data optimized for compression."""
        return {
            'symbol': self.symbol,
            'time_ms': self.time_ms,
            'rate': self.funding_rate,
            'premium': self.premium
        }


class DBTradingSignal(DatabaseModel):
    """
    Database-optimized trading signal model.
    
    Efficient storage for trading signals with metadata
    compression and performance tracking capabilities.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    signal_id: str = Field(
        ...,
        description="Unique signal identifier",
        max_length=100
    )
    symbol: str = Field(
        ...,
        description="Trading symbol",
        max_length=20
    )
    direction: str = Field(
        ...,
        description="Signal direction",
        max_length=20
    )
    signal_type: str = Field(
        ...,
        description="Signal type",
        max_length=30
    )
    confidence: str = Field(
        ...,
        description="Signal confidence as string"
    )
    strength: str = Field(
        ...,
        description="Signal strength as string"
    )
    price: str = Field(
        ...,
        description="Signal price as string"
    )
    target_price: Optional[str] = Field(
        None,
        description="Target price as string"
    )
    stop_loss: Optional[str] = Field(
        None,
        description="Stop loss price as string"
    )
    timeframe: str = Field(
        ...,
        description="Signal timeframe",
        max_length=10
    )
    timestamp: datetime = Field(
        ...,
        description="Signal generation timestamp"
    )
    expiry: Optional[datetime] = Field(
        None,
        description="Signal expiry timestamp"
    )
    source: str = Field(
        ...,
        description="Signal source",
        max_length=100
    )
    reason: str = Field(
        ...,
        description="Signal reasoning"
    )
    metadata_json: Optional[str] = Field(
        None,
        description="Signal metadata as JSON"
    )
    is_active: bool = Field(
        default=True,
        description="Whether signal is active"
    )
    
    @computed_field
    @property
    def metadata(self) -> Dict[str, Any]:
        """Parse metadata from JSON string."""
        if not self.metadata_json:
            return {}
        try:
            return json.loads(self.metadata_json)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    @computed_field
    @property
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """Calculate risk/reward ratio."""
        if not self.target_price or not self.stop_loss:
            return None
        
        target = Decimal(self.target_price)
        stop = Decimal(self.stop_loss)
        price = Decimal(self.price)
        
        if self.direction.lower() in ['buy', 'strong_buy']:
            reward = abs(target - price)
            risk = abs(price - stop)
        else:
            reward = abs(price - target)
            risk = abs(stop - price)
        
        if risk == 0:
            return None
        
        return reward / risk
    
    def get_compression_data(self) -> Dict[str, Any]:
        """Get data optimized for compression."""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction[0].upper(),  # Just first letter
            'type': self.signal_type,
            'confidence': self.confidence,
            'strength': self.strength,
            'price': self.price,
            'target': self.target_price,
            'stop': self.stop_loss,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'active': self.is_active
        }


class DBPosition(DatabaseModel):
    """
    Database-optimized position model.
    
    Efficient storage for trading positions with
    PnL tracking and performance analytics.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    position_id: str = Field(
        ...,
        description="Unique position identifier",
        max_length=100
    )
    symbol: str = Field(
        ...,
        description="Trading symbol",
        max_length=20
    )
    side: str = Field(
        ...,
        description="Position side (long/short)",
        max_length=10
    )
    size: str = Field(
        ...,
        description="Position size as string"
    )
    entry_price: str = Field(
        ...,
        description="Average entry price as string"
    )
    current_price: Optional[str] = Field(
        None,
        description="Current market price as string"
    )
    unrealized_pnl: Optional[str] = Field(
        None,
        description="Unrealized PnL as string"
    )
    realized_pnl: str = Field(
        default="0",
        description="Realized PnL as string"
    )
    margin_used: str = Field(
        ...,
        description="Margin used as string"
    )
    timestamp: datetime = Field(
        ...,
        description="Position open timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    is_open: bool = Field(
        default=True,
        description="Whether position is open"
    )
    
    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of position."""
        return Decimal(self.size) * Decimal(self.entry_price)
    
    @computed_field
    @property
    def leverage(self) -> Decimal:
        """Calculate position leverage."""
        notional = self.notional_value
        margin = Decimal(self.margin_used)
        if margin > 0:
            return notional / margin
        return Decimal('1')
    
    @computed_field
    @property
    def pnl_percentage(self) -> Optional[Decimal]:
        """Calculate PnL percentage."""
        if not self.current_price:
            return None
        
        entry = Decimal(self.entry_price)
        current = Decimal(self.current_price)
        
        if self.side.lower() == 'long':
            return ((current - entry) / entry) * Decimal('100')
        else:
            return ((entry - current) / entry) * Decimal('100')
    
    def get_compression_data(self) -> Dict[str, Any]:
        """Get data optimized for compression."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side[0],  # 'l' or 's'
            'size': self.size,
            'entry': self.entry_price,
            'current': self.current_price,
            'upnl': self.unrealized_pnl,
            'rpnl': self.realized_pnl,
            'margin': self.margin_used,
            'open': self.is_open
        }


class DBBatchOperation(DatabaseModel):
    """
    Database batch operation tracking model.
    
    Tracks batch insertions and updates for performance
    monitoring and data integrity verification.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    batch_id: str = Field(
        ...,
        description="Unique batch identifier",
        max_length=100
    )
    operation_type: str = Field(
        ...,
        description="Type of batch operation",
        max_length=20
    )
    table_name: str = Field(
        ...,
        description="Target table name",
        max_length=50
    )
    record_count: int = Field(
        ...,
        description="Number of records processed"
    )
    start_time: datetime = Field(
        ...,
        description="Batch start timestamp"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="Batch completion timestamp"
    )
    status: str = Field(
        default="running",
        description="Batch operation status",
        max_length=20
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if batch failed"
    )
    metrics_json: Optional[str] = Field(
        None,
        description="Performance metrics as JSON"
    )
    
    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate batch duration in seconds."""
        if not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    @computed_field
    @property
    def records_per_second(self) -> Optional[float]:
        """Calculate processing rate."""
        duration = self.duration_seconds
        if duration and duration > 0:
            return self.record_count / duration
        return None
    
    @computed_field
    @property
    def metrics(self) -> Dict[str, Any]:
        """Parse metrics from JSON string."""
        if not self.metrics_json:
            return {}
        try:
            return json.loads(self.metrics_json)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def mark_completed(self, success: bool = True, error_msg: Optional[str] = None) -> None:
        """Mark batch operation as completed."""
        self.end_time = datetime.now(timezone.utc)
        self.status = "completed" if success else "failed"
        if error_msg:
            self.error_message = error_msg


class DBArchiveRecord(DatabaseModel):
    """
    Database archive record model.
    
    Tracks archived data for long-term storage management
    and data lifecycle compliance.
    """
    
    id: Optional[int] = Field(
        None,
        description="Database sequence ID"
    )
    archive_id: str = Field(
        ...,
        description="Unique archive identifier",
        max_length=100
    )
    table_name: str = Field(
        ...,
        description="Source table name",
        max_length=50
    )
    date_range_start: datetime = Field(
        ...,
        description="Start of archived date range"
    )
    date_range_end: datetime = Field(
        ...,
        description="End of archived date range"
    )
    record_count: int = Field(
        ...,
        description="Number of archived records"
    )
    compressed_size_bytes: int = Field(
        ...,
        description="Compressed archive size in bytes"
    )
    original_size_bytes: int = Field(
        ...,
        description="Original data size in bytes"
    )
    archive_path: str = Field(
        ...,
        description="Path to archive file"
    )
    checksum: str = Field(
        ...,
        description="Archive file checksum",
        max_length=64
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Archive creation timestamp"
    )
    compression_algorithm: str = Field(
        default="zstd",
        description="Compression algorithm used",
        max_length=20
    )
    retention_policy: str = Field(
        ...,
        description="Data retention policy applied",
        max_length=50
    )
    
    @computed_field
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_size_bytes > 0:
            return self.compressed_size_bytes / self.original_size_bytes
        return 0.0
    
    @computed_field
    @property
    def space_saved_mb(self) -> float:
        """Calculate space saved in MB."""
        saved_bytes = self.original_size_bytes - self.compressed_size_bytes
        return saved_bytes / (1024 * 1024)
    
    @computed_field
    @property
    def age_days(self) -> int:
        """Calculate archive age in days."""
        return (datetime.now(timezone.utc) - self.created_at).days 