"""
Trading Operation Models

This module contains Pydantic models for trading operations including:
- Position: Position management with size, entry price, and PnL calculations
- Order: Order models with type, status, and execution tracking
- TradeExecution: Models for completed trade executions
- PortfolioState: Account overview and portfolio management
- Risk Parameters: Risk management models and validation
- Position Sizing: Position sizing and margin calculations

All models are designed for compatibility with HyperLiquid's trading API
and include comprehensive validation for real-time trading operations.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict
from enum import Enum

from .market_data import Timeframe


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stopMarket"
    STOP_LIMIT = "stopLimit"
    TAKE_PROFIT = "takeProfit"
    TAKE_PROFIT_LIMIT = "takeProfitLimit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partiallyFilled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


class Position(BaseModel):
    """
    Trading position model with PnL calculations and risk metrics.
    
    Represents an open or closed position with entry/exit tracking,
    real-time PnL calculations, and risk management features.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol for this position"
    )
    side: PositionSide = Field(
        ...,
        description="Position side (long or short)"
    )
    size: Decimal = Field(
        ...,
        description="Position size (positive for long, negative for short in some contexts)",
        ge=Decimal('0')
    )
    entry_price: Decimal = Field(
        ...,
        description="Average entry price for the position",
        gt=Decimal('0')
    )
    current_price: Optional[Decimal] = Field(
        None,
        description="Current market price for PnL calculation",
        gt=Decimal('0')
    )
    leverage: Decimal = Field(
        default=Decimal('1'),
        description="Position leverage",
        ge=Decimal('1'),
        le=Decimal('50')
    )
    margin_used: Optional[Decimal] = Field(
        None,
        description="Margin used for this position",
        ge=Decimal('0')
    )
    funding_rate: Optional[Decimal] = Field(
        None,
        description="Current funding rate for the position"
    )
    entry_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when position was opened"
    )
    exit_timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp when position was closed"
    )
    exit_price: Optional[Decimal] = Field(
        None,
        description="Exit price if position is closed",
        gt=Decimal('0')
    )
    is_open: bool = Field(
        default=True,
        description="Whether the position is currently open"
    )
    stop_loss: Optional[Decimal] = Field(
        None,
        description="Stop loss price",
        gt=Decimal('0')
    )
    take_profit: Optional[Decimal] = Field(
        None,
        description="Take profit price",
        gt=Decimal('0')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the position."""
        price = self.current_price or self.entry_price
        return self.size * price
    
    @computed_field
    @property
    def unrealized_pnl(self) -> Optional[Decimal]:
        """Calculate unrealized PnL if current price is available."""
        if not self.current_price or not self.is_open:
            return None
        
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.size
    
    @computed_field
    @property
    def unrealized_pnl_percent(self) -> Optional[Decimal]:
        """Calculate unrealized PnL percentage."""
        if not self.current_price or not self.is_open:
            return None
        
        if self.side == PositionSide.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * Decimal('100')
        else:  # SHORT
            return ((self.entry_price - self.current_price) / self.entry_price) * Decimal('100')
    
    @computed_field
    @property
    def realized_pnl(self) -> Optional[Decimal]:
        """Calculate realized PnL if position is closed."""
        if self.is_open or not self.exit_price:
            return None
        
        if self.side == PositionSide.LONG:
            return (self.exit_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.size
    
    @computed_field
    @property
    def realized_pnl_percent(self) -> Optional[Decimal]:
        """Calculate realized PnL percentage."""
        if self.is_open or not self.exit_price:
            return None
        
        if self.side == PositionSide.LONG:
            return ((self.exit_price - self.entry_price) / self.entry_price) * Decimal('100')
        else:  # SHORT
            return ((self.entry_price - self.exit_price) / self.entry_price) * Decimal('100')
    
    @computed_field
    @property
    def margin_requirement(self) -> Decimal:
        """Calculate margin requirement for the position."""
        if self.margin_used is not None:
            return self.margin_used
        
        # Calculate based on notional value and leverage
        return self.notional_value / self.leverage
    
    @computed_field
    @property
    def liquidation_price(self) -> Optional[Decimal]:
        """Estimate liquidation price (simplified calculation)."""
        if not self.margin_used:
            return None
        
        # Simplified liquidation price calculation
        # In reality, this would need to account for funding rates, fees, etc.
        maintenance_margin_rate = Decimal('0.005')  # 0.5% maintenance margin
        
        if self.side == PositionSide.LONG:
            # For long: liquidation when equity falls below maintenance margin
            loss_to_liquidation = self.margin_used - (self.notional_value * maintenance_margin_rate)
            return self.entry_price - (loss_to_liquidation / self.size)
        else:  # SHORT
            # For short: liquidation when price rises too much
            loss_to_liquidation = self.margin_used - (self.notional_value * maintenance_margin_rate)
            return self.entry_price + (loss_to_liquidation / self.size)
    
    def update_price(self, new_price: Decimal) -> None:
        """Update current price for PnL calculations."""
        self.current_price = new_price
    
    def close_position(self, exit_price: Decimal, exit_timestamp: Optional[datetime] = None) -> None:
        """Close the position with the given exit price."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp or datetime.now(timezone.utc)
        self.is_open = False
        self.current_price = exit_price
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'Position':
        """
        Create Position from HyperLiquid user info data.
        
        Expected format: {
            'position': {
                'coin': 'BTC',
                'szi': '1.5',  # Size (positive for long, negative for short)
                'entryPx': '50000.0',
                'positionValue': '75000.0',
                'unrealizedPnl': '1500.0',
                'leverage': {
                    'type': 'cross',
                    'value': 5
                }
            }
        }
        """
        position_data = data.get('position', data)
        
        # Determine side and size
        size_str = position_data.get('szi', '0')
        size = abs(Decimal(size_str))
        side = PositionSide.LONG if Decimal(size_str) >= 0 else PositionSide.SHORT
        
        leverage_data = position_data.get('leverage', {})
        leverage = Decimal(str(leverage_data.get('value', 1)))
        
        return cls(
            symbol=position_data.get('coin', 'UNKNOWN'),
            side=side,
            size=size,
            entry_price=Decimal(position_data.get('entryPx', '0')),
            leverage=leverage,
            margin_used=position_data.get('marginUsed') and Decimal(position_data['marginUsed']),
            is_open=size > 0,
        )


class Order(BaseModel):
    """
    Trading order model with type, status, and execution tracking.
    
    Represents a trading order with all necessary parameters for
    execution, tracking, and validation.
    """
    
    order_id: Optional[str] = Field(
        None,
        description="Unique order identifier from exchange"
    )
    client_order_id: Optional[str] = Field(
        None,
        description="Client-provided order identifier"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol for this order"
    )
    side: OrderSide = Field(
        ...,
        description="Order side (buy or sell)"
    )
    order_type: OrderType = Field(
        ...,
        description="Order type (market, limit, etc.)"
    )
    quantity: Decimal = Field(
        ...,
        description="Order quantity",
        gt=Decimal('0')
    )
    price: Optional[Decimal] = Field(
        None,
        description="Order price (required for limit orders)",
        gt=Decimal('0')
    )
    stop_price: Optional[Decimal] = Field(
        None,
        description="Stop price for stop orders",
        gt=Decimal('0')
    )
    time_in_force: TimeInForce = Field(
        default=TimeInForce.GTC,
        description="Time in force specification"
    )
    status: OrderStatus = Field(
        default=OrderStatus.PENDING,
        description="Current order status"
    )
    filled_quantity: Decimal = Field(
        default=Decimal('0'),
        description="Quantity filled so far",
        ge=Decimal('0')
    )
    average_fill_price: Optional[Decimal] = Field(
        None,
        description="Average price of fills",
        gt=Decimal('0')
    )
    created_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Order creation timestamp"
    )
    updated_timestamp: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    reduce_only: bool = Field(
        default=False,
        description="Whether this is a reduce-only order"
    )
    post_only: bool = Field(
        default=False,
        description="Whether this is a post-only order"
    )
    leverage: Optional[Decimal] = Field(
        None,
        description="Leverage for this order",
        ge=Decimal('1'),
        le=Decimal('50')
    )
    commission: Optional[Decimal] = Field(
        None,
        description="Commission paid on fills",
        ge=Decimal('0')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @field_validator('price')
    @classmethod
    def validate_price_for_limit_orders(cls, v, values):
        """Validate that limit orders have a price."""
        if hasattr(values, 'data'):
            order_type = values.data.get('order_type')
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                if v is None:
                    raise ValueError(f"Price is required for {order_type} orders")
        return v
    
    @computed_field
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    @computed_field
    @property
    def fill_percentage(self) -> Decimal:
        """Calculate percentage of order filled."""
        if self.quantity == 0:
            return Decimal('0')
        return (self.filled_quantity / self.quantity) * Decimal('100')
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity
    
    @computed_field
    @property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value of the order."""
        if self.price:
            return self.quantity * self.price
        return None
    
    @computed_field
    @property
    def filled_notional_value(self) -> Optional[Decimal]:
        """Calculate notional value of filled portion."""
        if self.average_fill_price:
            return self.filled_quantity * self.average_fill_price
        return None
    
    def update_fill(self, fill_quantity: Decimal, fill_price: Decimal, commission: Decimal = Decimal('0')) -> None:
        """Update order with a new fill."""
        self.filled_quantity += fill_quantity
        
        # Update average fill price
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            # Weighted average
            total_value = (self.filled_quantity - fill_quantity) * self.average_fill_price + fill_quantity * fill_price
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update commission
        if self.commission is None:
            self.commission = commission
        else:
            self.commission += commission
        
        # Update status
        if self.is_complete:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_timestamp = datetime.now(timezone.utc)
    
    def cancel(self) -> None:
        """Cancel the order."""
        if self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
            self.status = OrderStatus.CANCELLED
            self.updated_timestamp = datetime.now(timezone.utc)
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'Order':
        """
        Create Order from HyperLiquid order data.
        
        Expected format: {
            'order': {
                'oid': 'order_123',
                'cloid': 'client_order_123',
                'coin': 'BTC',
                'side': 'B',  # B for buy, A for sell
                'orderType': 'Limit',
                'sz': '1.0',
                'px': '50000.0',
                'timestamp': 1705316200000
            }
        }
        """
        order_data = data.get('order', data)
        
        # Map HyperLiquid side to our enum
        hl_side = order_data.get('side', 'B')
        side = OrderSide.BUY if hl_side == 'B' else OrderSide.SELL
        
        # Map HyperLiquid order type
        hl_order_type = order_data.get('orderType', 'Limit').lower()
        order_type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stopmarket': OrderType.STOP_MARKET,
            'stoplimit': OrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(hl_order_type, OrderType.LIMIT)
        
        # Convert timestamp
        timestamp = order_data.get('timestamp')
        created_timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc) if timestamp else datetime.now(timezone.utc)
        
        return cls(
            order_id=order_data.get('oid'),
            client_order_id=order_data.get('cloid'),
            symbol=order_data.get('coin', 'UNKNOWN'),
            side=side,
            order_type=order_type,
            quantity=Decimal(order_data.get('sz', '0')),
            price=order_data.get('px') and Decimal(order_data['px']),
            created_timestamp=created_timestamp,
            status=OrderStatus.OPEN,  # Default for new orders from exchange
        )


class TradeExecution(BaseModel):
    """
    Model for completed trade executions.
    
    Represents a trade that has been executed on the exchange,
    either as a full order fill or partial fill.
    """
    
    execution_id: str = Field(
        ...,
        description="Unique execution identifier"
    )
    order_id: Optional[str] = Field(
        None,
        description="Order ID that generated this execution"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    side: OrderSide = Field(
        ...,
        description="Execution side (buy or sell)"
    )
    quantity: Decimal = Field(
        ...,
        description="Executed quantity",
        gt=Decimal('0')
    )
    price: Decimal = Field(
        ...,
        description="Execution price",
        gt=Decimal('0')
    )
    timestamp: datetime = Field(
        ...,
        description="Execution timestamp"
    )
    commission: Decimal = Field(
        default=Decimal('0'),
        description="Commission paid",
        ge=Decimal('0')
    )
    commission_asset: str = Field(
        default="USD",
        description="Asset used for commission payment"
    )
    is_maker: bool = Field(
        default=False,
        description="Whether this execution was a maker trade"
    )
    liquidity: Literal["maker", "taker"] = Field(
        default="taker",
        description="Liquidity provision type"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the execution."""
        return self.quantity * self.price
    
    @computed_field
    @property
    def net_proceeds(self) -> Decimal:
        """Calculate net proceeds after commission."""
        gross_proceeds = self.notional_value
        if self.side == OrderSide.BUY:
            return gross_proceeds + self.commission  # Commission adds to cost for buys
        else:
            return gross_proceeds - self.commission  # Commission reduces proceeds for sells
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'TradeExecution':
        """
        Create TradeExecution from HyperLiquid fill data.
        
        Expected format: {
            'fill': {
                'tid': 'execution_123',
                'oid': 'order_123',
                'coin': 'BTC',
                'side': 'B',
                'sz': '1.0',
                'px': '50000.0',
                'time': 1705316200000,
                'fee': '25.0'
            }
        }
        """
        fill_data = data.get('fill', data)
        
        # Map HyperLiquid side to our enum
        hl_side = fill_data.get('side', 'B')
        side = OrderSide.BUY if hl_side == 'B' else OrderSide.SELL
        
        # Convert timestamp
        timestamp = fill_data.get('time')
        execution_timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc) if timestamp else datetime.now(timezone.utc)
        
        return cls(
            execution_id=fill_data.get('tid', ''),
            order_id=fill_data.get('oid'),
            symbol=fill_data.get('coin', 'UNKNOWN'),
            side=side,
            quantity=Decimal(fill_data.get('sz', '0')),
            price=Decimal(fill_data.get('px', '0')),
            timestamp=execution_timestamp,
            commission=Decimal(fill_data.get('fee', '0')),
        )


class RiskParameters(BaseModel):
    """
    Risk management parameters and validation.
    
    Defines risk limits and parameters for trading operations
    including position sizing, leverage limits, and exposure controls.
    """
    
    max_position_size: Decimal = Field(
        ...,
        description="Maximum position size per symbol",
        gt=Decimal('0')
    )
    max_leverage: Decimal = Field(
        default=Decimal('10'),
        description="Maximum allowed leverage",
        ge=Decimal('1'),
        le=Decimal('50')
    )
    max_portfolio_exposure: Decimal = Field(
        default=Decimal('100000'),
        description="Maximum total portfolio exposure",
        gt=Decimal('0')
    )
    stop_loss_percentage: Decimal = Field(
        default=Decimal('5'),
        description="Default stop loss percentage",
        gt=Decimal('0'),
        le=Decimal('50')
    )
    take_profit_percentage: Decimal = Field(
        default=Decimal('10'),
        description="Default take profit percentage",
        gt=Decimal('0')
    )
    max_drawdown_percentage: Decimal = Field(
        default=Decimal('20'),
        description="Maximum allowed drawdown percentage",
        gt=Decimal('0'),
        le=Decimal('100')
    )
    daily_loss_limit: Decimal = Field(
        default=Decimal('5000'),
        description="Daily loss limit",
        gt=Decimal('0')
    )
    risk_free_rate: Decimal = Field(
        default=Decimal('0.05'),
        description="Risk-free rate for calculations",
        ge=Decimal('0')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    def validate_position_size(self, size: Decimal, price: Decimal) -> bool:
        """Validate if position size is within limits."""
        notional_value = size * price
        return notional_value <= self.max_position_size
    
    def validate_leverage(self, leverage: Decimal) -> bool:
        """Validate if leverage is within limits."""
        return leverage <= self.max_leverage
    
    def calculate_max_position_size(self, price: Decimal, leverage: Decimal = Decimal('1')) -> Decimal:
        """Calculate maximum position size given price and leverage."""
        max_notional = self.max_position_size * leverage
        return max_notional / price
    
    def calculate_stop_loss_price(self, entry_price: Decimal, side: PositionSide) -> Decimal:
        """Calculate stop loss price based on default percentage."""
        if side == PositionSide.LONG:
            return entry_price * (Decimal('1') - self.stop_loss_percentage / Decimal('100'))
        else:  # SHORT
            return entry_price * (Decimal('1') + self.stop_loss_percentage / Decimal('100'))
    
    def calculate_take_profit_price(self, entry_price: Decimal, side: PositionSide) -> Decimal:
        """Calculate take profit price based on default percentage."""
        if side == PositionSide.LONG:
            return entry_price * (Decimal('1') + self.take_profit_percentage / Decimal('100'))
        else:  # SHORT
            return entry_price * (Decimal('1') - self.take_profit_percentage / Decimal('100'))


class PortfolioState(BaseModel):
    """
    Portfolio state and account overview.
    
    Provides a comprehensive view of the trading account including
    balance, positions, exposure, and performance metrics.
    """
    
    account_id: str = Field(
        ...,
        description="Account identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Portfolio state timestamp"
    )
    total_balance: Decimal = Field(
        ...,
        description="Total account balance",
        ge=Decimal('0')
    )
    available_balance: Decimal = Field(
        ...,
        description="Available balance for trading",
        ge=Decimal('0')
    )
    margin_used: Decimal = Field(
        default=Decimal('0'),
        description="Total margin used",
        ge=Decimal('0')
    )
    unrealized_pnl: Decimal = Field(
        default=Decimal('0'),
        description="Total unrealized PnL across all positions"
    )
    realized_pnl: Decimal = Field(
        default=Decimal('0'),
        description="Total realized PnL for the period"
    )
    positions: List[Position] = Field(
        default_factory=list,
        description="List of open positions"
    )
    pending_orders: List[Order] = Field(
        default_factory=list,
        description="List of pending orders"
    )
    daily_pnl: Decimal = Field(
        default=Decimal('0'),
        description="PnL for current trading day"
    )
    fees_paid: Decimal = Field(
        default=Decimal('0'),
        description="Total fees paid",
        ge=Decimal('0')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def equity(self) -> Decimal:
        """Calculate total equity (balance + unrealized PnL)."""
        return self.total_balance + self.unrealized_pnl
    
    @computed_field
    @property
    def margin_level(self) -> Optional[Decimal]:
        """Calculate margin level percentage."""
        if self.margin_used == 0:
            return None
        return (self.equity / self.margin_used) * Decimal('100')
    
    @computed_field
    @property
    def total_exposure(self) -> Decimal:
        """Calculate total portfolio exposure."""
        return sum(position.notional_value for position in self.positions)
    
    @computed_field
    @property
    def leverage_ratio(self) -> Decimal:
        """Calculate overall portfolio leverage ratio."""
        if self.equity == 0:
            return Decimal('0')
        return self.total_exposure / self.equity
    
    @computed_field
    @property
    def open_position_count(self) -> int:
        """Count of open positions."""
        return len([pos for pos in self.positions if pos.is_open])
    
    @computed_field
    @property
    def long_exposure(self) -> Decimal:
        """Calculate total long exposure."""
        return sum(
            pos.notional_value 
            for pos in self.positions 
            if pos.side == PositionSide.LONG and pos.is_open
        )
    
    @computed_field
    @property
    def short_exposure(self) -> Decimal:
        """Calculate total short exposure."""
        return sum(
            pos.notional_value 
            for pos in self.positions 
            if pos.side == PositionSide.SHORT and pos.is_open
        )
    
    @computed_field
    @property
    def net_exposure(self) -> Decimal:
        """Calculate net exposure (long - short)."""
        return self.long_exposure - self.short_exposure
    
    def update_from_positions(self) -> None:
        """Update portfolio state from current positions."""
        self.unrealized_pnl = sum(
            pos.unrealized_pnl or Decimal('0') 
            for pos in self.positions 
            if pos.is_open
        )
        
        self.margin_used = sum(
            pos.margin_requirement 
            for pos in self.positions 
            if pos.is_open
        )
    
    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        self.positions.append(position)
        self.update_from_positions()
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position from the portfolio."""
        self.positions = [pos for pos in self.positions if pos.symbol != symbol]
        self.update_from_positions()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        for position in self.positions:
            if position.symbol == symbol and position.is_open:
                return position
        return None
    
    @classmethod
    def from_hyperliquid(cls, data: Dict[str, Any]) -> 'PortfolioState':
        """
        Create PortfolioState from HyperLiquid user info data.
        
        Expected format: {
            'marginSummary': {
                'accountValue': '100000.0',
                'totalMarginUsed': '25000.0',
                'totalNtlPos': '125000.0',
                'totalUnrealizedPnl': '1500.0'
            },
            'assetPositions': [...]
        }
        """
        margin_summary = data.get('marginSummary', {})
        
        # Extract positions if available
        positions = []
        for pos_data in data.get('assetPositions', []):
            try:
                position = Position.from_hyperliquid(pos_data)
                positions.append(position)
            except Exception:
                continue  # Skip invalid position data
        
        return cls(
            account_id=data.get('user', 'unknown'),
            total_balance=Decimal(margin_summary.get('accountValue', '0')),
            available_balance=Decimal(margin_summary.get('accountValue', '0')) - Decimal(margin_summary.get('totalMarginUsed', '0')),
            margin_used=Decimal(margin_summary.get('totalMarginUsed', '0')),
            unrealized_pnl=Decimal(margin_summary.get('totalUnrealizedPnl', '0')),
            positions=positions,
        ) 