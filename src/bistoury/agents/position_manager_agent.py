"""
Position Manager Agent - Task 11 Implementation

The Position Manager executes trades and manages positions:
- Executes trades based on signals from strategy agents  
- Manages open positions with P&L tracking
- Implements stop-loss and take-profit management
- Portfolio state monitoring and reporting
- Order execution with slippage simulation
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from uuid import uuid4
from pydantic import BaseModel

from .base import BaseAgent, AgentType, AgentState
from ..models.trading import (
    Position, PositionSide, Order, OrderType, OrderStatus, OrderSide,
    TradeExecution, PortfolioState, RiskParameters, TimeInForce
)
from ..models.agent_messages import (
    Message, MessageType, MessagePriority, MessageFilter,
    TradingSignalPayload, SystemEventPayload
)


class PositionManagerConfig:
    """Configuration for Position Manager."""
    
    def __init__(
        self,
        initial_balance: Decimal = Decimal('100000'),
        slippage_rate: Decimal = Decimal('0.0005'),
        commission_rate: Decimal = Decimal('0.0005'),
        min_position_size: Decimal = Decimal('10'),
        max_position_size: Decimal = Decimal('10000'),
        enable_stop_loss: bool = True,
        enable_take_profit: bool = True,
        stop_loss_pct: Decimal = Decimal('2.0'),
        take_profit_pct: Decimal = Decimal('4.0')
    ):
        self.initial_balance = initial_balance
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.enable_stop_loss = enable_stop_loss
        self.enable_take_profit = enable_take_profit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct


class PositionManagerAgent(BaseAgent):
    """Position Manager Agent for trade execution and position management."""
    
    def __init__(
        self,
        name: str = "position_manager",
        config: Optional[PositionManagerConfig] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.POSITION_MANAGER,
            **kwargs
        )
        
        self.config = config or PositionManagerConfig()
        
        # Portfolio state
        self.portfolio = PortfolioState(
            account_id=self.name,
            total_balance=self.config.initial_balance,
            available_balance=self.config.initial_balance
        )
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.executions: List[TradeExecution] = []
        self.current_prices: Dict[str, Decimal] = {}
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = Decimal('0')
        
        # Control flags
        self._monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"Position Manager initialized with balance: {self.config.initial_balance}")
    
    async def _start(self) -> bool:
        """Start the position manager agent."""
        try:
            await self._setup_subscriptions()
            self._monitor_task = self.create_task(self._monitor_positions())
            self._set_state(AgentState.RUNNING)
            self.logger.info("Position Manager started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Position Manager: {e}", exc_info=True)
            return False
    
    async def _stop(self) -> None:
        """Stop the position manager agent."""
        try:
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            self._set_state(AgentState.STOPPED)
            self.logger.info("Position Manager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping Position Manager: {e}", exc_info=True)
    
    async def _health_check(self) -> None:
        """Perform health check."""
        try:
            healthy = (
                self.portfolio.total_balance > 0 and
                len(self.positions) <= 10
            )
            self.health.health_score = 1.0 if healthy else 0.5
            self.health.messages_processed = self.total_trades
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            self.health.health_score = 0.0
    
    async def _setup_subscriptions(self) -> None:
        """Set up message subscriptions."""
        if not self._message_bus:
            return
        
        # Subscribe to trading signals
        await self._message_bus.subscribe(
            agent_id=self.agent_id,
            filter=MessageFilter(
                message_types=[MessageType.SIGNAL_GENERATED],
                topics=["trading.signals"]
            ),
            handler=self._handle_trading_signal,
            is_async=True
        )
        
        # Subscribe to market data
        await self._message_bus.subscribe(
            agent_id=self.agent_id,
            filter=MessageFilter(
                message_types=[MessageType.DATA_PRICE_UPDATE],
                topics=["market.prices"]
            ),
            handler=self._handle_market_data,
            is_async=True
        )
    
    async def _handle_trading_signal(self, message: Message) -> None:
        """Handle incoming trading signals."""
        try:
            if not isinstance(message.payload, TradingSignalPayload):
                return
            
            signal = message.payload
            self.logger.info(f"Received signal: {signal.direction} {signal.symbol} ({signal.confidence})")
            
            if signal.confidence < 0.6:
                return
            
            await self._execute_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {e}", exc_info=True)
    
    async def _handle_market_data(self, message: Message) -> None:
        """Handle market data updates."""
        try:
            payload = message.payload
            if hasattr(payload, 'symbol') and hasattr(payload, 'price'):
                symbol = payload.symbol
                price = Decimal(str(payload.price))
                self.current_prices[symbol] = price
                
                # Update position prices
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.update_price(price)
                    await self._check_stop_take_profit(position)
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}", exc_info=True)
    
    async def _execute_signal(self, signal: TradingSignalPayload) -> None:
        """Execute a trading signal."""
        try:
            symbol = signal.symbol
            
            # Determine order side
            if signal.direction.upper() == "BUY":
                order_side = OrderSide.BUY
                position_side = PositionSide.LONG
            elif signal.direction.upper() == "SELL":
                order_side = OrderSide.SELL
                position_side = PositionSide.SHORT
            else:
                self.logger.warning(f"Unknown signal direction: {signal.direction}")
                return
            
            # Get current price
            current_price = self.current_prices.get(symbol)
            if not current_price:
                self.logger.warning(f"No current price for {symbol}")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, current_price)
            if position_size <= 0:
                return
            
            # Close existing opposite position
            existing_position = self.positions.get(symbol)
            if existing_position and existing_position.is_open:
                if (existing_position.side == PositionSide.LONG and order_side == OrderSide.SELL) or \
                   (existing_position.side == PositionSide.SHORT and order_side == OrderSide.BUY):
                    await self._close_position(symbol, current_price, "signal_reversal")
            
            # Create and execute order
            order = Order(
                client_order_id=str(uuid4()),
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=position_size,
                time_in_force=TimeInForce.IOC
            )
            
            execution = await self._execute_order(order, current_price)
            if execution:
                await self._create_position(execution, signal)
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}", exc_info=True)
    
    def _calculate_position_size(self, signal: TradingSignalPayload, price: Decimal) -> Decimal:
        """Calculate position size based on signal confidence."""
        try:
            # Base 2% of balance, adjusted by confidence
            base_pct = Decimal('0.02')
            confidence_mult = Decimal(str(signal.confidence)) * Decimal('2')
            
            available = self.portfolio.available_balance
            notional = available * base_pct * confidence_mult
            position_size = notional / price
            
            # Apply limits
            position_size = max(position_size, self.config.min_position_size)
            position_size = min(position_size, self.config.max_position_size)
            
            return position_size.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}", exc_info=True)
            return Decimal('0')
    
    async def _execute_order(self, order: Order, market_price: Decimal) -> Optional[TradeExecution]:
        """Execute order with slippage simulation."""
        try:
            # Apply slippage
            if order.side == OrderSide.BUY:
                execution_price = market_price * (Decimal('1') + self.config.slippage_rate)
            else:
                execution_price = market_price * (Decimal('1') - self.config.slippage_rate)
            
            # Calculate commission
            notional = order.quantity * execution_price
            commission = notional * self.config.commission_rate
            
            # Check if we have enough balance for buy orders
            if order.side == OrderSide.BUY:
                cost = notional + commission
                if cost > self.portfolio.available_balance:
                    self.logger.warning(f"Insufficient balance for order: {cost} > {self.portfolio.available_balance}")
                    return None
            
            execution = TradeExecution(
                execution_id=str(uuid4()),
                order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=datetime.now(timezone.utc),
                commission=commission,
                is_maker=False,
                liquidity="taker"
            )
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = execution_price
            order.commission = commission
            
            # Store execution
            self.executions.append(execution)
            self.total_trades += 1
            
            # Update portfolio
            if order.side == OrderSide.BUY:
                cost = notional + commission
                self.portfolio.available_balance = self.portfolio.available_balance - cost
            else:
                proceeds = notional - commission
                self.portfolio.available_balance = self.portfolio.available_balance + proceeds
            
            self.portfolio.fees_paid = self.portfolio.fees_paid + commission
            
            self.logger.info(f"Order executed: {order.side} {order.quantity} {order.symbol} @ {execution_price}")
            return execution
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}", exc_info=True)
            return None
    
    async def _create_position(self, execution: TradeExecution, signal: TradingSignalPayload) -> None:
        """Create new position."""
        try:
            position_side = PositionSide.LONG if execution.side == OrderSide.BUY else PositionSide.SHORT
            
            position = Position(
                symbol=execution.symbol,
                side=position_side,
                size=execution.quantity,
                entry_price=execution.price,
                current_price=execution.price,
                entry_timestamp=execution.timestamp
            )
            
            # Set stop loss and take profit
            if self.config.enable_stop_loss:
                if position_side == PositionSide.LONG:
                    position.stop_loss = execution.price * (Decimal('1') - self.config.stop_loss_pct / 100)
                else:
                    position.stop_loss = execution.price * (Decimal('1') + self.config.stop_loss_pct / 100)
            
            if self.config.enable_take_profit:
                if position_side == PositionSide.LONG:
                    position.take_profit = execution.price * (Decimal('1') + self.config.take_profit_pct / 100)
                else:
                    position.take_profit = execution.price * (Decimal('1') - self.config.take_profit_pct / 100)
            
            self.positions[execution.symbol] = position
            self.portfolio.add_position(position)
            
            self.logger.info(f"Created position: {position_side} {execution.quantity} {execution.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error creating position: {e}", exc_info=True)
    
    async def _check_stop_take_profit(self, position: Position) -> None:
        """Check stop loss and take profit."""
        try:
            if not position.is_open or not position.current_price:
                return
            
            current_price = position.current_price
            should_close = False
            reason = ""
            
            # Check stop loss
            if position.stop_loss:
                if (position.side == PositionSide.LONG and current_price <= position.stop_loss) or \
                   (position.side == PositionSide.SHORT and current_price >= position.stop_loss):
                    should_close = True
                    reason = "stop_loss"
            
            # Check take profit
            if not should_close and position.take_profit:
                if (position.side == PositionSide.LONG and current_price >= position.take_profit) or \
                   (position.side == PositionSide.SHORT and current_price <= position.take_profit):
                    should_close = True
                    reason = "take_profit"
            
            if should_close:
                await self._close_position(position.symbol, current_price, reason)
            
        except Exception as e:
            self.logger.error(f"Error checking stop/take profit: {e}", exc_info=True)
    
    async def _close_position(self, symbol: str, exit_price: Decimal, reason: str) -> None:
        """Close a position."""
        try:
            position = self.positions.get(symbol)
            if not position or not position.is_open:
                return
            
            # Create closing order
            close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            
            close_order = Order(
                client_order_id=str(uuid4()),
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position.size,
                reduce_only=True
            )
            
            execution = await self._execute_order(close_order, exit_price)
            if execution:
                position.close_position(exit_price)
                
                realized_pnl = position.realized_pnl or Decimal('0')
                self.total_pnl += realized_pnl
                
                if realized_pnl > 0:
                    self.winning_trades += 1
                
                self.portfolio.remove_position(symbol)
                self.portfolio.realized_pnl += realized_pnl
                
                self.logger.info(f"Position closed: {symbol} - {reason} - PnL: {realized_pnl}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}", exc_info=True)
    
    async def _monitor_positions(self) -> None:
        """Monitor positions and update portfolio."""
        try:
            while self.state == AgentState.RUNNING:
                # Update unrealized PnL
                total_unrealized = Decimal('0')
                for position in self.positions.values():
                    if position.is_open and position.symbol in self.current_prices:
                        position.update_price(self.current_prices[position.symbol])
                        if position.unrealized_pnl:
                            total_unrealized += position.unrealized_pnl
                
                self.portfolio.unrealized_pnl = total_unrealized
                self.portfolio.timestamp = datetime.now(timezone.utc)
                
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.logger.info("Position monitor cancelled")
        except Exception as e:
            self.logger.error(f"Error in position monitor: {e}", exc_info=True)
    
    # Public API
    async def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        return self.portfolio
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / max(self.total_trades, 1)) * 100,
            'total_pnl': float(self.total_pnl),
            'unrealized_pnl': float(self.portfolio.unrealized_pnl),
            'realized_pnl': float(self.portfolio.realized_pnl),
            'total_balance': float(self.portfolio.total_balance),
            'equity': float(self.portfolio.equity),
            'open_positions': len([p for p in self.positions.values() if p.is_open])
        } 