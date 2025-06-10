"""
Position Manager Agent - Task 11 Implementation

The Position Manager executes trades and manages positions:
- Executes trades based on signals from strategy agents  
- Manages open positions with P&L tracking
- Implements stop-loss and take-profit management
- Portfolio state monitoring and reporting
- Order execution with slippage simulation
- Enhanced with comprehensive performance analytics (Task 13.4)
"""

import asyncio
import math
import statistics
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from pydantic import BaseModel
from dataclasses import dataclass

from .base import BaseAgent, AgentType, AgentState, AgentHealth
from ..models.trading import (
    Position, PositionSide, Order, OrderType, OrderStatus, OrderSide,
    TradeExecution, PortfolioState, RiskParameters, TimeInForce
)
from ..models.agent_messages import (
    Message, MessageType, MessagePriority, MessageFilter,
    TradingSignalPayload, SystemEventPayload
)


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot at a point in time."""
    timestamp: datetime
    total_balance: Decimal
    total_pnl: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    equity: Decimal
    drawdown: Decimal
    drawdown_pct: Decimal


@dataclass
class TradeAnalysis:
    """Comprehensive trade analysis."""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    
    # Timing metrics
    avg_holding_period: timedelta
    avg_win_holding: timedelta
    avg_loss_holding: timedelta
    
    # Risk metrics
    max_consecutive_wins: int
    max_consecutive_losses: int


@dataclass
class AdvancedMetrics:
    """Advanced performance metrics."""
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    
    # Risk-adjusted ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Rolling metrics
    rolling_sharpe_30d: float
    rolling_return_30d: float
    rolling_volatility_30d: float


class PerformanceAnalyzer:
    """Internal performance analytics engine for Position Manager."""
    
    def __init__(self, initial_balance: Decimal):
        self.initial_balance = initial_balance
        self.snapshots: List[PerformanceSnapshot] = []
        self.daily_returns: List[float] = []
        self.peak_equity = initial_balance
        self.max_drawdown = Decimal('0')
        self.max_drawdown_duration = timedelta(0)
        self.drawdown_start: Optional[datetime] = None
        
    def add_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Add performance snapshot."""
        self.snapshots.append(snapshot)
        
        # Update peak and drawdown tracking
        if snapshot.equity > self.peak_equity:
            self.peak_equity = snapshot.equity
            if self.drawdown_start:
                # End of drawdown period
                duration = snapshot.timestamp - self.drawdown_start
                if duration > self.max_drawdown_duration:
                    self.max_drawdown_duration = duration
                self.drawdown_start = None
        elif snapshot.drawdown > self.max_drawdown:
            self.max_drawdown = snapshot.drawdown
            if not self.drawdown_start:
                self.drawdown_start = snapshot.timestamp
        
        # Calculate daily return if we have previous snapshot
        if len(self.snapshots) >= 2:
            prev = self.snapshots[-2]
            if prev.equity > 0:
                daily_return = float((snapshot.equity - prev.equity) / prev.equity)
                self.daily_returns.append(daily_return)
    
    def calculate_drawdown(self, current_equity: Decimal) -> Tuple[Decimal, Decimal]:
        """Calculate current drawdown."""
        if self.peak_equity <= 0:
            return Decimal('0'), Decimal('0')
        
        drawdown = self.peak_equity - current_equity
        drawdown_pct = (drawdown / self.peak_equity) * 100
        return max(drawdown, Decimal('0')), drawdown_pct
    
    def get_trade_analysis(self, executions: List[TradeExecution]) -> TradeAnalysis:
        """Analyze trade executions."""
        if not executions:
            return TradeAnalysis(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                total_pnl=0.0, avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
                profit_factor=0.0, avg_holding_period=timedelta(0),
                avg_win_holding=timedelta(0), avg_loss_holding=timedelta(0),
                max_consecutive_wins=0, max_consecutive_losses=0
            )
        
        # Group executions by position (symbol + timestamp proximity)
        positions = self._group_executions_by_position(executions)
        
        wins = []
        losses = []
        holding_periods = []
        win_holdings = []
        loss_holdings = []
        
        for position_trades in positions:
            if len(position_trades) >= 2:  # Need entry and exit
                entry = position_trades[0]
                exit_trade = position_trades[-1]
                
                # Calculate P&L
                if entry.side == OrderSide.BUY:
                    pnl = float((exit_trade.price - entry.price) * entry.quantity)
                else:
                    pnl = float((entry.price - exit_trade.price) * entry.quantity)
                
                # Account for commissions
                total_commission = sum(float(t.commission) for t in position_trades)
                pnl -= total_commission
                
                holding_period = exit_trade.timestamp - entry.timestamp
                holding_periods.append(holding_period)
                
                if pnl > 0:
                    wins.append(pnl)
                    win_holdings.append(holding_period)
                else:
                    losses.append(abs(pnl))
                    loss_holdings.append(holding_period)
        
        total_trades = len(wins) + len(losses)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        # Calculate average holding periods (convert to seconds for calculation)
        if holding_periods:
            avg_holding_seconds = statistics.mean([td.total_seconds() for td in holding_periods])
            avg_holding_period = timedelta(seconds=avg_holding_seconds)
        else:
            avg_holding_period = timedelta(0)
            
        if win_holdings:
            avg_win_seconds = statistics.mean([td.total_seconds() for td in win_holdings])
            avg_win_holding = timedelta(seconds=avg_win_seconds)
        else:
            avg_win_holding = timedelta(0)
            
        if loss_holdings:
            avg_loss_seconds = statistics.mean([td.total_seconds() for td in loss_holdings])
            avg_loss_holding = timedelta(seconds=avg_loss_seconds)
        else:
            avg_loss_holding = timedelta(0)
        
        return TradeAnalysis(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=(winning_trades / max(total_trades, 1)) * 100,
            total_pnl=sum(wins) - sum(losses),
            avg_win=statistics.mean(wins) if wins else 0.0,
            avg_loss=statistics.mean(losses) if losses else 0.0,
            largest_win=max(wins) if wins else 0.0,
            largest_loss=max(losses) if losses else 0.0,
            profit_factor=sum(wins) / max(sum(losses), 0.001),
            avg_holding_period=avg_holding_period,
            avg_win_holding=avg_win_holding,
            avg_loss_holding=avg_loss_holding,
            max_consecutive_wins=self._max_consecutive(wins, losses, True),
            max_consecutive_losses=self._max_consecutive(wins, losses, False)
        )
    
    def get_advanced_metrics(self) -> AdvancedMetrics:
        """Calculate advanced performance metrics."""
        if len(self.snapshots) < 2:
            return AdvancedMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                downside_volatility=0.0, max_drawdown=0.0, max_drawdown_duration=timedelta(0),
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                rolling_sharpe_30d=0.0, rolling_return_30d=0.0, rolling_volatility_30d=0.0
            )
        
        # Calculate returns
        current_equity = self.snapshots[-1].equity
        total_return = float((current_equity - self.initial_balance) / self.initial_balance)
        
        # Annualized return
        days = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).days
        if days > 0:
            annualized_return = (float(current_equity / self.initial_balance) ** (365.25 / days)) - 1
        else:
            annualized_return = 0.0
        
        # Volatility
        volatility = statistics.stdev(self.daily_returns) * math.sqrt(252) if len(self.daily_returns) > 1 else 0.0
        
        # Downside volatility (only negative returns)
        negative_returns = [r for r in self.daily_returns if r < 0]
        downside_volatility = statistics.stdev(negative_returns) * math.sqrt(252) if len(negative_returns) > 1 else 0.0
        
        # Risk-adjusted ratios (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annualized_return / max(volatility, 0.001)
        sortino_ratio = annualized_return / max(downside_volatility, 0.001)
        calmar_ratio = annualized_return / max(float(self.max_drawdown) / float(self.initial_balance), 0.001)
        
        # Rolling 30-day metrics
        rolling_returns_30d = self.daily_returns[-30:] if len(self.daily_returns) >= 30 else self.daily_returns
        rolling_return_30d = sum(rolling_returns_30d)
        rolling_volatility_30d = statistics.stdev(rolling_returns_30d) * math.sqrt(252) if len(rolling_returns_30d) > 1 else 0.0
        rolling_sharpe_30d = (rolling_return_30d * 12) / max(rolling_volatility_30d, 0.001)  # Annualized
        
        return AdvancedMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            downside_volatility=downside_volatility,
            max_drawdown=float(self.max_drawdown) / float(self.initial_balance),
            max_drawdown_duration=self.max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            rolling_sharpe_30d=rolling_sharpe_30d,
            rolling_return_30d=rolling_return_30d,
            rolling_volatility_30d=rolling_volatility_30d
        )
    
    def _group_executions_by_position(self, executions: List[TradeExecution]) -> List[List[TradeExecution]]:
        """Group executions by trading position."""
        # Group by symbol and pair buy/sell executions
        symbol_groups = {}
        for execution in sorted(executions, key=lambda x: x.timestamp):
            if execution.symbol not in symbol_groups:
                symbol_groups[execution.symbol] = []
            symbol_groups[execution.symbol].append(execution)
        
        # For each symbol, pair buy/sell executions to form complete positions
        positions = []
        for symbol_executions in symbol_groups.values():
            # Group into position pairs (entry + exit)
            current_position = []
            for execution in symbol_executions:
                current_position.append(execution)
                
                # If we have both buy and sell, consider it a complete position
                sides = set(exec.side for exec in current_position)
                if len(current_position) >= 2 and len(sides) == 2:
                    positions.append(current_position)
                    current_position = []
            
            # Add remaining executions as incomplete position if any
            if current_position:
                positions.append(current_position)
        
        return positions
    
    def _max_consecutive(self, wins: List[float], losses: List[float], count_wins: bool) -> int:
        """Calculate max consecutive wins or losses."""
        # This is a simplified implementation
        # Real implementation would need to track order of trades
        return max(len(wins), len(losses)) if count_wins else max(len(losses), 0)


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
        
        # Enhanced Performance Analytics (Task 13.4)
        self.performance_analyzer = PerformanceAnalyzer(self.config.initial_balance)
        self.last_snapshot_time: Optional[datetime] = None
        
        # Control flags
        self._monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"Position Manager initialized with balance: {self.config.initial_balance}")
    
    async def _start(self) -> bool:
        """Start the position manager agent."""
        try:
            await self._setup_subscriptions()
            self._monitor_task = self.create_task(self._monitor_positions())
            self._set_state(AgentState.RUNNING)
            
            # Create initial performance snapshot
            await self._create_performance_snapshot()
            
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
    
    async def _health_check(self) -> AgentHealth:
        """Perform health check and return health data."""
        try:
            healthy = (
                self.portfolio.total_balance > 0 and
                len(self.positions) <= 10
            )
            
            health_score = 1.0 if healthy else 0.5
            
            # Create and return AgentHealth object
            return AgentHealth(
                state=self.state,
                last_heartbeat=datetime.now(timezone.utc),
                cpu_usage=0.0,  # Could be implemented with psutil
                memory_usage=0.0,  # Could be implemented with psutil
                error_count=0,
                warning_count=0,
                messages_processed=self.total_trades,
                tasks_completed=self.total_trades,
                uptime_seconds=self.uptime,
                health_score=health_score,
                last_error=None,
                last_error_time=None
            )
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return AgentHealth(
                state=self.state,
                health_score=0.0,
                last_error=str(e),
                last_error_time=datetime.now(timezone.utc),
                error_count=1
            )
    
    async def handle_message(self, message: Message) -> None:
        """Handle incoming messages from the message bus."""
        try:
            if message.type == MessageType.SIGNAL_GENERATED:
                await self._handle_trading_signal(message)
            elif message.type in [MessageType.DATA_MARKET_UPDATE, MessageType.DATA_PRICE_UPDATE]:
                await self._handle_market_data(message)
            elif message.type == MessageType.SYSTEM_HEALTH_CHECK:
                # Respond to health check requests
                health = await self.get_health()
                # Could publish health response here if needed
            else:
                self.logger.debug(f"Unhandled message type: {message.type}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}", exc_info=True)
    
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
                
                # Create performance snapshot every hour or on significant change
                await self._maybe_create_performance_snapshot()
                
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.logger.info("Position monitor cancelled")
        except Exception as e:
            self.logger.error(f"Error in position monitor: {e}", exc_info=True)
    
    async def _create_performance_snapshot(self) -> None:
        """Create a performance snapshot."""
        try:
            current_time = datetime.now(timezone.utc)
            equity = self.portfolio.equity
            
            # Calculate current drawdown
            drawdown, drawdown_pct = self.performance_analyzer.calculate_drawdown(equity)
            
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                total_balance=self.portfolio.total_balance,
                total_pnl=self.total_pnl,
                unrealized_pnl=self.portfolio.unrealized_pnl,
                realized_pnl=self.portfolio.realized_pnl,
                equity=equity,
                drawdown=drawdown,
                drawdown_pct=drawdown_pct
            )
            
            self.performance_analyzer.add_snapshot(snapshot)
            self.last_snapshot_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error creating performance snapshot: {e}", exc_info=True)

    async def _maybe_create_performance_snapshot(self) -> None:
        """Create performance snapshot if needed."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Create snapshot every hour or if no snapshot exists
            if (not self.last_snapshot_time or 
                current_time - self.last_snapshot_time >= timedelta(hours=1)):
                await self._create_performance_snapshot()
                
        except Exception as e:
            self.logger.error(f"Error in maybe create performance snapshot: {e}", exc_info=True)
    
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
        """Get comprehensive performance metrics."""
        try:
            # Basic metrics (maintaining backward compatibility)
            basic_metrics = {
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
            
            # Enhanced analytics
            trade_analysis = self.performance_analyzer.get_trade_analysis(self.executions)
            advanced_metrics = self.performance_analyzer.get_advanced_metrics()
            
            # Current drawdown
            drawdown, drawdown_pct = self.performance_analyzer.calculate_drawdown(self.portfolio.equity)
            
            # Combine all metrics
            enhanced_metrics = {
                **basic_metrics,
                
                # Advanced P&L metrics
                'avg_win': trade_analysis.avg_win,
                'avg_loss': trade_analysis.avg_loss,
                'largest_win': trade_analysis.largest_win,
                'largest_loss': trade_analysis.largest_loss,
                'profit_factor': trade_analysis.profit_factor,
                
                # Timing metrics
                'avg_holding_period_hours': trade_analysis.avg_holding_period.total_seconds() / 3600,
                'avg_win_holding_hours': trade_analysis.avg_win_holding.total_seconds() / 3600,
                'avg_loss_holding_hours': trade_analysis.avg_loss_holding.total_seconds() / 3600,
                
                # Risk metrics
                'max_consecutive_wins': trade_analysis.max_consecutive_wins,
                'max_consecutive_losses': trade_analysis.max_consecutive_losses,
                'current_drawdown': float(drawdown),
                'current_drawdown_pct': float(drawdown_pct),
                
                # Advanced performance metrics
                'total_return': advanced_metrics.total_return,
                'annualized_return': advanced_metrics.annualized_return,
                'volatility': advanced_metrics.volatility,
                'downside_volatility': advanced_metrics.downside_volatility,
                'max_drawdown': advanced_metrics.max_drawdown,
                'max_drawdown_duration_hours': advanced_metrics.max_drawdown_duration.total_seconds() / 3600,
                
                # Risk-adjusted ratios
                'sharpe_ratio': advanced_metrics.sharpe_ratio,
                'sortino_ratio': advanced_metrics.sortino_ratio,
                'calmar_ratio': advanced_metrics.calmar_ratio,
                
                # Rolling metrics
                'rolling_sharpe_30d': advanced_metrics.rolling_sharpe_30d,
                'rolling_return_30d': advanced_metrics.rolling_return_30d,
                'rolling_volatility_30d': advanced_metrics.rolling_volatility_30d,
                
                # Metadata
                'snapshots_count': len(self.performance_analyzer.snapshots),
                'last_snapshot_time': self.last_snapshot_time.isoformat() if self.last_snapshot_time else None
            }
            
            return enhanced_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}", exc_info=True)
            # Return basic metrics on error
            return {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': (self.winning_trades / max(self.total_trades, 1)) * 100,
                'total_pnl': float(self.total_pnl),
                'error': str(e)
            }

    def get_trade_analysis(self) -> TradeAnalysis:
        """Get detailed trade analysis."""
        return self.performance_analyzer.get_trade_analysis(self.executions)

    def get_advanced_metrics(self) -> AdvancedMetrics:
        """Get advanced performance metrics."""
        return self.performance_analyzer.get_advanced_metrics()

    def get_performance_snapshots(self) -> List[PerformanceSnapshot]:
        """Get all performance snapshots."""
        return self.performance_analyzer.snapshots.copy()

    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            metrics = self.get_performance_metrics()
            trade_analysis = self.get_trade_analysis()
            advanced_metrics = self.get_advanced_metrics()
            
            # Create snapshot for current state
            await self._create_performance_snapshot()
            
            report = {
                'report_timestamp': datetime.now(timezone.utc).isoformat(),
                'account_summary': {
                    'initial_balance': float(self.config.initial_balance),
                    'current_equity': float(self.portfolio.equity),
                    'total_return': advanced_metrics.total_return,
                    'annualized_return': advanced_metrics.annualized_return
                },
                'trading_summary': {
                    'total_trades': trade_analysis.total_trades,
                    'winning_trades': trade_analysis.winning_trades,
                    'losing_trades': trade_analysis.losing_trades,
                    'win_rate': trade_analysis.win_rate,
                    'profit_factor': trade_analysis.profit_factor
                },
                'risk_metrics': {
                    'max_drawdown': advanced_metrics.max_drawdown,
                    'max_drawdown_duration_days': advanced_metrics.max_drawdown_duration.days,
                    'volatility': advanced_metrics.volatility,
                    'downside_volatility': advanced_metrics.downside_volatility,
                    'sharpe_ratio': advanced_metrics.sharpe_ratio,
                    'sortino_ratio': advanced_metrics.sortino_ratio,
                    'calmar_ratio': advanced_metrics.calmar_ratio
                },
                'current_positions': {
                    'open_positions': len([p for p in self.positions.values() if p.is_open]),
                    'unrealized_pnl': float(self.portfolio.unrealized_pnl),
                    'position_details': [
                        {
                            'symbol': p.symbol,
                            'side': p.side.value,
                            'size': float(p.size),
                            'entry_price': float(p.entry_price),
                            'current_price': float(p.current_price) if p.current_price else None,
                            'unrealized_pnl': float(p.unrealized_pnl) if p.unrealized_pnl else None
                        }
                        for p in self.positions.values() if p.is_open
                    ]
                },
                'recent_trades': [
                    {
                        'symbol': e.symbol,
                        'side': e.side.value,
                        'quantity': float(e.quantity),
                        'price': float(e.price),
                        'timestamp': e.timestamp.isoformat(),
                        'commission': float(e.commission)
                    }
                    for e in self.executions[-10:]  # Last 10 trades
                ]
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}", exc_info=True)
            return {'error': str(e), 'report_timestamp': datetime.now(timezone.utc).isoformat()} 