"""
Unit tests for Position Manager Agent

Tests the core functionality of the Position Manager including:
- Trade execution and position management
- Stop-loss and take-profit functionality  
- Portfolio state tracking
- Risk management and validation
- Message handling and agent lifecycle
"""

import pytest
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.bistoury.agents.position_manager_agent import (
    PositionManagerAgent, PositionManagerConfig
)
from src.bistoury.agents.base import AgentState
from src.bistoury.models.trading import (
    Position, PositionSide, Order, OrderSide, OrderType, 
    TradeExecution, PortfolioState
)
from src.bistoury.models.agent_messages import (
    Message, MessageType, TradingSignalPayload, MarketDataPayload
)


class TestPositionManagerConfig:
    """Test Position Manager configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PositionManagerConfig()
        
        assert config.initial_balance == Decimal('100000')
        assert config.slippage_rate == Decimal('0.0005')
        assert config.taker_fee_rate == Decimal('0.00045')
        assert config.maker_fee_rate == Decimal('0.00015')
        assert config.min_position_size == Decimal('10')
        assert config.max_position_size == Decimal('10000')
        assert config.enable_stop_loss is True
        assert config.enable_take_profit is True
        assert config.stop_loss_pct == Decimal('2.0')
        assert config.take_profit_pct == Decimal('4.0')
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PositionManagerConfig(
            initial_balance=Decimal('50000'),
            slippage_rate=Decimal('0.001'),
            taker_fee_rate=Decimal('0.001'),
            maker_fee_rate=Decimal('0.0005'),
            min_position_size=Decimal('5'),
            max_position_size=Decimal('5000'),
            enable_stop_loss=False,
            enable_take_profit=False,
            stop_loss_pct=Decimal('1.5'),
            take_profit_pct=Decimal('3.0')
        )
        
        assert config.initial_balance == Decimal('50000')
        assert config.slippage_rate == Decimal('0.001')
        assert config.taker_fee_rate == Decimal('0.001')
        assert config.maker_fee_rate == Decimal('0.0005')
        assert config.min_position_size == Decimal('5')
        assert config.max_position_size == Decimal('5000')
        assert config.enable_stop_loss is False
        assert config.enable_take_profit is False
        assert config.stop_loss_pct == Decimal('1.5')
        assert config.take_profit_pct == Decimal('3.0')


@pytest.fixture
def position_manager_config():
    """Create a test configuration for Position Manager."""
    return PositionManagerConfig(
        initial_balance=Decimal('100000'),
        slippage_rate=Decimal('0.0005'),
        taker_fee_rate=Decimal('0.00045'),  # HyperLiquid rates
        maker_fee_rate=Decimal('0.00015'),  # HyperLiquid rates
        min_position_size=Decimal('0.001'),  # Much smaller minimum for tests
        max_position_size=Decimal('1000'),
        enable_stop_loss=True,
        enable_take_profit=True,
        stop_loss_pct=Decimal('2.0'),
        take_profit_pct=Decimal('4.0')
    )


@pytest.fixture
def position_manager(position_manager_config):
    """Create a Position Manager instance for testing."""
    return PositionManagerAgent(
        name="test_position_manager",
        config=position_manager_config
    )


@pytest.fixture
def mock_message_bus():
    """Create a mock message bus."""
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def trading_signal():
    """Create a test trading signal."""
    return TradingSignalPayload(
        symbol="BTC",
        signal_type="momentum",
        direction="BUY",
        confidence=0.8,
        strength=7.5,
        timeframe="1h",
        strategy="test_strategy",
        reasoning="Test signal for buying BTC"
    )


class TestPositionManagerAgent:
    """Test Position Manager Agent functionality."""
    
    def test_initialization(self, position_manager, position_manager_config):
        """Test Position Manager initialization."""
        assert position_manager.name == "test_position_manager"
        assert position_manager.config == position_manager_config
        assert position_manager.portfolio.total_balance == Decimal('100000')
        assert position_manager.portfolio.available_balance == Decimal('100000')
        assert len(position_manager.positions) == 0
        assert len(position_manager.executions) == 0
        assert position_manager.total_trades == 0
        assert position_manager.winning_trades == 0
        assert position_manager.total_pnl == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, position_manager, mock_message_bus):
        """Test agent start and stop lifecycle."""
        position_manager._message_bus = mock_message_bus
        
        # Test start
        result = await position_manager._start()
        assert result is True
        assert position_manager.state == AgentState.RUNNING
        
        # Verify subscriptions were set up
        assert mock_message_bus.subscribe.call_count == 2  # signals + market data
        
        # Test stop
        await position_manager._stop()
        assert position_manager.state == AgentState.STOPPED
    
    @pytest.mark.asyncio
    async def test_health_check(self, position_manager):
        """Test health check functionality."""
        await position_manager._health_check()
        assert position_manager.health.health_score == 1.0
        assert position_manager.health.messages_processed == 0
        
        # Test with many positions (should still be healthy)
        for i in range(5):
            position = Position(
                symbol=f"BTC{i}",
                side=PositionSide.LONG,
                size=Decimal('100'),
                entry_price=Decimal('50000')
            )
            position_manager.positions[f"BTC{i}"] = position
        
        await position_manager._health_check()
        assert position_manager.health.health_score == 1.0
        
        # Test with too many positions (should be unhealthy)
        for i in range(5, 15):
            position = Position(
                symbol=f"BTC{i}",
                side=PositionSide.LONG,
                size=Decimal('100'),
                entry_price=Decimal('50000')
            )
            position_manager.positions[f"BTC{i}"] = position
        
        await position_manager._health_check()
        assert position_manager.health.health_score == 0.5
    
    def test_calculate_position_size(self, position_manager, trading_signal):
        """Test position size calculation."""
        price = Decimal('50000')
        
        # Test normal confidence (0.8)
        size = position_manager._calculate_position_size(trading_signal, price)
        expected_notional = Decimal('100000') * Decimal('0.02') * Decimal('0.8') * Decimal('2')
        expected_size = expected_notional / price
        
        # The actual calculation should be clamped to min_position_size
        expected_clamped = max(expected_size, position_manager.config.min_position_size)
        assert abs(size - expected_clamped) < Decimal('0.001')
        
        # Test minimum size enforcement
        low_confidence_signal = TradingSignalPayload(
            symbol="BTC",
            signal_type="momentum", 
            direction="BUY",
            confidence=0.1,  # Very low confidence
            strength=1.0,
            timeframe="1h",
            strategy="test_strategy",
            reasoning="Low confidence test"
        )
        size = position_manager._calculate_position_size(low_confidence_signal, price)
        assert size >= position_manager.config.min_position_size
        
        # Test maximum size enforcement
        high_confidence_signal = TradingSignalPayload(
            symbol="BTC",
            signal_type="momentum",
            direction="BUY", 
            confidence=1.0,  # Maximum confidence
            strength=10.0,
            timeframe="1h",
            strategy="test_strategy",
            reasoning="High confidence test"
        )
        size = position_manager._calculate_position_size(high_confidence_signal, price)
        assert size <= position_manager.config.max_position_size
    
    @pytest.mark.asyncio
    async def test_execute_order(self, position_manager):
        """Test order execution with slippage and commission."""
        order = Order(
            client_order_id="test_order_1",
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('0.1'),
            time_in_force="IOC"
        )
        
        market_price = Decimal('50000')
        execution = await position_manager._execute_order(order, market_price)
        
        assert execution is not None
        assert execution.symbol == "BTC"
        assert execution.side == OrderSide.BUY
        assert execution.quantity == Decimal('0.1')
        
        # Check slippage applied (buy should have higher price)
        expected_price = market_price * (Decimal('1') + position_manager.config.slippage_rate)
        assert execution.price == expected_price
        
        # Check commission calculated (market order = taker)
        notional = execution.quantity * execution.price
        expected_commission = notional * position_manager.config.taker_fee_rate
        assert execution.commission == expected_commission
        
        # Check portfolio updated
        cost = notional + expected_commission
        expected_balance = Decimal('100000') - cost
        assert position_manager.portfolio.available_balance == expected_balance
        
        # Check order status updated
        assert order.status.value == "filled"
        assert order.filled_quantity == Decimal('0.1')
        assert order.average_fill_price == expected_price
    
    @pytest.mark.asyncio
    async def test_create_position(self, position_manager, trading_signal):
        """Test position creation."""
        execution = TradeExecution(
            execution_id="exec_1",
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=Decimal('0.1'),
            price=Decimal('50025'),  # With slippage
            timestamp=datetime.now(timezone.utc),
            commission=Decimal('25.0125'),
            is_maker=False,
            liquidity="taker"
        )
        
        await position_manager._create_position(execution, trading_signal)
        
        assert "BTC" in position_manager.positions
        position = position_manager.positions["BTC"]
        
        assert position.symbol == "BTC"
        assert position.side == PositionSide.LONG
        assert position.size == Decimal('0.1')
        assert position.entry_price == Decimal('50025')
        assert position.is_open is True
        
        # Check stop loss and take profit set
        assert position.stop_loss is not None
        assert position.take_profit is not None
        
        # Verify stop loss calculation (2% below entry for long)
        expected_stop = Decimal('50025') * (Decimal('1') - Decimal('2.0') / 100)
        assert position.stop_loss == expected_stop
        
        # Verify take profit calculation (4% above entry for long)
        expected_tp = Decimal('50025') * (Decimal('1') + Decimal('4.0') / 100)
        assert position.take_profit == expected_tp
    
    @pytest.mark.asyncio
    async def test_check_stop_loss(self, position_manager):
        """Test stop loss functionality."""
        # Create a long position
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            current_price=Decimal('50000'),
            stop_loss=Decimal('49000')  # 2% stop loss
        )
        position_manager.positions["BTC"] = position
        position_manager.current_prices["BTC"] = Decimal('48500')  # Below stop loss
        
        # Update position price first
        position.update_price(Decimal('48500'))
        
        # Mock _close_position to track calls
        position_manager._close_position = AsyncMock()
        
        await position_manager._check_stop_take_profit(position)
        
        # Verify close_position was called with stop_loss reason
        position_manager._close_position.assert_called_once_with(
            "BTC", Decimal('48500'), "stop_loss"
        )
    
    @pytest.mark.asyncio
    async def test_check_take_profit(self, position_manager):
        """Test take profit functionality."""
        # Create a long position
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            current_price=Decimal('50000'),
            take_profit=Decimal('52000')  # 4% take profit
        )
        position_manager.positions["BTC"] = position
        position_manager.current_prices["BTC"] = Decimal('52500')  # Above take profit
        
        # Update position price first
        position.update_price(Decimal('52500'))
        
        # Mock _close_position to track calls
        position_manager._close_position = AsyncMock()
        
        await position_manager._check_stop_take_profit(position)
        
        # Verify close_position was called with take_profit reason
        position_manager._close_position.assert_called_once_with(
            "BTC", Decimal('52500'), "take_profit"
        )
    
    @pytest.mark.asyncio
    async def test_close_position(self, position_manager):
        """Test position closing."""
        # Create an open position
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            current_price=Decimal('52000')
        )
        position_manager.positions["BTC"] = position
        position_manager.portfolio.add_position(position)
        
        # Mock _execute_order to return a successful execution
        mock_execution = TradeExecution(
            execution_id="close_exec_1",
            symbol="BTC",
            side=OrderSide.SELL,
            quantity=Decimal('0.1'),
            price=Decimal('51900'),  # With slippage
            timestamp=datetime.now(timezone.utc),
            commission=Decimal('25.95'),
            is_maker=False,
            liquidity="taker"
        )
        position_manager._execute_order = AsyncMock(return_value=mock_execution)
        
        await position_manager._close_position("BTC", Decimal('52000'), "manual")
        
        # Verify position was closed
        assert not position.is_open
        assert position.exit_price == Decimal('52000')
        
        # Verify PnL calculated and updated
        expected_pnl = (Decimal('52000') - Decimal('50000')) * Decimal('0.1')
        assert position.realized_pnl == expected_pnl
        assert position_manager.total_pnl == expected_pnl
        assert position_manager.winning_trades == 1
        
        # Verify portfolio updated
        assert "BTC" not in [p.symbol for p in position_manager.portfolio.positions]
    
    @pytest.mark.asyncio
    async def test_handle_trading_signal_valid(self, position_manager, trading_signal, mock_message_bus):
        """Test handling valid trading signal."""
        position_manager._message_bus = mock_message_bus
        position_manager.current_prices["BTC"] = Decimal('50000')
        position_manager._execute_signal = AsyncMock()
        
        message = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="test_strategy",
            payload=trading_signal
        )
        
        await position_manager._handle_trading_signal(message)
        
        # Verify signal was processed
        position_manager._execute_signal.assert_called_once_with(trading_signal)
    
    @pytest.mark.asyncio
    async def test_handle_trading_signal_low_confidence(self, position_manager, mock_message_bus):
        """Test handling low confidence trading signal."""
        position_manager._message_bus = mock_message_bus
        position_manager._execute_signal = AsyncMock()
        
        low_confidence_signal = TradingSignalPayload(
            symbol="BTC",
            signal_type="momentum",
            direction="BUY",
            confidence=0.5,  # Below 0.6 threshold
            strength=5.0,
            timeframe="1h",
            strategy="test_strategy",
            reasoning="Low confidence signal"
        )
        
        message = Message(
            type=MessageType.SIGNAL_GENERATED,
            sender="test_strategy",
            payload=low_confidence_signal
        )
        
        await position_manager._handle_trading_signal(message)
        
        # Verify signal was NOT processed due to low confidence
        position_manager._execute_signal.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_handle_market_data(self, position_manager):
        """Test handling market data updates."""
        from src.bistoury.models.agent_messages import MarketDataPayload
        
        # Create a position to update
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            current_price=Decimal('50000')
        )
        position_manager.positions["BTC"] = position
        position_manager._check_stop_take_profit = AsyncMock()
        
        # Create market data message with proper payload
        market_payload = MarketDataPayload(
            symbol="BTC",
            price=Decimal('51000'),
            timestamp=datetime.now(timezone.utc),
            source="test"
        )
        
        message = Message(
            type=MessageType.DATA_PRICE_UPDATE,
            sender="data_collector",
            payload=market_payload
        )
        
        await position_manager._handle_market_data(message)
        
        # Verify price was updated
        assert position_manager.current_prices["BTC"] == Decimal('51000')
        assert position.current_price == Decimal('51000')
        
        # Verify stop/take profit check was called
        position_manager._check_stop_take_profit.assert_called_once_with(position)
    
    @pytest.mark.asyncio
    async def test_execute_signal_full_flow(self, position_manager, trading_signal):
        """Test full signal execution flow."""
        position_manager.current_prices["BTC"] = Decimal('50000')
        
        # Mock order execution
        mock_execution = TradeExecution(
            execution_id="exec_1",
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=Decimal('0.032'),  # Based on position size calculation
            price=Decimal('50025'),
            timestamp=datetime.now(timezone.utc),
            commission=Decimal('0.8004'),
            is_maker=False,
            liquidity="taker"
        )
        position_manager._execute_order = AsyncMock(return_value=mock_execution)
        position_manager._create_position = AsyncMock()
        
        await position_manager._execute_signal(trading_signal)
        
        # Verify order was executed
        position_manager._execute_order.assert_called_once()
        
        # Verify position was created
        position_manager._create_position.assert_called_once_with(mock_execution, trading_signal)
    
    @pytest.mark.asyncio
    async def test_signal_reversal(self, position_manager, trading_signal):
        """Test signal reversal closes opposite position."""
        # Create existing long position
        existing_position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            current_price=Decimal('50000')
        )
        existing_position.is_open = True
        position_manager.positions["BTC"] = existing_position
        position_manager.current_prices["BTC"] = Decimal('50000')
        
        # Create sell signal (opposite to existing long)
        sell_signal = TradingSignalPayload(
            symbol="BTC",
            signal_type="momentum",
            direction="SELL",
            confidence=0.8,
            strength=7.5,
            timeframe="1h",
            strategy="test_strategy",
            reasoning="Sell signal"
        )
        
        position_manager._close_position = AsyncMock()
        position_manager._execute_order = AsyncMock()
        
        await position_manager._execute_signal(sell_signal)
        
        # Verify existing position was closed due to reversal
        position_manager._close_position.assert_called_once_with(
            "BTC", Decimal('50000'), "signal_reversal"
        )
    
    def test_performance_metrics(self, position_manager):
        """Test performance metrics calculation."""
        # Set up some test data
        position_manager.total_trades = 10
        position_manager.winning_trades = 6
        position_manager.total_pnl = Decimal('1500')
        position_manager.portfolio.unrealized_pnl = Decimal('200')
        position_manager.portfolio.realized_pnl = Decimal('1300')
        
        # Add some positions
        for i in range(3):
            position = Position(
                symbol=f"BTC{i}",
                side=PositionSide.LONG,
                size=Decimal('0.1'),
                entry_price=Decimal('50000')
            )
            position.is_open = True
            position_manager.positions[f"BTC{i}"] = position
        
        metrics = position_manager.get_performance_metrics()
        
        assert metrics['total_trades'] == 10
        assert metrics['winning_trades'] == 6
        assert metrics['win_rate'] == 60.0
        assert metrics['total_pnl'] == 1500.0
        assert metrics['unrealized_pnl'] == 200.0
        assert metrics['realized_pnl'] == 1300.0
        assert metrics['open_positions'] == 3
    
    @pytest.mark.asyncio
    async def test_public_api_methods(self, position_manager):
        """Test public API methods."""
        # Add test position
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal('0.1'),
            entry_price=Decimal('50000')
        )
        position_manager.positions["BTC"] = position
        
        # Test get_portfolio_state
        portfolio = await position_manager.get_portfolio_state()
        assert isinstance(portfolio, PortfolioState)
        assert portfolio.account_id == "test_position_manager"
        
        # Test get_positions
        positions = await position_manager.get_positions()
        assert "BTC" in positions
        assert positions["BTC"] == position
        
        # Test get_position
        btc_position = await position_manager.get_position("BTC")
        assert btc_position == position
        
        # Test get_position for non-existent symbol
        none_position = await position_manager.get_position("ETH")
        assert none_position is None


@pytest.mark.asyncio
async def test_integration_position_lifecycle(position_manager, trading_signal):
    """Integration test for complete position lifecycle."""
    # Set initial market price 
    position_manager.current_prices["BTC"] = Decimal('50000')
    
    # Use a low confidence signal to get smaller position size
    small_signal = TradingSignalPayload(
        symbol="BTC",
        signal_type="momentum",
        direction="BUY",
        confidence=0.6,  # Lower confidence for smaller position
        strength=6.0,
        timeframe="1h",
        strategy="test_strategy",
        reasoning="Small test signal for buying BTC"
    )
    
    # Execute buy signal
    await position_manager._execute_signal(small_signal)
    
    # Verify position created
    assert "BTC" in position_manager.positions
    position = position_manager.positions["BTC"]
    assert position.side == PositionSide.LONG
    assert position.is_open is True
    
    # Simulate price movement that triggers take profit
    new_price = Decimal('52500')  # Above take profit
    position_manager.current_prices["BTC"] = new_price
    position.update_price(new_price)
    
    await position_manager._check_stop_take_profit(position)
    
    # Verify position was closed with profit
    assert not position.is_open
    assert position.realized_pnl > 0
    assert position_manager.winning_trades == 1
    assert position_manager.total_pnl > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 