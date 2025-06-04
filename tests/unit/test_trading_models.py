"""
Unit tests for trading operation models.

Tests position management, order tracking, trade executions, risk parameters,
portfolio state calculations, and HyperLiquid format conversions for all
trading-related models.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any

from src.bistoury.models.trading import (
    PositionSide,
    OrderType,
    OrderStatus,
    OrderSide,
    TimeInForce,
    Position,
    Order,
    TradeExecution,
    RiskParameters,
    PortfolioState,
)


class TestPositionSide:
    """Test PositionSide enum functionality."""
    
    def test_position_side_values(self):
        """Test position side enum values."""
        assert PositionSide.LONG == "long"
        assert PositionSide.SHORT == "short"


class TestOrderType:
    """Test OrderType enum functionality."""
    
    def test_order_type_values(self):
        """Test order type enum values."""
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP_MARKET == "stopMarket"
        assert OrderType.STOP_LIMIT == "stopLimit"
        assert OrderType.TAKE_PROFIT == "takeProfit"
        assert OrderType.TAKE_PROFIT_LIMIT == "takeProfitLimit"


class TestOrderStatus:
    """Test OrderStatus enum functionality."""
    
    def test_order_status_values(self):
        """Test order status enum values."""
        assert OrderStatus.PENDING == "pending"
        assert OrderStatus.OPEN == "open"
        assert OrderStatus.PARTIALLY_FILLED == "partiallyFilled"
        assert OrderStatus.FILLED == "filled"
        assert OrderStatus.CANCELLED == "cancelled"
        assert OrderStatus.REJECTED == "rejected"
        assert OrderStatus.EXPIRED == "expired"


class TestPosition:
    """Test Position model functionality."""
    
    def test_valid_long_position(self):
        """Test creation of valid long position."""
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("1.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            leverage=Decimal("5")
        )
        
        assert position.symbol == "BTC"
        assert position.side == PositionSide.LONG
        assert position.size == Decimal("1.5")
        assert position.entry_price == Decimal("50000")
        assert position.current_price == Decimal("51000")
        assert position.leverage == Decimal("5")
        assert position.is_open is True
    
    def test_valid_short_position(self):
        """Test creation of valid short position."""
        position = Position(
            symbol="ETH",
            side=PositionSide.SHORT,
            size=Decimal("10"),
            entry_price=Decimal("3000"),
            current_price=Decimal("2950"),
            leverage=Decimal("3")
        )
        
        assert position.side == PositionSide.SHORT
        assert position.size == Decimal("10")
        assert position.leverage == Decimal("3")
    
    def test_position_pnl_calculations_long(self):
        """Test PnL calculations for long position."""
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("2"),
            entry_price=Decimal("50000"),
            current_price=Decimal("52000")
        )
        
        # Unrealized PnL: (52000 - 50000) * 2 = 4000
        assert position.unrealized_pnl == Decimal("4000")
        
        # Unrealized PnL percentage: (2000 / 50000) * 100 = 4%
        assert position.unrealized_pnl_percent == Decimal("4")
        
        # Notional value: 2 * 52000 = 104000
        assert position.notional_value == Decimal("104000")
    
    def test_position_pnl_calculations_short(self):
        """Test PnL calculations for short position."""
        position = Position(
            symbol="ETH",
            side=PositionSide.SHORT,
            size=Decimal("5"),
            entry_price=Decimal("3000"),
            current_price=Decimal("2800")
        )
        
        # Unrealized PnL: (3000 - 2800) * 5 = 1000
        assert position.unrealized_pnl == Decimal("1000")
        
        # Unrealized PnL percentage: (200 / 3000) * 100 = 6.666...%
        expected_percent = Decimal("6.666666666666666666666666667")
        assert abs(position.unrealized_pnl_percent - expected_percent) < Decimal("0.000001")
    
    def test_position_realized_pnl(self):
        """Test realized PnL calculations."""
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("1"),
            entry_price=Decimal("50000"),
            is_open=False,
            exit_price=Decimal("55000")
        )
        
        # Realized PnL: (55000 - 50000) * 1 = 5000
        assert position.realized_pnl == Decimal("5000")
        
        # Realized PnL percentage: (5000 / 50000) * 100 = 10%
        assert position.realized_pnl_percent == Decimal("10")
    
    def test_position_margin_calculations(self):
        """Test margin requirement calculations."""
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("2"),
            entry_price=Decimal("50000"),
            leverage=Decimal("5")
        )
        
        # Margin requirement: (2 * 50000) / 5 = 20000
        assert position.margin_requirement == Decimal("20000")
    
    def test_position_liquidation_price_calculation(self):
        """Test liquidation price estimation."""
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("1"),
            entry_price=Decimal("50000"),
            margin_used=Decimal("10000")
        )
        
        # Should have a liquidation price
        liquidation_price = position.liquidation_price
        assert liquidation_price is not None
        assert liquidation_price < position.entry_price  # Long position liquidation below entry
    
    def test_position_price_update(self):
        """Test position price update functionality."""
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("1"),
            entry_price=Decimal("50000")
        )
        
        # Update price
        position.update_price(Decimal("51000"))
        assert position.current_price == Decimal("51000")
        assert position.unrealized_pnl == Decimal("1000")
    
    def test_position_close(self):
        """Test position closing functionality."""
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("1"),
            entry_price=Decimal("50000")
        )
        
        exit_time = datetime.now(timezone.utc)
        position.close_position(Decimal("52000"), exit_time)
        
        assert position.is_open is False
        assert position.exit_price == Decimal("52000")
        assert position.exit_timestamp == exit_time
        assert position.current_price == Decimal("52000")
        assert position.realized_pnl == Decimal("2000")
    
    def test_position_from_hyperliquid(self):
        """Test Position creation from HyperLiquid data."""
        hl_data = {
            'position': {
                'coin': 'BTC',
                'szi': '1.5',  # Positive for long
                'entryPx': '50000.0',
                'positionValue': '75000.0',
                'leverage': {
                    'type': 'cross',
                    'value': 5
                },
                'marginUsed': '15000.0'
            }
        }
        
        position = Position.from_hyperliquid(hl_data)
        
        assert position.symbol == "BTC"
        assert position.side == PositionSide.LONG
        assert position.size == Decimal("1.5")
        assert position.entry_price == Decimal("50000.0")
        assert position.leverage == Decimal("5")
        assert position.is_open is True
    
    def test_position_from_hyperliquid_short(self):
        """Test Position creation from HyperLiquid short data."""
        hl_data = {
            'position': {
                'coin': 'ETH',
                'szi': '-2.0',  # Negative for short
                'entryPx': '3000.0',
                'leverage': {
                    'value': 3
                }
            }
        }
        
        position = Position.from_hyperliquid(hl_data)
        
        assert position.symbol == "ETH"
        assert position.side == PositionSide.SHORT
        assert position.size == Decimal("2.0")  # Absolute value
        assert position.entry_price == Decimal("3000.0")
        assert position.leverage == Decimal("3")


class TestOrder:
    """Test Order model functionality."""
    
    def test_valid_limit_order(self):
        """Test creation of valid limit order."""
        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            time_in_force=TimeInForce.GTC
        )
        
        assert order.symbol == "BTC"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("1.0")
        assert order.price == Decimal("50000")
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == Decimal("0")
    
    def test_valid_market_order(self):
        """Test creation of valid market order."""
        order = Order(
            symbol="ETH",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("5.0")
        )
        
        assert order.order_type == OrderType.MARKET
        assert order.price is None  # Market orders don't have price
        assert order.side == OrderSide.SELL
    
    def test_order_computed_properties(self):
        """Test order computed properties."""
        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0.5")
        )
        
        # Remaining quantity: 2.0 - 0.5 = 1.5
        assert order.remaining_quantity == Decimal("1.5")
        
        # Fill percentage: (0.5 / 2.0) * 100 = 25%
        assert order.fill_percentage == Decimal("25")
        
        # Is complete: false (not fully filled)
        assert order.is_complete is False
        
        # Notional value: 2.0 * 50000 = 100000
        assert order.notional_value == Decimal("100000")
    
    def test_order_fill_update(self):
        """Test order fill update functionality."""
        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2.0"),
            price=Decimal("50000")
        )
        
        # First fill
        order.update_fill(Decimal("0.5"), Decimal("49900"), Decimal("10"))
        
        assert order.filled_quantity == Decimal("0.5")
        assert order.average_fill_price == Decimal("49900")
        assert order.commission == Decimal("10")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # Second fill
        order.update_fill(Decimal("1.5"), Decimal("50100"), Decimal("25"))
        
        assert order.filled_quantity == Decimal("2.0")
        assert order.is_complete is True
        assert order.status == OrderStatus.FILLED
        assert order.commission == Decimal("35")
        
        # Calculate weighted average price: (0.5 * 49900 + 1.5 * 50100) / 2.0 = 50050
        assert order.average_fill_price == Decimal("50050")
    
    def test_order_cancellation(self):
        """Test order cancellation."""
        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.OPEN
        )
        
        order.cancel()
        
        assert order.status == OrderStatus.CANCELLED
        assert order.updated_timestamp is not None
    
    def test_order_from_hyperliquid(self):
        """Test Order creation from HyperLiquid data."""
        hl_data = {
            'order': {
                'oid': 'order_123',
                'cloid': 'client_order_123',
                'coin': 'BTC',
                'side': 'B',  # B for buy
                'orderType': 'Limit',
                'sz': '1.5',
                'px': '50000.0',
                'timestamp': 1705316200000
            }
        }
        
        order = Order.from_hyperliquid(hl_data)
        
        assert order.order_id == "order_123"
        assert order.client_order_id == "client_order_123"
        assert order.symbol == "BTC"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("1.5")
        assert order.price == Decimal("50000.0")
        assert order.status == OrderStatus.OPEN
    
    def test_order_from_hyperliquid_sell(self):
        """Test Order creation from HyperLiquid sell data."""
        hl_data = {
            'order': {
                'coin': 'ETH',
                'side': 'A',  # A for sell
                'orderType': 'Market',
                'sz': '2.0'
            }
        }
        
        order = Order.from_hyperliquid(hl_data)
        
        assert order.symbol == "ETH"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET


class TestTradeExecution:
    """Test TradeExecution model functionality."""
    
    def test_valid_trade_execution(self):
        """Test creation of valid trade execution."""
        execution = TradeExecution(
            execution_id="exec_123",
            order_id="order_456",
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            commission=Decimal("25"),
            is_maker=True,
            liquidity="maker"
        )
        
        assert execution.execution_id == "exec_123"
        assert execution.order_id == "order_456"
        assert execution.symbol == "BTC"
        assert execution.side == OrderSide.BUY
        assert execution.quantity == Decimal("1.0")
        assert execution.price == Decimal("50000")
        assert execution.commission == Decimal("25")
        assert execution.is_maker is True
        assert execution.liquidity == "maker"
    
    def test_trade_execution_calculations(self):
        """Test trade execution value calculations."""
        execution = TradeExecution(
            execution_id="exec_123",
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            commission=Decimal("50")
        )
        
        # Notional value: 2.0 * 50000 = 100000
        assert execution.notional_value == Decimal("100000")
        
        # Net proceeds for buy: 100000 + 50 = 100050 (commission adds to cost)
        assert execution.net_proceeds == Decimal("100050")
    
    def test_trade_execution_sell_calculations(self):
        """Test trade execution calculations for sell orders."""
        execution = TradeExecution(
            execution_id="exec_456",
            symbol="ETH",
            side=OrderSide.SELL,
            quantity=Decimal("5.0"),
            price=Decimal("3000"),
            timestamp=datetime.now(timezone.utc),
            commission=Decimal("75")
        )
        
        # Notional value: 5.0 * 3000 = 15000
        assert execution.notional_value == Decimal("15000")
        
        # Net proceeds for sell: 15000 - 75 = 14925 (commission reduces proceeds)
        assert execution.net_proceeds == Decimal("14925")
    
    def test_trade_execution_from_hyperliquid(self):
        """Test TradeExecution creation from HyperLiquid data."""
        hl_data = {
            'fill': {
                'tid': 'execution_123',
                'oid': 'order_123',
                'coin': 'BTC',
                'side': 'B',  # B for buy
                'sz': '1.0',
                'px': '50000.0',
                'time': 1705316200000,
                'fee': '25.0'
            }
        }
        
        execution = TradeExecution.from_hyperliquid(hl_data)
        
        assert execution.execution_id == "execution_123"
        assert execution.order_id == "order_123"
        assert execution.symbol == "BTC"
        assert execution.side == OrderSide.BUY
        assert execution.quantity == Decimal("1.0")
        assert execution.price == Decimal("50000.0")
        assert execution.commission == Decimal("25.0")


class TestRiskParameters:
    """Test RiskParameters model functionality."""
    
    def test_valid_risk_parameters(self):
        """Test creation of valid risk parameters."""
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_leverage=Decimal("10"),
            max_portfolio_exposure=Decimal("500000"),
            stop_loss_percentage=Decimal("5"),
            take_profit_percentage=Decimal("15"),
            daily_loss_limit=Decimal("10000")
        )
        
        assert risk_params.max_position_size == Decimal("100000")
        assert risk_params.max_leverage == Decimal("10")
        assert risk_params.stop_loss_percentage == Decimal("5")
        assert risk_params.take_profit_percentage == Decimal("15")
    
    def test_risk_validation_methods(self):
        """Test risk validation methods."""
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_leverage=Decimal("5")
        )
        
        # Position size validation
        assert risk_params.validate_position_size(Decimal("2"), Decimal("40000")) is True  # 80k < 100k
        assert risk_params.validate_position_size(Decimal("3"), Decimal("40000")) is False  # 120k > 100k
        
        # Leverage validation
        assert risk_params.validate_leverage(Decimal("3")) is True
        assert risk_params.validate_leverage(Decimal("10")) is False  # > max_leverage of 5
    
    def test_position_sizing_calculations(self):
        """Test position sizing calculations."""
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_leverage=Decimal("10")
        )
        
        # Max position size at price 50000 with leverage 2: (100000 * 2) / 50000 = 4
        max_size = risk_params.calculate_max_position_size(Decimal("50000"), Decimal("2"))
        assert max_size == Decimal("4")
    
    def test_stop_loss_calculations(self):
        """Test stop loss price calculations."""
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            stop_loss_percentage=Decimal("5")
        )
        
        entry_price = Decimal("50000")
        
        # Long stop loss: 50000 * (1 - 0.05) = 47500
        long_stop = risk_params.calculate_stop_loss_price(entry_price, PositionSide.LONG)
        assert long_stop == Decimal("47500")
        
        # Short stop loss: 50000 * (1 + 0.05) = 52500
        short_stop = risk_params.calculate_stop_loss_price(entry_price, PositionSide.SHORT)
        assert short_stop == Decimal("52500")
    
    def test_take_profit_calculations(self):
        """Test take profit price calculations."""
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            take_profit_percentage=Decimal("10")
        )
        
        entry_price = Decimal("50000")
        
        # Long take profit: 50000 * (1 + 0.10) = 55000
        long_tp = risk_params.calculate_take_profit_price(entry_price, PositionSide.LONG)
        assert long_tp == Decimal("55000")
        
        # Short take profit: 50000 * (1 - 0.10) = 45000
        short_tp = risk_params.calculate_take_profit_price(entry_price, PositionSide.SHORT)
        assert short_tp == Decimal("45000")


class TestPortfolioState:
    """Test PortfolioState model functionality."""
    
    def test_valid_portfolio_state(self):
        """Test creation of valid portfolio state."""
        positions = [
            Position(
                symbol="BTC",
                side=PositionSide.LONG,
                size=Decimal("1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000")
            ),
            Position(
                symbol="ETH",
                side=PositionSide.SHORT,
                size=Decimal("5"),
                entry_price=Decimal("3000"),
                current_price=Decimal("2950")
            )
        ]
        
        portfolio = PortfolioState(
            account_id="test_account",
            total_balance=Decimal("100000"),
            available_balance=Decimal("75000"),
            margin_used=Decimal("25000"),
            positions=positions
        )
        
        assert portfolio.account_id == "test_account"
        assert portfolio.total_balance == Decimal("100000")
        assert portfolio.available_balance == Decimal("75000")
        assert len(portfolio.positions) == 2
    
    def test_portfolio_computed_properties(self):
        """Test portfolio computed properties."""
        positions = [
            Position(
                symbol="BTC",
                side=PositionSide.LONG,
                size=Decimal("2"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000")
            ),
            Position(
                symbol="ETH",
                side=PositionSide.SHORT,
                size=Decimal("10"),
                entry_price=Decimal("3000"),
                current_price=Decimal("2900")
            )
        ]
        
        portfolio = PortfolioState(
            account_id="test_account",
            total_balance=Decimal("100000"),
            available_balance=Decimal("75000"),
            margin_used=Decimal("25000"),
            unrealized_pnl=Decimal("3000"),
            positions=positions
        )
        
        # Equity: 100000 + 3000 = 103000
        assert portfolio.equity == Decimal("103000")
        
        # Margin level: (103000 / 25000) * 100 = 412%
        assert portfolio.margin_level == Decimal("412")
        
        # Total exposure: (2 * 51000) + (10 * 2900) = 102000 + 29000 = 131000
        assert portfolio.total_exposure == Decimal("131000")
        
        # Leverage ratio: 131000 / 103000 ≈ 1.27
        expected_leverage = Decimal("131000") / Decimal("103000")
        assert abs(portfolio.leverage_ratio - expected_leverage) < Decimal("0.01")
        
        # Open position count
        assert portfolio.open_position_count == 2
        
        # Long exposure: 2 * 51000 = 102000
        assert portfolio.long_exposure == Decimal("102000")
        
        # Short exposure: 10 * 2900 = 29000
        assert portfolio.short_exposure == Decimal("29000")
        
        # Net exposure: 102000 - 29000 = 73000
        assert portfolio.net_exposure == Decimal("73000")
    
    def test_portfolio_position_management(self):
        """Test portfolio position management methods."""
        portfolio = PortfolioState(
            account_id="test_account",
            total_balance=Decimal("100000"),
            available_balance=Decimal("75000")
        )
        
        # Add position
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000")
        )
        
        portfolio.add_position(position)
        assert len(portfolio.positions) == 1
        assert portfolio.get_position("BTC") is not None
        
        # Remove position
        portfolio.remove_position("BTC")
        assert len(portfolio.positions) == 0
        assert portfolio.get_position("BTC") is None
    
    def test_portfolio_from_hyperliquid(self):
        """Test PortfolioState creation from HyperLiquid data."""
        hl_data = {
            'user': 'test_user',
            'marginSummary': {
                'accountValue': '100000.0',
                'totalMarginUsed': '25000.0',
                'totalNtlPos': '125000.0',
                'totalUnrealizedPnl': '1500.0'
            },
            'assetPositions': [
                {
                    'position': {
                        'coin': 'BTC',
                        'szi': '1.0',
                        'entryPx': '50000.0',
                        'leverage': {'value': 5}
                    }
                }
            ]
        }
        
        portfolio = PortfolioState.from_hyperliquid(hl_data)
        
        assert portfolio.account_id == "test_user"
        assert portfolio.total_balance == Decimal("100000.0")
        assert portfolio.margin_used == Decimal("25000.0")
        assert portfolio.unrealized_pnl == Decimal("1500.0")
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].symbol == "BTC"


class TestTradingIntegration:
    """Test integration between trading models."""
    
    def test_complete_trading_workflow(self):
        """Test a complete trading workflow with all models."""
        # 1. Create risk parameters
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_leverage=Decimal("10"),
            stop_loss_percentage=Decimal("5"),
            take_profit_percentage=Decimal("10")
        )
        
        # 2. Create a limit order
        order = Order(
            order_id="order_123",
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        # 3. Simulate order fill
        execution = TradeExecution(
            execution_id="exec_123",
            order_id="order_123",
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("49950"),
            timestamp=datetime.now(timezone.utc),
            commission=Decimal("25")
        )
        
        # Update order with fill
        order.update_fill(execution.quantity, execution.price, execution.commission)
        
        # 4. Create position from filled order
        position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=execution.quantity,
            entry_price=execution.price,
            current_price=Decimal("51000")
        )
        
        # 5. Create portfolio state
        portfolio = PortfolioState(
            account_id="test_account",
            total_balance=Decimal("100000"),
            available_balance=Decimal("50000"),
            positions=[position],
            pending_orders=[order] if not order.is_complete else []
        )
        
        # Verify complete workflow
        assert order.is_complete is True
        assert order.status == OrderStatus.FILLED
        assert position.unrealized_pnl == Decimal("1050")  # (51000 - 49950) * 1
        assert portfolio.open_position_count == 1
        assert portfolio.total_exposure == Decimal("51000")
        
        # Test risk validation
        assert risk_params.validate_position_size(position.size, position.entry_price) is True
        assert risk_params.validate_leverage(position.leverage) is True
        
        print(f"Order filled: {order.fill_percentage}%")
        print(f"Position PnL: {position.unrealized_pnl}")
        print(f"Portfolio equity: {portfolio.equity}")
    
    def test_round_trip_hyperliquid_conversions(self):
        """Test round-trip conversions with HyperLiquid format."""
        # Create original models
        original_position = Position(
            symbol="BTC",
            side=PositionSide.LONG,
            size=Decimal("1.5"),
            entry_price=Decimal("50000"),
            leverage=Decimal("5")
        )
        
        original_order = Order(
            order_id="order_123",
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        # Convert to HyperLiquid format and back (conceptually)
        # This validates that our from_hyperliquid methods work correctly
        hl_position_data = {
            'position': {
                'coin': original_position.symbol,
                'szi': str(original_position.size),
                'entryPx': str(original_position.entry_price),
                'leverage': {'value': int(original_position.leverage)}
            }
        }
        
        hl_order_data = {
            'order': {
                'oid': original_order.order_id,
                'coin': original_order.symbol,
                'side': 'B',
                'orderType': 'Limit',
                'sz': str(original_order.quantity),
                'px': str(original_order.price)
            }
        }
        
        # Convert back from HyperLiquid format
        converted_position = Position.from_hyperliquid(hl_position_data)
        converted_order = Order.from_hyperliquid(hl_order_data)
        
        # Verify conversions
        assert converted_position.symbol == original_position.symbol
        assert converted_position.size == original_position.size
        assert converted_position.entry_price == original_position.entry_price
        assert converted_position.leverage == original_position.leverage
        
        assert converted_order.order_id == original_order.order_id
        assert converted_order.symbol == original_order.symbol
        assert converted_order.quantity == original_order.quantity
        assert converted_order.price == original_order.price 