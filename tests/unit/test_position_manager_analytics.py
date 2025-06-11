"""
Test suite for Position Manager Agent Enhanced Performance Analytics (Task 13.4)

Tests the comprehensive performance analytics capabilities added to the Position Manager Agent.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.bistoury.agents.position_manager_agent import (
    PositionManagerAgent, PositionManagerConfig,
    PerformanceAnalyzer, PerformanceSnapshot, TradeAnalysis, AdvancedMetrics
)
from src.bistoury.models.trading import (
    Position, PositionSide, Order, OrderType, OrderStatus, OrderSide,
    TradeExecution, PortfolioState, TimeInForce
)
from src.bistoury.models.agent_messages import TradingSignalPayload


class TestPerformanceAnalyzer:
    """Test the PerformanceAnalyzer class."""
    
    def test_performance_analyzer_initialization(self):
        """Test PerformanceAnalyzer initialization."""
        initial_balance = Decimal('100000')
        analyzer = PerformanceAnalyzer(initial_balance)
        
        assert analyzer.initial_balance == initial_balance
        assert analyzer.snapshots == []
        assert analyzer.daily_returns == []
        assert analyzer.peak_equity == initial_balance
        assert analyzer.max_drawdown == Decimal('0')
        assert analyzer.max_drawdown_duration == timedelta(0)
        assert analyzer.drawdown_start is None
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        analyzer = PerformanceAnalyzer(Decimal('100000'))
        
        # No drawdown when at peak
        drawdown, drawdown_pct = analyzer.calculate_drawdown(Decimal('100000'))
        assert drawdown == Decimal('0')
        assert drawdown_pct == Decimal('0')
        
        # Drawdown when below peak
        drawdown, drawdown_pct = analyzer.calculate_drawdown(Decimal('95000'))
        assert drawdown == Decimal('5000')
        assert drawdown_pct == Decimal('5')
    
    def test_snapshot_tracking(self):
        """Test performance snapshot tracking."""
        analyzer = PerformanceAnalyzer(Decimal('100000'))
        
        snapshot1 = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_balance=Decimal('100000'),
            total_pnl=Decimal('0'),
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0'),
            equity=Decimal('100000'),
            drawdown=Decimal('0'),
            drawdown_pct=Decimal('0')
        )
        
        analyzer.add_snapshot(snapshot1)
        assert len(analyzer.snapshots) == 1
        assert analyzer.peak_equity == Decimal('100000')
        
        # Add snapshot with profit
        snapshot2 = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc) + timedelta(hours=1),
            total_balance=Decimal('105000'),
            total_pnl=Decimal('5000'),
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('5000'),
            equity=Decimal('105000'),
            drawdown=Decimal('0'),
            drawdown_pct=Decimal('0')
        )
        
        analyzer.add_snapshot(snapshot2)
        assert len(analyzer.snapshots) == 2
        assert analyzer.peak_equity == Decimal('105000')
        assert len(analyzer.daily_returns) == 1
        assert analyzer.daily_returns[0] == 0.05  # 5% return
    
    def test_trade_analysis_empty(self):
        """Test trade analysis with no executions."""
        analyzer = PerformanceAnalyzer(Decimal('100000'))
        analysis = analyzer.get_trade_analysis([])
        
        assert analysis.total_trades == 0
        assert analysis.winning_trades == 0
        assert analysis.losing_trades == 0
        assert analysis.win_rate == 0.0
        assert analysis.total_pnl == 0.0
        assert analysis.avg_win == 0.0
        assert analysis.avg_loss == 0.0
        assert analysis.profit_factor == 0.0
    
    def test_trade_analysis_with_executions(self):
        """Test trade analysis with sample executions."""
        analyzer = PerformanceAnalyzer(Decimal('100000'))
        
        # Create sample executions (buy and sell for one profitable trade)
        buy_execution = TradeExecution(
            execution_id="1",
            order_id="order1",
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=Decimal('1'),
            price=Decimal('50000'),
            timestamp=datetime.now(timezone.utc),
            commission=Decimal('25'),
            is_maker=False,
            liquidity="taker"
        )
        
        sell_execution = TradeExecution(
            execution_id="2",
            order_id="order2",
            symbol="BTC",
            side=OrderSide.SELL,
            quantity=Decimal('1'),
            price=Decimal('52000'),
            timestamp=datetime.now(timezone.utc) + timedelta(hours=2),
            commission=Decimal('26'),
            is_maker=False,
            liquidity="taker"
        )
        
        executions = [buy_execution, sell_execution]
        analysis = analyzer.get_trade_analysis(executions)
        
        # Should detect one profitable trade
        assert analysis.total_trades == 1
        assert analysis.winning_trades == 1
        assert analysis.losing_trades == 0
        assert analysis.win_rate == 100.0
        assert analysis.total_pnl == 1949.0  # 2000 profit - 51 commission
        assert analysis.avg_win == 1949.0
        assert analysis.largest_win == 1949.0
        assert analysis.profit_factor > 1000  # Very high since no losses
    
    def test_advanced_metrics_empty(self):
        """Test advanced metrics with no data."""
        analyzer = PerformanceAnalyzer(Decimal('100000'))
        metrics = analyzer.get_advanced_metrics()
        
        assert metrics.total_return == 0.0
        assert metrics.annualized_return == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0


class TestPositionManagerAnalytics:
    """Test the enhanced Position Manager Agent analytics."""
    
    @pytest.fixture
    def config(self):
        """Position Manager configuration for testing."""
        return PositionManagerConfig(
            initial_balance=Decimal('100000'),
            slippage_rate=Decimal('0.0005'),
            taker_fee_rate=Decimal('0.00045'),  # HyperLiquid rates
            maker_fee_rate=Decimal('0.00015'),  # HyperLiquid rates
            enable_stop_loss=True,
            enable_take_profit=True,
            stop_loss_pct=Decimal('2.0'),
            take_profit_pct=Decimal('4.0'),
            min_position_size=Decimal('0.001'),  # Much smaller minimum position
            max_position_size=Decimal('1.0')     # Smaller maximum too
        )
    
    @pytest.fixture
    def position_manager(self, config):
        """Create Position Manager Agent for testing."""
        agent = PositionManagerAgent(name="test_pm", config=config)
        agent._message_bus = AsyncMock()
        return agent
    
    def test_performance_analyzer_initialization(self, position_manager):
        """Test that PerformanceAnalyzer is properly initialized."""
        assert position_manager.performance_analyzer is not None
        assert position_manager.performance_analyzer.initial_balance == Decimal('100000')
        assert position_manager.last_snapshot_time is None
    
    def test_basic_performance_metrics(self, position_manager):
        """Test basic performance metrics retrieval."""
        metrics = position_manager.get_performance_metrics()
        
        # Check basic metrics are present
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics
        assert 'equity' in metrics
        
        # Check enhanced metrics are present
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'volatility' in metrics
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        
        # Initial values should be zero/empty
        assert metrics['total_trades'] == 0
        assert metrics['winning_trades'] == 0
        assert metrics['total_pnl'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0
    
    def test_trade_analysis_access(self, position_manager):
        """Test trade analysis method access."""
        analysis = position_manager.get_trade_analysis()
        
        assert isinstance(analysis, TradeAnalysis)
        assert analysis.total_trades == 0
        assert analysis.winning_trades == 0
        assert analysis.win_rate == 0.0
    
    def test_advanced_metrics_access(self, position_manager):
        """Test advanced metrics method access."""
        metrics = position_manager.get_advanced_metrics()
        
        assert isinstance(metrics, AdvancedMetrics)
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
    
    def test_performance_snapshots_access(self, position_manager):
        """Test performance snapshots access."""
        snapshots = position_manager.get_performance_snapshots()
        
        assert isinstance(snapshots, list)
        assert len(snapshots) == 0  # No snapshots initially
    
    @pytest.mark.asyncio
    async def test_performance_snapshot_creation(self, position_manager):
        """Test performance snapshot creation."""
        await position_manager._create_performance_snapshot()
        
        assert position_manager.last_snapshot_time is not None
        assert len(position_manager.performance_analyzer.snapshots) == 1
        
        snapshot = position_manager.performance_analyzer.snapshots[0]
        assert snapshot.total_balance == Decimal('100000')
        assert snapshot.equity == Decimal('100000')
        assert snapshot.drawdown == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self, position_manager):
        """Test comprehensive performance report generation."""
        report = await position_manager.generate_performance_report()
        
        # Check report structure
        assert 'report_timestamp' in report
        assert 'account_summary' in report
        assert 'trading_summary' in report
        assert 'risk_metrics' in report
        assert 'current_positions' in report
        assert 'recent_trades' in report
        
        # Check account summary
        account = report['account_summary']
        assert account['initial_balance'] == 100000.0
        assert account['current_equity'] == 100000.0
        assert account['total_return'] == 0.0
        
        # Check trading summary
        trading = report['trading_summary']
        assert trading['total_trades'] == 0
        assert trading['win_rate'] == 0.0
        
        # Check risk metrics
        risk = report['risk_metrics']
        assert 'sharpe_ratio' in risk
        assert 'max_drawdown' in risk
        assert 'volatility' in risk
    
    @pytest.mark.asyncio
    async def test_performance_tracking_during_trade(self, position_manager):
        """Test that performance is tracked during trade execution."""
        # Mock market price
        position_manager.current_prices["BTC"] = Decimal('50000')
        
        # Create and execute a trade signal with smaller position
        signal = TradingSignalPayload(
            symbol="BTC",
            signal_type="CANDLESTICK",
            direction="BUY",
            confidence=0.1,  # Lower confidence to get smaller position size
            strength=1.0,     # Lower strength
            timeframe="15m",
            strategy="candlestick_strategy",
            reasoning="Weak bullish pattern detected"
        )
        
        await position_manager._execute_signal(signal)
        
        # Check that execution was recorded
        assert len(position_manager.executions) == 1
        assert position_manager.total_trades == 1
        
        # Check metrics update
        metrics = position_manager.get_performance_metrics()
        assert metrics['total_trades'] == 1
        assert metrics['open_positions'] == 1
    
    def test_performance_metrics_error_handling(self, position_manager):
        """Test error handling in performance metrics."""
        # Temporarily break the analyzer to test error handling
        position_manager.performance_analyzer = None
        
        metrics = position_manager.get_performance_metrics()
        
        # Should return basic metrics and error
        assert 'error' in metrics
        assert 'total_trades' in metrics
        assert metrics['total_trades'] == 0


class TestPerformanceIntegration:
    """Integration tests for performance analytics."""
    
    @pytest.mark.asyncio
    async def test_full_trading_cycle_analytics(self):
        """Test analytics through a complete trading cycle."""
        config = PositionManagerConfig(initial_balance=Decimal('100000'))
        agent = PositionManagerAgent(name="test_pm", config=config)
        agent._message_bus = AsyncMock()
        
        # Initialize
        await agent._create_performance_snapshot()
        initial_metrics = agent.get_performance_metrics()
        
        # Add some mock trade executions
        buy_execution = TradeExecution(
            execution_id="1",
            order_id="order1",
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=Decimal('1'),
            price=Decimal('50000'),
            timestamp=datetime.now(timezone.utc),
            commission=Decimal('25'),
            is_maker=False,
            liquidity="taker"
        )
        
        sell_execution = TradeExecution(
            execution_id="2",
            order_id="order2",
            symbol="BTC",
            side=OrderSide.SELL,
            quantity=Decimal('1'),
            price=Decimal('52000'),
            timestamp=datetime.now(timezone.utc) + timedelta(hours=2),
            commission=Decimal('26'),
            is_maker=False,
            liquidity="taker"
        )
        
        agent.executions.extend([buy_execution, sell_execution])
        agent.total_trades = 1
        agent.winning_trades = 1
        agent.total_pnl = Decimal('1949')  # 2000 - 51 commission
        
        # Create another snapshot to see changes
        await agent._create_performance_snapshot()
        final_metrics = agent.get_performance_metrics()
        
        # Verify analytics captured the trade
        assert final_metrics['total_trades'] == 1
        assert final_metrics['winning_trades'] == 1
        assert final_metrics['win_rate'] == 100.0
        
        # Generate comprehensive report
        report = await agent.generate_performance_report()
        assert report['trading_summary']['total_trades'] == 1
        assert len(report['recent_trades']) == 2  # Buy and sell executions


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 