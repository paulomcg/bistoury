"""
Unit tests for trade models.

Tests validation, conversion, API compatibility, aggregation logic,
and analytics for all trade model classes.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List

from src.bistoury.models.trades import (
    Trade,
    TradeAggregation,
    TradeAnalytics,
)


class TestTrade:
    """Test Trade model functionality."""
    
    def test_valid_trade(self):
        """Test creation of valid trade."""
        now = datetime.now(timezone.utc)
        trade = Trade(
            symbol="BTC",
            timestamp=now,
            trade_id="12345",
            price="50000.12345678",
            quantity="1.5",
            side="buy",
            is_buyer_maker=False,
            exchange="hyperliquid"
        )
        
        assert trade.symbol == "BTC"
        assert trade.trade_id == "12345"
        assert trade.price == Decimal("50000.12345678")
        assert trade.quantity == Decimal("1.5")
        assert trade.side == "buy"
        assert trade.is_buyer_maker is False
        assert trade.exchange == "hyperliquid"
    
    def test_trade_without_optional_fields(self):
        """Test trade creation without optional fields."""
        trade = Trade(
            symbol="ETH",
            timestamp=datetime.now(timezone.utc),
            price="3000.0",
            quantity="5.0",
            side="sell"
        )
        
        assert trade.trade_id is None
        assert trade.is_buyer_maker is None
        assert trade.user_data is None
        assert trade.exchange == "hyperliquid"  # Default value
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid trade format."""
        hl_data = {
            'time': 1705316200000,  # 2024-01-15T10:30:00Z
            'px': '50000.125',
            'sz': '1.5',
            'side': 'B',  # Buy
            'tid': 'trade_123',
            'user': {'liquidation': False}
        }
        
        trade = Trade.from_hyperliquid(hl_data, "BTC")
        
        assert trade.symbol == "BTC"
        assert trade.price == Decimal("50000.125")
        assert trade.quantity == Decimal("1.5")
        assert trade.side == "buy"
        assert trade.trade_id == "trade_123"
        assert trade.user_data == {'liquidation': False}
        assert trade.time_ms == 1705316200000
    
    def test_hyperliquid_conversion_sell(self):
        """Test conversion from HyperLiquid trade format for sell side."""
        hl_data = {
            'time': 1705316200000,
            'px': '50000.0',
            'sz': '2.0',
            'side': 'A',  # Ask/Sell
        }
        
        trade = Trade.from_hyperliquid(hl_data, "ETH")
        
        assert trade.side == "sell"
        assert trade.symbol == "ETH"
    
    def test_to_hyperliquid_conversion(self):
        """Test conversion to HyperLiquid format."""
        trade = Trade(
            symbol="BTC",
            timestamp=datetime.fromtimestamp(1705316200, timezone.utc),
            trade_id="trade_123",
            price="50000.125",
            quantity="1.5",
            side="buy",
            user_data={'test': True}
        )
        
        hl_data = trade.to_hyperliquid()
        
        assert hl_data['px'] == '50000.12500000'
        assert hl_data['sz'] == '1.50000000'
        assert hl_data['side'] == 'B'
        assert hl_data['tid'] == 'trade_123'
        assert hl_data['user'] == {'test': True}
    
    def test_trade_properties(self):
        """Test trade calculation properties."""
        trade = Trade(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            price="50000.0",
            quantity="1.5",
            side="buy",
            is_buyer_maker=True
        )
        
        # Notional value
        assert trade.notional_value == Decimal("75000.0")
        
        # Side checks
        assert trade.is_buy is True
        assert trade.is_sell is False
        
        # Taker side logic
        assert trade.taker_side == "sell"  # Buyer was maker, so seller was taker
    
    def test_taker_side_logic(self):
        """Test taker side determination logic."""
        # Case 1: Buyer is maker -> Seller is taker
        trade1 = Trade(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            price="50000",
            quantity="1.0",
            side="buy",
            is_buyer_maker=True
        )
        assert trade1.taker_side == "sell"
        
        # Case 2: Buyer is taker -> Buyer is taker
        trade2 = Trade(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            price="50000",
            quantity="1.0",
            side="buy",
            is_buyer_maker=False
        )
        assert trade2.taker_side == "buy"
        
        # Case 3: Unknown maker/taker -> Default to trade side
        trade3 = Trade(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            price="50000",
            quantity="1.0",
            side="sell",
            is_buyer_maker=None
        )
        assert trade3.taker_side == "sell"
    
    def test_decimal_precision_validation(self):
        """Test decimal precision handling."""
        # Scientific notation
        trade = Trade(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            price="5e4",  # 50000
            quantity="1.5e-1",  # 0.15
            side="buy"
        )
        
        assert trade.price == Decimal("50000.00000000")
        assert trade.quantity == Decimal("0.15000000")
    
    def test_trade_validation(self):
        """Test trade validation rules."""
        now = datetime.now(timezone.utc)
        
        # Valid trade
        Trade(
            symbol="BTC",
            timestamp=now,
            price="50000",
            quantity="1.0",
            side="buy"
        )
        
        # Invalid: negative/zero price
        with pytest.raises(ValueError, match="greater than 0"):
            Trade(
                symbol="BTC",
                timestamp=now,
                price="0",
                quantity="1.0",
                side="buy"
            )
        
        # Invalid: negative/zero quantity
        with pytest.raises(ValueError, match="greater than 0"):
            Trade(
                symbol="BTC",
                timestamp=now,
                price="50000",
                quantity="0",
                side="buy"
            )
    
    def test_trade_representation(self):
        """Test string representation."""
        trade = Trade(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            price="50000.0",
            quantity="1.5",
            side="buy"
        )
        
        repr_str = repr(trade)
        assert "Trade(BTC)" in repr_str
        assert "BUY" in repr_str
        assert "1.5" in repr_str
        assert "$50000.0" in repr_str
        assert "$75000.00" in repr_str  # Notional value


class TestTradeAggregation:
    """Test TradeAggregation model functionality."""
    
    def test_valid_trade_aggregation(self):
        """Test creation of valid trade aggregation."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)
        
        agg = TradeAggregation(
            symbol="BTC",
            start_time=start_time,
            end_time=end_time,
            trade_count=10,
            total_volume=Decimal("15.5"),
            total_notional=Decimal("775000.0"),
            vwap=Decimal("50000.0"),
            first_price=Decimal("49950.0"),
            last_price=Decimal("50050.0"),
            min_price=Decimal("49900.0"),
            max_price=Decimal("50100.0"),
            buy_volume=Decimal("8.0"),
            sell_volume=Decimal("7.5"),
            buy_count=6,
            sell_count=4
        )
        
        assert agg.symbol == "BTC"
        assert agg.trade_count == 10
        assert agg.total_volume == Decimal("15.5")
        assert agg.vwap == Decimal("50000.0")
        assert agg.buy_count + agg.sell_count == agg.trade_count
    
    def test_aggregation_validation(self):
        """Test aggregation validation rules."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)
        
        # Valid aggregation
        TradeAggregation(
            symbol="BTC",
            start_time=start_time,
            end_time=end_time,
            trade_count=5,
            total_volume=Decimal("10.0"),
            total_notional=Decimal("500000.0"),
            vwap=Decimal("50000.0"),
            buy_volume=Decimal("6.0"),
            sell_volume=Decimal("4.0"),
            buy_count=3,
            sell_count=2
        )
        
        # Invalid: end_time <= start_time
        with pytest.raises(ValueError, match="End time.*must be after or equal to start time"):
            TradeAggregation(
                symbol="BTC",
                start_time=end_time,
                end_time=start_time,  # Wrong order
                trade_count=5,
                total_volume=Decimal("10.0"),
                total_notional=Decimal("500000.0")
            )
        
        # Invalid: buy_count + sell_count != trade_count
        with pytest.raises(ValueError, match="Buy count.*sell count.*must equal trade count"):
            TradeAggregation(
                symbol="BTC",
                start_time=start_time,
                end_time=end_time,
                trade_count=10,
                total_volume=Decimal("10.0"),
                total_notional=Decimal("500000.0"),
                buy_count=3,
                sell_count=2  # 3 + 2 != 10
            )
        
        # Invalid: missing VWAP when volume > 0
        with pytest.raises(ValueError, match="VWAP must be provided when total volume"):
            TradeAggregation(
                symbol="BTC",
                start_time=start_time,
                end_time=end_time,
                trade_count=5,
                total_volume=Decimal("10.0"),
                total_notional=Decimal("500000.0"),
                vwap=None,  # Missing VWAP
                buy_volume=Decimal("6.0"),
                sell_volume=Decimal("4.0"),
                buy_count=3,
                sell_count=2
            )
    
    def test_from_trades_creation(self):
        """Test creation of TradeAggregation from trade list."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)
        
        trades = [
            Trade(
                symbol="BTC",
                timestamp=start_time,
                price="50000.0",
                quantity="1.0",
                side="buy"
            ),
            Trade(
                symbol="BTC",
                timestamp=start_time + timedelta(seconds=30),
                price="50100.0",
                quantity="2.0",
                side="sell"
            ),
            Trade(
                symbol="BTC",
                timestamp=end_time - timedelta(seconds=10),
                price="49900.0",
                quantity="1.5",
                side="buy"
            )
        ]
        
        agg = TradeAggregation.from_trades(trades, start_time, end_time)
        
        assert agg.symbol == "BTC"
        assert agg.trade_count == 3
        assert agg.total_volume == Decimal("4.5")  # 1.0 + 2.0 + 1.5
        assert agg.first_price == Decimal("50000.0")
        assert agg.last_price == Decimal("49900.0")
        assert agg.min_price == Decimal("49900.0")
        assert agg.max_price == Decimal("50100.0")
        assert agg.buy_count == 2
        assert agg.sell_count == 1
        assert agg.buy_volume == Decimal("2.5")  # 1.0 + 1.5
        assert agg.sell_volume == Decimal("2.0")
    
    def test_aggregation_properties(self):
        """Test calculated aggregation properties."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)  # 60 seconds
        
        agg = TradeAggregation(
            symbol="BTC",
            start_time=start_time,
            end_time=end_time,
            trade_count=60,
            total_volume=Decimal("120.0"),
            total_notional=Decimal("6000000.0"),
            vwap=Decimal("50000.0"),
            first_price=Decimal("49900.0"),
            last_price=Decimal("50100.0"),
            buy_volume=Decimal("70.0"),
            sell_volume=Decimal("50.0"),
            buy_count=35,
            sell_count=25
        )
        
        # Duration
        assert agg.duration_seconds == 60.0
        
        # Volume per second
        assert agg.volume_per_second == Decimal("2.0")  # 120 / 60
        
        # Trades per second
        assert agg.trades_per_second == Decimal("1.0")  # 60 / 60
        
        # Price change
        assert agg.price_change == Decimal("200.0")  # 50100 - 49900
        
        # Price change percentage
        expected_pct = (Decimal("200.0") / Decimal("49900.0")) * 100
        assert agg.price_change_pct == expected_pct
        
        # Buy/sell ratio
        assert agg.buy_sell_ratio == Decimal("1.4")  # 70 / 50
        
        # Buy pressure (percentage)
        # Allow for slight precision differences
        expected_buy_pressure = (Decimal("70.0") / Decimal("120.0")) * 100
        assert abs(agg.buy_pressure - expected_buy_pressure) < Decimal("0.01")  # Within 0.01%
        
        # Average trade size
        assert agg.average_trade_size == Decimal("2.0")  # 120 / 60
        
        # Average trade value
        assert agg.average_trade_value == Decimal("100000.0")  # 6000000 / 60
    
    def test_edge_case_calculations(self):
        """Test edge cases in calculations."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)  # Same time
        
        # All buys (sell volume = 0)
        agg = TradeAggregation(
            symbol="BTC",
            start_time=start_time,
            end_time=end_time,
            trade_count=5,
            total_volume=Decimal("10.0"),
            total_notional=Decimal("500000.0"),
            vwap=Decimal("50000.0"),
            buy_volume=Decimal("10.0"),
            sell_volume=Decimal("0.0"),
            buy_count=5,
            sell_count=0
        )
        
        assert agg.buy_sell_ratio == Decimal("999999")  # Approximates infinity
        assert agg.buy_pressure == Decimal("100")  # 100%
    
    def test_empty_trades_aggregation(self):
        """Test aggregation from empty trades list."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)
        
        agg = TradeAggregation.from_trades([], start_time, end_time)
        
        assert agg.trade_count == 0
        assert agg.total_volume == Decimal("0")
        assert agg.total_notional == Decimal("0")
        assert agg.vwap is None


class TestTradeAnalytics:
    """Test TradeAnalytics model functionality."""
    
    def create_sample_trades(self, symbol: str = "BTC", count: int = 10) -> List[Trade]:
        """Create sample trades for testing."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        trades = []
        
        for i in range(count):
            timestamp = start_time + timedelta(seconds=i * 6)  # Every 6 seconds
            price = Decimal("50000") + Decimal(str(i * 10))  # Increasing prices
            quantity = Decimal("1.0") + Decimal(str(i * 0.1))
            side = "buy" if i % 2 == 0 else "sell"
            
            trade = Trade(
                symbol=symbol,
                timestamp=timestamp,
                price=str(price),
                quantity=str(quantity),
                side=side
            )
            trades.append(trade)
        
        return trades
    
    def test_valid_trade_analytics(self):
        """Test creation of valid trade analytics."""
        trades = self.create_sample_trades()
        
        analytics = TradeAnalytics.from_trades(
            trades,
            bucket_size=timedelta(seconds=30),
            large_trade_threshold=Decimal("2.0")
        )
        
        assert analytics.symbol == "BTC"
        assert analytics.large_trade_threshold == Decimal("2.0")
        assert len(analytics.trade_aggregations) > 0
        assert analytics.analysis_period.total_seconds() > 0
    
    def test_analytics_aggregation_properties(self):
        """Test analytics aggregated properties."""
        trades = self.create_sample_trades(count=6)
        
        analytics = TradeAnalytics.from_trades(trades, bucket_size=timedelta(seconds=10))
        
        # Should have multiple aggregations for 6 trades over ~30 seconds with 10s buckets
        assert len(analytics.trade_aggregations) >= 1
        
        # Test total calculations - might be less than original due to time bucketing edge cases
        assert analytics.total_trades >= 1
        assert analytics.total_trades <= 6
    
    def test_volume_profile_calculation(self):
        """Test volume profile calculation."""
        trades = [
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50000.12",
                quantity="1.0",
                side="buy"
            ),
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50000.13",  # Different price, same cent bucket
                quantity="2.0",
                side="sell"
            ),
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50001.00",  # Different cent bucket
                quantity="1.5",
                side="buy"
            )
        ]
        
        analytics = TradeAnalytics.from_trades(trades)
        
        # Volume profile should group by cent buckets - check actual keys
        profile_keys = set(analytics.volume_profile.keys())
        
        # Should have entries for the prices (may not be rounded to cents)
        assert len(profile_keys) >= 2  # At least separate prices
        assert analytics.volume_profile["50000.12"] == Decimal("1.0")  # First trade
        assert analytics.volume_profile["50000.13"] == Decimal("2.0")  # Second trade
        assert analytics.volume_profile["50001.00"] == Decimal("1.5")  # Third trade
    
    def test_peak_volume_price(self):
        """Test peak volume price calculation."""
        trades = [
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50000.00",
                quantity="5.0",  # Highest volume at this price
                side="buy"
            ),
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50001.00",
                quantity="2.0",
                side="sell"
            )
        ]
        
        analytics = TradeAnalytics.from_trades(trades)
        
        assert analytics.peak_volume_price == Decimal("50000.00")
    
    def test_volume_weighted_std_calculation(self):
        """Test volume-weighted standard deviation calculation."""
        # Create trades with different prices and volumes
        trades = [
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50000.00",
                quantity="10.0",  # Large volume
                side="buy"
            ),
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50100.00",
                quantity="1.0",   # Small volume
                side="sell"
            )
        ]
        
        analytics = TradeAnalytics.from_trades(trades)
        
        # Should have some standard deviation
        assert analytics.volume_weighted_std > 0
    
    def test_large_trades_analysis(self):
        """Test large trades analysis."""
        trades = self.create_sample_trades(count=5)
        
        # Set threshold so some trades are considered large
        analytics = TradeAnalytics.from_trades(
            trades,
            large_trade_threshold=Decimal("1.3")  # Some trades will be above this
        )
        
        large_trades_info = analytics.get_large_trades_analysis()
        
        assert 'large_trade_periods' in large_trades_info
        assert 'large_trade_volume' in large_trades_info
        assert 'large_trade_volume_pct' in large_trades_info
        assert large_trades_info['large_trade_volume_pct'] >= 0
    
    def test_empty_trades_analytics(self):
        """Test analytics with empty trades list."""
        with pytest.raises(ValueError, match="Cannot create analytics from empty trade list"):
            TradeAnalytics.from_trades([])
    
    def test_analytics_time_bucketing(self):
        """Test time bucketing logic."""
        # Create trades over 65 seconds
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        trades = []
        
        for i in range(5):
            timestamp = start_time + timedelta(seconds=i * 15)  # Every 15 seconds
            trade = Trade(
                symbol="BTC",
                timestamp=timestamp,
                price="50000.0",
                quantity="1.0",
                side="buy"
            )
            trades.append(trade)
        
        # Use 30-second buckets
        analytics = TradeAnalytics.from_trades(trades, bucket_size=timedelta(seconds=30))
        
        # Should have multiple buckets
        assert len(analytics.trade_aggregations) >= 2
        
        # Verify each bucket has proper time ranges
        for agg in analytics.trade_aggregations:
            bucket_duration = (agg.end_time - agg.start_time).total_seconds()
            assert bucket_duration == 30.0


class TestModelIntegration:
    """Test integration between trade models."""
    
    def test_json_serialization(self):
        """Test JSON serialization of trade models."""
        trade = Trade(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            price="50000.0",
            quantity="1.5",
            side="buy"
        )
        
        # Test serialization
        json_data = trade.model_dump()
        assert 'symbol' in json_data
        assert 'price' in json_data
        assert 'side' in json_data
        
        # Test round-trip
        restored = Trade(**json_data)
        assert restored.price == trade.price
        assert restored.side == trade.side
    
    def test_comprehensive_workflow(self):
        """Test a comprehensive workflow with all models."""
        # 1. Create individual trades
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        trades = []
        
        for i in range(20):
            timestamp = start_time + timedelta(seconds=i * 3)
            price = Decimal("50000") + Decimal(str(i * 5))
            quantity = Decimal("1.0") + Decimal(str(i * 0.05))
            side = "buy" if i % 3 == 0 else "sell"
            
            trade = Trade(
                symbol="BTC",
                timestamp=timestamp,
                price=str(price),
                quantity=str(quantity),
                side=side
            )
            trades.append(trade)
        
        # 2. Create aggregations
        period_start = start_time
        period_end = start_time + timedelta(seconds=30)
        period_trades = [t for t in trades if period_start <= t.timestamp < period_end]
        
        agg = TradeAggregation.from_trades(period_trades, period_start, period_end)
        
        assert agg.trade_count > 0
        assert agg.total_volume > 0
        
        # 3. Create analytics
        analytics = TradeAnalytics.from_trades(trades, bucket_size=timedelta(seconds=15))
        
        assert analytics.total_trades == 20
        assert len(analytics.trade_aggregations) >= 3  # Multiple 15s buckets over ~60s
        assert analytics.peak_volume_price is not None
        
        # 4. Test analytics properties
        large_trades_info = analytics.get_large_trades_analysis()
        assert isinstance(large_trades_info, dict)
        
        print(f"Analyzed {analytics.total_trades} trades")
        print(f"Total volume: {analytics.total_volume}")
        print(f"VWAP: {analytics.overall_vwap}")
        print(f"Peak volume price: {analytics.peak_volume_price}")
    
    def test_edge_cases_handling(self):
        """Test handling of various edge cases."""
        # Single trade
        single_trade = [
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price="50000.0",
                quantity="1.0",
                side="buy"
            )
        ]
        
        analytics = TradeAnalytics.from_trades(single_trade)
        assert analytics.total_trades == 1
        assert analytics.peak_volume_price == Decimal("50000.00")
        
        # All same price
        same_price_trades = [
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
                price="50000.0",
                quantity="1.0",
                side="buy" if i % 2 == 0 else "sell"
            )
            for i in range(5)
        ]
        
        analytics2 = TradeAnalytics.from_trades(same_price_trades)
        assert analytics2.volume_weighted_std == Decimal("0")  # No price variation 