"""
Unit tests for core market data models.

Tests validation, conversion, API compatibility, and business logic
for all market data model classes.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any

from src.bistoury.models.market_data import (
    CandlestickData,
    Ticker,
    SymbolInfo,
    MarketData,
    Timeframe,
    PriceLevel,
)


class TestTimeframe:
    """Test Timeframe enum functionality."""
    
    def test_timeframe_values(self):
        """Test timeframe string values."""
        assert Timeframe.ONE_MINUTE == "1m"
        assert Timeframe.FIVE_MINUTES == "5m"
        assert Timeframe.FIFTEEN_MINUTES == "15m"
        assert Timeframe.ONE_HOUR == "1h"
        assert Timeframe.FOUR_HOURS == "4h"
        assert Timeframe.ONE_DAY == "1d"
    
    def test_timeframe_seconds(self):
        """Test timeframe conversion to seconds."""
        assert Timeframe.ONE_MINUTE.seconds == 60
        assert Timeframe.FIVE_MINUTES.seconds == 300
        assert Timeframe.FIFTEEN_MINUTES.seconds == 900
        assert Timeframe.ONE_HOUR.seconds == 3600
        assert Timeframe.FOUR_HOURS.seconds == 14400
        assert Timeframe.ONE_DAY.seconds == 86400
    
    def test_timeframe_milliseconds(self):
        """Test timeframe conversion to milliseconds."""
        assert Timeframe.ONE_MINUTE.milliseconds == 60000
        assert Timeframe.FIVE_MINUTES.milliseconds == 300000


class TestPriceLevel:
    """Test PriceLevel model validation and functionality."""
    
    def test_valid_price_level(self):
        """Test creation of valid price level."""
        level = PriceLevel(
            price="50000.12345678",
            quantity="1.5",
            count=5
        )
        
        assert level.price == Decimal("50000.12345678")
        assert level.quantity == Decimal("1.5")
        assert level.count == 5
    
    def test_price_level_validation(self):
        """Test price level validation rules."""
        # Test positive price requirement
        with pytest.raises(ValueError, match="greater than 0"):
            PriceLevel(price="0", quantity="1")
        
        with pytest.raises(ValueError, match="greater than 0"):
            PriceLevel(price="-100", quantity="1")
    
    def test_decimal_conversion(self):
        """Test conversion from various formats to Decimal."""
        # String input
        level1 = PriceLevel(price="50000.123", quantity="1.5")
        assert level1.price == Decimal("50000.12300000")
        
        # Float input (should work but not recommended)
        level2 = PriceLevel(price=50000.123, quantity=1.5)
        assert isinstance(level2.price, Decimal)
        assert isinstance(level2.quantity, Decimal)
        
        # Scientific notation
        level3 = PriceLevel(price="5e4", quantity="1.5e-1")
        assert level3.price == Decimal("50000.00000000")
        assert level3.quantity == Decimal("0.15000000")
    
    def test_sorting_and_comparison(self):
        """Test price level sorting and comparison."""
        level1 = PriceLevel(price="50000", quantity="1")
        level2 = PriceLevel(price="51000", quantity="2")
        level3 = PriceLevel(price="50000", quantity="3")
        
        assert level1 < level2
        assert level1 == level3  # Equal by price only
        
        levels = [level2, level1, level3]
        sorted_levels = sorted(levels)
        assert sorted_levels[0].price == Decimal("50000")
        assert sorted_levels[2].price == Decimal("51000")


class TestMarketData:
    """Test MarketData base class functionality."""
    
    def test_valid_market_data(self):
        """Test creation of valid market data."""
        now = datetime.now(timezone.utc)
        data = MarketData(
            symbol="BTC",
            timestamp=now
        )
        
        assert data.symbol == "BTC"
        assert data.timestamp == now
        assert data.time_ms == int(now.timestamp() * 1000)
    
    def test_symbol_validation(self):
        """Test symbol validation rules."""
        now = datetime.now(timezone.utc)
        
        # Valid symbols
        MarketData(symbol="BTC", timestamp=now)
        MarketData(symbol="ETH", timestamp=now)
        MarketData(symbol="BTC1", timestamp=now)
        MarketData(symbol="DOGE", timestamp=now)
        
        # Invalid symbols
        with pytest.raises(ValueError, match="pattern"):
            MarketData(symbol="btc", timestamp=now)
        
        with pytest.raises(ValueError, match="pattern"):
            MarketData(symbol="BTC-USD", timestamp=now)
        
        with pytest.raises(ValueError, match="at least 1 character"):
            MarketData(symbol="", timestamp=now)
        
        with pytest.raises(ValueError, match="at most 20 character"):
            MarketData(symbol="A" * 21, timestamp=now)
    
    def test_timestamp_conversion(self):
        """Test timestamp format conversions."""
        # ISO string format
        data1 = MarketData(
            symbol="BTC",
            timestamp="2024-01-15T10:30:00Z"
        )
        assert data1.timestamp.tzinfo == timezone.utc
        
        # ISO string with timezone
        data2 = MarketData(
            symbol="BTC", 
            timestamp="2024-01-15T10:30:00+00:00"
        )
        assert data2.timestamp.tzinfo == timezone.utc
        
        # Milliseconds timestamp
        timestamp_ms = 1705316200000  # 2024-01-15T10:30:00Z
        data3 = MarketData(
            symbol="BTC",
            timestamp=timestamp_ms
        )
        assert data3.timestamp.year == 2024
        assert data3.timestamp.month == 1
        assert data3.timestamp.day == 15
        
        # Datetime object
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        data4 = MarketData(symbol="BTC", timestamp=dt)
        assert data4.timestamp == dt
    
    def test_time_ms_auto_generation(self):
        """Test automatic time_ms generation from timestamp."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        data = MarketData(symbol="BTC", timestamp=dt)
        
        expected_ms = int(dt.timestamp() * 1000)
        assert data.time_ms == expected_ms
    
    def test_explicit_time_ms(self):
        """Test explicit time_ms override."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        custom_ms = 1234567890000
        
        data = MarketData(
            symbol="BTC",
            timestamp=dt,
            time_ms=custom_ms
        )
        
        assert data.time_ms == custom_ms


class TestCandlestickData:
    """Test CandlestickData model validation and functionality."""
    
    def test_valid_candlestick(self):
        """Test creation of valid candlestick data."""
        now = datetime.now(timezone.utc)
        candle = CandlestickData(
            symbol="BTC",
            timestamp=now,
            timeframe=Timeframe.ONE_HOUR,
            open="50000.0",
            high="51000.0",
            low="49500.0",
            close="50500.0",
            volume="100.5",
            trade_count=150
        )
        
        assert candle.symbol == "BTC"
        assert candle.timeframe == Timeframe.ONE_HOUR
        assert candle.open == Decimal("50000.00000000")
        assert candle.high == Decimal("51000.00000000")
        assert candle.low == Decimal("49500.00000000")
        assert candle.close == Decimal("50500.00000000")
        assert candle.volume == Decimal("100.50000000")
        assert candle.trade_count == 150
    
    def test_ohlc_validation(self):
        """Test OHLC price relationship validation."""
        now = datetime.now(timezone.utc)
        
        # Valid OHLC relationships
        CandlestickData(
            symbol="BTC",
            timestamp=now,
            timeframe=Timeframe.ONE_HOUR,
            open="50000",
            high="51000",  # High is highest
            low="49000",   # Low is lowest
            close="50500",
            volume="100"
        )
        
        # Invalid: High < Open
        with pytest.raises(ValueError, match="High price.*must be"):
            CandlestickData(
                symbol="BTC",
                timestamp=now,
                timeframe=Timeframe.ONE_HOUR,
                open="51000",
                high="50000",  # High < Open
                low="49000",
                close="50500",
                volume="100"
            )
        
        # Invalid: High < Close
        with pytest.raises(ValueError, match="High price.*must be"):
            CandlestickData(
                symbol="BTC",
                timestamp=now,
                timeframe=Timeframe.ONE_HOUR,
                open="50000",
                high="50000",  # High < Close
                low="49000",
                close="51000",
                volume="100"
            )
        
        # Invalid: Low > Open
        with pytest.raises(ValueError, match="Low price.*must be"):
            CandlestickData(
                symbol="BTC",
                timestamp=now,
                timeframe=Timeframe.ONE_HOUR,
                open="49000",
                high="51000",
                low="50000",  # Low > Open
                close="50500",
                volume="100"
            )
    
    def test_hyperliquid_conversion(self):
        """Test conversion to/from HyperLiquid format."""
        # Sample HyperLiquid candlestick data
        hl_data = {
            't': 1705316200000,  # 2024-01-15T10:30:00Z
            's': 'BTC',
            'o': '50000.12345678',
            'h': '51000.87654321',
            'l': '49500.11111111',
            'c': '50500.99999999',
            'v': '100.5',
            'n': 150
        }
        
        candle = CandlestickData.from_hyperliquid(hl_data, Timeframe.ONE_HOUR)
        
        assert candle.symbol == "BTC"
        assert candle.timeframe == Timeframe.ONE_HOUR
        assert candle.open == Decimal("50000.12345678")
        assert candle.high == Decimal("51000.87654321")
        assert candle.low == Decimal("49500.11111111")
        assert candle.close == Decimal("50500.99999999")
        assert candle.volume == Decimal("100.5")
        assert candle.trade_count == 150
        
        # Convert back to HyperLiquid format
        converted = candle.to_hyperliquid()
        
        assert converted['t'] == 1705316200000
        assert converted['s'] == 'BTC'
        assert converted['o'] == '50000.12345678'
        assert converted['h'] == '51000.87654321'
        assert converted['l'] == '49500.11111111'
        assert converted['c'] == '50500.99999999'
        assert converted['v'] == '100.50000000'
        assert converted['n'] == 150
    
    def test_candlestick_properties(self):
        """Test calculated candlestick properties."""
        candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open="50000",
            high="51000",
            low="49000",
            close="50500",
            volume="100"
        )
        
        # Body size
        assert candle.body_size == Decimal("500")  # |50500 - 50000|
        
        # Shadows
        assert candle.upper_shadow == Decimal("500")  # 51000 - max(50000, 50500)
        assert candle.lower_shadow == Decimal("1000")  # min(50000, 50500) - 49000
        
        # Direction
        assert candle.is_bullish is True
        assert candle.is_bearish is False
        
        # Test bearish candle
        bear_candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open="50500",
            high="51000", 
            low="49000",
            close="50000",
            volume="100"
        )
        
        assert bear_candle.is_bullish is False
        assert bear_candle.is_bearish is True
    
    def test_doji_detection(self):
        """Test doji candlestick pattern detection."""
        # Perfect doji (open == close)
        doji = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open="50000",
            high="50100",
            low="49900",
            close="50000",
            volume="100"
        )
        assert doji.is_doji is True
        
        # Near doji (small body relative to range)
        near_doji = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open="50000",
            high="51000",
            low="49000",
            close="50010",  # Small body relative to 2000 range
            volume="100"
        )
        assert near_doji.is_doji is True
        
        # Not a doji
        not_doji = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open="50000",
            high="50100",
            low="49900",
            close="50050",  # Large body relative to 200 range
            volume="100"
        )
        assert not_doji.is_doji is False


class TestTicker:
    """Test Ticker model validation and functionality."""
    
    def test_valid_ticker(self):
        """Test creation of valid ticker data."""
        now = datetime.now(timezone.utc)
        ticker = Ticker(
            symbol="BTC",
            timestamp=now,
            last_price="50000.0",
            bid_price="49995.0",
            ask_price="50005.0",
            volume_24h="1000.5",
            price_change_24h="500.0",
            price_change_pct_24h="1.01",
            high_24h="51000.0",
            low_24h="49000.0",
            open_24h="49500.0",
            trade_count_24h=5000
        )
        
        assert ticker.symbol == "BTC"
        assert ticker.last_price == Decimal("50000.0")
        assert ticker.bid_price == Decimal("49995.0")
        assert ticker.ask_price == Decimal("50005.0")
        assert ticker.volume_24h == Decimal("1000.5")
        assert ticker.price_change_24h == Decimal("500.0")
        assert ticker.price_change_pct_24h == Decimal("1.01")
        assert ticker.high_24h == Decimal("51000.0")
        assert ticker.low_24h == Decimal("49000.0")
        assert ticker.trade_count_24h == 5000
    
    def test_bid_ask_validation(self):
        """Test bid/ask price validation."""
        now = datetime.now(timezone.utc)
        
        # Valid spread
        Ticker(
            symbol="BTC",
            timestamp=now,
            last_price="50000",
            bid_price="49995",
            ask_price="50005",
            volume_24h="1000",
            price_change_24h="500",
            price_change_pct_24h="1.0",
            high_24h="51000",
            low_24h="49000"
        )
        
        # Invalid: bid >= ask
        with pytest.raises(ValueError, match="Bid price.*must be less than ask price"):
            Ticker(
                symbol="BTC",
                timestamp=now,
                last_price="50000",
                bid_price="50005",  # bid > ask
                ask_price="50000",
                volume_24h="1000",
                price_change_24h="500",
                price_change_pct_24h="1.0",
                high_24h="51000",
                low_24h="49000"
            )
    
    def test_ticker_calculations(self):
        """Test ticker calculation properties."""
        ticker = Ticker(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            last_price="50000",
            bid_price="49990",
            ask_price="50010",
            volume_24h="1000",
            price_change_24h="500",
            price_change_pct_24h="1.0",
            high_24h="51000",
            low_24h="49000"
        )
        
        # Mid price
        assert ticker.mid_price == Decimal("50000")  # (49990 + 50010) / 2
        
        # Spread
        assert ticker.spread == Decimal("20")  # 50010 - 49990
        
        # Spread percentage
        expected_spread_pct = (Decimal("20") / Decimal("50000")) * 100
        assert ticker.spread_pct == expected_spread_pct
    
    def test_optional_bid_ask(self):
        """Test ticker with optional bid/ask prices."""
        ticker = Ticker(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            last_price="50000",
            volume_24h="1000",
            price_change_24h="500",
            price_change_pct_24h="1.0",
            high_24h="51000",
            low_24h="49000"
        )
        
        assert ticker.bid_price is None
        assert ticker.ask_price is None
        assert ticker.mid_price is None
        assert ticker.spread is None
        assert ticker.spread_pct is None


class TestSymbolInfo:
    """Test SymbolInfo model validation and functionality."""
    
    def test_valid_symbol_info(self):
        """Test creation of valid symbol info."""
        info = SymbolInfo(
            symbol="BTC",
            name="Bitcoin",
            max_leverage="50.0",
            only_cross=False,
            sz_decimals=3,
            price_decimals=2,
            min_order_size="0.001",
            max_order_size="1000.0",
            tick_size="0.01",
            step_size="0.001",
            is_active=True,
            base_asset="BTC",
            quote_asset="USD"
        )
        
        assert info.symbol == "BTC"
        assert info.name == "Bitcoin"
        assert info.max_leverage == Decimal("50.0")
        assert info.only_cross is False
        assert info.sz_decimals == 3
        assert info.price_decimals == 2
        assert info.min_order_size == Decimal("0.001")
        assert info.max_order_size == Decimal("1000.0")
        assert info.tick_size == Decimal("0.01")
        assert info.step_size == Decimal("0.001")
        assert info.base_asset == "BTC"
        assert info.quote_asset == "USD"
    
    def test_order_size_validation(self):
        """Test order size range validation."""
        # Valid range
        SymbolInfo(
            symbol="BTC",
            min_order_size="0.001",
            max_order_size="1000.0"
        )
        
        # Invalid: min >= max
        with pytest.raises(ValueError, match="Min order size.*must be less than max order size"):
            SymbolInfo(
                symbol="BTC",
                min_order_size="1000.0",
                max_order_size="0.001"
            )
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid metadata."""
        hl_data = {
            'name': 'BTC',
            'maxLeverage': 50,
            'onlyCross': False,
            'szDecimals': 3
        }
        
        info = SymbolInfo.from_hyperliquid(hl_data)
        
        assert info.symbol == "BTC"
        assert info.max_leverage == 50
        assert info.only_cross is False
        assert info.sz_decimals == 3
    
    def test_price_validation_method(self):
        """Test price validation according to symbol specs."""
        info = SymbolInfo(
            symbol="BTC",
            tick_size="0.01",
            price_decimals=2
        )
        
        # Valid price
        validated = info.validate_price("50000.12")
        assert validated == Decimal("50000.12")
        
        # Price requiring rounding to tick size
        validated = info.validate_price("50000.123")
        assert validated == Decimal("50000.13")  # Rounded up to next tick
        
        # Price requiring decimal rounding
        info_with_decimals = SymbolInfo(
            symbol="BTC",
            price_decimals=1
        )
        validated = info_with_decimals.validate_price("50000.987")
        assert validated == Decimal("50001.0")
    
    def test_quantity_validation_method(self):
        """Test quantity validation according to symbol specs."""
        info = SymbolInfo(
            symbol="BTC",
            min_order_size="0.001",
            max_order_size="1000.0",
            step_size="0.001",
            sz_decimals=3
        )
        
        # Valid quantity
        validated = info.validate_quantity("0.123")
        assert validated == Decimal("0.123")
        
        # Below minimum
        with pytest.raises(ValueError, match="below minimum"):
            info.validate_quantity("0.0005")
        
        # Above maximum
        with pytest.raises(ValueError, match="above maximum"):
            info.validate_quantity("2000.0")
        
        # Requiring step size rounding
        validated = info.validate_quantity("0.1234")
        assert validated == Decimal("0.124")  # Rounded up to next step
        
        # Requiring decimal places rounding
        validated = info.validate_quantity("0.12345")
        assert validated == Decimal("0.124")


class TestModelIntegration:
    """Test model integration and edge cases."""
    
    def test_json_serialization(self):
        """Test JSON serialization of all models."""
        now = datetime.now(timezone.utc)
        
        # Test CandlestickData
        candle = CandlestickData(
            symbol="BTC",
            timestamp=now,
            timeframe=Timeframe.ONE_HOUR,
            open="50000",
            high="51000",
            low="49000",
            close="50500",
            volume="100"
        )
        
        json_data = candle.model_dump()
        # In Pydantic v2, decimal fields may not be automatically converted to strings
        assert json_data['timeframe'] == "1h"
        
        # Test round-trip conversion
        restored = CandlestickData(**json_data)
        assert restored.open == candle.open
        assert restored.timeframe == candle.timeframe
    
    def test_edge_case_values(self):
        """Test handling of edge case values."""
        now = datetime.now(timezone.utc)
        
        # Very small values
        candle = CandlestickData(
            symbol="SHIB",
            timestamp=now,
            timeframe=Timeframe.ONE_MINUTE,
            open="0.00000001",
            high="0.00000002",
            low="0.00000001",
            close="0.00000002",
            volume="1000000000"
        )
        
        assert candle.open == Decimal("0.00000001")
        assert candle.volume == Decimal("1000000000.00000000")
        
        # Zero volume (allowed)
        zero_vol_candle = CandlestickData(
            symbol="BTC",
            timestamp=now,
            timeframe=Timeframe.ONE_HOUR,
            open="50000",
            high="50000",
            low="50000",
            close="50000",
            volume="0"
        )
        assert zero_vol_candle.volume == Decimal("0")
    
    def test_scientific_notation_handling(self):
        """Test handling of scientific notation in inputs."""
        level = PriceLevel(
            price="5.0e4",  # 50000
            quantity="1.5e-3"  # 0.0015
        )
        
        assert level.price == Decimal("50000.00000000")
        assert level.quantity == Decimal("0.00150000") 