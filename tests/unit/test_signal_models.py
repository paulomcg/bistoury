"""
Unit tests for trading signal models.

Tests signal generation, pattern recognition, analysis context,
signal aggregation, and all computed properties and validation
for comprehensive signal processing functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List

from src.bistoury.models.signals import (
    SignalDirection,
    SignalType,
    ConfidenceLevel,
    PatternType,
    TradingSignal,
    CandlestickPattern,
    AnalysisContext,
    SignalAggregation,
)
from src.bistoury.models.market_data import Timeframe, CandlestickData


class TestSignalDirection:
    """Test SignalDirection enum functionality."""
    
    def test_signal_direction_values(self):
        """Test signal direction enum values."""
        assert SignalDirection.BUY == "buy"
        assert SignalDirection.SELL == "sell"
        assert SignalDirection.HOLD == "hold"
        assert SignalDirection.STRONG_BUY == "strong_buy"
        assert SignalDirection.STRONG_SELL == "strong_sell"


class TestSignalType:
    """Test SignalType enum functionality."""
    
    def test_signal_type_values(self):
        """Test signal type enum values."""
        assert SignalType.TECHNICAL == "technical"
        assert SignalType.FUNDAMENTAL == "fundamental"
        assert SignalType.MOMENTUM == "momentum"
        assert SignalType.PATTERN == "pattern"
        assert SignalType.BREAKOUT == "breakout"


class TestTradingSignal:
    """Test TradingSignal model functionality."""
    
    def test_valid_buy_signal(self):
        """Test creation of valid buy signal."""
        signal = TradingSignal(
            signal_id="signal_123",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("75"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            target_price=Decimal("55000"),
            stop_loss=Decimal("47000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test_strategy",
            reason="Technical indicators show bullish momentum"
        )
        
        assert signal.signal_id == "signal_123"
        assert signal.symbol == "BTC"
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == Decimal("75")
        assert signal.strength == Decimal("0.8")
        assert signal.price == Decimal("50000")
        assert signal.target_price == Decimal("55000")
        assert signal.stop_loss == Decimal("47000")
        assert signal.is_active is True
    
    def test_valid_sell_signal(self):
        """Test creation of valid sell signal."""
        signal = TradingSignal(
            signal_id="signal_456",
            symbol="ETH",
            direction=SignalDirection.SELL,
            signal_type=SignalType.MOMENTUM,
            confidence=Decimal("85"),
            strength=Decimal("0.9"),
            price=Decimal("3000"),
            timeframe=Timeframe.FIFTEEN_MINUTES,
            source="momentum_strategy",
            reason="Momentum indicators showing weakness"
        )
        
        assert signal.direction == SignalDirection.SELL
        assert signal.signal_type == SignalType.MOMENTUM
        assert signal.confidence == Decimal("85")
        assert signal.strength == Decimal("0.9")
    
    def test_computed_properties(self):
        """Test signal computed properties."""
        signal = TradingSignal(
            signal_id="signal_789",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("85"),
            strength=Decimal("0.9"),
            price=Decimal("50000"),
            target_price=Decimal("55000"),
            stop_loss=Decimal("47000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test_strategy",
            reason="Strong bullish signal"
        )
        
        # Confidence level: 85% = HIGH
        assert signal.confidence_level == ConfidenceLevel.VERY_HIGH
        
        # Risk/reward ratio: (55000-50000) / (50000-47000) = 5000/3000 ≈ 1.67
        expected_rr = Decimal("5000") / Decimal("3000")
        assert abs(signal.risk_reward_ratio - expected_rr) < Decimal("0.01")
        
        # Signal score: 0.85 * 0.9 = 0.765
        assert signal.signal_score == Decimal("0.765")
        
        # Age should be very small (just created)
        assert signal.age.total_seconds() < 5
        
        # Not expired (no expiry set)
        assert signal.is_expired is False
    
    def test_confidence_levels(self):
        """Test confidence level categorization."""
        # Very Low: 0-20%
        signal_very_low = TradingSignal(
            signal_id="test",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("15"),
            strength=Decimal("0.5"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test",
            reason="test"
        )
        assert signal_very_low.confidence_level == ConfidenceLevel.VERY_LOW
        
        # Low: 20-40%
        signal_low = TradingSignal(
            signal_id="test",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("35"),
            strength=Decimal("0.5"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test",
            reason="test"
        )
        assert signal_low.confidence_level == ConfidenceLevel.LOW
        
        # Medium: 40-60%
        signal_medium = TradingSignal(
            signal_id="test",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("55"),
            strength=Decimal("0.5"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test",
            reason="test"
        )
        assert signal_medium.confidence_level == ConfidenceLevel.MEDIUM
        
        # High: 60-80%
        signal_high = TradingSignal(
            signal_id="test",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("75"),
            strength=Decimal("0.5"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test",
            reason="test"
        )
        assert signal_high.confidence_level == ConfidenceLevel.HIGH
        
        # Very High: 80-100%
        signal_very_high = TradingSignal(
            signal_id="test",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("90"),
            strength=Decimal("0.5"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test",
            reason="test"
        )
        assert signal_very_high.confidence_level == ConfidenceLevel.VERY_HIGH
    
    def test_expiry_functionality(self):
        """Test signal expiry functionality."""
        # Signal with expiry in future
        future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        signal = TradingSignal(
            signal_id="signal_exp",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("75"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test_strategy",
            reason="Test signal",
            expiry=future_expiry
        )
        
        assert signal.is_expired is False
        assert signal.time_to_expiry is not None
        assert signal.time_to_expiry.total_seconds() > 3500  # Close to 1 hour
        
        # Signal with expiry in past
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        signal_expired = TradingSignal(
            signal_id="signal_exp2",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("75"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test_strategy",
            reason="Test signal",
            expiry=past_expiry
        )
        
        assert signal_expired.is_expired is True
    
    def test_metadata_management(self):
        """Test signal metadata management."""
        signal = TradingSignal(
            signal_id="signal_meta",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("75"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="test_strategy",
            reason="Test signal"
        )
        
        # Add metadata
        signal.add_metadata("indicator_1", "RSI_oversold")
        signal.add_metadata("indicator_2", "MACD_bullish_cross")
        
        assert signal.metadata["indicator_1"] == "RSI_oversold"
        assert signal.metadata["indicator_2"] == "MACD_bullish_cross"
        
        # Update status
        signal.update_status(False, "Market conditions changed")
        
        assert signal.is_active is False
        assert signal.metadata["status_change_reason"] == "Market conditions changed"
        assert "status_change_time" in signal.metadata
    
    def test_create_buy_signal_classmethod(self):
        """Test create_buy_signal class method."""
        signal = TradingSignal.create_buy_signal(
            signal_id="buy_signal_1",
            symbol="BTC",
            price=Decimal("50000"),
            confidence=Decimal("80"),
            strength=Decimal("0.9"),
            source="test_strategy",
            reason="Strong bullish momentum",
            target_price=Decimal("55000"),
            stop_loss=Decimal("47000")
        )
        
        assert signal.direction == SignalDirection.BUY
        assert signal.signal_type == SignalType.TECHNICAL
        assert signal.symbol == "BTC"
        assert signal.confidence == Decimal("80")
        assert signal.strength == Decimal("0.9")
        assert signal.target_price == Decimal("55000")
        assert signal.stop_loss == Decimal("47000")
    
    def test_create_sell_signal_classmethod(self):
        """Test create_sell_signal class method."""
        signal = TradingSignal.create_sell_signal(
            signal_id="sell_signal_1",
            symbol="ETH",
            price=Decimal("3000"),
            confidence=Decimal("70"),
            strength=Decimal("0.7"),
            source="momentum_strategy",
            reason="Bearish momentum detected",
            signal_type=SignalType.MOMENTUM
        )
        
        assert signal.direction == SignalDirection.SELL
        assert signal.signal_type == SignalType.MOMENTUM
        assert signal.symbol == "ETH"
        assert signal.confidence == Decimal("70")
        assert signal.strength == Decimal("0.7")


class TestCandlestickPattern:
    """Test CandlestickPattern model functionality."""
    
    def test_valid_hammer_pattern(self):
        """Test creation of valid hammer pattern."""
        # Create sample candlestick data
        candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("49500"),
            high=Decimal("50200"),
            low=Decimal("49000"),
            close=Decimal("50000"),
            volume=Decimal("150")
        )
        
        pattern = CandlestickPattern(
            pattern_id="hammer_123",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            candles=[candle],
            timeframe=Timeframe.ONE_HOUR,
            confidence=Decimal("85"),
            reliability=Decimal("0.75"),
            bullish_probability=Decimal("0.8"),
            bearish_probability=Decimal("0.2"),
            completion_price=Decimal("50000"),
            volume_confirmation=True
        )
        
        assert pattern.pattern_id == "hammer_123"
        assert pattern.symbol == "BTC"
        assert pattern.pattern_type == PatternType.HAMMER
        assert pattern.confidence == Decimal("85")
        assert pattern.reliability == Decimal("0.75")
        assert pattern.volume_confirmation is True
    
    def test_pattern_computed_properties(self):
        """Test pattern computed properties."""
        # Create sample candlestick data
        candle1 = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("49500"),
            high=Decimal("50500"),
            low=Decimal("49000"),
            close=Decimal("50000"),
            volume=Decimal("100")
        )
        
        candle2 = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49800"),
            close=Decimal("50800"),
            volume=Decimal("200")
        )
        
        pattern = CandlestickPattern(
            pattern_id="engulfing_123",
            symbol="BTC",
            pattern_type=PatternType.ENGULFING,
            candles=[candle1, candle2],
            timeframe=Timeframe.ONE_HOUR,
            confidence=Decimal("90"),
            reliability=Decimal("0.8"),
            bullish_probability=Decimal("0.85"),
            bearish_probability=Decimal("0.15"),
            completion_price=Decimal("50800")
        )
        
        # Pattern strength: 0.9 * 0.8 = 0.72
        assert pattern.pattern_strength == Decimal("0.72")
        
        # Directional bias: 0.85 > 0.15 + 0.2, so BUY
        assert pattern.directional_bias == SignalDirection.BUY
        
        # Pattern size: 2 candles
        assert pattern.pattern_size == 2
        
        # Price range: 51000 - 49000 = 2000
        assert pattern.price_range == Decimal("2000")
        
        # Volume average: (100 + 200) / 2 = 150
        assert pattern.volume_average == Decimal("150")
        
        # Is reversal pattern
        assert pattern.is_reversal_pattern is True
        
        # Is not continuation pattern
        assert pattern.is_continuation_pattern is False
    
    def test_pattern_classification(self):
        """Test pattern classification methods."""
        # Create sample candlestick data for testing
        candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("49500"),
            high=Decimal("50200"),
            low=Decimal("49000"),
            close=Decimal("50000"),
            volume=Decimal("150")
        )
        
        # Reversal pattern
        reversal_pattern = CandlestickPattern(
            pattern_id="hammer_test",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            candles=[candle],
            timeframe=Timeframe.ONE_HOUR,
            confidence=Decimal("80"),
            reliability=Decimal("0.7"),
            bullish_probability=Decimal("0.8"),
            bearish_probability=Decimal("0.2"),
            completion_price=Decimal("50000")
        )
        
        assert reversal_pattern.is_reversal_pattern is True
        assert reversal_pattern.is_continuation_pattern is False
        
        # Continuation pattern
        continuation_pattern = CandlestickPattern(
            pattern_id="flag_test",
            symbol="BTC",
            pattern_type=PatternType.FLAG,
            candles=[candle],
            timeframe=Timeframe.ONE_HOUR,
            confidence=Decimal("80"),
            reliability=Decimal("0.7"),
            bullish_probability=Decimal("0.6"),
            bearish_probability=Decimal("0.4"),
            completion_price=Decimal("50000")
        )
        
        assert continuation_pattern.is_continuation_pattern is True
        assert continuation_pattern.is_reversal_pattern is False
    
    def test_to_trading_signal_conversion(self):
        """Test pattern to trading signal conversion."""
        # Create sample candlestick
        candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("49500"),
            high=Decimal("50200"),
            low=Decimal("49000"),
            close=Decimal("50000"),
            volume=Decimal("150")
        )
        
        pattern = CandlestickPattern(
            pattern_id="hammer_signal",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            candles=[candle],
            timeframe=Timeframe.ONE_HOUR,
            confidence=Decimal("85"),
            reliability=Decimal("0.8"),
            bullish_probability=Decimal("0.9"),
            bearish_probability=Decimal("0.1"),
            completion_price=Decimal("50000"),
            volume_confirmation=True
        )
        
        signal = pattern.to_trading_signal("signal_from_pattern")
        
        assert signal.signal_id == "signal_from_pattern"
        assert signal.symbol == "BTC"
        assert signal.direction == SignalDirection.BUY
        assert signal.signal_type == SignalType.PATTERN
        assert signal.confidence == Decimal("68")  # 85 * 0.8
        assert signal.strength == Decimal("0.68")  # 0.85 * 0.8
        assert signal.price == Decimal("50000")
        assert signal.timeframe == Timeframe.ONE_HOUR
        assert "Hammer" in signal.reason
        assert signal.metadata["pattern_type"] == "hammer"
        assert signal.metadata["volume_confirmation"] is True


class TestAnalysisContext:
    """Test AnalysisContext model functionality."""
    
    def test_valid_analysis_context(self):
        """Test creation of valid analysis context."""
        # Create sample candlestick data for different timeframes
        candle_1h = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("50000"),
            high=Decimal("50500"),
            low=Decimal("49500"),
            close=Decimal("50200"),
            volume=Decimal("100")
        )
        
        candle_4h = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.FOUR_HOURS,
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50200"),
            volume=Decimal("500")
        )
        
        context = AnalysisContext(
            symbol="BTC",
            timeframes={
                Timeframe.ONE_HOUR: candle_1h,
                Timeframe.FOUR_HOURS: candle_4h
            },
            trends={
                Timeframe.ONE_HOUR: SignalDirection.BUY,
                Timeframe.FOUR_HOURS: SignalDirection.BUY
            },
            volatility={
                Timeframe.ONE_HOUR: Decimal("0.02"),
                Timeframe.FOUR_HOURS: Decimal("0.05")
            },
            support_levels=[Decimal("49000"), Decimal("48000")],
            resistance_levels=[Decimal("51000"), Decimal("52000")],
            market_regime="trending",
            sentiment_score=Decimal("0.3"),
            liquidity_score=Decimal("0.8")
        )
        
        assert context.symbol == "BTC"
        assert len(context.timeframes) == 2
        assert context.market_regime == "trending"
        assert context.sentiment_score == Decimal("0.3")
        assert context.liquidity_score == Decimal("0.8")
    
    def test_context_computed_properties(self):
        """Test analysis context computed properties."""
        # Create context with mixed trends
        candle_1h = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("50000"),
            high=Decimal("50500"),
            low=Decimal("49500"),
            close=Decimal("50200"),
            volume=Decimal("100")
        )
        
        candle_4h = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.FOUR_HOURS,
            open=Decimal("51000"),
            high=Decimal("51500"),
            low=Decimal("50000"),
            close=Decimal("50200"),
            volume=Decimal("500")
        )
        
        context = AnalysisContext(
            symbol="BTC",
            timeframes={
                Timeframe.ONE_HOUR: candle_1h,
                Timeframe.FOUR_HOURS: candle_4h
            },
            trends={
                Timeframe.ONE_HOUR: SignalDirection.BUY,
                Timeframe.FOUR_HOURS: SignalDirection.SELL
            },
            volatility={
                Timeframe.ONE_HOUR: Decimal("0.02"),
                Timeframe.FOUR_HOURS: Decimal("0.05")
            },
            support_levels=[Decimal("49000"), Decimal("48000")],
            resistance_levels=[Decimal("51000"), Decimal("52000")]
        )
        
        # Current price from shortest timeframe (1h)
        assert context.current_price == Decimal("50200")
        
        # Trend alignment: 1 bullish out of 2 = 0.5
        assert context.trend_alignment == Decimal("0.5")
        
        # Average volatility: (0.02 + 0.05) / 2 = 0.035
        assert context.average_volatility == Decimal("0.035")
        
        # Trend strength: 0.5 - abs(0.5 - 0.5) = 0.5 - 0 = 0.5
        assert context.trend_strength == Decimal("0.5")  # Mixed trend gives 0.5 strength
        
        # Nearest support: 49000 (below current price 50200)
        assert context.nearest_support == Decimal("49000")
        
        # Nearest resistance: 51000 (above current price 50200)
        assert context.nearest_resistance == Decimal("51000")
    
    def test_market_environment_checks(self):
        """Test market environment classification methods."""
        # Bullish environment (alignment > 0.6)
        bullish_context = AnalysisContext(
            symbol="BTC",
            timeframes={},
            trends={
                Timeframe.ONE_HOUR: SignalDirection.BUY,
                Timeframe.FOUR_HOURS: SignalDirection.BUY,
                Timeframe.ONE_DAY: SignalDirection.BUY
            }
        )
        
        assert bullish_context.trend_alignment == Decimal("1")
        assert bullish_context.is_bullish_environment() is True
        assert bullish_context.is_bearish_environment() is False
        
        # Bearish environment (alignment < 0.4)
        bearish_context = AnalysisContext(
            symbol="BTC",
            timeframes={},
            trends={
                Timeframe.ONE_HOUR: SignalDirection.SELL,
                Timeframe.FOUR_HOURS: SignalDirection.SELL,
                Timeframe.ONE_DAY: SignalDirection.SELL
            }
        )
        
        assert bearish_context.trend_alignment == Decimal("0")
        assert bearish_context.is_bearish_environment() is True
        assert bearish_context.is_bullish_environment() is False
    
    def test_market_regime_checks(self):
        """Test market regime classification methods."""
        # Trending market
        trending_context = AnalysisContext(
            symbol="BTC",
            timeframes={},
            trends={
                Timeframe.ONE_HOUR: SignalDirection.BUY,
                Timeframe.FOUR_HOURS: SignalDirection.BUY,
                Timeframe.ONE_DAY: SignalDirection.BUY
            },
            market_regime="trending"
        )
        
        assert trending_context.trend_strength == Decimal("1")
        assert trending_context.is_trending_market() is True
        assert trending_context.is_ranging_market() is False
        
        # Ranging market
        ranging_context = AnalysisContext(
            symbol="BTC",
            timeframes={},
            market_regime="ranging"
        )
        
        assert ranging_context.is_ranging_market() is True
        assert ranging_context.is_trending_market() is False
    
    def test_timeframe_management(self):
        """Test timeframe data management."""
        context = AnalysisContext(
            symbol="BTC",
            timeframes={}
        )
        
        # Add timeframe data
        candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_HOUR,
            open=Decimal("50000"),
            high=Decimal("50500"),
            low=Decimal("49500"),
            close=Decimal("50200"),
            volume=Decimal("100")
        )
        
        context.add_timeframe(
            timeframe=Timeframe.ONE_HOUR,
            candle=candle,
            trend=SignalDirection.BUY,
            volatility=Decimal("0.02")
        )
        
        assert len(context.timeframes) == 1
        assert context.trends[Timeframe.ONE_HOUR] == SignalDirection.BUY
        assert context.volatility[Timeframe.ONE_HOUR] == Decimal("0.02")
        
        # Get timeframe data
        retrieved_candle = context.get_timeframe_data(Timeframe.ONE_HOUR)
        assert retrieved_candle is not None
        assert retrieved_candle.close == Decimal("50200")


class TestSignalAggregation:
    """Test SignalAggregation model functionality."""
    
    def test_valid_signal_aggregation(self):
        """Test creation of valid signal aggregation."""
        # Create sample signals
        signal1 = TradingSignal(
            signal_id="signal_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("80"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="rsi_strategy",
            reason="RSI oversold"
        )
        
        signal2 = TradingSignal(
            signal_id="signal_2",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.MOMENTUM,
            confidence=Decimal("70"),
            strength=Decimal("0.7"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="macd_strategy",
            reason="MACD bullish cross"
        )
        
        aggregation = SignalAggregation(
            aggregation_id="agg_123",
            symbol="BTC",
            signals=[signal1, signal2],
            weights={
                "rsi_strategy": Decimal("0.6"),
                "macd_strategy": Decimal("0.4")
            },
            min_confidence_threshold=Decimal("50"),
            conflict_resolution="weighted_average"
        )
        
        assert aggregation.aggregation_id == "agg_123"
        assert aggregation.symbol == "BTC"
        assert len(aggregation.signals) == 2
        assert aggregation.weights["rsi_strategy"] == Decimal("0.6")
        assert aggregation.conflict_resolution == "weighted_average"
    
    def test_signal_filtering(self):
        """Test signal filtering by status and confidence."""
        # Create signals with different confidence levels and statuses
        signal1 = TradingSignal(
            signal_id="signal_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("80"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_1",
            reason="Strong signal",
            is_active=True
        )
        
        signal2 = TradingSignal(
            signal_id="signal_2",
            symbol="BTC",
            direction=SignalDirection.SELL,
            signal_type=SignalType.MOMENTUM,
            confidence=Decimal("20"),  # Below threshold
            strength=Decimal("0.2"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_2",
            reason="Weak signal",
            is_active=True
        )
        
        signal3 = TradingSignal(
            signal_id="signal_3",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("70"),
            strength=Decimal("0.7"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_3",
            reason="Good signal",
            is_active=False  # Inactive
        )
        
        aggregation = SignalAggregation(
            aggregation_id="agg_filter",
            symbol="BTC",
            signals=[signal1, signal2, signal3],
            min_confidence_threshold=Decimal("50")
        )
        
        # Active signals: only signal1 (signal2 below threshold, signal3 inactive)
        assert len(aggregation.active_signals) == 1
        assert aggregation.active_signals[0].signal_id == "signal_1"
        
        # Bullish signals: signal1
        assert len(aggregation.bullish_signals) == 1
        
        # Bearish signals: none (signal2 filtered out)
        assert len(aggregation.bearish_signals) == 0
        
        # Neutral signals: none
        assert len(aggregation.neutral_signals) == 0
    
    def test_consensus_calculation(self):
        """Test consensus direction and confidence calculation."""
        # Create signals with mixed directions but weighted towards buy
        signal1 = TradingSignal(
            signal_id="signal_1",
            symbol="BTC",
            direction=SignalDirection.STRONG_BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("90"),
            strength=Decimal("0.9"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_1",
            reason="Very strong buy signal"
        )
        
        signal2 = TradingSignal(
            signal_id="signal_2",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.MOMENTUM,
            confidence=Decimal("70"),
            strength=Decimal("0.7"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_2",
            reason="Moderate buy signal"
        )
        
        signal3 = TradingSignal(
            signal_id="signal_3",
            symbol="BTC",
            direction=SignalDirection.SELL,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("60"),
            strength=Decimal("0.6"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_3",
            reason="Weak sell signal"
        )
        
        aggregation = SignalAggregation(
            aggregation_id="agg_consensus",
            symbol="BTC",
            signals=[signal1, signal2, signal3],
            weights={
                "strategy_1": Decimal("0.5"),
                "strategy_2": Decimal("0.3"),
                "strategy_3": Decimal("0.2")
            }
        )
        
        # Should be bullish consensus due to strong buy signals with higher weights
        assert aggregation.consensus_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        
        # Consensus confidence should be weighted average
        assert aggregation.consensus_confidence > Decimal("70")
        
        # Consensus strength should reflect agreement level
        assert aggregation.consensus_strength > Decimal("0.5")
    
    def test_signal_management(self):
        """Test signal addition and management."""
        aggregation = SignalAggregation(
            aggregation_id="agg_mgmt",
            symbol="BTC",
            signals=[]
        )
        
        # Add signal with weight
        signal = TradingSignal(
            signal_id="signal_new",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("80"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="new_strategy",
            reason="New signal"
        )
        
        aggregation.add_signal(signal, Decimal("0.7"))
        
        assert len(aggregation.signals) == 1
        assert aggregation.weights["new_strategy"] == Decimal("0.7")
        
        # Set weight for source
        aggregation.set_weight("new_strategy", Decimal("0.8"))
        assert aggregation.weights["new_strategy"] == Decimal("0.8")
        
        # Get strongest signal
        strongest = aggregation.get_strongest_signal()
        assert strongest is not None
        assert strongest.signal_id == "signal_new"
    
    def test_aggregated_signal_creation(self):
        """Test creation of aggregated signal."""
        # Create multiple signals
        signal1 = TradingSignal(
            signal_id="signal_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("80"),
            strength=Decimal("0.8"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_1",
            reason="Signal 1"
        )
        
        signal2 = TradingSignal(
            signal_id="signal_2",
            symbol="BTC",
            direction=SignalDirection.BUY,
            signal_type=SignalType.MOMENTUM,
            confidence=Decimal("70"),
            strength=Decimal("0.7"),
            price=Decimal("50000"),
            timeframe=Timeframe.ONE_HOUR,
            source="strategy_2",
            reason="Signal 2"
        )
        
        aggregation = SignalAggregation(
            aggregation_id="agg_final",
            symbol="BTC",
            signals=[signal1, signal2]
        )
        
        # Create aggregated signal
        final_signal = aggregation.create_aggregated_signal("final_signal")
        
        assert final_signal.signal_id == "final_signal"
        assert final_signal.symbol == "BTC"
        assert final_signal.direction == aggregation.consensus_direction
        assert final_signal.confidence == aggregation.consensus_confidence
        assert final_signal.strength == aggregation.consensus_strength
        assert final_signal.source == "signal_aggregation"
        assert final_signal.metadata["aggregation_id"] == "agg_final"
        assert final_signal.metadata["signal_count"] == 2
        assert "strategy_1" in final_signal.metadata["signal_sources"]
        assert "strategy_2" in final_signal.metadata["signal_sources"]
    
    def test_signal_type_counting(self):
        """Test signal counting by type."""
        signals = [
            TradingSignal(
                signal_id="tech_1",
                symbol="BTC",
                direction=SignalDirection.BUY,
                signal_type=SignalType.TECHNICAL,
                confidence=Decimal("80"),
                strength=Decimal("0.8"),
                price=Decimal("50000"),
                timeframe=Timeframe.ONE_HOUR,
                source="tech_strategy",
                reason="Technical signal"
            ),
            TradingSignal(
                signal_id="mom_1",
                symbol="BTC",
                direction=SignalDirection.BUY,
                signal_type=SignalType.MOMENTUM,
                confidence=Decimal("70"),
                strength=Decimal("0.7"),
                price=Decimal("50000"),
                timeframe=Timeframe.ONE_HOUR,
                source="mom_strategy",
                reason="Momentum signal"
            ),
            TradingSignal(
                signal_id="tech_2",
                symbol="BTC",
                direction=SignalDirection.SELL,
                signal_type=SignalType.TECHNICAL,
                confidence=Decimal("60"),
                strength=Decimal("0.6"),
                price=Decimal("50000"),
                timeframe=Timeframe.ONE_HOUR,
                source="tech_strategy_2",
                reason="Another technical signal"
            )
        ]
        
        aggregation = SignalAggregation(
            aggregation_id="agg_count",
            symbol="BTC",
            signals=signals
        )
        
        type_counts = aggregation.signal_count_by_type
        
        assert type_counts[SignalType.TECHNICAL] == 2
        assert type_counts[SignalType.MOMENTUM] == 1
        assert len(type_counts) == 2 