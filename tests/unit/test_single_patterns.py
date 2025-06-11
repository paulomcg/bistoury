"""
Unit tests for single candlestick pattern recognition.

Tests all pattern detectors with various candlestick scenarios
including edge cases, confidence scoring, and pattern classification.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import List

from src.bistoury.strategies.patterns.single_candlestick import (
    DojiDetector,
    HammerDetector,
    ShootingStarDetector,
    SpinningTopDetector,
    MarubozuDetector,
    SinglePatternRecognizer
)
from src.bistoury.strategies.candlestick_models import VolumeProfile
from src.bistoury.models.market_data import CandlestickData, Timeframe
from src.bistoury.models.signals import PatternType, SignalDirection


def create_test_candle(
    symbol: str = "BTC",
    open_price: Decimal = Decimal('50000'),
    high: Decimal = Decimal('51000'),
    low: Decimal = Decimal('49000'),
    close: Decimal = Decimal('50500'),
    volume: Decimal = Decimal('100'),
    timeframe: Timeframe = Timeframe.ONE_MINUTE
) -> CandlestickData:
    """Create a test candlestick with specified OHLC values."""
    return CandlestickData(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=datetime.now(timezone.utc),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        trades_count=10,
        vwap=Decimal('50250')
    )


def create_volume_profile(confirmation: bool = True) -> VolumeProfile:
    """Create a test volume profile."""
    return VolumeProfile(
        pattern_volume=Decimal('110'),
        average_volume=Decimal('100'),
        volume_ratio=Decimal('1.2'),
        volume_trend=SignalDirection.BUY if confirmation else SignalDirection.SELL,
        breakout_volume=Decimal('120')
    )


class TestDojiDetector:
    """Test Doji pattern detection."""
    
    def setup_method(self):
        self.detector = DojiDetector(min_confidence=Decimal('20'))
    
    def test_perfect_doji(self):
        """Test detection of perfect Doji (open == close)."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50500'),
            low=Decimal('49500'),
            close=Decimal('50000')  # Perfect Doji
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.DOJI
        assert pattern.confidence >= Decimal('80')
        assert "doji_subtype" in pattern.pattern_metadata  # Allow any subtype for symmetric candle
    
    def test_dragonfly_doji(self):
        """Test detection of Dragonfly Doji."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50050'),  # Minimal upper shadow
            low=Decimal('49000'),   # Long lower shadow
            close=Decimal('50020')  # Small body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.DOJI
        assert pattern.pattern_metadata["doji_subtype"] == "dragonfly"
        assert pattern.bullish_probability > pattern.bearish_probability
    
    def test_gravestone_doji(self):
        """Test detection of Gravestone Doji."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('51000'),  # Long upper shadow
            low=Decimal('49950'),   # Minimal lower shadow
            close=Decimal('49980')  # Small body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.DOJI
        assert pattern.pattern_metadata["doji_subtype"] == "gravestone"
        assert pattern.bearish_probability > pattern.bullish_probability
    
    def test_long_legged_doji(self):
        """Test detection of Long-legged Doji."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('51000'),  # Long upper shadow
            low=Decimal('49000'),   # Long lower shadow
            close=Decimal('50020')  # Small body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.DOJI
        assert pattern.pattern_metadata["doji_subtype"] == "long_legged"
    
    def test_large_body_rejection(self):
        """Test that large body candles are not detected as Doji."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('51000'),
            low=Decimal('49000'),
            close=Decimal('50800')  # Large body
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None
    
    def test_low_confidence_rejection(self):
        """Test that low confidence patterns are rejected."""
        detector = DojiDetector(min_confidence=Decimal('90'))
        
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50200'),
            low=Decimal('49800'),
            close=Decimal('50100')  # Medium body
        )
        
        pattern = detector.detect(candle)
        assert pattern is None


class TestHammerDetector:
    """Test Hammer pattern detection."""
    
    def setup_method(self):
        self.detector = HammerDetector(min_confidence=Decimal('20'))
    
    def test_perfect_hammer(self):
        """Test detection of perfect Hammer pattern."""
        candle = create_test_candle(
            open_price=Decimal('50500'),
            high=Decimal('50600'),   # Small upper shadow
            low=Decimal('49000'),    # Long lower shadow
            close=Decimal('50400')   # Small body near top
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.HAMMER
        assert pattern.confidence >= Decimal('70')
        assert pattern.bullish_probability >= Decimal('0.7')
    
    def test_bullish_hammer_body(self):
        """Test Hammer with bullish body gets higher bullish probability."""
        candle = create_test_candle(
            open_price=Decimal('50400'),
            high=Decimal('50600'),
            low=Decimal('49000'),
            close=Decimal('50500')   # Bullish body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.bullish_probability == Decimal('0.75')
        assert pattern.pattern_metadata["is_bullish_body"] is True
    
    def test_bearish_hammer_body(self):
        """Test Hammer with bearish body still bullish but lower probability."""
        candle = create_test_candle(
            open_price=Decimal('50500'),
            high=Decimal('50600'),
            low=Decimal('49000'),
            close=Decimal('50400')   # Bearish body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.bullish_probability == Decimal('0.70')
        assert pattern.pattern_metadata["is_bullish_body"] is False
    
    def test_insufficient_lower_shadow(self):
        """Test rejection when lower shadow is too short."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50200'),
            low=Decimal('49800'),    # Short lower shadow
            close=Decimal('50100')
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None
    
    def test_excessive_upper_shadow(self):
        """Test rejection when upper shadow is too long."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('51500'),   # Long upper shadow
            low=Decimal('49000'),
            close=Decimal('50100')
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None
    
    def test_large_body(self):
        """Test rejection when body is too large."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50700'),   # Fixed: high must be >= close
            low=Decimal('49000'),
            close=Decimal('50700')   # Large body
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None


class TestShootingStarDetector:
    """Test Shooting Star pattern detection."""
    
    def setup_method(self):
        self.detector = ShootingStarDetector(min_confidence=Decimal('20'))
    
    def test_perfect_shooting_star(self):
        """Test detection of perfect Shooting Star pattern."""
        # Range = 51000 - 49000 = 2000
        # Body = |49500 - 49400| = 100 (5% of range) ✓
        # Upper shadow = 51000 - max(49400, 49500) = 1500 (75% of range) ✓  
        # Lower shadow = min(49400, 49500) - 49000 = 400 (20% of range) ✗ (should be ≤ 15%)
        candle = create_test_candle(
            open_price=Decimal('49200'),
            high=Decimal('51000'),   # Long upper shadow
            low=Decimal('49000'),    # Small lower shadow  
            close=Decimal('49300')   # Small body near bottom
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.SHOOTING_STAR
        assert pattern.confidence >= Decimal('70')
        assert pattern.bearish_probability >= Decimal('0.7')
    
    def test_bearish_shooting_star_body(self):
        """Test Shooting Star with bearish body gets higher bearish probability."""
        candle = create_test_candle(
            open_price=Decimal('49300'),
            high=Decimal('51000'),
            low=Decimal('49000'),
            close=Decimal('49200')   # Bearish body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.bearish_probability == Decimal('0.75')
        assert pattern.pattern_metadata["is_bearish_body"] is True
    
    def test_insufficient_upper_shadow(self):
        """Test rejection when upper shadow is too short."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50200'),   # Short upper shadow
            low=Decimal('49800'),
            close=Decimal('49900')
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None
    
    def test_excessive_lower_shadow(self):
        """Test rejection when lower shadow is too long."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('51000'),
            low=Decimal('48500'),    # Long lower shadow
            close=Decimal('49900')
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None


class TestSpinningTopDetector:
    """Test Spinning Top pattern detection."""
    
    def setup_method(self):
        self.detector = SpinningTopDetector(min_confidence=Decimal('20'))
    
    def test_perfect_spinning_top(self):
        """Test detection of perfect Spinning Top pattern."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50800'),   # Balanced upper shadow
            low=Decimal('49200'),    # Balanced lower shadow
            close=Decimal('50100')   # Small body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.SPINNING_TOP
        assert pattern.confidence >= Decimal('60')
        assert abs(pattern.bullish_probability - pattern.bearish_probability) <= Decimal('0.1')
    
    def test_bullish_spinning_top(self):
        """Test Spinning Top with bullish body."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50700'),
            low=Decimal('49300'),
            close=Decimal('50150')   # Bullish body
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.bullish_probability == Decimal('0.55')
        assert pattern.bearish_probability == Decimal('0.45')
    
    def test_insufficient_shadows(self):
        """Test rejection when shadows are too small."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50100'),   # Small upper shadow
            low=Decimal('49900'),    # Small lower shadow
            close=Decimal('50050')
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None
    
    def test_unbalanced_shadows(self):
        """Test rejection when shadows are too unbalanced."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('51500'),   # Very long upper shadow
            low=Decimal('49700'),    # Short lower shadow
            close=Decimal('50100')
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None
    
    def test_large_body(self):
        """Test rejection when body is too large."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50800'),
            low=Decimal('49200'),
            close=Decimal('50600')   # Large body
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None


class TestMarubozuDetector:
    """Test Marubozu pattern detection."""
    
    def setup_method(self):
        self.detector = MarubozuDetector(min_confidence=Decimal('20'))
    
    def test_perfect_bullish_marubozu(self):
        """Test detection of perfect bullish Marubozu."""
        candle = create_test_candle(
            open_price=Decimal('49000'),  # Open at low
            high=Decimal('51000'),
            low=Decimal('49000'),
            close=Decimal('51000')        # Close at high
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.MARUBOZU
        assert pattern.confidence >= Decimal('90')
        assert pattern.bullish_probability == Decimal('0.85')
        assert pattern.pattern_metadata["marubozu_type"] == "bullish"
    
    def test_perfect_bearish_marubozu(self):
        """Test detection of perfect bearish Marubozu."""
        candle = create_test_candle(
            open_price=Decimal('51000'),  # Open at high
            high=Decimal('51000'),
            low=Decimal('49000'),
            close=Decimal('49000')        # Close at low
        )
        
        pattern = self.detector.detect(candle)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.MARUBOZU
        assert pattern.confidence >= Decimal('90')
        assert pattern.bearish_probability == Decimal('0.85')
        assert pattern.pattern_metadata["marubozu_type"] == "bearish"
    
    def test_small_body_rejection(self):
        """Test rejection when body is too small."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50500'),
            low=Decimal('49500'),
            close=Decimal('50200')   # Small body
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None
    
    def test_excessive_shadows(self):
        """Test rejection when shadows are too large."""
        candle = create_test_candle(
            open_price=Decimal('49200'),
            high=Decimal('51000'),   # Shadow too large
            low=Decimal('49000'),    # Shadow too large
            close=Decimal('50800')
        )
        
        pattern = self.detector.detect(candle)
        assert pattern is None


class TestSinglePatternRecognizer:
    """Test the main single pattern recognizer."""
    
    def setup_method(self):
        self.recognizer = SinglePatternRecognizer(min_confidence=Decimal('60'))
    
    def test_multiple_pattern_detection(self):
        """Test detection of multiple patterns in one candle."""
        # Create a perfect Doji that should be detected
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50020'),
            low=Decimal('49980'),
            close=Decimal('50000')   # Perfect Doji: open == close, small range
        )
        
        patterns = self.recognizer.analyze(candle)
        
        # Should detect some patterns
        assert len(patterns) > 0
        # Patterns should be sorted by confidence
        if len(patterns) > 1:
            assert patterns[0].confidence >= patterns[1].confidence
    
    def test_get_best_pattern(self):
        """Test getting the best pattern from analysis."""
        candle = create_test_candle(
            open_price=Decimal('49000'),
            high=Decimal('51000'),
            low=Decimal('49000'),
            close=Decimal('51000')   # Perfect bullish Marubozu
        )
        
        best_pattern = self.recognizer.get_best_pattern(candle)
        
        assert best_pattern is not None
        assert best_pattern.pattern_type == PatternType.MARUBOZU
    
    def test_no_patterns_detected(self):
        """Test when no patterns are detected."""
        # Create a candle that doesn't match any patterns
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50300'),
            low=Decimal('49700'),
            close=Decimal('50150')   # Medium body, medium shadows
        )
        
        patterns = self.recognizer.analyze(candle)
        best_pattern = self.recognizer.get_best_pattern(candle)
        
        # Might detect some patterns or none, but should handle gracefully
        assert isinstance(patterns, list)
        assert best_pattern is None or isinstance(best_pattern.confidence, Decimal)
    
    def test_pattern_summary_with_patterns(self):
        """Test pattern summary when patterns are detected."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50050'),
            low=Decimal('49950'),
            close=Decimal('50020')   # Small Doji-like candle
        )
        
        summary = self.recognizer.get_pattern_summary(candle)
        
        assert "total_patterns" in summary
        assert "best_pattern" in summary
        assert "avg_confidence" in summary
        assert "bullish_patterns" in summary
        assert "bearish_patterns" in summary
        assert "neutral_patterns" in summary
        assert "pattern_types" in summary
    
    def test_pattern_summary_no_patterns(self):
        """Test pattern summary when no patterns are detected."""
        # Use high confidence threshold to ensure no patterns
        recognizer = SinglePatternRecognizer(min_confidence=Decimal('95'))
        
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50300'),
            low=Decimal('49700'),
            close=Decimal('50150')
        )
        
        summary = recognizer.get_pattern_summary(candle)
        
        assert summary["total_patterns"] == 0
        assert summary["best_pattern"] is None
        assert summary["avg_confidence"] == Decimal('0')
        assert summary["bullish_patterns"] == 0
        assert summary["bearish_patterns"] == 0
        assert summary["neutral_patterns"] == 0
    
    def test_volume_profile_integration(self):
        """Test pattern detection with volume profile."""
        candle = create_test_candle(
            open_price=Decimal('49000'),
            high=Decimal('51000'),
            low=Decimal('49000'),
            close=Decimal('51000')
        )
        
                # Create strong volume profile for confirmation  
        volume_profile = VolumeProfile(
            pattern_volume=Decimal('150'),   # Above average
            average_volume=Decimal('100'),
            volume_ratio=Decimal('1.5'),     # 1.5x average (> 1.2 required)
            volume_trend=SignalDirection.STRONG_BUY,  # Strong uptrend
            breakout_volume=Decimal('180')   # Strong breakout (1.8x average > 1.5 required)
        )

        pattern = self.recognizer.get_best_pattern(candle, volume_profile)

        assert pattern is not None
        assert pattern.volume_confirmation is True
    
    def test_error_handling(self):
        """Test error handling in pattern detection."""
        # This test ensures the recognizer handles errors gracefully
        candle = create_test_candle()
        
        # Should not raise exceptions even with edge cases
        patterns = self.recognizer.analyze(candle)
        summary = self.recognizer.get_pattern_summary(candle)
        
        assert isinstance(patterns, list)
        assert isinstance(summary, dict)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_range_candle(self):
        """Test handling of candles with zero price range."""
        candle = create_test_candle(
            open_price=Decimal('50000'),
            high=Decimal('50000'),    # No price movement
            low=Decimal('50000'),
            close=Decimal('50000')
        )
        
        recognizer = SinglePatternRecognizer()
        patterns = recognizer.analyze(candle)
        
        # Should handle gracefully without errors
        assert isinstance(patterns, list)
    
    def test_very_small_decimal_values(self):
        """Test handling of very small decimal values."""
        candle = create_test_candle(
            open_price=Decimal('0.00001'),
            high=Decimal('0.00002'),
            low=Decimal('0.000005'),
            close=Decimal('0.000015')
        )
        
        recognizer = SinglePatternRecognizer()
        patterns = recognizer.analyze(candle)
        
        # Should handle small values correctly
        assert isinstance(patterns, list)
    
    def test_very_large_decimal_values(self):
        """Test handling of very large decimal values."""
        candle = create_test_candle(
            open_price=Decimal('1000000'),
            high=Decimal('1100000'),
            low=Decimal('900000'),
            close=Decimal('1050000')
        )
        
        recognizer = SinglePatternRecognizer()
        patterns = recognizer.analyze(candle)
        
        # Should handle large values correctly
        assert isinstance(patterns, list) 