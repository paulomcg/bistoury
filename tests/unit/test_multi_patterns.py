"""
Unit tests for multi-candlestick pattern recognition.

Tests all pattern detectors with various candlestick scenarios
including edge cases, confidence scoring, and pattern classification.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List

from src.bistoury.strategies.patterns.multi_candlestick import (
    EngulfingDetector,
    HaramiDetector,
    PiercingLineDetector,
    DarkCloudCoverDetector,
    MorningStarDetector,
    EveningStarDetector,
    MultiPatternRecognizer
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
    timeframe: Timeframe = Timeframe.ONE_MINUTE,
    timestamp_offset_seconds: int = 0
) -> CandlestickData:
    """Create a test candlestick with specified OHLC values."""
    base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    timestamp = base_time + timedelta(seconds=timestamp_offset_seconds)
    
    return CandlestickData(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
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
        pattern_volume=Decimal('150'),
        average_volume=Decimal('100'),
        volume_ratio=Decimal('1.5'),
        volume_trend=SignalDirection.STRONG_BUY if confirmation else SignalDirection.SELL,
        breakout_volume=Decimal('180')
    )


class TestEngulfingDetector:
    """Test EngulfingDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        return EngulfingDetector()
    
    def test_bullish_engulfing_detection(self, detector):
        """Test detection of bullish engulfing pattern."""
        # Small bearish candle followed by large bullish candle
        candles = [
            create_test_candle(
                open_price=Decimal('50500'),
                high=Decimal('50600'),
                low=Decimal('50400'),
                close=Decimal('50450'),  # Small bearish body
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50300'),  # Opens lower
                high=Decimal('50700'),
                low=Decimal('50250'),
                close=Decimal('50650'),  # Large bullish body that engulfs first
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.ENGULFING
        assert pattern.confidence >= Decimal('60')
        assert pattern.bullish_probability > pattern.bearish_probability
        assert pattern.pattern_metadata["engulfing_type"] == "bullish"
        assert pattern.pattern_metadata["engulfing_ratio"] > 1.5
    
    def test_bearish_engulfing_detection(self, detector):
        """Test detection of bearish engulfing pattern."""
        # Small bullish candle followed by large bearish candle
        candles = [
            create_test_candle(
                open_price=Decimal('50400'),
                high=Decimal('50600'),
                low=Decimal('50350'),
                close=Decimal('50550'),  # Small bullish body
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50700'),  # Opens higher
                high=Decimal('50750'),
                low=Decimal('50200'),
                close=Decimal('50250'),  # Large bearish body that engulfs first
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.ENGULFING
        assert pattern.bearish_probability > pattern.bullish_probability
        assert pattern.pattern_metadata["engulfing_type"] == "bearish"
    
    def test_insufficient_engulfing(self, detector):
        """Test rejection when engulfing is insufficient."""
        # Second candle doesn't fully engulf first
        candles = [
            create_test_candle(
                open_price=Decimal('50000'),
                close=Decimal('50500'),  # Bullish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50400'),  # Doesn't engulf bottom
                close=Decimal('50100'),   # Doesn't engulf top
                high=Decimal('50600'),
                low=Decimal('50050'),
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None
    
    def test_same_direction_rejection(self, detector):
        """Test rejection when both candles have same direction."""
        candles = [
            create_test_candle(
                open_price=Decimal('50000'),
                close=Decimal('50500'),  # Bullish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49500'),
                close=Decimal('51000'),  # Also bullish
                high=Decimal('51100'),
                low=Decimal('49400'),
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None


class TestHaramiDetector:
    """Test HaramiDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        return HaramiDetector()
    
    def test_bullish_harami_detection(self, detector):
        """Test detection of bullish harami pattern."""
        # Large bearish candle followed by small contained candle
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                high=Decimal('51100'),
                low=Decimal('49500'),
                close=Decimal('49800'),  # Large bearish body
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50200'),
                high=Decimal('50400'),
                low=Decimal('50100'),
                close=Decimal('50300'),  # Small contained body
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.HARAMI
        assert pattern.confidence >= Decimal('60')
        assert pattern.bullish_probability > pattern.bearish_probability  # Reversal of bearish first candle
        assert pattern.pattern_metadata["harami_type"] == "bullish"
        assert pattern.pattern_metadata["containment_ratio"] < 0.6
    
    def test_bearish_harami_detection(self, detector):
        """Test detection of bearish harami pattern."""
        # Large bullish candle followed by small contained candle
        candles = [
            create_test_candle(
                open_price=Decimal('49800'),
                high=Decimal('51200'),
                low=Decimal('49700'),
                close=Decimal('51000'),  # Large bullish body
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50400'),
                high=Decimal('50600'),
                low=Decimal('50100'),  # Fixed: low must be <= min(open, close)
                close=Decimal('50200'),  # Small contained body
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.bearish_probability > pattern.bullish_probability  # Reversal of bullish first candle
        assert pattern.pattern_metadata["harami_type"] == "bearish"
    
    def test_large_second_candle_rejection(self, detector):
        """Test rejection when second candle is too large."""
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                close=Decimal('49500'),  # Large bearish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50800'),
                close=Decimal('49700'),  # Too large relative to first (>60% containment)
                high=Decimal('50900'),
                low=Decimal('49600'),
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None
    
    def test_not_contained_rejection(self, detector):
        """Test rejection when second candle is not contained."""
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                close=Decimal('50000'),  # Bearish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50200'),
                close=Decimal('51200'),  # Extends beyond first candle's body
                high=Decimal('51300'),
                low=Decimal('50100'),
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None


class TestPiercingLineDetector:
    """Test PiercingLineDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        return PiercingLineDetector(min_confidence=Decimal('40'))  # Lower threshold for testing
    
    def test_piercing_line_detection(self, detector):
        """Test detection of piercing line pattern."""
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                high=Decimal('51200'),
                low=Decimal('50000'),
                close=Decimal('50200'),  # Bearish candle
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49800'),  # Gap down below first candle's low
                high=Decimal('50800'),
                low=Decimal('49700'),
                close=Decimal('50700'),  # Close above midpoint (50600)
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.PIERCING_LINE
        assert pattern.confidence >= Decimal('40')  # Match detector threshold
        assert pattern.bullish_probability > pattern.bearish_probability
        assert pattern.pattern_metadata["pattern_type"] == "piercing_line"
        assert pattern.pattern_metadata["pierce_ratio"] > 0
    
    def test_wrong_directions_rejection(self, detector):
        """Test rejection when candles have wrong directions."""
        candles = [
            create_test_candle(
                open_price=Decimal('50000'),
                close=Decimal('51000'),  # Bullish (should be bearish)
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49500'),
                close=Decimal('50800'),  # Bullish
                high=Decimal('50900'),
                low=Decimal('49400'),
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None
    
    def test_no_gap_down_rejection(self, detector):
        """Test rejection when there's no gap down."""
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                low=Decimal('50000'),
                close=Decimal('50200'),  # Bearish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50100'),  # Opens above first candle's low (no gap)
                close=Decimal('50700'),   # Bullish
                high=Decimal('50800'),
                low=Decimal('50000'),
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None
    
    def test_insufficient_piercing_rejection(self, detector):
        """Test rejection when piercing is insufficient."""
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                low=Decimal('50000'),
                close=Decimal('50200'),  # Bearish, midpoint = 50600
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49500'),  # Gap down
                close=Decimal('50500'),   # Below midpoint (insufficient piercing)
                high=Decimal('50600'),
                low=Decimal('49400'),
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None


class TestDarkCloudCoverDetector:
    """Test DarkCloudCoverDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        return DarkCloudCoverDetector(min_confidence=Decimal('40'))  # Lower threshold for testing
    
    def test_dark_cloud_cover_detection(self, detector):
        """Test detection of dark cloud cover pattern."""
        candles = [
            create_test_candle(
                open_price=Decimal('50200'),
                high=Decimal('51200'),
                low=Decimal('50000'),
                close=Decimal('51000'),  # Bullish candle
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('51300'),  # Gap up above first candle's high
                high=Decimal('51500'),
                low=Decimal('50300'),
                close=Decimal('50500'),  # Close below midpoint (50600)
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.DARK_CLOUD_COVER
        assert pattern.confidence >= Decimal('40')  # Match detector threshold
        assert pattern.bearish_probability > pattern.bullish_probability
        assert pattern.pattern_metadata["pattern_type"] == "dark_cloud_cover"
        assert pattern.pattern_metadata["cover_ratio"] > 0
    
    def test_wrong_directions_rejection(self, detector):
        """Test rejection when candles have wrong directions."""
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                close=Decimal('50000'),  # Bearish (should be bullish)
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('51500'),
                high=Decimal('51600'),  # Fixed: high must be >= max(open, close)
                low=Decimal('50400'),
                close=Decimal('50500'),  # Bearish
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None
    
    def test_no_gap_up_rejection(self, detector):
        """Test rejection when there's no gap up."""
        candles = [
            create_test_candle(
                open_price=Decimal('50200'),
                high=Decimal('51000'),
                close=Decimal('51000'),  # Bullish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50900'),  # Opens below first candle's high (no gap)
                close=Decimal('50300'),   # Bearish
                timestamp_offset_seconds=60
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None


class TestMorningStarDetector:
    """Test MorningStarDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        return MorningStarDetector(min_confidence=Decimal('40'))  # Lower threshold for testing
    
    def test_morning_star_detection(self, detector):
        """Test detection of morning star pattern."""
        candles = [
            # Large bearish candle
            create_test_candle(
                open_price=Decimal('51500'),
                high=Decimal('51700'),
                low=Decimal('50000'),
                close=Decimal('50200'),
                timestamp_offset_seconds=0
            ),
            # Small star candle that gaps down
            create_test_candle(
                open_price=Decimal('49900'),  # Gap down from first close
                high=Decimal('50000'),
                low=Decimal('49700'),
                close=Decimal('49800'),  # Small body
                timestamp_offset_seconds=60
            ),
            # Bullish recovery candle
            create_test_candle(
                open_price=Decimal('49900'),
                high=Decimal('51200'),
                low=Decimal('49800'),
                close=Decimal('51000'),  # Close above midpoint (50850)
                timestamp_offset_seconds=120
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.MORNING_STAR
        assert pattern.confidence >= Decimal('40')  # Match detector threshold
        assert pattern.bullish_probability > pattern.bearish_probability
        assert pattern.pattern_metadata["pattern_type"] == "morning_star"
        assert pattern.pattern_metadata["star_body_ratio"] < 0.3
    
    def test_wrong_first_candle_rejection(self, detector):
        """Test rejection when first candle is not bearish."""
        candles = [
            create_test_candle(
                open_price=Decimal('50000'),
                close=Decimal('51000'),  # Bullish (should be bearish)
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49500'),
                close=Decimal('49600'),
                timestamp_offset_seconds=60
            ),
            create_test_candle(
                open_price=Decimal('49700'),
                close=Decimal('50800'),
                timestamp_offset_seconds=120
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None
    
    def test_large_star_rejection(self, detector):
        """Test rejection when star candle is too large."""
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                close=Decimal('50000'),  # Bearish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49500'),
                close=Decimal('50200'),  # Large star body (>30% of first)
                high=Decimal('50300'),
                low=Decimal('49400'),
                timestamp_offset_seconds=60
            ),
            create_test_candle(
                open_price=Decimal('50000'),
                close=Decimal('50800'),  # Bullish
                timestamp_offset_seconds=120
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None


class TestEveningStarDetector:
    """Test EveningStarDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        return EveningStarDetector(min_confidence=Decimal('40'))  # Lower threshold for testing
    
    def test_evening_star_detection(self, detector):
        """Test detection of evening star pattern."""
        candles = [
            # Large bullish candle
            create_test_candle(
                open_price=Decimal('50200'),
                high=Decimal('51700'),
                low=Decimal('50000'),
                close=Decimal('51500'),
                timestamp_offset_seconds=0
            ),
            # Small star candle that gaps up
            create_test_candle(
                open_price=Decimal('51600'),  # Gap up from first close
                high=Decimal('51800'),
                low=Decimal('51550'),
                close=Decimal('51700'),  # Small body
                timestamp_offset_seconds=60
            ),
            # Bearish decline candle
            create_test_candle(
                open_price=Decimal('51600'),
                high=Decimal('51700'),
                low=Decimal('50200'),
                close=Decimal('50500'),  # Close below midpoint (50850)
                timestamp_offset_seconds=120
            )
        ]
        
        pattern = detector.detect(candles)
        
        assert pattern is not None
        assert pattern.pattern_type == PatternType.EVENING_STAR
        assert pattern.confidence >= Decimal('40')  # Match detector threshold
        assert pattern.bearish_probability > pattern.bullish_probability
        assert pattern.pattern_metadata["pattern_type"] == "evening_star"
        assert pattern.pattern_metadata["star_body_ratio"] < 0.3
    
    def test_wrong_third_candle_rejection(self, detector):
        """Test rejection when third candle is not bearish."""
        candles = [
            create_test_candle(
                open_price=Decimal('50000'),
                close=Decimal('51000'),  # Bullish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('51200'),
                high=Decimal('51400'),  # Fixed: high must be >= max(open, close)
                low=Decimal('51100'),
                close=Decimal('51300'),  # Small star
                timestamp_offset_seconds=60
            ),
            create_test_candle(
                open_price=Decimal('51100'),
                high=Decimal('51900'),  # Fixed: high must be >= max(open, close)
                low=Decimal('51000'),
                close=Decimal('51800'),  # Bullish (should be bearish)
                timestamp_offset_seconds=120
            )
        ]
        
        pattern = detector.detect(candles)
        assert pattern is None


class TestMultiPatternRecognizer:
    """Test MultiPatternRecognizer functionality."""
    
    @pytest.fixture
    def recognizer(self):
        return MultiPatternRecognizer(min_confidence=Decimal('40'))  # Lower threshold for testing
    
    def test_engulfing_pattern_recognition(self, recognizer):
        """Test recognition of engulfing pattern in sequence."""
        candles = [
            create_test_candle(
                open_price=Decimal('50500'),
                close=Decimal('50450'),  # Small bearish
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50300'),
                close=Decimal('50650'),  # Large bullish engulfing
                high=Decimal('50700'),
                low=Decimal('50250'),
                timestamp_offset_seconds=60
            )
        ]
        
        patterns = recognizer.analyze(candles)
        
        assert len(patterns) >= 1
        assert any(p.pattern_type == PatternType.ENGULFING for p in patterns)
    
    def test_morning_star_pattern_recognition(self, recognizer):
        """Test recognition of morning star pattern."""
        candles = [
            create_test_candle(
                open_price=Decimal('51500'),
                close=Decimal('50200'),  # Large bearish
                high=Decimal('51700'),
                low=Decimal('50000'),
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49900'),
                close=Decimal('49800'),  # Small star with gap down
                high=Decimal('50000'),
                low=Decimal('49700'),
                timestamp_offset_seconds=60
            ),
            create_test_candle(
                open_price=Decimal('49900'),
                close=Decimal('51000'),  # Bullish recovery
                high=Decimal('51200'),
                low=Decimal('49800'),
                timestamp_offset_seconds=120
            )
        ]
        
        patterns = recognizer.analyze(candles)
        
        assert len(patterns) >= 1
        assert any(p.pattern_type == PatternType.MORNING_STAR for p in patterns)
    
    def test_multiple_pattern_detection(self, recognizer):
        """Test detection when multiple patterns could apply."""
        # Create sequence that might trigger multiple patterns
        candles = [
            create_test_candle(
                open_price=Decimal('51000'),
                close=Decimal('50000'),  # Large bearish
                high=Decimal('51200'),
                low=Decimal('49800'),
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('49700'),  # Gap down
                close=Decimal('50800'),   # Large bullish (piercing line + engulfing)
                high=Decimal('50900'),
                low=Decimal('49600'),
                timestamp_offset_seconds=60
            )
        ]
        
        patterns = recognizer.analyze(candles)
        
        # Should detect at least one pattern
        assert len(patterns) >= 1
        
        # Check that patterns are sorted by confidence
        if len(patterns) > 1:
            for i in range(1, len(patterns)):
                assert patterns[i-1].confidence >= patterns[i].confidence
    
    def test_best_pattern_selection(self, recognizer):
        """Test selection of best pattern."""
        candles = [
            create_test_candle(
                open_price=Decimal('50500'),
                close=Decimal('50450'),
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50300'),
                close=Decimal('50650'),
                high=Decimal('50700'),
                low=Decimal('50250'),
                timestamp_offset_seconds=60
            )
        ]
        
        best_pattern = recognizer.get_best_pattern(candles)
        patterns = recognizer.analyze(candles)
        
        if patterns:
            assert best_pattern is not None
            assert best_pattern.pattern_type == patterns[0].pattern_type  # Should be same pattern type
            assert best_pattern.confidence == patterns[0].confidence  # Should be highest confidence
        else:
            assert best_pattern is None
    
    def test_pattern_summary(self, recognizer):
        """Test pattern summary generation."""
        candles = [
            create_test_candle(
                open_price=Decimal('50500'),
                close=Decimal('50450'),
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50300'),
                close=Decimal('50650'),
                high=Decimal('50700'),
                low=Decimal('50250'),
                timestamp_offset_seconds=60
            )
        ]
        
        summary = recognizer.get_pattern_summary(candles)
        
        assert "total_patterns" in summary
        assert "best_pattern" in summary
        assert "avg_confidence" in summary
        assert "bullish_patterns" in summary
        assert "bearish_patterns" in summary
        assert "reversal_patterns" in summary
        assert "pattern_types" in summary
    
    def test_volume_profile_integration(self, recognizer):
        """Test pattern recognition with volume profile."""
        candles = [
            create_test_candle(
                open_price=Decimal('50500'),
                close=Decimal('50450'),
                timestamp_offset_seconds=0
            ),
            create_test_candle(
                open_price=Decimal('50300'),
                close=Decimal('50650'),
                high=Decimal('50700'),
                low=Decimal('50250'),
                timestamp_offset_seconds=60
            )
        ]
        
        volume_profile = VolumeProfile(
            pattern_volume=Decimal('150'),
            average_volume=Decimal('100'),
            volume_ratio=Decimal('1.5'),
            volume_trend=SignalDirection.STRONG_BUY,
            breakout_volume=Decimal('180')
        )
        
        patterns = recognizer.analyze(candles, volume_profile)
        
        if patterns:
            # Should have volume confirmation
            pattern = patterns[0]
            assert pattern.volume_confirmation is True
    
    def test_insufficient_candles(self, recognizer):
        """Test behavior with insufficient candles."""
        # Only one candle
        candles = [
            create_test_candle()
        ]
        
        patterns = recognizer.analyze(candles)
        assert len(patterns) == 0
        
        summary = recognizer.get_pattern_summary(candles)
        assert summary["total_patterns"] == 0
        assert summary["best_pattern"] is None
    
    def test_detector_requirements(self, recognizer):
        """Test that detectors have correct candle requirements."""
        # Two-candle detectors
        for detector in recognizer.two_candle_detectors:
            assert detector.get_required_candles() == 2
        
        # Three-candle detectors
        for detector in recognizer.three_candle_detectors:
            assert detector.get_required_candles() == 3
    
    def test_candle_sequence_validation(self, recognizer):
        """Test candle sequence validation."""
        # Create invalid sequence (wrong order)
        candles = [
            create_test_candle(timestamp_offset_seconds=60),  # Later timestamp
            create_test_candle(timestamp_offset_seconds=0)    # Earlier timestamp
        ]
        
        patterns = recognizer.analyze(candles)
        # Should not detect any patterns due to invalid sequence
        assert len(patterns) == 0 