"""
Unit tests for candlestick strategy foundation models.

Tests the extended models for pattern analysis, multi-timeframe confluence,
volume analysis, and strategy configuration.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List

from src.bistoury.strategies.candlestick_models import (
    PatternStrength,
    TimeframePriority,
    PatternConfluence,
    VolumeProfile,
    PatternQuality,
    MultiTimeframePattern,
    StrategyConfiguration
)
from src.bistoury.models.market_data import CandlestickData, Timeframe
from src.bistoury.models.signals import (
    CandlestickPattern,
    PatternType,
    SignalDirection,
    SignalType,
    AnalysisContext,
    TradingSignal
)


class TestPatternStrength:
    """Test PatternStrength enum functionality."""
    
    def test_pattern_strength_values(self):
        """Test pattern strength enum values."""
        assert PatternStrength.VERY_WEAK == "very_weak"
        assert PatternStrength.WEAK == "weak"
        assert PatternStrength.MODERATE == "moderate"
        assert PatternStrength.STRONG == "strong"
        assert PatternStrength.VERY_STRONG == "very_strong"
    
    def test_from_confidence(self):
        """Test conversion from confidence percentage to strength."""
        assert PatternStrength.from_confidence(Decimal('10')) == PatternStrength.VERY_WEAK
        assert PatternStrength.from_confidence(Decimal('30')) == PatternStrength.WEAK
        assert PatternStrength.from_confidence(Decimal('50')) == PatternStrength.MODERATE
        assert PatternStrength.from_confidence(Decimal('70')) == PatternStrength.STRONG
        assert PatternStrength.from_confidence(Decimal('90')) == PatternStrength.VERY_STRONG
        
        # Test boundary conditions
        assert PatternStrength.from_confidence(Decimal('20')) == PatternStrength.WEAK
        assert PatternStrength.from_confidence(Decimal('19.9')) == PatternStrength.VERY_WEAK


class TestTimeframePriority:
    """Test TimeframePriority enum functionality."""
    
    def test_timeframe_priority_values(self):
        """Test timeframe priority values."""
        assert TimeframePriority.ONE_MINUTE == 1
        assert TimeframePriority.FIVE_MINUTES == 2
        assert TimeframePriority.FIFTEEN_MINUTES == 3
        assert TimeframePriority.ONE_HOUR == 2
        assert TimeframePriority.FOUR_HOURS == 1
        assert TimeframePriority.ONE_DAY == 1


class TestPatternConfluence:
    """Test PatternConfluence model functionality."""
    
    def create_sample_pattern(self, pattern_type: PatternType, confidence: Decimal = Decimal('75')) -> CandlestickPattern:
        """Helper to create sample candlestick patterns."""
        candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.FIVE_MINUTES,
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("100")
        )
        
        return CandlestickPattern(
            pattern_id=f"{pattern_type.value}_test",
            symbol="BTC",
            pattern_type=pattern_type,
            candles=[candle],
            timeframe=Timeframe.FIVE_MINUTES,
            confidence=confidence,
            reliability=Decimal("0.8"),
            bullish_probability=Decimal("0.7"),
            bearish_probability=Decimal("0.3"),
            completion_price=Decimal("50050")
        )
    
    def test_valid_confluence(self):
        """Test creation of valid pattern confluence."""
        primary = self.create_sample_pattern(PatternType.HAMMER, Decimal('80'))
        supporting = [
            self.create_sample_pattern(PatternType.DOJI, Decimal('70')),
            self.create_sample_pattern(PatternType.SPINNING_TOP, Decimal('65'))
        ]
        conflicting = [self.create_sample_pattern(PatternType.SHOOTING_STAR, Decimal('60'))]
        
        confluence = PatternConfluence(
            primary_pattern=primary,
            supporting_patterns=supporting,
            conflicting_patterns=conflicting,
            confluence_score=Decimal('75')
        )
        
        assert confluence.primary_pattern.pattern_type == PatternType.HAMMER
        assert len(confluence.supporting_patterns) == 2
        assert len(confluence.conflicting_patterns) == 1
        assert confluence.confluence_score == Decimal('75')
    
    def test_computed_properties(self):
        """Test confluence computed properties."""
        primary = self.create_sample_pattern(PatternType.HAMMER)
        supporting = [
            self.create_sample_pattern(PatternType.DOJI),
            self.create_sample_pattern(PatternType.SPINNING_TOP)
        ]
        conflicting = [self.create_sample_pattern(PatternType.SHOOTING_STAR)]
        
        confluence = PatternConfluence(
            primary_pattern=primary,
            supporting_patterns=supporting,
            conflicting_patterns=conflicting,
            confluence_score=Decimal('70')
        )
        
        # Net support: 2 supporting - 1 conflicting = 1
        assert confluence.net_support == 1
        
        # Confluence strength: 70% = STRONG
        assert confluence.confluence_strength == PatternStrength.STRONG
        
        # Strong confluence: >= 60%
        assert confluence.is_strong_confluence is True
        
        # Test weak confluence
        weak_confluence = PatternConfluence(
            primary_pattern=primary,
            supporting_patterns=[],
            conflicting_patterns=conflicting,
            confluence_score=Decimal('50')
        )
        
        assert weak_confluence.is_strong_confluence is False


class TestVolumeProfile:
    """Test VolumeProfile model functionality."""
    
    def test_valid_volume_profile(self):
        """Test creation of valid volume profile."""
        profile = VolumeProfile(
            pattern_volume=Decimal('1200'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.2'),
            volume_trend=SignalDirection.BUY,
            breakout_volume=Decimal('1800')
        )
        
        assert profile.pattern_volume == Decimal('1200')
        assert profile.average_volume == Decimal('1000')
        assert profile.volume_ratio == Decimal('1.2')
        assert profile.volume_trend == SignalDirection.BUY
        assert profile.breakout_volume == Decimal('1800')
    
    def test_volume_computed_properties(self):
        """Test volume profile computed properties."""
        # Above average volume
        profile = VolumeProfile(
            pattern_volume=Decimal('1200'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.2'),
            volume_trend=SignalDirection.BUY
        )
        
        assert profile.is_above_average is True
        
        # Below average volume
        profile_low = VolumeProfile(
            pattern_volume=Decimal('800'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('0.8'),
            volume_trend=SignalDirection.SELL
        )
        
        assert profile_low.is_above_average is False
    
    def test_volume_confirmation(self):
        """Test volume confirmation logic."""
        # Strong confirmation with breakout
        strong_profile = VolumeProfile(
            pattern_volume=Decimal('1300'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.3'),
            volume_trend=SignalDirection.BUY,
            breakout_volume=Decimal('1600')
        )
        
        assert strong_profile.volume_confirmation is True
        
        # Weak confirmation without breakout
        weak_profile = VolumeProfile(
            pattern_volume=Decimal('1100'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.1'),
            volume_trend=SignalDirection.SELL
        )
        
        assert weak_profile.volume_confirmation is False
        
        # No confirmation with low volume
        no_confirm_profile = VolumeProfile(
            pattern_volume=Decimal('900'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('0.9'),
            volume_trend=SignalDirection.BUY
        )
        
        assert no_confirm_profile.volume_confirmation is False


class TestPatternQuality:
    """Test PatternQuality model functionality."""
    
    def test_valid_quality(self):
        """Test creation of valid pattern quality."""
        quality = PatternQuality(
            technical_score=Decimal('80'),
            volume_score=Decimal('75'),
            context_score=Decimal('70'),
            overall_score=Decimal('75')
        )
        
        assert quality.technical_score == Decimal('80')
        assert quality.volume_score == Decimal('75')
        assert quality.context_score == Decimal('70')
        assert quality.overall_score == Decimal('75')
    
    def test_quality_grading(self):
        """Test pattern quality grading system."""
        # A+ grade
        excellent = PatternQuality(
            technical_score=Decimal('95'),
            volume_score=Decimal('90'),
            context_score=Decimal('92'),
            overall_score=Decimal('92')
        )
        assert excellent.quality_grade == "A+"
        
        # A grade
        very_good = PatternQuality(
            technical_score=Decimal('85'),
            volume_score=Decimal('80'),
            context_score=Decimal('82'),
            overall_score=Decimal('82')
        )
        assert very_good.quality_grade == "A"
        
        # B grade
        good = PatternQuality(
            technical_score=Decimal('75'),
            volume_score=Decimal('70'),
            context_score=Decimal('72'),
            overall_score=Decimal('72')
        )
        assert good.quality_grade == "B"
        
        # C grade
        fair = PatternQuality(
            technical_score=Decimal('65'),
            volume_score=Decimal('60'),
            context_score=Decimal('62'),
            overall_score=Decimal('62')
        )
        assert fair.quality_grade == "C"
        
        # D grade
        poor = PatternQuality(
            technical_score=Decimal('55'),
            volume_score=Decimal('50'),
            context_score=Decimal('52'),
            overall_score=Decimal('52')
        )
        assert poor.quality_grade == "D"
        
        # F grade
        failing = PatternQuality(
            technical_score=Decimal('45'),
            volume_score=Decimal('40'),
            context_score=Decimal('42'),
            overall_score=Decimal('42')
        )
        assert failing.quality_grade == "F"
    
    def test_high_quality_threshold(self):
        """Test high quality threshold detection."""
        # High quality
        high_quality = PatternQuality(
            technical_score=Decimal('75'),
            volume_score=Decimal('70'),
            context_score=Decimal('72'),
            overall_score=Decimal('72')
        )
        assert high_quality.is_high_quality is True
        
        # Low quality
        low_quality = PatternQuality(
            technical_score=Decimal('65'),
            volume_score=Decimal('60'),
            context_score=Decimal('62'),
            overall_score=Decimal('62')
        )
        assert low_quality.is_high_quality is False


class TestStrategyConfiguration:
    """Test StrategyConfiguration model functionality."""
    
    def test_default_configuration(self):
        """Test default strategy configuration."""
        config = StrategyConfiguration()
        
        # Default timeframes
        assert config.timeframes == [Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES]
        assert config.primary_timeframe == Timeframe.FIVE_MINUTES
        
        # Default thresholds
        assert config.min_pattern_confidence == Decimal('60')
        assert config.min_confluence_score == Decimal('50')
        assert config.min_quality_score == Decimal('70')
        
        # Default volume settings
        assert config.volume_lookback_periods == 20
        assert config.min_volume_ratio == Decimal('1.2')
        
        # Default risk management
        assert config.default_stop_loss_pct == Decimal('2.0')
        assert config.default_take_profit_pct == Decimal('4.0')
        assert config.min_risk_reward_ratio == Decimal('1.5')
        
        # Default pattern settings
        assert config.enable_single_patterns is True
        assert config.enable_multi_patterns is True
        assert config.enable_complex_patterns is False
    
    def test_custom_configuration(self):
        """Test custom strategy configuration."""
        config = StrategyConfiguration(
            timeframes=[Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES],
            primary_timeframe=Timeframe.FIFTEEN_MINUTES,
            min_pattern_confidence=Decimal('70'),
            min_confluence_score=Decimal('60'),
            volume_lookback_periods=30,
            default_stop_loss_pct=Decimal('1.5'),
            default_take_profit_pct=Decimal('3.0'),
            enable_complex_patterns=True
        )
        
        assert config.timeframes == [Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES]
        assert config.primary_timeframe == Timeframe.FIFTEEN_MINUTES
        assert config.min_pattern_confidence == Decimal('70')
        assert config.min_confluence_score == Decimal('60')
        assert config.volume_lookback_periods == 30
        assert config.default_stop_loss_pct == Decimal('1.5')
        assert config.default_take_profit_pct == Decimal('3.0')
        assert config.enable_complex_patterns is True
    
    def test_risk_reward_calculation(self):
        """Test risk/reward ratio calculation."""
        config = StrategyConfiguration(
            default_stop_loss_pct=Decimal('2.0'),
            default_take_profit_pct=Decimal('6.0')
        )
        
        # Risk/reward = take_profit / stop_loss = 6.0 / 2.0 = 3.0
        assert config.target_risk_reward == Decimal('3.0')
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration
        valid_config = StrategyConfiguration(
            min_pattern_confidence=Decimal('75'),
            default_stop_loss_pct=Decimal('1.5'),
            volume_lookback_periods=15
        )
        
        assert valid_config.min_pattern_confidence == Decimal('75')
        assert valid_config.default_stop_loss_pct == Decimal('1.5')
        assert valid_config.volume_lookback_periods == 15
        
        # Test boundary values
        boundary_config = StrategyConfiguration(
            min_pattern_confidence=Decimal('0'),      # Minimum
            min_confluence_score=Decimal('100'),      # Maximum
            min_quality_score=Decimal('50'),
            default_stop_loss_pct=Decimal('0.1'),     # Minimum
            default_take_profit_pct=Decimal('20.0'),  # Maximum
            volume_lookback_periods=1                  # Minimum
        )
        
        assert boundary_config.min_pattern_confidence == Decimal('0')
        assert boundary_config.min_confluence_score == Decimal('100')
        assert boundary_config.default_stop_loss_pct == Decimal('0.1')
        assert boundary_config.default_take_profit_pct == Decimal('20.0')
        assert boundary_config.volume_lookback_periods == 1


class TestMultiTimeframePattern:
    """Test MultiTimeframePattern model functionality."""
    
    def create_sample_analysis_context(self) -> AnalysisContext:
        """Helper to create sample analysis context."""
        candle_1m = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.ONE_MINUTE,
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("100")
        )
        
        return AnalysisContext(
            symbol="BTC",
            timeframes={Timeframe.ONE_MINUTE: candle_1m},
            trends={Timeframe.ONE_MINUTE: SignalDirection.BUY},
            volatility={Timeframe.ONE_MINUTE: Decimal("0.02")},
            current_price=Decimal("50050"),
            market_regime="trending",
            support_levels=[Decimal("49000"), Decimal("48500")],
            resistance_levels=[Decimal("50500"), Decimal("51000")]
        )
    
    def create_sample_pattern(self) -> CandlestickPattern:
        """Helper to create sample pattern."""
        candle = CandlestickData(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            timeframe=Timeframe.FIVE_MINUTES,
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("100")
        )
        
        return CandlestickPattern(
            pattern_id="hammer_test",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            candles=[candle],
            timeframe=Timeframe.FIVE_MINUTES,
            confidence=Decimal("80"),
            reliability=Decimal("0.75"),
            bullish_probability=Decimal("0.8"),
            bearish_probability=Decimal("0.2"),
            completion_price=Decimal("50050")
        )
    
    def test_signal_strength_calculation(self):
        """Test signal strength calculation."""
        primary_pattern = self.create_sample_pattern()
        
        confluence = PatternConfluence(
            primary_pattern=primary_pattern,
            supporting_patterns=[],
            conflicting_patterns=[],
            confluence_score=Decimal('70')
        )
        
        volume_profile = VolumeProfile(
            pattern_volume=Decimal('1200'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.2'),
            volume_trend=SignalDirection.BUY
        )
        
        quality = PatternQuality(
            technical_score=Decimal('80'),
            volume_score=Decimal('75'),
            context_score=Decimal('70'),
            overall_score=Decimal('75')
        )
        
        context = self.create_sample_analysis_context()
        
        mt_pattern = MultiTimeframePattern(
            symbol="BTC",
            primary_timeframe=Timeframe.FIVE_MINUTES,
            primary_pattern=primary_pattern,
            confluence=confluence,
            volume_profile=volume_profile,
            quality=quality,
            market_context=context,
            entry_price=Decimal("50050")
        )
        
        # Signal strength should be calculated from quality, confluence, and context
        expected_strength = (
            (Decimal('75') / Decimal('100')) * Decimal('0.4') +  # Pattern component
            (Decimal('70') / Decimal('100')) * Decimal('0.3') +  # Confluence component
            context.trend_strength * Decimal('0.3')              # Context component
        )
        
        assert abs(mt_pattern.signal_strength - expected_strength) < Decimal('0.01')
    
    def test_trading_signal_conversion(self):
        """Test conversion to trading signal."""
        primary_pattern = self.create_sample_pattern()
        
        confluence = PatternConfluence(
            primary_pattern=primary_pattern,
            supporting_patterns=[],
            conflicting_patterns=[],
            confluence_score=Decimal('75')
        )
        
        volume_profile = VolumeProfile(
            pattern_volume=Decimal('1300'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.3'),
            volume_trend=SignalDirection.BUY,
            breakout_volume=Decimal('1600')
        )
        
        quality = PatternQuality(
            technical_score=Decimal('80'),
            volume_score=Decimal('85'),
            context_score=Decimal('75'),
            overall_score=Decimal('80')
        )
        
        context = self.create_sample_analysis_context()
        
        mt_pattern = MultiTimeframePattern(
            symbol="BTC",
            primary_timeframe=Timeframe.FIVE_MINUTES,
            primary_pattern=primary_pattern,
            confluence=confluence,
            volume_profile=volume_profile,
            quality=quality,
            market_context=context,
            entry_price=Decimal("50050"),
            stop_loss=Decimal("49050"),
            take_profit=Decimal("52050"),
            risk_reward_ratio=Decimal("2.0")
        )
        
        signal = mt_pattern.to_trading_signal("test_signal_123")
        
        assert signal.signal_id == "test_signal_123"
        assert signal.symbol == "BTC"
        assert signal.direction == SignalDirection.BUY  # Based on pattern bias
        assert signal.signal_type == SignalType.PATTERN
        assert signal.price == Decimal("50050")
        assert signal.target_price == Decimal("52050")
        assert signal.stop_loss == Decimal("49050")
        assert signal.timeframe == Timeframe.FIVE_MINUTES
        assert signal.source == "multi_timeframe_candlestick_strategy"
        assert "Hammer" in signal.reason
        
        # Check metadata
        assert signal.metadata["pattern_type"] == "hammer"
        assert signal.metadata["quality_score"] == 80.0
        assert signal.metadata["volume_confirmed"] is True
        assert signal.metadata["risk_reward_ratio"] == 2.0 