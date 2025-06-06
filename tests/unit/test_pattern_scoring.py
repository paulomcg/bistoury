"""
Test suite for Pattern Strength and Confidence Scoring System.

This test suite covers:
- Technical scoring validation
- Volume confirmation scoring
- Market context scoring
- Historical performance tracking
- Composite scoring functionality
- PatternScoringEngine operations
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List

from src.bistoury.strategies.pattern_scoring import (
    TechnicalScoring,
    VolumeScoring,
    MarketContextScoring,
    HistoricalPerformance,
    CompositePatternScore,
    PatternScoringEngine,
    PatternOutcome,
    MarketSession,
    VolatilityRegime,
    TrendStrength
)
from src.bistoury.models.signals import (
    CandlestickPattern,
    CandlestickData,
    Timeframe,
    SignalDirection,
    PatternType
)
from src.bistoury.strategies.candlestick_models import PatternStrength, VolumeProfile


class TestTechnicalScoring:
    """Test technical scoring functionality."""
    
    def test_technical_scoring_creation(self):
        """Test technical scoring model creation."""
        scoring = TechnicalScoring(
            body_size_score=Decimal('80'),
            shadow_ratio_score=Decimal('75'),
            price_position_score=Decimal('70'),
            symmetry_score=Decimal('85'),
            textbook_compliance=Decimal('90')
        )
        
        assert scoring.body_size_score == Decimal('80')
        assert scoring.shadow_ratio_score == Decimal('75')
        assert scoring.price_position_score == Decimal('70')
        assert scoring.symmetry_score == Decimal('85')
        assert scoring.textbook_compliance == Decimal('90')
    
    def test_overall_technical_score_calculation(self):
        """Test overall technical score calculation."""
        scoring = TechnicalScoring(
            body_size_score=Decimal('80'),
            shadow_ratio_score=Decimal('60'),
            price_position_score=Decimal('70'),
            symmetry_score=Decimal('90'),
            textbook_compliance=Decimal('75')
        )
        
        expected_average = (80 + 60 + 70 + 90 + 75) / 5
        assert scoring.overall_technical_score == Decimal(str(expected_average))
    
    def test_technical_scoring_validation(self):
        """Test technical scoring validation constraints."""
        # Test valid range
        valid_scoring = TechnicalScoring(
            body_size_score=Decimal('0'),
            shadow_ratio_score=Decimal('100'),
            price_position_score=Decimal('50'),
            symmetry_score=Decimal('25'),
            textbook_compliance=Decimal('75')
        )
        assert valid_scoring.body_size_score == Decimal('0')
        assert valid_scoring.shadow_ratio_score == Decimal('100')
        
        # Test invalid range
        with pytest.raises(ValueError):
            TechnicalScoring(
                body_size_score=Decimal('-10'),  # Invalid: below 0
                shadow_ratio_score=Decimal('50'),
                price_position_score=Decimal('50'),
                symmetry_score=Decimal('50'),
                textbook_compliance=Decimal('50')
            )
        
        with pytest.raises(ValueError):
            TechnicalScoring(
                body_size_score=Decimal('50'),
                shadow_ratio_score=Decimal('150'),  # Invalid: above 100
                price_position_score=Decimal('50'),
                symmetry_score=Decimal('50'),
                textbook_compliance=Decimal('50')
            )


class TestVolumeScoring:
    """Test volume scoring functionality."""
    
    def test_volume_scoring_creation(self):
        """Test volume scoring model creation."""
        scoring = VolumeScoring(
            volume_spike_score=Decimal('85'),
            volume_trend_score=Decimal('70'),
            breakout_volume_score=Decimal('90'),
            relative_volume_score=Decimal('75')
        )
        
        assert scoring.volume_spike_score == Decimal('85')
        assert scoring.volume_trend_score == Decimal('70')
        assert scoring.breakout_volume_score == Decimal('90')
        assert scoring.relative_volume_score == Decimal('75')
    
    def test_overall_volume_score_calculation(self):
        """Test overall volume score calculation."""
        scoring = VolumeScoring(
            volume_spike_score=Decimal('80'),
            volume_trend_score=Decimal('60'),
            breakout_volume_score=Decimal('90'),
            relative_volume_score=Decimal('70')
        )
        
        expected_average = (80 + 60 + 90 + 70) / 4
        assert scoring.overall_volume_score == Decimal(str(expected_average))


class TestMarketContextScoring:
    """Test market context scoring functionality."""
    
    def test_context_scoring_creation(self):
        """Test context scoring model creation."""
        scoring = MarketContextScoring(
            trend_alignment_score=Decimal('90'),
            volatility_score=Decimal('75'),
            session_score=Decimal('85'),
            support_resistance_score=Decimal('80'),
            momentum_score=Decimal('70')
        )
        
        assert scoring.trend_alignment_score == Decimal('90')
        assert scoring.volatility_score == Decimal('75')
        assert scoring.session_score == Decimal('85')
        assert scoring.support_resistance_score == Decimal('80')
        assert scoring.momentum_score == Decimal('70')
    
    def test_overall_context_score_calculation(self):
        """Test overall context score calculation."""
        scoring = MarketContextScoring(
            trend_alignment_score=Decimal('80'),
            volatility_score=Decimal('70'),
            session_score=Decimal('90'),
            support_resistance_score=Decimal('60'),
            momentum_score=Decimal('75')
        )
        
        expected_average = (80 + 70 + 90 + 60 + 75) / 5
        assert scoring.overall_context_score == Decimal(str(expected_average))


class TestHistoricalPerformance:
    """Test historical performance tracking."""
    
    def test_historical_performance_creation(self):
        """Test historical performance model creation."""
        performance = HistoricalPerformance(
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            total_occurrences=50,
            successful_trades=32,
            success_rate=Decimal('64'),
            average_profit_loss=Decimal('2.5'),
            average_duration_hours=Decimal('18'),
            reliability_score=Decimal('75')
        )
        
        assert performance.pattern_type == PatternType.HAMMER
        assert performance.timeframe == Timeframe.FIVE_MINUTES
        assert performance.total_occurrences == 50
        assert performance.successful_trades == 32
        assert performance.success_rate == Decimal('64')
        assert performance.average_profit_loss == Decimal('2.5')
        assert performance.average_duration_hours == Decimal('18')
        assert performance.reliability_score == Decimal('75')
    
    def test_confidence_multiplier_calculation(self):
        """Test confidence multiplier calculation."""
        # High success rate with good sample size
        performance = HistoricalPerformance(
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            total_occurrences=100,  # Large sample
            successful_trades=70,
            success_rate=Decimal('70'),  # Above baseline
            reliability_score=Decimal('75')
        )
        
        # Expected: (70/50) * 1.1 = 1.4 * 1.1 = 1.54
        expected_multiplier = (Decimal('70') / Decimal('50')) * Decimal('1.1')
        assert performance.confidence_multiplier == expected_multiplier
        
        # Low success rate with small sample
        performance_low = HistoricalPerformance(
            pattern_type=PatternType.DOJI,
            timeframe=Timeframe.ONE_MINUTE,
            total_occurrences=10,  # Small sample
            successful_trades=3,
            success_rate=Decimal('30'),  # Below baseline
            reliability_score=Decimal('40')
        )
        
        # Expected: (30/50) * 0.9 = 0.6 * 0.9 = 0.54
        expected_multiplier_low = (Decimal('30') / Decimal('50')) * Decimal('0.9')
        assert performance_low.confidence_multiplier == expected_multiplier_low


class TestCompositePatternScore:
    """Test composite pattern scoring functionality."""
    
    @pytest.fixture
    def sample_pattern(self):
        """Create a sample candlestick pattern."""
        candle = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=datetime.now(timezone.utc),
            open=Decimal('50000'),
            high=Decimal('50200'),
            low=Decimal('49800'),
            close=Decimal('50100'),
            volume=Decimal('1000')
        )
        
        return CandlestickPattern(
            pattern_id="test_hammer",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=datetime.now(timezone.utc),
            candles=[candle],
            confidence=Decimal('75'),
            reliability=Decimal('0.8'),
            bullish_probability=Decimal('0.8'),
            bearish_probability=Decimal('0.2'),
            completion_price=Decimal('50100'),
            pattern_metadata={"strength": "strong"}
        )
    
    @pytest.fixture
    def sample_scoring_components(self):
        """Create sample scoring components."""
        technical = TechnicalScoring(
            body_size_score=Decimal('80'),
            shadow_ratio_score=Decimal('75'),
            price_position_score=Decimal('85'),
            symmetry_score=Decimal('70'),
            textbook_compliance=Decimal('75')
        )
        
        volume = VolumeScoring(
            volume_spike_score=Decimal('70'),
            volume_trend_score=Decimal('65'),
            breakout_volume_score=Decimal('80'),
            relative_volume_score=Decimal('75')
        )
        
        context = MarketContextScoring(
            trend_alignment_score=Decimal('90'),
            volatility_score=Decimal('75'),
            session_score=Decimal('85'),
            support_resistance_score=Decimal('80'),
            momentum_score=Decimal('70')
        )
        
        historical = HistoricalPerformance(
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            total_occurrences=50,
            successful_trades=35,
            success_rate=Decimal('70'),
            reliability_score=Decimal('75')
        )
        
        return technical, volume, context, historical
    
    def test_composite_score_creation(self, sample_pattern, sample_scoring_components):
        """Test composite score creation."""
        technical, volume, context, historical = sample_scoring_components
        
        composite = CompositePatternScore(
            pattern=sample_pattern,
            technical_scoring=technical,
            volume_scoring=volume,
            context_scoring=context,
            historical_performance=historical
        )
        
        assert composite.pattern == sample_pattern
        assert composite.technical_scoring == technical
        assert composite.volume_scoring == volume
        assert composite.context_scoring == context
        assert composite.historical_performance == historical
    
    def test_weighted_confidence_score_calculation(self, sample_pattern, sample_scoring_components):
        """Test weighted confidence score calculation."""
        technical, volume, context, historical = sample_scoring_components
        
        composite = CompositePatternScore(
            pattern=sample_pattern,
            technical_scoring=technical,
            volume_scoring=volume,
            context_scoring=context,
            historical_performance=historical
        )
        
        # Calculate expected score
        tech_score = technical.overall_technical_score  # 77
        vol_score = volume.overall_volume_score  # 72.5
        ctx_score = context.overall_context_score  # 80
        hist_score = historical.reliability_score  # 75
        
        weighted = (
            tech_score * Decimal('0.3') +
            vol_score * Decimal('0.25') +
            ctx_score * Decimal('0.25') +
            hist_score * Decimal('0.2')
        )
        
        # Apply historical multiplier
        multiplier = historical.confidence_multiplier
        expected = weighted * multiplier
        
        # Should be capped at 100
        expected = min(Decimal('100'), max(Decimal('0'), expected))
        
        assert composite.weighted_confidence_score == expected
    
    def test_pattern_strength_classification(self, sample_pattern, sample_scoring_components):
        """Test pattern strength classification."""
        technical, volume, context, historical = sample_scoring_components
        
        composite = CompositePatternScore(
            pattern=sample_pattern,
            technical_scoring=technical,
            volume_scoring=volume,
            context_scoring=context,
            historical_performance=historical
        )
        
        # Should classify based on weighted confidence score
        confidence = composite.weighted_confidence_score
        expected_strength = PatternStrength.from_confidence(confidence)
        
        assert composite.pattern_strength == expected_strength
    
    def test_is_tradeable_threshold(self, sample_pattern, sample_scoring_components):
        """Test tradeable threshold logic."""
        technical, volume, context, historical = sample_scoring_components
        
        # High confidence scenario
        high_technical = TechnicalScoring(
            body_size_score=Decimal('90'),
            shadow_ratio_score=Decimal('85'),
            price_position_score=Decimal('95'),
            symmetry_score=Decimal('80'),
            textbook_compliance=Decimal('90')
        )
        
        composite_high = CompositePatternScore(
            pattern=sample_pattern,
            technical_scoring=high_technical,
            volume_scoring=volume,
            context_scoring=context,
            historical_performance=historical
        )
        
        # Should be tradeable if confidence >= 60
        if composite_high.weighted_confidence_score >= Decimal('60'):
            assert composite_high.is_tradeable is True
        else:
            assert composite_high.is_tradeable is False
    
    def test_risk_adjusted_confidence(self, sample_pattern, sample_scoring_components):
        """Test risk-adjusted confidence calculation."""
        technical, volume, context, historical = sample_scoring_components
        
        # High volatility scenario
        high_vol_context = MarketContextScoring(
            trend_alignment_score=Decimal('90'),
            volatility_score=Decimal('90'),  # High volatility
            session_score=Decimal('85'),
            support_resistance_score=Decimal('80'),
            momentum_score=Decimal('70')
        )
        
        composite = CompositePatternScore(
            pattern=sample_pattern,
            technical_scoring=technical,
            volume_scoring=volume,
            context_scoring=high_vol_context,
            historical_performance=historical
        )
        
        base_confidence = composite.weighted_confidence_score
        risk_adjusted = composite.risk_adjusted_confidence
        
        # Should be adjusted down for high volatility
        assert risk_adjusted <= base_confidence
        
        # Misaligned trend scenario
        misaligned_context = MarketContextScoring(
            trend_alignment_score=Decimal('20'),  # Poor alignment
            volatility_score=Decimal('75'),
            session_score=Decimal('85'),
            support_resistance_score=Decimal('80'),
            momentum_score=Decimal('70')
        )
        
        composite_misaligned = CompositePatternScore(
            pattern=sample_pattern,
            technical_scoring=technical,
            volume_scoring=volume,
            context_scoring=misaligned_context,
            historical_performance=historical
        )
        
        base_confidence_misaligned = composite_misaligned.weighted_confidence_score
        risk_adjusted_misaligned = composite_misaligned.risk_adjusted_confidence
        
        # Should be significantly adjusted down for trend misalignment
        assert risk_adjusted_misaligned <= base_confidence_misaligned * Decimal('0.8')


class TestPatternScoringEngine:
    """Test the main pattern scoring engine."""
    
    @pytest.fixture
    def scoring_engine(self):
        """Create a pattern scoring engine."""
        return PatternScoringEngine()
    
    @pytest.fixture
    def sample_candles(self):
        """Create sample candlestick data."""
        base_time = datetime.now(timezone.utc)
        candles = []
        
        for i in range(25):
            timestamp = base_time - timedelta(minutes=5 * i)
            # Create uptrend with some volatility
            base_price = Decimal('50000') + (i * Decimal('50'))
            
            candle = CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=timestamp,
                open=base_price,
                high=base_price + Decimal('100'),
                low=base_price - Decimal('50'),
                close=base_price + Decimal('75'),
                volume=Decimal('1000') + (i * Decimal('100'))
            )
            candles.append(candle)
        
        return list(reversed(candles))  # Chronological order
    
    @pytest.fixture
    def sample_hammer_pattern(self, sample_candles):
        """Create a sample hammer pattern."""
        candle = sample_candles[-1]  # Latest candle
        
        return CandlestickPattern(
            pattern_id="test_hammer",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=candle.timestamp,
            candles=[candle],
            confidence=Decimal('75'),
            reliability=Decimal('0.8'),
            bullish_probability=Decimal('0.8'),
            bearish_probability=Decimal('0.2'),
            completion_price=candle.close,
            pattern_metadata={"strength": "strong"}
        )
    
    def test_score_pattern_basic(self, scoring_engine, sample_hammer_pattern, sample_candles):
        """Test basic pattern scoring."""
        score = scoring_engine.score_pattern(
            pattern=sample_hammer_pattern,
            market_data=sample_candles
        )
        
        assert isinstance(score, CompositePatternScore)
        assert score.pattern == sample_hammer_pattern
        assert isinstance(score.technical_scoring, TechnicalScoring)
        assert isinstance(score.volume_scoring, VolumeScoring)
        assert isinstance(score.context_scoring, MarketContextScoring)
        assert isinstance(score.historical_performance, HistoricalPerformance)
    
    def test_score_pattern_with_volume_profile(self, scoring_engine, sample_hammer_pattern, sample_candles):
        """Test pattern scoring with volume profile."""
        volume_profile = VolumeProfile(
            pattern_volume=Decimal('1500'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.5'),
            volume_trend=SignalDirection.BUY,
            breakout_volume=Decimal('1800')
        )
        
        score = scoring_engine.score_pattern(
            pattern=sample_hammer_pattern,
            market_data=sample_candles,
            volume_profile=volume_profile
        )
        
        # Volume scoring should benefit from volume profile
        assert score.volume_scoring.breakout_volume_score > Decimal('50')
    
    def test_technical_scoring_calculation(self, scoring_engine, sample_hammer_pattern):
        """Test technical scoring calculation."""
        technical_scoring = scoring_engine._calculate_technical_scoring(sample_hammer_pattern)
        
        assert isinstance(technical_scoring, TechnicalScoring)
        assert Decimal('0') <= technical_scoring.body_size_score <= Decimal('100')
        assert Decimal('0') <= technical_scoring.shadow_ratio_score <= Decimal('100')
        assert Decimal('0') <= technical_scoring.price_position_score <= Decimal('100')
        assert Decimal('0') <= technical_scoring.symmetry_score <= Decimal('100')
        assert Decimal('0') <= technical_scoring.textbook_compliance <= Decimal('100')
    
    def test_technical_scoring_pattern_specific(self, scoring_engine):
        """Test pattern-specific technical scoring."""
        # Create a perfect doji (small body)
        base_time = datetime.now(timezone.utc)
        doji_candle = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=base_time,
            open=Decimal('50000'),
            high=Decimal('50100'),
            low=Decimal('49900'),
            close=Decimal('50005'),  # Very close to open (small body)
            volume=Decimal('1000')
        )
        
        doji_pattern = CandlestickPattern(
            pattern_id="test_doji",
            symbol="BTC",
            pattern_type=PatternType.DOJI,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=base_time,
            candles=[doji_candle],
            confidence=Decimal('80'),
            reliability=Decimal('0.75'),
            bullish_probability=Decimal('0.5'),
            bearish_probability=Decimal('0.5'),
            completion_price=doji_candle.close,
            pattern_metadata={}
        )
        
        technical_scoring = scoring_engine._calculate_technical_scoring(doji_pattern)
        
        # Doji should score high on body size (because it's small)
        assert technical_scoring.body_size_score > Decimal('70')
    
    def test_volume_scoring_calculation(self, scoring_engine, sample_hammer_pattern, sample_candles):
        """Test volume scoring calculation."""
        volume_scoring = scoring_engine._calculate_volume_scoring(
            sample_hammer_pattern, sample_candles, None
        )
        
        assert isinstance(volume_scoring, VolumeScoring)
        assert Decimal('0') <= volume_scoring.volume_spike_score <= Decimal('100')
        assert Decimal('0') <= volume_scoring.volume_trend_score <= Decimal('100')
        assert Decimal('0') <= volume_scoring.breakout_volume_score <= Decimal('100')
        assert Decimal('0') <= volume_scoring.relative_volume_score <= Decimal('100')
    
    def test_context_scoring_calculation(self, scoring_engine, sample_hammer_pattern, sample_candles):
        """Test context scoring calculation."""
        context_scoring = scoring_engine._calculate_context_scoring(
            sample_hammer_pattern, sample_candles
        )
        
        assert isinstance(context_scoring, MarketContextScoring)
        assert Decimal('0') <= context_scoring.trend_alignment_score <= Decimal('100')
        assert Decimal('0') <= context_scoring.volatility_score <= Decimal('100')
        assert Decimal('0') <= context_scoring.session_score <= Decimal('100')
        assert Decimal('0') <= context_scoring.support_resistance_score <= Decimal('100')
        assert Decimal('0') <= context_scoring.momentum_score <= Decimal('100')
    
    def test_historical_performance_tracking(self, scoring_engine):
        """Test historical performance tracking."""
        # Get initial performance (should be default)
        initial_performance = scoring_engine._get_historical_performance(
            PatternType.HAMMER, Timeframe.FIVE_MINUTES
        )
        
        assert initial_performance.total_occurrences == 0
        assert initial_performance.success_rate == Decimal('55')  # Default
        
        # Record some outcomes
        base_time = datetime.now(timezone.utc)
        
        # Successful outcome
        outcome1 = PatternOutcome(
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            detection_time=base_time,
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            exit_time=base_time + timedelta(hours=2),
            success=True,
            profit_loss_pct=Decimal('2.0'),
            duration_hours=2.0
        )
        
        # Failed outcome
        outcome2 = PatternOutcome(
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            detection_time=base_time + timedelta(hours=1),
            entry_price=Decimal('51000'),
            exit_price=Decimal('50500'),
            exit_time=base_time + timedelta(hours=3),
            success=False,
            profit_loss_pct=Decimal('-1.0'),
            duration_hours=2.0
        )
        
        scoring_engine.record_pattern_outcome(outcome1)
        scoring_engine.record_pattern_outcome(outcome2)
        
        # Get updated performance
        updated_performance = scoring_engine._get_historical_performance(
            PatternType.HAMMER, Timeframe.FIVE_MINUTES
        )
        
        assert updated_performance.total_occurrences == 2
        assert updated_performance.successful_trades == 1
        assert updated_performance.success_rate == Decimal('50')  # 1 out of 2
    
    def test_batch_scoring(self, scoring_engine, sample_candles):
        """Test batch pattern scoring."""
        # Create multiple patterns
        base_time = datetime.now(timezone.utc)
        patterns = []
        
        for i, pattern_type in enumerate([PatternType.HAMMER, PatternType.DOJI, PatternType.SHOOTING_STAR]):
            candle = sample_candles[-1]
            pattern = CandlestickPattern(
                pattern_id=f"test_{pattern_type.value}_{i}",
                symbol="BTC",
                pattern_type=pattern_type,
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=base_time,
                candles=[candle],
                confidence=Decimal('70') + i * Decimal('5'),
                reliability=Decimal('0.75'),
                bullish_probability=Decimal('0.6') + i * Decimal('0.1'),
                bearish_probability=Decimal('0.4') - i * Decimal('0.1'),
                completion_price=candle.close,
                pattern_metadata={}
            )
            patterns.append(pattern)
        
        # Score all patterns
        scores = scoring_engine.batch_score_patterns(patterns, sample_candles)
        
        assert len(scores) == len(patterns)
        assert all(isinstance(score, CompositePatternScore) for score in scores)
        assert all(score.pattern in patterns for score in scores)
    
    def test_performance_summary(self, scoring_engine):
        """Test performance summary generation."""
        # Record some outcomes for different patterns
        base_time = datetime.now(timezone.utc)
        
        outcomes = [
            PatternOutcome(
                pattern_type=PatternType.HAMMER,
                timeframe=Timeframe.FIVE_MINUTES,
                detection_time=base_time,
                entry_price=Decimal('50000'),
                success=True
            ),
            PatternOutcome(
                pattern_type=PatternType.HAMMER,
                timeframe=Timeframe.FIVE_MINUTES,
                detection_time=base_time + timedelta(hours=1),
                entry_price=Decimal('51000'),
                success=False
            ),
            PatternOutcome(
                pattern_type=PatternType.DOJI,
                timeframe=Timeframe.ONE_MINUTE,
                detection_time=base_time + timedelta(hours=2),
                entry_price=Decimal('50500'),
                success=True
            )
        ]
        
        for outcome in outcomes:
            scoring_engine.record_pattern_outcome(outcome)
        
        summary = scoring_engine.get_performance_summary()
        
        assert summary["total_patterns_tracked"] == 3
        assert summary["overall_success_rate"] == round(2/3 * 100, 2)  # 2 successful out of 3
        assert "pattern_breakdown" in summary
        assert f"{PatternType.HAMMER.value}_{Timeframe.FIVE_MINUTES.value}" in summary["pattern_breakdown"]
        assert f"{PatternType.DOJI.value}_{Timeframe.ONE_MINUTE.value}" in summary["pattern_breakdown"]
    
    def test_session_scoring(self, scoring_engine):
        """Test market session scoring."""
        # Test different session times
        test_times = [
            (14, Decimal('100')),  # London/NY overlap (13-16)
            (10, Decimal('90')),   # London session (8-16, but not overlap)
            (17, Decimal('85')),   # NY session (13-21, but not overlap)
            (2, Decimal('70')),    # Asian session (22-08)
            (23, Decimal('70'))    # Asian session (22-08)
        ]
        
        for hour, expected_score in test_times:
            test_time = datetime.now(timezone.utc).replace(hour=hour, minute=0, second=0, microsecond=0)
            score = scoring_engine._calculate_session_score(test_time)
            assert score == expected_score
    
    def test_empty_data_handling(self, scoring_engine):
        """Test handling of empty or insufficient data."""
        # Create pattern with minimal candle
        empty_candle = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=datetime.now(timezone.utc),
            open=Decimal('50000'),
            high=Decimal('50000'),
            low=Decimal('50000'),
            close=Decimal('50000'),
            volume=Decimal('0')  # Zero volume
        )
        
        empty_pattern = CandlestickPattern(
            pattern_id="empty",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=datetime.now(timezone.utc),
            candles=[empty_candle],  # Minimal candle with zero volume
            confidence=Decimal('50'),
            reliability=Decimal('0.5'),
            bullish_probability=Decimal('0.5'),
            bearish_probability=Decimal('0.5'),
            completion_price=Decimal('50000'),
            pattern_metadata={}
        )
        
        # Should handle gracefully with default scores
        technical_scoring = scoring_engine._calculate_technical_scoring(empty_pattern)
        # Empty candles (no body/shadows) should have 0 for size/ratio scores, 50 for others
        assert technical_scoring.body_size_score == Decimal('0')
        assert technical_scoring.shadow_ratio_score == Decimal('0')
        assert technical_scoring.price_position_score == Decimal('50')
        assert technical_scoring.textbook_compliance == Decimal('50')
        
        # Test with insufficient market data
        empty_data: List[CandlestickData] = []
        volume_scoring = scoring_engine._calculate_volume_scoring(
            empty_pattern, empty_data, None
        )
        assert all(score == Decimal('50') for score in [
            volume_scoring.volume_spike_score,
            volume_scoring.volume_trend_score,
            volume_scoring.breakout_volume_score,
            volume_scoring.relative_volume_score
        ])


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.fixture
    def scoring_engine(self):
        """Create a scoring engine with some historical data."""
        engine = PatternScoringEngine()
        
        # Add some historical performance data
        base_time = datetime.now(timezone.utc)
        outcomes = [
            PatternOutcome(
                pattern_type=PatternType.HAMMER,
                timeframe=Timeframe.FIVE_MINUTES,
                detection_time=base_time - timedelta(days=i),
                entry_price=Decimal('50000'),
                exit_price=Decimal('50000') + (Decimal('500') if i % 2 == 0 else Decimal('-300')),
                success=i % 2 == 0,  # 50% success rate
                profit_loss_pct=Decimal('1.0') if i % 2 == 0 else Decimal('-0.6')
            )
            for i in range(20)
        ]
        
        for outcome in outcomes:
            engine.record_pattern_outcome(outcome)
        
        return engine
    
    def test_high_confidence_bullish_scenario(self, scoring_engine):
        """Test scenario with high confidence bullish pattern."""
        # Create strong uptrend data
        base_time = datetime.now(timezone.utc)
        candles = []
        
        for i in range(25):
            timestamp = base_time - timedelta(minutes=5 * (24 - i))
            price = Decimal('50000') + (i * Decimal('100'))  # Strong uptrend
            
            candle = CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=timestamp,
                open=price,
                high=price + Decimal('50'),
                low=price - Decimal('25'),
                close=price + Decimal('40'),
                volume=Decimal('1500') + (i * Decimal('50'))  # Increasing volume
            )
            candles.append(candle)
        
        # Create bullish hammer pattern at peak time (London/NY overlap)
        pattern_time = base_time.replace(hour=14, minute=30)  # Peak session
        pattern_candle = candles[-1]
        pattern_candle.timestamp = pattern_time
        
        hammer_pattern = CandlestickPattern(
            pattern_id="strong_hammer",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=pattern_time,
            candles=[pattern_candle],
            confidence=Decimal('85'),
            reliability=Decimal('0.9'),
            bullish_probability=Decimal('0.9'),
            bearish_probability=Decimal('0.1'),
            completion_price=pattern_candle.close,
            pattern_metadata={"strength": "very_strong"}
        )
        
        # Score the pattern
        score = scoring_engine.score_pattern(hammer_pattern, candles)
        
        # Should have high confidence
        assert score.weighted_confidence_score > Decimal('65')
        assert score.is_tradeable is True
        assert score.pattern_strength in [PatternStrength.STRONG, PatternStrength.VERY_STRONG]
        
        # Technical scoring should be strong for hammer
        assert score.technical_scoring.overall_technical_score > Decimal('60')
        
        # Context scoring should benefit from trend alignment and session
        assert score.context_scoring.trend_alignment_score > Decimal('80')
        assert score.context_scoring.session_score == Decimal('100')  # Peak session
        
        # Volume scoring should benefit from increasing volume
        assert score.volume_scoring.overall_volume_score > Decimal('60')
    
    def test_low_confidence_conflicting_scenario(self, scoring_engine):
        """Test scenario with low confidence and conflicting signals."""
        # Create sideways/conflicting data
        base_time = datetime.now(timezone.utc)
        candles = []
        
        for i in range(25):
            timestamp = base_time - timedelta(minutes=5 * (24 - i))
            # Sideways movement with high volatility
            base_price = Decimal('50000')
            volatility = Decimal('200') * (1 if i % 2 == 0 else -1)
            price = base_price + volatility
            
            candle = CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=timestamp,
                open=price,
                high=price + Decimal('100'),
                low=price - Decimal('100'),
                close=price + (Decimal('50') if i % 2 == 0 else Decimal('-50')),
                volume=Decimal('500') + (i * Decimal('10'))  # Low, inconsistent volume
            )
            candles.append(candle)
        
        # Create weak pattern during off-hours (21:00-22:00 UTC gap)
        pattern_time = base_time.replace(hour=21, minute=30)  # True off-hours
        pattern_candle = candles[-1]
        pattern_candle.timestamp = pattern_time
        
        weak_pattern = CandlestickPattern(
            pattern_id="weak_doji",
            symbol="BTC",
            pattern_type=PatternType.DOJI,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=pattern_time,
            candles=[pattern_candle],
            confidence=Decimal('35'),  # Very low confidence
            reliability=Decimal('0.3'),
            bullish_probability=Decimal('0.52'),
            bearish_probability=Decimal('0.48'),
            completion_price=pattern_candle.close,
            pattern_metadata={"strength": "weak"}
        )
        
        # Score the pattern
        score = scoring_engine.score_pattern(weak_pattern, candles)
        
        # Should have low confidence
        assert score.weighted_confidence_score < Decimal('65')
        assert score.is_tradeable is False
        assert score.pattern_strength in [PatternStrength.VERY_WEAK, PatternStrength.WEAK, PatternStrength.MODERATE]
        
        # Context scoring should be penalized
        assert score.context_scoring.session_score == Decimal('40')  # True off-hours penalty
        assert score.context_scoring.volatility_score <= Decimal('70')  # High volatility penalty
        
        # Risk-adjusted should be lower or equal (may be equal if no risk adjustments triggered)
        assert score.risk_adjusted_confidence <= score.weighted_confidence_score
    
    def test_volume_confirmation_scenario(self, scoring_engine):
        """Test scenario where volume confirmation is critical."""
        # Create data with volume breakout
        base_time = datetime.now(timezone.utc)
        candles = []
        
        for i in range(25):
            timestamp = base_time - timedelta(minutes=5 * (24 - i))
            price = Decimal('50000') + (i * Decimal('25'))
            
            # Volume spike in recent candles
            volume = Decimal('1000')
            if i >= 20:  # Recent candles
                volume = Decimal('2500')  # 2.5x average volume
            
            candle = CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=timestamp,
                open=price,
                high=price + Decimal('50'),
                low=price - Decimal('25'),
                close=price + Decimal('35'),
                volume=volume
            )
            candles.append(candle)
        
        # Create pattern with volume profile
        pattern_candle = candles[-1]
        volume_profile = VolumeProfile(
            pattern_volume=Decimal('2500'),
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('2.5'),  # Strong volume confirmation
            volume_trend=SignalDirection.BUY,
            breakout_volume=Decimal('2500')
        )
        
        pattern = CandlestickPattern(
            pattern_id="volume_confirmed",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=pattern_candle.timestamp,
            candles=[pattern_candle],
            confidence=Decimal('70'),
            reliability=Decimal('0.8'),
            bullish_probability=Decimal('0.75'),
            bearish_probability=Decimal('0.25'),
            completion_price=pattern_candle.close,
            pattern_metadata={}
        )
        
        # Score with volume profile
        score = scoring_engine.score_pattern(pattern, candles, volume_profile)
        
        # Volume scoring should be very high
        assert score.volume_scoring.volume_spike_score >= Decimal('80')
        assert score.volume_scoring.breakout_volume_score > Decimal('90')
        assert score.volume_scoring.overall_volume_score > Decimal('80')
        
        # Overall confidence should benefit from volume confirmation
        assert score.weighted_confidence_score > Decimal('65')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])