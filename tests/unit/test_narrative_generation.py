"""
Unit tests for narrative generation module.

Tests the generation of human-readable trading narratives from pattern analysis
and signal data for LLM consumption.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List

from src.bistoury.models.signals import (
    CandlestickData, CandlestickPattern, TradingSignal, SignalDirection, 
    PatternType, Timeframe, SignalType
)
from src.bistoury.strategies.candlestick_models import PatternStrength, VolumeProfile
from src.bistoury.strategies.pattern_scoring import (
    CompositePatternScore, TechnicalScoring, VolumeScoring, 
    MarketContextScoring, HistoricalPerformance
)
from src.bistoury.strategies.timeframe_analyzer import (
    TimeframeAnalysisResult, ConfluenceAnalysis, TrendAlignment
)
from src.bistoury.strategies.signal_generator import (
    GeneratedSignal, SignalEntryPoint, SignalRiskManagement, 
    SignalTiming, RiskLevel
)
from src.bistoury.strategies.narrative_generator import (
    NarrativeGenerator, NarrativeConfiguration, NarrativeStyle,
    PatternNarrative, TimeframeNarrative, TradingNarrative
)


class TestNarrativeConfiguration:
    """Test narrative configuration options."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = NarrativeConfiguration()
        assert config.style == NarrativeStyle.COMPREHENSIVE
        assert config.include_technical_details is True
        assert config.include_risk_metrics is True
        assert config.include_confidence_scores is True
        assert config.include_historical_context is True
        assert config.max_pattern_count == 5
        assert config.focus_timeframe is None
        assert config.emphasize_strengths is True
        assert config.include_warnings is True
    
    def test_custom_configuration(self):
        """Test custom configuration creation."""
        config = NarrativeConfiguration(
            style=NarrativeStyle.CONCISE,
            include_technical_details=False,
            max_pattern_count=3,
            focus_timeframe=Timeframe.FIFTEEN_MINUTES
        )
        assert config.style == NarrativeStyle.CONCISE
        assert config.include_technical_details is False
        assert config.max_pattern_count == 3
        assert config.focus_timeframe == Timeframe.FIFTEEN_MINUTES


@pytest.fixture
def sample_candle():
    """Create sample candlestick data."""
    return CandlestickData(
        symbol="BTC",
        timeframe=Timeframe.FIFTEEN_MINUTES,
        timestamp=datetime.now(timezone.utc),
        open=Decimal('50000'),
        high=Decimal('50200'),
        low=Decimal('49800'),
        close=Decimal('50100'),
        volume=Decimal('1000')
    )


@pytest.fixture  
def sample_pattern(sample_candle):
    """Create sample candlestick pattern."""
    return CandlestickPattern(
        pattern_id="test_hammer_123",
        pattern_type=PatternType.HAMMER,
        confidence=Decimal('75'),
        reliability=Decimal('0.8'),
        bullish_probability=Decimal('0.75'),
        bearish_probability=Decimal('0.25'),
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.FIFTEEN_MINUTES,
        symbol="BTC",
        completion_price=Decimal('50100'),
        candles=[sample_candle]
    )


@pytest.fixture
def sample_pattern_score(sample_pattern):
    """Create sample pattern score."""
    technical = TechnicalScoring(
        body_size_score=Decimal('80'),
        shadow_ratio_score=Decimal('85'),
        price_position_score=Decimal('75'),
        symmetry_score=Decimal('70'),
        textbook_compliance=Decimal('90')
    )
    
    volume = VolumeScoring(
        volume_spike_score=Decimal('70'),
        volume_trend_score=Decimal('65'),
        breakout_volume_score=Decimal('75'),
        relative_volume_score=Decimal('80')
    )
    
    context = MarketContextScoring(
        trend_alignment_score=Decimal('85'),
        volatility_score=Decimal('60'),
        session_score=Decimal('90'),
        support_resistance_score=Decimal('75'),
        momentum_score=Decimal('70')
    )
    
    historical = HistoricalPerformance(
        pattern_type=PatternType.HAMMER,
        timeframe=Timeframe.FIFTEEN_MINUTES,
        total_occurrences=50,
        successful_trades=35,
        success_rate=Decimal('70'),
        reliability_score=Decimal('75')
    )
    
    return CompositePatternScore(
        pattern=sample_pattern,
        technical_scoring=technical,
        volume_scoring=volume,
        context_scoring=context,
        historical_performance=historical
    )


class TestPatternNarrative:
    """Test pattern narrative generation."""
    
    def test_pattern_narrative_creation(self, sample_pattern, sample_pattern_score):
        """Test creating pattern narrative from pattern and score."""
        config = NarrativeConfiguration()
        narrative = PatternNarrative.from_pattern(sample_pattern, sample_pattern_score, config)
        
        assert narrative.pattern_type == PatternType.HAMMER
        assert narrative.timeframe == Timeframe.FIFTEEN_MINUTES
        assert "15-minute" in narrative.description
        assert "Hammer" in narrative.description
        assert "bullish reversal" in narrative.description
        assert len(narrative.significance) > 0
        assert len(narrative.strength_assessment) > 0
        assert len(narrative.reliability_note) > 0
        assert len(narrative.context_relevance) > 0
    
    def test_timeframe_to_text_conversion(self):
        """Test timeframe to text conversion."""
        assert PatternNarrative._timeframe_to_text(Timeframe.ONE_MINUTE) == "1-minute"
        assert PatternNarrative._timeframe_to_text(Timeframe.FIVE_MINUTES) == "5-minute"
        assert PatternNarrative._timeframe_to_text(Timeframe.FIFTEEN_MINUTES) == "15-minute"
        assert PatternNarrative._timeframe_to_text(Timeframe.ONE_HOUR) == "1-hour"
        assert PatternNarrative._timeframe_to_text(Timeframe.FOUR_HOURS) == "4-hour"
        assert PatternNarrative._timeframe_to_text(Timeframe.ONE_DAY) == "daily"


class TestTradingNarrative:
    """Test complete trading narrative generation."""
    
    def test_narrative_sections_property(self):
        """Test narrative sections property."""
        narrative = TradingNarrative(
            executive_summary="Test summary",
            market_overview="Test overview", 
            pattern_analysis="Test pattern",
            volume_analysis="Test volume",
            risk_assessment="Test risk",
            entry_strategy="Test entry",
            exit_strategy="Test exit",
            confidence_rationale="Test confidence"
        )
        
        sections = narrative.narrative_sections
        assert sections["executive_summary"] == "Test summary"
        assert sections["market_overview"] == "Test overview"
        assert sections["pattern_analysis"] == "Test pattern"
        assert sections["volume_analysis"] == "Test volume"
    
    def test_full_narrative_generation(self):
        """Test full narrative text generation."""
        narrative = TradingNarrative(
            executive_summary="Test summary",
            market_overview="Test overview",
            pattern_analysis="Test pattern", 
            volume_analysis="Test volume",
            risk_assessment="Test risk",
            entry_strategy="Test entry",
            exit_strategy="Test exit",
            confidence_rationale="Test confidence",
            supporting_factors=["Factor 1", "Factor 2"],
            conflicting_factors=["Conflict 1"],
            key_warnings=["Warning 1", "Warning 2"]
        )
        
        full_text = narrative.full_narrative
        assert "EXECUTIVE SUMMARY:" in full_text
        assert "SUPPORTING FACTORS:" in full_text
        assert "• Factor 1" in full_text
        assert "⚠️ Warning 1" in full_text


class TestNarrativeGenerator:
    """Test the main narrative generator."""
    
    @pytest.fixture
    def sample_signal(self, sample_pattern, sample_pattern_score):
        """Create a sample generated signal."""
        # Create signal components
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50150'),
            entry_timing=SignalTiming.CONFIRMATION,
            entry_zone_low=Decimal('50100'),
            entry_zone_high=Decimal('50200'),
            max_slippage_pct=Decimal('0.1'),
            entry_window_minutes=30
        )
        
        risk_management = SignalRiskManagement(
            stop_loss_price=Decimal('49750'),
            take_profit_price=Decimal('50750'),
            risk_amount=Decimal('400'),
            reward_amount=Decimal('600'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        # Create base signal
        base_signal = TradingSignal.create_buy_signal(
            signal_id="test_signal_123",
            symbol="BTC",
            price=Decimal('50150'),
            confidence=Decimal('75'),
            strength=Decimal('0.75'),
            source="test",
            reason="Test hammer pattern",
            timeframe=Timeframe.FIFTEEN_MINUTES,
            target_price=Decimal('50750'),
            stop_loss=Decimal('49750'),
            signal_type=SignalType.PATTERN
        )
        
        return GeneratedSignal(
            base_signal=base_signal,
            entry_point=entry_point,
            risk_management=risk_management,
            source_pattern=sample_pattern,
            pattern_score=sample_pattern_score
        )
    
    @pytest.fixture
    def generator(self):
        """Create narrative generator."""
        return NarrativeGenerator()
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        # Default config
        generator = NarrativeGenerator()
        assert generator.config.style == NarrativeStyle.COMPREHENSIVE
        
        # Custom config
        config = NarrativeConfiguration(style=NarrativeStyle.CONCISE)
        generator = NarrativeGenerator(config)
        assert generator.config.style == NarrativeStyle.CONCISE
    
    def test_signal_narrative_generation(self, generator, sample_signal):
        """Test generating narrative from signal."""
        narrative = generator.generate_signal_narrative(sample_signal)
        
        # Check all required sections are present
        assert len(narrative.executive_summary) > 0
        assert len(narrative.market_overview) > 0
        assert len(narrative.pattern_analysis) > 0
        assert len(narrative.volume_analysis) > 0
        assert len(narrative.risk_assessment) > 0
        assert len(narrative.entry_strategy) > 0
        assert len(narrative.exit_strategy) > 0
        assert len(narrative.confidence_rationale) > 0
        
        # Check content quality
        assert "BULLISH" in narrative.executive_summary
        assert "Hammer" in narrative.executive_summary
        assert "BTC" in narrative.executive_summary
        assert "100%" in narrative.executive_summary
    
    def test_executive_summary_generation(self, generator, sample_signal):
        """Test executive summary generation."""
        summary = generator._generate_executive_summary(sample_signal)
        
        assert "BULLISH" in summary
        assert "Hammer" in summary
        assert "BTC" in summary
        assert "100%" in summary
        assert "50150" in summary
    
    def test_volume_analysis_generation(self, generator, sample_signal):
        """Test volume analysis generation."""
        analysis = generator._generate_volume_analysis(sample_signal.pattern_score)
        
        assert len(analysis) > 0
        assert "Volume analysis" in analysis
        assert "/100" in analysis  # Should mention scores
    
    def test_quick_narrative_generation(self, generator, sample_signal):
        """Test quick narrative generation."""
        quick = generator.generate_quick_narrative(sample_signal, NarrativeStyle.CONCISE)
        
        assert len(quick) > 0
        assert "BULLISH" in quick
        assert "Hammer" in quick
        assert "100%" in quick
        assert "50150" in quick
    
    def test_pattern_summary_generation(self, generator, sample_candle):
        """Test pattern summary generation."""
        # Create multiple patterns
        patterns = []
        scores = []
        
        for i, pattern_type in enumerate([PatternType.HAMMER, PatternType.DOJI]):
            pattern = CandlestickPattern(
                pattern_id=f"test_{pattern_type.value}_{i}",
                pattern_type=pattern_type,
                confidence=Decimal(str(80 - i * 10)),
                reliability=Decimal('0.8'),
                bullish_probability=Decimal('0.75'),
                bearish_probability=Decimal('0.25'),
                timestamp=datetime.now(timezone.utc),
                timeframe=Timeframe.FIFTEEN_MINUTES,
                symbol="BTC",
                completion_price=Decimal('50100'),
                candles=[sample_candle]
            )
            patterns.append(pattern)
            
            # Create simple score
            technical = TechnicalScoring(
                body_size_score=Decimal(str(80 - i * 10)),
                shadow_ratio_score=Decimal('75'),
                price_position_score=Decimal('75'),
                symmetry_score=Decimal('70'),
                textbook_compliance=Decimal('80')
            )
            
            volume = VolumeScoring(
                volume_spike_score=Decimal('70'),
                volume_trend_score=Decimal('65'),
                breakout_volume_score=Decimal('75'),
                relative_volume_score=Decimal('80')
            )
            
            context = MarketContextScoring(
                trend_alignment_score=Decimal('75'),
                volatility_score=Decimal('60'),
                session_score=Decimal('80'),
                support_resistance_score=Decimal('75'),
                momentum_score=Decimal('70')
            )
            
            historical = HistoricalPerformance(
                pattern_type=pattern_type,
                timeframe=Timeframe.FIFTEEN_MINUTES,
                total_occurrences=50,
                successful_trades=35,
                success_rate=Decimal('70'),
                reliability_score=Decimal('75')
            )
            
            score = CompositePatternScore(
                pattern=pattern,
                technical_scoring=technical,
                volume_scoring=volume,
                context_scoring=context,
                historical_performance=historical
            )
            scores.append(score)
        
        summary = generator.generate_pattern_summary(patterns, scores)
        
        assert len(summary) > 0
        assert "Analysis of 2 candlestick patterns" in summary
        assert "Hammer" in summary
        assert "Doji" in summary
    
    def test_empty_pattern_summary(self, generator):
        """Test pattern summary with no patterns."""
        summary = generator.generate_pattern_summary([], [])
        assert "No significant patterns detected" in summary 