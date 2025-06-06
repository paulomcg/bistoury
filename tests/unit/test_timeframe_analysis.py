"""
Unit tests for Multi-Timeframe Analysis Engine

Tests cover:
- TimeframeSynchronization: Data alignment and completeness
- TrendAlignment: Trend analysis across timeframes  
- ConfluenceAnalysis: Pattern confluence detection
- TimeframeAnalysisResult: Complete analysis results
- TimeframeAnalyzer: Main analysis engine

All tests designed to validate latency requirements (<2s) and accuracy.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List

from src.bistoury.models.market_data import CandlestickData, Timeframe
from src.bistoury.models.signals import (
    CandlestickPattern, 
    PatternType, 
    SignalDirection, 
    SignalType
)
from src.bistoury.strategies.candlestick_models import (
    PatternStrength,
    TimeframePriority,
    PatternConfluence,
    VolumeProfile,
    PatternQuality,
    MultiTimeframePattern,
    StrategyConfiguration
)
from src.bistoury.strategies.timeframe_analyzer import (
    TimeframeSynchronization,
    TrendAlignment,
    ConfluenceAnalysis,
    TimeframeAnalysisResult,
    TimeframeAnalyzer
)


class TestTimeframeSynchronization:
    """Test TimeframeSynchronization data alignment and management."""
    
    @pytest.fixture
    def sample_timeframe_data(self) -> Dict[Timeframe, List[CandlestickData]]:
        """Create sample timeframe data for testing."""
        base_time = datetime.now(timezone.utc)
        
        # Create candlestick data for different timeframes
        timeframe_data = {}
        
        # 1-minute data (60 candles = 1 hour)
        timeframe_data[Timeframe.ONE_MINUTE] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.ONE_MINUTE,
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(f'{50000 + i}'),
                high=Decimal(f'{50100 + i}'),
                low=Decimal(f'{49900 + i}'),
                close=Decimal(f'{50050 + i}'),
                volume=Decimal('1000')
            )
            for i in range(60)
        ]
        
        # 5-minute data (12 candles = 1 hour)
        timeframe_data[Timeframe.FIVE_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=base_time + timedelta(minutes=i*5),
                open=Decimal(f'{50000 + i*5}'),
                high=Decimal(f'{50200 + i*5}'),
                low=Decimal(f'{49800 + i*5}'),
                close=Decimal(f'{50100 + i*5}'),
                volume=Decimal('5000')
            )
            for i in range(12)
        ]
        
        # 15-minute data (4 candles = 1 hour)
        timeframe_data[Timeframe.FIFTEEN_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIFTEEN_MINUTES,
                timestamp=base_time + timedelta(minutes=i*15),
                open=Decimal(f'{50000 + i*15}'),
                high=Decimal(f'{50300 + i*15}'),
                low=Decimal(f'{49700 + i*15}'),
                close=Decimal(f'{50150 + i*15}'),
                volume=Decimal('15000')
            )
            for i in range(4)
        ]
        
        return timeframe_data
    
    def test_timeframe_synchronization_creation(self, sample_timeframe_data):
        """Test basic creation and properties."""
        base_timestamp = datetime.now(timezone.utc)
        
        sync = TimeframeSynchronization(
            base_timestamp=base_timestamp,
            timeframe_data=sample_timeframe_data,
            sync_window_minutes=60
        )
        
        assert sync.base_timestamp == base_timestamp
        assert len(sync.synchronized_timeframes) == 3
        assert Timeframe.ONE_MINUTE in sync.synchronized_timeframes
        assert Timeframe.FIVE_MINUTES in sync.synchronized_timeframes
        assert Timeframe.FIFTEEN_MINUTES in sync.synchronized_timeframes
    
    def test_data_completeness_calculation(self, sample_timeframe_data):
        """Test data completeness percentage calculation."""
        sync = TimeframeSynchronization(
            base_timestamp=datetime.now(timezone.utc),
            timeframe_data=sample_timeframe_data,
            sync_window_minutes=60
        )
        
        completeness = sync.data_completeness
        
        # All timeframes should have 100% completeness for 60-minute window
        assert completeness[Timeframe.ONE_MINUTE] == Decimal('100')  # 60 expected, 60 actual
        assert completeness[Timeframe.FIVE_MINUTES] == Decimal('100')  # 12 expected, 12 actual  
        assert completeness[Timeframe.FIFTEEN_MINUTES] == Decimal('100')  # 4 expected, 4 actual
    
    def test_get_latest_candles(self, sample_timeframe_data):
        """Test getting latest candles from timeframe data."""
        sync = TimeframeSynchronization(
            base_timestamp=datetime.now(timezone.utc),
            timeframe_data=sample_timeframe_data,
            sync_window_minutes=60
        )
        
        # Get latest 1-minute candle
        latest_1m = sync.get_latest_candles(Timeframe.ONE_MINUTE, 1)
        assert len(latest_1m) == 1
        assert latest_1m[0].close == Decimal('50109')  # Last candle's close price
        
        # Get latest 3 5-minute candles
        latest_5m = sync.get_latest_candles(Timeframe.FIVE_MINUTES, 3)
        assert len(latest_5m) == 3
        assert latest_5m[-1].close == Decimal('50155')  # Last 5m candle's close
    
    def test_get_candles_in_window(self, sample_timeframe_data):
        """Test getting candles within a specific time window."""
        base_time = datetime.now(timezone.utc)
        sync = TimeframeSynchronization(
            base_timestamp=base_time,
            timeframe_data=sample_timeframe_data,
            sync_window_minutes=60
        )
        
        # Get 1-minute candles in first 15 minutes
        start_time = base_time
        end_time = base_time + timedelta(minutes=15)
        
        candles_in_window = sync.get_candles_in_window(Timeframe.ONE_MINUTE, start_time, end_time)
        # Should be all candles from start_time to end_time (inclusive)
        expected_count = len([c for c in sample_timeframe_data[Timeframe.ONE_MINUTE] 
                             if start_time <= c.timestamp <= end_time])
        assert len(candles_in_window) == expected_count
        
        # Verify candles are sorted by timestamp
        timestamps = [c.timestamp for c in candles_in_window]
        assert timestamps == sorted(timestamps)


class TestTrendAlignment:
    """Test TrendAlignment trend analysis across timeframes."""
    
    @pytest.fixture
    def sample_trend_alignment(self) -> TrendAlignment:
        """Create sample trend alignment for testing."""
        return TrendAlignment(
            timeframe_trends={
                Timeframe.ONE_MINUTE: SignalDirection.BUY,
                Timeframe.FIVE_MINUTES: SignalDirection.BUY,
                Timeframe.FIFTEEN_MINUTES: SignalDirection.STRONG_BUY
            },
            alignment_score=Decimal('85'),
            dominant_trend=SignalDirection.BUY,
            trend_strength=PatternStrength.STRONG,
            conflicting_timeframes=[]
        )
    
    def test_trend_alignment_creation(self, sample_trend_alignment):
        """Test basic creation and properties."""
        assert sample_trend_alignment.alignment_score == Decimal('85')
        assert sample_trend_alignment.dominant_trend == SignalDirection.BUY
        assert sample_trend_alignment.trend_strength == PatternStrength.STRONG
        assert len(sample_trend_alignment.conflicting_timeframes) == 0
    
    def test_is_aligned_property(self, sample_trend_alignment):
        """Test alignment detection logic."""
        # Strong alignment (85 > 70)
        assert sample_trend_alignment.is_aligned is True
        
        # Weak alignment
        weak_alignment = TrendAlignment(
            timeframe_trends={
                Timeframe.ONE_MINUTE: SignalDirection.HOLD,
                Timeframe.FIVE_MINUTES: SignalDirection.BUY,
                Timeframe.FIFTEEN_MINUTES: SignalDirection.SELL
            },
            alignment_score=Decimal('40'),
            dominant_trend=SignalDirection.HOLD,
            trend_strength=PatternStrength.WEAK,
            conflicting_timeframes=[Timeframe.FIFTEEN_MINUTES]
        )
        assert weak_alignment.is_aligned is False
    
    def test_bullish_bearish_timeframes(self, sample_trend_alignment):
        """Test bullish and bearish timeframe identification."""
        bullish_tfs = sample_trend_alignment.bullish_timeframes
        bearish_tfs = sample_trend_alignment.bearish_timeframes
        
        assert len(bullish_tfs) == 3
        assert Timeframe.ONE_MINUTE in bullish_tfs
        assert Timeframe.FIVE_MINUTES in bullish_tfs
        assert Timeframe.FIFTEEN_MINUTES in bullish_tfs
        assert len(bearish_tfs) == 0
        
        # Test mixed trends
        mixed_alignment = TrendAlignment(
            timeframe_trends={
                Timeframe.ONE_MINUTE: SignalDirection.BUY,
                Timeframe.FIVE_MINUTES: SignalDirection.SELL,
                Timeframe.FIFTEEN_MINUTES: SignalDirection.HOLD
            },
            alignment_score=Decimal('50'),
            dominant_trend=SignalDirection.HOLD,
            trend_strength=PatternStrength.MODERATE,
            conflicting_timeframes=[]
        )
        
        assert len(mixed_alignment.bullish_timeframes) == 1
        assert len(mixed_alignment.bearish_timeframes) == 1
        assert Timeframe.ONE_MINUTE in mixed_alignment.bullish_timeframes
        assert Timeframe.FIVE_MINUTES in mixed_alignment.bearish_timeframes


class TestConfluenceAnalysis:
    """Test ConfluenceAnalysis pattern confluence detection."""
    
    @pytest.fixture
    def sample_patterns(self) -> Dict[Timeframe, List[CandlestickPattern]]:
        """Create sample patterns for confluence testing."""
        patterns = {}
        base_time = datetime.now(timezone.utc)
        
        # Create sample candles for patterns
        sample_candle = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.ONE_MINUTE,
            timestamp=base_time,
            open=Decimal('50000'),
            high=Decimal('50100'),
            low=Decimal('49900'),
            close=Decimal('50050'),
            volume=Decimal('1000')
        )
        
        # 1-minute patterns
        patterns[Timeframe.ONE_MINUTE] = [
            CandlestickPattern(
                pattern_id="test_hammer_1m",
                symbol="BTC",
                pattern_type=PatternType.HAMMER,
                candles=[sample_candle],
                timeframe=Timeframe.ONE_MINUTE,
                confidence=Decimal('75'),
                reliability=Decimal('0.8'),
                bullish_probability=Decimal('0.8'),
                bearish_probability=Decimal('0.2'),
                timestamp=base_time,
                completion_price=Decimal('50050'),
                pattern_metadata={"test": True}
            )
        ]
        
        # 5-minute patterns (same pattern type for confluence)
        sample_candle_5m = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=base_time,
            open=Decimal('50000'),
            high=Decimal('50100'),
            low=Decimal('49900'),
            close=Decimal('50050'),
            volume=Decimal('5000')
        )
        
        patterns[Timeframe.FIVE_MINUTES] = [
            CandlestickPattern(
                pattern_id="test_hammer_5m",
                symbol="BTC",
                pattern_type=PatternType.HAMMER,
                candles=[sample_candle_5m],
                timeframe=Timeframe.FIVE_MINUTES,
                confidence=Decimal('80'),
                reliability=Decimal('0.85'),
                bullish_probability=Decimal('0.85'),
                bearish_probability=Decimal('0.15'),
                timestamp=base_time,
                completion_price=Decimal('50050'),
                pattern_metadata={"test": True}
            )
        ]
        
        # 15-minute patterns (different pattern type)
        sample_candle_15m = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.FIFTEEN_MINUTES,
            timestamp=base_time,
            open=Decimal('50000'),
            high=Decimal('50100'),
            low=Decimal('49900'),
            close=Decimal('50050'),
            volume=Decimal('15000')
        )
        
        patterns[Timeframe.FIFTEEN_MINUTES] = [
            CandlestickPattern(
                pattern_id="test_doji_15m",
                symbol="BTC",
                pattern_type=PatternType.DOJI,
                candles=[sample_candle_15m],
                timeframe=Timeframe.FIFTEEN_MINUTES,
                confidence=Decimal('70'),
                reliability=Decimal('0.7'),
                bullish_probability=Decimal('0.5'),
                bearish_probability=Decimal('0.5'),
                timestamp=base_time,
                completion_price=Decimal('50050'),
                pattern_metadata={"test": True}
            )
        ]
        
        return patterns
    
    def test_confluence_analysis_creation(self, sample_patterns):
        """Test basic creation and properties."""
        confluence = ConfluenceAnalysis(
            primary_timeframe=Timeframe.FIVE_MINUTES,
            patterns_by_timeframe=sample_patterns,
            confluence_patterns=[sample_patterns[Timeframe.FIVE_MINUTES][0]],  # Hammer with highest confidence
            confluence_score=Decimal('70'),
            weighted_confidence=Decimal('75')
        )
        
        assert confluence.primary_timeframe == Timeframe.FIVE_MINUTES
        assert len(confluence.patterns_by_timeframe) == 3
        assert len(confluence.confluence_patterns) == 1
        assert confluence.confluence_score == Decimal('70')
        assert confluence.weighted_confidence == Decimal('75')
    
    def test_has_strong_confluence(self, sample_patterns):
        """Test strong confluence detection logic."""
        # Strong confluence (score >= 60, patterns >= 2, weighted >= 65)
        strong_confluence = ConfluenceAnalysis(
            primary_timeframe=Timeframe.FIVE_MINUTES,
            patterns_by_timeframe=sample_patterns,
            confluence_patterns=[
                sample_patterns[Timeframe.ONE_MINUTE][0],
                sample_patterns[Timeframe.FIVE_MINUTES][0]
            ],
            confluence_score=Decimal('75'),
            weighted_confidence=Decimal('80')
        )
        assert strong_confluence.has_strong_confluence is True
        
        # Weak confluence (too few patterns)
        weak_confluence = ConfluenceAnalysis(
            primary_timeframe=Timeframe.FIVE_MINUTES,
            patterns_by_timeframe=sample_patterns,
            confluence_patterns=[sample_patterns[Timeframe.FIVE_MINUTES][0]],
            confluence_score=Decimal('75'),
            weighted_confidence=Decimal('80')
        )
        assert weak_confluence.has_strong_confluence is False
    
    def test_pattern_type_distribution(self, sample_patterns):
        """Test pattern type distribution calculation."""
        confluence = ConfluenceAnalysis(
            primary_timeframe=Timeframe.FIVE_MINUTES,
            patterns_by_timeframe=sample_patterns,
            confluence_patterns=[],
            confluence_score=Decimal('50'),
            weighted_confidence=Decimal('60')
        )
        
        distribution = confluence.pattern_type_distribution
        assert distribution[PatternType.HAMMER] == 2  # 1m and 5m
        assert distribution[PatternType.DOJI] == 1    # 15m
    
    def test_dominant_signal_direction(self, sample_patterns):
        """Test dominant signal direction calculation."""
        # Bullish confluence (Hammer patterns are bullish)
        bullish_confluence = ConfluenceAnalysis(
            primary_timeframe=Timeframe.FIVE_MINUTES,
            patterns_by_timeframe=sample_patterns,
            confluence_patterns=[
                sample_patterns[Timeframe.ONE_MINUTE][0],  # 75 confidence, 0.8 bullish
                sample_patterns[Timeframe.FIVE_MINUTES][0]  # 80 confidence, 0.85 bullish
            ],
            confluence_score=Decimal('70'),
            weighted_confidence=Decimal('75')
        )
        assert bullish_confluence.dominant_signal_direction == SignalDirection.BUY
        
        # Neutral confluence (no patterns)
        neutral_confluence = ConfluenceAnalysis(
            primary_timeframe=Timeframe.FIVE_MINUTES,
            patterns_by_timeframe={},
            confluence_patterns=[],
            confluence_score=Decimal('0'),
            weighted_confidence=Decimal('0')
        )
        assert neutral_confluence.dominant_signal_direction is None


class TestTimeframeAnalysisResult:
    """Test TimeframeAnalysisResult complete analysis results."""
    
    @pytest.fixture
    def sample_timeframe_data(self) -> Dict[Timeframe, List[CandlestickData]]:
        """Create sample timeframe data for testing."""
        base_time = datetime.now(timezone.utc)
        
        # Create candlestick data for different timeframes
        timeframe_data = {}
        
        # 1-minute data (60 candles = 1 hour)
        timeframe_data[Timeframe.ONE_MINUTE] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.ONE_MINUTE,
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(f'{50000 + i}'),
                high=Decimal(f'{50100 + i}'),
                low=Decimal(f'{49900 + i}'),
                close=Decimal(f'{50050 + i}'),
                volume=Decimal('1000')
            )
            for i in range(60)
        ]
        
        # 5-minute data (12 candles = 1 hour)
        timeframe_data[Timeframe.FIVE_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=base_time + timedelta(minutes=i*5),
                open=Decimal(f'{50000 + i*5}'),
                high=Decimal(f'{50200 + i*5}'),
                low=Decimal(f'{49800 + i*5}'),
                close=Decimal(f'{50100 + i*5}'),
                volume=Decimal('5000')
            )
            for i in range(12)
        ]
        
        # 15-minute data (4 candles = 1 hour)
        timeframe_data[Timeframe.FIFTEEN_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIFTEEN_MINUTES,
                timestamp=base_time + timedelta(minutes=i*15),
                open=Decimal(f'{50000 + i*15}'),
                high=Decimal(f'{50300 + i*15}'),
                low=Decimal(f'{49700 + i*15}'),
                close=Decimal(f'{50150 + i*15}'),
                volume=Decimal('15000')
            )
            for i in range(4)
        ]
        
        return timeframe_data
    
    @pytest.fixture
    def sample_patterns(self) -> Dict[Timeframe, List[CandlestickPattern]]:
        """Create sample patterns for confluence testing."""
        patterns = {}
        base_time = datetime.now(timezone.utc)
        
        # Add patterns for all three timeframes
        for timeframe in [Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES]:
            confidence = Decimal('75') if timeframe == Timeframe.ONE_MINUTE else (
                Decimal('80') if timeframe == Timeframe.FIVE_MINUTES else Decimal('70')
            )
            volume = Decimal('1000') if timeframe == Timeframe.ONE_MINUTE else (
                Decimal('5000') if timeframe == Timeframe.FIVE_MINUTES else Decimal('15000')
            )
            
            sample_candle = CandlestickData(
                symbol="BTC",
                timeframe=timeframe,
                timestamp=base_time,
                open=Decimal('50000'),
                high=Decimal('50100'),
                low=Decimal('49900'),
                close=Decimal('50050'),
                volume=volume
            )
            
            patterns[timeframe] = [
                CandlestickPattern(
                    pattern_id=f"test_hammer_{timeframe.value}",
                    symbol="BTC",
                    pattern_type=PatternType.HAMMER,
                    candles=[sample_candle],
                    timeframe=timeframe,
                    confidence=confidence,
                    reliability=Decimal('0.8'),
                    bullish_probability=Decimal('0.8'),
                    bearish_probability=Decimal('0.2'),
                    timestamp=base_time,
                    completion_price=Decimal('50050'),
                    pattern_metadata={"is_reversal_pattern": True, "is_continuation_pattern": False}
                )
            ]
        
        return patterns
    
    @pytest.fixture
    def sample_analysis_result(self, sample_timeframe_data, sample_patterns) -> TimeframeAnalysisResult:
        """Create sample analysis result for testing."""
        # Create components
        sync = TimeframeSynchronization(
            base_timestamp=datetime.now(timezone.utc),
            timeframe_data=sample_timeframe_data,
            sync_window_minutes=60
        )
        
        trend_alignment = TrendAlignment(
            timeframe_trends={
                Timeframe.ONE_MINUTE: SignalDirection.BUY,
                Timeframe.FIVE_MINUTES: SignalDirection.BUY,
                Timeframe.FIFTEEN_MINUTES: SignalDirection.BUY
            },
            alignment_score=Decimal('85'),
            dominant_trend=SignalDirection.BUY,
            trend_strength=PatternStrength.STRONG,
            conflicting_timeframes=[]
        )
        
        confluence = ConfluenceAnalysis(
            primary_timeframe=Timeframe.FIVE_MINUTES,
            patterns_by_timeframe=sample_patterns,
            confluence_patterns=[sample_patterns[Timeframe.FIVE_MINUTES][0]],
            confluence_score=Decimal('70'),
            weighted_confidence=Decimal('75')
        )
        
        return TimeframeAnalysisResult(
            symbol="BTC",
            synchronization=sync,
            trend_alignment=trend_alignment,
            confluence_analysis=confluence,
            single_patterns=sample_patterns,
            multi_patterns={},  # Empty for this test
            analysis_duration_ms=500,  # Under 2s requirement
            data_quality_score=Decimal('95')
        )
    
    def test_analysis_result_creation(self, sample_analysis_result):
        """Test basic creation and properties."""
        assert sample_analysis_result.symbol == "BTC"
        assert sample_analysis_result.analysis_duration_ms == 500
        assert sample_analysis_result.data_quality_score == Decimal('95')
    
    def test_meets_latency_requirement(self, sample_analysis_result):
        """Test latency requirement validation."""
        # Under 2s (good)
        assert sample_analysis_result.meets_latency_requirement is True
        
        # Over 2s (bad)
        slow_result = TimeframeAnalysisResult(
            symbol="BTC",
            synchronization=sample_analysis_result.synchronization,
            trend_alignment=sample_analysis_result.trend_alignment,
            confluence_analysis=sample_analysis_result.confluence_analysis,
            single_patterns=sample_analysis_result.single_patterns,
            multi_patterns={},
            analysis_duration_ms=3000,  # 3 seconds
            data_quality_score=Decimal('95')
        )
        assert slow_result.meets_latency_requirement is False
    
    def test_total_patterns_detected(self, sample_analysis_result):
        """Test pattern counting across timeframes."""
        # 3 single patterns (one per timeframe), 0 multi patterns
        assert sample_analysis_result.total_patterns_detected == 3
    
    def test_best_timeframe_pattern(self, sample_analysis_result):
        """Test finding highest confidence pattern."""
        best_pattern = sample_analysis_result.best_timeframe_pattern
        assert best_pattern is not None
        assert best_pattern.confidence == Decimal('80')  # 5m Hammer has highest confidence
        assert best_pattern.pattern_type == PatternType.HAMMER
    
    def test_trading_recommendation(self, sample_analysis_result):
        """Test overall trading recommendation calculation."""
        recommendation = sample_analysis_result.trading_recommendation
        # All trends bullish, confluence bullish, best pattern bullish
        assert recommendation in [SignalDirection.BUY, SignalDirection.STRONG_BUY]


class TestTimeframeAnalyzer:
    """Test TimeframeAnalyzer main analysis engine."""
    
    @pytest.fixture
    def analyzer(self) -> TimeframeAnalyzer:
        """Create analyzer instance for testing."""
        config = StrategyConfiguration(
            timeframes=[Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES],
            primary_timeframe=Timeframe.FIVE_MINUTES,
            min_pattern_confidence=Decimal('40'),  # Lower for testing
            min_confluence_score=Decimal('50'),
            min_quality_score=Decimal('60')
        )
        return TimeframeAnalyzer(config)
    
    @pytest.fixture
    def test_timeframe_data(self) -> Dict[Timeframe, List[CandlestickData]]:
        """Create test data with patterns for analysis."""
        base_time = datetime.now(timezone.utc)
        
        # Create data that will generate detectable patterns
        timeframe_data = {}
        
        # 1-minute data with potential hammer pattern
        timeframe_data[Timeframe.ONE_MINUTE] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.ONE_MINUTE,
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal('50000'),
                high=Decimal('50010'),  # Small upper shadow
                low=Decimal('49700'),   # Long lower shadow for hammer
                close=Decimal('49980'), # Close near high
                volume=Decimal('1000')
            )
            for i in range(10)
        ]
        
        # 5-minute data
        timeframe_data[Timeframe.FIVE_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=base_time + timedelta(minutes=i*5),
                open=Decimal('50000'),
                high=Decimal('50020'),
                low=Decimal('49750'),
                close=Decimal('49990'),
                volume=Decimal('5000')
            )
            for i in range(5)
        ]
        
        # 15-minute data
        timeframe_data[Timeframe.FIFTEEN_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIFTEEN_MINUTES,
                timestamp=base_time + timedelta(minutes=i*15),
                open=Decimal('50000'),
                high=Decimal('50050'),
                low=Decimal('49800'),
                close=Decimal('50000'),
                volume=Decimal('15000')
            )
            for i in range(3)
        ]
        
        return timeframe_data
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config is not None
        assert analyzer.single_recognizer is not None
        assert analyzer.multi_recognizer is not None
        assert analyzer._analysis_count == 0
    
    @pytest.mark.asyncio
    async def test_analyze_basic_functionality(self, analyzer, test_timeframe_data):
        """Test basic analysis functionality."""
        result = await analyzer.analyze(
            symbol="BTC",
            timeframe_data=test_timeframe_data,
            primary_timeframe=Timeframe.FIVE_MINUTES
        )
        
        assert result.symbol == "BTC"
        assert result.meets_latency_requirement is True  # Should complete quickly
        assert result.data_quality_score > Decimal('0')
        assert len(result.synchronization.synchronized_timeframes) == 3
    
    @pytest.mark.asyncio
    async def test_analyze_with_missing_primary_timeframe(self, analyzer, test_timeframe_data):
        """Test error handling for missing primary timeframe."""
        # Remove primary timeframe from data
        limited_data = {k: v for k, v in test_timeframe_data.items() if k != Timeframe.FIVE_MINUTES}
        
        with pytest.raises(ValueError, match="Primary timeframe .* not found"):
            await analyzer.analyze(
                symbol="BTC",
                timeframe_data=limited_data,
                primary_timeframe=Timeframe.FIVE_MINUTES
            )
    
    @pytest.mark.asyncio
    async def test_analyze_empty_data(self, analyzer):
        """Test analysis with empty data."""
        empty_data = {
            Timeframe.ONE_MINUTE: [],
            Timeframe.FIVE_MINUTES: [],
            Timeframe.FIFTEEN_MINUTES: []
        }
        
        result = await analyzer.analyze(
            symbol="BTC",
            timeframe_data=empty_data
        )
        
        assert result.total_patterns_detected == 0
        assert result.data_quality_score == Decimal('0')
    
    def test_create_volume_profile(self, analyzer, test_timeframe_data):
        """Test volume profile creation."""
        candles = test_timeframe_data[Timeframe.ONE_MINUTE]
        volume_profile = analyzer._create_volume_profile(candles)
        
        assert volume_profile.pattern_volume > Decimal('0')
        assert volume_profile.average_volume > Decimal('0')
        assert volume_profile.volume_ratio > Decimal('0')
    
    def test_analyze_trend_alignment(self, analyzer, test_timeframe_data):
        """Test trend alignment analysis."""
        trend_alignment = analyzer._analyze_trend_alignment(test_timeframe_data)
        
        assert trend_alignment.alignment_score >= Decimal('0')
        assert trend_alignment.alignment_score <= Decimal('100')
        assert len(trend_alignment.timeframe_trends) == 3
        assert trend_alignment.dominant_trend in [
            SignalDirection.BUY, SignalDirection.SELL, 
            SignalDirection.STRONG_BUY, SignalDirection.STRONG_SELL,
            SignalDirection.HOLD
        ]
    
    def test_performance_stats(self, analyzer):
        """Test performance statistics tracking."""
        # Initial stats
        initial_stats = analyzer.performance_stats
        assert initial_stats["total_analyses"] == 0
        assert initial_stats["avg_analysis_time_ms"] == 0
        
        # Update metrics
        analyzer._update_performance_metrics(500)
        analyzer._update_performance_metrics(750)
        
        updated_stats = analyzer.performance_stats
        assert updated_stats["total_analyses"] == 2
        assert updated_stats["avg_analysis_time_ms"] == 625  # (500 + 750) / 2
        assert updated_stats["max_analysis_time_ms"] == 750
    
    @pytest.mark.asyncio
    async def test_pattern_confluence_detection(self, analyzer, test_timeframe_data):
        """Test pattern confluence detection across timeframes."""
        result = await analyzer.analyze(
            symbol="BTC",
            timeframe_data=test_timeframe_data
        )
        
        confluence = result.confluence_analysis
        assert confluence.primary_timeframe == Timeframe.FIVE_MINUTES
        assert confluence.confluence_score >= Decimal('0')
        assert confluence.weighted_confidence >= Decimal('0')
    
    def test_analyzer_cleanup(self, analyzer):
        """Test proper cleanup of resources."""
        # Analyzer should have thread pool
        assert hasattr(analyzer, '_thread_pool')
        
        # Deletion should cleanup resources
        del analyzer
        # Test passes if no exceptions are raised


class TestIntegrationScenarios:
    """Integration tests for complete analysis scenarios."""
    
    @pytest.mark.asyncio
    async def test_bullish_confluence_scenario(self):
        """Test scenario with strong bullish confluence across timeframes."""
        analyzer = TimeframeAnalyzer()
        base_time = datetime.now(timezone.utc)
        
        # Create strongly bullish data across all timeframes
        bullish_data = {}
        
        for timeframe, minutes in [(Timeframe.ONE_MINUTE, 1), (Timeframe.FIVE_MINUTES, 5), (Timeframe.FIFTEEN_MINUTES, 15)]:
            candles = []
            for i in range(10):
                # Strong uptrend with higher highs and higher lows
                candles.append(CandlestickData(
                    symbol="BTC",
                    timeframe=timeframe,
                    timestamp=base_time + timedelta(minutes=i*minutes),
                    open=Decimal(f'{50000 + i*50}'),    # Rising opens
                    high=Decimal(f'{50200 + i*50}'),    # Rising highs
                    low=Decimal(f'{49900 + i*40}'),     # Rising lows
                    close=Decimal(f'{50150 + i*50}'),   # Rising closes
                    volume=Decimal('1000')
                ))
            bullish_data[timeframe] = candles
        
        result = await analyzer.analyze("BTC", bullish_data)
        
        # Should detect bullish trend alignment
        assert result.trend_alignment.dominant_trend in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        assert result.trading_recommendation in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
    
    @pytest.mark.asyncio
    async def test_conflicting_timeframes_scenario(self):
        """Test scenario with conflicting trends across timeframes."""
        analyzer = TimeframeAnalyzer()
        base_time = datetime.now(timezone.utc)
        
        # Create conflicting data
        conflicting_data = {}
        
        # 1m: Bearish
        conflicting_data[Timeframe.ONE_MINUTE] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.ONE_MINUTE,
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(f'{50000 - i*10}'),    # Falling
                high=Decimal(f'{50050 - i*10}'),
                low=Decimal(f'{49900 - i*10}'),
                close=Decimal(f'{49950 - i*10}'),
                volume=Decimal('1000')
            )
            for i in range(10)
        ]
        
        # 5m: Bullish
        conflicting_data[Timeframe.FIVE_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=base_time + timedelta(minutes=i*5),
                open=Decimal(f'{50000 + i*20}'),    # Rising
                high=Decimal(f'{50100 + i*20}'),
                low=Decimal(f'{49950 + i*15}'),
                close=Decimal(f'{50080 + i*20}'),
                volume=Decimal('5000')
            )
            for i in range(5)
        ]
        
        # 15m: Neutral
        conflicting_data[Timeframe.FIFTEEN_MINUTES] = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.FIFTEEN_MINUTES,
                timestamp=base_time + timedelta(minutes=i*15),
                open=Decimal('50000'),  # Sideways
                high=Decimal('50050'),
                low=Decimal('49950'),
                close=Decimal('50000'),
                volume=Decimal('15000')
            )
            for i in range(3)
        ]
        
        result = await analyzer.analyze("BTC", conflicting_data)
        
        # Should detect conflict and lower alignment score
        assert result.trend_alignment.alignment_score < Decimal('80')
        assert len(result.trend_alignment.conflicting_timeframes) > 0
        assert result.trading_recommendation == SignalDirection.HOLD  # Should be cautious
    
    @pytest.mark.asyncio
    async def test_performance_requirement_compliance(self):
        """Test that analysis meets <2s latency requirement."""
        analyzer = TimeframeAnalyzer()
        
        # Create realistic amount of data
        base_time = datetime.now(timezone.utc)
        large_dataset = {}
        
        for timeframe, minutes in [(Timeframe.ONE_MINUTE, 1), (Timeframe.FIVE_MINUTES, 5), (Timeframe.FIFTEEN_MINUTES, 15)]:
            candles = []
            for i in range(100):  # Larger dataset
                candles.append(CandlestickData(
                    symbol="BTC",
                    timeframe=timeframe,
                    timestamp=base_time + timedelta(minutes=i*minutes),
                    open=Decimal(f'{50000 + (i % 10) * 50}'),
                    high=Decimal(f'{50100 + (i % 10) * 50}'),
                    low=Decimal(f'{49900 + (i % 10) * 50}'),
                    close=Decimal(f'{50050 + (i % 10) * 50}'),
                    volume=Decimal('1000')
                ))
            large_dataset[timeframe] = candles
        
        # Measure actual performance
        import time
        start_time = time.time()
        result = await analyzer.analyze("BTC", large_dataset)
        end_time = time.time()
        
        actual_duration = (end_time - start_time) * 1000  # Convert to ms
        
        # Should meet latency requirement
        assert actual_duration < 2000  # Under 2 seconds
        assert result.meets_latency_requirement is True
        assert result.analysis_duration_ms < 2000


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 