"""
Multi-Timeframe Analysis Engine

This module implements comprehensive multi-timeframe analysis for candlestick patterns,
providing pattern confluence detection, trend alignment analysis, and signal confirmation
across multiple timeframes.

Key features:
- Synchronizes data across 1m, 5m, 15m timeframes  
- Detects pattern confluence with priority weighting (15m > 5m > 1m)
- Analyzes trend alignment between timeframes
- Provides pattern confirmation logic across timeframes
- Optimized for <2s analysis time for real-time trading
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, computed_field, ConfigDict
from enum import Enum
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from ..models.market_data import CandlestickData, Timeframe
from ..models.signals import (
    CandlestickPattern, 
    PatternType, 
    SignalDirection, 
    SignalType,
    AnalysisContext,
    TradingSignal
)
from .candlestick_models import (
    PatternStrength,
    TimeframePriority,
    PatternConfluence,
    VolumeProfile,
    PatternQuality,
    MultiTimeframePattern,
    StrategyConfiguration
)
from .patterns.single_candlestick import SinglePatternRecognizer
from .patterns.multi_candlestick import MultiPatternRecognizer


class TimeframeSynchronization(BaseModel):
    """
    Handles synchronization of data across multiple timeframes.
    
    Ensures that data from different timeframes is properly aligned
    for confluence analysis and pattern detection.
    """
    
    base_timestamp: datetime = Field(
        ...,
        description="Base timestamp for synchronization"
    )
    timeframe_data: Dict[Timeframe, List[CandlestickData]] = Field(
        ...,
        description="Candlestick data organized by timeframe"
    )
    sync_window_minutes: int = Field(
        default=60,
        description="Synchronization window in minutes",
        ge=1,
        le=1440
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def synchronized_timeframes(self) -> List[Timeframe]:
        """Get list of available synchronized timeframes."""
        return list(self.timeframe_data.keys())
    
    @computed_field
    @property
    def data_completeness(self) -> Dict[Timeframe, Decimal]:
        """Calculate data completeness percentage for each timeframe."""
        completeness = {}
        
        for timeframe, candles in self.timeframe_data.items():
            expected_count = self._calculate_expected_candles(timeframe)
            actual_count = len(candles)
            completeness[timeframe] = Decimal(min(100, (actual_count / expected_count) * 100))
        
        return completeness
    
    def _calculate_expected_candles(self, timeframe: Timeframe) -> int:
        """Calculate expected number of candles for the sync window."""
        timeframe_minutes = {
            Timeframe.ONE_MINUTE: 1,
            Timeframe.FIVE_MINUTES: 5,
            Timeframe.FIFTEEN_MINUTES: 15,
            Timeframe.ONE_HOUR: 60,
            Timeframe.FOUR_HOURS: 240,
            Timeframe.ONE_DAY: 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 5)
        return max(1, self.sync_window_minutes // minutes)
    
    def get_latest_candles(self, timeframe: Timeframe, count: int = 1) -> List[CandlestickData]:
        """Get the latest N candles for a specific timeframe."""
        candles = self.timeframe_data.get(timeframe, [])
        return candles[-count:] if candles else []
    
    def get_candles_in_window(self, timeframe: Timeframe, start_time: datetime, end_time: datetime) -> List[CandlestickData]:
        """Get candles within a specific time window."""
        candles = self.timeframe_data.get(timeframe, [])
        
        filtered_candles = [
            candle for candle in candles
            if start_time <= candle.timestamp <= end_time
        ]
        
        return sorted(filtered_candles, key=lambda c: c.timestamp)


class TrendAlignment(BaseModel):
    """
    Analyzes trend alignment across multiple timeframes.
    
    Determines if trends are aligned across timeframes and provides
    confidence scores for trend direction consistency.
    """
    
    timeframe_trends: Dict[Timeframe, SignalDirection] = Field(
        ...,
        description="Trend direction for each timeframe"
    )
    alignment_score: Decimal = Field(
        ...,
        description="Overall trend alignment score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    dominant_trend: SignalDirection = Field(
        ...,
        description="Dominant trend across timeframes"
    )
    trend_strength: PatternStrength = Field(
        ...,
        description="Strength of the trend alignment"
    )
    conflicting_timeframes: List[Timeframe] = Field(
        default_factory=list,
        description="Timeframes with conflicting trends"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def is_aligned(self) -> bool:
        """Check if trends are well aligned across timeframes."""
        return self.alignment_score >= Decimal('70')
    
    @computed_field
    @property
    def bullish_timeframes(self) -> List[Timeframe]:
        """Get timeframes showing bullish trends."""
        return [
            tf for tf, trend in self.timeframe_trends.items()
            if trend in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        ]
    
    @computed_field
    @property
    def bearish_timeframes(self) -> List[Timeframe]:
        """Get timeframes showing bearish trends."""
        return [
            tf for tf, trend in self.timeframe_trends.items()
            if trend in [SignalDirection.SELL, SignalDirection.STRONG_SELL]
        ]


class ConfluenceAnalysis(BaseModel):
    """
    Analyzes pattern confluence across multiple timeframes.
    
    Identifies when similar patterns appear across different timeframes
    and calculates confluence strength based on pattern alignment.
    """
    
    primary_timeframe: Timeframe = Field(
        ...,
        description="Primary timeframe for analysis"
    )
    patterns_by_timeframe: Dict[Timeframe, List[CandlestickPattern]] = Field(
        ...,
        description="Detected patterns organized by timeframe"
    )
    confluence_patterns: List[CandlestickPattern] = Field(
        ...,
        description="Patterns that appear across multiple timeframes"
    )
    confluence_score: Decimal = Field(
        ...,
        description="Overall confluence strength (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    weighted_confidence: Decimal = Field(
        ...,
        description="Timeframe-weighted confidence score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def has_strong_confluence(self) -> bool:
        """Check if confluence is strong enough for trading."""
        return (
            self.confluence_score >= Decimal('60') and
            len(self.confluence_patterns) >= 2 and
            self.weighted_confidence >= Decimal('65')
        )
    
    @computed_field
    @property
    def pattern_type_distribution(self) -> Dict[PatternType, int]:
        """Get distribution of pattern types across timeframes."""
        distribution = defaultdict(int)
        
        for patterns in self.patterns_by_timeframe.values():
            for pattern in patterns:
                distribution[pattern.pattern_type] += 1
        
        return dict(distribution)
    
    @computed_field
    @property
    def dominant_signal_direction(self) -> Optional[SignalDirection]:
        """Determine dominant signal direction from confluence."""
        if not self.confluence_patterns:
            return None
        
        bullish_weight = Decimal('0')
        bearish_weight = Decimal('0')
        
        for pattern in self.confluence_patterns:
            if pattern.bullish_probability > pattern.bearish_probability:
                bullish_weight += pattern.confidence
            else:
                bearish_weight += pattern.confidence
        
        if bullish_weight > bearish_weight * Decimal('1.2'):
            return SignalDirection.BUY
        elif bearish_weight > bullish_weight * Decimal('1.2'):
            return SignalDirection.SELL
        else:
            return SignalDirection.HOLD


class TimeframeAnalysisResult(BaseModel):
    """
    Complete result of multi-timeframe analysis.
    
    Contains all analysis components including synchronization,
    pattern detection, confluence analysis, and trend alignment.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol analyzed"
    )
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the analysis was performed"
    )
    
    # Core analysis components
    synchronization: TimeframeSynchronization = Field(
        ...,
        description="Timeframe data synchronization results"
    )
    trend_alignment: TrendAlignment = Field(
        ...,
        description="Trend alignment analysis across timeframes"
    )
    confluence_analysis: ConfluenceAnalysis = Field(
        ...,
        description="Pattern confluence analysis"
    )
    
    # Pattern analysis by timeframe
    single_patterns: Dict[Timeframe, List[CandlestickPattern]] = Field(
        ...,
        description="Single candlestick patterns by timeframe"
    )
    multi_patterns: Dict[Timeframe, List[CandlestickPattern]] = Field(
        ...,
        description="Multi-candlestick patterns by timeframe"
    )
    
    # Performance metrics
    analysis_duration_ms: int = Field(
        ...,
        description="Analysis execution time in milliseconds",
        ge=0
    )
    data_quality_score: Decimal = Field(
        ...,
        description="Overall data quality score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def meets_latency_requirement(self) -> bool:
        """Check if analysis meets <2s latency requirement."""
        return self.analysis_duration_ms < 2000
    
    @computed_field
    @property
    def total_patterns_detected(self) -> int:
        """Get total number of patterns detected across all timeframes."""
        single_count = sum(len(patterns) for patterns in self.single_patterns.values())
        multi_count = sum(len(patterns) for patterns in self.multi_patterns.values())
        return single_count + multi_count
    
    @computed_field
    @property
    def best_timeframe_pattern(self) -> Optional[CandlestickPattern]:
        """Get the highest confidence pattern across all timeframes."""
        all_patterns = []
        
        for patterns in self.single_patterns.values():
            all_patterns.extend(patterns)
        for patterns in self.multi_patterns.values():
            all_patterns.extend(patterns)
        
        if not all_patterns:
            return None
        
        return max(all_patterns, key=lambda p: p.confidence)
    
    @computed_field
    @property
    def trading_recommendation(self) -> SignalDirection:
        """Get overall trading recommendation based on all analysis."""
        trend_direction = self.trend_alignment.dominant_trend
        
        # When patterns are detected, use comprehensive analysis
        if self.total_patterns_detected > 0:
            # Weight factors for different components
            confluence_weight = Decimal('0.4')
            trend_weight = Decimal('0.3')
            pattern_weight = Decimal('0.3')
            
            confluence_direction = self.confluence_analysis.dominant_signal_direction or SignalDirection.HOLD
            
            # Calculate pattern direction from best pattern
            best_pattern = self.best_timeframe_pattern
            pattern_direction = SignalDirection.HOLD
            if best_pattern:
                if best_pattern.bullish_probability > best_pattern.bearish_probability:
                    pattern_direction = SignalDirection.BUY
                elif best_pattern.bearish_probability > best_pattern.bullish_probability:
                    pattern_direction = SignalDirection.SELL
            
            # Score each direction
            buy_score = Decimal('0')
            sell_score = Decimal('0')
            
            # Confluence scoring
            if confluence_direction == SignalDirection.BUY:
                buy_score += confluence_weight * self.confluence_analysis.confluence_score
            elif confluence_direction == SignalDirection.SELL:
                sell_score += confluence_weight * self.confluence_analysis.confluence_score
            
            # Trend scoring
            if trend_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                buy_score += trend_weight * self.trend_alignment.alignment_score
            elif trend_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                sell_score += trend_weight * self.trend_alignment.alignment_score
            
            # Pattern scoring
            if pattern_direction == SignalDirection.BUY and best_pattern:
                buy_score += pattern_weight * best_pattern.confidence
            elif pattern_direction == SignalDirection.SELL and best_pattern:
                sell_score += pattern_weight * best_pattern.confidence
            
            # Determine final recommendation
            if buy_score > sell_score * Decimal('1.1'):
                if buy_score > 60:
                    return SignalDirection.STRONG_BUY
                elif buy_score > 30:
                    return SignalDirection.BUY
                else:
                    return SignalDirection.HOLD
            elif sell_score > buy_score * Decimal('1.1'):
                if sell_score > 60:
                    return SignalDirection.STRONG_SELL
                elif sell_score > 30:
                    return SignalDirection.SELL
                else:
                    return SignalDirection.HOLD
            else:
                return SignalDirection.HOLD
        
        else:
            # When no patterns detected, rely more heavily on trend analysis
            alignment_score = self.trend_alignment.alignment_score
            
            # Strong trend alignment with high confidence
            if alignment_score > 80 and self.trend_alignment.is_aligned:
                if trend_direction == SignalDirection.STRONG_BUY:
                    return SignalDirection.STRONG_BUY
                elif trend_direction == SignalDirection.BUY:
                    return SignalDirection.BUY
                elif trend_direction == SignalDirection.STRONG_SELL:
                    return SignalDirection.STRONG_SELL
                elif trend_direction == SignalDirection.SELL:
                    return SignalDirection.SELL
            
            # Moderate trend alignment
            elif alignment_score > 60 and self.trend_alignment.is_aligned:
                if trend_direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY]:
                    return SignalDirection.BUY
                elif trend_direction in [SignalDirection.STRONG_SELL, SignalDirection.SELL]:
                    return SignalDirection.SELL
            
            # Default to hold for weak signals
            return SignalDirection.HOLD
    

    
    def to_multi_timeframe_pattern(self, 
                                  entry_price: Decimal,
                                  stop_loss: Optional[Decimal] = None,
                                  take_profit: Optional[Decimal] = None) -> MultiTimeframePattern:
        """Convert analysis result to MultiTimeframePattern for signal generation."""
        # Get best pattern from confluence analysis
        best_pattern = self.confluence_analysis.confluence_patterns[0] if self.confluence_analysis.confluence_patterns else None
        
        if not best_pattern:
            # Fallback to best overall pattern
            best_pattern = self.best_timeframe_pattern
        
        if not best_pattern:
            # Create a synthetic pattern if none found
            best_pattern = CandlestickPattern(
                pattern_type=PatternType.DOJI,
                confidence=Decimal('50'),
                reliability=Decimal('0.5'),
                bullish_probability=Decimal('0.5'),
                bearish_probability=Decimal('0.5'),
                timestamp=self.analysis_timestamp,
                timeframe=self.confluence_analysis.primary_timeframe,
                metadata={"synthetic": True}
            )
        
        # Create confluence based on analysis
        confluence = PatternConfluence(
            primary_pattern=best_pattern,
            supporting_patterns=self.confluence_analysis.confluence_patterns[1:] if len(self.confluence_analysis.confluence_patterns) > 1 else [],
            conflicting_patterns=[],  # Would need to analyze conflicting patterns
            confluence_score=self.confluence_analysis.confluence_score
        )
        
        # Create volume profile (would need volume data)
        volume_profile = VolumeProfile(
            pattern_volume=Decimal('1000'),  # Would calculate from actual data
            average_volume=Decimal('1000'),
            volume_ratio=Decimal('1.0'),
            volume_trend=SignalDirection.HOLD
        )
        
        # Create quality assessment
        quality = PatternQuality(
            technical_score=best_pattern.confidence,
            volume_score=Decimal('70'),  # Would calculate from volume analysis
            context_score=self.trend_alignment.alignment_score,
            overall_score=(best_pattern.confidence + self.trend_alignment.alignment_score) / Decimal('2')
        )
        
        # Create market context
        market_context = AnalysisContext(
            current_price=entry_price,
            volume_24h=Decimal('1000000'),  # Would get from market data
            price_change_24h=Decimal('0'),
            trend_direction=self.trend_alignment.dominant_trend,
            volatility_score=Decimal('50'),
            market_session="continuous",
            timeframe=self.confluence_analysis.primary_timeframe
        )
        
        # Calculate risk/reward
        risk_reward_ratio = None
        if stop_loss and take_profit:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0:
                risk_reward_ratio = reward / risk
        
        return MultiTimeframePattern(
            symbol=self.symbol,
            primary_timeframe=self.confluence_analysis.primary_timeframe,
            analysis_time=self.analysis_timestamp,
            primary_pattern=best_pattern,
            confluence=confluence,
            volume_profile=volume_profile,
            quality=quality,
            market_context=market_context,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio
        )


class TimeframeAnalyzer:
    """
    Main class for multi-timeframe candlestick pattern analysis.
    
    Coordinates the complete analysis workflow including data synchronization,
    pattern detection across timeframes, confluence analysis, trend alignment,
    and final signal generation with proper latency optimization.
    """
    
    def __init__(self, config: Optional[StrategyConfiguration] = None):
        """
        Initialize the TimeframeAnalyzer.
        
        Args:
            config: Strategy configuration. If None, uses default configuration.
        """
        self.config = config or StrategyConfiguration()
        
        # Initialize pattern recognizers with configuration
        self.single_recognizer = SinglePatternRecognizer(
            min_confidence=self.config.min_pattern_confidence
        )
        self.multi_recognizer = MultiPatternRecognizer(
            min_confidence=self.config.min_pattern_confidence
        )
        
        # Performance tracking
        self._analysis_count = 0
        self._total_analysis_time = 0.0
        self._max_analysis_time = 0.0
        
        # Thread pool for parallel pattern analysis
        self._thread_pool = ThreadPoolExecutor(max_workers=3)
    
    async def analyze(self, 
                     symbol: str,
                     timeframe_data: Dict[Timeframe, List[CandlestickData]],
                     primary_timeframe: Optional[Timeframe] = None) -> TimeframeAnalysisResult:
        """
        Perform comprehensive multi-timeframe analysis.
        
        Args:
            symbol: Trading symbol to analyze
            timeframe_data: Candlestick data organized by timeframe
            primary_timeframe: Primary timeframe for analysis (uses config default if None)
            
        Returns:
            Complete analysis result with all components
        """
        start_time = time.time()
        
        # Use primary timeframe from config if not specified
        if primary_timeframe is None:
            primary_timeframe = self.config.primary_timeframe
        
        # Ensure primary timeframe is in the data
        if primary_timeframe not in timeframe_data:
            raise ValueError(f"Primary timeframe {primary_timeframe} not found in provided data")
        
        # Create synchronization
        sync = TimeframeSynchronization(
            base_timestamp=datetime.now(timezone.utc),
            timeframe_data=timeframe_data,
            sync_window_minutes=self.config.volume_lookback_periods
        )
        
        # Perform parallel pattern analysis for each timeframe
        tasks = []
        for timeframe in self.config.timeframes:
            if timeframe in timeframe_data and timeframe_data[timeframe]:
                task = asyncio.create_task(
                    self._analyze_timeframe_patterns(timeframe, timeframe_data[timeframe])
                )
                tasks.append((timeframe, task))
        
        # Wait for all pattern analysis to complete
        pattern_results = {}
        for timeframe, task in tasks:
            single_patterns, multi_patterns = await task
            pattern_results[timeframe] = {
                'single': single_patterns,
                'multi': multi_patterns
            }
        
        # Extract patterns by type
        single_patterns = {tf: results['single'] for tf, results in pattern_results.items()}
        multi_patterns = {tf: results['multi'] for tf, results in pattern_results.items()}
        
        # Perform trend alignment analysis
        trend_alignment = self._analyze_trend_alignment(timeframe_data)
        
        # Perform confluence analysis
        confluence_analysis = self._analyze_pattern_confluence(
            primary_timeframe, single_patterns, multi_patterns
        )
        
        # Calculate data quality score
        data_quality = self._calculate_data_quality(sync)
        
        # Record performance metrics
        end_time = time.time()
        analysis_duration = int((end_time - start_time) * 1000)  # Convert to milliseconds
        
        self._update_performance_metrics(analysis_duration)
        
        return TimeframeAnalysisResult(
            symbol=symbol,
            analysis_timestamp=datetime.now(timezone.utc),
            synchronization=sync,
            trend_alignment=trend_alignment,
            confluence_analysis=confluence_analysis,
            single_patterns=single_patterns,
            multi_patterns=multi_patterns,
            analysis_duration_ms=analysis_duration,
            data_quality_score=data_quality
        )
    
    async def _analyze_timeframe_patterns(self, 
                                        timeframe: Timeframe, 
                                        candles: List[CandlestickData]) -> Tuple[List[CandlestickPattern], List[CandlestickPattern]]:
        """
        Analyze patterns for a specific timeframe.
        
        Args:
            timeframe: Timeframe being analyzed
            candles: Candlestick data for the timeframe
            
        Returns:
            Tuple of (single_patterns, multi_patterns)
        """
        if not candles:
            return [], []
        
        # Create volume profile if we have enough data
        volume_profile = self._create_volume_profile(candles) if len(candles) >= 3 else None
        
        # Run pattern analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Analyze single patterns on latest candle
        single_task = loop.run_in_executor(
            self._thread_pool,
            self.single_recognizer.analyze,
            candles[-1],
            volume_profile
        )
        
        # Analyze multi patterns on recent candles
        multi_task = loop.run_in_executor(
            self._thread_pool,
            self.multi_recognizer.analyze,
            candles[-5:] if len(candles) >= 5 else candles,  # Use last 5 candles or all available
            volume_profile
        )
        
        # Wait for both analyses to complete
        single_patterns, multi_patterns = await asyncio.gather(single_task, multi_task)
        
        return single_patterns, multi_patterns
    
    def _create_volume_profile(self, candles: List[CandlestickData]) -> VolumeProfile:
        """Create volume profile from candlestick data."""
        if len(candles) < 3:
            # Not enough data for meaningful volume analysis
            return VolumeProfile(
                pattern_volume=Decimal('1000'),
                average_volume=Decimal('1000'),
                volume_ratio=Decimal('1.0'),
                volume_trend=SignalDirection.HOLD
            )
        
        # Calculate recent and average volumes
        recent_candles = candles[-3:]
        pattern_volume = sum(candle.volume for candle in recent_candles)
        
        lookback_periods = min(self.config.volume_lookback_periods, len(candles))
        avg_volume = sum(candle.volume for candle in candles[-lookback_periods:]) / lookback_periods
        
        volume_ratio = pattern_volume / (avg_volume * 3) if avg_volume > 0 else Decimal('1.0')
        
        # Determine volume trend
        if len(candles) >= 6:
            recent_avg = sum(candle.volume for candle in candles[-3:]) / 3
            older_avg = sum(candle.volume for candle in candles[-6:-3]) / 3
            
            if recent_avg > older_avg * Decimal('1.2'):
                volume_trend = SignalDirection.BUY
            elif recent_avg < older_avg * Decimal('0.8'):
                volume_trend = SignalDirection.SELL
            else:
                volume_trend = SignalDirection.HOLD
        else:
            volume_trend = SignalDirection.HOLD
        
        return VolumeProfile(
            pattern_volume=pattern_volume,
            average_volume=avg_volume * 3,  # For 3-candle comparison
            volume_ratio=volume_ratio,
            volume_trend=volume_trend
        )
    
    def _analyze_trend_alignment(self, timeframe_data: Dict[Timeframe, List[CandlestickData]]) -> TrendAlignment:
        """Analyze trend alignment across timeframes."""
        timeframe_trends = {}
        priority_weights = {}
        
        for timeframe, candles in timeframe_data.items():
            if not candles or len(candles) < 5:
                timeframe_trends[timeframe] = SignalDirection.HOLD
                continue
            
            # Simple trend analysis using recent price action
            recent_candles = candles[-5:]
            
            # Calculate price momentum
            start_price = recent_candles[0].close
            end_price = recent_candles[-1].close
            price_change = (end_price - start_price) / start_price * 100
            
            # Calculate higher highs and lower lows
            highs = [candle.high for candle in recent_candles]
            lows = [candle.low for candle in recent_candles]
            
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            
            # Determine trend (more sensitive thresholds)
            if price_change > 0.5 and higher_highs >= 3:
                timeframe_trends[timeframe] = SignalDirection.STRONG_BUY
            elif price_change > 0.1 and higher_highs >= 2:
                timeframe_trends[timeframe] = SignalDirection.BUY
            elif price_change < -0.5 and lower_lows >= 3:
                timeframe_trends[timeframe] = SignalDirection.STRONG_SELL
            elif price_change < -0.1 and lower_lows >= 2:
                timeframe_trends[timeframe] = SignalDirection.SELL
            else:
                timeframe_trends[timeframe] = SignalDirection.HOLD
            
            # Get priority weight for this timeframe
            priority_weights[timeframe] = getattr(TimeframePriority, timeframe.name, 1)
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(timeframe_trends, priority_weights)
        
        # Determine dominant trend
        dominant_trend = self._calculate_dominant_trend(timeframe_trends, priority_weights)
        
        # Find conflicting timeframes (more comprehensive detection)
        conflicting_timeframes = []
        bullish_trends = [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        bearish_trends = [SignalDirection.SELL, SignalDirection.STRONG_SELL]
        
        # Check for conflicts between bullish and bearish trends
        has_bullish = any(trend in bullish_trends for trend in timeframe_trends.values())
        has_bearish = any(trend in bearish_trends for trend in timeframe_trends.values())
        
        if has_bullish and has_bearish:
            # Add all non-HOLD timeframes as potentially conflicting
            for timeframe, trend in timeframe_trends.items():
                if trend != SignalDirection.HOLD and trend != dominant_trend:
                    conflicting_timeframes.append(timeframe)
        else:
            # Original logic for less severe conflicts
            for timeframe, trend in timeframe_trends.items():
                if trend != dominant_trend and trend != SignalDirection.HOLD:
                    conflicting_timeframes.append(timeframe)
        
        # Determine trend strength
        trend_strength = PatternStrength.from_confidence(alignment_score)
        
        return TrendAlignment(
            timeframe_trends=timeframe_trends,
            alignment_score=alignment_score,
            dominant_trend=dominant_trend,
            trend_strength=trend_strength,
            conflicting_timeframes=conflicting_timeframes
        )
    
    def _calculate_alignment_score(self, 
                                 timeframe_trends: Dict[Timeframe, SignalDirection],
                                 priority_weights: Dict[Timeframe, int]) -> Decimal:
        """Calculate trend alignment score based on timeframe priority weights."""
        if not timeframe_trends:
            return Decimal('0')
        
        # Find the most common trend (excluding HOLD)
        non_hold_trends = [trend for trend in timeframe_trends.values() if trend != SignalDirection.HOLD]
        if not non_hold_trends:
            return Decimal('50')  # Neutral when all are HOLD
        
        # Count trend occurrences with weights
        trend_weights = defaultdict(Decimal)
        total_weight = Decimal('0')
        
        for timeframe, trend in timeframe_trends.items():
            weight = Decimal(priority_weights.get(timeframe, 1))
            trend_weights[trend] += weight
            total_weight += weight
        
        # Find dominant trend
        dominant_trend = max(trend_weights.items(), key=lambda x: x[1])[0]
        dominant_weight = trend_weights[dominant_trend]
        
        # Calculate alignment as percentage of total weight supporting dominant trend
        alignment_percentage = (dominant_weight / total_weight) * 100 if total_weight > 0 else Decimal('0')
        
        return min(Decimal('100'), alignment_percentage)
    
    def _calculate_dominant_trend(self, 
                                timeframe_trends: Dict[Timeframe, SignalDirection],
                                priority_weights: Dict[Timeframe, int]) -> SignalDirection:
        """Calculate dominant trend based on weighted voting."""
        trend_weights = defaultdict(Decimal)
        
        for timeframe, trend in timeframe_trends.items():
            weight = Decimal(priority_weights.get(timeframe, 1))
            trend_weights[trend] += weight
        
        if not trend_weights:
            return SignalDirection.HOLD
        
        return max(trend_weights.items(), key=lambda x: x[1])[0]
    
    def _analyze_pattern_confluence(self, 
                                  primary_timeframe: Timeframe,
                                  single_patterns: Dict[Timeframe, List[CandlestickPattern]],
                                  multi_patterns: Dict[Timeframe, List[CandlestickPattern]]) -> ConfluenceAnalysis:
        """Analyze pattern confluence across timeframes."""
        # Combine all patterns by timeframe
        patterns_by_timeframe = {}
        for timeframe in self.config.timeframes:
            combined_patterns = []
            combined_patterns.extend(single_patterns.get(timeframe, []))
            combined_patterns.extend(multi_patterns.get(timeframe, []))
            patterns_by_timeframe[timeframe] = combined_patterns
        
        # Find confluence patterns (patterns that appear in multiple timeframes)
        confluence_patterns = self._find_confluence_patterns(patterns_by_timeframe)
        
        # Calculate confluence score
        confluence_score = self._calculate_confluence_score(patterns_by_timeframe, confluence_patterns)
        
        # Calculate weighted confidence based on timeframe priority
        weighted_confidence = self._calculate_weighted_confidence(patterns_by_timeframe)
        
        return ConfluenceAnalysis(
            primary_timeframe=primary_timeframe,
            patterns_by_timeframe=patterns_by_timeframe,
            confluence_patterns=confluence_patterns,
            confluence_score=confluence_score,
            weighted_confidence=weighted_confidence
        )
    
    def _find_confluence_patterns(self, patterns_by_timeframe: Dict[Timeframe, List[CandlestickPattern]]) -> List[CandlestickPattern]:
        """Find patterns that appear across multiple timeframes."""
        # Group patterns by type
        pattern_groups = defaultdict(list)
        
        for timeframe, patterns in patterns_by_timeframe.items():
            for pattern in patterns:
                pattern_groups[pattern.pattern_type].append((timeframe, pattern))
        
        # Find patterns that appear in multiple timeframes
        confluence_patterns = []
        for pattern_type, timeframe_patterns in pattern_groups.items():
            if len(timeframe_patterns) >= 2:  # At least 2 timeframes
                # Choose the highest confidence pattern of this type
                best_pattern = max(timeframe_patterns, key=lambda x: x[1].confidence)[1]
                confluence_patterns.append(best_pattern)
        
        # Sort by confidence
        confluence_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return confluence_patterns
    
    def _calculate_confluence_score(self, 
                                  patterns_by_timeframe: Dict[Timeframe, List[CandlestickPattern]],
                                  confluence_patterns: List[CandlestickPattern]) -> Decimal:
        """Calculate overall confluence strength score."""
        if not confluence_patterns:
            return Decimal('0')
        
        # Base score from number of confluence patterns
        base_score = min(Decimal('40'), Decimal(len(confluence_patterns)) * Decimal('20'))
        
        # Confidence bonus from pattern quality
        if confluence_patterns:
            avg_confidence = sum(p.confidence for p in confluence_patterns) / len(confluence_patterns)
            confidence_bonus = (avg_confidence - Decimal('60')) / Decimal('2') if avg_confidence > 60 else Decimal('0')
        else:
            confidence_bonus = Decimal('0')
        
        # Timeframe coverage bonus
        total_timeframes = len(self.config.timeframes)
        timeframes_with_patterns = len([tf for tf, patterns in patterns_by_timeframe.items() if patterns])
        coverage_bonus = (Decimal(timeframes_with_patterns) / Decimal(total_timeframes)) * Decimal('20')
        
        confluence_score = base_score + confidence_bonus + coverage_bonus
        return min(Decimal('100'), confluence_score)
    
    def _calculate_weighted_confidence(self, patterns_by_timeframe: Dict[Timeframe, List[CandlestickPattern]]) -> Decimal:
        """Calculate weighted confidence based on timeframe priority."""
        weighted_sum = Decimal('0')
        total_weight = Decimal('0')
        
        for timeframe, patterns in patterns_by_timeframe.items():
            if not patterns:
                continue
            
            # Get timeframe priority weight
            priority = getattr(TimeframePriority, timeframe.name, 1)
            weight = Decimal(priority)
            
            # Use highest confidence pattern for this timeframe
            max_confidence = max(p.confidence for p in patterns)
            
            weighted_sum += max_confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else Decimal('0')
    
    def _calculate_data_quality(self, sync: TimeframeSynchronization) -> Decimal:
        """Calculate overall data quality score."""
        completeness_scores = list(sync.data_completeness.values())
        
        if not completeness_scores:
            return Decimal('0')
        
        # Average completeness across all timeframes
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        
        # Penalty for missing timeframes
        expected_timeframes = len(self.config.timeframes)
        actual_timeframes = len(sync.synchronized_timeframes)
        completeness_penalty = Decimal('10') * (expected_timeframes - actual_timeframes)
        
        quality_score = avg_completeness - completeness_penalty
        return max(Decimal('0'), min(Decimal('100'), quality_score))
    
    def _update_performance_metrics(self, analysis_duration_ms: int):
        """Update internal performance tracking metrics."""
        self._analysis_count += 1
        self._total_analysis_time += analysis_duration_ms
        self._max_analysis_time = max(self._max_analysis_time, analysis_duration_ms)
    
    @property
    def performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the analyzer."""
        if self._analysis_count == 0:
            return {
                "total_analyses": 0,
                "avg_analysis_time_ms": 0,
                "max_analysis_time_ms": 0,
                "latency_compliance_rate": 0.0
            }
        
        avg_time = self._total_analysis_time / self._analysis_count
        
        return {
            "total_analyses": self._analysis_count,
            "avg_analysis_time_ms": int(avg_time),
            "max_analysis_time_ms": self._max_analysis_time,
            "latency_compliance_rate": 1.0 if self._max_analysis_time < 2000 else 0.8  # Estimated
        }
    
    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False) 