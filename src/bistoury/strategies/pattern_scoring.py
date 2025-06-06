"""
Pattern Strength and Confidence Scoring System

This module implements sophisticated scoring algorithms for evaluating
candlestick pattern reliability and trading confidence based on multiple
criteria including technical strength, volume confirmation, market context,
and historical success rates.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
import statistics
from dataclasses import dataclass

from pydantic import BaseModel, Field, computed_field, ConfigDict

from ..models.signals import CandlestickPattern, CandlestickData, Timeframe, SignalDirection, PatternType
from .candlestick_models import PatternStrength, VolumeProfile, PatternQuality


class MarketSession(str, Enum):
    """Market session classifications for context scoring."""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OFF_HOURS = "off_hours"


class VolatilityRegime(str, Enum):
    """Market volatility classifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class TrendStrength(str, Enum):
    """Trend strength classifications."""
    NO_TREND = "no_trend"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class PatternOutcome:
    """Historical pattern outcome for success rate tracking."""
    pattern_type: PatternType
    timeframe: Timeframe
    detection_time: datetime
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    success: Optional[bool] = None
    profit_loss_pct: Optional[Decimal] = None
    duration_hours: Optional[float] = None


class TechnicalScoring(BaseModel):
    """Technical criteria scoring for pattern strength."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    body_size_score: Decimal = Field(
        ...,
        description="Score based on candlestick body size (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    shadow_ratio_score: Decimal = Field(
        ...,
        description="Score based on shadow to body ratios (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    price_position_score: Decimal = Field(
        ...,
        description="Score based on price position within range (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    symmetry_score: Decimal = Field(
        ...,
        description="Score based on pattern symmetry (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    textbook_compliance: Decimal = Field(
        ...,
        description="How well pattern matches textbook definition (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    
    @computed_field
    @property
    def overall_technical_score(self) -> Decimal:
        """Calculate overall technical score."""
        scores = [
            self.body_size_score,
            self.shadow_ratio_score,
            self.price_position_score,
            self.symmetry_score,
            self.textbook_compliance
        ]
        return sum(scores) / len(scores)


class VolumeScoring(BaseModel):
    """Volume confirmation scoring for patterns."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    volume_spike_score: Decimal = Field(
        ...,
        description="Score based on volume spike during pattern (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    volume_trend_score: Decimal = Field(
        ...,
        description="Score based on volume trend alignment (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    breakout_volume_score: Decimal = Field(
        ...,
        description="Score based on breakout confirmation volume (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    relative_volume_score: Decimal = Field(
        ...,
        description="Score vs average volume (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    
    @computed_field
    @property
    def overall_volume_score(self) -> Decimal:
        """Calculate overall volume score."""
        scores = [
            self.volume_spike_score,
            self.volume_trend_score,
            self.breakout_volume_score,
            self.relative_volume_score
        ]
        return sum(scores) / len(scores)


class MarketContextScoring(BaseModel):
    """Market context scoring for environmental factors."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    trend_alignment_score: Decimal = Field(
        ...,
        description="Score based on alignment with broader trend (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    volatility_score: Decimal = Field(
        ...,
        description="Score based on current volatility regime (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    session_score: Decimal = Field(
        ...,
        description="Score based on market session timing (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    support_resistance_score: Decimal = Field(
        ...,
        description="Score based on proximity to key levels (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    momentum_score: Decimal = Field(
        ...,
        description="Score based on price momentum context (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    
    @computed_field
    @property
    def overall_context_score(self) -> Decimal:
        """Calculate overall context score."""
        scores = [
            self.trend_alignment_score,
            self.volatility_score,
            self.session_score,
            self.support_resistance_score,
            self.momentum_score
        ]
        return sum(scores) / len(scores)


class HistoricalPerformance(BaseModel):
    """Historical pattern performance metrics."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    pattern_type: PatternType = Field(..., description="Pattern type")
    timeframe: Timeframe = Field(..., description="Timeframe")
    total_occurrences: int = Field(default=0, description="Total pattern occurrences")
    successful_trades: int = Field(default=0, description="Number of successful trades")
    success_rate: Decimal = Field(
        default=Decimal('50'),
        description="Historical success rate percentage",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    average_profit_loss: Decimal = Field(
        default=Decimal('0'),
        description="Average profit/loss percentage"
    )
    average_duration_hours: Decimal = Field(
        default=Decimal('24'),
        description="Average holding duration in hours",
        ge=Decimal('0')
    )
    reliability_score: Decimal = Field(
        default=Decimal('50'),
        description="Overall reliability score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    
    @computed_field
    @property
    def confidence_multiplier(self) -> Decimal:
        """Get confidence multiplier based on historical performance."""
        base_multiplier = self.success_rate / Decimal('50')  # 50% is baseline
        
        # Adjust for sample size
        if self.total_occurrences >= 100:
            sample_bonus = Decimal('1.1')
        elif self.total_occurrences >= 50:
            sample_bonus = Decimal('1.05')
        elif self.total_occurrences >= 20:
            sample_bonus = Decimal('1.0')
        else:
            sample_bonus = Decimal('0.9')  # Penalty for small sample
        
        return base_multiplier * sample_bonus


class CompositePatternScore(BaseModel):
    """Composite pattern scoring combining all factors."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    pattern: CandlestickPattern = Field(..., description="Pattern being scored")
    technical_scoring: TechnicalScoring = Field(..., description="Technical criteria scores")
    volume_scoring: VolumeScoring = Field(..., description="Volume confirmation scores")
    context_scoring: MarketContextScoring = Field(..., description="Market context scores")
    historical_performance: HistoricalPerformance = Field(..., description="Historical performance metrics")
    
    # Weight factors for combining scores
    technical_weight: Decimal = Field(default=Decimal('0.3'), description="Technical scoring weight")
    volume_weight: Decimal = Field(default=Decimal('0.25'), description="Volume scoring weight")
    context_weight: Decimal = Field(default=Decimal('0.25'), description="Context scoring weight")
    historical_weight: Decimal = Field(default=Decimal('0.2'), description="Historical performance weight")
    
    scoring_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When scoring was performed"
    )
    
    @computed_field
    @property
    def weighted_confidence_score(self) -> Decimal:
        """Calculate weighted composite confidence score."""
        technical_score = self.technical_scoring.overall_technical_score
        volume_score = self.volume_scoring.overall_volume_score
        context_score = self.context_scoring.overall_context_score
        historical_score = self.historical_performance.reliability_score
        
        weighted_score = (
            technical_score * self.technical_weight +
            volume_score * self.volume_weight +
            context_score * self.context_weight +
            historical_score * self.historical_weight
        )
        
        # Apply historical multiplier
        final_score = weighted_score * self.historical_performance.confidence_multiplier
        
        return min(Decimal('100'), max(Decimal('0'), final_score))
    
    @computed_field
    @property
    def pattern_strength(self) -> PatternStrength:
        """Get pattern strength classification."""
        score = self.weighted_confidence_score
        return PatternStrength.from_confidence(score)
    
    @computed_field
    @property
    def is_tradeable(self) -> bool:
        """Determine if pattern meets minimum trading threshold."""
        return self.weighted_confidence_score >= Decimal('60')
    
    @computed_field
    @property
    def risk_adjusted_confidence(self) -> Decimal:
        """Get risk-adjusted confidence based on market conditions."""
        base_confidence = self.weighted_confidence_score
        
        # Reduce confidence in high volatility
        if self.context_scoring.volatility_score < 30:
            volatility_adjustment = Decimal('0.9')
        elif self.context_scoring.volatility_score > 80:
            volatility_adjustment = Decimal('0.85')
        else:
            volatility_adjustment = Decimal('1.0')
        
        # Reduce confidence if trend misaligned
        if self.context_scoring.trend_alignment_score < 40:
            trend_adjustment = Decimal('0.8')
        else:
            trend_adjustment = Decimal('1.0')
        
        adjusted = base_confidence * volatility_adjustment * trend_adjustment
        return min(Decimal('100'), max(Decimal('0'), adjusted))


class PatternScoringEngine:
    """Main engine for pattern strength and confidence scoring."""
    
    def __init__(self):
        """Initialize the scoring engine."""
        # Historical performance tracking
        self._pattern_history: Dict[Tuple[PatternType, Timeframe], List[PatternOutcome]] = defaultdict(list)
        self._performance_cache: Dict[Tuple[PatternType, Timeframe], HistoricalPerformance] = {}
        
        # Market context cache
        self._volatility_window = 20  # Periods for volatility calculation
        self._volume_window = 20  # Periods for volume analysis
        
    def score_pattern(self, 
                     pattern: CandlestickPattern,
                     market_data: List[CandlestickData],
                     volume_profile: Optional[VolumeProfile] = None) -> CompositePatternScore:
        """
        Score a pattern using all available criteria.
        
        Args:
            pattern: The candlestick pattern to score
            market_data: Historical market data for context
            volume_profile: Optional volume profile for enhanced scoring
            
        Returns:
            Complete composite scoring result
        """
        # Calculate technical scoring
        technical_scoring = self._calculate_technical_scoring(pattern)
        
        # Calculate volume scoring
        volume_scoring = self._calculate_volume_scoring(pattern, market_data, volume_profile)
        
        # Calculate market context scoring
        context_scoring = self._calculate_context_scoring(pattern, market_data)
        
        # Get historical performance
        historical_performance = self._get_historical_performance(pattern.pattern_type, pattern.timeframe)
        
        return CompositePatternScore(
            pattern=pattern,
            technical_scoring=technical_scoring,
            volume_scoring=volume_scoring,
            context_scoring=context_scoring,
            historical_performance=historical_performance
        )
    
    def _calculate_technical_scoring(self, pattern: CandlestickPattern) -> TechnicalScoring:
        """Calculate technical criteria scoring."""
        if not pattern.candles:
            return TechnicalScoring(
                body_size_score=Decimal('50'),
                shadow_ratio_score=Decimal('50'),
                price_position_score=Decimal('50'),
                symmetry_score=Decimal('50'),
                textbook_compliance=Decimal('50')
            )
        
        candle = pattern.candles[-1]  # Use last/most recent candle
        
        # Body size scoring
        body_size = abs(candle.close - candle.open)
        total_range = candle.high - candle.low
        body_ratio = body_size / total_range if total_range > 0 else Decimal('0')
        
        # Pattern-specific body size expectations
        if pattern.pattern_type in [PatternType.DOJI]:
            # Doji should have small body
            body_size_score = max(Decimal('0'), Decimal('100') - (body_ratio * Decimal('200')))
        elif pattern.pattern_type in [PatternType.MARUBOZU]:
            # Marubozu should have large body
            body_size_score = body_ratio * Decimal('100')
        else:
            # General patterns - moderate body size preferred
            if body_ratio < Decimal('0.3'):
                body_size_score = body_ratio * Decimal('200')  # Scale up small bodies
            else:
                body_size_score = Decimal('100') - ((body_ratio - Decimal('0.3')) * Decimal('100'))
        
        # Shadow ratio scoring
        upper_shadow = candle.high - max(candle.open, candle.close)
        lower_shadow = min(candle.open, candle.close) - candle.low
        
        if pattern.pattern_type == PatternType.HAMMER:
            # Hammer should have long lower shadow, short upper shadow
            lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else Decimal('0')
            upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else Decimal('0')
            shadow_ratio_score = (lower_shadow_ratio * Decimal('100')) - (upper_shadow_ratio * Decimal('50'))
        elif pattern.pattern_type == PatternType.SHOOTING_STAR:
            # Shooting star should have long upper shadow, short lower shadow
            upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else Decimal('0')
            lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else Decimal('0')
            shadow_ratio_score = (upper_shadow_ratio * Decimal('100')) - (lower_shadow_ratio * Decimal('50'))
        else:
            # General shadow scoring
            shadow_balance = abs(upper_shadow - lower_shadow) / total_range if total_range > 0 else Decimal('0')
            shadow_ratio_score = max(Decimal('0'), Decimal('100') - (shadow_balance * Decimal('100')))
        
        # Price position scoring (where close is relative to high/low)
        if total_range > 0:
            close_position = (candle.close - candle.low) / total_range
            if pattern.pattern_type in [PatternType.HAMMER]:
                # Hammer should close near high
                price_position_score = close_position * Decimal('100')
            elif pattern.pattern_type in [PatternType.SHOOTING_STAR]:
                # Shooting star should close near low
                price_position_score = (Decimal('1') - close_position) * Decimal('100')
            else:
                # General preference for middle positioning
                price_position_score = Decimal('100') - abs(close_position - Decimal('0.5')) * Decimal('200')
        else:
            price_position_score = Decimal('50')
        
        # Symmetry scoring (for multi-candle patterns)
        symmetry_score = Decimal('75')  # Default for single candle patterns
        
        # Textbook compliance (use existing confidence as base)
        textbook_compliance = pattern.confidence
        
        return TechnicalScoring(
            body_size_score=max(Decimal('0'), min(Decimal('100'), body_size_score)),
            shadow_ratio_score=max(Decimal('0'), min(Decimal('100'), shadow_ratio_score)),
            price_position_score=max(Decimal('0'), min(Decimal('100'), price_position_score)),
            symmetry_score=symmetry_score,
            textbook_compliance=textbook_compliance
        )
    
    def _calculate_volume_scoring(self, 
                                pattern: CandlestickPattern,
                                market_data: List[CandlestickData],
                                volume_profile: Optional[VolumeProfile]) -> VolumeScoring:
        """Calculate volume confirmation scoring."""
        if not market_data or len(market_data) < self._volume_window:
            return VolumeScoring(
                volume_spike_score=Decimal('50'),
                volume_trend_score=Decimal('50'),
                breakout_volume_score=Decimal('50'),
                relative_volume_score=Decimal('50')
            )
        
        # Get recent volume data
        recent_data = market_data[-self._volume_window:]
        pattern_volume = pattern.candles[-1].volume if pattern.candles else Decimal('0')
        
        # Calculate average volume
        avg_volume = sum(candle.volume for candle in recent_data) / len(recent_data)
        
        # Volume spike scoring
        if avg_volume > 0:
            volume_ratio = pattern_volume / avg_volume
            if volume_ratio > Decimal('2.0'):
                volume_spike_score = Decimal('100')
            elif volume_ratio > Decimal('1.5'):
                volume_spike_score = Decimal('80')
            elif volume_ratio > Decimal('1.2'):
                volume_spike_score = Decimal('60')
            else:
                volume_spike_score = volume_ratio * Decimal('50')
        else:
            volume_spike_score = Decimal('50')
        
        # Volume trend scoring
        if len(recent_data) >= 5:
            early_volumes = [candle.volume for candle in recent_data[:5]]
            late_volumes = [candle.volume for candle in recent_data[-5:]]
            early_avg = sum(early_volumes) / len(early_volumes)
            late_avg = sum(late_volumes) / len(late_volumes)
            
            if early_avg > 0:
                trend_ratio = late_avg / early_avg
                volume_trend_score = min(Decimal('100'), trend_ratio * Decimal('50'))
            else:
                volume_trend_score = Decimal('50')
        else:
            volume_trend_score = Decimal('50')
        
        # Breakout volume scoring (use volume profile if available)
        if volume_profile:
            breakout_volume_score = min(Decimal('100'), volume_profile.volume_ratio * Decimal('50'))
        else:
            breakout_volume_score = volume_spike_score  # Fallback to spike score
        
        # Relative volume scoring
        relative_volume_score = min(Decimal('100'), volume_spike_score)
        
        return VolumeScoring(
            volume_spike_score=max(Decimal('0'), min(Decimal('100'), volume_spike_score)),
            volume_trend_score=max(Decimal('0'), min(Decimal('100'), volume_trend_score)),
            breakout_volume_score=max(Decimal('0'), min(Decimal('100'), breakout_volume_score)),
            relative_volume_score=max(Decimal('0'), min(Decimal('100'), relative_volume_score))
        )
    
    def _calculate_context_scoring(self, 
                                 pattern: CandlestickPattern,
                                 market_data: List[CandlestickData]) -> MarketContextScoring:
        """Calculate market context scoring."""
        if not market_data or len(market_data) < self._volatility_window:
            return MarketContextScoring(
                trend_alignment_score=Decimal('50'),
                volatility_score=Decimal('50'),
                session_score=Decimal('50'),
                support_resistance_score=Decimal('50'),
                momentum_score=Decimal('50')
            )
        
        recent_data = market_data[-self._volatility_window:]
        
        # Trend alignment scoring
        trend_alignment_score = self._calculate_trend_alignment(pattern, recent_data)
        
        # Volatility scoring
        volatility_score = self._calculate_volatility_score(recent_data)
        
        # Session scoring
        session_score = self._calculate_session_score(pattern.timestamp)
        
        # Support/resistance scoring
        support_resistance_score = self._calculate_support_resistance_score(pattern, recent_data)
        
        # Momentum scoring
        momentum_score = self._calculate_momentum_score(recent_data)
        
        return MarketContextScoring(
            trend_alignment_score=trend_alignment_score,
            volatility_score=volatility_score,
            session_score=session_score,
            support_resistance_score=support_resistance_score,
            momentum_score=momentum_score
        )
    
    def _calculate_trend_alignment(self, 
                                 pattern: CandlestickPattern,
                                 market_data: List[CandlestickData]) -> Decimal:
        """Calculate trend alignment score."""
        if len(market_data) < 10:
            return Decimal('50')
        
        # Calculate trend direction using simple moving average
        recent_closes = [candle.close for candle in market_data[-10:]]
        early_avg = sum(recent_closes[:5]) / 5
        late_avg = sum(recent_closes[-5:]) / 5
        
        trend_direction = "bullish" if late_avg > early_avg else "bearish"
        
        # Determine pattern bias
        pattern_bias = "bullish" if pattern.bullish_probability > pattern.bearish_probability else "bearish"
        
        # Score alignment
        if trend_direction == pattern_bias:
            price_change = abs(late_avg - early_avg) / early_avg * 100 if early_avg > 0 else Decimal('0')
            alignment_score = Decimal('80') + min(Decimal('20'), price_change * Decimal('2'))
        else:
            alignment_score = Decimal('30')  # Penalty for misalignment
        
        return max(Decimal('0'), min(Decimal('100'), alignment_score))
    
    def _calculate_volatility_score(self, market_data: List[CandlestickData]) -> Decimal:
        """Calculate volatility score (higher is better for pattern reliability)."""
        if len(market_data) < 5:
            return Decimal('50')
        
        # Calculate price volatility
        closes = [float(candle.close) for candle in market_data]
        volatility = Decimal(str(statistics.stdev(closes))) / Decimal(str(statistics.mean(closes))) * 100
        
        # Optimal volatility range: 1-3%
        if volatility < Decimal('0.5'):
            score = Decimal('40')  # Too low volatility
        elif volatility <= Decimal('1.0'):
            score = Decimal('60')
        elif volatility <= Decimal('3.0'):
            score = Decimal('100')  # Optimal range
        elif volatility <= Decimal('5.0'):
            score = Decimal('70')
        else:
            score = Decimal('30')  # Too high volatility
        
        return score
    
    def _calculate_session_score(self, timestamp: datetime) -> Decimal:
        """Calculate market session score."""
        # Convert to UTC hour
        hour = timestamp.hour
        
        # Market sessions (UTC):
        # Asian: 22:00-08:00
        # London: 08:00-16:00
        # New York: 13:00-21:00
        # Overlap (London/NY): 13:00-16:00
        
        if 13 <= hour < 16:  # London/NY overlap
            return Decimal('100')
        elif 8 <= hour < 16:  # London session
            return Decimal('90')
        elif 13 <= hour < 21:  # New York session
            return Decimal('85')
        elif 22 <= hour or hour < 8:  # Asian session
            return Decimal('70')
        else:  # Off hours
            return Decimal('40')
    
    def _calculate_support_resistance_score(self, 
                                          pattern: CandlestickPattern,
                                          market_data: List[CandlestickData]) -> Decimal:
        """Calculate support/resistance proximity score."""
        if not pattern.candles or len(market_data) < 20:
            return Decimal('50')
        
        pattern_price = pattern.candles[-1].close
        
        # Find recent highs and lows
        recent_highs = [candle.high for candle in market_data[-20:]]
        recent_lows = [candle.low for candle in market_data[-20:]]
        
        max_high = max(recent_highs)
        min_low = min(recent_lows)
        
        # Calculate distance to key levels
        range_size = max_high - min_low
        if range_size == 0:
            return Decimal('50')
        
        # Distance to support (min_low)
        distance_to_support = abs(pattern_price - min_low) / range_size
        
        # Distance to resistance (max_high)
        distance_to_resistance = abs(pattern_price - max_high) / range_size
        
        # Score higher when near key levels
        min_distance = min(distance_to_support, distance_to_resistance)
        
        if min_distance < Decimal('0.05'):  # Very close to key level
            return Decimal('95')
        elif min_distance < Decimal('0.1'):
            return Decimal('80')
        elif min_distance < Decimal('0.2'):
            return Decimal('65')
        else:
            return Decimal('50')
    
    def _calculate_momentum_score(self, market_data: List[CandlestickData]) -> Decimal:
        """Calculate price momentum score."""
        if len(market_data) < 5:
            return Decimal('50')
        
        # Calculate momentum using price change over recent periods
        closes = [candle.close for candle in market_data[-5:]]
        
        momentum_score = Decimal('50')
        
        # Calculate consecutive moves
        consecutive_moves = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                consecutive_moves += 1
            elif closes[i] < closes[i-1]:
                consecutive_moves -= 1
        
        # Score based on momentum strength
        if abs(consecutive_moves) >= 3:
            momentum_score = Decimal('80')
        elif abs(consecutive_moves) >= 2:
            momentum_score = Decimal('65')
        else:
            momentum_score = Decimal('50')
        
        return momentum_score
    
    def _get_historical_performance(self, 
                                  pattern_type: PatternType,
                                  timeframe: Timeframe) -> HistoricalPerformance:
        """Get or create historical performance metrics."""
        key = (pattern_type, timeframe)
        
        if key in self._performance_cache:
            return self._performance_cache[key]
        
        # Calculate performance from historical data
        outcomes = self._pattern_history[key]
        
        if not outcomes:
            # Default performance for new patterns
            performance = HistoricalPerformance(
                pattern_type=pattern_type,
                timeframe=timeframe,
                total_occurrences=0,
                successful_trades=0,
                success_rate=Decimal('55'),  # Slightly optimistic default
                average_profit_loss=Decimal('1.5'),
                average_duration_hours=Decimal('24'),
                reliability_score=Decimal('60')
            )
        else:
            successful = sum(1 for outcome in outcomes if outcome.success is True)
            total = len(outcomes)
            success_rate = Decimal(successful) / Decimal(total) * 100 if total > 0 else Decimal('50')
            
            # Calculate average profit/loss
            completed_outcomes = [o for o in outcomes if o.profit_loss_pct is not None]
            if completed_outcomes:
                avg_pnl = sum(o.profit_loss_pct for o in completed_outcomes) / len(completed_outcomes)
                avg_duration = sum(o.duration_hours for o in completed_outcomes if o.duration_hours) / len(completed_outcomes)
            else:
                avg_pnl = Decimal('0')
                avg_duration = Decimal('24')
            
            # Calculate reliability score
            reliability = min(Decimal('100'), success_rate + (avg_pnl * Decimal('5')))
            
            performance = HistoricalPerformance(
                pattern_type=pattern_type,
                timeframe=timeframe,
                total_occurrences=total,
                successful_trades=successful,
                success_rate=success_rate,
                average_profit_loss=avg_pnl,
                average_duration_hours=avg_duration,
                reliability_score=reliability
            )
        
        self._performance_cache[key] = performance
        return performance
    
    def record_pattern_outcome(self, outcome: PatternOutcome):
        """Record a pattern outcome for historical tracking."""
        key = (outcome.pattern_type, outcome.timeframe)
        self._pattern_history[key].append(outcome)
        
        # Invalidate cache to force recalculation
        if key in self._performance_cache:
            del self._performance_cache[key]
    
    def batch_score_patterns(self, 
                           patterns: List[CandlestickPattern],
                           market_data: List[CandlestickData],
                           volume_profiles: Optional[Dict[str, VolumeProfile]] = None) -> List[CompositePatternScore]:
        """Score multiple patterns efficiently."""
        scores = []
        
        for pattern in patterns:
            volume_profile = None
            if volume_profiles and hasattr(pattern, 'pattern_id'):
                volume_profile = volume_profiles.get(pattern.pattern_id)
            
            score = self.score_pattern(pattern, market_data, volume_profile)
            scores.append(score)
        
        return scores
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of historical performance tracking."""
        total_patterns = sum(len(outcomes) for outcomes in self._pattern_history.values())
        total_successful = sum(
            sum(1 for outcome in outcomes if outcome.success is True)
            for outcomes in self._pattern_history.values()
        )
        
        overall_success_rate = (total_successful / total_patterns * 100) if total_patterns > 0 else 0
        
        pattern_breakdown = {}
        for (pattern_type, timeframe), outcomes in self._pattern_history.items():
            successful = sum(1 for outcome in outcomes if outcome.success is True)
            total = len(outcomes)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            pattern_breakdown[f"{pattern_type.value}_{timeframe.value}"] = {
                "total_occurrences": total,
                "successful_trades": successful,
                "success_rate": round(success_rate, 2)
            }
        
        return {
            "total_patterns_tracked": total_patterns,
            "overall_success_rate": round(overall_success_rate, 2),
            "pattern_breakdown": pattern_breakdown,
            "tracking_period_days": 30  # Assuming 30-day tracking window
        }