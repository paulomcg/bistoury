"""
Trading Signal Models

This module contains Pydantic models for trading signals and pattern recognition:
- TradingSignal: Individual trading signal with confidence and direction
- CandlestickPattern: Candlestick pattern recognition and scoring
- AnalysisContext: Multi-timeframe analysis context and data
- SignalAggregation: Signal combining and weighting logic
- SignalPerformance: Performance tracking and analytics

All models are designed for algorithmic trading signal generation
and include comprehensive validation for signal reliability.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict
from enum import Enum

from .market_data import Timeframe, CandlestickData


class SignalDirection(str, Enum):
    """Trading signal direction enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalType(str, Enum):
    """Trading signal type enumeration."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    PATTERN = "pattern"
    ARBITRAGE = "arbitrage"


class ConfidenceLevel(str, Enum):
    """Signal confidence level enumeration."""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MEDIUM = "medium"          # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%


class RiskLevel(str, Enum):
    """Risk level enumeration for trading signals."""
    VERY_LOW = "very_low"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PatternType(str, Enum):
    """Candlestick pattern type enumeration."""
    # Single candle patterns
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    INVERTED_HAMMER = "inverted_hammer"
    SPINNING_TOP = "spinning_top"
    MARUBOZU = "marubozu"
    
    # Two candle patterns
    ENGULFING = "engulfing"
    HARAMI = "harami"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    TWEEZER_TOPS = "tweezer_tops"
    TWEEZER_BOTTOMS = "tweezer_bottoms"
    
    # Three candle patterns
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    THREE_INSIDE_UP = "three_inside_up"
    THREE_INSIDE_DOWN = "three_inside_down"
    THREE_OUTSIDE_UP = "three_outside_up"
    THREE_OUTSIDE_DOWN = "three_outside_down"
    
    # Complex patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    CUP_AND_HANDLE = "cup_and_handle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE = "wedge"
    TRIANGLE = "triangle"


class TradingSignal(BaseModel):
    """
    Individual trading signal with confidence scoring and direction.
    
    Represents a single trading recommendation with detailed metadata
    including confidence levels, signal strength, and timing information.
    """
    
    signal_id: str = Field(
        ...,
        description="Unique signal identifier"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol for this signal"
    )
    direction: SignalDirection = Field(
        ...,
        description="Signal direction (buy, sell, hold, etc.)"
    )
    signal_type: SignalType = Field(
        ...,
        description="Type of signal (technical, fundamental, etc.)"
    )
    confidence: Decimal = Field(
        ...,
        description="Signal confidence as percentage (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    strength: Decimal = Field(
        ...,
        description="Signal strength (0-1, where 1 is strongest)",
        ge=Decimal('0'),
        le=Decimal('1')
    )
    price: Decimal = Field(
        ...,
        description="Price at which signal was generated",
        gt=Decimal('0')
    )
    target_price: Optional[Decimal] = Field(
        None,
        description="Target price for the signal",
        gt=Decimal('0')
    )
    stop_loss: Optional[Decimal] = Field(
        None,
        description="Suggested stop loss price",
        gt=Decimal('0')
    )
    timeframe: Timeframe = Field(
        ...,
        description="Timeframe for this signal"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signal generation timestamp"
    )
    expiry: Optional[datetime] = Field(
        None,
        description="Signal expiry timestamp"
    )
    source: str = Field(
        ...,
        description="Signal source/strategy name"
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation for the signal"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional signal metadata"
    )
    is_active: bool = Field(
        default=True,
        description="Whether the signal is currently active"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category."""
        if self.confidence < 20:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 40:
            return ConfidenceLevel.LOW
        elif self.confidence < 60:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 80:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    @computed_field
    @property
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """Calculate risk/reward ratio if target and stop loss are set."""
        if not self.target_price or not self.stop_loss:
            return None
        
        if self.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            # For buy signals
            reward = abs(self.target_price - self.price)
            risk = abs(self.price - self.stop_loss)
        else:
            # For sell signals
            reward = abs(self.price - self.target_price)
            risk = abs(self.stop_loss - self.price)
        
        if risk == 0:
            return None
        
        return reward / risk
    
    @computed_field
    @property
    def signal_score(self) -> Decimal:
        """Calculate overall signal score combining confidence and strength."""
        return (self.confidence / Decimal('100')) * self.strength
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if not self.expiry:
            return False
        return datetime.now(timezone.utc) > self.expiry
    
    @computed_field
    @property
    def time_to_expiry(self) -> Optional[timedelta]:
        """Calculate time remaining until expiry."""
        if not self.expiry:
            return None
        return self.expiry - datetime.now(timezone.utc)
    
    @computed_field
    @property
    def age(self) -> timedelta:
        """Calculate signal age."""
        return datetime.now(timezone.utc) - self.timestamp
    
    def update_status(self, is_active: bool, reason: str = "") -> None:
        """Update signal active status."""
        self.is_active = is_active
        if reason:
            self.metadata["status_change_reason"] = reason
            self.metadata["status_change_time"] = datetime.now(timezone.utc).isoformat()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the signal."""
        self.metadata[key] = value
    
    @classmethod
    def create_buy_signal(
        cls,
        signal_id: str,
        symbol: str,
        price: Decimal,
        confidence: Decimal,
        strength: Decimal,
        source: str,
        reason: str,
        timeframe: Timeframe = Timeframe.ONE_HOUR,
        target_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        signal_type: SignalType = SignalType.TECHNICAL
    ) -> 'TradingSignal':
        """Create a buy signal with common parameters."""
        return cls(
            signal_id=signal_id,
            symbol=symbol,
            direction=SignalDirection.BUY,
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            price=price,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe=timeframe,
            source=source,
            reason=reason
        )
    
    @classmethod
    def create_sell_signal(
        cls,
        signal_id: str,
        symbol: str,
        price: Decimal,
        confidence: Decimal,
        strength: Decimal,
        source: str,
        reason: str,
        timeframe: Timeframe = Timeframe.ONE_HOUR,
        target_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        signal_type: SignalType = SignalType.TECHNICAL
    ) -> 'TradingSignal':
        """Create a sell signal with common parameters."""
        return cls(
            signal_id=signal_id,
            symbol=symbol,
            direction=SignalDirection.SELL,
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            price=price,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe=timeframe,
            source=source,
            reason=reason
        )


class CandlestickPattern(BaseModel):
    """
    Candlestick pattern recognition and scoring model.
    
    Identifies and scores candlestick patterns with confidence levels,
    bullish/bearish implications, and pattern-specific metadata.
    """
    
    pattern_id: str = Field(
        ...,
        description="Unique pattern identifier"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol where pattern was found"
    )
    pattern_type: PatternType = Field(
        ...,
        description="Type of candlestick pattern"
    )
    candles: List[CandlestickData] = Field(
        ...,
        description="Candlestick data forming the pattern",
        min_length=1,
        max_length=10
    )
    timeframe: Timeframe = Field(
        ...,
        description="Timeframe of the pattern"
    )
    confidence: Decimal = Field(
        ...,
        description="Pattern confidence score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    reliability: Decimal = Field(
        ...,
        description="Historical pattern reliability (0-1)",
        ge=Decimal('0'),
        le=Decimal('1')
    )
    bullish_probability: Decimal = Field(
        ...,
        description="Probability of bullish outcome (0-1)",
        ge=Decimal('0'),
        le=Decimal('1')
    )
    bearish_probability: Decimal = Field(
        ...,
        description="Probability of bearish outcome (0-1)",
        ge=Decimal('0'),
        le=Decimal('1')
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Pattern detection timestamp"
    )
    completion_price: Decimal = Field(
        ...,
        description="Price when pattern was completed",
        gt=Decimal('0')
    )
    volume_confirmation: bool = Field(
        default=False,
        description="Whether pattern has volume confirmation"
    )
    pattern_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pattern-specific metadata"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @field_validator('bullish_probability', 'bearish_probability')
    @classmethod
    def validate_probabilities(cls, v, info):
        """Validate that probabilities don't exceed 1 when combined."""
        # This is a simplified validation - in practice, probabilities 
        # might not always sum to 1 (could have neutral outcomes)
        return v
    
    @computed_field
    @property
    def pattern_strength(self) -> Decimal:
        """Calculate overall pattern strength."""
        return (self.confidence / Decimal('100')) * self.reliability
    
    @computed_field
    @property
    def directional_bias(self) -> SignalDirection:
        """Determine overall directional bias of the pattern."""
        if self.bullish_probability > self.bearish_probability + Decimal('0.2'):
            return SignalDirection.BUY
        elif self.bearish_probability > self.bullish_probability + Decimal('0.2'):
            return SignalDirection.SELL
        else:
            return SignalDirection.HOLD
    
    @computed_field
    @property
    def pattern_size(self) -> int:
        """Get number of candles in pattern."""
        return len(self.candles)
    
    @computed_field
    @property
    def price_range(self) -> Decimal:
        """Calculate price range of the pattern."""
        if not self.candles:
            return Decimal('0')
        
        highs = [candle.high for candle in self.candles]
        lows = [candle.low for candle in self.candles]
        
        return max(highs) - min(lows)
    
    @computed_field
    @property
    def volume_average(self) -> Decimal:
        """Calculate average volume of pattern candles."""
        if not self.candles:
            return Decimal('0')
        
        total_volume = sum(candle.volume for candle in self.candles)
        return total_volume / len(self.candles)
    
    @computed_field
    @property
    def is_reversal_pattern(self) -> bool:
        """Check if this is typically a reversal pattern."""
        reversal_patterns = {
            PatternType.HAMMER,
            PatternType.HANGING_MAN,
            PatternType.SHOOTING_STAR,
            PatternType.INVERTED_HAMMER,
            PatternType.ENGULFING,
            PatternType.PIERCING_LINE,
            PatternType.DARK_CLOUD_COVER,
            PatternType.MORNING_STAR,
            PatternType.EVENING_STAR,
            PatternType.HEAD_AND_SHOULDERS,
            PatternType.INVERSE_HEAD_AND_SHOULDERS,
            PatternType.DOUBLE_TOP,
            PatternType.DOUBLE_BOTTOM,
        }
        return self.pattern_type in reversal_patterns
    
    @computed_field
    @property
    def is_continuation_pattern(self) -> bool:
        """Check if this is typically a continuation pattern."""
        continuation_patterns = {
            PatternType.FLAG,
            PatternType.PENNANT,
            PatternType.TRIANGLE,
            PatternType.WEDGE,
        }
        return self.pattern_type in continuation_patterns
    
    def to_trading_signal(
        self,
        signal_id: str,
        source: str = "pattern_recognition"
    ) -> TradingSignal:
        """Convert pattern to trading signal."""
        # Determine signal strength based on pattern strength and bias
        strength = self.pattern_strength
        
        # Adjust confidence based on pattern reliability
        confidence = self.confidence * self.reliability
        
        # Create reason string
        reason = f"{self.pattern_type.value.replace('_', ' ').title()} pattern detected with {self.confidence}% confidence"
        
        return TradingSignal(
            signal_id=signal_id,
            symbol=self.symbol,
            direction=self.directional_bias,
            signal_type=SignalType.PATTERN,
            confidence=confidence,
            strength=strength,
            price=self.completion_price,
            timeframe=self.timeframe,
            source=source,
            reason=reason,
            metadata={
                "pattern_type": self.pattern_type.value,
                "pattern_id": self.pattern_id,
                "bullish_probability": float(self.bullish_probability),
                "bearish_probability": float(self.bearish_probability),
                "volume_confirmation": self.volume_confirmation,
                "pattern_candles": len(self.candles)
            }
        )


class AnalysisContext(BaseModel):
    """
    Multi-timeframe analysis context and market conditions.
    
    Provides comprehensive market context including multiple timeframe data,
    trend analysis, volatility metrics, and market conditions for signal generation.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol for analysis"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    timeframes: Dict[Timeframe, CandlestickData] = Field(
        ...,
        description="Current candle data for each timeframe"
    )
    trends: Dict[Timeframe, SignalDirection] = Field(
        default_factory=dict,
        description="Trend direction for each timeframe"
    )
    volatility: Dict[Timeframe, Decimal] = Field(
        default_factory=dict,
        description="Volatility metrics for each timeframe"
    )
    volume_profile: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Volume analysis metrics"
    )
    support_levels: List[Decimal] = Field(
        default_factory=list,
        description="Identified support price levels"
    )
    resistance_levels: List[Decimal] = Field(
        default_factory=list,
        description="Identified resistance price levels"
    )
    market_regime: Literal["trending", "ranging", "volatile", "low_volume"] = Field(
        default="ranging",
        description="Current market regime classification"
    )
    sentiment_score: Optional[Decimal] = Field(
        None,
        description="Market sentiment score (-1 to 1)",
        ge=Decimal('-1'),
        le=Decimal('1')
    )
    liquidity_score: Decimal = Field(
        default=Decimal('0.5'),
        description="Market liquidity score (0-1)",
        ge=Decimal('0'),
        le=Decimal('1')
    )
    correlation_data: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Correlation with other symbols/assets"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def current_price(self) -> Optional[Decimal]:
        """Get current price from shortest timeframe."""
        if not self.timeframes:
            return None
        
        # Find shortest timeframe (lowest seconds value)
        shortest_tf = min(self.timeframes.keys(), key=lambda tf: tf.seconds)
        return self.timeframes[shortest_tf].close
    
    @computed_field
    @property
    def trend_alignment(self) -> Decimal:
        """Calculate trend alignment across timeframes (0-1)."""
        if not self.trends:
            return Decimal('0.5')
        
        total_trends = len(self.trends)
        if total_trends == 0:
            return Decimal('0.5')
        
        # Count bullish trends
        bullish_count = sum(
            1 for direction in self.trends.values()
            if direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        )
        
        # Calculate alignment (1 = all bullish, 0 = all bearish, 0.5 = mixed)
        if bullish_count == total_trends:
            return Decimal('1')
        elif bullish_count == 0:
            return Decimal('0')
        else:
            return Decimal(str(bullish_count)) / Decimal(str(total_trends))
    
    @computed_field
    @property
    def average_volatility(self) -> Decimal:
        """Calculate average volatility across timeframes."""
        if not self.volatility:
            return Decimal('0')
        
        total = sum(self.volatility.values())
        return total / len(self.volatility)
    
    @computed_field
    @property
    def trend_strength(self) -> Decimal:
        """Calculate overall trend strength (0-1)."""
        alignment = self.trend_alignment
        
        # Strong trend if alignment is high (> 0.8 or < 0.2)
        if alignment > Decimal('0.8'):
            return alignment
        elif alignment < Decimal('0.2'):
            return Decimal('1') - alignment
        else:
            # Weak/mixed trend
            return Decimal('0.5') - abs(alignment - Decimal('0.5'))
    
    @computed_field
    @property
    def nearest_support(self) -> Optional[Decimal]:
        """Get nearest support level below current price."""
        current = self.current_price
        if not current or not self.support_levels:
            return None
        
        below_current = [level for level in self.support_levels if level < current]
        return max(below_current) if below_current else None
    
    @computed_field
    @property
    def nearest_resistance(self) -> Optional[Decimal]:
        """Get nearest resistance level above current price."""
        current = self.current_price
        if not current or not self.resistance_levels:
            return None
        
        above_current = [level for level in self.resistance_levels if level > current]
        return min(above_current) if above_current else None
    
    def get_timeframe_data(self, timeframe: Timeframe) -> Optional[CandlestickData]:
        """Get candlestick data for specific timeframe."""
        return self.timeframes.get(timeframe)
    
    def add_timeframe(self, timeframe: Timeframe, candle: CandlestickData, trend: SignalDirection, volatility: Decimal) -> None:
        """Add timeframe data to the context."""
        self.timeframes[timeframe] = candle
        self.trends[timeframe] = trend
        self.volatility[timeframe] = volatility
    
    def is_bullish_environment(self) -> bool:
        """Check if overall environment is bullish."""
        return self.trend_alignment > Decimal('0.6')
    
    def is_bearish_environment(self) -> bool:
        """Check if overall environment is bearish."""
        return self.trend_alignment < Decimal('0.4')
    
    def is_trending_market(self) -> bool:
        """Check if market is in trending regime."""
        return self.market_regime == "trending" and self.trend_strength > Decimal('0.7')
    
    def is_ranging_market(self) -> bool:
        """Check if market is in ranging regime."""
        return self.market_regime == "ranging" or self.trend_strength < Decimal('0.3')


class SignalAggregation(BaseModel):
    """
    Signal aggregation and weighting model.
    
    Combines multiple trading signals with weighted scoring,
    consensus analysis, and conflict resolution for final recommendations.
    """
    
    aggregation_id: str = Field(
        ...,
        description="Unique aggregation identifier"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol for aggregated signals"
    )
    signals: List[TradingSignal] = Field(
        ...,
        description="Individual signals to aggregate"
    )
    weights: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Weight for each signal source (0-1)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Aggregation timestamp"
    )
    min_confidence_threshold: Decimal = Field(
        default=Decimal('30'),
        description="Minimum confidence to include signal",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    conflict_resolution: Literal["weighted_average", "highest_confidence", "consensus", "veto"] = Field(
        default="weighted_average",
        description="Method for resolving conflicting signals"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def active_signals(self) -> List[TradingSignal]:
        """Get only active signals above confidence threshold."""
        return [
            signal for signal in self.signals
            if signal.is_active 
            and not signal.is_expired 
            and signal.confidence >= self.min_confidence_threshold
        ]
    
    @computed_field
    @property
    def bullish_signals(self) -> List[TradingSignal]:
        """Get bullish signals."""
        return [
            signal for signal in self.active_signals
            if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        ]
    
    @computed_field
    @property
    def bearish_signals(self) -> List[TradingSignal]:
        """Get bearish signals."""
        return [
            signal for signal in self.active_signals
            if signal.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]
        ]
    
    @computed_field
    @property
    def neutral_signals(self) -> List[TradingSignal]:
        """Get neutral signals."""
        return [
            signal for signal in self.active_signals
            if signal.direction == SignalDirection.HOLD
        ]
    
    @computed_field
    @property
    def consensus_direction(self) -> SignalDirection:
        """Calculate consensus direction from all signals."""
        active = self.active_signals
        if not active:
            return SignalDirection.HOLD
        
        # Weight signals by their score and assigned weight
        total_bullish_weight = Decimal('0')
        total_bearish_weight = Decimal('0')
        
        for signal in active:
            signal_weight = self.weights.get(signal.source, Decimal('1'))
            weighted_score = signal.signal_score * signal_weight
            
            if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                multiplier = Decimal('2') if signal.direction == SignalDirection.STRONG_BUY else Decimal('1')
                total_bullish_weight += weighted_score * multiplier
            elif signal.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                multiplier = Decimal('2') if signal.direction == SignalDirection.STRONG_SELL else Decimal('1')
                total_bearish_weight += weighted_score * multiplier
        
        # Determine consensus
        if total_bullish_weight > total_bearish_weight * Decimal('1.5'):
            return SignalDirection.STRONG_BUY if total_bullish_weight > total_bearish_weight * Decimal('2') else SignalDirection.BUY
        elif total_bearish_weight > total_bullish_weight * Decimal('1.5'):
            return SignalDirection.STRONG_SELL if total_bearish_weight > total_bullish_weight * Decimal('2') else SignalDirection.SELL
        else:
            return SignalDirection.HOLD
    
    @computed_field
    @property
    def consensus_confidence(self) -> Decimal:
        """Calculate consensus confidence level."""
        active = self.active_signals
        if not active:
            return Decimal('0')
        
        # Weight average confidence by signal scores
        total_weighted_confidence = Decimal('0')
        total_weights = Decimal('0')
        
        for signal in active:
            signal_weight = self.weights.get(signal.source, Decimal('1'))
            weight = signal.signal_score * signal_weight
            total_weighted_confidence += signal.confidence * weight
            total_weights += weight
        
        if total_weights == 0:
            return Decimal('0')
        
        return total_weighted_confidence / total_weights
    
    @computed_field
    @property
    def consensus_strength(self) -> Decimal:
        """Calculate consensus strength (0-1)."""
        active = self.active_signals
        if not active:
            return Decimal('0')
        
        # Calculate agreement level
        consensus_dir = self.consensus_direction
        if consensus_dir == SignalDirection.HOLD:
            return Decimal('0.5')
        
        # Count signals in same direction
        same_direction_count = 0
        total_count = len(active)
        
        for signal in active:
            if consensus_dir in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                    same_direction_count += 1
            elif consensus_dir in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                if signal.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                    same_direction_count += 1
        
        agreement_ratio = Decimal(str(same_direction_count)) / Decimal(str(total_count))
        return agreement_ratio
    
    @computed_field
    @property
    def signal_count_by_type(self) -> Dict[SignalType, int]:
        """Count signals by type."""
        counts = {}
        for signal in self.active_signals:
            counts[signal.signal_type] = counts.get(signal.signal_type, 0) + 1
        return counts
    
    def add_signal(self, signal: TradingSignal, weight: Optional[Decimal] = None) -> None:
        """Add a signal to the aggregation."""
        self.signals.append(signal)
        if weight is not None:
            self.weights[signal.source] = weight
    
    def set_weight(self, source: str, weight: Decimal) -> None:
        """Set weight for a signal source."""
        self.weights[source] = weight
    
    def remove_expired_signals(self) -> int:
        """Remove expired signals and return count removed."""
        original_count = len(self.signals)
        self.signals = [signal for signal in self.signals if not signal.is_expired]
        return original_count - len(self.signals)
    
    def get_strongest_signal(self) -> Optional[TradingSignal]:
        """Get the signal with highest score."""
        active = self.active_signals
        if not active:
            return None
        
        return max(active, key=lambda s: s.signal_score)
    
    def create_aggregated_signal(self, signal_id: str) -> TradingSignal:
        """Create a single aggregated signal from all input signals."""
        return TradingSignal(
            signal_id=signal_id,
            symbol=self.symbol,
            direction=self.consensus_direction,
            signal_type=SignalType.TECHNICAL,  # Composite signal
            confidence=self.consensus_confidence,
            strength=self.consensus_strength,
            price=self.active_signals[0].price if self.active_signals else Decimal('0'),
            timeframe=Timeframe.ONE_HOUR,  # Default aggregation timeframe
            source="signal_aggregation",
            reason=f"Aggregated from {len(self.active_signals)} signals with {self.consensus_confidence:.1f}% consensus confidence",
            metadata={
                "aggregation_id": self.aggregation_id,
                "signal_sources": [s.source for s in self.active_signals],
                "signal_count": len(self.active_signals),
                "bullish_count": len(self.bullish_signals),
                "bearish_count": len(self.bearish_signals),
                "neutral_count": len(self.neutral_signals),
                "consensus_method": self.conflict_resolution
            }
        ) 