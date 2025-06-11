"""
Candlestick Strategy Foundation Models

This module extends the existing Pydantic models with additional functionality
specifically for candlestick pattern analysis and multi-timeframe strategies.

Built on top of:
- CandlestickData (from models.market_data)
- CandlestickPattern (from models.signals)
- PatternType (from models.signals)
- AnalysisContext (from models.signals)
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, computed_field, ConfigDict
from enum import Enum

from ..models.market_data import CandlestickData, Timeframe
from ..models.signals import (
    CandlestickPattern, 
    PatternType, 
    SignalDirection, 
    SignalType,
    AnalysisContext,
    TradingSignal
)


class PatternStrength(str, Enum):
    """Pattern strength classification."""
    VERY_WEAK = "very_weak"      # 0-20%
    WEAK = "weak"                # 20-40%
    MODERATE = "moderate"        # 40-60%
    STRONG = "strong"            # 60-80%
    VERY_STRONG = "very_strong"  # 80-100%

    @classmethod
    def from_confidence(cls, confidence: Decimal) -> 'PatternStrength':
        """Convert confidence percentage to strength level."""
        if confidence < 20:
            return cls.VERY_WEAK
        elif confidence < 40:
            return cls.WEAK
        elif confidence < 60:
            return cls.MODERATE
        elif confidence < 80:
            return cls.STRONG
        else:
            return cls.VERY_STRONG


class TimeframePriority(int, Enum):
    """Priority weights for different timeframes in analysis."""
    ONE_MINUTE = 1       # Least priority for long-term signals
    FIVE_MINUTES = 2
    FIFTEEN_MINUTES = 3  # Highest priority for swing trades
    ONE_HOUR = 2
    FOUR_HOURS = 1
    ONE_DAY = 1


class PatternConfluence(BaseModel):
    """
    Pattern confluence analysis across multiple timeframes.
    
    Analyzes how patterns align across different timeframes to determine
    the strength and reliability of trading signals.
    """
    
    primary_pattern: CandlestickPattern = Field(
        ...,
        description="Main pattern on the analysis timeframe"
    )
    supporting_patterns: List[CandlestickPattern] = Field(
        default_factory=list,
        description="Supporting patterns from other timeframes"
    )
    conflicting_patterns: List[CandlestickPattern] = Field(
        default_factory=list,
        description="Patterns that conflict with the primary signal"
    )
    confluence_score: Decimal = Field(
        ...,
        description="Overall confluence strength (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
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
    def net_support(self) -> int:
        """Calculate net supporting patterns (supporting - conflicting)."""
        return len(self.supporting_patterns) - len(self.conflicting_patterns)
    
    @computed_field
    @property
    def confluence_strength(self) -> PatternStrength:
        """Get confluence strength classification."""
        return PatternStrength.from_confidence(self.confluence_score)
    
    @computed_field
    @property
    def is_strong_confluence(self) -> bool:
        """Check if confluence is strong enough for trading."""
        return self.confluence_score >= Decimal('60')


class VolumeProfile(BaseModel):
    """
    Volume analysis for pattern confirmation.
    
    Analyzes volume characteristics during pattern formation
    to determine the strength and validity of the pattern.
    """
    
    pattern_volume: Decimal = Field(
        ...,
        description="Total volume during pattern formation",
        ge=Decimal('0')
    )
    average_volume: Decimal = Field(
        ...,
        description="Average volume for comparison period",
        ge=Decimal('0')
    )
    volume_ratio: Decimal = Field(
        ...,
        description="Pattern volume / average volume ratio",
        ge=Decimal('0')
    )
    volume_trend: SignalDirection = Field(
        ...,
        description="Volume trend during pattern formation"
    )
    breakout_volume: Optional[Decimal] = Field(
        None,
        description="Volume on pattern breakout/completion",
        ge=Decimal('0')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def is_above_average(self) -> bool:
        """Check if pattern volume is above average."""
        return self.volume_ratio > Decimal('1.0')
    
    @computed_field
    @property
    def volume_confirmation(self) -> bool:
        """Check if volume confirms the pattern."""
        # Strong volume confirmation requires:
        # 1. Above average volume during pattern formation
        # 2. Increasing volume trend
        # 3. Strong breakout volume (if available)
        volume_confirmed = (
            self.volume_ratio > Decimal('1.2') and
            self.volume_trend in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        )
        
        if self.breakout_volume is not None:
            breakout_ratio = self.breakout_volume / self.average_volume
            volume_confirmed = volume_confirmed and breakout_ratio > Decimal('1.5')
        
        return volume_confirmed


class PatternQuality(BaseModel):
    """
    Overall pattern quality assessment.
    
    Combines technical analysis, volume confirmation, and market context
    to provide a comprehensive quality score for pattern reliability.
    """
    
    technical_score: Decimal = Field(
        ...,
        description="Technical pattern quality score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    volume_score: Decimal = Field(
        ...,
        description="Volume confirmation score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    context_score: Decimal = Field(
        ...,
        description="Market context relevance score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    overall_score: Decimal = Field(
        ...,
        description="Composite quality score (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def quality_grade(self) -> str:
        """Get letter grade for pattern quality."""
        if self.overall_score >= 90:
            return "A+"
        elif self.overall_score >= 80:
            return "A"
        elif self.overall_score >= 70:
            return "B"
        elif self.overall_score >= 60:
            return "C"
        elif self.overall_score >= 50:
            return "D"
        else:
            return "F"
    
    @computed_field
    @property
    def is_high_quality(self) -> bool:
        """Check if pattern meets high quality threshold."""
        return self.overall_score >= Decimal('70')


class MultiTimeframePattern(BaseModel):
    """
    Multi-timeframe pattern analysis result.
    
    Combines pattern analysis across multiple timeframes to generate
    higher-confidence trading signals with proper context.
    """
    
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    primary_timeframe: Timeframe = Field(
        ...,
        description="Primary analysis timeframe"
    )
    analysis_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the analysis was performed"
    )
    
    # Pattern analysis
    primary_pattern: CandlestickPattern = Field(
        ...,
        description="Primary pattern on main timeframe"
    )
    confluence: PatternConfluence = Field(
        ...,
        description="Multi-timeframe confluence analysis"
    )
    volume_profile: VolumeProfile = Field(
        ...,
        description="Volume analysis for pattern"
    )
    quality: PatternQuality = Field(
        ...,
        description="Overall pattern quality assessment"
    )
    
    # Market context
    market_context: AnalysisContext = Field(
        ...,
        description="Current market conditions and context"
    )
    
    # Risk/reward
    entry_price: Decimal = Field(
        ...,
        description="Suggested entry price",
        gt=Decimal('0')
    )
    stop_loss: Optional[Decimal] = Field(
        None,
        description="Suggested stop loss level",
        gt=Decimal('0')
    )
    take_profit: Optional[Decimal] = Field(
        None,
        description="Suggested take profit level",
        gt=Decimal('0')
    )
    risk_reward_ratio: Optional[Decimal] = Field(
        None,
        description="Risk to reward ratio",
        gt=Decimal('0')
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
    def signal_strength(self) -> Decimal:
        """Calculate overall signal strength (0-1)."""
        # Combine pattern quality, confluence, and market context
        pattern_weight = Decimal('0.4')
        confluence_weight = Decimal('0.3')
        context_weight = Decimal('0.3')
        
        pattern_component = (self.quality.overall_score / Decimal('100')) * pattern_weight
        confluence_component = (self.confluence.confluence_score / Decimal('100')) * confluence_weight
        context_component = (self.market_context.trend_strength) * context_weight
        
        return pattern_component + confluence_component + context_component
    
    @computed_field
    @property
    def is_tradeable(self) -> bool:
        """Check if pattern meets minimum trading criteria."""
        return (
            self.quality.is_high_quality and
            self.confluence.is_strong_confluence and
            self.volume_profile.volume_confirmation and
            self.signal_strength >= Decimal('0.6')
        )
    
    @computed_field
    @property
    def direction_confidence(self) -> Decimal:
        """Get confidence in the directional bias."""
        base_confidence = self.primary_pattern.confidence
        
        # Adjust based on confluence
        confluence_adjustment = (self.confluence.confluence_score - Decimal('50')) / Decimal('100')
        
        # Adjust based on market context alignment
        context_adjustment = Decimal('0')
        if self.market_context.is_bullish_environment() and self.primary_pattern.directional_bias == SignalDirection.BUY:
            context_adjustment = Decimal('10')
        elif self.market_context.is_bearish_environment() and self.primary_pattern.directional_bias == SignalDirection.SELL:
            context_adjustment = Decimal('10')
        
        adjusted_confidence = base_confidence + (confluence_adjustment * Decimal('20')) + context_adjustment
        
        # Clamp to 0-100 range
        return max(Decimal('0'), min(Decimal('100'), adjusted_confidence))
    
    def to_trading_signal(self, signal_id: str) -> TradingSignal:
        """Convert multi-timeframe pattern to trading signal."""
        return TradingSignal(
            signal_id=signal_id,
            symbol=self.symbol,
            direction=self.primary_pattern.directional_bias,
            signal_type=SignalType.PATTERN,
            confidence=self.direction_confidence,
            strength=self.signal_strength,
            price=self.entry_price,
            target_price=self.take_profit,
            stop_loss=self.stop_loss,
            timeframe=self.primary_timeframe,
            source="multi_timeframe_candlestick_strategy",
            reason=f"Multi-timeframe {self.primary_pattern.pattern_type.value.replace('_', ' ').title()} pattern with {self.confluence.net_support} supporting patterns",
            metadata={
                "pattern_type": self.primary_pattern.pattern_type.value,
                "pattern_id": self.primary_pattern.pattern_id,
                "confluence_score": float(self.confluence.confluence_score),
                "quality_score": float(self.quality.overall_score),
                "volume_confirmed": self.volume_profile.volume_confirmation,
                "supporting_patterns": len(self.confluence.supporting_patterns),
                "conflicting_patterns": len(self.confluence.conflicting_patterns),
                "risk_reward_ratio": float(self.risk_reward_ratio) if self.risk_reward_ratio else None,
                "signal_strength": float(self.signal_strength)
            }
        )


class StrategyConfiguration(BaseModel):
    """
    Configuration for candlestick strategy parameters.
    
    Defines all configurable parameters for pattern recognition,
    analysis, and signal generation. Loads from centralized config files.
    """
    
    # Timeframes to analyze
    timeframes: List[Timeframe] = Field(
        default=[Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES],
        description="Timeframes to include in analysis"
    )
    primary_timeframe: Timeframe = Field(
        default=Timeframe.FIVE_MINUTES,
        description="Primary timeframe for signal generation"
    )
    
    # Pattern detection thresholds
    min_pattern_confidence: Decimal = Field(
        default=Decimal('60'),
        description="Minimum confidence for pattern detection",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    min_confluence_score: Decimal = Field(
        default=Decimal('50'),
        description="Minimum confluence score for signal generation",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    min_quality_score: Decimal = Field(
        default=Decimal('70'),
        description="Minimum quality score for trading",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    
    # Volume analysis
    volume_lookback_periods: int = Field(
        default=20,
        description="Number of periods for volume average calculation",
        ge=1
    )
    min_volume_ratio: Decimal = Field(
        default=Decimal('1.2'),
        description="Minimum volume ratio for confirmation",
        ge=Decimal('0')
    )
    
    # Risk management
    default_stop_loss_pct: Decimal = Field(
        default=Decimal('2.0'),
        description="Default stop loss percentage",
        ge=Decimal('0.1'),
        le=Decimal('10.0')
    )
    default_take_profit_pct: Decimal = Field(
        default=Decimal('4.0'),
        description="Default take profit percentage",
        ge=Decimal('0.1'),
        le=Decimal('20.0')
    )
    min_risk_reward_ratio: Decimal = Field(
        default=Decimal('1.5'),
        description="Minimum risk to reward ratio",
        ge=Decimal('1.0')
    )
    
    # Pattern-specific settings
    enable_single_patterns: bool = Field(
        default=True,
        description="Enable single candlestick pattern detection"
    )
    enable_multi_patterns: bool = Field(
        default=True,
        description="Enable multi-candlestick pattern detection"
    )
    enable_complex_patterns: bool = Field(
        default=False,
        description="Enable complex chart pattern detection"
    )
    
    @classmethod
    def from_config_manager(cls) -> 'StrategyConfiguration':
        """Create configuration from centralized config manager."""
        try:
            from ..config_manager import get_config_manager
            config_manager = get_config_manager()
            
            # Timeframe mapping
            timeframe_map = {
                "1m": Timeframe.ONE_MINUTE,
                "5m": Timeframe.FIVE_MINUTES, 
                "15m": Timeframe.FIFTEEN_MINUTES,
                "1h": Timeframe.ONE_HOUR,
                "4h": Timeframe.FOUR_HOURS,
                "1d": Timeframe.ONE_DAY
            }
            
            # Load timeframe settings
            primary_tf_str = config_manager.get('strategy', 'timeframe_analysis', 'primary_timeframe', default='5m')
            primary_timeframe = timeframe_map.get(primary_tf_str, Timeframe.FIVE_MINUTES)
            
            default_tf_list = config_manager.get_list('strategy', 'timeframe_analysis', 'default_timeframes', default=['1m', '5m', '15m'])
            timeframes = [timeframe_map.get(tf, Timeframe.FIVE_MINUTES) for tf in default_tf_list]
            
            # Load other configurations
            return cls(
                timeframes=timeframes,
                primary_timeframe=primary_timeframe,
                min_pattern_confidence=config_manager.get_decimal('strategy', 'pattern_detection', 'min_pattern_confidence', default=60),
                min_confluence_score=config_manager.get_decimal('strategy', 'pattern_detection', 'min_confluence_score', default=50),
                min_quality_score=config_manager.get_decimal('strategy', 'pattern_detection', 'min_quality_score', default=70),
                volume_lookback_periods=config_manager.get_int('strategy', 'volume_analysis', 'volume_lookback_periods', default=20),
                min_volume_ratio=config_manager.get_decimal('strategy', 'volume_analysis', 'min_volume_ratio', default=1.2),
                default_stop_loss_pct=config_manager.get_decimal('strategy', 'risk_management', 'default_stop_loss_pct', default=2.0),
                default_take_profit_pct=config_manager.get_decimal('strategy', 'risk_management', 'default_take_profit_pct', default=4.0),
                min_risk_reward_ratio=config_manager.get_decimal('strategy', 'risk_management', 'min_risk_reward_ratio', default=1.5),
                enable_single_patterns=config_manager.get_bool('strategy', 'pattern_detection', 'enable_single_patterns', default=True),
                enable_multi_patterns=config_manager.get_bool('strategy', 'pattern_detection', 'enable_multi_patterns', default=True),
                enable_complex_patterns=config_manager.get_bool('strategy', 'pattern_detection', 'enable_complex_patterns', default=False)
            )
        except Exception as e:
            # Fallback to defaults if config loading fails
            import logging
            logging.warning(f"Failed to load strategy config from centralized manager: {e}")
            return cls()
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def target_risk_reward(self) -> Decimal:
        """Calculate target risk/reward ratio."""
        return self.default_take_profit_pct / self.default_stop_loss_pct 