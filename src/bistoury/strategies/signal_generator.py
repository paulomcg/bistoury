"""
Trading Signal Generation

This module implements comprehensive signal generation from candlestick pattern analysis,
converting scored patterns and multi-timeframe confluence into actionable trading signals.

Key features:
- Signal generation from pattern confirmations and scoring
- Entry point calculation with optimal timing
- Stop-loss and take-profit level suggestions
- Signal filtering based on minimum confidence thresholds
- Signal persistence and historical tracking
- Signal expiration and invalidation logic
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, computed_field, ConfigDict
from enum import Enum
import uuid
from collections import defaultdict

from ..models.market_data import CandlestickData, Timeframe
from ..models.signals import (
    TradingSignal, 
    SignalDirection, 
    SignalType,
    CandlestickPattern,
    PatternType,
    AnalysisContext
)
from .candlestick_models import PatternStrength, VolumeProfile, MultiTimeframePattern
from .pattern_scoring import CompositePatternScore, PatternScoringEngine
from .timeframe_analyzer import TimeframeAnalysisResult


class SignalTiming(str, Enum):
    """Signal timing classification for entry optimization."""
    IMMEDIATE = "immediate"      # Enter immediately on pattern completion
    CONFIRMATION = "confirmation"  # Wait for next candle confirmation
    BREAKOUT = "breakout"         # Wait for price breakout from pattern
    PULLBACK = "pullback"         # Wait for pullback to entry zone
    DELAYED = "delayed"           # Delayed entry after full pattern validation


class SignalValidation(str, Enum):
    """Signal validation status for quality control."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class RiskLevel(str, Enum):
    """Risk level classification for position sizing."""
    VERY_LOW = "very_low"    # 0.5% risk
    LOW = "low"              # 1% risk
    MEDIUM = "medium"        # 2% risk
    HIGH = "high"            # 3% risk
    VERY_HIGH = "very_high"  # 5% risk


class SignalConfiguration(BaseModel):
    """Configuration for signal generation parameters."""
    
    min_confidence_threshold: Decimal = Field(
        default=Decimal('60'),
        description="Minimum pattern confidence to generate signal",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    min_strength_threshold: PatternStrength = Field(
        default=PatternStrength.MODERATE,
        description="Minimum pattern strength to generate signal"
    )
    require_volume_confirmation: bool = Field(
        default=True,
        description="Whether to require volume confirmation for signals"
    )
    require_trend_alignment: bool = Field(
        default=True,
        description="Whether to require trend alignment across timeframes"
    )
    min_risk_reward_ratio: Decimal = Field(
        default=Decimal('1.5'),
        description="Minimum risk/reward ratio for signals",
        ge=Decimal('1.0')
    )
    max_risk_per_trade: Decimal = Field(
        default=Decimal('2.0'),
        description="Maximum risk percentage per trade",
        ge=Decimal('0.1'),
        le=Decimal('10.0')
    )
    signal_expiry_hours: int = Field(
        default=24,
        description="Hours after which signals expire",
        ge=1,
        le=168  # 1 week
    )
    enable_multi_timeframe: bool = Field(
        default=True,
        description="Whether to use multi-timeframe analysis"
    )
    priority_timeframe: Timeframe = Field(
        default=Timeframe.FIFTEEN_MINUTES,
        description="Primary timeframe for signal generation"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )


class SignalEntryPoint(BaseModel):
    """Entry point calculation for trading signals."""
    
    entry_price: Decimal = Field(
        ...,
        description="Recommended entry price",
        gt=Decimal('0')
    )
    entry_timing: SignalTiming = Field(
        ...,
        description="Recommended entry timing"
    )
    entry_zone_low: Decimal = Field(
        ...,
        description="Lower bound of entry zone",
        gt=Decimal('0')
    )
    entry_zone_high: Decimal = Field(
        ...,
        description="Upper bound of entry zone",
        gt=Decimal('0')
    )
    max_slippage_pct: Decimal = Field(
        default=Decimal('0.1'),
        description="Maximum acceptable slippage percentage",
        ge=Decimal('0'),
        le=Decimal('5.0')
    )
    entry_window_minutes: int = Field(
        default=30,
        description="Entry window in minutes",
        ge=1,
        le=1440
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def entry_zone_size(self) -> Decimal:
        """Calculate the size of the entry zone."""
        return abs(self.entry_zone_high - self.entry_zone_low)
    
    @computed_field
    @property
    def zone_midpoint(self) -> Decimal:
        """Get the midpoint of the entry zone."""
        return (self.entry_zone_low + self.entry_zone_high) / Decimal('2')


class SignalRiskManagement(BaseModel):
    """Risk management parameters for trading signals."""
    
    stop_loss_price: Decimal = Field(
        ...,
        description="Stop loss price level",
        gt=Decimal('0')
    )
    take_profit_price: Optional[Decimal] = Field(
        None,
        description="Take profit price level",
        gt=Decimal('0')
    )
    risk_amount: Decimal = Field(
        ...,
        description="Risk amount per share/unit",
        gt=Decimal('0')
    )
    reward_amount: Optional[Decimal] = Field(
        None,
        description="Potential reward amount per share/unit",
        gt=Decimal('0')
    )
    risk_percentage: Decimal = Field(
        ...,
        description="Risk as percentage of account",
        ge=Decimal('0.1'),
        le=Decimal('10.0')
    )
    position_size_suggestion: Decimal = Field(
        ...,
        description="Suggested position size",
        gt=Decimal('0')
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Risk level classification"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """Calculate risk/reward ratio."""
        if self.reward_amount and self.risk_amount > 0:
            return self.reward_amount / self.risk_amount
        return None
    
    @computed_field
    @property
    def max_loss_amount(self) -> Decimal:
        """Calculate maximum loss amount."""
        return self.risk_amount * self.position_size_suggestion


class GeneratedSignal(BaseModel):
    """Complete generated trading signal with all components."""
    
    signal_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique signal identifier"
    )
    base_signal: TradingSignal = Field(
        ...,
        description="Base trading signal"
    )
    entry_point: SignalEntryPoint = Field(
        ...,
        description="Entry point calculation"
    )
    risk_management: SignalRiskManagement = Field(
        ...,
        description="Risk management parameters"
    )
    source_pattern: CandlestickPattern = Field(
        ...,
        description="Source pattern that generated the signal"
    )
    pattern_score: CompositePatternScore = Field(
        ...,
        description="Pattern scoring analysis"
    )
    timeframe_analysis: Optional[TimeframeAnalysisResult] = Field(
        None,
        description="Multi-timeframe analysis results"
    )
    validation_status: SignalValidation = Field(
        default=SignalValidation.PENDING,
        description="Signal validation status"
    )
    generation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signal generation timestamp"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
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
    def is_valid(self) -> bool:
        """Check if signal is valid and tradeable."""
        return (
            self.validation_status == SignalValidation.VALID and
            self.base_signal.is_active and
            not self.base_signal.is_expired
        )
    
    @computed_field
    @property
    def signal_quality_score(self) -> Decimal:
        """Calculate overall signal quality score."""
        base_score = self.pattern_score.weighted_confidence_score
        
        # Adjust for risk/reward ratio
        if self.risk_management.risk_reward_ratio:
            if self.risk_management.risk_reward_ratio >= Decimal('2.0'):
                base_score += Decimal('10')
            elif self.risk_management.risk_reward_ratio >= Decimal('1.5'):
                base_score += Decimal('5')
        
        # Adjust for timeframe confluence
        if self.timeframe_analysis and self.timeframe_analysis.confluence_analysis.has_strong_confluence:
            base_score += Decimal('15')
        
        return min(Decimal('100'), base_score)
    
    def update_validation_status(self, status: SignalValidation, reason: str = ""):
        """Update signal validation status."""
        self.validation_status = status
        self.last_updated = datetime.now(timezone.utc)
        
        if reason:
            self.base_signal.add_metadata("validation_reason", reason)


class SignalDatabase:
    """In-memory database for signal persistence and tracking."""
    
    def __init__(self):
        self._signals: Dict[str, GeneratedSignal] = {}
        self._signals_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self._expired_signals: List[str] = []
        self._performance_tracking: Dict[str, Dict[str, Any]] = {}
    
    def store_signal(self, signal: GeneratedSignal) -> None:
        """Store a generated signal."""
        self._signals[signal.signal_id] = signal
        self._signals_by_symbol[signal.base_signal.symbol].append(signal.signal_id)
    
    def get_signal(self, signal_id: str) -> Optional[GeneratedSignal]:
        """Retrieve a signal by ID."""
        return self._signals.get(signal_id)
    
    def get_signals_by_symbol(self, symbol: str, active_only: bool = True) -> List[GeneratedSignal]:
        """Get all signals for a symbol."""
        signal_ids = self._signals_by_symbol.get(symbol, [])
        signals = [self._signals[sid] for sid in signal_ids if sid in self._signals]
        
        if active_only:
            signals = [s for s in signals if s.is_valid]
        
        return sorted(signals, key=lambda s: s.generation_timestamp, reverse=True)
    
    def get_active_signals(self) -> List[GeneratedSignal]:
        """Get all active signals."""
        return [signal for signal in self._signals.values() if signal.is_valid]
    
    def expire_old_signals(self, max_age_hours: int = 24) -> int:
        """Expire signals older than max_age_hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        expired_count = 0
        
        for signal in self._signals.values():
            if signal.generation_timestamp < cutoff_time and signal.validation_status != SignalValidation.EXPIRED:
                signal.update_validation_status(SignalValidation.EXPIRED, "Automatic expiry due to age")
                self._expired_signals.append(signal.signal_id)
                expired_count += 1
        
        return expired_count
    
    def cleanup_expired_signals(self, keep_days: int = 7) -> int:
        """Remove expired signals older than keep_days."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=keep_days)
        removed_count = 0
        
        expired_to_remove = []
        for signal_id in self._expired_signals:
            signal = self._signals.get(signal_id)
            if signal and signal.generation_timestamp < cutoff_time:
                expired_to_remove.append(signal_id)
        
        for signal_id in expired_to_remove:
            signal = self._signals.pop(signal_id, None)
            if signal:
                # Remove from symbol mapping
                symbol_signals = self._signals_by_symbol.get(signal.base_signal.symbol, [])
                if signal_id in symbol_signals:
                    symbol_signals.remove(signal_id)
                
                self._expired_signals.remove(signal_id)
                removed_count += 1
        
        return removed_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_signals = len(self._signals)
        active_signals = len(self.get_active_signals())
        expired_signals = len(self._expired_signals)
        
        return {
            "total_signals": total_signals,
            "active_signals": active_signals,
            "expired_signals": expired_signals,
            "symbols_tracked": len(self._signals_by_symbol),
            "database_size_mb": self._estimate_size_mb()
        }
    
    def _estimate_size_mb(self) -> float:
        """Estimate database size in MB (rough approximation)."""
        # Rough estimate: ~2KB per signal
        return (len(self._signals) * 2) / 1024


class SignalGenerator:
    """
    Main signal generation engine.
    
    Converts pattern analysis and multi-timeframe data into actionable
    trading signals with comprehensive risk management and validation.
    """
    
    def __init__(self, config: Optional[SignalConfiguration] = None):
        self.config = config or SignalConfiguration()
        self.pattern_scorer = PatternScoringEngine()
        self.signal_db = SignalDatabase()
        self._signal_counter = 0
    
    def generate_signal_from_pattern(self,
                                   pattern: CandlestickPattern,
                                   market_data: List[CandlestickData],
                                   volume_profile: Optional[VolumeProfile] = None,
                                   timeframe_analysis: Optional[TimeframeAnalysisResult] = None,
                                   account_balance: Optional[Decimal] = None) -> Optional[GeneratedSignal]:
        """
        Generate a trading signal from a candlestick pattern.
        
        Args:
            pattern: The candlestick pattern to analyze
            market_data: Recent market data for context
            volume_profile: Volume analysis (optional)
            timeframe_analysis: Multi-timeframe analysis results (optional)
            account_balance: Account balance for position sizing (optional)
        
        Returns:
            Generated signal if pattern meets criteria, None otherwise
        """
        # Score the pattern
        pattern_score = self.pattern_scorer.score_pattern(pattern, market_data, volume_profile)
        
        # Apply initial filters
        if not self._passes_initial_filters(pattern_score, timeframe_analysis):
            return None
        
        # Calculate entry point
        entry_point = self._calculate_entry_point(pattern, market_data, pattern_score)
        
        # Calculate risk management
        risk_management = self._calculate_risk_management(
            pattern, entry_point, pattern_score, account_balance
        )
        
        # Create base trading signal
        base_signal = self._create_base_signal(pattern, pattern_score, entry_point, risk_management)
        
        # Create generated signal
        generated_signal = GeneratedSignal(
            base_signal=base_signal,
            entry_point=entry_point,
            risk_management=risk_management,
            source_pattern=pattern,
            pattern_score=pattern_score,
            timeframe_analysis=timeframe_analysis
        )
        
        # Validate signal
        validation_status = self._validate_signal(generated_signal)
        generated_signal.update_validation_status(validation_status)
        
        # Store signal
        if validation_status == SignalValidation.VALID:
            self.signal_db.store_signal(generated_signal)
        
        return generated_signal
    
    def generate_signals_from_timeframe_analysis(self,
                                               timeframe_analysis: TimeframeAnalysisResult,
                                               account_balance: Optional[Decimal] = None) -> List[GeneratedSignal]:
        """
        Generate signals from multi-timeframe analysis results.
        
        Args:
            timeframe_analysis: Complete timeframe analysis
            account_balance: Account balance for position sizing
        
        Returns:
            List of generated signals
        """
        signals = []
        
        # Process patterns from each timeframe
        for timeframe in [Timeframe.FIFTEEN_MINUTES, Timeframe.FIVE_MINUTES, Timeframe.ONE_MINUTE]:
            # Single patterns
            single_patterns = timeframe_analysis.single_patterns.get(timeframe, [])
            for pattern in single_patterns:
                market_data = timeframe_analysis.synchronization.timeframe_data.get(timeframe, [])
                
                signal = self.generate_signal_from_pattern(
                    pattern=pattern,
                    market_data=market_data,
                    timeframe_analysis=timeframe_analysis,
                    account_balance=account_balance
                )
                
                if signal:
                    signals.append(signal)
            
            # Multi patterns
            multi_patterns = timeframe_analysis.multi_patterns.get(timeframe, [])
            for pattern in multi_patterns:
                market_data = timeframe_analysis.synchronization.timeframe_data.get(timeframe, [])
                
                signal = self.generate_signal_from_pattern(
                    pattern=pattern,
                    market_data=market_data,
                    timeframe_analysis=timeframe_analysis,
                    account_balance=account_balance
                )
                
                if signal:
                    signals.append(signal)
        
        # Rank signals by quality and remove duplicates
        signals = self._rank_and_deduplicate_signals(signals)
        
        return signals
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[GeneratedSignal]:
        """Get active signals, optionally filtered by symbol."""
        if symbol:
            return self.signal_db.get_signals_by_symbol(symbol, active_only=True)
        return self.signal_db.get_active_signals()
    
    def expire_old_signals(self) -> int:
        """Expire signals older than configured expiry time."""
        return self.signal_db.expire_old_signals(self.config.signal_expiry_hours)
    
    def cleanup_database(self) -> int:
        """Clean up old expired signals."""
        return self.signal_db.cleanup_expired_signals()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        db_stats = self.signal_db.get_statistics()
        
        return {
            **db_stats,
            "signals_generated": self._signal_counter,
            "config": {
                "min_confidence": float(self.config.min_confidence_threshold),
                "min_strength": self.config.min_strength_threshold.value,
                "expiry_hours": self.config.signal_expiry_hours
            }
        }
    
    def _passes_initial_filters(self,
                              pattern_score: CompositePatternScore,
                              timeframe_analysis: Optional[TimeframeAnalysisResult]) -> bool:
        """Check if pattern passes initial filtering criteria."""
        # Confidence threshold
        if pattern_score.weighted_confidence_score < self.config.min_confidence_threshold:
            return False
        
        # Strength threshold
        strength_values = {
            PatternStrength.VERY_WEAK: 0,
            PatternStrength.WEAK: 1,
            PatternStrength.MODERATE: 2,
            PatternStrength.STRONG: 3,
            PatternStrength.VERY_STRONG: 4
        }
        
        if strength_values[pattern_score.pattern_strength] < strength_values[self.config.min_strength_threshold]:
            return False
        
        # Volume confirmation
        if self.config.require_volume_confirmation:
            if pattern_score.volume_scoring.overall_volume_score < Decimal('50'):
                return False
        
        # Trend alignment (if timeframe analysis available)
        if self.config.require_trend_alignment and timeframe_analysis:
            if not timeframe_analysis.trend_alignment.is_aligned:
                return False
        
        # Pattern must meet our own tradeable threshold (don't rely on hardcoded is_tradeable)
        # This allows the signal generator to have its own configurable threshold
        
        return True
    
    def _calculate_entry_point(self,
                             pattern: CandlestickPattern,
                             market_data: List[CandlestickData],
                             pattern_score: CompositePatternScore) -> SignalEntryPoint:
        """Calculate optimal entry point for the signal."""
        current_price = pattern.completion_price
        pattern_type = pattern.pattern_type
        
        # Determine entry timing based on pattern type and strength
        if pattern_score.pattern_strength == PatternStrength.VERY_STRONG:
            entry_timing = SignalTiming.IMMEDIATE
        elif pattern_score.volume_scoring.overall_volume_score > Decimal('80'):
            entry_timing = SignalTiming.CONFIRMATION
        else:
            entry_timing = SignalTiming.BREAKOUT
        
        # Calculate entry zone based on pattern
        if pattern.directional_bias in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            # Bullish pattern - entry above pattern high
            pattern_high = max(candle.high for candle in pattern.candles)
            entry_price = pattern_high + (pattern_high * Decimal('0.001'))  # Small buffer
            entry_zone_low = pattern_high
            entry_zone_high = pattern_high + (pattern_high * Decimal('0.005'))
        else:
            # Bearish pattern - entry below pattern low
            pattern_low = min(candle.low for candle in pattern.candles)
            entry_price = pattern_low - (pattern_low * Decimal('0.001'))  # Small buffer
            entry_zone_high = pattern_low
            entry_zone_low = pattern_low - (pattern_low * Decimal('0.005'))
        
        return SignalEntryPoint(
            entry_price=entry_price,
            entry_timing=entry_timing,
            entry_zone_low=entry_zone_low,
            entry_zone_high=entry_zone_high,
            max_slippage_pct=Decimal('0.1'),
            entry_window_minutes=30 if entry_timing == SignalTiming.IMMEDIATE else 60
        )
    
    def _calculate_risk_management(self,
                                 pattern: CandlestickPattern,
                                 entry_point: SignalEntryPoint,
                                 pattern_score: CompositePatternScore,
                                 account_balance: Optional[Decimal]) -> SignalRiskManagement:
        """Calculate risk management parameters."""
        direction = pattern.directional_bias
        entry_price = entry_point.entry_price
        
        # Calculate stop loss based on pattern structure
        if direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            # For bullish patterns, stop below pattern low
            pattern_low = min(candle.low for candle in pattern.candles)
            stop_loss_price = pattern_low - (pattern_low * Decimal('0.002'))  # Small buffer
        else:
            # For bearish patterns, stop above pattern high
            pattern_high = max(candle.high for candle in pattern.candles)
            stop_loss_price = pattern_high + (pattern_high * Decimal('0.002'))  # Small buffer
        
        # Calculate risk amount per unit
        risk_amount = abs(entry_price - stop_loss_price)
        
        # Calculate take profit (aim for minimum risk/reward ratio)
        reward_amount = risk_amount * self.config.min_risk_reward_ratio
        
        if direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            take_profit_price = entry_price + reward_amount
        else:
            take_profit_price = entry_price - reward_amount
        
        # Determine risk level based on pattern confidence
        if pattern_score.weighted_confidence_score >= Decimal('90'):
            risk_level = RiskLevel.HIGH
            risk_percentage = Decimal('3.0')
        elif pattern_score.weighted_confidence_score >= Decimal('80'):
            risk_level = RiskLevel.MEDIUM
            risk_percentage = Decimal('2.0')
        elif pattern_score.weighted_confidence_score >= Decimal('70'):
            risk_level = RiskLevel.LOW
            risk_percentage = Decimal('1.0')
        else:
            risk_level = RiskLevel.VERY_LOW
            risk_percentage = Decimal('0.5')
        
        # Cap risk percentage to configured maximum
        risk_percentage = min(risk_percentage, self.config.max_risk_per_trade)
        
        # Calculate position size suggestion
        if account_balance:
            max_risk_amount = account_balance * (risk_percentage / Decimal('100'))
            position_size_suggestion = max_risk_amount / risk_amount
        else:
            position_size_suggestion = Decimal('100')  # Default suggestion
        
        return SignalRiskManagement(
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_percentage=risk_percentage,
            position_size_suggestion=position_size_suggestion,
            risk_level=risk_level
        )
    
    def _create_base_signal(self,
                          pattern: CandlestickPattern,
                          pattern_score: CompositePatternScore,
                          entry_point: SignalEntryPoint,
                          risk_management: SignalRiskManagement) -> TradingSignal:
        """Create the base trading signal."""
        self._signal_counter += 1
        
        # Calculate signal expiry
        expiry = datetime.now(timezone.utc) + timedelta(hours=self.config.signal_expiry_hours)
        
        # Create signal reason
        reason = self._generate_signal_reason(pattern, pattern_score)
        
        # Determine signal direction and strength
        direction = pattern.directional_bias
        confidence = pattern_score.weighted_confidence_score
        strength = min(Decimal('1'), confidence / Decimal('100'))
        
        signal = TradingSignal.create_buy_signal(
            signal_id=f"signal_{self._signal_counter}_{pattern.symbol}_{int(datetime.now().timestamp())}",
            symbol=pattern.symbol,
            price=entry_point.entry_price,
            confidence=confidence,
            strength=strength,
            source="candlestick_pattern_generator",
            reason=reason,
            timeframe=pattern.timeframe,
            target_price=risk_management.take_profit_price,
            stop_loss=risk_management.stop_loss_price,
            signal_type=SignalType.PATTERN
        ) if direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else TradingSignal.create_sell_signal(
            signal_id=f"signal_{self._signal_counter}_{pattern.symbol}_{int(datetime.now().timestamp())}",
            symbol=pattern.symbol,
            price=entry_point.entry_price,
            confidence=confidence,
            strength=strength,
            source="candlestick_pattern_generator",
            reason=reason,
            timeframe=pattern.timeframe,
            target_price=risk_management.take_profit_price,
            stop_loss=risk_management.stop_loss_price,
            signal_type=SignalType.PATTERN
        )
        
        # Set expiry
        signal.expiry = expiry
        
        # Add metadata
        signal.add_metadata("pattern_type", pattern.pattern_type.value)
        signal.add_metadata("pattern_confidence", float(pattern.confidence))
        signal.add_metadata("pattern_score", float(pattern_score.weighted_confidence_score))
        signal.add_metadata("risk_reward_ratio", float(risk_management.risk_reward_ratio) if risk_management.risk_reward_ratio else None)
        signal.add_metadata("entry_timing", entry_point.entry_timing.value)
        signal.add_metadata("risk_level", risk_management.risk_level.value)
        
        return signal
    
    def _generate_signal_reason(self,
                              pattern: CandlestickPattern,
                              pattern_score: CompositePatternScore) -> str:
        """Generate human-readable reason for the signal."""
        pattern_name = pattern.pattern_type.value.replace('_', ' ').title()
        direction = "bullish" if pattern.directional_bias in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else "bearish"
        strength = pattern_score.pattern_strength.value.replace('_', ' ')
        confidence = int(pattern_score.weighted_confidence_score)
        
        return (f"{strength.title()} {direction} {pattern_name} pattern detected "
                f"with {confidence}% confidence. Pattern shows good technical structure "
                f"and volume confirmation.")
    
    def _validate_signal(self, signal: GeneratedSignal) -> SignalValidation:
        """Validate the generated signal."""
        # Check risk/reward ratio
        if signal.risk_management.risk_reward_ratio:
            if signal.risk_management.risk_reward_ratio < self.config.min_risk_reward_ratio:
                return SignalValidation.INVALID
        
        # Check if signal is reasonable
        if signal.entry_point.entry_price <= 0:
            return SignalValidation.INVALID
        
        if signal.risk_management.stop_loss_price <= 0:
            return SignalValidation.INVALID
        
        # Check entry zone makes sense
        if signal.entry_point.entry_zone_low >= signal.entry_point.entry_zone_high:
            return SignalValidation.INVALID
        
        return SignalValidation.VALID
    
    def _rank_and_deduplicate_signals(self, signals: List[GeneratedSignal]) -> List[GeneratedSignal]:
        """Rank signals by quality and remove duplicates."""
        if not signals:
            return signals
        
        # Group by symbol and pattern type to identify duplicates
        signal_groups: Dict[Tuple[str, PatternType], List[GeneratedSignal]] = defaultdict(list)
        
        for signal in signals:
            key = (signal.base_signal.symbol, signal.source_pattern.pattern_type)
            signal_groups[key].append(signal)
        
        # For each group, keep only the highest quality signal
        final_signals = []
        for group_signals in signal_groups.values():
            if len(group_signals) == 1:
                final_signals.extend(group_signals)
            else:
                # Sort by quality score and keep the best
                best_signal = max(group_signals, key=lambda s: s.signal_quality_score)
                final_signals.append(best_signal)
        
        # Sort all signals by quality score
        return sorted(final_signals, key=lambda s: s.signal_quality_score, reverse=True)