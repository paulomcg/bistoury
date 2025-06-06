"""
Signal Manager Models - Task 9.1

Core data models for signal aggregation, weighting, and temporal storage.
Implements Phase 1 mathematical foundation with narrative preservation
for future Phase 2-3 evolution.
"""

from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Deque, Tuple
from pydantic import BaseModel, Field, ConfigDict, computed_field

from ..models.signals import (
    TradingSignal, 
    SignalDirection, 
    Timeframe,
    RiskLevel
)
from ..strategies.signal_generator import GeneratedSignal
from ..strategies.narrative_generator import TradingNarrative


class ConflictType(str, Enum):
    """Types of signal conflicts detected during aggregation"""
    DIRECTION = "direction"          # Opposing buy/sell directions
    TIMING = "timing"               # Conflicting entry/exit timing
    CONFIDENCE = "confidence"       # Major confidence disparities
    MAGNITUDE = "magnitude"         # Conflicting position sizes
    TIMEFRAME = "timeframe"         # Cross-timeframe conflicts
    STRATEGY = "strategy"           # Strategy-specific conflicts


class ConflictResolution(str, Enum):
    """Methods for resolving signal conflicts"""
    WEIGHTED_AVERAGE = "weighted_average"     # Use weighted confidence
    HIGHEST_CONFIDENCE = "highest_confidence" # Take most confident signal
    TIMEFRAME_PRIORITY = "timeframe_priority" # Prioritize by timeframe
    STRATEGY_PRIORITY = "strategy_priority"   # Prioritize by strategy performance
    MANUAL_OVERRIDE = "manual_override"       # Require manual intervention
    IGNORE_CONFLICT = "ignore_conflict"       # Proceed with aggregation
    HOLD_SIGNAL = "hold_signal"              # Generate HOLD when conflicted


class SignalQualityGrade(str, Enum):
    """Quality grades for aggregated signals"""
    A_PLUS = "A+"     # Exceptional quality, high confidence across strategies
    A = "A"           # Excellent quality, strong consensus
    B_PLUS = "B+"     # Good quality, moderate consensus
    B = "B"           # Fair quality, some conflicts resolved
    C = "C"           # Poor quality, significant conflicts
    D = "D"           # Very poor quality, major conflicts
    F = "F"           # Failed quality, conflicting signals


class SignalConflict(BaseModel):
    """Represents a conflict between strategy signals"""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    strategy_a: str = Field(..., description="First conflicting strategy ID")
    strategy_b: str = Field(..., description="Second conflicting strategy ID")
    conflict_type: ConflictType = Field(..., description="Type of conflict detected")
    severity: float = Field(..., ge=0.0, le=1.0, description="Conflict severity (0-1)")
    resolution: ConflictResolution = Field(..., description="How conflict was resolved")
    
    # Conflict details
    signal_a_direction: SignalDirection = Field(..., description="Direction from strategy A")
    signal_b_direction: SignalDirection = Field(..., description="Direction from strategy B")
    signal_a_confidence: float = Field(..., description="Confidence from strategy A")
    signal_b_confidence: float = Field(..., description="Confidence from strategy B")
    
    # Resolution metrics
    confidence_impact: float = Field(default=0.0, description="Impact on final confidence")
    resolution_weight: float = Field(default=1.0, description="Weight applied during resolution")
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def model_post_init(self, __context):
        """Calculate derived conflict metrics"""
        # Calculate severity based on confidence difference and direction opposition
        if self.conflict_type == ConflictType.DIRECTION:
            opposing_directions = (
                (self.signal_a_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY] and
                 self.signal_b_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]) or
                (self.signal_a_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL] and
                 self.signal_b_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
            )
            if opposing_directions:
                # Severity based on how confident both signals are
                avg_confidence = (self.signal_a_confidence + self.signal_b_confidence) / 2
                self.severity = min(1.0, avg_confidence / 100.0)
            else:
                self.severity = 0.2  # Minor direction conflict
        
        elif self.conflict_type == ConflictType.CONFIDENCE:
            # Severity based on confidence spread
            confidence_diff = abs(self.signal_a_confidence - self.signal_b_confidence)
            self.severity = min(1.0, confidence_diff / 100.0)
        
        # Calculate confidence impact (penalty for conflicts)
        self.confidence_impact = self.severity * -10.0  # Up to -10% confidence penalty


class SignalWeight(BaseModel):
    """Dynamic weighting system for strategy importance"""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    strategy_id: str = Field(..., description="Strategy identifier")
    base_weight: float = Field(..., ge=0.0, le=1.0, description="Static strategy importance")
    
    # Performance-based modifiers
    performance_modifier: float = Field(default=1.0, description="Recent performance multiplier")
    confidence_modifier: float = Field(default=1.0, description="Signal confidence multiplier") 
    timeframe_modifier: float = Field(default=1.0, description="Timeframe priority multiplier")
    volume_modifier: float = Field(default=1.0, description="Volume confirmation multiplier")
    
    # Calculated final weight
    final_weight: float = Field(default=0.0, description="Computed final weight")
    
    # Performance tracking
    recent_success_rate: float = Field(default=0.5, description="Recent success rate (0-1)")
    total_signals: int = Field(default=0, description="Total signals generated")
    successful_signals: int = Field(default=0, description="Successful signals count")
    
    # Timing and metadata
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def model_post_init(self, __context):
        """Calculate final weight from all modifiers"""
        self.final_weight = (
            self.base_weight * 
            self.performance_modifier * 
            self.confidence_modifier *
            self.timeframe_modifier *
            self.volume_modifier
        )
        
        # Ensure weight stays within bounds
        self.final_weight = max(0.0, min(1.0, self.final_weight))
    
    def update_performance(self, success: bool):
        """Update performance tracking with new signal result"""
        self.total_signals += 1
        if success:
            self.successful_signals += 1
            
        # Calculate recent success rate (last 50 signals or total if less)
        if self.total_signals > 0:
            self.recent_success_rate = self.successful_signals / self.total_signals
            
        # Update performance modifier based on success rate
        if self.recent_success_rate > 0.7:
            self.performance_modifier = 1.2  # Boost performing strategies
        elif self.recent_success_rate > 0.5:
            self.performance_modifier = 1.0  # Neutral
        elif self.recent_success_rate > 0.3:
            self.performance_modifier = 0.8  # Reduce poor performers
        else:
            self.performance_modifier = 0.5  # Significantly reduce failing strategies
            
        self.last_updated = datetime.now(timezone.utc)
        self.model_post_init(None)  # Recalculate final weight


class SignalQuality(BaseModel):
    """Quality assessment for aggregated signals"""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    # Core quality metrics
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score")
    grade: SignalQualityGrade = Field(..., description="Letter grade classification")
    
    # Component scores
    consensus_score: float = Field(..., description="Strategy consensus score")
    confidence_score: float = Field(..., description="Weighted confidence score")
    conflict_penalty: float = Field(..., description="Penalty from conflicts")
    temporal_consistency: float = Field(..., description="Consistency over time")
    
    # Quality factors
    contributing_strategies: int = Field(..., description="Number of contributing strategies")
    conflicts_detected: int = Field(..., description="Number of conflicts detected")
    major_conflicts: int = Field(..., description="Number of major conflicts")
    
    # Tradability assessment
    is_tradeable: bool = Field(..., description="Whether signal meets trading thresholds")
    confidence_threshold: float = Field(default=60.0, description="Minimum confidence for trading")
    quality_threshold: float = Field(default=70.0, description="Minimum quality for trading")
    
    @computed_field
    @property
    def quality_summary(self) -> str:
        """Human-readable quality summary"""
        if self.grade in [SignalQualityGrade.A_PLUS, SignalQualityGrade.A]:
            return f"Excellent signal quality ({self.grade}) with {self.contributing_strategies} strategies in consensus"
        elif self.grade in [SignalQualityGrade.B_PLUS, SignalQualityGrade.B]:
            return f"Good signal quality ({self.grade}) with minor conflicts resolved"
        elif self.grade == SignalQualityGrade.C:
            return f"Fair signal quality ({self.grade}) with significant conflicts"
        else:
            return f"Poor signal quality ({self.grade}) - not recommended for trading"
    
    @classmethod
    def calculate_quality(cls,
                         consensus_score: float,
                         confidence_score: float,
                         conflicts: List[SignalConflict],
                         contributing_strategies: int) -> "SignalQuality":
        """Calculate signal quality from component metrics"""
        
        # Calculate conflict penalty
        conflict_penalty = 0.0
        major_conflicts = 0
        for conflict in conflicts:
            conflict_penalty += conflict.severity * 5.0  # Up to 5% penalty per conflict
            if conflict.severity > 0.7:
                major_conflicts += 1
        
        # Calculate temporal consistency (placeholder for Phase 1)
        temporal_consistency = 85.0  # Will be enhanced in Phase 2
        
        # Calculate overall score
        overall_score = (
            consensus_score * 0.3 +
            confidence_score * 0.4 +
            temporal_consistency * 0.2 +
            (max(0, 100 - conflict_penalty) * 0.1)
        )
        
        # Determine grade
        if overall_score >= 90:
            grade = SignalQualityGrade.A_PLUS
        elif overall_score >= 80:
            grade = SignalQualityGrade.A
        elif overall_score >= 75:
            grade = SignalQualityGrade.B_PLUS
        elif overall_score >= 65:
            grade = SignalQualityGrade.B
        elif overall_score >= 50:
            grade = SignalQualityGrade.C
        elif overall_score >= 30:
            grade = SignalQualityGrade.D
        else:
            grade = SignalQualityGrade.F
        
        # Determine tradability
        is_tradeable = (
            overall_score >= 70.0 and
            confidence_score >= 60.0 and
            major_conflicts == 0 and
            contributing_strategies >= 2
        )
        
        return cls(
            overall_score=overall_score,
            grade=grade,
            consensus_score=consensus_score,
            confidence_score=confidence_score,
            conflict_penalty=conflict_penalty,
            temporal_consistency=temporal_consistency,
            contributing_strategies=contributing_strategies,
            conflicts_detected=len(conflicts),
            major_conflicts=major_conflicts,
            is_tradeable=is_tradeable
        )


class AggregatedSignal(BaseModel):
    """Mathematical aggregation of multiple strategy signals"""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    # Core signal properties
    direction: SignalDirection = Field(..., description="Aggregated signal direction")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Weighted confidence score")
    weight: float = Field(..., ge=0.0, le=1.0, description="Overall signal strength")
    
    # Aggregation metadata
    contributing_strategies: List[str] = Field(..., description="Contributing strategy IDs")
    strategy_weights: Dict[str, float] = Field(..., description="Individual strategy weights")
    conflicts: List[SignalConflict] = Field(default_factory=list, description="Detected conflicts")
    
    # Quality assessment
    quality: SignalQuality = Field(..., description="Signal quality assessment")
    
    # Risk and execution
    risk_level: RiskLevel = Field(..., description="Aggregated risk level")
    position_size_multiplier: float = Field(default=1.0, description="Position sizing adjustment")
    
    # Timing and expiry
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expiry: datetime = Field(..., description="Signal expiration time")
    
    # Phase 2 preparation - narrative context preservation
    source_narratives: List[str] = Field(default_factory=list, description="Narrative IDs for Phase 2")
    narrative_summary: Optional[str] = Field(None, description="Brief narrative summary")
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        return datetime.now(timezone.utc) >= self.expiry
    
    @computed_field
    @property
    def time_to_expiry(self) -> timedelta:
        """Time remaining until expiry"""
        return self.expiry - datetime.now(timezone.utc)
    
    @computed_field 
    @property
    def consensus_level(self) -> str:
        """Describe the level of strategy consensus"""
        if len(self.conflicts) == 0:
            return "full_consensus"
        elif len([c for c in self.conflicts if c.severity > 0.7]) == 0:
            return "strong_consensus"
        elif len(self.conflicts) <= 2:
            return "moderate_consensus"
        else:
            return "weak_consensus"
    
    def to_trading_signal(self) -> TradingSignal:
        """Convert to basic TradingSignal for backward compatibility"""
        return TradingSignal(
            symbol="",  # Will be set by caller
            direction=self.direction,
            confidence=self.confidence,
            timeframe=Timeframe.MULTI,  # Aggregated across timeframes
            reasoning=f"Aggregated from {len(self.contributing_strategies)} strategies",
            risk_level=self.risk_level,
            timestamp=self.timestamp
        )


class TemporalSignalBuffer:
    """Stores signals and narratives for temporal analysis and Phase 2 evolution"""
    
    def __init__(self, max_age: timedelta = timedelta(minutes=15)):
        self.max_age = max_age
        self.signals: Deque[AggregatedSignal] = deque()
        self.narratives: Deque[TradingNarrative] = deque() 
        self.signal_narrative_map: Dict[str, str] = {}  # signal_id -> narrative_id
    
    def add_signal(self, signal: AggregatedSignal, narrative: TradingNarrative):
        """Add signal and preserve narrative for future evolution"""
        # Store signal
        self.signals.append(signal)
        
        # Store narrative with timestamp-based ID
        narrative_id = f"narrative_{signal.timestamp.isoformat()}"
        self.narratives.append(narrative)
        
        # Map signal to narrative for Phase 2 retrieval
        signal_id = f"signal_{signal.timestamp.isoformat()}"
        self.signal_narrative_map[signal_id] = narrative_id
        
        # Add narrative ID to signal for future reference
        signal.source_narratives.append(narrative_id)
        
        # Cleanup expired entries
        self._cleanup_expired()
    
    def get_recent_signals(self, max_count: Optional[int] = None) -> List[AggregatedSignal]:
        """Get recent signals, optionally limited by count"""
        signals = list(self.signals)
        if max_count:
            signals = signals[-max_count:]
        return signals
    
    def get_narrative_timeline(self) -> List[TradingNarrative]:
        """Get chronological narrative evolution for Phase 2 analysis"""
        return list(self.narratives)
    
    def get_signal_with_narrative(self, signal_timestamp: datetime) -> Optional[Tuple[AggregatedSignal, TradingNarrative]]:
        """Get signal with its corresponding narrative"""
        signal_id = f"signal_{signal_timestamp.isoformat()}"
        narrative_id = self.signal_narrative_map.get(signal_id)
        
        if not narrative_id:
            return None
            
        # Find signal and narrative
        target_signal = None
        target_narrative = None
        
        for signal in self.signals:
            if signal.timestamp == signal_timestamp:
                target_signal = signal
                break
                
        for narrative in self.narratives:
            if f"narrative_{narrative.generation_timestamp.isoformat()}" == narrative_id:
                target_narrative = narrative
                break
                
        if target_signal and target_narrative:
            return (target_signal, target_narrative)
        return None
    
    def _cleanup_expired(self):
        """Remove signals and narratives older than max_age"""
        cutoff_time = datetime.now(timezone.utc) - self.max_age
        
        # Clean signals
        while self.signals and self.signals[0].timestamp < cutoff_time:
            expired_signal = self.signals.popleft()
            # Clean up mapping
            signal_id = f"signal_{expired_signal.timestamp.isoformat()}"
            self.signal_narrative_map.pop(signal_id, None)
        
        # Clean narratives
        while self.narratives and self.narratives[0].generation_timestamp < cutoff_time:
            self.narratives.popleft()
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring"""
        return {
            "signal_count": len(self.signals),
            "narrative_count": len(self.narratives),
            "oldest_signal": self.signals[0].timestamp if self.signals else None,
            "newest_signal": self.signals[-1].timestamp if self.signals else None,
            "buffer_age_minutes": self.max_age.total_seconds() / 60,
            "memory_usage_estimate": len(self.signals) * 1024 + len(self.narratives) * 4096  # Rough estimate
        }


class SignalManagerConfiguration(BaseModel):
    """Configuration for signal manager behavior"""
    
    model_config = ConfigDict(
        json_encoders={
            timedelta: lambda td: td.total_seconds(),
        }
    )
    
    # Strategy weights (can be updated dynamically)
    strategy_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "candlestick_strategy": 0.4,
            "funding_rate_strategy": 0.3,
            "order_flow_strategy": 0.2,
            "volume_profile_strategy": 0.1
        },
        description="Base weights for each strategy"
    )
    
    # Aggregation parameters
    min_strategies_for_signal: int = Field(default=2, description="Minimum strategies needed for signal")
    max_signal_age: timedelta = Field(default=timedelta(minutes=5), description="Maximum age for signal aggregation")
    confidence_threshold: float = Field(default=60.0, description="Minimum confidence for trading signals")
    quality_threshold: float = Field(default=70.0, description="Minimum quality for trading signals")
    
    # Conflict resolution
    default_conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.WEIGHTED_AVERAGE,
        description="Default method for resolving conflicts"
    )
    max_conflict_severity: float = Field(default=0.8, description="Maximum conflict severity to allow")
    
    # Temporal buffer settings
    temporal_buffer_age: timedelta = Field(default=timedelta(minutes=15), description="How long to keep signals/narratives")
    max_buffer_size: int = Field(default=1000, description="Maximum buffer size")
    
    # Performance tracking
    performance_window: int = Field(default=50, description="Number of recent signals for performance calculation")
    auto_adjust_weights: bool = Field(default=True, description="Automatically adjust strategy weights based on performance")
    weight_adjustment_frequency: timedelta = Field(default=timedelta(hours=1), description="How often to adjust weights")
    
    # Phase 2 preparation
    preserve_narratives: bool = Field(default=True, description="Store narratives for Phase 2 evolution")
    narrative_compression: bool = Field(default=False, description="Compress stored narratives")
    
    # Risk management
    max_position_size_multiplier: float = Field(default=2.0, description="Maximum position size multiplier")
    emergency_stop_threshold: float = Field(default=0.05, description="Emergency stop loss threshold")
    
    def update_strategy_weight(self, strategy_id: str, new_weight: float):
        """Update weight for a specific strategy"""
        if 0.0 <= new_weight <= 1.0:
            self.strategy_weights[strategy_id] = new_weight
        else:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {new_weight}")
    
    def get_strategy_weight(self, strategy_id: str) -> float:
        """Get weight for a specific strategy"""
        return self.strategy_weights.get(strategy_id, 0.1)  # Default weight if not found 