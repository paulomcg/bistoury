"""
Signal Manager Aggregation Engine - Task 9.2

Core mathematical signal aggregation and conflict resolution.
Implements Phase 1 bootstrap strategy with proven quantitative finance algorithms.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

from .models import (
    AggregatedSignal,
    SignalWeight,
    SignalConflict,
    ConflictType,
    ConflictResolution,
    SignalQuality,
    SignalQualityGrade,
    SignalManagerConfiguration,
)
from ..models.signals import TradingSignal, SignalDirection, RiskLevel
from ..strategies.signal_generator import GeneratedSignal
from ..strategies.narrative_generator import TradingNarrative


@dataclass
class SignalInput:
    """Container for signal and associated narrative"""
    signal: GeneratedSignal
    narrative: TradingNarrative
    strategy_id: str
    timestamp: datetime
    weight: float = 1.0
    
    @property
    def confidence(self) -> float:
        """Get signal confidence from pattern score"""
        return float(self.signal.pattern_score.confidence_score)
    
    @property
    def direction(self):
        """Get signal direction from base signal"""
        return self.signal.base_signal.direction


class ConflictResolver:
    """Resolves conflicts between contradictory trading signals"""
    
    def __init__(self, config: SignalManagerConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_conflicts(self, signals: List[SignalInput]) -> List[SignalConflict]:
        """Detect conflicts between signals"""
        conflicts = []
        
        for i, signal_a in enumerate(signals):
            for j, signal_b in enumerate(signals[i + 1:], i + 1):
                conflict = self._analyze_signal_pair(signal_a, signal_b)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _analyze_signal_pair(self, signal_a: SignalInput, signal_b: SignalInput) -> Optional[SignalConflict]:
        """Analyze two signals for conflicts"""
        conflicts_detected = []
        
        # Direction conflicts
        direction_conflict = self._check_direction_conflict(signal_a, signal_b)
        if direction_conflict:
            conflicts_detected.append(direction_conflict)
        
        # Confidence conflicts (large disparities)
        confidence_conflict = self._check_confidence_conflict(signal_a, signal_b)
        if confidence_conflict:
            conflicts_detected.append(confidence_conflict)
        
        # Timing conflicts (different entry/exit timing)
        timing_conflict = self._check_timing_conflict(signal_a, signal_b)
        if timing_conflict:
            conflicts_detected.append(timing_conflict)
        
        # Return the most severe conflict
        if conflicts_detected:
            return max(conflicts_detected, key=lambda c: c.severity)
        
        return None
    
    def _check_direction_conflict(self, signal_a: SignalInput, signal_b: SignalInput) -> Optional[SignalConflict]:
        """Check for direction conflicts between signals"""
        dir_a = signal_a.direction
        dir_b = signal_b.direction
        
        # Check for opposing directions
        opposing_pairs = [
            (SignalDirection.BUY, SignalDirection.SELL),
            (SignalDirection.BUY, SignalDirection.STRONG_SELL),
            (SignalDirection.STRONG_BUY, SignalDirection.SELL),
            (SignalDirection.STRONG_BUY, SignalDirection.STRONG_SELL),
        ]
        
        if (dir_a, dir_b) in opposing_pairs or (dir_b, dir_a) in opposing_pairs:
            # Calculate severity based on confidence levels
            avg_confidence = (signal_a.confidence + signal_b.confidence) / 2
            severity = min(1.0, avg_confidence / 100.0)
            
            return SignalConflict(
                strategy_a=signal_a.strategy_id,
                strategy_b=signal_b.strategy_id,
                conflict_type=ConflictType.DIRECTION,
                severity=severity,
                resolution=self.config.default_conflict_resolution,
                signal_a_direction=dir_a,
                signal_b_direction=dir_b,
                signal_a_confidence=signal_a.confidence,
                signal_b_confidence=signal_b.confidence
            )
        
        return None
    
    def _check_confidence_conflict(self, signal_a: SignalInput, signal_b: SignalInput) -> Optional[SignalConflict]:
        """Check for major confidence disparities"""
        conf_diff = abs(signal_a.confidence - signal_b.confidence)
        
        # Conflict if confidence difference > 40%
        if conf_diff > 40.0:
            severity = min(1.0, conf_diff / 100.0)
            
            return SignalConflict(
                strategy_a=signal_a.strategy_id,
                strategy_b=signal_b.strategy_id,
                conflict_type=ConflictType.CONFIDENCE,
                severity=severity,
                resolution=self.config.default_conflict_resolution,
                signal_a_direction=signal_a.direction,
                signal_b_direction=signal_b.direction,
                signal_a_confidence=signal_a.confidence,
                signal_b_confidence=signal_b.confidence
            )
        
        return None
    
    def _check_timing_conflict(self, signal_a: SignalInput, signal_b: SignalInput) -> Optional[SignalConflict]:
        """Check for timing conflicts between signals"""
        time_diff = abs((signal_a.timestamp - signal_b.timestamp).total_seconds())
        
        # Conflict if signals are too far apart (>5 minutes)
        if time_diff > 300:  # 5 minutes
            severity = min(1.0, time_diff / 1800)  # Max severity at 30 minutes
            
            return SignalConflict(
                strategy_a=signal_a.strategy_id,
                strategy_b=signal_b.strategy_id,
                conflict_type=ConflictType.TIMING,
                severity=severity * 0.5,  # Timing conflicts are less severe
                resolution=self.config.default_conflict_resolution,
                signal_a_direction=signal_a.direction,
                signal_b_direction=signal_b.direction,
                signal_a_confidence=signal_a.confidence,
                signal_b_confidence=signal_b.confidence
            )
        
        return None
    
    def resolve_conflicts(self, signals: List[SignalInput], conflicts: List[SignalConflict]) -> Tuple[SignalDirection, float, Dict[str, float]]:
        """Resolve conflicts and return aggregated direction and confidence"""
        if not signals:
            return SignalDirection.HOLD, 0.0, {}
        
        if not conflicts:
            # No conflicts - simple weighted average
            return self._simple_aggregation(signals)
        
        # Check if conflicts are too severe
        severe_conflicts = [c for c in conflicts if c.severity > self.config.max_conflict_severity]
        if severe_conflicts:
            self.logger.warning(f"Severe conflicts detected, defaulting to HOLD: {len(severe_conflicts)} conflicts")
            return SignalDirection.HOLD, 50.0, {}
        
        # Resolve based on configured method
        if self.config.default_conflict_resolution == ConflictResolution.WEIGHTED_AVERAGE:
            return self._weighted_average_resolution(signals, conflicts)
        elif self.config.default_conflict_resolution == ConflictResolution.HIGHEST_CONFIDENCE:
            return self._highest_confidence_resolution(signals)
        elif self.config.default_conflict_resolution == ConflictResolution.HOLD_SIGNAL:
            return SignalDirection.HOLD, 50.0, {}
        else:
            # Default to weighted average
            return self._weighted_average_resolution(signals, conflicts)
    
    def _simple_aggregation(self, signals: List[SignalInput]) -> Tuple[SignalDirection, float, Dict[str, float]]:
        """Simple weighted aggregation without conflicts"""
        if not signals:
            return SignalDirection.HOLD, 0.0, {}
        
        # Convert directions to numerical values for averaging
        direction_values = {
            SignalDirection.STRONG_SELL: -2,
            SignalDirection.SELL: -1,
            SignalDirection.HOLD: 0,
            SignalDirection.BUY: 1,
            SignalDirection.STRONG_BUY: 2
        }
        
        total_weighted_direction = 0.0
        total_weighted_confidence = 0.0
        total_weights = 0.0
        strategy_weights = {}
        
        for signal_input in signals:
            weight = signal_input.weight
            direction_val = direction_values.get(signal_input.direction, 0)
            confidence = signal_input.confidence
            
            total_weighted_direction += direction_val * weight * confidence / 100.0
            total_weighted_confidence += confidence * weight
            total_weights += weight
            strategy_weights[signal_input.strategy_id] = weight
        
        if total_weights == 0:
            return SignalDirection.HOLD, 0.0, {}
        
        # Calculate final direction and confidence
        avg_direction = total_weighted_direction / total_weights
        avg_confidence = total_weighted_confidence / total_weights
        
        # Convert back to direction enum
        if avg_direction <= -1.5:
            final_direction = SignalDirection.STRONG_SELL
        elif avg_direction <= -0.5:
            final_direction = SignalDirection.SELL
        elif avg_direction >= 1.5:
            final_direction = SignalDirection.STRONG_BUY
        elif avg_direction >= 0.5:
            final_direction = SignalDirection.BUY
        else:
            final_direction = SignalDirection.HOLD
        
        return final_direction, avg_confidence, strategy_weights
    
    def _weighted_average_resolution(self, signals: List[SignalInput], conflicts: List[SignalConflict]) -> Tuple[SignalDirection, float, Dict[str, float]]:
        """Resolve conflicts using weighted average with conflict penalties"""
        # Apply conflict penalties to confidence
        penalty_by_strategy = defaultdict(float)
        
        for conflict in conflicts:
            penalty = conflict.severity * 10.0  # Up to 10% penalty
            penalty_by_strategy[conflict.strategy_a] += penalty
            penalty_by_strategy[conflict.strategy_b] += penalty
        
        # Apply penalties directly to calculations without creating new objects
        # Use simple aggregation logic but apply penalties to confidence values
        direction_values = {
            SignalDirection.STRONG_SELL: -2,
            SignalDirection.SELL: -1,
            SignalDirection.HOLD: 0,
            SignalDirection.BUY: 1,
            SignalDirection.STRONG_BUY: 2
        }
        
        total_weighted_direction = 0.0
        total_weighted_confidence = 0.0
        total_weights = 0.0
        strategy_weights = {}
        
        for signal_input in signals:
            weight = signal_input.weight
            direction_val = direction_values.get(signal_input.direction, 0)
            
            # Apply conflict penalty
            penalty = penalty_by_strategy.get(signal_input.strategy_id, 0.0)
            adjusted_confidence = max(0.0, signal_input.confidence - penalty)
            
            total_weighted_direction += direction_val * weight * adjusted_confidence / 100.0
            total_weighted_confidence += adjusted_confidence * weight
            total_weights += weight
            strategy_weights[signal_input.strategy_id] = weight
        
        if total_weights == 0:
            return SignalDirection.HOLD, 0.0, {}
        
        # Calculate final direction and confidence
        avg_direction = total_weighted_direction / total_weights
        avg_confidence = total_weighted_confidence / total_weights
        
        # Convert back to direction enum
        if avg_direction > 1.5:
            final_direction = SignalDirection.STRONG_BUY
        elif avg_direction > 0.5:
            final_direction = SignalDirection.BUY
        elif avg_direction < -1.5:
            final_direction = SignalDirection.STRONG_SELL
        elif avg_direction < -0.5:
            final_direction = SignalDirection.SELL
        else:
            final_direction = SignalDirection.HOLD
        
        return final_direction, avg_confidence, strategy_weights
    
    def _highest_confidence_resolution(self, signals: List[SignalInput]) -> Tuple[SignalDirection, float, Dict[str, float]]:
        """Resolve conflicts by taking the highest confidence signal"""
        if not signals:
            return SignalDirection.HOLD, 0.0, {}
        
        # Find signal with highest confidence
        best_signal = max(signals, key=lambda s: s.confidence)
        
        strategy_weights = {s.strategy_id: 0.0 for s in signals}
        strategy_weights[best_signal.strategy_id] = 1.0
        
        return best_signal.direction, best_signal.confidence, strategy_weights


class SignalValidator:
    """Validates and filters signals based on quality criteria"""
    
    def __init__(self, config: SignalManagerConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_signal(self, signal_input: SignalInput) -> bool:
        """Validate individual signal meets minimum criteria"""
        # Check basic signal validity
        if not self._is_signal_valid(signal_input.signal):
            self.logger.debug(f"Signal validation failed for {signal_input.strategy_id}: invalid signal")
            return False
        
        # Check confidence threshold
        if signal_input.confidence < self.config.confidence_threshold:
            self.logger.debug(f"Signal validation failed for {signal_input.strategy_id}: confidence {signal_input.confidence} < {self.config.confidence_threshold}")
            return False
        
        # Check signal age
        age = datetime.now(timezone.utc) - signal_input.timestamp
        if age > self.config.max_signal_age:
            self.logger.debug(f"Signal validation failed for {signal_input.strategy_id}: age {age} > {self.config.max_signal_age}")
            return False
        
        return True
    
    def _is_signal_valid(self, signal: GeneratedSignal) -> bool:
        """Check if signal has valid data"""
        try:
            # Check required fields exist
            if not hasattr(signal, 'base_signal') or not signal.base_signal:
                return False
            
            if not hasattr(signal, 'pattern_score') or not signal.pattern_score:
                return False
            
            # Check direction exists
            if not hasattr(signal.base_signal, 'direction') or not signal.base_signal.direction:
                return False
            
            # Check confidence range
            confidence = signal.pattern_score.confidence_score
            if confidence < 0 or confidence > 100:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return False
    
    def filter_signals(self, signals: List[SignalInput]) -> List[SignalInput]:
        """Filter signals based on validation criteria"""
        valid_signals = []
        
        for signal_input in signals:
            if self.validate_signal(signal_input):
                valid_signals.append(signal_input)
            else:
                self.logger.debug(f"Filtered out signal from {signal_input.strategy_id}")
        
        return valid_signals
    
    def deduplicate_signals(self, signals: List[SignalInput]) -> List[SignalInput]:
        """Remove duplicate signals from same strategy"""
        seen_strategies = set()
        deduplicated = []
        
        # Sort by confidence (highest first) then by timestamp (newest first)
        sorted_signals = sorted(signals, 
                              key=lambda s: (-s.confidence, -s.timestamp.timestamp()))
        
        for signal_input in sorted_signals:
            if signal_input.strategy_id not in seen_strategies:
                deduplicated.append(signal_input)
                seen_strategies.add(signal_input.strategy_id)
            else:
                self.logger.debug(f"Deduplicated signal from {signal_input.strategy_id}")
        
        return deduplicated


class SignalScorer:
    """Scores composite signal quality and tradability"""
    
    def __init__(self, config: SignalManagerConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_quality(self, 
                         signals: List[SignalInput],
                         conflicts: List[SignalConflict],
                         final_confidence: float,
                         strategy_weights: Dict[str, float]) -> SignalQuality:
        """Calculate comprehensive signal quality score"""
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(signals, conflicts)
        
        # Use provided confidence score
        confidence_score = final_confidence
        
        # Calculate quality using existing method
        return SignalQuality.calculate_quality(
            consensus_score=consensus_score,
            confidence_score=confidence_score,
            conflicts=conflicts,
            contributing_strategies=len(signals)
        )
    
    def _calculate_consensus_score(self, signals: List[SignalInput], conflicts: List[SignalConflict]) -> float:
        """Calculate how well strategies agree"""
        if len(signals) <= 1:
            return 100.0 if signals else 0.0
        
        # Base score starts high
        consensus_score = 100.0
        
        # Penalty for each conflict
        for conflict in conflicts:
            penalty = conflict.severity * 15.0  # Up to 15% penalty per conflict
            consensus_score -= penalty
        
        # Bonus for multiple strategies in agreement
        if len(signals) >= 3:
            consensus_score += 10.0  # Bonus for multiple strategies
        
        return max(0.0, min(100.0, consensus_score))


class WeightManager:
    """Manages dynamic strategy weights based on performance"""
    
    def __init__(self, config: SignalManagerConfiguration):
        self.config = config
        self.weights: Dict[str, SignalWeight] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.performance_window))
        self.last_weight_update = datetime.now(timezone.utc)
        self.logger = logging.getLogger(__name__)
    
    def get_weight(self, strategy_id: str) -> SignalWeight:
        """Get current weight for strategy"""
        if strategy_id not in self.weights:
            base_weight = self.config.get_strategy_weight(strategy_id)
            self.weights[strategy_id] = SignalWeight(
                strategy_id=strategy_id,
                base_weight=base_weight
            )
        
        return self.weights[strategy_id]
    
    def update_performance(self, strategy_id: str, success: bool):
        """Update strategy performance tracking"""
        weight = self.get_weight(strategy_id)
        weight.update_performance(success)
        self.weights[strategy_id] = weight
        
        # Track performance history
        self.performance_history[strategy_id].append({
            'timestamp': datetime.now(timezone.utc),
            'success': success
        })
        
        self.logger.debug(f"Updated performance for {strategy_id}: success={success}, rate={weight.recent_success_rate}")
    
    def should_update_weights(self) -> bool:
        """Check if weights should be updated"""
        if not self.config.auto_adjust_weights:
            return False
        
        time_since_update = datetime.now(timezone.utc) - self.last_weight_update
        return time_since_update >= self.config.weight_adjustment_frequency
    
    def update_weights(self):
        """Update all strategy weights based on recent performance"""
        if not self.should_update_weights():
            return
        
        self.logger.info("Updating strategy weights based on performance")
        
        for strategy_id, weight in self.weights.items():
            # Weight already updated via update_performance
            self.logger.info(f"Strategy {strategy_id}: weight={weight.final_weight:.3f}, success_rate={weight.recent_success_rate:.3f}")
        
        self.last_weight_update = datetime.now(timezone.utc)
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get all current strategy weights"""
        return {strategy_id: weight.final_weight for strategy_id, weight in self.weights.items()}
    
    def update_weight(self, strategy_id: str, new_weight: float):
        """Manually update base weight for a strategy"""
        if not (0.0 <= new_weight <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {new_weight}")
        
        weight = self.get_weight(strategy_id)
        weight.base_weight = new_weight
        weight.model_post_init(None)  # Recalculate final weight
        self.weights[strategy_id] = weight
        
        self.logger.info(f"Manually updated base weight for {strategy_id} to {new_weight}, final weight: {weight.final_weight}")
        
        # Update config for persistence
        self.config.update_strategy_weight(strategy_id, new_weight)


class SignalAggregator:
    """Main signal aggregation engine combining all components"""
    
    def __init__(self, config: Optional[SignalManagerConfiguration] = None):
        self.config = config or SignalManagerConfiguration()
        self.conflict_resolver = ConflictResolver(self.config)
        self.validator = SignalValidator(self.config)
        self.scorer = SignalScorer(self.config)
        self.weight_manager = WeightManager(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_signals_processed = 0
        self.total_conflicts_resolved = 0
        self.total_signals_filtered = 0
        
    async def aggregate_signals(self, 
                               signal_inputs: List[SignalInput],
                               symbol: str) -> Optional[AggregatedSignal]:
        """Main aggregation method - processes signals into aggregated result"""
        
        if not signal_inputs:
            self.logger.debug("No signals to aggregate")
            return None
        
        self.logger.debug(f"Aggregating {len(signal_inputs)} signals for {symbol}")
        
        # 1. Validate and filter signals
        valid_signals = self.validator.filter_signals(signal_inputs)
        if not valid_signals:
            self.logger.debug("No valid signals after filtering")
            self.total_signals_filtered += len(signal_inputs)
            return None
        
        # 2. Deduplicate signals
        unique_signals = self.validator.deduplicate_signals(valid_signals)
        
        # 3. Check minimum strategy requirement
        if len(unique_signals) < self.config.min_strategies_for_signal:
            self.logger.debug(f"Insufficient strategies: {len(unique_signals)} < {self.config.min_strategies_for_signal}")
            return None
        
        # 4. Apply dynamic weights
        self._apply_dynamic_weights(unique_signals)
        
        # 5. Detect conflicts
        conflicts = self.conflict_resolver.detect_conflicts(unique_signals)
        self.total_conflicts_resolved += len(conflicts)
        
        # 6. Resolve conflicts and aggregate
        final_direction, final_confidence, strategy_weights = self.conflict_resolver.resolve_conflicts(unique_signals, conflicts)
        
        # 7. Calculate quality score
        quality = self.scorer.calculate_quality(unique_signals, conflicts, final_confidence, strategy_weights)
        
        # 8. Check if signal meets trading thresholds
        if not quality.is_tradeable:
            self.logger.debug(f"Signal not tradeable: quality={quality.overall_score}, grade={quality.grade}")
            return None
        
        # 9. Create aggregated signal
        aggregated_signal = await self._create_aggregated_signal(
            symbol=symbol,
            direction=final_direction,
            confidence=final_confidence,
            quality=quality,
            conflicts=conflicts,
            strategy_weights=strategy_weights,
            contributing_strategies=[s.strategy_id for s in unique_signals],
            source_signals=unique_signals
        )
        
        self.total_signals_processed += 1
        self.logger.info(f"Aggregated signal created: {final_direction} with {final_confidence:.1f}% confidence, quality {quality.grade}")
        
        return aggregated_signal
    
    def _apply_dynamic_weights(self, signals: List[SignalInput]):
        """Apply current dynamic weights to signals"""
        for signal_input in signals:
            weight_obj = self.weight_manager.get_weight(signal_input.strategy_id)
            signal_input.weight = weight_obj.final_weight
    
    async def _create_aggregated_signal(self,
                                      symbol: str,
                                      direction: SignalDirection,
                                      confidence: float,
                                      quality: SignalQuality,
                                      conflicts: List[SignalConflict],
                                      strategy_weights: Dict[str, float],
                                      contributing_strategies: List[str],
                                      source_signals: List[SignalInput]) -> AggregatedSignal:
        """Create the final aggregated signal"""
        
        # Calculate risk level based on confidence and quality
        risk_level = self._calculate_risk_level(confidence, quality, conflicts)
        
        # Calculate signal weight (overall strength)
        signal_weight = min(1.0, (confidence / 100.0) * (quality.overall_score / 100.0))
        
        # Set expiry time
        expiry_time = datetime.now(timezone.utc) + self.config.max_signal_age
        
        # Create brief narrative summary
        narrative_summary = self._create_narrative_summary(source_signals, direction, confidence)
        
        return AggregatedSignal(
            direction=direction,
            confidence=confidence,
            weight=signal_weight,
            contributing_strategies=contributing_strategies,
            strategy_weights=strategy_weights,
            conflicts=conflicts,
            quality=quality,
            risk_level=risk_level,
            expiry=expiry_time,
            narrative_summary=narrative_summary
        )
    
    def _calculate_risk_level(self, confidence: float, quality: SignalQuality, conflicts: List[SignalConflict]) -> RiskLevel:
        """Calculate risk level based on signal characteristics"""
        # Start with confidence-based risk
        if confidence >= 85 and quality.grade in [SignalQualityGrade.A_PLUS, SignalQualityGrade.A]:
            base_risk = RiskLevel.LOW
        elif confidence >= 70 and quality.grade in [SignalQualityGrade.A, SignalQualityGrade.B_PLUS]:
            base_risk = RiskLevel.MEDIUM
        elif confidence >= 60:
            base_risk = RiskLevel.HIGH
        else:
            base_risk = RiskLevel.VERY_HIGH
        
        # Increase risk for conflicts
        major_conflicts = len([c for c in conflicts if c.severity > 0.7])
        if major_conflicts > 0:
            if base_risk == RiskLevel.LOW:
                base_risk = RiskLevel.MEDIUM
            elif base_risk == RiskLevel.MEDIUM:
                base_risk = RiskLevel.HIGH
            elif base_risk == RiskLevel.HIGH:
                base_risk = RiskLevel.VERY_HIGH
        
        return base_risk
    
    def _create_narrative_summary(self, signals: List[SignalInput], direction: SignalDirection, confidence: float) -> str:
        """Create brief narrative summary for aggregated signal"""
        strategy_names = [s.strategy_id.replace('_', ' ').title() for s in signals]
        
        if len(strategy_names) == 1:
            strategy_text = strategy_names[0]
        elif len(strategy_names) == 2:
            strategy_text = f"{strategy_names[0]} and {strategy_names[1]}"
        else:
            strategy_text = f"{', '.join(strategy_names[:-1])}, and {strategy_names[-1]}"
        
        direction_text = direction.value.replace('_', ' ').title()
        
        return f"{direction_text} signal from {strategy_text} with {confidence:.1f}% confidence"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get aggregator performance statistics"""
        return {
            "total_signals_processed": self.total_signals_processed,
            "total_conflicts_resolved": self.total_conflicts_resolved,
            "total_signals_filtered": self.total_signals_filtered,
            "strategy_weights": self.weight_manager.get_all_weights(),
            "strategy_performance": {
                strategy_id: {
                    "success_rate": weight.recent_success_rate,
                    "total_signals": weight.total_signals,
                    "final_weight": weight.final_weight
                }
                for strategy_id, weight in self.weight_manager.weights.items()
            }
        }
    
    def update_strategy_performance(self, strategy_id: str, success: bool):
        """Update strategy performance for weight adjustment"""
        self.weight_manager.update_performance(strategy_id, success)
        
        # Check if weights should be updated
        if self.weight_manager.should_update_weights():
            self.weight_manager.update_weights() 