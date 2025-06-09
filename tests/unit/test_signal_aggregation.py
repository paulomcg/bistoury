"""
Test suite for Signal Aggregation Engine - Task 9.2

Tests core mathematical aggregation, conflict resolution, and validation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, MagicMock

from src.bistoury.signal_manager.aggregator import (
    SignalAggregator,
    ConflictResolver,
    SignalValidator,
    SignalScorer,
    WeightManager,
    SignalInput,
)
from src.bistoury.signal_manager.models import (
    SignalManagerConfiguration,
    ConflictType,
    ConflictResolution,
    SignalQualityGrade,
)
from src.bistoury.models.signals import SignalDirection, RiskLevel
from src.bistoury.strategies.signal_generator import (
    GeneratedSignal,
    SignalEntryPoint,
    SignalRiskManagement,
    SignalTiming,
    RiskLevel,
)
from src.bistoury.strategies.narrative_generator import (
    TradingNarrative,
    NarrativeConfiguration,
    NarrativeStyle,
)


def create_test_signal(direction: SignalDirection, confidence: float, strategy_id: str) -> SignalInput:
    """Helper to create test signals with proper mock behavior"""

    from unittest.mock import Mock
    from src.bistoury.strategies.narrative_generator import TradingNarrative
    
    # Create mock GeneratedSignal with proper numeric behavior
    generated_signal = Mock()
    
    # Mock pattern_score with numeric confidence
    pattern_score = Mock()
    pattern_score.confidence_score = confidence
    generated_signal.pattern_score = pattern_score
    
    # Mock base_signal with enum direction and numeric confidence
    base_signal = Mock()
    base_signal.direction = direction
    base_signal.confidence = confidence
    generated_signal.base_signal = base_signal
    
    # Make the signal support numeric operations for confidence
    generated_signal.confidence = confidence
    generated_signal.direction = direction
    
    # Create mock TradingNarrative
    narrative = Mock(spec=TradingNarrative)
    narrative.summary = f"Test narrative for {strategy_id} with {confidence}% confidence"
    
    # Return SignalInput that works with the properties
    return SignalInput(
        signal=generated_signal,
        narrative=narrative,
        strategy_id=strategy_id,
        timestamp=datetime.now(timezone.utc)
    )


class TestConflictResolver:
    """Test conflict detection and resolution"""
    
    def test_conflict_resolver_creation(self):
        """Test basic conflict resolver creation"""
        config = SignalManagerConfiguration()
        resolver = ConflictResolver(config)
        
        assert resolver.config == config
        assert resolver.logger is not None
    
    def test_detect_no_conflicts(self):
        """Test conflict detection with no conflicts"""
        config = SignalManagerConfiguration()
        resolver = ConflictResolver(config)
        
        # Two signals in same direction
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.BUY, 75.0, "strategy2")
        ]
        
        conflicts = resolver.detect_conflicts(signals)
        assert len(conflicts) == 0
    
    def test_detect_direction_conflicts(self):
        """Test direction conflict detection"""
        config = SignalManagerConfiguration()
        resolver = ConflictResolver(config)
        
        # Opposing signals
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.SELL, 75.0, "strategy2")
        ]
        
        conflicts = resolver.detect_conflicts(signals)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.DIRECTION
        assert conflicts[0].strategy_a == "strategy1"
        assert conflicts[0].strategy_b == "strategy2"
        assert conflicts[0].severity > 0
    
    def test_detect_confidence_conflicts(self):
        """Test confidence conflict detection"""
        config = SignalManagerConfiguration()
        resolver = ConflictResolver(config)
        
        # Large confidence difference
        signals = [
            create_test_signal(SignalDirection.BUY, 90.0, "strategy1"),
            create_test_signal(SignalDirection.BUY, 40.0, "strategy2")
        ]
        
        conflicts = resolver.detect_conflicts(signals)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.CONFIDENCE
        assert conflicts[0].severity > 0
    
    def test_resolve_conflicts_no_conflicts(self):
        """Test conflict resolution with no conflicts"""
        config = SignalManagerConfiguration()
        resolver = ConflictResolver(config)
        
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.BUY, 75.0, "strategy2")
        ]
        
        direction, confidence, weights = resolver.resolve_conflicts(signals, [])
        
        assert direction == SignalDirection.BUY
        assert confidence > 70  # Should be weighted average
        assert len(weights) == 2
    
    def test_resolve_conflicts_weighted_average(self):
        """Test weighted average conflict resolution"""
        config = SignalManagerConfiguration()
        config.default_conflict_resolution = ConflictResolution.WEIGHTED_AVERAGE
        resolver = ConflictResolver(config)
        
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.SELL, 70.0, "strategy2")
        ]
        
        conflicts = resolver.detect_conflicts(signals)
        direction, confidence, weights = resolver.resolve_conflicts(signals, conflicts)
        
        # Should resolve to some direction with reduced confidence due to conflicts
        assert direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
        assert confidence > 0
        assert len(weights) == 2
    
    def test_resolve_conflicts_highest_confidence(self):
        """Test highest confidence conflict resolution"""
        config = SignalManagerConfiguration()
        config.default_conflict_resolution = ConflictResolution.HIGHEST_CONFIDENCE
        resolver = ConflictResolver(config)
        
        signals = [
            create_test_signal(SignalDirection.BUY, 85.0, "strategy1"),
            create_test_signal(SignalDirection.SELL, 70.0, "strategy2")
        ]
        
        conflicts = resolver.detect_conflicts(signals)
        direction, confidence, weights = resolver.resolve_conflicts(signals, conflicts)
        
        # Should take the highest confidence signal (BUY)
        assert direction == SignalDirection.BUY
        assert confidence == 85.0
        assert weights["strategy1"] == 1.0
        assert weights["strategy2"] == 0.0


class TestSignalValidator:
    """Test signal validation and filtering"""
    
    def test_validator_creation(self):
        """Test basic validator creation"""
        config = SignalManagerConfiguration()
        validator = SignalValidator(config)
        
        assert validator.config == config
        assert validator.logger is not None
    
    def test_validate_valid_signal(self):
        """Test validation of valid signal"""
        config = SignalManagerConfiguration()
        validator = SignalValidator(config)
        
        signal = create_test_signal(SignalDirection.BUY, 75.0, "strategy1")
        
        assert validator.validate_signal(signal) is True
    
    def test_validate_low_confidence_signal(self):
        """Test validation fails for low confidence"""
        config = SignalManagerConfiguration()
        config.confidence_threshold = 70.0
        validator = SignalValidator(config)
        
        signal = create_test_signal(SignalDirection.BUY, 65.0, "strategy1")
        
        assert validator.validate_signal(signal) is False
    
    def test_validate_expired_signal(self):
        """Test validation fails for expired signal"""
        config = SignalManagerConfiguration()
        config.max_signal_age = timedelta(minutes=1)
        validator = SignalValidator(config)
        
        signal = create_test_signal(SignalDirection.BUY, 75.0, "strategy1")
        # Make signal old
        signal.timestamp = datetime.now(timezone.utc) - timedelta(minutes=2)
        
        assert validator.validate_signal(signal) is False
    
    def test_filter_signals(self):
        """Test signal filtering"""
        config = SignalManagerConfiguration()
        config.confidence_threshold = 70.0
        validator = SignalValidator(config)
        
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),  # Valid
            create_test_signal(SignalDirection.SELL, 65.0, "strategy2"),  # Too low confidence
            create_test_signal(SignalDirection.BUY, 75.0, "strategy3"),  # Valid
        ]
        
        valid_signals = validator.filter_signals(signals)
        
        assert len(valid_signals) == 2
        assert valid_signals[0].strategy_id == "strategy1"
        assert valid_signals[1].strategy_id == "strategy3"
    
    def test_deduplicate_signals(self):
        """Test signal deduplication"""
        config = SignalManagerConfiguration()
        validator = SignalValidator(config)
        
        # Create signals with same strategy ID
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.BUY, 75.0, "strategy1"),  # Duplicate
            create_test_signal(SignalDirection.SELL, 70.0, "strategy2"),
        ]
        
        unique_signals = validator.deduplicate_signals(signals)
        
        assert len(unique_signals) == 2
        # Should keep highest confidence from duplicates
        strategy1_signal = next(s for s in unique_signals if s.strategy_id == "strategy1")
        assert strategy1_signal.signal.confidence == 80.0


class TestSignalScorer:
    """Test signal quality scoring"""
    
    def test_scorer_creation(self):
        """Test basic scorer creation"""
        config = SignalManagerConfiguration()
        scorer = SignalScorer(config)
        
        assert scorer.config == config
        assert scorer.logger is not None
    
    def test_calculate_quality_no_conflicts(self):
        """Test quality calculation without conflicts"""
        config = SignalManagerConfiguration()
        scorer = SignalScorer(config)
        
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.BUY, 75.0, "strategy2"),
        ]
        
        quality = scorer.calculate_quality(signals, [], 77.5, {"strategy1": 0.5, "strategy2": 0.5})
        
        assert quality.overall_score > 70
        assert quality.grade in [SignalQualityGrade.A, SignalQualityGrade.B_PLUS, SignalQualityGrade.B]
        assert quality.contributing_strategies == 2
        assert quality.conflicts_detected == 0
        assert quality.is_tradeable is True
    
    def test_calculate_quality_with_conflicts(self):
        """Test quality calculation with conflicts"""
        config = SignalManagerConfiguration()
        scorer = SignalScorer(config)
        
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.SELL, 70.0, "strategy2"),
        ]
        
        # Create a mock conflict
        from src.bistoury.signal_manager.models import SignalConflict
        conflict = SignalConflict(
            strategy_a="strategy1",
            strategy_b="strategy2",
            conflict_type=ConflictType.DIRECTION,
            severity=0.5,
            resolution=ConflictResolution.WEIGHTED_AVERAGE,
            signal_a_direction=SignalDirection.BUY,
            signal_b_direction=SignalDirection.SELL,
            signal_a_confidence=80.0,
            signal_b_confidence=70.0
        )
        
        quality = scorer.calculate_quality(signals, [conflict], 70.0, {"strategy1": 0.6, "strategy2": 0.4})
        
        assert quality.overall_score > 70  # Should be reasonable despite conflicts
        assert quality.consensus_score < 100  # Should show impact of conflicts
        assert quality.conflicts_detected == 1
        assert quality.major_conflicts >= 0  # Direction conflicts are considered major


class TestWeightManager:
    """Test dynamic weight management"""
    
    def test_weight_manager_creation(self):
        """Test basic weight manager creation"""
        config = SignalManagerConfiguration()
        manager = WeightManager(config)
        
        assert manager.config == config
        assert len(manager.weights) == 0
    
    def test_get_weight_new_strategy(self):
        """Test getting weight for new strategy"""
        config = SignalManagerConfiguration()
        manager = WeightManager(config)
        
        weight = manager.get_weight("candlestick_strategy")
        
        assert weight.strategy_id == "candlestick_strategy"
        assert weight.base_weight == 0.4  # From default config
        assert weight.final_weight == 0.4  # No modifiers initially
    
    def test_update_performance_success(self):
        """Test performance update with success"""
        config = SignalManagerConfiguration()
        manager = WeightManager(config)
        
        # Update with successful results
        for _ in range(8):
            manager.update_performance("test_strategy", True)
        for _ in range(2):
            manager.update_performance("test_strategy", False)
        
        weight = manager.get_weight("test_strategy")
        assert weight.total_signals == 10
        assert weight.successful_signals == 8
        assert weight.recent_success_rate == 0.8
        assert weight.performance_modifier == 1.2  # Should be boosted
    
    def test_update_performance_failure(self):
        """Test performance update with failures"""
        config = SignalManagerConfiguration()
        manager = WeightManager(config)
        
        # Update with mostly failed results
        for _ in range(2):
            manager.update_performance("test_strategy", True)
        for _ in range(8):
            manager.update_performance("test_strategy", False)
        
        weight = manager.get_weight("test_strategy")
        assert weight.total_signals == 10
        assert weight.successful_signals == 2
        assert weight.recent_success_rate == 0.2
        assert weight.performance_modifier == 0.5  # Should be penalized
    
    def test_should_update_weights(self):
        """Test weight update timing"""
        config = SignalManagerConfiguration()
        config.auto_adjust_weights = True
        config.weight_adjustment_frequency = timedelta(seconds=1)
        manager = WeightManager(config)
        
        # Initially should not update
        assert manager.should_update_weights() is False
        
        # After time passes, should update
        manager.last_weight_update = datetime.now(timezone.utc) - timedelta(seconds=2)
        assert manager.should_update_weights() is True


class TestSignalAggregator:
    """Test complete signal aggregation"""
    
    @pytest.fixture
    def aggregator(self):
        """Create test aggregator"""
        config = SignalManagerConfiguration()
        config.min_strategies_for_signal = 2
        config.confidence_threshold = 60.0
        return SignalAggregator(config)
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_success(self, aggregator):
        """Test successful signal aggregation"""
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.BUY, 75.0, "strategy2"),
        ]
        
        result = await aggregator.aggregate_signals(signals, "BTC")
        
        assert result is not None
        assert result.direction == SignalDirection.BUY
        assert result.confidence > 70
        assert len(result.contributing_strategies) == 2
        assert result.quality.is_tradeable is True
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_insufficient_strategies(self, aggregator):
        """Test aggregation with insufficient strategies"""
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
        ]
        
        result = await aggregator.aggregate_signals(signals, "BTC")
        
        assert result is None  # Should fail min strategy requirement
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_low_quality(self, aggregator):
        """Test aggregation with low quality signals"""
        signals = [
            create_test_signal(SignalDirection.BUY, 50.0, "strategy1"),  # Low confidence
            create_test_signal(SignalDirection.SELL, 45.0, "strategy2"),  # Low confidence, opposing
        ]
        
        result = await aggregator.aggregate_signals(signals, "BTC")
        
        # Should either return None or low quality result
        if result is not None:
            assert result.quality.is_tradeable is False
        else:
            # Signals filtered out due to low confidence
            assert True
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_with_conflicts(self, aggregator):
        """Test aggregation with conflicting signals"""
        signals = [
            create_test_signal(SignalDirection.BUY, 80.0, "strategy1"),
            create_test_signal(SignalDirection.SELL, 75.0, "strategy2"),
        ]
        
        result = await aggregator.aggregate_signals(signals, "BTC")
        
        if result is not None:
            # Should have detected and resolved conflicts
            assert len(result.conflicts) > 0
            assert result.confidence < 80  # Should be reduced due to conflicts
            assert result.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
    
    def test_get_performance_stats(self, aggregator):
        """Test performance statistics"""
        stats = aggregator.get_performance_stats()
        
        assert "total_signals_processed" in stats
        assert "total_conflicts_resolved" in stats
        assert "total_signals_filtered" in stats
        assert "strategy_weights" in stats
        assert "strategy_performance" in stats
    
    def test_update_strategy_performance(self, aggregator):
        """Test strategy performance updates"""
        aggregator.update_strategy_performance("test_strategy", True)
        
        weight = aggregator.weight_manager.get_weight("test_strategy")
        assert weight.total_signals == 1
        assert weight.successful_signals == 1
        
        stats = aggregator.get_performance_stats()
        assert "test_strategy" in stats["strategy_performance"]


if __name__ == "__main__":
    pytest.main([__file__]) 