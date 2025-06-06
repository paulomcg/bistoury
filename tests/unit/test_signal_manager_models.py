"""
Test suite for Signal Manager Models - Task 9.1

Tests core signal aggregation, weighting, and temporal storage models.
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.bistoury.signal_manager.models import (
    AggregatedSignal,
    SignalWeight,
    SignalConflict,
    ConflictType,
    ConflictResolution,
    TemporalSignalBuffer,
    SignalQuality,
    SignalQualityGrade,
    SignalManagerConfiguration,
)
from src.bistoury.models.signals import (
    TradingSignal,
    SignalDirection,
    Timeframe,
    RiskLevel
)
from src.bistoury.strategies.narrative_generator import (
    TradingNarrative,
    NarrativeConfiguration,
    NarrativeStyle
)


class TestSignalConflict:
    """Test signal conflict detection and resolution"""
    
    def test_signal_conflict_creation(self):
        """Test basic signal conflict creation"""
        conflict = SignalConflict(
            strategy_a="candlestick",
            strategy_b="funding_rate", 
            conflict_type=ConflictType.DIRECTION,
            severity=0.8,
            resolution=ConflictResolution.WEIGHTED_AVERAGE,
            signal_a_direction=SignalDirection.BUY,
            signal_b_direction=SignalDirection.SELL,
            signal_a_confidence=85.0,
            signal_b_confidence=75.0
        )
        
        assert conflict.strategy_a == "candlestick"
        assert conflict.strategy_b == "funding_rate"
        assert conflict.conflict_type == ConflictType.DIRECTION
        assert conflict.severity == 0.8
        assert conflict.resolution == ConflictResolution.WEIGHTED_AVERAGE
        assert conflict.signal_a_direction == SignalDirection.BUY
        assert conflict.signal_b_direction == SignalDirection.SELL
        assert conflict.signal_a_confidence == 85.0
        assert conflict.signal_b_confidence == 75.0


class TestSignalWeight:
    """Test dynamic signal weighting system"""
    
    def test_signal_weight_creation(self):
        """Test basic signal weight creation and calculation"""
        weight = SignalWeight(
            strategy_id="candlestick",
            base_weight=0.4
        )
        
        assert weight.strategy_id == "candlestick"
        assert weight.base_weight == 0.4
        assert weight.performance_modifier == 1.0
        assert weight.confidence_modifier == 1.0
        assert weight.timeframe_modifier == 1.0
        assert weight.volume_modifier == 1.0
        assert weight.final_weight == 0.4  # base_weight * all modifiers
        assert weight.recent_success_rate == 0.5
        assert weight.total_signals == 0
        assert weight.successful_signals == 0


class TestSignalQuality:
    """Test signal quality assessment and grading"""
    
    def test_signal_quality_creation(self):
        """Test basic signal quality creation"""
        quality = SignalQuality(
            overall_score=85.0,
            grade=SignalQualityGrade.A,
            consensus_score=90.0,
            confidence_score=80.0,
            conflict_penalty=5.0,
            temporal_consistency=85.0,
            contributing_strategies=3,
            conflicts_detected=1,
            major_conflicts=0,
            is_tradeable=True
        )
        
        assert quality.overall_score == 85.0
        assert quality.grade == SignalQualityGrade.A
        assert quality.consensus_score == 90.0
        assert quality.confidence_score == 80.0
        assert quality.is_tradeable is True


class TestAggregatedSignal:
    """Test aggregated signal model"""
    
    def test_aggregated_signal_creation(self):
        """Test basic aggregated signal creation"""
        quality = SignalQuality(
            overall_score=85.0,
            grade=SignalQualityGrade.A,
            consensus_score=90.0,
            confidence_score=80.0,
            conflict_penalty=0.0,
            temporal_consistency=85.0,
            contributing_strategies=3,
            conflicts_detected=0,
            major_conflicts=0,
            is_tradeable=True
        )
        
        expiry_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        
        signal = AggregatedSignal(
            direction=SignalDirection.BUY,
            confidence=82.5,
            weight=0.75,
            contributing_strategies=["candlestick", "funding_rate", "volume"],
            strategy_weights={"candlestick": 0.4, "funding_rate": 0.3, "volume": 0.3},
            quality=quality,
            risk_level=RiskLevel.MEDIUM,
            expiry=expiry_time
        )
        
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == 82.5
        assert signal.weight == 0.75
        assert len(signal.contributing_strategies) == 3
        assert signal.quality.grade == SignalQualityGrade.A
        assert signal.risk_level == RiskLevel.MEDIUM
        assert not signal.is_expired
        assert signal.time_to_expiry.total_seconds() > 0


class TestTemporalSignalBuffer:
    """Test temporal signal and narrative buffer"""
    
    def test_buffer_creation(self):
        """Test basic buffer creation"""
        buffer = TemporalSignalBuffer(max_age=timedelta(minutes=10))
        
        assert buffer.max_age == timedelta(minutes=10)
        assert len(buffer.signals) == 0
        assert len(buffer.narratives) == 0
        assert len(buffer.signal_narrative_map) == 0


class TestSignalManagerConfiguration:
    """Test signal manager configuration"""
    
    def test_configuration_creation(self):
        """Test basic configuration creation with defaults"""
        config = SignalManagerConfiguration()
        
        assert len(config.strategy_weights) == 4
        assert config.strategy_weights["candlestick_strategy"] == 0.4
        assert config.strategy_weights["funding_rate_strategy"] == 0.3
        assert config.min_strategies_for_signal == 2
        assert config.max_signal_age == timedelta(minutes=5)
        assert config.confidence_threshold == 60.0
        assert config.quality_threshold == 70.0
        assert config.default_conflict_resolution == ConflictResolution.WEIGHTED_AVERAGE
        assert config.max_conflict_severity == 0.8
        assert config.temporal_buffer_age == timedelta(minutes=15)
        assert config.preserve_narratives is True
        assert config.auto_adjust_weights is True


if __name__ == "__main__":
    pytest.main([__file__]) 