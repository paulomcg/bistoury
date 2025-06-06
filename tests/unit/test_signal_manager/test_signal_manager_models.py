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
    
    def test_conflict_severity_calculation_direction(self):
        """Test automatic severity calculation for direction conflicts"""
        conflict = SignalConflict(
            strategy_a="strategy1",
            strategy_b="strategy2",
            conflict_type=ConflictType.DIRECTION,
            severity=0.0,  # Will be recalculated
            resolution=ConflictResolution.WEIGHTED_AVERAGE,
            signal_a_direction=SignalDirection.STRONG_BUY,
            signal_b_direction=SignalDirection.STRONG_SELL,
            signal_a_confidence=90.0,
            signal_b_confidence=80.0
        )
        
        # Should calculate severity based on opposing directions and confidence
        expected_severity = min(1.0, (90.0 + 80.0) / 2 / 100.0)
        assert abs(conflict.severity - expected_severity) < 0.01
        assert conflict.confidence_impact < 0  # Should be negative penalty
    
    def test_conflict_severity_calculation_confidence(self):
        """Test automatic severity calculation for confidence conflicts"""
        conflict = SignalConflict(
            strategy_a="strategy1",
            strategy_b="strategy2",
            conflict_type=ConflictType.CONFIDENCE,
            severity=0.0,  # Will be recalculated
            resolution=ConflictResolution.WEIGHTED_AVERAGE,
            signal_a_direction=SignalDirection.BUY,
            signal_b_direction=SignalDirection.BUY,
            signal_a_confidence=90.0,
            signal_b_confidence=40.0
        )
        
        # Should calculate severity based on confidence difference
        expected_severity = min(1.0, abs(90.0 - 40.0) / 100.0)
        assert abs(conflict.severity - expected_severity) < 0.01
        assert conflict.confidence_impact < 0


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
    
    def test_weight_calculation_with_modifiers(self):
        """Test final weight calculation with modifiers"""
        weight = SignalWeight(
            strategy_id="test",
            base_weight=0.5,
            performance_modifier=1.2,
            confidence_modifier=0.9,
            timeframe_modifier=1.1,
            volume_modifier=0.8
        )
        
        expected_final = 0.5 * 1.2 * 0.9 * 1.1 * 0.8
        assert abs(weight.final_weight - expected_final) < 0.001
    
    def test_performance_update_success(self):
        """Test performance tracking with successful signals"""
        weight = SignalWeight(strategy_id="test", base_weight=0.5)
        
        # Add successful signals
        for _ in range(8):
            weight.update_performance(True)
        for _ in range(2):
            weight.update_performance(False)
        
        assert weight.total_signals == 10
        assert weight.successful_signals == 8
        assert weight.recent_success_rate == 0.8
        assert weight.performance_modifier == 1.2  # Should boost high performers
        assert weight.final_weight == 0.5 * 1.2  # Boosted weight
    
    def test_performance_update_failure(self):
        """Test performance tracking with failed signals"""
        weight = SignalWeight(strategy_id="test", base_weight=0.5)
        
        # Add mostly failed signals
        for _ in range(2):
            weight.update_performance(True)
        for _ in range(8):
            weight.update_performance(False)
        
        assert weight.total_signals == 10
        assert weight.successful_signals == 2
        assert weight.recent_success_rate == 0.2
        assert weight.performance_modifier == 0.5  # Should penalize poor performers
        assert weight.final_weight == 0.5 * 0.5  # Reduced weight
    
    def test_weight_bounds_enforcement(self):
        """Test that final weight is kept within bounds"""
        weight = SignalWeight(
            strategy_id="test",
            base_weight=0.8,
            performance_modifier=2.0,  # Would push above 1.0
            confidence_modifier=1.5
        )
        
        assert weight.final_weight <= 1.0
        assert weight.final_weight >= 0.0


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
    
    def test_quality_summary_excellent(self):
        """Test quality summary for excellent signals"""
        quality = SignalQuality(
            overall_score=92.0,
            grade=SignalQualityGrade.A_PLUS,
            consensus_score=95.0,
            confidence_score=90.0,
            conflict_penalty=0.0,
            temporal_consistency=90.0,
            contributing_strategies=4,
            conflicts_detected=0,
            major_conflicts=0,
            is_tradeable=True
        )
        
        summary = quality.quality_summary
        assert "Excellent signal quality" in summary
        assert "A+" in summary
        assert "4 strategies" in summary
        assert "consensus" in summary
    
    def test_quality_summary_poor(self):
        """Test quality summary for poor signals"""
        quality = SignalQuality(
            overall_score=25.0,
            grade=SignalQualityGrade.D,
            consensus_score=30.0,
            confidence_score=40.0,
            conflict_penalty=20.0,
            temporal_consistency=50.0,
            contributing_strategies=2,
            conflicts_detected=3,
            major_conflicts=2,
            is_tradeable=False
        )
        
        summary = quality.quality_summary
        assert "Poor signal quality" in summary
        assert "D" in summary
        assert "not recommended" in summary
    
    def test_calculate_quality_excellent(self):
        """Test quality calculation for excellent signals"""
        conflicts = []  # No conflicts
        
        quality = SignalQuality.calculate_quality(
            consensus_score=95.0,
            confidence_score=90.0,
            conflicts=conflicts,
            contributing_strategies=4
        )
        
        assert quality.grade in [SignalQualityGrade.A_PLUS, SignalQualityGrade.A]
        assert quality.overall_score >= 80.0
        assert quality.is_tradeable is True
        assert quality.conflicts_detected == 0
        assert quality.major_conflicts == 0
    
    def test_calculate_quality_with_conflicts(self):
        """Test quality calculation with conflicts"""
        conflicts = [
            SignalConflict(
                strategy_a="strategy1",
                strategy_b="strategy2",
                conflict_type=ConflictType.DIRECTION,
                severity=0.8,
                resolution=ConflictResolution.WEIGHTED_AVERAGE,
                signal_a_direction=SignalDirection.BUY,
                signal_b_direction=SignalDirection.SELL,
                signal_a_confidence=70.0,
                signal_b_confidence=60.0
            )
        ]
        
        quality = SignalQuality.calculate_quality(
            consensus_score=70.0,
            confidence_score=65.0,
            conflicts=conflicts,
            contributing_strategies=2
        )
        
        assert quality.conflicts_detected == 1
        assert quality.conflict_penalty > 0
        assert quality.overall_score < 70.0  # Should be reduced by conflicts
        assert quality.grade in [SignalQualityGrade.B, SignalQualityGrade.C]
    
    def test_calculate_quality_not_tradeable(self):
        """Test quality calculation for non-tradeable signals"""
        conflicts = [
            SignalConflict(
                strategy_a="strategy1",
                strategy_b="strategy2",
                conflict_type=ConflictType.DIRECTION,
                severity=0.9,  # Major conflict
                resolution=ConflictResolution.WEIGHTED_AVERAGE,
                signal_a_direction=SignalDirection.BUY,
                signal_b_direction=SignalDirection.SELL,
                signal_a_confidence=50.0,
                signal_b_confidence=45.0
            )
        ]
        
        quality = SignalQuality.calculate_quality(
            consensus_score=40.0,
            confidence_score=45.0,
            conflicts=conflicts,
            contributing_strategies=2
        )
        
        assert quality.is_tradeable is False
        assert quality.major_conflicts > 0
        assert quality.overall_score < 70.0


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
    
    def test_aggregated_signal_consensus_level(self):
        """Test consensus level computation"""
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
        
        # No conflicts - full consensus
        signal = AggregatedSignal(
            direction=SignalDirection.BUY,
            confidence=85.0,
            weight=0.8,
            contributing_strategies=["strategy1", "strategy2"],
            strategy_weights={"strategy1": 0.5, "strategy2": 0.5},
            quality=quality,
            risk_level=RiskLevel.MEDIUM,
            expiry=datetime.now(timezone.utc) + timedelta(minutes=5),
            conflicts=[]
        )
        
        assert signal.consensus_level == "full_consensus"
        
        # Minor conflicts - strong consensus
        minor_conflict = SignalConflict(
            strategy_a="strategy1",
            strategy_b="strategy2",
            conflict_type=ConflictType.CONFIDENCE,
            severity=0.3,
            resolution=ConflictResolution.WEIGHTED_AVERAGE,
            signal_a_direction=SignalDirection.BUY,
            signal_b_direction=SignalDirection.BUY,
            signal_a_confidence=80.0,
            signal_b_confidence=60.0
        )
        
        signal.conflicts = [minor_conflict]
        assert signal.consensus_level == "strong_consensus"
    
    def test_aggregated_signal_to_trading_signal(self):
        """Test conversion to basic TradingSignal"""
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
        
        aggregated = AggregatedSignal(
            direction=SignalDirection.STRONG_BUY,
            confidence=88.0,
            weight=0.9,
            contributing_strategies=["candlestick", "funding_rate"],
            strategy_weights={"candlestick": 0.6, "funding_rate": 0.4},
            quality=quality,
            risk_level=RiskLevel.HIGH,
            expiry=datetime.now(timezone.utc) + timedelta(minutes=5)
        )
        
        trading_signal = aggregated.to_trading_signal()
        
        assert isinstance(trading_signal, TradingSignal)
        assert trading_signal.direction == SignalDirection.STRONG_BUY
        assert trading_signal.confidence == 88.0
        assert trading_signal.timeframe == Timeframe.MULTI
        assert trading_signal.risk_level == RiskLevel.HIGH
        assert "Aggregated from 2 strategies" in trading_signal.reasoning
    
    def test_aggregated_signal_expiry(self):
        """Test signal expiry functionality"""
        quality = SignalQuality(
            overall_score=85.0,
            grade=SignalQualityGrade.A,
            consensus_score=90.0,
            confidence_score=80.0,
            conflict_penalty=0.0,
            temporal_consistency=85.0,
            contributing_strategies=2,
            conflicts_detected=0,
            major_conflicts=0,
            is_tradeable=True
        )
        
        # Signal that expires in the past
        past_expiry = datetime.now(timezone.utc) - timedelta(minutes=1)
        
        expired_signal = AggregatedSignal(
            direction=SignalDirection.BUY,
            confidence=80.0,
            weight=0.7,
            contributing_strategies=["strategy1"],
            strategy_weights={"strategy1": 1.0},
            quality=quality,
            risk_level=RiskLevel.MEDIUM,
            expiry=past_expiry
        )
        
        assert expired_signal.is_expired is True
        assert expired_signal.time_to_expiry.total_seconds() < 0


class TestTemporalSignalBuffer:
    """Test temporal signal and narrative buffer"""
    
    def test_buffer_creation(self):
        """Test basic buffer creation"""
        buffer = TemporalSignalBuffer(max_age=timedelta(minutes=10))
        
        assert buffer.max_age == timedelta(minutes=10)
        assert len(buffer.signals) == 0
        assert len(buffer.narratives) == 0
        assert len(buffer.signal_narrative_map) == 0
    
    def test_add_signal_and_narrative(self):
        """Test adding signals and narratives to buffer"""
        buffer = TemporalSignalBuffer()
        
        # Create test signal
        quality = SignalQuality(
            overall_score=85.0,
            grade=SignalQualityGrade.A,
            consensus_score=90.0,
            confidence_score=80.0,
            conflict_penalty=0.0,
            temporal_consistency=85.0,
            contributing_strategies=2,
            conflicts_detected=0,
            major_conflicts=0,
            is_tradeable=True
        )
        
        signal = AggregatedSignal(
            direction=SignalDirection.BUY,
            confidence=80.0,
            weight=0.7,
            contributing_strategies=["candlestick"],
            strategy_weights={"candlestick": 1.0},
            quality=quality,
            risk_level=RiskLevel.MEDIUM,
            expiry=datetime.now(timezone.utc) + timedelta(minutes=5)
        )
        
        # Create test narrative
        narrative = TradingNarrative(
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=80.0,
            timeframe="15m",
            executive_summary="Test bullish signal",
            market_overview="Market showing strength",
            pattern_analysis="Strong bullish pattern detected",
            timeframe_analysis="Higher timeframes align",
            volume_analysis="Volume confirms pattern",
            risk_assessment="Moderate risk entry",
            entry_exit_strategy="Entry at $50000, exit at $52000",
            supporting_factors=["Strong momentum", "Volume confirmation"],
            conflicting_factors=["Some resistance above"],
            key_warnings=["Watch for reversal signs"],
            generation_timestamp=datetime.now(timezone.utc),
            style=NarrativeStyle.COMPREHENSIVE,
            config=NarrativeConfiguration()
        )
        
        buffer.add_signal(signal, narrative)
        
        assert len(buffer.signals) == 1
        assert len(buffer.narratives) == 1
        assert len(buffer.signal_narrative_map) == 1
        assert len(signal.source_narratives) == 1
    
    def test_get_recent_signals(self):
        """Test retrieving recent signals"""
        buffer = TemporalSignalBuffer()
        
        # Add multiple signals
        for i in range(5):
            quality = SignalQuality(
                overall_score=80.0 + i,
                grade=SignalQualityGrade.A,
                consensus_score=85.0,
                confidence_score=75.0 + i,
                conflict_penalty=0.0,
                temporal_consistency=85.0,
                contributing_strategies=2,
                conflicts_detected=0,
                major_conflicts=0,
                is_tradeable=True
            )
            
            signal = AggregatedSignal(
                direction=SignalDirection.BUY,
                confidence=75.0 + i,
                weight=0.7,
                contributing_strategies=[f"strategy{i}"],
                strategy_weights={f"strategy{i}": 1.0},
                quality=quality,
                risk_level=RiskLevel.MEDIUM,
                expiry=datetime.now(timezone.utc) + timedelta(minutes=5)
            )
            
            narrative = TradingNarrative(
                symbol="BTC",
                direction=SignalDirection.BUY,
                confidence=75.0 + i,
                timeframe="15m",
                executive_summary=f"Test signal {i}",
                market_overview="Market analysis",
                pattern_analysis="Pattern detected",
                timeframe_analysis="Timeframes align",
                volume_analysis="Volume confirms",
                risk_assessment="Risk assessment",
                entry_exit_strategy="Entry/exit strategy",
                supporting_factors=["Factor 1"],
                conflicting_factors=["Factor 2"],
                key_warnings=["Warning 1"],
                generation_timestamp=datetime.now(timezone.utc),
                style=NarrativeStyle.COMPREHENSIVE,
                config=NarrativeConfiguration()
            )
            
            buffer.add_signal(signal, narrative)
        
        # Test retrieving all signals
        all_signals = buffer.get_recent_signals()
        assert len(all_signals) == 5
        
        # Test retrieving limited signals
        recent_signals = buffer.get_recent_signals(max_count=3)
        assert len(recent_signals) == 3
        assert recent_signals[-1].confidence == 79.0  # Most recent
    
    def test_get_narrative_timeline(self):
        """Test retrieving narrative timeline"""
        buffer = TemporalSignalBuffer()
        
        # Add signals with narratives
        for i in range(3):
            quality = SignalQuality(
                overall_score=80.0,
                grade=SignalQualityGrade.A,
                consensus_score=85.0,
                confidence_score=75.0,
                conflict_penalty=0.0,
                temporal_consistency=85.0,
                contributing_strategies=2,
                conflicts_detected=0,
                major_conflicts=0,
                is_tradeable=True
            )
            
            signal = AggregatedSignal(
                direction=SignalDirection.BUY,
                confidence=75.0,
                weight=0.7,
                contributing_strategies=["strategy"],
                strategy_weights={"strategy": 1.0},
                quality=quality,
                risk_level=RiskLevel.MEDIUM,
                expiry=datetime.now(timezone.utc) + timedelta(minutes=5)
            )
            
            narrative = TradingNarrative(
                symbol="BTC",
                direction=SignalDirection.BUY,
                confidence=75.0,
                timeframe="15m",
                executive_summary=f"Timeline narrative {i}",
                market_overview="Market analysis",
                pattern_analysis="Pattern analysis",
                timeframe_analysis="Timeframe analysis",
                volume_analysis="Volume analysis",
                risk_assessment="Risk assessment",
                entry_exit_strategy="Entry/exit strategy",
                supporting_factors=["Supporting factor"],
                conflicting_factors=["Conflicting factor"],
                key_warnings=["Key warning"],
                generation_timestamp=datetime.now(timezone.utc),
                style=NarrativeStyle.COMPREHENSIVE,
                config=NarrativeConfiguration()
            )
            
            buffer.add_signal(signal, narrative)
        
        timeline = buffer.get_narrative_timeline()
        assert len(timeline) == 3
        assert all(isinstance(n, TradingNarrative) for n in timeline)
        assert timeline[0].executive_summary == "Timeline narrative 0"
        assert timeline[2].executive_summary == "Timeline narrative 2"
    
    def test_buffer_statistics(self):
        """Test buffer statistics reporting"""
        buffer = TemporalSignalBuffer()
        
        # Initially empty
        stats = buffer.get_buffer_stats()
        assert stats["signal_count"] == 0
        assert stats["narrative_count"] == 0
        assert stats["oldest_signal"] is None
        assert stats["newest_signal"] is None
        assert stats["buffer_age_minutes"] == 15.0
        assert stats["memory_usage_estimate"] == 0
        
        # Add one signal
        quality = SignalQuality(
            overall_score=80.0,
            grade=SignalQualityGrade.A,
            consensus_score=85.0,
            confidence_score=75.0,
            conflict_penalty=0.0,
            temporal_consistency=85.0,
            contributing_strategies=2,
            conflicts_detected=0,
            major_conflicts=0,
            is_tradeable=True
        )
        
        signal = AggregatedSignal(
            direction=SignalDirection.BUY,
            confidence=75.0,
            weight=0.7,
            contributing_strategies=["strategy"],
            strategy_weights={"strategy": 1.0},
            quality=quality,
            risk_level=RiskLevel.MEDIUM,
            expiry=datetime.now(timezone.utc) + timedelta(minutes=5)
        )
        
        narrative = TradingNarrative(
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=75.0,
            timeframe="15m",
            executive_summary="Test narrative",
            market_overview="Market analysis",
            pattern_analysis="Pattern analysis",
            timeframe_analysis="Timeframe analysis",
            volume_analysis="Volume analysis",
            risk_assessment="Risk assessment",
            entry_exit_strategy="Entry/exit strategy",
            supporting_factors=["Supporting factor"],
            conflicting_factors=["Conflicting factor"],
            key_warnings=["Key warning"],
            generation_timestamp=datetime.now(timezone.utc),
            style=NarrativeStyle.COMPREHENSIVE,
            config=NarrativeConfiguration()
        )
        
        buffer.add_signal(signal, narrative)
        
        stats = buffer.get_buffer_stats()
        assert stats["signal_count"] == 1
        assert stats["narrative_count"] == 1
        assert stats["oldest_signal"] is not None
        assert stats["newest_signal"] is not None
        assert stats["memory_usage_estimate"] > 0


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
    
    def test_configuration_custom_values(self):
        """Test configuration with custom values"""
        custom_weights = {
            "custom_strategy": 0.6,
            "another_strategy": 0.4
        }
        
        config = SignalManagerConfiguration(
            strategy_weights=custom_weights,
            min_strategies_for_signal=3,
            confidence_threshold=70.0,
            quality_threshold=80.0,
            max_conflict_severity=0.6,
            preserve_narratives=False,
            auto_adjust_weights=False
        )
        
        assert config.strategy_weights == custom_weights
        assert config.min_strategies_for_signal == 3
        assert config.confidence_threshold == 70.0
        assert config.quality_threshold == 80.0
        assert config.max_conflict_severity == 0.6
        assert config.preserve_narratives is False
        assert config.auto_adjust_weights is False
    
    def test_update_strategy_weight(self):
        """Test updating strategy weights"""
        config = SignalManagerConfiguration()
        
        # Valid weight update
        config.update_strategy_weight("candlestick_strategy", 0.5)
        assert config.strategy_weights["candlestick_strategy"] == 0.5
        
        # Add new strategy
        config.update_strategy_weight("new_strategy", 0.2)
        assert config.strategy_weights["new_strategy"] == 0.2
        
        # Invalid weight should raise error
        with pytest.raises(ValueError):
            config.update_strategy_weight("test", -0.1)
        
        with pytest.raises(ValueError):
            config.update_strategy_weight("test", 1.5)
    
    def test_get_strategy_weight(self):
        """Test getting strategy weights"""
        config = SignalManagerConfiguration()
        
        # Existing strategy
        weight = config.get_strategy_weight("candlestick_strategy")
        assert weight == 0.4
        
        # Non-existing strategy should return default
        weight = config.get_strategy_weight("unknown_strategy")
        assert weight == 0.1  # Default weight


if __name__ == "__main__":
    pytest.main([__file__]) 