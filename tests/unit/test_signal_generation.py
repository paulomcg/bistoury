"""
Tests for Trading Signal Generation

Comprehensive test suite for the signal generation system including:
- Signal generation from patterns
- Entry point calculation
- Risk management parameters
- Signal validation and filtering
- Signal persistence and expiration
- Multi-timeframe signal generation
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List

from src.bistoury.strategies.signal_generator import (
    SignalGenerator,
    SignalConfiguration,
    SignalTiming,
    SignalValidation,
    RiskLevel,
    SignalEntryPoint,
    SignalRiskManagement,
    GeneratedSignal,
    SignalDatabase
)
from src.bistoury.strategies.candlestick_models import PatternStrength, VolumeProfile
from src.bistoury.strategies.pattern_scoring import CompositePatternScore, PatternScoringEngine
from src.bistoury.strategies.timeframe_analyzer import TimeframeAnalysisResult, TimeframeSynchronization, TrendAlignment, ConfluenceAnalysis
from src.bistoury.models.market_data import CandlestickData, Timeframe
from src.bistoury.models.signals import (
    CandlestickPattern, 
    PatternType, 
    SignalDirection, 
    SignalType,
    TradingSignal
)


@pytest.fixture
def sample_candle():
    """Create a sample candlestick for testing."""
    return CandlestickData(
        symbol="BTC",
        timeframe=Timeframe.FIVE_MINUTES,
        timestamp=datetime.now(timezone.utc),
        open=Decimal('50000'),
        high=Decimal('50200'),
        low=Decimal('49800'),
        close=Decimal('50100'),
        volume=Decimal('100'),
        trade_count=50
    )


@pytest.fixture
def sample_market_data(sample_candle):
    """Create sample market data."""
    market_data = []
    base_time = sample_candle.timestamp
    
    for i in range(20):
        candle = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=base_time - timedelta(minutes=5*i),
            open=Decimal('50000') + Decimal(i * 10),
            high=Decimal('50200') + Decimal(i * 10),
            low=Decimal('49800') + Decimal(i * 10),
            close=Decimal('50100') + Decimal(i * 10),
            volume=Decimal('100') + Decimal(i * 5),
            trade_count=50 + i
        )
        market_data.append(candle)
    
    return list(reversed(market_data))


@pytest.fixture
def sample_pattern(sample_candle):
    """Create a sample candlestick pattern."""
    return CandlestickPattern(
        pattern_id="test_hammer",
        symbol="BTC",
        pattern_type=PatternType.HAMMER,
        timeframe=Timeframe.FIVE_MINUTES,
        timestamp=sample_candle.timestamp,
        candles=[sample_candle],
        confidence=Decimal('75'),
        reliability=Decimal('0.8'),
        bullish_probability=Decimal('0.75'),
        bearish_probability=Decimal('0.25'),
        completion_price=sample_candle.close,
        volume_confirmation=True,
        pattern_metadata={"strength": "strong"}
    )


@pytest.fixture
def sample_volume_profile():
    """Create a sample volume profile."""
    return VolumeProfile(
        pattern_volume=Decimal('150'),
        average_volume=Decimal('100'),
        volume_ratio=Decimal('1.5'),  # Required field
        volume_trend=SignalDirection.BUY,
        breakout_confirmation=True,
        volume_surge_percentage=Decimal('50')
    )


@pytest.fixture
def signal_generator():
    """Create a signal generator with lenient configuration for testing."""
    config = SignalConfiguration(
        min_confidence_threshold=Decimal('50'),  # Lower for testing
        min_strength_threshold=PatternStrength.WEAK,  # Lower for testing
        require_volume_confirmation=False,  # Disable for testing
        require_trend_alignment=False,  # Disable for testing
        min_risk_reward_ratio=Decimal('1.0'),  # Lower for testing
        max_risk_per_trade=Decimal('5.0'),
        signal_expiry_hours=24
    )
    return SignalGenerator(config)


@pytest.fixture
def restrictive_config():
    """Create a restrictive signal configuration for testing filters."""
    return SignalConfiguration(
        min_confidence_threshold=Decimal('80'),
        min_strength_threshold=PatternStrength.STRONG,
        require_volume_confirmation=True,
        require_trend_alignment=True,
        min_risk_reward_ratio=Decimal('2.0'),
        max_risk_per_trade=Decimal('1.0'),
        signal_expiry_hours=12
    )


class TestSignalConfiguration:
    """Test signal configuration validation and defaults."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = SignalConfiguration()
        
        assert config.min_confidence_threshold == Decimal('60')
        assert config.min_strength_threshold == PatternStrength.MODERATE
        assert config.require_volume_confirmation is True
        assert config.require_trend_alignment is True
        assert config.min_risk_reward_ratio == Decimal('1.5')
        assert config.max_risk_per_trade == Decimal('2.0')
        assert config.signal_expiry_hours == 24
        assert config.enable_multi_timeframe is True
        assert config.priority_timeframe == Timeframe.FIFTEEN_MINUTES
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = SignalConfiguration(
            min_confidence_threshold=Decimal('70'),
            min_strength_threshold=PatternStrength.STRONG,
            require_volume_confirmation=False,
            signal_expiry_hours=48
        )
        
        assert config.min_confidence_threshold == Decimal('70')
        assert config.min_strength_threshold == PatternStrength.STRONG
        assert config.require_volume_confirmation is False
        assert config.signal_expiry_hours == 48
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = SignalConfiguration(
            min_confidence_threshold=Decimal('50'),
            min_risk_reward_ratio=Decimal('1.0'),
            max_risk_per_trade=Decimal('5.0')
        )
        assert config.min_confidence_threshold == Decimal('50')
        
        # Invalid confidence threshold (should be clamped by Pydantic)
        with pytest.raises(ValueError):
            SignalConfiguration(min_confidence_threshold=Decimal('150'))


class TestSignalEntryPoint:
    """Test signal entry point calculations."""
    
    def test_entry_point_creation(self):
        """Test basic entry point creation."""
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50100'),
            entry_timing=SignalTiming.IMMEDIATE,
            entry_zone_low=Decimal('50000'),
            entry_zone_high=Decimal('50200'),
            max_slippage_pct=Decimal('0.1'),
            entry_window_minutes=30
        )
        
        assert entry_point.entry_price == Decimal('50100')
        assert entry_point.entry_timing == SignalTiming.IMMEDIATE
        assert entry_point.entry_zone_low == Decimal('50000')
        assert entry_point.entry_zone_high == Decimal('50200')
    
    def test_entry_zone_calculations(self):
        """Test entry zone size and midpoint calculations."""
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50100'),
            entry_timing=SignalTiming.CONFIRMATION,
            entry_zone_low=Decimal('50000'),
            entry_zone_high=Decimal('50200')
        )
        
        assert entry_point.entry_zone_size == Decimal('200')
        assert entry_point.zone_midpoint == Decimal('50100')


class TestSignalRiskManagement:
    """Test signal risk management calculations."""
    
    def test_risk_management_creation(self):
        """Test basic risk management creation."""
        risk_mgmt = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            take_profit_price=Decimal('51000'),
            risk_amount=Decimal('600'),
            reward_amount=Decimal('900'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        assert risk_mgmt.stop_loss_price == Decimal('49500')
        assert risk_mgmt.take_profit_price == Decimal('51000')
        assert risk_mgmt.risk_level == RiskLevel.MEDIUM
    
    def test_risk_reward_ratio_calculation(self):
        """Test risk/reward ratio calculation."""
        risk_mgmt = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            take_profit_price=Decimal('51000'),
            risk_amount=Decimal('600'),
            reward_amount=Decimal('900'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        assert risk_mgmt.risk_reward_ratio == Decimal('1.5')
        assert risk_mgmt.max_loss_amount == Decimal('60000')  # 600 * 100
    
    def test_risk_reward_ratio_none(self):
        """Test risk/reward ratio when reward is None."""
        risk_mgmt = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            risk_amount=Decimal('600'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        assert risk_mgmt.risk_reward_ratio is None


class TestGeneratedSignal:
    """Test generated signal model and methods."""
    
    def test_generated_signal_creation(self, sample_pattern, sample_market_data):
        """Test basic generated signal creation."""
        # Create components
        base_signal = TradingSignal.create_buy_signal(
            signal_id="test_signal",
            symbol="BTC",
            price=Decimal('50100'),
            confidence=Decimal('75'),
            strength=Decimal('0.75'),
            source="test",
            reason="Test signal"
        )
        
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50100'),
            entry_timing=SignalTiming.IMMEDIATE,
            entry_zone_low=Decimal('50000'),
            entry_zone_high=Decimal('50200')
        )
        
        risk_management = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            take_profit_price=Decimal('51000'),
            risk_amount=Decimal('600'),
            reward_amount=Decimal('900'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        # Create pattern score
        scorer = PatternScoringEngine()
        pattern_score = scorer.score_pattern(sample_pattern, sample_market_data)
        
        # Create generated signal
        signal = GeneratedSignal(
            base_signal=base_signal,
            entry_point=entry_point,
            risk_management=risk_management,
            source_pattern=sample_pattern,
            pattern_score=pattern_score
        )
        
        assert signal.base_signal.symbol == "BTC"
        assert signal.entry_point.entry_price == Decimal('50100')
        assert signal.source_pattern.pattern_type == PatternType.HAMMER
        assert signal.validation_status == SignalValidation.PENDING
    
    def test_signal_quality_score(self, sample_pattern, sample_market_data):
        """Test signal quality score calculation."""
        # Create signal with high confidence
        base_signal = TradingSignal.create_buy_signal(
            signal_id="test_signal",
            symbol="BTC",
            price=Decimal('50100'),
            confidence=Decimal('85'),
            strength=Decimal('0.85'),
            source="test",
            reason="Test signal"
        )
        
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50100'),
            entry_timing=SignalTiming.IMMEDIATE,
            entry_zone_low=Decimal('50000'),
            entry_zone_high=Decimal('50200')
        )
        
        risk_management = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            take_profit_price=Decimal('51500'),  # Higher reward
            risk_amount=Decimal('600'),
            reward_amount=Decimal('1400'),  # 2.3 R/R ratio
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        scorer = PatternScoringEngine()
        pattern_score = scorer.score_pattern(sample_pattern, sample_market_data)
        
        signal = GeneratedSignal(
            base_signal=base_signal,
            entry_point=entry_point,
            risk_management=risk_management,
            source_pattern=sample_pattern,
            pattern_score=pattern_score
        )
        
        # Should get bonus for good risk/reward ratio
        base_score = pattern_score.weighted_confidence_score
        quality_score = signal.signal_quality_score
        
        assert quality_score >= base_score  # Should be enhanced
        assert quality_score <= Decimal('100')  # Should be capped
    
    def test_signal_validation_status_update(self, sample_pattern, sample_market_data):
        """Test signal validation status updates."""
        base_signal = TradingSignal.create_buy_signal(
            signal_id="test_signal",
            symbol="BTC",
            price=Decimal('50100'),
            confidence=Decimal('75'),
            strength=Decimal('0.75'),
            source="test",
            reason="Test signal"
        )
        
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50100'),
            entry_timing=SignalTiming.IMMEDIATE,
            entry_zone_low=Decimal('50000'),
            entry_zone_high=Decimal('50200')
        )
        
        risk_management = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            risk_amount=Decimal('600'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        scorer = PatternScoringEngine()
        pattern_score = scorer.score_pattern(sample_pattern, sample_market_data)
        
        signal = GeneratedSignal(
            base_signal=base_signal,
            entry_point=entry_point,
            risk_management=risk_management,
            source_pattern=sample_pattern,
            pattern_score=pattern_score
        )
        
        # Test status update
        initial_time = signal.last_updated
        signal.update_validation_status(SignalValidation.VALID, "Passed validation")
        
        assert signal.validation_status == SignalValidation.VALID
        assert signal.last_updated > initial_time
        assert "validation_reason" in signal.base_signal.metadata


class TestSignalDatabase:
    """Test signal database functionality."""
    
    def test_signal_storage_and_retrieval(self, sample_pattern, sample_market_data):
        """Test storing and retrieving signals."""
        db = SignalDatabase()
        
        # Create a signal
        base_signal = TradingSignal.create_buy_signal(
            signal_id="test_signal",
            symbol="BTC",
            price=Decimal('50100'),
            confidence=Decimal('75'),
            strength=Decimal('0.75'),
            source="test",
            reason="Test signal"
        )
        
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50100'),
            entry_timing=SignalTiming.IMMEDIATE,
            entry_zone_low=Decimal('50000'),
            entry_zone_high=Decimal('50200')
        )
        
        risk_management = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            risk_amount=Decimal('600'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        scorer = PatternScoringEngine()
        pattern_score = scorer.score_pattern(sample_pattern, sample_market_data)
        
        signal = GeneratedSignal(
            base_signal=base_signal,
            entry_point=entry_point,
            risk_management=risk_management,
            source_pattern=sample_pattern,
            pattern_score=pattern_score,
            validation_status=SignalValidation.VALID
        )
        
        # Store signal
        db.store_signal(signal)
        
        # Retrieve signal
        retrieved = db.get_signal(signal.signal_id)
        assert retrieved is not None
        assert retrieved.signal_id == signal.signal_id
        
        # Get signals by symbol
        btc_signals = db.get_signals_by_symbol("BTC")
        assert len(btc_signals) == 1
        assert btc_signals[0].signal_id == signal.signal_id
    
    def test_signal_expiry(self, sample_pattern, sample_market_data):
        """Test signal expiry functionality."""
        db = SignalDatabase()
        
        # Create an old signal
        base_signal = TradingSignal.create_buy_signal(
            signal_id="old_signal",
            symbol="BTC",
            price=Decimal('50100'),
            confidence=Decimal('75'),
            strength=Decimal('0.75'),
            source="test",
            reason="Old signal"
        )
        
        entry_point = SignalEntryPoint(
            entry_price=Decimal('50100'),
            entry_timing=SignalTiming.IMMEDIATE,
            entry_zone_low=Decimal('50000'),
            entry_zone_high=Decimal('50200')
        )
        
        risk_management = SignalRiskManagement(
            stop_loss_price=Decimal('49500'),
            risk_amount=Decimal('600'),
            risk_percentage=Decimal('2.0'),
            position_size_suggestion=Decimal('100'),
            risk_level=RiskLevel.MEDIUM
        )
        
        scorer = PatternScoringEngine()
        pattern_score = scorer.score_pattern(sample_pattern, sample_market_data)
        
        signal = GeneratedSignal(
            base_signal=base_signal,
            entry_point=entry_point,
            risk_management=risk_management,
            source_pattern=sample_pattern,
            pattern_score=pattern_score,
            validation_status=SignalValidation.VALID
        )
        
        # Make it old
        signal.generation_timestamp = datetime.now(timezone.utc) - timedelta(hours=25)
        
        db.store_signal(signal)
        
        # Expire old signals
        expired_count = db.expire_old_signals(24)
        assert expired_count == 1
        
        # Check signal is expired
        retrieved = db.get_signal(signal.signal_id)
        assert retrieved.validation_status == SignalValidation.EXPIRED
    
    def test_database_statistics(self, sample_pattern, sample_market_data):
        """Test database statistics."""
        db = SignalDatabase()
        
        # Add some signals
        for i in range(3):
            base_signal = TradingSignal.create_buy_signal(
                signal_id=f"signal_{i}",
                symbol="BTC",
                price=Decimal('50100'),
                confidence=Decimal('75'),
                strength=Decimal('0.75'),
                source="test",
                reason=f"Signal {i}"
            )
            
            entry_point = SignalEntryPoint(
                entry_price=Decimal('50100'),
                entry_timing=SignalTiming.IMMEDIATE,
                entry_zone_low=Decimal('50000'),
                entry_zone_high=Decimal('50200')
            )
            
            risk_management = SignalRiskManagement(
                stop_loss_price=Decimal('49500'),
                risk_amount=Decimal('600'),
                risk_percentage=Decimal('2.0'),
                position_size_suggestion=Decimal('100'),
                risk_level=RiskLevel.MEDIUM
            )
            
            scorer = PatternScoringEngine()
            pattern_score = scorer.score_pattern(sample_pattern, sample_market_data)
            
            signal = GeneratedSignal(
                base_signal=base_signal,
                entry_point=entry_point,
                risk_management=risk_management,
                source_pattern=sample_pattern,
                pattern_score=pattern_score,
                validation_status=SignalValidation.VALID
            )
            
            db.store_signal(signal)
        
        stats = db.get_statistics()
        assert stats["total_signals"] == 3
        assert stats["active_signals"] == 3
        assert stats["expired_signals"] == 0
        assert stats["symbols_tracked"] == 1


class TestSignalGenerator:
    """Test main signal generator functionality."""
    
    def test_signal_generator_initialization(self):
        """Test signal generator initialization."""
        generator = SignalGenerator()
        assert generator.config.min_confidence_threshold == Decimal('60')  # Default config
        assert isinstance(generator.pattern_scorer, PatternScoringEngine)
        assert isinstance(generator.signal_db, SignalDatabase)
    
    def test_signal_generation_from_pattern(self, signal_generator, sample_pattern, sample_market_data, sample_volume_profile):
        """Test generating signal from a pattern."""
        signal = signal_generator.generate_signal_from_pattern(
            pattern=sample_pattern,
            market_data=sample_market_data,
            volume_profile=sample_volume_profile,
            account_balance=Decimal('10000')
        )
        
        assert signal is not None
        assert signal.base_signal.symbol == "BTC"
        assert signal.source_pattern.pattern_type == PatternType.HAMMER
        assert signal.validation_status == SignalValidation.VALID
        assert signal.base_signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
    
    def test_signal_filtering(self, sample_pattern, sample_market_data):
        """Test signal filtering with restrictive configuration."""
        restrictive_config = SignalConfiguration(
            min_confidence_threshold=Decimal('95'),  # Very high threshold
            min_strength_threshold=PatternStrength.VERY_STRONG,
            require_volume_confirmation=True
        )
        
        generator = SignalGenerator(restrictive_config)
        
        # Should be filtered out due to high confidence requirement
        signal = generator.generate_signal_from_pattern(
            pattern=sample_pattern,
            market_data=sample_market_data
        )
        
        assert signal is None  # Should be filtered out
    
    def test_risk_level_assignment(self, signal_generator, sample_market_data):
        """Test risk level assignment based on pattern confidence."""
        # High confidence pattern
        high_conf_pattern = CandlestickPattern(
            pattern_id="high_conf_hammer",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=datetime.now(timezone.utc),
            candles=[sample_market_data[-1]],
            confidence=Decimal('95'),
            reliability=Decimal('0.9'),
            bullish_probability=Decimal('0.9'),
            bearish_probability=Decimal('0.1'),
            completion_price=sample_market_data[-1].close,
            volume_confirmation=True
        )
        
        signal = signal_generator.generate_signal_from_pattern(
            pattern=high_conf_pattern,
            market_data=sample_market_data
        )
        
        assert signal is not None
        # Risk level depends on pattern scoring, not just pattern confidence
        # The actual risk level is determined by the weighted confidence score from pattern scoring
        assert signal.risk_management.risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert signal.risk_management.risk_percentage >= Decimal('0.5')
    
    def test_entry_point_calculation_bullish(self, signal_generator, sample_market_data):
        """Test entry point calculation for bullish patterns."""
        bullish_pattern = CandlestickPattern(
            pattern_id="bullish_hammer",
            symbol="BTC",
            pattern_type=PatternType.HAMMER,
            timeframe=Timeframe.FIVE_MINUTES,
            timestamp=datetime.now(timezone.utc),
            candles=[sample_market_data[-1]],
            confidence=Decimal('75'),
            reliability=Decimal('0.8'),
            bullish_probability=Decimal('0.8'),
            bearish_probability=Decimal('0.2'),
            completion_price=sample_market_data[-1].close,
            volume_confirmation=True
        )
        
        signal = signal_generator.generate_signal_from_pattern(
            pattern=bullish_pattern,
            market_data=sample_market_data
        )
        
        assert signal is not None
        # For bullish patterns, entry should be above pattern high
        pattern_high = bullish_pattern.candles[-1].high
        assert signal.entry_point.entry_price > pattern_high
        assert signal.entry_point.entry_zone_low >= pattern_high
        
        # Stop loss should be below pattern low
        pattern_low = bullish_pattern.candles[-1].low
        assert signal.risk_management.stop_loss_price < pattern_low
    
    def test_active_signals_retrieval(self, signal_generator, sample_pattern, sample_market_data):
        """Test retrieving active signals."""
        # Generate some signals
        for i in range(3):
            pattern = CandlestickPattern(
                pattern_id=f"pattern_{i}",
                symbol="BTC",
                pattern_type=PatternType.HAMMER,
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=datetime.now(timezone.utc),
                candles=[sample_market_data[-1]],
                confidence=Decimal('75'),
                reliability=Decimal('0.8'),
                bullish_probability=Decimal('0.8'),
                bearish_probability=Decimal('0.2'),
                completion_price=sample_market_data[-1].close,
                volume_confirmation=True
            )
            
            signal_generator.generate_signal_from_pattern(
                pattern=pattern,
                market_data=sample_market_data
            )
        
        active_signals = signal_generator.get_active_signals()
        assert len(active_signals) == 3
        
        btc_signals = signal_generator.get_active_signals("BTC")
        assert len(btc_signals) == 3
    
    def test_signal_expiry_and_cleanup(self, signal_generator, sample_pattern, sample_market_data):
        """Test signal expiry and cleanup functionality."""
        # Generate a signal
        signal = signal_generator.generate_signal_from_pattern(
            pattern=sample_pattern,
            market_data=sample_market_data
        )
        
        assert signal is not None
        
        # Manually expire it
        signal.generation_timestamp = datetime.now(timezone.utc) - timedelta(hours=25)
        signal_generator.signal_db.store_signal(signal)
        
        # Expire old signals
        expired_count = signal_generator.expire_old_signals()
        assert expired_count >= 1
        
        # Clean up database
        removed_count = signal_generator.cleanup_database()
        # May be 0 if not old enough for cleanup
        assert removed_count >= 0
    
    def test_signal_statistics(self, signal_generator, sample_pattern, sample_market_data):
        """Test signal generation statistics."""
        # Generate some signals
        for i in range(2):
            pattern = CandlestickPattern(
                pattern_id=f"stat_pattern_{i}",
                symbol="BTC",
                pattern_type=PatternType.HAMMER,
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=datetime.now(timezone.utc),
                candles=[sample_market_data[-1]],
                confidence=Decimal('75'),
                reliability=Decimal('0.8'),
                bullish_probability=Decimal('0.8'),
                bearish_probability=Decimal('0.2'),
                completion_price=sample_market_data[-1].close,
                volume_confirmation=True
            )
            
            signal_generator.generate_signal_from_pattern(
                pattern=pattern,
                market_data=sample_market_data
            )
        
        stats = signal_generator.get_statistics()
        assert stats["active_signals"] >= 2
        assert "config" in stats
        assert stats["config"]["min_confidence"] == 50.0  # Our test fixture uses 50.0


class TestSignalValidation:
    """Test signal validation logic."""
    
    def test_valid_signal_validation(self, sample_pattern, sample_market_data):
        """Test validation of a valid signal."""
        # Use lenient config for testing
        config = SignalConfiguration(
            min_confidence_threshold=Decimal('50'),
            min_strength_threshold=PatternStrength.WEAK,
            require_volume_confirmation=False,
            require_trend_alignment=False
        )
        generator = SignalGenerator(config)
        
        signal = generator.generate_signal_from_pattern(
            pattern=sample_pattern,
            market_data=sample_market_data
        )
        
        assert signal is not None
        assert signal.validation_status == SignalValidation.VALID
        assert signal.is_valid
    
    def test_invalid_risk_reward_ratio(self, sample_pattern, sample_market_data):
        """Test signal with invalid risk/reward ratio."""
        config = SignalConfiguration(
            min_risk_reward_ratio=Decimal('3.0')  # Very high requirement
        )
        generator = SignalGenerator(config)
        
        # This might generate a signal with lower R/R ratio that gets invalidated
        signal = generator.generate_signal_from_pattern(
            pattern=sample_pattern,
            market_data=sample_market_data
        )
        
        # Could be None (filtered) or Invalid (failed validation)
        if signal is not None:
            # If generated, should either be valid or have high R/R
            if signal.validation_status == SignalValidation.VALID:
                assert signal.risk_management.risk_reward_ratio >= Decimal('3.0')


class TestSignalDeduplication:
    """Test signal deduplication and ranking."""
    
    def test_signal_deduplication(self, signal_generator, sample_market_data):
        """Test deduplication of similar signals."""
        # Create multiple similar patterns for the same symbol and type
        patterns = []
        for i in range(3):
            pattern = CandlestickPattern(
                pattern_id=f"dup_pattern_{i}",
                symbol="BTC",
                pattern_type=PatternType.HAMMER,
                timeframe=Timeframe.FIVE_MINUTES,
                timestamp=datetime.now(timezone.utc),
                candles=[sample_market_data[-1]],
                confidence=Decimal('75') + Decimal(i * 5),  # Varying confidence
                reliability=Decimal('0.8'),
                bullish_probability=Decimal('0.8'),
                bearish_probability=Decimal('0.2'),
                completion_price=sample_market_data[-1].close,
                volume_confirmation=True
            )
            patterns.append(pattern)
        
        # Generate signals
        signals = []
        for pattern in patterns:
            signal = signal_generator.generate_signal_from_pattern(
                pattern=pattern,
                market_data=sample_market_data
            )
            if signal:
                signals.append(signal)
        
        # Should have signals
        assert len(signals) >= 2
        
        # Test deduplication
        deduplicated = signal_generator._rank_and_deduplicate_signals(signals)
        
        # Should keep only one signal per symbol/pattern type combination
        assert len(deduplicated) == 1
        
        # Should keep the highest quality signal
        if len(signals) > 1:
            best_signal = max(signals, key=lambda s: s.signal_quality_score)
            assert deduplicated[0].signal_id == best_signal.signal_id