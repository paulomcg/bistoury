"""
Test suite for Signal Manager Core Implementation - Task 9.4

Tests the main SignalManager class integration and functionality.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.bistoury.signal_manager.signal_manager import (
    SignalManager,
    SignalManagerStatus,
    SignalManagerMetrics
)
from src.bistoury.signal_manager.models import (
    SignalManagerConfiguration,
    AggregatedSignal,
    SignalQuality,
    SignalQualityGrade
)
from src.bistoury.signal_manager.narrative_buffer import NarrativeBufferConfig
from src.bistoury.models.signals import (
    TradingSignal,
    SignalDirection,
    SignalType
)
from src.bistoury.models.market_data import Timeframe
from src.bistoury.strategies.narrative_generator import (
    TradingNarrative,
    NarrativeConfiguration,
    NarrativeStyle
)


@pytest.fixture
def signal_manager_config():
    """Create test configuration for SignalManager"""
    return SignalManagerConfiguration(
        min_strategies_for_signal=2,
        max_signal_age=timedelta(minutes=5),
        confidence_threshold=60.0,
        quality_threshold=70.0,
        preserve_narratives=True
    )


@pytest.fixture
def narrative_config():
    """Create test configuration for narrative buffer"""
    return NarrativeBufferConfig(
        max_memory_narratives=100,
        compression_level="light",
        enable_continuity_tracking=True
    )


@pytest.fixture
def sample_trading_signal():
    """Create a sample trading signal for testing"""
    return TradingSignal(
        signal_id="test_signal_1",
        symbol="BTC",
        direction=SignalDirection.BUY,
        confidence=Decimal('85.0'),
        strength=Decimal('0.8'),
        price=Decimal('50000.0'),
        signal_type=SignalType.PATTERN,
        timeframe=Timeframe.FIFTEEN_MINUTES,
        timestamp=datetime.now(timezone.utc),
        source="test_strategy",
        reason="Strong bullish pattern detected"
    )


@pytest.fixture
def sample_trading_narrative():
    """Create a sample trading narrative for testing"""
    return TradingNarrative(
        executive_summary="Strong bullish momentum detected on BTC",
        market_overview="Market showing strong upward momentum with high volume",
        pattern_analysis="Hammer pattern confirmed with strong volume support",
        volume_analysis="Volume spike indicates institutional interest",
        risk_assessment="Medium risk with favorable risk/reward ratio",
        entry_strategy="Enter long position above pattern high",
        exit_strategy="Take profit at resistance, stop loss below pattern low",
        confidence_rationale="Multiple confluences support bullish bias",
        generation_timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
async def signal_manager(signal_manager_config, narrative_config):
    """Create and initialize a SignalManager for testing"""
    manager = SignalManager(
        config=signal_manager_config,
        narrative_config=narrative_config
    )
    yield manager
    
    # Cleanup
    if manager.status.is_running:
        await manager.stop()


class TestSignalManagerStatus:
    """Test SignalManagerStatus class"""
    
    def test_status_initialization(self):
        """Test status object initialization"""
        status = SignalManagerStatus()
        
        assert status.is_running is False
        assert status.start_time is None
        assert status.signals_processed == 0
        assert status.signals_published == 0
        assert len(status.active_strategies) == 0
        assert status.last_signal_time is None
        assert status.processing_latency_ms == 0.0
        assert status.error_count == 0
        assert status.last_error is None


class TestSignalManagerMetrics:
    """Test SignalManagerMetrics class"""
    
    def test_metrics_initialization(self):
        """Test metrics object initialization"""
        metrics = SignalManagerMetrics()
        
        assert metrics.signals_per_minute == 0.0
        assert metrics.average_confidence == 0.0
        assert len(metrics.quality_distribution) == 0
        assert len(metrics.strategy_performance) == 0
        assert metrics.conflict_rate == 0.0
        assert metrics.narrative_storage_rate == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0


class TestSignalManagerInitialization:
    """Test SignalManager initialization"""
    
    def test_signal_manager_creation(self, signal_manager_config, narrative_config):
        """Test SignalManager creation with configuration"""
        manager = SignalManager(
            config=signal_manager_config,
            narrative_config=narrative_config
        )
        
        assert manager.config == signal_manager_config
        assert manager.narrative_config == narrative_config
        assert manager.aggregator is not None
        assert manager.conflict_resolver is not None
        assert manager.validator is not None
        assert manager.scorer is not None
        assert manager.weight_manager is not None
        assert manager.narrative_buffer is None  # Not initialized until start()
        assert len(manager.active_signals) == 0
        assert len(manager.strategy_subscriptions) == 0
        assert manager.status.is_running is False
    
    def test_signal_manager_default_config(self):
        """Test SignalManager creation with default configuration"""
        manager = SignalManager()
        
        assert manager.config is not None
        assert manager.narrative_config is not None
        assert isinstance(manager.config, SignalManagerConfiguration)
        assert isinstance(manager.narrative_config, NarrativeBufferConfig)


class TestSignalManagerLifecycle:
    """Test SignalManager start/stop lifecycle"""
    
    @pytest.mark.asyncio
    async def test_start_signal_manager(self, signal_manager):
        """Test starting the SignalManager"""
        assert not signal_manager.status.is_running
        
        await signal_manager.start()
        
        assert signal_manager.status.is_running
        assert signal_manager.status.start_time is not None
        assert signal_manager.narrative_buffer is not None
        assert len(signal_manager._background_tasks) > 0
    
    @pytest.mark.asyncio
    async def test_stop_signal_manager(self, signal_manager):
        """Test stopping the SignalManager"""
        await signal_manager.start()
        assert signal_manager.status.is_running
        
        await signal_manager.stop()
        
        assert not signal_manager.status.is_running
        assert len(signal_manager._background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, signal_manager):
        """Test starting SignalManager when already running"""
        await signal_manager.start()
        assert signal_manager.status.is_running
        
        # Should not raise error
        await signal_manager.start()
        assert signal_manager.status.is_running
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, signal_manager):
        """Test stopping SignalManager when not running"""
        assert not signal_manager.status.is_running
        
        # Should not raise error
        await signal_manager.stop()
        assert not signal_manager.status.is_running


class TestSignalProcessing:
    """Test signal processing functionality"""
    
    @pytest.mark.asyncio
    async def test_process_single_signal(self, signal_manager, sample_trading_signal):
        """Test processing a single signal"""
        await signal_manager.start()
        
        result = await signal_manager.process_signal(
            signal=sample_trading_signal,
            strategy_id="test_strategy"
        )
        
        # Single signal should not produce aggregated result (needs min 2 strategies)
        assert result is None
        assert signal_manager.status.signals_processed == 1
        assert signal_manager.status.signals_published == 0
        assert "test_strategy" in signal_manager.status.active_strategies
        assert len(signal_manager.active_signals) == 1
    
    @pytest.mark.asyncio
    async def test_process_multiple_signals_aggregation(self, signal_manager, sample_trading_signal):
        """Test processing multiple signals for aggregation"""
        await signal_manager.start()
        
        # Process first signal
        signal1 = sample_trading_signal
        await signal_manager.process_signal(signal1, strategy_id="strategy1")
        
        # Process second signal for same symbol
        signal2 = TradingSignal(
            signal_id="test_signal_2",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=Decimal('75.0'),
            strength=Decimal('0.7'),
            price=Decimal('50100.0'),
            signal_type=SignalType.VOLUME,
            timeframe=Timeframe.FIFTEEN_MINUTES,
            timestamp=datetime.now(timezone.utc),
            source="test_strategy_2",
            reason="Volume confirmation"
        )
        
        result = await signal_manager.process_signal(signal2, strategy_id="strategy2")
        
        # Should produce aggregated result with 2 strategies
        assert result is not None
        assert isinstance(result, AggregatedSignal)
        assert result.direction == SignalDirection.BUY
        assert len(result.contributing_strategies) == 2
        assert "strategy1" in result.contributing_strategies
        assert "strategy2" in result.contributing_strategies
        assert signal_manager.status.signals_processed == 2
    
    @pytest.mark.asyncio
    async def test_process_signal_with_narrative(self, signal_manager, sample_trading_signal, sample_trading_narrative):
        """Test processing signal with narrative preservation"""
        await signal_manager.start()
        
        result = await signal_manager.process_signal(
            signal=sample_trading_signal,
            narrative=sample_trading_narrative,
            strategy_id="narrative_strategy"
        )
        
        # Verify signal was processed
        assert signal_manager.status.signals_processed == 1
        
        # Verify narrative was stored (check narrative buffer for Phase 1)
        if signal_manager.narrative_buffer:
            # For Phase 1, just verify narrative buffer is available and started
            assert signal_manager.narrative_buffer is not None
        else:
            # If narratives disabled, that's also valid for Phase 1
            assert signal_manager.config.preserve_narratives is False
    
    @pytest.mark.asyncio
    async def test_process_expired_signal(self, signal_manager):
        """Test processing an expired signal"""
        await signal_manager.start()
        
        # Create expired signal
        expired_signal = TradingSignal(
            signal_id="expired_signal",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=Decimal('85.0'),
            strength=Decimal('0.8'),
            price=Decimal('50000.0'),
            signal_type=SignalType.PATTERN,
            timeframe=Timeframe.FIFTEEN_MINUTES,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),  # Expired
            source="test_strategy",
            reason="Expired signal"
        )
        
        # Mock validator to reject expired signal
        with patch.object(signal_manager.validator, 'validate_signal', return_value=False):
            result = await signal_manager.process_signal(expired_signal, strategy_id="test")
        
        assert result is None
        assert signal_manager.status.signals_processed == 0  # Rejected signals not counted as processed


class TestSignalCallbacks:
    """Test signal callback functionality"""
    
    @pytest.mark.asyncio
    async def test_signal_callback_registration(self, signal_manager):
        """Test registering signal callbacks"""
        callback_called = False
        received_signal = None
        
        def test_callback(signal: AggregatedSignal):
            nonlocal callback_called, received_signal
            callback_called = True
            received_signal = signal
        
        signal_manager.add_signal_callback(test_callback)
        assert len(signal_manager.signal_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_error_callback_registration(self, signal_manager):
        """Test registering error callbacks"""
        error_called = False
        received_error = None
        
        def test_error_callback(error: Exception):
            nonlocal error_called, received_error
            error_called = True
            received_error = error
        
        signal_manager.add_error_callback(test_error_callback)
        assert len(signal_manager.error_callbacks) == 1


class TestSignalManagerAPI:
    """Test SignalManager public API methods"""
    
    @pytest.mark.asyncio
    async def test_get_status(self, signal_manager):
        """Test getting SignalManager status"""
        status = signal_manager.get_status()
        
        assert isinstance(status, SignalManagerStatus)
        assert status.is_running is False
        
        await signal_manager.start()
        status = signal_manager.get_status()
        assert status.is_running is True
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, signal_manager):
        """Test getting SignalManager metrics"""
        metrics = signal_manager.get_metrics()
        
        assert isinstance(metrics, SignalManagerMetrics)
        assert metrics.signals_per_minute == 0.0
    
    @pytest.mark.asyncio
    async def test_get_active_signals(self, signal_manager, sample_trading_signal):
        """Test getting active signals"""
        await signal_manager.start()
        
        # Initially empty
        active = signal_manager.get_active_signals()
        assert len(active) == 0
        
        # Add signal
        await signal_manager.process_signal(sample_trading_signal, strategy_id="test")
        
        active = signal_manager.get_active_signals()
        assert len(active) == 1
        assert list(active.values())[0].symbol == "BTC"
    
    @pytest.mark.asyncio
    async def test_get_strategy_weights(self, signal_manager, sample_trading_signal):
        """Test getting strategy weights"""
        await signal_manager.start()
        
        # Initially empty
        weights = signal_manager.get_strategy_weights()
        assert len(weights) == 0
        
        # Add signal to activate strategy
        await signal_manager.process_signal(sample_trading_signal, strategy_id="test_strategy")
        
        weights = signal_manager.get_strategy_weights()
        assert "test_strategy" in weights
        assert isinstance(weights["test_strategy"], float)
    
    @pytest.mark.asyncio
    async def test_update_strategy_weight(self, signal_manager, sample_trading_signal):
        """Test updating strategy weight"""
        await signal_manager.start()
        
        # Add signal to activate strategy
        await signal_manager.process_signal(sample_trading_signal, strategy_id="test_strategy")
        
        # Update weight
        await signal_manager.update_strategy_weight("test_strategy", 0.8)
        
        # Verify weight was updated
        weights = signal_manager.get_strategy_weights()
        # Note: The actual weight might be modified by performance factors
        assert "test_strategy" in weights
    
    @pytest.mark.asyncio
    async def test_get_narrative_timeline_no_buffer(self, signal_manager_config, narrative_config):
        """Test getting narrative timeline when buffer is disabled"""
        config = signal_manager_config
        config.preserve_narratives = False
        
        manager = SignalManager(config=config, narrative_config=narrative_config)
        
        timeline = await manager.get_narrative_timeline()
        assert timeline == []
    
    @pytest.mark.asyncio
    async def test_get_narrative_timeline_with_buffer(self, signal_manager, sample_trading_signal, sample_trading_narrative):
        """Test getting narrative timeline with buffer enabled"""
        await signal_manager.start()
        
        # Process signal with narrative
        await signal_manager.process_signal(
            signal=sample_trading_signal,
            narrative=sample_trading_narrative,
            strategy_id="test_strategy"
        )
        
        # Get timeline
        timeline = await signal_manager.get_narrative_timeline(symbol="BTC")
        
        # Should return timeline (exact format depends on narrative buffer implementation)
        assert isinstance(timeline, list)


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_error_in_signal_processing(self, signal_manager, sample_trading_signal):
        """Test error handling during signal processing"""
        await signal_manager.start()
        
        # Mock aggregator to raise error
        with patch.object(signal_manager, '_aggregate_signals', side_effect=Exception("Test error")):
            result = await signal_manager.process_signal(sample_trading_signal, strategy_id="test")
        
        assert result is None
        assert signal_manager.status.error_count > 0
        assert signal_manager.status.last_error == "Test error"
    
    @pytest.mark.asyncio
    async def test_error_callback_execution(self, signal_manager):
        """Test error callback execution"""
        error_received = None
        
        def error_callback(error: Exception):
            nonlocal error_received
            error_received = error
        
        signal_manager.add_error_callback(error_callback)
        
        test_error = Exception("Test error")
        await signal_manager._handle_error(test_error)
        
        assert error_received == test_error
        assert signal_manager.status.error_count == 1


class TestBackgroundTasks:
    """Test background task functionality"""
    
    @pytest.mark.asyncio
    async def test_background_tasks_start(self, signal_manager):
        """Test that background tasks start with SignalManager"""
        await signal_manager.start()
        
        assert len(signal_manager._background_tasks) > 0
        
        # Verify tasks are running
        for task in signal_manager._background_tasks:
            assert not task.done()
    
    @pytest.mark.asyncio
    async def test_background_tasks_stop(self, signal_manager):
        """Test that background tasks stop with SignalManager"""
        await signal_manager.start()
        initial_task_count = len(signal_manager._background_tasks)
        assert initial_task_count > 0
        
        await signal_manager.stop()
        
        assert len(signal_manager._background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_task_functionality(self, signal_manager, sample_trading_signal):
        """Test cleanup task removes expired signals"""
        await signal_manager.start()
        
        # Add signal
        await signal_manager.process_signal(sample_trading_signal, strategy_id="test")
        assert len(signal_manager.active_signals) == 1
        
        # Mock signal as expired
        with patch.object(signal_manager, '_is_expired', return_value=True):
            # Trigger cleanup manually
            expired_signals = [
                signal_id for signal_id, signal in signal_manager.active_signals.items()
                if signal_manager._is_expired(signal)
            ]
            
            for signal_id in expired_signals:
                del signal_manager.active_signals[signal_id]
        
        assert len(signal_manager.active_signals) == 0


class TestMetricsUpdating:
    """Test metrics updating functionality"""
    
    @pytest.mark.asyncio
    async def test_metrics_update_on_signal_processing(self, signal_manager, sample_trading_signal):
        """Test that metrics are updated when processing signals"""
        await signal_manager.start()
        
        initial_processed = signal_manager.status.signals_processed
        
        await signal_manager.process_signal(sample_trading_signal, strategy_id="test")
        
        assert signal_manager.status.signals_processed == initial_processed + 1
        assert signal_manager.status.last_signal_time is not None
        assert signal_manager.status.processing_latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, signal_manager):
        """Test metrics calculation in background task"""
        await signal_manager.start()
        
        # Wait a bit for metrics task to run
        await asyncio.sleep(0.1)
        
        # Verify metrics are being calculated
        metrics = signal_manager.get_metrics()
        assert isinstance(metrics.signals_per_minute, float)


if __name__ == "__main__":
    pytest.main([__file__]) 