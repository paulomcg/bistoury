"""
Signal Manager Core Implementation - Task 9.4

Main SignalManager class integrating aggregation and narrative preservation
for the bootstrap strategy Phase 1 implementation.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from uuid import uuid4

from src.bistoury.signal_manager.models import (
    AggregatedSignal,
    SignalManagerConfiguration,
    TemporalSignalBuffer,
    SignalQuality,
    SignalWeight,
    SignalConflict
)
from src.bistoury.signal_manager.aggregator import (
    SignalAggregator,
    ConflictResolver,
    SignalValidator,
    SignalScorer,
    WeightManager
)
from src.bistoury.signal_manager.narrative_buffer import (
    NarrativeBuffer,
    NarrativeBufferConfig
)
from src.bistoury.models.signals import TradingSignal, SignalDirection, RiskLevel, Timeframe
from src.bistoury.strategies.narrative_generator import TradingNarrative


class SignalManagerStatus:
    """Signal Manager operational status tracking"""
    
    def __init__(self):
        self.is_running: bool = False
        self.start_time: Optional[datetime] = None
        self.signals_processed: int = 0
        self.signals_published: int = 0
        self.active_strategies: Set[str] = set()
        self.last_signal_time: Optional[datetime] = None
        self.processing_latency_ms: float = 0.0
        self.error_count: int = 0
        self.last_error: Optional[str] = None


class SignalManagerMetrics:
    """Performance and health metrics for Signal Manager"""
    
    def __init__(self):
        self.signals_per_minute: float = 0.0
        self.average_confidence: float = 0.0
        self.quality_distribution: Dict[str, int] = {}
        self.strategy_performance: Dict[str, float] = {}
        self.conflict_rate: float = 0.0
        self.narrative_storage_rate: float = 0.0
        self.memory_usage_mb: float = 0.0
        self.cpu_usage_percent: float = 0.0


class SignalManager:
    """
    Main Signal Manager class integrating mathematical aggregation and narrative preservation.
    
    Implements the bootstrap strategy Phase 1:
    - Mathematical signal aggregation for immediate trading capability
    - Narrative preservation for future Phase 2 temporal evolution
    - Real-time signal processing with sub-second latency
    - Strategy subscription and dynamic configuration management
    """
    
    def __init__(
        self,
        config: Optional[SignalManagerConfiguration] = None,
        narrative_config: Optional[NarrativeBufferConfig] = None
    ):
        """Initialize Signal Manager with configuration"""
        self.config = config or SignalManagerConfiguration()
        self.narrative_config = narrative_config or NarrativeBufferConfig()
        
        # Core components
        self.aggregator = SignalAggregator(self.config)
        self.conflict_resolver = ConflictResolver(self.config)
        self.validator = SignalValidator(self.config)
        self.scorer = SignalScorer(self.config)
        self.weight_manager = WeightManager(self.config)
        
        # Narrative preservation system
        self.narrative_buffer: Optional[NarrativeBuffer] = None
        
        # Signal storage and management
        self.temporal_buffer = TemporalSignalBuffer(max_age=self.config.temporal_buffer_age)
        self.active_signals: Dict[str, TradingSignal] = {}
        self.strategy_subscriptions: Set[str] = set()
        
        # Status and metrics
        self.status = SignalManagerStatus()
        self.metrics = SignalManagerMetrics()
        
        # Event callbacks
        self.signal_callbacks: List[Callable[[AggregatedSignal], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    async def start(self) -> None:
        """Start the Signal Manager"""
        if self.status.is_running:
            self.logger.warning("Signal Manager is already running")
            return
            
        try:
            self.logger.info("Starting Signal Manager...")
            
            # Initialize narrative buffer
            if self.config.preserve_narratives:
                self.narrative_buffer = NarrativeBuffer(self.narrative_config)
                await self.narrative_buffer.start()
                self.logger.info("Narrative buffer initialized")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Update status
            self.status.is_running = True
            self.status.start_time = datetime.now(timezone.utc)
            
            self.logger.info("Signal Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Signal Manager: {e}")
            await self._handle_error(e)
            raise
    
    async def stop(self) -> None:
        """Stop the Signal Manager"""
        if not self.status.is_running:
            self.logger.warning("Signal Manager is not running")
            return
            
        try:
            self.logger.info("Stopping Signal Manager...")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop narrative buffer
            if self.narrative_buffer:
                await self.narrative_buffer.stop()
                self.logger.info("Narrative buffer stopped")
            
            # Update status
            self.status.is_running = False
            
            self.logger.info("Signal Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Signal Manager: {e}")
            await self._handle_error(e)
    
    async def process_signal(
        self,
        signal: TradingSignal,
        narrative: Optional[TradingNarrative] = None,
        strategy_id: str = "unknown"
    ) -> Optional[AggregatedSignal]:
        """
        Process a new trading signal with optional narrative.
        
        This is the main entry point for signal processing, implementing
        the dual-path architecture of the bootstrap strategy.
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate signal (Bootstrap Phase 1 simple validation)
            if not self._validate_trading_signal(signal):
                self.logger.warning(f"Signal validation failed for {strategy_id}")
                return None
            
            # Store in temporal buffer (simplified for bootstrap strategy)
            # Will be enhanced in Phase 2 with proper AggregatedSignal storage
            
            # Preserve narrative if provided
            if narrative and self.narrative_buffer:
                await self._store_narrative(signal, narrative, strategy_id)
            
            # Add to active signals
            signal_id = f"{strategy_id}_{signal.symbol}_{int(signal.timestamp.timestamp())}"
            self.active_signals[signal_id] = signal
            
            # Update strategy subscription
            self.strategy_subscriptions.add(strategy_id)
            self.status.active_strategies.add(strategy_id)
            
            # Aggregate with existing signals
            aggregated = await self._aggregate_signals(signal, strategy_id)
            
            # Update metrics
            await self._update_metrics(signal, aggregated, start_time)
            
            # Publish if meets criteria
            if aggregated and aggregated.quality.is_tradeable:
                await self._publish_signal(aggregated)
                self.status.signals_published += 1
            
            self.status.signals_processed += 1
            self.status.last_signal_time = datetime.now(timezone.utc)
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error processing signal from {strategy_id}: {e}")
            await self._handle_error(e)
            return None
    
    async def _aggregate_signals(
        self,
        new_signal: TradingSignal,
        strategy_id: str
    ) -> Optional[AggregatedSignal]:
        """Aggregate new signal with existing signals (Bootstrap Strategy Phase 1)"""
        
        # Get relevant signals for the same symbol
        relevant_signals = [
            (sid, sig) for sid, sig in self.active_signals.items()
            if sig.symbol == new_signal.symbol and not self._is_expired(sig)
        ]
        
        if len(relevant_signals) < self.config.min_strategies_for_signal:
            self.logger.debug(f"Insufficient signals for aggregation: {len(relevant_signals)}")
            return None
        
        # Extract signals and strategy IDs
        signals = [sig for _, sig in relevant_signals]
        strategy_ids = [sid.split('_')[0] for sid, _ in relevant_signals]
        
        # Bootstrap Strategy Phase 1: Simple mathematical aggregation
        return await self._simple_mathematical_aggregation(signals, strategy_ids)
    
    async def _simple_mathematical_aggregation(
        self,
        signals: List[TradingSignal], 
        strategy_ids: List[str]
    ) -> Optional[AggregatedSignal]:
        """Simple mathematical aggregation for bootstrap strategy Phase 1"""
        
        # Calculate weights for strategies
        weights = {}
        total_weight = 0.0
        for strategy_id in strategy_ids:
            weight_obj = self.weight_manager.get_weight(strategy_id)
            weights[strategy_id] = weight_obj.final_weight
            total_weight += weight_obj.final_weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {sid: w / total_weight for sid, w in weights.items()}
        
        # Aggregate direction and confidence
        buy_weight = 0.0
        sell_weight = 0.0
        total_confidence = 0.0
        
        for signal, strategy_id in zip(signals, strategy_ids):
            weight = weights.get(strategy_id, 0.0)
            confidence = float(signal.confidence)
            
            if signal.direction == SignalDirection.BUY:
                buy_weight += weight * confidence
            elif signal.direction == SignalDirection.SELL:
                sell_weight += weight * confidence
            
            total_confidence += confidence * weight
        
        # Determine final direction
        if buy_weight > sell_weight:
            final_direction = SignalDirection.BUY
            final_confidence = min(100.0, buy_weight)
        elif sell_weight > buy_weight:
            final_direction = SignalDirection.SELL
            final_confidence = min(100.0, sell_weight)
        else:
            final_direction = SignalDirection.HOLD
            final_confidence = 50.0
        
        # Create basic quality assessment
        quality = SignalQuality.calculate_quality(
            consensus_score=85.0,  # Simple default for Phase 1
            confidence_score=final_confidence,
            conflicts=[],  # No conflicts in Phase 1
            contributing_strategies=len(strategy_ids)
        )
        
        # Create aggregated signal
        return AggregatedSignal(
            direction=final_direction,
            confidence=final_confidence,
            weight=min(1.0, final_confidence / 100.0),
            contributing_strategies=strategy_ids,
            strategy_weights=weights,
            conflicts=[],  # No conflicts in Phase 1 bootstrap
            quality=quality,
            risk_level=RiskLevel.MEDIUM,  # Default for Phase 1
            expiry=datetime.now(timezone.utc) + self.config.max_signal_age,
            narrative_summary=f"Bootstrap Phase 1: {final_direction.value} from {len(strategy_ids)} strategies"
        )
    
    async def _store_narrative(
        self,
        signal: TradingSignal,
        narrative: TradingNarrative,
        strategy_id: str
    ) -> None:
        """Store narrative in the preservation system"""
        if not self.narrative_buffer:
            return
            
        try:
            # Generate signal ID
            signal_id = f"{strategy_id}_{signal.symbol}_{int(signal.timestamp.timestamp())}"
            
            # Store narrative with metadata
            await self.narrative_buffer.store_narrative(
                signal_id=signal_id,
                strategy_id=strategy_id,
                symbol=signal.symbol,
                direction=signal.direction,
                confidence=signal.confidence,
                timeframe=signal.timeframe,
                narrative=narrative
            )
            
            # Update temporal buffer mapping (simplified for bootstrap strategy)
            # Will be enhanced in Phase 2 with proper narrative timeline integration
            
        except Exception as e:
            self.logger.error(f"Error storing narrative: {e}")
            await self._handle_error(e)
    
    async def _publish_signal(self, signal: AggregatedSignal) -> None:
        """Publish aggregated signal to subscribers"""
        try:
            # Call registered callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    self.logger.error(f"Error in signal callback: {e}")
            
            self.logger.info(
                f"Published {signal.direction.value} signal for strategies "
                f"{signal.contributing_strategies} with confidence {signal.confidence:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error publishing signal: {e}")
            await self._handle_error(e)
    
    async def _update_metrics(
        self,
        signal: TradingSignal,
        aggregated: Optional[AggregatedSignal],
        start_time: datetime
    ) -> None:
        """Update performance metrics"""
        try:
            # Processing latency
            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.status.processing_latency_ms = latency
            
            # Update metrics
            if aggregated:
                self.metrics.average_confidence = (
                    self.metrics.average_confidence * 0.9 + aggregated.confidence * 0.1
                )
                
                # Quality distribution
                grade = aggregated.quality.grade.value
                self.metrics.quality_distribution[grade] = (
                    self.metrics.quality_distribution.get(grade, 0) + 1
                )
            
            # Strategy performance tracking
            # Performance is tracked via WeightManager when signals succeed/fail
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        tasks = [
            self._cleanup_task(),
            self._metrics_task(),
            self._health_check_task()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks"""
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
    
    async def _cleanup_task(self) -> None:
        """Background task for cleaning up expired signals"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up expired signals
                expired_signals = [
                    signal_id for signal_id, signal in self.active_signals.items()
                    if self._is_expired(signal)
                ]
                
                for signal_id in expired_signals:
                    del self.active_signals[signal_id]
                
                if expired_signals:
                    self.logger.debug(f"Cleaned up {len(expired_signals)} expired signals")
                
                # Clean up temporal buffer if available
                if self.temporal_buffer:
                    self.temporal_buffer._cleanup_expired()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_task(self) -> None:
        """Background task for updating metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Calculate signals per minute
                if self.status.start_time:
                    runtime_minutes = (
                        datetime.now(timezone.utc) - self.status.start_time
                    ).total_seconds() / 60
                    
                    if runtime_minutes > 0:
                        self.metrics.signals_per_minute = (
                            self.status.signals_processed / runtime_minutes
                        )
                
                # Update strategy performance
                for strategy_id in self.status.active_strategies:
                    weight = self.weight_manager.get_weight(strategy_id)
                    self.metrics.strategy_performance[strategy_id] = weight.recent_success_rate
                
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics task: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_task(self) -> None:
        """Background task for health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                # Check if we're receiving signals
                if self.status.last_signal_time:
                    time_since_last = (
                        datetime.now(timezone.utc) - self.status.last_signal_time
                    ).total_seconds()
                    
                    if time_since_last > 300:  # 5 minutes
                        self.logger.warning(
                            f"No signals received for {time_since_last:.0f} seconds"
                        )
                
                # Check processing latency
                if self.status.processing_latency_ms > 1000:  # 1 second
                    self.logger.warning(
                        f"High processing latency: {self.status.processing_latency_ms:.1f}ms"
                    )
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check task: {e}")
                await asyncio.sleep(120)
    
    def _validate_trading_signal(self, signal: TradingSignal) -> bool:
        """Simple validation for TradingSignal objects (Bootstrap Phase 1)"""
        try:
            # Check required fields
            if not signal.signal_id or not signal.symbol or not signal.direction:
                return False
            
            # Check confidence range
            confidence = float(signal.confidence)
            if confidence < 0 or confidence > 100:
                return False
            
            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                return False
            
            # Check signal age
            age = datetime.now(timezone.utc) - signal.timestamp
            if age > self.config.max_signal_age:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return False
    
    def _is_expired(self, signal: TradingSignal) -> bool:
        """Check if a signal has expired"""
        age = datetime.now(timezone.utc) - signal.timestamp
        return age > self.config.max_signal_age
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors and notify callbacks"""
        self.status.error_count += 1
        self.status.last_error = str(error)
        
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    # Public API methods
    
    def add_signal_callback(self, callback: Callable[[AggregatedSignal], None]) -> None:
        """Add a callback for published signals"""
        self.signal_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback for errors"""
        self.error_callbacks.append(callback)
    
    def get_status(self) -> SignalManagerStatus:
        """Get current status"""
        return self.status
    
    def get_metrics(self) -> SignalManagerMetrics:
        """Get current metrics"""
        return self.metrics
    
    def get_active_signals(self) -> Dict[str, TradingSignal]:
        """Get currently active signals"""
        return self.active_signals.copy()
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return {
            strategy_id: self.weight_manager.get_weight(strategy_id).final_weight
            for strategy_id in self.status.active_strategies
        }
    
    async def update_strategy_weight(self, strategy_id: str, weight: float) -> None:
        """Update strategy weight dynamically"""
        try:
            self.weight_manager.update_weight(strategy_id, weight)
            self.logger.info(f"Updated weight for {strategy_id} to {weight}")
        except Exception as e:
            self.logger.error(f"Error updating strategy weight: {e}")
            await self._handle_error(e)
    
    async def get_narrative_timeline(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        timeframe: Optional[Timeframe] = None,
        limit: int = 100
    ) -> List[Any]:
        """Get narrative timeline for analysis"""
        if not self.narrative_buffer:
            return []
        
        try:
            return await self.narrative_buffer.get_timeline(
                symbol=symbol,
                strategy_id=strategy_id,
                timeframe=timeframe,
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Error retrieving narrative timeline: {e}")
            await self._handle_error(e)
            return [] 