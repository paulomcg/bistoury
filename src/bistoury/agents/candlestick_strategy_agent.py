"""
Candlestick Strategy Agent

Integrates candlestick pattern analysis into the multi-agent framework.
Consumes real-time market data and generates trading signals based on pattern recognition.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
import yaml
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict, deque

from ..agents.base import BaseAgent, AgentType, AgentCapability, AgentHealth, AgentState
from ..agents.messaging import MessageType, Message, MessagePriority
from ..models.signals import CandlestickData, TradingSignal, AnalysisContext, SignalDirection, SignalType
from ..models.agent_messages import TradingSignalPayload
from ..strategies.candlestick_models import (
    StrategyConfiguration, 
    MultiTimeframePattern,
    PatternStrength,
    TimeframePriority
)
from ..strategies.patterns.single_candlestick import SinglePatternRecognizer
from ..strategies.patterns.multi_candlestick import MultiPatternRecognizer
from ..strategies.timeframe_analyzer import TimeframeAnalyzer
from ..strategies.pattern_scoring import PatternScoringEngine
from ..strategies.signal_generator import SignalGenerator, SignalConfiguration
from ..strategies.narrative_generator import NarrativeGenerator, NarrativeConfiguration, NarrativeStyle
from ..models.market_data import Timeframe


@dataclass
class CandlestickStrategyConfig:
    """Configuration for the candlestick strategy agent."""
    
    # Market data configuration
    symbols: List[str] = field(default_factory=lambda: ["BTC", "ETH"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    
    # Pattern analysis configuration
    min_confidence_threshold: float = 0.60
    min_pattern_strength: PatternStrength = PatternStrength.MODERATE
    enable_single_patterns: bool = True
    enable_multi_patterns: bool = True
    
    # Signal generation configuration
    signal_expiry_minutes: int = 15
    max_signals_per_symbol: int = 3
    
    # Narrative configuration
    narrative_style: NarrativeStyle = NarrativeStyle.TECHNICAL
    include_technical_details: bool = True
    include_risk_metrics: bool = True
    
    # Performance configuration
    max_data_buffer_size: int = 1000
    data_cleanup_interval_seconds: int = 300
    health_check_interval_seconds: int = 30
    
    # Agent configuration
    agent_name: str = "candlestick_strategy"
    agent_version: str = "1.0.0"

    def __post_init__(self):
        # Ensure that the symbols and timeframes are valid
        self.symbols = [symbol for symbol in self.symbols if symbol in ["BTC", "ETH"]]
        self.timeframes = [timeframe for timeframe in self.timeframes if timeframe in ["1m", "5m", "15m"]]

    def to_dict(self):
        return asdict(self)

    def to_yaml(self):
        return yaml.dump(self.to_dict())


@dataclass
class StrategyPerformanceMetrics:
    """Performance metrics for the strategy agent."""
    
    signals_generated: int = 0
    patterns_detected: int = 0
    high_confidence_signals: int = 0
    processing_latency_ms: float = 0.0
    data_messages_received: int = 0
    errors_count: int = 0
    uptime_seconds: float = 0.0
    last_signal_time: Optional[datetime] = None
    average_confidence: float = 0.0
    
    def update_processing_latency(self, latency_ms: float):
        """Update processing latency with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        self.processing_latency_ms = (alpha * latency_ms + 
                                    (1 - alpha) * self.processing_latency_ms)


class CandlestickStrategyAgent(BaseAgent):
    """
    Candlestick strategy agent that performs real-time pattern analysis
    and generates trading signals based on candlestick patterns.
    """
    
    def __init__(self, config: Optional[CandlestickStrategyConfig] = None):
        # Initialize configuration - store it first before BaseAgent overwrites it
        strategy_config = config or CandlestickStrategyConfig()

        # Initialize base agent
        super().__init__(
            name=strategy_config.agent_name,
            agent_type=AgentType.STRATEGY
        )
        
        # Restore our strategy configuration after BaseAgent initialization
        self.config = strategy_config
        
        # Set version and capabilities after base initialization
        self.metadata.version = self.config.agent_version
        self.capabilities = [
            AgentCapability("pattern_recognition", "Candlestick pattern analysis", "1.0.0"),
            AgentCapability("signal_generation", "Trading signal generation", "1.0.0"),
            AgentCapability("narrative_generation", "Human-readable analysis", "1.0.0"),
            AgentCapability("multi_timeframe", "Multi-timeframe analysis", "1.0.0"),
            AgentCapability("real_time_processing", "Real-time data processing", "1.0.0")
        ]
        
        # Initialize strategy components
        self._initialize_strategy_components()
        
        # Initialize data management
        self.market_data_buffer: Dict[str, Dict[str, List[CandlestickData]]] = {}
        self.active_signals: Dict[str, List] = {}
        self.performance_metrics = StrategyPerformanceMetrics()
        
        # Initialize subscriptions
        self.subscribed_topics: Set[str] = set()
        
        # Message bus (will be set externally)
        self._message_bus = None
        
    def _initialize_strategy_components(self):
        """Initialize all strategy analysis components."""
        
        # Create strategy configuration
        strategy_config = StrategyConfiguration(
            confidence_threshold=self.config.min_confidence_threshold,
            pattern_strength_threshold=self.config.min_pattern_strength,
            timeframe_priorities={
                "1m": TimeframePriority.ONE_MINUTE,
                "5m": TimeframePriority.FIVE_MINUTES, 
                "15m": TimeframePriority.FIFTEEN_MINUTES
            }
        )
        
        # Initialize pattern recognizers
        self.single_pattern_recognizer = SinglePatternRecognizer()
        self.multi_pattern_recognizer = MultiPatternRecognizer()
        
        # Initialize timeframe analyzer
        self.timeframe_analyzer = TimeframeAnalyzer()
        
        # Initialize pattern scoring engine
        self.pattern_scoring_engine = PatternScoringEngine()
        
        # Initialize signal generator
        signal_config = SignalConfiguration(
            confidence_threshold=self.config.min_confidence_threshold,
            strength_threshold=self.config.min_pattern_strength,
            signal_expiry_minutes=self.config.signal_expiry_minutes
        )
        self.signal_generator = SignalGenerator(signal_config)
        
        # Initialize narrative generator
        narrative_config = NarrativeConfiguration(
            style=self.config.narrative_style,
            include_technical_details=self.config.include_technical_details,
            include_risk_metrics=self.config.include_risk_metrics
        )
        self.narrative_generator = NarrativeGenerator(narrative_config)
        
    async def _start(self):
        """Start the candlestick strategy agent."""
        self.logger.info(f"Starting CandlestickStrategyAgent {self.agent_id}")
        
        try:
            # Skip legacy topic subscriptions - using MessageBus subscriptions instead
            # await self._setup_data_subscriptions()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("CandlestickStrategyAgent started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start CandlestickStrategyAgent: {e}")
            return False
        
    async def _stop(self):
        """Stop the candlestick strategy agent."""
        self.logger.info(f"Stopping CandlestickStrategyAgent {self.agent_id}")
        
        # Unsubscribe from topics
        await self._cleanup_subscriptions()
        
        # Clear data buffers
        self.market_data_buffer.clear()
        self.active_signals.clear()
        
        self.logger.info("CandlestickStrategyAgent stopped successfully")
        
    async def _health_check(self) -> AgentHealth:
        """Perform health check and return health data."""
        return AgentHealth(
            state=self.state,
            last_heartbeat=datetime.now(timezone.utc),
            cpu_usage=0.0,  # Could be implemented with psutil
            memory_usage=0.0,  # Could be implemented with psutil
            error_count=self.performance_metrics.errors_count,
            warning_count=0,
            messages_processed=self.performance_metrics.data_messages_received,
            tasks_completed=self.performance_metrics.signals_generated,
            uptime_seconds=self.performance_metrics.uptime_seconds,
            health_score=1.0 - min(0.5, self.performance_metrics.errors_count * 0.1),
            last_error=None,
            last_error_time=None
        )
        
    async def _setup_data_subscriptions(self):
        """Set up subscriptions to relevant market data topics."""
        topics = []
        
        # Subscribe to candlestick data for each symbol and timeframe
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                topic = f"market_data.candle.{symbol}.{timeframe}"
                topics.append(topic)
                
            # Subscribe to volume data for pattern confirmation
            volume_topic = f"market_data.volume.{symbol}"
            topics.append(volume_topic)
            
        # Subscribe to all topics
        for topic in topics:
            await self.subscribe_to_topic(topic)
            self.subscribed_topics.add(topic)
            
        self.logger.info(f"Subscribed to {len(topics)} market data topics")
        
    async def _cleanup_subscriptions(self):
        """Clean up topic subscriptions."""
        for topic in self.subscribed_topics:
            await self.unsubscribe_from_topic(topic)
        self.subscribed_topics.clear()
        
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        
        async def data_cleanup_task():
            while self.state == AgentState.RUNNING:
                await asyncio.sleep(self.config.data_cleanup_interval_seconds)
                if self.state == AgentState.RUNNING:
                    await self._cleanup_old_data()
                
        async def health_monitoring_task():
            while self.state == AgentState.RUNNING:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                if self.state == AgentState.RUNNING:
                    await self._update_health_metrics()
                
        # Start tasks using the base agent's task management
        self.create_task(data_cleanup_task())
        self.create_task(health_monitoring_task())
        
    async def handle_message(self, message: Message):
        """Handle incoming messages from the message bus."""
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(f"🧠 Strategy received message: type={message.type}, topic={message.topic}")
        
        try:
            if message.type == MessageType.DATA_MARKET_UPDATE:
                await self._handle_market_data(message)
                self.performance_metrics.data_messages_received += 1
                
            elif message.type == MessageType.SYSTEM_CONFIG_UPDATE:
                await self._handle_configuration_update(message)
                
            elif message.type == MessageType.SYSTEM_HEALTH_CHECK:
                await self._handle_health_check_request(message)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.performance_metrics.errors_count += 1
            
        finally:
            # Update processing latency
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            self.performance_metrics.update_processing_latency(latency_ms)
            
    async def _handle_market_data(self, message: Message):
        """Handle incoming market data updates."""
        payload = message.payload
        
        # Extract data details from the MarketDataPayload object
        # Data is stored in the 'data' dictionary within the payload
        if hasattr(payload, 'data') and isinstance(payload.data, dict):
            data_type = payload.data.get("data_type")
            timeframe = payload.data.get("timeframe")
            symbol = payload.symbol  # This is a direct attribute
            
            self.logger.info(f"📊 Strategy received market data: {symbol} {timeframe} ({data_type})")
            
            if data_type == "candlestick" and symbol in self.config.symbols:
                await self._process_candlestick_data(symbol, timeframe, payload.data)
                
            elif data_type == "volume" and symbol in self.config.symbols:
                await self._process_volume_data(symbol, payload.data)
            
    async def _process_candlestick_data(self, symbol: str, timeframe: str, payload: Dict[str, Any]):
        """Process new candlestick data and perform pattern analysis."""
        
        try:
            # Parse candlestick data
            candle_data = CandlestickData(**payload.get("candle_data", {}))
            
            # Add to data buffer
            self._add_to_buffer(symbol, timeframe, candle_data)
            
            # Check if we have enough data for analysis
            if await self._has_sufficient_data(symbol):
                # Perform pattern analysis
                await self._analyze_patterns(symbol)
                
        except Exception as e:
            self.logger.error(f"Error processing candlestick data for {symbol}: {e}")
            self.performance_metrics.errors_count += 1
            
    async def _process_volume_data(self, symbol: str, payload: Dict[str, Any]):
        """Process volume data for pattern confirmation."""
        # Volume data can be used for pattern confirmation
        # Implementation depends on volume data structure
        pass
        
    def _add_to_buffer(self, symbol: str, timeframe: str, candle: CandlestickData):
        """Add candlestick data to the buffer."""
        
        if symbol not in self.market_data_buffer:
            self.market_data_buffer[symbol] = {}
            
        if timeframe not in self.market_data_buffer[symbol]:
            self.market_data_buffer[symbol][timeframe] = []
            
        # Add new candle
        self.market_data_buffer[symbol][timeframe].append(candle)
        
        # Maintain buffer size
        max_size = self.config.max_data_buffer_size
        if len(self.market_data_buffer[symbol][timeframe]) > max_size:
            self.market_data_buffer[symbol][timeframe] = \
                self.market_data_buffer[symbol][timeframe][-max_size:]
        
        buffer_size = len(self.market_data_buffer[symbol][timeframe])
        self.logger.info(f"📈 Buffer updated: {symbol} {timeframe} now has {buffer_size} candles")
                
    async def _has_sufficient_data(self, symbol: str) -> bool:
        """Check if we have sufficient data for pattern analysis."""
        
        if symbol not in self.market_data_buffer:
            self.logger.debug(f"❌ No buffer for {symbol}")
            return False
            
        # Check each required timeframe
        for timeframe in self.config.timeframes:
            if timeframe not in self.market_data_buffer[symbol]:
                self.logger.debug(f"❌ No data for {symbol} {timeframe}")
                return False
                
            # Need at least 2 candles for reliable pattern analysis
            buffer_size = len(self.market_data_buffer[symbol][timeframe])
            if buffer_size < 2:  # Reduced to 2 for immediate testing
                self.logger.debug(f"❌ Insufficient data for {symbol} {timeframe}: {buffer_size}/2 candles")
                return False
        
        self.logger.info(f"✅ Sufficient data for {symbol}, starting pattern analysis")
        return True
        
    async def _analyze_patterns(self, symbol: str):
        """Perform comprehensive pattern analysis for a symbol."""
        
        try:
            self.logger.info(f"🔍 Starting pattern analysis for {symbol}")
            
            # Prepare timeframe data - convert strings to Timeframe enum
            timeframe_data = {}
            
            timeframe_mapping = {
                "1m": Timeframe.ONE_MINUTE,
                "5m": Timeframe.FIVE_MINUTES,
                "15m": Timeframe.FIFTEEN_MINUTES
            }
            
            for timeframe_str in self.config.timeframes:
                if timeframe_str in timeframe_mapping:
                    timeframe_enum = timeframe_mapping[timeframe_str]
                    candles = self.market_data_buffer[symbol][timeframe_str]
                    timeframe_data[timeframe_enum] = candles[-20:]  # Last 20 candles
                    self.logger.info(f"📊 Prepared {len(candles[-20:])} candles for {timeframe_str} analysis")
                
            # Set primary timeframe to the first (and likely only) timeframe we have data for
            primary_timeframe = list(timeframe_data.keys())[0] if timeframe_data else None
            self.logger.info(f"🎯 Using primary timeframe: {primary_timeframe}")
            
            # Perform multi-timeframe analysis
            self.logger.info(f"🔬 Performing timeframe analysis...")
            analysis_result = await self.timeframe_analyzer.analyze(
                symbol,
                timeframe_data,
                primary_timeframe=primary_timeframe
            )
            
            self.logger.info(f"📈 Analysis result: {analysis_result.total_patterns_detected} patterns detected")
            self.logger.info(f"📊 Data quality score: {analysis_result.data_quality_score}")
            self.logger.info(f"⏱️ Meets latency requirement: {analysis_result.meets_latency_requirement}")
            
            # Check if analysis meets our criteria
            has_patterns = analysis_result.total_patterns_detected > 0
            meets_quality = analysis_result.data_quality_score >= Decimal("60")
            meets_latency = analysis_result.meets_latency_requirement
            
            self.logger.info(f"🎯 Analysis criteria: has_patterns={has_patterns}, meets_quality={meets_quality}, meets_latency={meets_latency}")
            
            if has_patterns and meets_quality and meets_latency:
                self.logger.info(f"✅ Analysis meets criteria, generating signals...")
                await self._generate_signals(symbol, analysis_result)
            else:
                self.logger.info(f"❌ Analysis doesn't meet criteria, skipping signal generation")
                
        except Exception as e:
            self.logger.error(f"Error analyzing patterns for {symbol}: {e}")
            self.performance_metrics.errors_count += 1
            
    async def _generate_signals(self, symbol: str, analysis_result):
        """Generate trading signals from pattern analysis."""
        
        try:
            self.logger.info(f"🚀 Generating signals for {symbol}")
            
            # Get all patterns from the analysis
            all_patterns = []
            
            # Collect single patterns with safe dict access
            if hasattr(analysis_result, 'single_patterns') and hasattr(analysis_result.single_patterns, 'items'):
                for timeframe, patterns in analysis_result.single_patterns.items():
                    if patterns:  # Ensure patterns is not None or empty
                        all_patterns.extend(patterns)
                        self.logger.info(f"📊 Found {len(patterns)} single patterns in {timeframe}")
            
            # Collect multi patterns with safe dict access
            if hasattr(analysis_result, 'multi_patterns') and hasattr(analysis_result.multi_patterns, 'items'):
                for timeframe, patterns in analysis_result.multi_patterns.items():
                    if patterns:  # Ensure patterns is not None or empty
                        all_patterns.extend(patterns)
                        self.logger.info(f"📊 Found {len(patterns)} multi patterns in {timeframe}")

            self.logger.info(f"📊 Total patterns collected: {len(all_patterns)}")

            # Get market data for pattern scoring - handle mock objects safely
            market_data = []
            if (hasattr(self, 'market_data_buffer') and 
                isinstance(self.market_data_buffer, dict) and 
                symbol in self.market_data_buffer and
                isinstance(self.market_data_buffer[symbol], dict)):
                
                # Use the most recent timeframe data available
                for timeframe in self.config.timeframes:
                    if (timeframe in self.market_data_buffer[symbol] and 
                        self.market_data_buffer[symbol][timeframe] and
                        hasattr(self.market_data_buffer[symbol][timeframe], '__iter__')):
                        market_data = self.market_data_buffer[symbol][timeframe]
                        break

            # Filter patterns that meet our criteria
            qualifying_patterns = []
            for pattern in all_patterns:
                # Score the pattern using the scoring engine with market data (optional)
                if market_data and hasattr(self, 'pattern_scoring_engine'):
                    try:
                        pattern_score = self.pattern_scoring_engine.score_pattern(pattern, market_data)
                        self.logger.debug(f"🔍 Pattern {pattern.pattern_type} scored: {pattern_score}")
                    except Exception:
                        # If scoring fails, continue without score
                        pass
                
                # Check if pattern meets criteria
                if (hasattr(pattern, 'confidence') and hasattr(pattern, 'reliability') and
                    pattern.confidence >= Decimal(str(self.config.min_confidence_threshold)) and
                    pattern.reliability >= Decimal("0.5")):  # Basic reliability threshold
                    qualifying_patterns.append(pattern)
                    self.logger.info(f"✅ Pattern qualified: {pattern.pattern_type} (confidence: {pattern.confidence})")
                else:
                    self.logger.info(f"❌ Pattern failed criteria: {pattern.pattern_type} (confidence: {getattr(pattern, 'confidence', 'N/A')})")
                        
            self.logger.info(f"🎯 Qualifying patterns: {len(qualifying_patterns)}")
            
            if qualifying_patterns:
                # Get best pattern for signal generation
                best_pattern = max(qualifying_patterns, key=lambda p: p.confidence)
                self.logger.info(f"🏆 Best pattern: {best_pattern.pattern_type} (confidence: {best_pattern.confidence})")
                
                # Generate signal using the analysis result
                signal = await self._create_trading_signal(symbol, best_pattern, analysis_result)
                
                if signal:
                    self.logger.info(f"📈 Signal created: {signal.direction} at {signal.price}")
                    await self._publish_signal(symbol, signal)
                    self.performance_metrics.signals_generated += 1
                    self.logger.info(f"🚀 Signal published successfully!")
                else:
                    self.logger.warning(f"⚠️ Failed to create signal from pattern")
            else:
                self.logger.info(f"❌ No qualifying patterns found")
                
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}", exc_info=True)
            self.performance_metrics.errors_count += 1
            
    async def _create_trading_signal(self, symbol: str, pattern: 'CandlestickPattern', analysis_result) -> Optional['TradingSignal']:
        """Create a TradingSignal from pattern analysis."""
        
        try:
            from decimal import Decimal
            
            self.logger.info(f"🔨 Creating signal for pattern: {pattern.pattern_type}")
            self.logger.info(f"📊 Pattern probabilities: bullish={getattr(pattern, 'bullish_probability', 'N/A')}, bearish={getattr(pattern, 'bearish_probability', 'N/A')}")
            
            # Determine signal direction
            if pattern.bullish_probability > pattern.bearish_probability:
                direction = SignalDirection.BUY
                self.logger.info(f"📈 Signal direction: BUY")
            elif pattern.bearish_probability > pattern.bullish_probability:
                direction = SignalDirection.SELL
                self.logger.info(f"📉 Signal direction: SELL")
            else:
                # For neutral patterns like doji, generate a BUY signal for testing
                direction = SignalDirection.BUY
                self.logger.info(f"⚖️ Neutral pattern detected, generating BUY signal for testing")
            
            # Calculate entry price (would use latest market price in real implementation)
            entry_price = Decimal("50000.0")  # Mock price
            
            # Calculate stop loss and take profit based on pattern
            if direction == SignalDirection.BUY:
                stop_loss = entry_price * Decimal("0.98")  # 2% stop
                take_profit = entry_price * Decimal("1.04")  # 4% target
            else:
                stop_loss = entry_price * Decimal("1.02")  # 2% stop
                take_profit = entry_price * Decimal("0.96")  # 4% target
            
            # Create signal ID
            signal_id = f"candlestick_{symbol}_{pattern.pattern_type.value}_{int(datetime.now(timezone.utc).timestamp())}"
            
            # Create TradingSignal
            signal = TradingSignal(
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                confidence=pattern.confidence,
                strength=pattern.confidence / Decimal("100"),  # Convert to 0-1 scale
                price=entry_price,
                signal_type=SignalType.PATTERN,
                timeframe=pattern.timeframe,
                timestamp=datetime.now(timezone.utc),
                source="candlestick_strategy",
                reason=f"{pattern.pattern_type.value} pattern detected with {pattern.confidence}% confidence"
            )
            
            self.logger.info(f"✅ Signal created successfully: {signal.signal_id}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating trading signal: {e}", exc_info=True)
            return None
        
    async def _publish_signal(self, symbol: str, signal):
        """Publish a trading signal via message bus."""
        
        try:
            # Generate narrative for the signal
            narrative = await self._generate_signal_narrative(signal)
            
            # Create signal payload for message bus
            signal_payload = TradingSignalPayload(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                direction=signal.direction.value,
                confidence=float(signal.confidence) / 100.0,  # Convert percentage to 0-1 range
                strength=float(signal.strength),
                timeframe=signal.timeframe.value,
                strategy="candlestick_strategy",
                reasoning=signal.reason,
                metadata={
                    "signal_id": signal.signal_id,
                    "price": float(signal.price),
                    "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
                    "take_profit": float(signal.target_price) if signal.target_price else None,
                    "expiry_time": signal.expiry.isoformat() if signal.expiry else None,
                    "narrative": narrative
                },
                timestamp=signal.timestamp
            )
            
            # Publish via message bus
            if self._message_bus:
                result = await self._message_bus.publish(
                    topic=f"signals.{symbol}",
                    message_type=MessageType.SIGNAL_GENERATED,
                    payload=signal_payload,
                    sender=self.name,
                    priority=MessagePriority.HIGH
                )
                
                if result:
                    self.logger.info(f"✅ Successfully published signal to topic: signals.{symbol}")
                else:
                    self.logger.warning(f"❌ Failed to publish signal to topic: signals.{symbol}")
            else:
                self.logger.error("❌ No message bus available for signal publishing")
            
            # Track active signals
            if symbol not in self.active_signals:
                self.active_signals[symbol] = []
            self.active_signals[symbol].append(signal)
            
            # Maintain signal limit per symbol
            max_signals = self.config.max_signals_per_symbol
            if len(self.active_signals[symbol]) > max_signals:
                self.active_signals[symbol] = self.active_signals[symbol][-max_signals:]
                
            self.performance_metrics.last_signal_time = datetime.now(timezone.utc)
            
            self.logger.info(f"Published trading signal for {symbol}: {signal.signal_id}")
            
        except Exception as e:
            self.logger.error(f"Error publishing signal for {symbol}: {e}")
            self.performance_metrics.errors_count += 1
            
    async def _generate_signal_narrative(self, signal) -> Dict[str, str]:
        """Generate a simple narrative for the signal."""
        
        try:
            # Simple narrative generation (placeholder for full narrative generator)
            executive_summary = f"{signal.direction.value.title()} signal for {signal.symbol} with {signal.confidence}% confidence"
            
            pattern_analysis = f"Candlestick pattern analysis indicates {signal.direction.value} momentum based on {signal.reason}"
            
            risk_assessment = f"Signal strength: {float(signal.strength):.2f}, recommended position size based on risk tolerance"
            
            entry_strategy = f"Enter {signal.direction.value} position around {signal.price}"
            
            exit_strategy = f"Manage position according to risk management rules"
            
            return {
                "executive_summary": executive_summary,
                "pattern_analysis": pattern_analysis,
                "risk_assessment": risk_assessment,
                "entry_strategy": entry_strategy,
                "exit_strategy": exit_strategy
            }
            
        except Exception as e:
            self.logger.error(f"Error generating narrative: {e}")
            return {
                "executive_summary": f"Signal generated for {signal.symbol}",
                "pattern_analysis": "Pattern analysis completed",
                "risk_assessment": "Risk assessment required",
                "entry_strategy": "Entry strategy needed",
                "exit_strategy": "Exit strategy needed"
            }
            
    async def _handle_configuration_update(self, message: Message):
        """Handle configuration update messages."""
        
        try:
            config_updates = message.payload.get("config_updates", {})
            
            # Update configuration
            for key, value in config_updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"Updated configuration: {key} = {value}")
                    
            # Re-initialize components if needed
            if any(key in config_updates for key in [
                "min_confidence_threshold", "min_pattern_strength", 
                "narrative_style", "signal_expiry_minutes"
            ]):
                self._initialize_strategy_components()
                self.logger.info("Reinitialized strategy components after config update")
                
        except Exception as e:
            self.logger.error(f"Error handling configuration update: {e}")
            
    async def _handle_health_check_request(self, message: Message):
        """Handle health check requests."""
        
        try:
            health_data = {
                "agent_id": self.agent_id,
                "status": self.state.name,
                "performance_metrics": {
                    "signals_generated": self.performance_metrics.signals_generated,
                    "patterns_detected": self.performance_metrics.patterns_detected,
                    "high_confidence_signals": self.performance_metrics.high_confidence_signals,
                    "processing_latency_ms": self.performance_metrics.processing_latency_ms,
                    "data_messages_received": self.performance_metrics.data_messages_received,
                    "errors_count": self.performance_metrics.errors_count,
                    "average_confidence": self.performance_metrics.average_confidence
                },
                "configuration": {
                    "symbols": self.config.symbols,
                    "timeframes": self.config.timeframes,
                    "min_confidence_threshold": self.config.min_confidence_threshold
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Respond to health check
            await self.publish_message(
                message_type=MessageType.AGENT_HEALTH_UPDATE,
                payload=health_data,
                topic=f"health.{self.agent_id}",
                priority=MessagePriority.NORMAL
            )
            
        except Exception as e:
            self.logger.error(f"Error handling health check request: {e}")
            
    async def _cleanup_old_data(self):
        """Clean up old data from buffers."""
        
        try:
            current_time = datetime.now(timezone.utc)
            
            # Clean up market data buffer (keep only recent data)
            for symbol in self.market_data_buffer:
                for timeframe in self.market_data_buffer[symbol]:
                    candles = self.market_data_buffer[symbol][timeframe]
                    
                    # Keep only last 100 candles to save memory
                    if len(candles) > 100:
                        self.market_data_buffer[symbol][timeframe] = candles[-100:]
                        
            # Clean up expired signals
            for symbol in list(self.active_signals.keys()):
                active_signals = []
                for signal in self.active_signals[symbol]:
                    if signal.expiry and signal.expiry > current_time:
                        active_signals.append(signal)
                        
                self.active_signals[symbol] = active_signals
                
                # Remove empty symbol entries
                if not self.active_signals[symbol]:
                    del self.active_signals[symbol]
                    
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}")
            
    async def _update_health_metrics(self):
        """Update health and performance metrics."""
        
        try:
            # Calculate average confidence of active signals
            all_signals = []
            for signals in self.active_signals.values():
                all_signals.extend(signals)
                
            if all_signals:
                total_confidence = sum(float(s.confidence) for s in all_signals)
                self.performance_metrics.average_confidence = total_confidence / len(all_signals)
            else:
                self.performance_metrics.average_confidence = 0.0
                
            # Update uptime
            if hasattr(self, '_start_time'):
                current_time = datetime.now(timezone.utc)
                
                # Handle both datetime and float timestamps
                if isinstance(self._start_time, datetime):
                    self.performance_metrics.uptime_seconds = (
                        current_time - self._start_time
                    ).total_seconds()
                elif isinstance(self._start_time, (int, float)):
                    # Convert float timestamp to uptime seconds
                    self.performance_metrics.uptime_seconds = (
                        current_time.timestamp() - self._start_time
                    )
                else:
                    # Fallback: initialize start time to now
                    self._start_time = current_time
                    self.performance_metrics.uptime_seconds = 0.0
                
        except Exception as e:
            self.logger.error(f"Error updating health metrics: {e}")
            
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics."""
        
        return {
            "agent_info": {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "status": self.state.name,
                "version": self.config.agent_version
            },
            "performance": {
                "signals_generated": self.performance_metrics.signals_generated,
                "patterns_detected": self.performance_metrics.patterns_detected,
                "high_confidence_signals": self.performance_metrics.high_confidence_signals,
                "processing_latency_ms": round(self.performance_metrics.processing_latency_ms, 2),
                "data_messages_received": self.performance_metrics.data_messages_received,
                "errors_count": self.performance_metrics.errors_count,
                "uptime_seconds": round(self.performance_metrics.uptime_seconds, 2),
                "average_confidence": round(self.performance_metrics.average_confidence, 3)
            },
            "configuration": {
                "symbols": self.config.symbols,
                "timeframes": self.config.timeframes,
                "min_confidence_threshold": self.config.min_confidence_threshold,
                "min_pattern_strength": self.config.min_pattern_strength.value,
                "narrative_style": self.config.narrative_style.value
            },
            "active_data": {
                "symbols_tracking": list(self.market_data_buffer.keys()),
                "active_signals_count": sum(len(signals) for signals in self.active_signals.values()),
                "subscribed_topics_count": len(self.subscribed_topics)
            }
        }

    async def save_state(self, file_path: Optional[str] = None) -> bool:
        """Override state saving to handle config serialization properly."""
        try:
            state_data = {
                "agent_id": self.id,
                "agent_type": self.agent_type.value,
                "state": self.state.value,
                "health": self.health.value,
                "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else asdict(self.config),
                "performance_metrics": asdict(self.performance_metrics),
                "last_saved": datetime.now(timezone.utc).isoformat()
            }
            
            if file_path is None:
                file_path = f"data/agents/{self.name}_state.yaml"
            
            # Ensure the directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                yaml.dump(state_data, f, default_flow_style=False)
            
            self.logger.info(f"Agent state saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False

    def set_message_bus(self, message_bus) -> None:
        """Set the message bus for communication."""
        self._message_bus = message_bus
        self.logger.info("Message bus connected to CandlestickStrategyAgent") 