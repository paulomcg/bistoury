"""
Paper Trading Engine Core

Main orchestration engine that coordinates data flow between historical data,
signal generation, and trade execution for mathematical paper trading.
"""

import asyncio
import logging
import signal
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set, Callable, Any
from pathlib import Path
import json

from .config import PaperTradingConfig, TradingMode
from ..database import DatabaseManager
from ..agents.position_manager_agent import PositionManagerAgent
from ..agents.candlestick_strategy_agent import CandlestickStrategyAgent
from ..agents.messaging import MessageBus
from ..agents.registry import AgentRegistry
from ..agents.orchestrator import AgentOrchestrator
from ..signal_manager.signal_manager import SignalManager
from ..models.market_data import CandlestickData, Timeframe
from ..models.signals import TradingSignal, SignalDirection
from ..models.agent_messages import MessageType, MarketDataPayload, TradingSignalPayload


class PaperTradingEngineStatus:
    """Status tracking for paper trading engine"""
    
    def __init__(self):
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        self.signals_processed = 0
        self.trades_executed = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self.components_status: Dict[str, str] = {}


class PaperTradingEngine:
    """
    Main paper trading engine that orchestrates mathematical trading
    without LLM decision layer. Coordinates historical data replay,
    signal generation, and trade execution.
    """
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.status = PaperTradingEngineStatus()
        
        # Core components
        self.db_manager: Optional[DatabaseManager] = None
        self.message_bus: Optional[MessageBus] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.signal_manager: Optional[SignalManager] = None
        
        # Strategy agents
        self.candlestick_agent: Optional[CandlestickStrategyAgent] = None
        self.position_manager: Optional[PositionManagerAgent] = None
        
        # State management
        self.shutdown_event = asyncio.Event()
        self.tasks: Set[asyncio.Task] = set()
        self.signal_handlers_registered = False
        
        # Historical replay state
        self.replay_start_time: Optional[datetime] = None
        self.replay_current_time: Optional[datetime] = None
        self.processed_data_count = 0
        
        # Performance tracking
        self.performance_stats = {
            "signals_received": 0,
            "signals_traded": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "total_pnl": Decimal("0.0"),
            "win_rate": Decimal("0.0")
        }
    
    async def initialize(self) -> None:
        """Initialize all components and prepare for trading"""
        try:
            self.logger.info(f"Initializing Paper Trading Engine: {self.config.session_name}")
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize messaging and orchestration
            await self._initialize_messaging()
            
            # Initialize signal manager
            await self._initialize_signal_manager()
            
            # Initialize trading agents
            await self._initialize_agents()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.logger.info("Paper Trading Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Paper Trading Engine: {e}")
            raise
    
    async def start(self) -> None:
        """Start the paper trading engine"""
        if self.status.is_running:
            self.logger.warning("Paper Trading Engine is already running")
            return
        
        try:
            await self.initialize()
            
            self.status.is_running = True
            self.status.start_time = datetime.now(timezone.utc)
            self.logger.info("Starting Paper Trading Engine...")
            
            # Start orchestrator and all agents
            if self.orchestrator:
                await self.orchestrator.start_all_agents()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Start appropriate trading mode
            if self.config.trading_mode == TradingMode.HISTORICAL:
                await self._start_historical_replay()
            elif self.config.trading_mode == TradingMode.LIVE_PAPER:
                await self._start_live_paper_trading()
            elif self.config.trading_mode == TradingMode.BACKTEST:
                await self._start_backtesting()
            
            self.logger.info("Paper Trading Engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Paper Trading Engine: {e}")
            self.status.errors += 1
            self.status.last_error = str(e)
            raise
    
    async def stop(self) -> None:
        """Stop the paper trading engine gracefully"""
        if not self.status.is_running:
            self.logger.warning("Paper Trading Engine is not running")
            return
        
        try:
            self.logger.info("Stopping Paper Trading Engine...")
            self.shutdown_event.set()
            
            # Cancel all background tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Stop orchestrator and all agents
            if self.orchestrator:
                await self.orchestrator.stop_all_agents()
            
            # Save final state
            await self._save_state()
            
            self.status.is_running = False
            self.logger.info("Paper Trading Engine stopped gracefully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Paper Trading Engine: {e}")
            raise
    
    async def _initialize_database(self) -> None:
        """Initialize database connection"""
        db_path = self.config.database_path or "data/bistoury.db"
        self.db_manager = DatabaseManager(db_path)
        await self.db_manager.initialize()
        self.status.components_status["database"] = "initialized"
        self.logger.info(f"Database initialized: {db_path}")
    
    async def _initialize_messaging(self) -> None:
        """Initialize messaging infrastructure"""
        self.message_bus = MessageBus()
        self.agent_registry = AgentRegistry(self.message_bus)
        self.orchestrator = AgentOrchestrator(self.agent_registry, self.message_bus)
        
        self.status.components_status["messaging"] = "initialized"
        self.logger.info("Messaging infrastructure initialized")
    
    async def _initialize_signal_manager(self) -> None:
        """Initialize signal manager for signal aggregation"""
        # Use bootstrap strategy configuration for mathematical signals
        from ..signal_manager.models import SignalManagerConfiguration
        
        signal_config = SignalManagerConfiguration(
            strategy_weights=self.config.strategy_weights,
            min_confidence=self.config.trading_params.min_confidence,
            signal_ttl_minutes=5,  # 5 minute signal TTL
            enable_narrative_buffer=False  # Disable for Phase 1
        )
        
        self.signal_manager = SignalManager(signal_config)
        
        # Subscribe to trading signals
        if self.message_bus:
            await self.message_bus.subscribe(
                "signals.*", 
                self._handle_trading_signal,
                self
            )
        
        self.status.components_status["signal_manager"] = "initialized"
        self.logger.info("Signal Manager initialized")
    
    async def _initialize_agents(self) -> None:
        """Initialize all trading agents"""
        if not self.agent_registry or not self.message_bus:
            raise RuntimeError("Messaging infrastructure not initialized")
        
        # Initialize Position Manager Agent
        position_config = {
            "initial_balance": float(self.config.risk_params.initial_balance),
            "max_positions": self.config.trading_params.max_concurrent_positions,
            "default_stop_loss": float(self.config.risk_params.default_stop_loss_percent),
            "default_take_profit": float(self.config.risk_params.default_take_profit_percent),
        }
        
        self.position_manager = PositionManagerAgent(
            agent_id="position_manager",
            config=position_config,
            message_bus=self.message_bus
        )
        
        # Subscribe to position updates
        await self.message_bus.subscribe(
            "positions.*",
            self._handle_position_update,
            self
        )
        
        # Initialize Candlestick Strategy Agent if enabled
        if "candlestick_strategy" in self.config.enabled_strategies:
            candlestick_config = {
                "symbols": self.config.historical_config.symbols if self.config.historical_config else ["BTC"],
                "timeframes": ["1m", "5m", "15m"],
                "min_confidence": float(self.config.trading_params.min_confidence),
                "buffer_size": 1000
            }
            
            self.candlestick_agent = CandlestickStrategyAgent(
                agent_id="candlestick_strategy",
                config=candlestick_config,
                message_bus=self.message_bus
            )
        
        # Register agents with orchestrator
        if self.position_manager:
            await self.orchestrator.register_agent(self.position_manager)
        if self.candlestick_agent:
            await self.orchestrator.register_agent(self.candlestick_agent)
        
        self.status.components_status["agents"] = "initialized"
        self.logger.info("Trading agents initialized")
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and reporting tasks"""
        # Performance reporting task
        if self.config.performance_reporting_interval:
            task = asyncio.create_task(self._performance_reporting_loop())
            self.tasks.add(task)
        
        # State saving task
        if self.config.save_state_interval:
            task = asyncio.create_task(self._state_saving_loop())
            self.tasks.add(task)
        
        # Health monitoring task
        task = asyncio.create_task(self._health_monitoring_loop())
        self.tasks.add(task)
        
        self.logger.info("Background tasks started")
    
    async def _start_historical_replay(self) -> None:
        """Start historical data replay mode"""
        if not self.config.historical_config:
            raise ValueError("Historical config required for historical replay")
        
        self.logger.info(f"Starting historical replay: {self.config.historical_config.start_date} to {self.config.historical_config.end_date}")
        
        # Create and start historical replay task
        task = asyncio.create_task(self._historical_replay_loop())
        self.tasks.add(task)
    
    async def _start_live_paper_trading(self) -> None:
        """Start live paper trading mode"""
        # TODO: Implement live paper trading in Task 13.5
        self.logger.info("Live paper trading mode not yet implemented")
        raise NotImplementedError("Live paper trading mode - Task 13.5")
    
    async def _start_backtesting(self) -> None:
        """Start backtesting mode"""
        # TODO: Implement backtesting in Task 13.6
        self.logger.info("Backtesting mode not yet implemented")
        raise NotImplementedError("Backtesting mode - Task 13.6")
    
    async def _historical_replay_loop(self) -> None:
        """Main loop for replaying historical market data"""
        if not self.config.historical_config or not self.db_manager:
            return
        
        config = self.config.historical_config
        self.replay_start_time = config.start_date
        self.replay_current_time = config.start_date
        
        try:
            # Query historical candlestick data
            for symbol in config.symbols:
                for timeframe in config.timeframes:
                    await self._replay_symbol_timeframe(symbol, timeframe)
                    
                    # Check for shutdown
                    if self.shutdown_event.is_set():
                        break
                
                if self.shutdown_event.is_set():
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in historical replay: {e}")
            self.status.errors += 1
            self.status.last_error = str(e)
    
    async def _replay_symbol_timeframe(self, symbol: str, timeframe: Timeframe) -> None:
        """Replay data for specific symbol and timeframe"""
        if not self.db_manager or not self.config.historical_config:
            return
        
        # Convert timeframe to table name
        timeframe_table = f"candles_{timeframe.value}"
        
        # Query data from database
        query = f"""
        SELECT timestamp, open_price, high_price, low_price, close_price, volume, trade_count
        FROM {timeframe_table}
        WHERE symbol = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
        """
        
        async with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                query, 
                (symbol, self.config.historical_config.start_date, self.config.historical_config.end_date)
            )
            
            for row in cursor.fetchall():
                if self.shutdown_event.is_set():
                    break
                
                # Create candlestick data
                candle_data = CandlestickData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.fromisoformat(row[0]),
                    open_price=Decimal(str(row[1])),
                    high_price=Decimal(str(row[2])),
                    low_price=Decimal(str(row[3])),
                    close_price=Decimal(str(row[4])),
                    volume=Decimal(str(row[5])),
                    trade_count=int(row[6])
                )
                
                # Send market data to candlestick strategy agent
                if self.message_bus and self.candlestick_agent:
                    await self.message_bus.publish_message(
                        message_type=MessageType.MARKET_DATA_UPDATE,
                        topic=f"market_data.{symbol}.{timeframe.value}",
                        payload=MarketDataPayload(
                            symbol=symbol,
                            timeframe=timeframe.value,
                            timestamp=candle_data.timestamp,
                            data_type="candlestick",
                            data=candle_data.model_dump()
                        ),
                        sender="paper_trading_engine"
                    )
                
                self.processed_data_count += 1
                self.replay_current_time = candle_data.timestamp
                
                # Simulate replay speed (optional throttling)
                if self.config.historical_config.replay_speed < 100.0:
                    await asyncio.sleep(0.01 / self.config.historical_config.replay_speed)
    
    async def _handle_trading_signal(self, message) -> None:
        """Handle incoming trading signals from strategy agents"""
        try:
            if message.message_type != MessageType.SIGNAL_GENERATED:
                return
            
            payload = message.payload
            self.status.signals_processed += 1
            self.performance_stats["signals_received"] += 1
            
            # Apply signal filtering
            if not self.config.should_trade_signal(
                Decimal(str(payload.confidence)), 
                getattr(payload, 'quality_grade', 'C')
            ):
                self.logger.debug(f"Signal filtered out: confidence={payload.confidence}")
                return
            
            # Calculate position size
            base_size = self.config.trading_params.base_position_size
            position_size = self.config.calculate_position_size(
                base_size, 
                Decimal(str(payload.confidence))
            )
            
            # Send trading signal to position manager
            if self.message_bus and self.position_manager:
                await self.message_bus.publish_message(
                    message_type=MessageType.TRADE_SIGNAL,
                    topic=f"trades.{payload.symbol}",
                    payload=TradingSignalPayload(
                        signal_id=getattr(payload, 'signal_id', f"signal_{self.status.signals_processed}"),
                        symbol=payload.symbol,
                        direction=payload.direction,
                        confidence=payload.confidence,
                        entry_price=getattr(payload, 'entry_price', None),
                        stop_loss=getattr(payload, 'stop_loss', None),
                        take_profit=getattr(payload, 'take_profit', None),
                        position_size=float(position_size),
                        strategy=payload.strategy,
                        reasoning=getattr(payload, 'reasoning', "Mathematical signal processing"),
                        timestamp=payload.timestamp
                    ),
                    sender="paper_trading_engine"
                )
            
            self.performance_stats["signals_traded"] += 1
            self.logger.info(f"Processed trading signal: {payload.symbol} {payload.direction} confidence={payload.confidence}")
            
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {e}")
            self.status.errors += 1
            self.status.last_error = str(e)
    
    async def _handle_position_update(self, message) -> None:
        """Handle position updates from position manager"""
        try:
            if message.message_type in [MessageType.POSITION_OPENED, MessageType.POSITION_CLOSED]:
                payload = message.payload
                
                if message.message_type == MessageType.POSITION_OPENED:
                    self.performance_stats["positions_opened"] += 1
                    self.status.trades_executed += 1
                elif message.message_type == MessageType.POSITION_CLOSED:
                    self.performance_stats["positions_closed"] += 1
                    # Update P&L if available
                    if hasattr(payload, 'pnl'):
                        self.performance_stats["total_pnl"] += Decimal(str(payload.pnl))
                
                self.logger.info(f"Position update: {message.message_type.value} - {payload.symbol}")
                
        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")
            self.status.errors += 1
            self.status.last_error = str(e)
    
    async def _performance_reporting_loop(self) -> None:
        """Background task for periodic performance reporting"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.performance_reporting_interval.total_seconds())
                
                if self.status.is_running:
                    await self._report_performance()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance reporting: {e}")
    
    async def _state_saving_loop(self) -> None:
        """Background task for periodic state saving"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.save_state_interval.total_seconds())
                
                if self.status.is_running:
                    await self._save_state()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in state saving: {e}")
    
    async def _health_monitoring_loop(self) -> None:
        """Background task for health monitoring"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Health check every 30 seconds
                
                if self.status.is_running:
                    await self._check_health()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
    
    async def _report_performance(self) -> None:
        """Generate and log performance report"""
        runtime = (datetime.now(timezone.utc) - self.status.start_time).total_seconds() if self.status.start_time else 0
        
        report = {
            "session": self.config.session_name,
            "runtime_seconds": runtime,
            "signals_processed": self.status.signals_processed,
            "trades_executed": self.status.trades_executed,
            "performance": dict(self.performance_stats),
            "replay_progress": {
                "processed_data_points": self.processed_data_count,
                "current_time": self.replay_current_time.isoformat() if self.replay_current_time else None
            }
        }
        
        self.logger.info(f"Performance Report: {json.dumps(report, indent=2, default=str)}")
    
    async def _save_state(self) -> None:
        """Save current engine state"""
        state = {
            "session_name": self.config.session_name,
            "start_time": self.status.start_time.isoformat() if self.status.start_time else None,
            "current_time": datetime.now(timezone.utc).isoformat(),
            "signals_processed": self.status.signals_processed,
            "trades_executed": self.status.trades_executed,
            "performance_stats": {k: float(v) if isinstance(v, Decimal) else v for k, v in self.performance_stats.items()},
            "replay_state": {
                "current_time": self.replay_current_time.isoformat() if self.replay_current_time else None,
                "processed_count": self.processed_data_count
            }
        }
        
        # Save to file (create directory if needed)
        state_dir = Path("data/paper_trading_states")
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = state_dir / f"{self.config.session_name}_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.debug(f"State saved to {state_file}")
    
    async def _check_health(self) -> None:
        """Perform health checks on all components"""
        # TODO: Implement comprehensive health checks
        self.status.current_time = datetime.now(timezone.utc)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        if self.signal_handlers_registered:
            return
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self.signal_handlers_registered = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "is_running": self.status.is_running,
            "start_time": self.status.start_time.isoformat() if self.status.start_time else None,
            "current_time": self.status.current_time.isoformat() if self.status.current_time else None,
            "signals_processed": self.status.signals_processed,
            "trades_executed": self.status.trades_executed,
            "errors": self.status.errors,
            "last_error": self.status.last_error,
            "components_status": self.status.components_status,
            "performance_stats": {k: float(v) if isinstance(v, Decimal) else v for k, v in self.performance_stats.items()},
            "replay_progress": {
                "start_time": self.replay_start_time.isoformat() if self.replay_start_time else None,
                "current_time": self.replay_current_time.isoformat() if self.replay_current_time else None,
                "processed_count": self.processed_data_count
            }
        } 