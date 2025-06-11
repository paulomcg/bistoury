"""
Collector Agent implementation for the Bistoury multi-agent trading system.

This module integrates the EnhancedDataCollector with the BaseAgent framework,
providing comprehensive data collection capabilities within the multi-agent architecture.
"""

import asyncio
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field

from ..hyperliquid.collector import EnhancedDataCollector, CollectorConfig
from ..hyperliquid.client import HyperLiquidIntegration
from ..database import DatabaseManager, get_database_switcher
from ..logger import get_logger
from ..models.agent_messages import (
    Message, MessageType, MessagePriority, MarketDataPayload,
    SystemEventPayload
)
from ..models.market_data import CandlestickData, Timeframe
from .base import BaseAgent, AgentType, AgentState, AgentHealth

logger = get_logger(__name__)


@dataclass
class CollectorAgentConfig:
    """Configuration for the Collector Agent."""
    
    # Data collection settings
    symbols: Set[str] = field(default_factory=lambda: {'BTC', 'ETH', 'SOL'})
    intervals: Set[str] = field(default_factory=lambda: {'1m', '5m', '15m', '1h', '4h', '1d'})
    
    # Performance settings
    buffer_size: int = 1000
    flush_interval: float = 30.0
    max_batch_size: int = 5000
    
    # Monitoring settings
    stats_interval: float = 60.0
    health_check_interval: float = 30.0
    
    # Collection features
    collect_orderbooks: bool = True
    collect_funding_rates: bool = True
    collect_historical_on_start: bool = False
    historical_days: int = 7
    
    # Messaging settings
    publish_data_updates: bool = True
    publish_stats_updates: bool = True
    data_update_interval: float = 5.0
    
    # Historical replay mode (for paper trading)
    historical_replay_mode: bool = False
    replay_start_date: Optional[datetime] = None
    replay_end_date: Optional[datetime] = None
    replay_speed: float = 1.0  # 1.0 = real-time, 100.0 = 100x speed


class CollectorAgent(BaseAgent):
    """
    Collector Agent for the Bistoury trading system.
    
    Integrates the EnhancedDataCollector with the multi-agent framework to provide:
    - Real-time market data collection from HyperLiquid
    - Database storage with optimized models
    - Health monitoring and status reporting
    - Message bus integration for data distribution
    - Agent lifecycle management
    - Configuration management
    """
    
    def __init__(
        self,
        hyperliquid: HyperLiquidIntegration,
        db_manager: DatabaseManager,
        config: Optional[Dict[str, Any]] = None,
        name: str = "collector",
        persist_state: bool = True
    ):
        """
        Initialize the Collector Agent.
        
        Args:
            hyperliquid: HyperLiquid API integration instance
            db_manager: Database manager for data storage
            config: Agent configuration dictionary
            name: Agent name for identification
            persist_state: Whether to persist agent state
        """
        super().__init__(
            name=name,
            agent_type=AgentType.COLLECTOR,
            config=config,
            persist_state=persist_state
        )
        
        # External dependencies
        self.hyperliquid = hyperliquid
        self.db_manager = db_manager
        
        # Parse configuration
        agent_config = config or {}
        collector_config_dict = agent_config.get('collector', {})
        
        # Convert symbols list to set if needed
        if 'symbols' in collector_config_dict and isinstance(collector_config_dict['symbols'], list):
            collector_config_dict['symbols'] = set(collector_config_dict['symbols'])
        
        self.collector_config = CollectorAgentConfig(**collector_config_dict)
        
        # Create collector configuration
        collector_config = CollectorConfig(
            symbols=self.collector_config.symbols,
            intervals=self.collector_config.intervals,
            buffer_size=self.collector_config.buffer_size,
            flush_interval=self.collector_config.flush_interval,
            max_batch_size=self.collector_config.max_batch_size,
            orderbook_symbols=self.collector_config.symbols if self.collector_config.collect_orderbooks else set(),
            enable_validation=True,
            enable_monitoring=True
        )
        
        # Initialize the enhanced data collector
        self.collector = EnhancedDataCollector(
            hyperliquid=self.hyperliquid,
            db_manager=self.db_manager,
            config=collector_config
        )
        
        # Agent-specific state
        self._last_stats_update = datetime.now(timezone.utc)
        self._last_data_publish = datetime.now(timezone.utc)
        self._collection_start_time: Optional[datetime] = None
        self.replay_completed = False  # Track historical replay completion
        self._last_published_candle_ts: Dict[Tuple[str, str], datetime] = {}
        
        # Update metadata
        self.metadata.description = "Real-time market data collection agent"
        self.metadata.version = "1.0.0"
        self.metadata.capabilities = [
            "market_data_collection",
            "real_time_feeds",
            "historical_data",
            "database_storage",
            "health_monitoring"
        ]
        self.metadata.dependencies = ["hyperliquid_api", "database"]
        
        # Only log initialization if not in live mode
        if not os.getenv('BISTOURY_LIVE_MODE'):
            logger.info(f"CollectorAgent '{name}' initialized with {len(self.collector_config.symbols)} symbols")
    
    async def _start(self) -> bool:
        """
        Start the collector agent and begin data collection.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("🚀 Starting CollectorAgent...")
            
            if self.collector_config.historical_replay_mode:
                # Historical replay mode for paper trading
                self.logger.info(f"📺 Starting in historical replay mode: {self.collector_config.replay_speed}x speed")
                self.logger.info(f"📅 Replay date range: {self.collector_config.replay_start_date} to {self.collector_config.replay_end_date}")
                
                # Start historical replay task
                self.create_task(self._historical_replay_loop())
                self.logger.info("📡 Historical replay task created and started")
                
                # Start background tasks (but not the live collector)
                self.create_task(self._monitor_collector_health())
                if self.collector_config.publish_data_updates:
                    self.create_task(self._publish_data_updates())
                if self.collector_config.publish_stats_updates:
                    self.create_task(self._publish_stats_updates())
                
            else:
                # Live collection mode  
                # Start the enhanced data collector
                if not await self.collector.start():
                    self.logger.error("Failed to start EnhancedDataCollector")
                    return False
                
                # Start background tasks
                self.create_task(self._monitor_collector_health())
                self.create_task(self._publish_data_updates())
                self.create_task(self._publish_stats_updates())
                
                # Collect historical data if configured
                if self.collector_config.collect_historical_on_start:
                    self.create_task(self._collect_initial_historical_data())
            
            self._collection_start_time = datetime.now(timezone.utc)
            
            # Send startup notification
            if self._message_bus:
                mode = "historical_replay" if self.collector_config.historical_replay_mode else "live"
                await self._send_system_message(
                    MessageType.AGENT_STARTED,
                    f"CollectorAgent started in {mode} mode with {len(self.collector_config.symbols)} symbols"
                )
            
            self.logger.info("✅ CollectorAgent started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start CollectorAgent: {e}", exc_info=True)
            return False
    
    async def _stop(self) -> None:
        """Stop the collector agent and clean up resources."""
        try:
            self.logger.info("Stopping CollectorAgent...")
            
            if not self.collector_config.historical_replay_mode:
                # Only stop the live collector if we're not in historical mode
                await self.collector.stop()
            
            # Send shutdown notification
            if self._message_bus:
                await self._send_system_message(
                    MessageType.AGENT_STOPPED,
                    "CollectorAgent stopped gracefully"
                )
            
            self.logger.info("CollectorAgent stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping CollectorAgent: {e}")
    
    async def _health_check(self) -> AgentHealth:
        """
        Perform health check and return current health status.
        
        Returns:
            AgentHealth: Current health metrics
        """
        try:
            # Get collector statistics
            collector_stats = self.collector.get_stats()
            
            # Calculate performance metrics
            uptime = self.uptime
            
            # Calculate health score based on various factors
            health_score = 1.0
            
            # Check if collector is running
            if not self.collector.running:
                health_score *= 0.3
            
            # Check for recent errors
            if collector_stats.get('errors', 0) > 0:
                error_rate = collector_stats['errors'] / max(collector_stats.get('batches_processed', 1), 1)
                health_score *= max(0.1, 1.0 - error_rate)
            
            # Check data collection activity
            last_activity = collector_stats.get('last_activity')
            if last_activity:
                try:
                    if isinstance(last_activity, str):
                        last_activity_dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                    else:
                        last_activity_dt = last_activity
                    
                    time_since_activity = (datetime.now(timezone.utc) - last_activity_dt).total_seconds()
                    if time_since_activity > 300:  # 5 minutes
                        health_score *= 0.5
                except (ValueError, TypeError):
                    health_score *= 0.7
            
            # Update health object
            self.health.health_score = health_score
            self.health.uptime_seconds = uptime
            self.health.last_heartbeat = datetime.now(timezone.utc)
            
            # Add collector-specific metrics
            self.health.tasks_completed = collector_stats.get('batches_processed', 0)
            self.health.error_count = collector_stats.get('errors', 0)
            
            return self.health
            
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            self.health.health_score = 0.1
            self.health.last_error = str(e)
            self.health.last_error_time = datetime.now(timezone.utc)
            return self.health
    
    async def _monitor_collector_health(self) -> None:
        """Monitor the underlying collector health and update agent state."""
        while not self._stop_event.is_set():
            try:
                # Perform health check
                await self._health_check()
                
                # Check if collector has crashed - but not in historical replay mode
                # In historical replay mode, the live collector is not started
                if not self.collector_config.historical_replay_mode:
                    if not self.collector.running and self.state == AgentState.RUNNING:
                        self.logger.warning("Collector has stopped unexpectedly")
                        self._set_state(AgentState.ERROR)
                        
                        if self._message_bus:
                            await self._send_system_message(
                                MessageType.AGENT_ERROR,
                                "Collector has stopped unexpectedly"
                            )
                
                # Wait for next health check
                await asyncio.sleep(self.collector_config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.collector_config.health_check_interval)
    
    async def _publish_data_updates(self) -> None:
        """Publish data update messages to the message bus."""
        if not self.collector_config.publish_data_updates or not self._message_bus:
            return
        
        while not self._stop_event.is_set():
            try:
                # Get current collector stats
                stats = self.collector.get_stats()
                
                # Check if there's new data since last publish
                current_time = datetime.now(timezone.utc)
                time_since_last = (current_time - self._last_data_publish).total_seconds()
                
                if time_since_last >= self.collector_config.data_update_interval:
                    # Create market data update message
                    payload = MarketDataPayload(
                        symbol="SYSTEM",
                        timestamp=current_time,
                        source="collector_agent",
                        data={
                            "candles_collected": stats.get('candles_collected', 0),
                            "trades_collected": stats.get('trades_collected', 0),
                            "orderbooks_collected": stats.get('orderbooks_collected', 0),
                            "total_candles_stored": stats.get('total_candles_stored', 0),
                            "active_symbols": list(self.collector_config.symbols)
                        }
                    )
                    
                    message = Message(
                        type=MessageType.DATA_MARKET_UPDATE,
                        sender=self.name,
                        topic="data.collection",
                        priority=MessagePriority.NORMAL,
                        payload=payload
                    )
                    
                    await self._message_bus.send_message(message)
                    self._last_data_publish = current_time
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error publishing data updates: {e}")
                await asyncio.sleep(self.collector_config.data_update_interval)
    
    async def _publish_stats_updates(self) -> None:
        """Publish statistics updates to the message bus."""
        if not self.collector_config.publish_stats_updates or not self._message_bus:
            return
        
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.collector_config.stats_interval)
                
                # Get comprehensive stats
                stats = self.collector.get_stats()
                health = await self._health_check()
                
                # Create system event message
                payload = SystemEventPayload(
                    event_type="stats_update",
                    component="collector_agent",
                    status="running" if self.is_running else "stopped",
                    description=f"CollectorAgent statistics update",
                    metadata={
                        "agent_name": self.name,
                        "health_score": health.health_score,
                        "uptime_seconds": health.uptime_seconds,
                        "collector_stats": stats,
                        "symbols": list(self.collector_config.symbols),
                        "intervals": list(self.collector_config.intervals)
                    }
                )
                
                message = Message(
                    type=MessageType.AGENT_HEALTH_UPDATE,
                    sender=self.name,
                    topic="agent.health",
                    priority=MessagePriority.NORMAL,
                    payload=payload
                )
                
                await self._message_bus.send_message(message)
                
            except Exception as e:
                self.logger.error(f"Error publishing stats updates: {e}")
    
    async def _collect_initial_historical_data(self) -> None:
        """Collect initial historical data for configured symbols."""
        try:
            self.logger.info(f"Starting initial historical data collection for {self.collector_config.historical_days} days")
            
            for symbol in self.collector_config.symbols:
                try:
                    result = await self.collector.collect_enhanced_historical_data(
                        symbol=symbol,
                        days_back=self.collector_config.historical_days,
                        intervals=list(self.collector_config.intervals)
                    )
                    
                    self.logger.info(f"Collected historical data for {symbol}: {result}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to collect historical data for {symbol}: {e}")
            
            if self._message_bus:
                await self._send_system_message(
                    MessageType.DATA_MARKET_UPDATE,
                    f"Initial historical data collection completed for {len(self.collector_config.symbols)} symbols"
                )
            
        except Exception as e:
            self.logger.error(f"Error in initial historical data collection: {e}")
    
    async def _send_system_message(self, message_type: MessageType, description: str) -> None:
        """Send a system message via the message bus."""
        if not self._message_bus:
            return
        
        try:
            payload = SystemEventPayload(
                event_type=message_type.value,
                component="collector_agent",
                status=self.state.value,
                description=description,
                metadata={
                    "agent_name": self.name,
                    "agent_id": self.agent_id,
                    "symbols": list(self.collector_config.symbols)
                }
            )
            
            message = Message(
                type=message_type,
                sender=self.name,
                topic="agent.lifecycle",
                priority=MessagePriority.HIGH if message_type in [
                    MessageType.AGENT_ERROR, MessageType.SYSTEM_SHUTDOWN
                ] else MessagePriority.NORMAL,
                payload=payload
            )
            
            await self._message_bus.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending system message: {e}")
    
    # Public API methods for external control
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the collection configuration."""
        self.collector_config.symbols.add(symbol.upper())
        self.collector.add_symbol(symbol.upper())
        self.logger.info(f"Added symbol {symbol} to collection")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the collection configuration."""
        self.collector_config.symbols.discard(symbol.upper())
        self.collector.remove_symbol(symbol.upper())
        self.logger.info(f"Removed symbol {symbol} from collection")
    
    async def collect_historical_data(
        self, 
        symbol: str, 
        days_back: int = 7,
        intervals: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Collect historical data for a specific symbol.
        
        Args:
            symbol: Trading symbol to collect data for
            days_back: Number of days back to collect
            intervals: List of intervals to collect (defaults to agent config)
            
        Returns:
            Dict with collection results by interval
        """
        intervals = intervals or list(self.collector_config.intervals)
        
        result = await self.collector.collect_enhanced_historical_data(
            symbol=symbol,
            days_back=days_back,
            intervals=intervals
        )
        
        if self._message_bus:
            await self._send_system_message(
                MessageType.DATA_MARKET_UPDATE,
                f"Historical data collection completed for {symbol}"
            )
        
        return result
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        stats = self.collector.get_stats()
        stats.update({
            "agent_name": self.name,
            "agent_state": self.state.value,
            "symbols": list(self.collector_config.symbols),
            "intervals": list(self.collector_config.intervals),
            "collection_uptime": (
                (datetime.now(timezone.utc) - self._collection_start_time).total_seconds()
                if self._collection_start_time else 0
            )
        })
        return stats
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current agent configuration."""
        return {
            "collector": {
                "symbols": list(self.collector_config.symbols),
                "intervals": list(self.collector_config.intervals),
                "buffer_size": self.collector_config.buffer_size,
                "flush_interval": self.collector_config.flush_interval,
                "max_batch_size": self.collector_config.max_batch_size,
                "stats_interval": self.collector_config.stats_interval,
                "health_check_interval": self.collector_config.health_check_interval,
                "collect_orderbooks": self.collector_config.collect_orderbooks,
                "collect_funding_rates": self.collector_config.collect_funding_rates,
                "publish_data_updates": self.collector_config.publish_data_updates,
                "publish_stats_updates": self.collector_config.publish_stats_updates
            }
        }
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """Update agent configuration."""
        collector_updates = config_updates.get('collector', {})
        
        # Update configuration
        for key, value in collector_updates.items():
            if hasattr(self.collector_config, key):
                setattr(self.collector_config, key, value)
                self.logger.info(f"Updated config {key} = {value}")
        
        # Update base config
        self.update_config(config_updates)
    
    def set_message_bus(self, message_bus) -> None:
        """Set the message bus for communication."""
        self._message_bus = message_bus
        self.logger.info("Message bus connected to CollectorAgent")
    
    async def _historical_replay_loop(self) -> None:
        """Replay historical candle data from database at specified speed."""
        try:
            self.logger.info("🎬 Starting historical replay loop")
            
            # Get database manager - switch_to_database returns DatabaseManager directly
            switcher = get_database_switcher()
            db_manager = switcher.switch_to_database('production')  # Returns DatabaseManager instance
            conn = db_manager.get_connection()
            
            self.logger.info(f"Starting historical replay from {self.collector_config.replay_start_date} to {self.collector_config.replay_end_date}")
            
            # Process each symbol and timeframe
            for symbol in self.collector_config.symbols:
                for interval in self.collector_config.intervals:
                    
                    self.logger.info(f"🔍 Processing {symbol} {interval}")
                    
                    # Map interval string to Timeframe enum
                    timeframe_mapping = {
                        "1m": Timeframe.ONE_MINUTE,
                        "5m": Timeframe.FIVE_MINUTES, 
                        "15m": Timeframe.FIFTEEN_MINUTES,
                        "1h": Timeframe.ONE_HOUR,
                        "4h": Timeframe.FOUR_HOURS,
                        "1d": Timeframe.ONE_DAY
                    }
                    
                    if interval not in timeframe_mapping:
                        self.logger.warning(f"⚠️  Unknown interval {interval}, skipping")
                        continue
                        
                    timeframe = timeframe_mapping[interval]
                    
                    # Query historical candle data
                    table_name = f"candles_{interval}"
                    
                    query = f"""
                    SELECT timestamp_start, open_price, high_price, low_price, close_price, volume, trade_count
                    FROM {table_name}
                    WHERE symbol = ? 
                    """
                    
                    params = [symbol]
                    
                    if self.collector_config.replay_start_date:
                        query += " AND timestamp_start >= ?"
                        params.append(self.collector_config.replay_start_date)
                        
                    if self.collector_config.replay_end_date:
                        query += " AND timestamp_start <= ?"
                        params.append(self.collector_config.replay_end_date)
                        
                    query += " ORDER BY timestamp_start ASC"
                    
                    try:
                        cursor = conn.execute(query, params)
                        rows = cursor.fetchall()
                        
                        self.logger.info(f"Found {len(rows)} candles for {symbol} {interval}")
                        
                        for row in rows:
                            if self._stop_event.is_set():
                                self.logger.info("🛑 Stop event set, ending replay")
                                return
                                
                            # Create candlestick data
                            timestamp = row[0]
                            if isinstance(timestamp, str):
                                timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            elif isinstance(timestamp, datetime):
                                timestamp_dt = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
                            else:
                                timestamp_dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                            
                            candle_data = CandlestickData(
                                symbol=symbol,
                                timeframe=timeframe,
                                timestamp=timestamp_dt,
                                open=float(row[1]),
                                high=float(row[2]),
                                low=float(row[3]),
                                close=float(row[4]),
                                volume=float(row[5]),
                                trade_count=int(row[6])
                            )
                            
                            # Publish candle data via message bus
                            await self._publish_candle_data(symbol, interval, candle_data)
                            
                            # Sleep based on replay speed (simulate time between candles)
                            if self.collector_config.replay_speed > 0:
                                # Calculate delay: 15m interval = 900 seconds real time
                                interval_seconds = {
                                    "1m": 60,
                                    "5m": 300, 
                                    "15m": 900,
                                    "1h": 3600,
                                    "4h": 14400,
                                    "1d": 86400
                                }.get(interval, 900)
                                
                                # Apply replay speed (higher speed = shorter delay)
                                delay = interval_seconds / self.collector_config.replay_speed
                                await asyncio.sleep(delay)
                                
                    except Exception as e:
                        self.logger.error(f"Error replaying {symbol} {interval}: {e}")
                        
            self.logger.info("Historical replay completed")
            
            # Mark replay as complete
            self.replay_completed = True
            
            # Signal completion to the session manager
            if self._message_bus:
                await self._send_system_message(
                    MessageType.SYSTEM_HISTORICAL_REPLAY_COMPLETE,
                    f"Historical replay completed for {len(self.collector_config.symbols)} symbols"
                )
            
        except Exception as e:
            self.logger.error(f"Error in historical replay loop: {e}", exc_info=True)
    
    async def _publish_candle_data(self, symbol: str, interval: str, candle_data: CandlestickData) -> None:
        """Publish candle data via message bus."""
        if not self._message_bus:
            self.logger.warning(f"No message bus available for publishing {symbol} {interval} data")
            return
            
        try:
            if (symbol, interval) in self._last_published_candle_ts and self._last_published_candle_ts[(symbol, interval)] == candle_data.timestamp:
                # Duplicate/in-progress candle, skip publishing to avoid unfinished updates
                return
            # Record this timestamp as the latest published for deduplication
            self._last_published_candle_ts[(symbol, interval)] = candle_data.timestamp
            
            # Create market data payload
            payload = MarketDataPayload(
                symbol=symbol,
                price=candle_data.close,
                volume=candle_data.volume,
                timestamp=candle_data.timestamp,
                source="collector_agent_historical",
                data={
                    "timeframe": interval,
                    "data_type": "candlestick",
                    "candle_data": candle_data.model_dump()
                }
            )
            
            # Publish to topic that strategy agents expect
            topic = f"market_data.{symbol}.{interval}"
            
            self.logger.info(f"📤 Publishing message: topic={topic}, symbol={symbol}, price={candle_data.close}")
            
            # Use publish() method instead of send_message()
            result = await self._message_bus.publish(
                topic=topic,
                message_type=MessageType.DATA_MARKET_UPDATE,
                payload=payload,
                sender=self.name,
                priority=MessagePriority.NORMAL
            )
            
            if result:
                self.logger.debug(f"✅ Successfully published {symbol} {interval} data")
            else:
                self.logger.warning(f"❌ Failed to publish {symbol} {interval} data")
            
        except Exception as e:
            self.logger.error(f"Error publishing candle data: {e}", exc_info=True) 