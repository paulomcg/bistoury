"""
Market Data Simulator for Paper Trading Engine

Simple data replay system that feeds historical data to the Collector Agent
as if it came from HyperLiquid API. The Collector Agent then distributes
data to other agents via the existing message bus architecture.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path

from .config import HistoricalReplayConfig, TradingMode
from ..database import DatabaseManager, get_database_switcher, get_compatible_query
from ..models.market_data import CandlestickData, Timeframe
from ..models.trades import Trade
from ..models.orderbook import OrderBookSnapshot
from ..logger import get_logger


class MarketDataSimulator:
    """
    Market Data Replay System for Paper Trading
    
    Reads historical data from database and feeds it to the Collector Agent
    as if it came from HyperLiquid API. The Collector Agent then handles
    distribution to other agents via the existing message bus.
    
    Architecture:
    Database → Market Data Simulator → Collector Agent → Message Bus → Other Agents
    """
    
    def __init__(self, database_name: str = "test", config: Optional[HistoricalReplayConfig] = None):
        self.database_name = database_name
        self.config = config or HistoricalReplayConfig()
        self.logger = get_logger(__name__)
        
        # Database connections
        self.db_manager: Optional[DatabaseManager] = None
        self.data_query = None
        
        # Single target: Collector Agent (preserves existing architecture)
        self.collector_agent = None
        
        # Simulation state
        self.is_running = False
        self.current_time: Optional[datetime] = None
        self.replay_speed = 1.0  # 1.0 = real-time, 10.0 = 10x speed
        self.events_replayed = 0
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.symbols_processed: List[str] = []
        
    async def initialize(self) -> None:
        """Initialize database connections and validate data availability"""
        try:
            # Get database switcher and switch to specified database
            switcher = get_database_switcher()
            await switcher.switch_to_database(self.database_name)
            
            # Get database manager and data query
            self.db_manager = switcher.get_current_manager()
            self.data_query = get_compatible_query(self.db_manager, "test_legacy")
            
            # Validate data availability
            symbols = await self.data_query.get_symbols()
            self.logger.info(f"Initialized MarketDataSimulator with {len(symbols)} symbols from {self.database_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MarketDataSimulator: {e}")
            raise
    
    def set_collector_agent(self, collector_agent) -> None:
        """Set the Collector Agent as the single data recipient"""
        self.collector_agent = collector_agent
        self.logger.info("Collector Agent registered as data recipient")
    
    def set_replay_speed(self, speed: float) -> None:
        """Set replay speed (1.0 = real-time, 10.0 = 10x speed)"""
        self.replay_speed = max(0.1, min(speed, 100.0))
        self.logger.info(f"Replay speed set to {self.replay_speed}x")
    
    async def start_simulation(self, symbols: List[str]) -> None:
        """Start replaying historical data to the Collector Agent"""
        if self.is_running:
            return
            
        if not self.collector_agent:
            raise ValueError("Collector Agent must be set before starting simulation")
            
        self.is_running = True
        self.start_time = datetime.now()
        self.events_replayed = 0
        self.symbols_processed = symbols.copy()
        
        self.logger.info(f"Starting data replay for symbols: {symbols} at {self.replay_speed}x speed")
        
        try:
            # Create tasks for each symbol
            tasks = []
            
            for symbol in symbols:
                # Each symbol gets its own replay task
                tasks.append(asyncio.create_task(
                    self._replay_symbol_data(symbol)
                ))
            
            # Wait for all symbols to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.is_running = False
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            self.logger.info(f"Replay completed: {self.events_replayed} events in {duration:.1f}s")
    
    async def stop_simulation(self) -> None:
        """Stop the data replay"""
        self.is_running = False
        self.logger.info("Market data replay stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replay statistics"""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        return {
            "is_running": self.is_running,
            "events_replayed": self.events_replayed,
            "symbols_processed": self.symbols_processed,
            "replay_speed": self.replay_speed,
            "duration_seconds": duration,
            "events_per_second": self.events_replayed / duration if duration > 0 else 0,
            "collector_agent_connected": self.collector_agent is not None
        }
    
    async def _replay_symbol_data(self, symbol: str) -> None:
        """Replay all data types for a single symbol in chronological order"""
        # Get all data for this symbol and sort by timestamp
        all_data = []
        
        # Collect candles for configured timeframes
        for timeframe in [Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES]:
            async for candle in self._get_historical_candles(symbol, timeframe):
                all_data.append(('candle', candle.timestamp, candle))
        
        # Collect trades
        async for trade in self._get_historical_trades(symbol):
            all_data.append(('trade', trade.timestamp, trade))
        
        # Collect orderbooks
        async for orderbook in self._get_historical_orderbooks(symbol):
            all_data.append(('orderbook', orderbook.timestamp, orderbook))
        
        # Sort all data by timestamp
        all_data.sort(key=lambda x: x[1])
        
        self.logger.info(f"Replaying {len(all_data)} data points for {symbol}")
        
        # Replay data in chronological order to Collector Agent only
        last_timestamp = None
        for data_type, timestamp, data in all_data:
            if not self.is_running:
                break
            
            # Calculate delay based on real time differences
            if last_timestamp is not None and self.replay_speed > 0:
                time_diff = (timestamp - last_timestamp).total_seconds()
                # Apply replay speed
                delay = time_diff / self.replay_speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))  # Cap at 1 second for very slow replay
            
            # Feed data to Collector Agent only (it handles distribution)
            try:
                if data_type == 'candle':
                    await self._feed_candle_to_collector(data)
                elif data_type == 'trade':
                    await self._feed_trade_to_collector(data)
                elif data_type == 'orderbook':
                    await self._feed_orderbook_to_collector(data)
                    
                self.events_replayed += 1
                
            except Exception as e:
                self.logger.error(f"Error feeding {data_type} data to Collector Agent: {e}")
            
            last_timestamp = timestamp
            
    async def _feed_candle_to_collector(self, candle: CandlestickData) -> None:
        """Feed candlestick data to Collector Agent (simulates HyperLiquid API)"""
        if hasattr(self.collector_agent, 'enhanced_collector'):
            # Feed to EnhancedDataCollector within CollectorAgent
            collector = self.collector_agent.enhanced_collector
            if hasattr(collector, '_process_candle_data'):
                await collector._process_candle_data(candle)
        elif hasattr(self.collector_agent, '_process_candle_data'):
            # Direct method call
            await self.collector_agent._process_candle_data(candle)
        else:
            self.logger.warning("Collector Agent doesn't have candle processing method")
    
    async def _feed_trade_to_collector(self, trade: Trade) -> None:
        """Feed trade data to Collector Agent (simulates HyperLiquid API)"""
        if hasattr(self.collector_agent, 'enhanced_collector'):
            # Feed to EnhancedDataCollector within CollectorAgent
            collector = self.collector_agent.enhanced_collector
            if hasattr(collector, '_process_trade_data'):
                await collector._process_trade_data(trade)
        elif hasattr(self.collector_agent, '_process_trade_data'):
            # Direct method call
            await self.collector_agent._process_trade_data(trade)
        else:
            self.logger.warning("Collector Agent doesn't have trade processing method")
    
    async def _feed_orderbook_to_collector(self, orderbook: OrderBookSnapshot) -> None:
        """Feed order book data to Collector Agent (simulates HyperLiquid API)"""
        if hasattr(self.collector_agent, 'enhanced_collector'):
            # Feed to EnhancedDataCollector within CollectorAgent
            collector = self.collector_agent.enhanced_collector
            if hasattr(collector, '_process_orderbook_data'):
                await collector._process_orderbook_data(orderbook)
        elif hasattr(self.collector_agent, '_process_orderbook_data'):
            # Direct method call
            await self.collector_agent._process_orderbook_data(orderbook)
        else:
            self.logger.warning("Collector Agent doesn't have orderbook processing method")

    async def _get_historical_candles(self, symbol: str, timeframe: Timeframe) -> AsyncGenerator[CandlestickData, None]:
        """Get historical candlestick data from database"""
        try:
            # Map timeframe to table name
            timeframe_map = {
                Timeframe.ONE_MINUTE: "candles_1m",
                Timeframe.FIVE_MINUTES: "candles_5m", 
                Timeframe.FIFTEEN_MINUTES: "candles_15m"
            }
            
            table_name = timeframe_map.get(timeframe)
            if not table_name:
                return
            
            # Get candles from database
            candles = await self.data_query.get_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_time=self.config.start_date,
                end_time=self.config.end_date
            )
            
            for candle in candles:
                yield candle
                
        except Exception as e:
            self.logger.error(f"Error getting historical candles for {symbol} {timeframe}: {e}")
    
    async def _get_historical_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        """Get historical trade data from database"""
        try:
            trades = await self.data_query.get_trades(
                symbol=symbol,
                start_time=self.config.start_date,
                end_time=self.config.end_date
            )
            
            for trade in trades:
                yield trade
                
        except Exception as e:
            self.logger.error(f"Error getting historical trades for {symbol}: {e}")
    
    async def _get_historical_orderbooks(self, symbol: str) -> AsyncGenerator[OrderBookSnapshot, None]:
        """Get historical order book data from database"""
        try:
            orderbooks = await self.data_query.get_orderbook_snapshots(
                symbol=symbol,
                start_time=self.config.start_date,
                end_time=self.config.end_date
            )
            
            for orderbook in orderbooks:
                yield orderbook
                
        except Exception as e:
            self.logger.error(f"Error getting historical orderbooks for {symbol}: {e}") 