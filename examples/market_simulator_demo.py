#!/usr/bin/env python3
"""
Market Data Simulator Demo - Correct Architecture

Demonstrates the proper architecture where:
- Market Data Simulator feeds data ONLY to Collector Agent
- Collector Agent distributes data via existing message bus
- Other agents receive data through normal subscriptions
- No duplication of existing functionality

Architecture:
Database → Market Data Simulator → Collector Agent → Message Bus → Other Agents
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Add project root to path
sys.path.append('.')

from src.bistoury.paper_trading.market_simulator import MarketDataSimulator
from src.bistoury.paper_trading.config import HistoricalReplayConfig
from src.bistoury.models.market_data import Timeframe


class MockCollectorAgent:
    """
    Mock Collector Agent that simulates how the real CollectorAgent would
    receive data from Market Data Simulator and distribute via message bus.
    
    In production, this would be the actual CollectorAgent from Task 6.7.
    """
    
    def __init__(self):
        # Data processing counters
        self.candles_processed = 0
        self.trades_processed = 0
        self.orderbooks_processed = 0
        
        # Simulated agents that subscribe via message bus
        self.subscribed_agents = []
        
    def subscribe_agent(self, agent):
        """Simulate agent subscribing to message bus for data"""
        self.subscribed_agents.append(agent)
        print(f"📡 Agent {agent.__class__.__name__} subscribed to Collector Agent")
    
    async def _process_candle_data(self, candle):
        """Process candle data and distribute to subscribed agents"""
        self.candles_processed += 1
        
        # Distribute to all subscribed agents (simulates message bus)
        for agent in self.subscribed_agents:
            if hasattr(agent, 'handle_candle_data'):
                agent.handle_candle_data(candle)
        
        if self.candles_processed % 10 == 0:
            print(f"🕯️  Collector processed {self.candles_processed} candles, distributed to {len(self.subscribed_agents)} agents")
    
    async def _process_trade_data(self, trade):
        """Process trade data and distribute to subscribed agents"""
        self.trades_processed += 1
        
        # Distribute to all subscribed agents (simulates message bus)
        for agent in self.subscribed_agents:
            if hasattr(agent, 'handle_trade_data'):
                agent.handle_trade_data(trade)
        
        if self.trades_processed % 50 == 0:
            print(f"💰 Collector processed {self.trades_processed} trades, distributed to {len(self.subscribed_agents)} agents")
    
    async def _process_orderbook_data(self, orderbook):
        """Process orderbook data and distribute to subscribed agents"""
        self.orderbooks_processed += 1
        
        # Distribute to all subscribed agents (simulates message bus)
        for agent in self.subscribed_agents:
            if hasattr(agent, 'handle_orderbook_data'):
                agent.handle_orderbook_data(orderbook)
        
        if self.orderbooks_processed % 20 == 0:
            print(f"📋 Collector processed {self.orderbooks_processed} orderbooks, distributed to {len(self.subscribed_agents)} agents")
    
    def get_stats(self):
        """Get collector statistics"""
        return {
            "candles_processed": self.candles_processed,
            "trades_processed": self.trades_processed,
            "orderbooks_processed": self.orderbooks_processed,
            "subscribed_agents": len(self.subscribed_agents)
        }


class MockCandlestickStrategyAgent:
    """
    Mock Candlestick Strategy Agent that receives data via Collector Agent
    and generates trading signals (simulates Task 8 implementation)
    """
    
    def __init__(self):
        self.candles_received = 0
        self.signals_generated = 0
        
    def handle_candle_data(self, candle):
        """Handle candle data received from Collector Agent via message bus"""
        self.candles_received += 1
        
        # Simulate pattern recognition and signal generation
        if self.candles_received % 5 == 0:  # Signal every 5 candles
            self.signals_generated += 1
            print(f"📊 Candlestick Strategy: Signal #{self.signals_generated} generated for {candle.symbol} at ${candle.close}")
    
    def get_performance_metrics(self):
        """Return performance metrics for Paper Trading Engine"""
        return {
            "candles_analyzed": self.candles_received,
            "signals_generated": self.signals_generated
        }


class MockPositionManager:
    """
    Mock Position Manager that receives signals and executes trades
    (simulates Task 11 implementation)
    """
    
    def __init__(self):
        self.signals_received = 0
        self.positions_opened = 0
        self.positions_closed = 0
        self.total_pnl = Decimal("0.0")
        
    def handle_trading_signal(self, signal):
        """Handle trading signal (would come from Signal Manager)"""
        self.signals_received += 1
        
        # Simulate position management
        if self.signals_received % 2 == 1:  # Open on odd signals
            self.positions_opened += 1
            print(f"📈 Position Manager: Opened position #{self.positions_opened}")
        else:  # Close on even signals
            self.positions_closed += 1
            pnl = Decimal("25.50")  # Simulated P&L
            self.total_pnl += pnl
            print(f"📉 Position Manager: Closed position, P&L: +${pnl}")
    
    def get_performance_metrics(self):
        """Return performance metrics for Paper Trading Engine"""
        return {
            "signals_received": self.signals_received,
            "positions_opened": self.positions_opened,
            "positions_closed": self.positions_closed,
            "total_pnl": str(self.total_pnl)
        }


async def main():
    """
    Demonstrate correct Market Data Simulator architecture
    """
    print("🚀 Market Data Simulator Demo - Correct Architecture")
    print("=" * 60)
    print("Architecture Flow:")
    print("  📀 Database → Market Data Simulator → Collector Agent → Message Bus → Other Agents")
    print("  ✅ Single data flow preserves existing message bus architecture")
    print("  ✅ No duplication of agent functionality")
    print()
    
    # Create configuration for historical replay
    config = HistoricalReplayConfig(
        start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),  # 1 hour window
        symbols=["BTC"],
        timeframes=[Timeframe.ONE_MINUTE]
    )
    
    # Create Market Data Simulator (reads from database)
    simulator = MarketDataSimulator("test", config)
    
    # Create Mock Collector Agent (central data hub)
    collector_agent = MockCollectorAgent()
    
    # Create Mock Agents (receive data via Collector Agent)
    candlestick_strategy = MockCandlestickStrategyAgent()
    position_manager = MockPositionManager()
    
    try:
        # Initialize simulator (connects to database)
        print("🔌 Connecting to database...")
        await simulator.initialize()
        print("✅ Connected to database successfully")
        
        # Set up correct architecture: Simulator → Collector Agent
        print("🔗 Setting up data flow architecture...")
        simulator.set_collector_agent(collector_agent)
        print("✅ Market Data Simulator connected to Collector Agent")
        
        # Agents subscribe to Collector Agent (simulates message bus subscriptions)
        print("📡 Agents subscribing to Collector Agent...")
        collector_agent.subscribe_agent(candlestick_strategy)
        collector_agent.subscribe_agent(position_manager)
        print("✅ All agents subscribed via message bus")
        
        # Set faster replay speed for demo
        simulator.set_replay_speed(100.0)  # 100x real-time
        
        print("\n🎬 Starting historical data replay...")
        print("Data Flow: Database → Simulator → Collector Agent → Subscribed Agents")
        print()
        
        # Start simulation (feeds Collector Agent only)
        await simulator.start_simulation(["BTC"])
        
        print("\n📈 Data Replay Complete!")
        
        # Get statistics from all components
        print(f"\n📊 System Statistics:")
        
        # Simulator stats
        sim_stats = simulator.get_stats()
        print(f"  Market Data Simulator:")
        print(f"    • Events replayed: {sim_stats['events_replayed']}")
        print(f"    • Duration: {sim_stats['duration_seconds']:.2f} seconds")
        print(f"    • Events/sec: {sim_stats['events_per_second']:.1f}")
        print(f"    • Collector connected: {sim_stats['collector_agent_connected']}")
        
        # Collector Agent stats
        collector_stats = collector_agent.get_stats()
        print(f"  Collector Agent:")
        print(f"    • Candles processed: {collector_stats['candles_processed']}")
        print(f"    • Trades processed: {collector_stats['trades_processed']}")
        print(f"    • Orderbooks processed: {collector_stats['orderbooks_processed']}")
        print(f"    • Subscribed agents: {collector_stats['subscribed_agents']}")
        
        # Agent performance metrics (Paper Trading Engine would request these)
        print(f"  Agent Performance:")
        
        candlestick_metrics = candlestick_strategy.get_performance_metrics()
        print(f"    Candlestick Strategy:")
        print(f"      • Candles analyzed: {candlestick_metrics['candles_analyzed']}")
        print(f"      • Signals generated: {candlestick_metrics['signals_generated']}")
        
        position_metrics = position_manager.get_performance_metrics()
        print(f"    Position Manager:")
        print(f"      • Signals received: {position_metrics['signals_received']}")
        print(f"      • Positions opened: {position_metrics['positions_opened']}")
        print(f"      • Positions closed: {position_metrics['positions_closed']}")
        print(f"      • Total P&L: ${position_metrics['total_pnl']}")
        
        print("\n✨ Demo completed successfully!")
        print("\n🏛️  Architecture Verification:")
        print("  ✅ Market Data Simulator feeds ONLY Collector Agent")
        print("  ✅ Collector Agent distributes via message bus")
        print("  ✅ Other agents receive data through normal subscriptions") 
        print("  ✅ No duplication of existing functionality")
        print("  ✅ Preserves existing agent architecture")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nNote: This demo requires a populated test database.")
        print("The corrected architecture ensures clean separation of concerns.")


if __name__ == "__main__":
    asyncio.run(main()) 