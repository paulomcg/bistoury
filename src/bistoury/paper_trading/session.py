"""
Paper trading session implementation.
Orchestrates historical data replay with proper agent management.
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from ..config import Config
from ..database import get_database_switcher
from ..agents.orchestrator import AgentOrchestrator
from ..agents.registry import AgentRegistry
from ..agents.messaging import MessageBus
from ..agents.collector_agent import CollectorAgent, CollectorAgentConfig
from ..agents.candlestick_strategy_agent import CandlestickStrategyAgent, CandlestickStrategyConfig
from ..agents.position_manager_agent import PositionManagerAgent, PositionManagerConfig
from ..models.orchestrator_config import OrchestratorConfig

console = Console()


async def run_historical_paper_trading(
    symbol: str,
    timeframe: str,
    duration: int,
    balance: float,
    speed: float,
    min_confidence: float,
    config: Config,
    logger
) -> None:
    """Run a historical paper trading session using the orchestrator."""
    
    # Initialize components
    console.print("🔧 Initializing components...")
    
    # Determine date range from available data
    start_date, end_date = await _get_available_date_range(symbol, timeframe)
    if not start_date:
        console.print("[red]❌ No historical data available[/red]")
        return
    
    console.print(f"📅 Using available data: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Initialize message bus and registry
    message_bus = MessageBus(enable_persistence=False)
    await message_bus.start()
    
    agent_registry = AgentRegistry(message_bus)
    await agent_registry.start()
    
    # Initialize orchestrator
    orchestrator_config = OrchestratorConfig()
    orchestrator = AgentOrchestrator(
        config=orchestrator_config,
        registry=agent_registry,
        message_bus=message_bus
    )
    
    # Create agents with proper configuration
    agents = await _create_paper_trading_agents(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        speed=speed,
        balance=balance,
        min_confidence=min_confidence,
        message_bus=message_bus
    )
    
    # Register agents with orchestrator
    for agent in agents:
        await agent_registry.register_agent(agent)
    
    console.print("✅ Components initialized")
    
    # Start the session
    console.print("🚀 Starting paper trading session...")
    
    # Set up signal handling for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        console.print("\n🛑 Shutdown signal received...")
        shutdown_event.set()
    
    # Register signal handlers
    if sys.platform != 'win32':
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
    
    try:
        # Start all agents via orchestrator
        await orchestrator.start_all_agents()
        
        # Create live display for session monitoring
        with Live(_create_session_display(agents), refresh_per_second=1) as live:
            # Run for specified duration or until shutdown
            start_time = asyncio.get_event_loop().time()
            
            while not shutdown_event.is_set():
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                if elapsed >= duration:
                    console.print(f"\n⏰ Session duration ({duration}s) completed")
                    break
                
                # Update live display
                live.update(_create_session_display(agents, elapsed, duration))
                
                await asyncio.sleep(1.0)
                
    except KeyboardInterrupt:
        console.print("\n🛑 Session interrupted by user")
        raise
    finally:
        # Clean shutdown
        console.print("🔄 Stopping agents...")
        await orchestrator.stop_all_agents()
        
        console.print("📊 Generating session report...")
        await _generate_session_report(agents)
        
        await agent_registry.stop()
        await message_bus.stop()
        
        console.print("✅ Paper trading session completed")


async def _create_paper_trading_agents(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    speed: float,
    balance: float,
    min_confidence: float,
    message_bus: MessageBus
) -> list:
    """Create and configure agents for paper trading."""
    
    agents = []
    
    # 1. CollectorAgent in historical replay mode
    collector_config = CollectorAgentConfig(
        symbols={symbol},
        intervals={timeframe},
        historical_replay_mode=True,
        replay_start_date=start_date,
        replay_end_date=end_date,
        replay_speed=speed,
        publish_data_updates=True,
        data_update_interval=0.1  # Fast updates for paper trading
    )
    
    collector_agent = CollectorAgent(
        hyperliquid=None,  # Not needed for historical mode
        db_manager=None,   # Will use database switcher
        config={"collector": collector_config.__dict__},
        name="paper_collector"
    )
    collector_agent._message_bus = message_bus
    agents.append(collector_agent)
    
    # 2. CandlestickStrategyAgent
    strategy_config = CandlestickStrategyConfig(
        symbols=[symbol],
        timeframes=[timeframe],
        min_confidence_threshold=min_confidence,
        agent_name="paper_strategy"
    )
    
    strategy_agent = CandlestickStrategyAgent(config=strategy_config)
    strategy_agent._message_bus = message_bus
    agents.append(strategy_agent)
    
    # 3. PositionManagerAgent
    position_config = PositionManagerConfig(
        initial_balance=balance,
        enable_stop_loss=True,
        enable_take_profit=True
    )
    
    position_agent = PositionManagerAgent(
        name="paper_position_manager",
        config=position_config
    )
    position_agent._message_bus = message_bus
    agents.append(position_agent)
    
    return agents


async def _get_available_date_range(symbol: str, timeframe: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Get the available date range for historical data."""
    try:
        # Switch to production database
        switcher = get_database_switcher()
        db_manager = switcher.switch_to_database('production')
        conn = db_manager.get_connection()
        
        # Query date range from candle data
        table_name = f"candles_{timeframe}"
        query = f"""
        SELECT MIN(timestamp_start) as min_date, MAX(timestamp_start) as max_date
        FROM {table_name}
        WHERE symbol = ?
        """
        
        cursor = conn.execute(query, [symbol])
        row = cursor.fetchone()
        
        if row and row[0]:
            min_date = datetime.fromisoformat(str(row[0]).replace('Z', '+00:00'))
            max_date = datetime.fromisoformat(str(row[1]).replace('Z', '+00:00'))
            return min_date, max_date
        
        return None, None
        
    except Exception as e:
        console.print(f"[red]Error getting date range: {e}[/red]")
        return None, None


def _create_session_display(agents: list, elapsed: float = 0, duration: int = 60) -> Panel:
    """Create a live display panel for the session."""
    
    # Create main table
    table = Table(title="📊 Paper Trading Session", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add session metrics
    progress = (elapsed / duration) * 100 if duration > 0 else 0
    table.add_row("⏱️ Progress", f"{elapsed:.1f}s / {duration}s ({progress:.1f}%)")
    
    # Add agent metrics
    for agent in agents:
        if hasattr(agent, 'name'):
            # Get agent-specific metrics
            if 'collector' in agent.name.lower():
                table.add_row(f"📊 {agent.name}", "Collecting data...")
            elif 'strategy' in agent.name.lower():
                table.add_row(f"🧠 {agent.name}", "Analyzing patterns...")
            elif 'position' in agent.name.lower():
                table.add_row(f"💰 {agent.name}", "Managing positions...")
    
    return Panel(table, title="🚀 Live Session Monitor", border_style="green")


async def _generate_session_report(agents: list):
    """Generate and display a session report."""
    
    table = Table(title="📋 Session Report", show_header=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Metrics")
    
    for agent in agents:
        if hasattr(agent, 'name'):
            # Get agent health
            try:
                health = await agent.get_health()
                status = "✅ Healthy" if health.is_healthy() else "⚠️ Issues"
                metrics = f"Messages: {health.messages_processed}"
            except:
                status = "❌ Error"
                metrics = "N/A"
                
            table.add_row(agent.name, status, metrics)
    
    console.print(table) 