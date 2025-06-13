"""
Paper trading session implementation.
Orchestrates historical data replay with proper agent management.
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from decimal import Decimal
import time

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
from ..models.agent_registry import (
    AgentRegistration, AgentCapability, AgentCapabilityType, 
    AgentCompatibility, AgentType
)

console = Console()


async def run_historical_paper_trading(
    symbol: str,
    timeframe: str,
    duration: int,
    balance: float,
    speed: float,
    min_confidence: float,
    config: Config,
    logger,
    live_mode: bool = False
) -> None:
    """Run a historical paper trading session using the orchestrator."""
    
    # Configure logging based on mode - VERY EARLY suppression for live mode
    import logging
    root_logger = logging.getLogger()
    original_level = root_logger.level
    
    if live_mode:
        # IMMEDIATE aggressive suppression before any components are created
        root_logger.setLevel(logging.ERROR)
        
        # Immediately suppress all existing loggers 
        for name, logger_obj in logging.Logger.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                logger_obj.setLevel(logging.ERROR)
        
        # Suppress specific known loggers that will be created
        for logger_name in ['bistoury', 'bistoury.cli', 'bistoury.database', 'bistoury.database.connection', 'agent', 'agent.paper_collector', 'agent.paper_strategy', 'agent.paper_position_manager']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
    
    if live_mode:
        # Aggressive logging suppression for clean dashboard
        root_logger.setLevel(logging.ERROR)
        
        # Remove ALL console handlers from root logger and all child loggers
        def remove_console_handlers(logger_obj):
            for handler in logger_obj.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler.stream.name in ['<stdout>', '<stderr>']:
                    logger_obj.removeHandler(handler)
        
        # Remove from root logger
        remove_console_handlers(root_logger)
        
        # Remove from all existing loggers
        for name, logger_obj in logging.Logger.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                logger_obj.setLevel(logging.ERROR)
                remove_console_handlers(logger_obj)
        
        # Specifically suppress the verbose loggers
        for logger_name in ['bistoury', 'bistoury.hyperliquid', 'bistoury.hyperliquid.collector', 'bistoury.hyperliquid.client', 'bistoury.database', 'agent']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            remove_console_handlers(logger)
    else:
        # Normal logging mode - DEBUG for most, INFO only for position manager
        root_logger.setLevel(logging.DEBUG)
        
        # Set most loggers to DEBUG (less verbose)
        for logger_name in ['bistoury', 'bistoury.hyperliquid', 'bistoury.hyperliquid.collector', 'bistoury.hyperliquid.client', 'bistoury.database', 'agent.paper_collector', 'agent.paper_strategy', 'agent']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
        
        # Keep position manager at INFO for useful trading info
        position_logger = logging.getLogger('agent.paper_position_manager')
        position_logger.setLevel(logging.INFO)
    
    # Set up log capture for Rich display (only in live mode) - exact same as collector
    if live_mode:
        from rich.console import Console
        from rich.logging import RichHandler
        import io
        from collections import deque
        import time
        
        # Create log buffer for Rich display - keep more messages and add timestamps
        log_messages = deque(maxlen=50)  # Keep last 50 error messages
        message_timestamps = deque(maxlen=50)  # Track when each message was added
        
        class LogCapture(logging.Handler):
            def emit(self, record):
                try:
                    # Capture WARNING+ messages for live mode
                    if record.levelno >= logging.WARNING:
                        msg = self.format(record)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_messages.append(f"[{timestamp}] {record.levelname}: {msg}")
                        message_timestamps.append(time.time())  # Track when message was added
                except Exception:
                    pass

        # Add log capture handler with appropriate level
        log_capture = LogCapture()
        log_capture.setLevel(logging.WARNING)
        root_logger.addHandler(log_capture)
    
    # Initialize components
    if not live_mode:
        console.print("🔧 Initializing components...")
    
    # Determine date range from available data
    start_date, end_date = await _get_available_date_range(symbol, timeframe)
    if not start_date:
        console.print("[red]❌ No historical data available[/red]")
        return
    
    # Calculate realistic session duration based on data range and speed
    # If duration is the default (60), use all available data instead
    use_all_data = (duration == 60)  # Default CLI value indicates "use all data"
    
    if use_all_data:
        # Calculate time span of available data in seconds
        data_time_span = (end_date - start_date).total_seconds()
        # Convert to replay time based on speed (higher speed = shorter session)
        calculated_duration = int(data_time_span / speed)
        
        # Reasonable bounds: minimum 30 seconds, maximum 10 minutes for replay
        calculated_duration = max(30, min(calculated_duration, 600))
        actual_duration = calculated_duration
        
        if not live_mode:
            console.print(f"📅 Using ALL available data: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
            console.print(f"⏱️  Calculated session duration: {actual_duration}s (data span: {data_time_span:.0f}s @ {speed}x speed)")
    else:
        # Use specified duration
        actual_duration = duration
        if not live_mode:
            console.print(f"📅 Using available data: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
            console.print(f"⏱️  Using specified duration: {actual_duration}s")
    
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
    
    # Apply logging configuration AFTER agent creation
    if live_mode:
        # Re-apply aggressive suppression after agent creation for live mode
        def remove_console_handlers(logger_obj):
            for handler in logger_obj.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler.stream.name in ['<stdout>', '<stderr>']:
                    logger_obj.removeHandler(handler)
        
        # Suppress ALL agent loggers aggressively
        for name, logger_obj in logging.Logger.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                logger_obj.setLevel(logging.ERROR)
                remove_console_handlers(logger_obj)
        
        # Extra suppression for specific agent loggers
        for logger_name in ['agent.paper_collector', 'agent.paper_strategy', 'agent.paper_position_manager', 'agent']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            remove_console_handlers(logger)
    else:
        # Set most loggers to DEBUG (less verbose)
        for logger_name in ['bistoury', 'bistoury.hyperliquid', 'bistoury.hyperliquid.collector', 'bistoury.hyperliquid.client', 'bistoury.database', 'agent.paper_collector', 'agent']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
        
        # Strategy agent is very verbose - set to WARNING to only show important messages
        strategy_logger = logging.getLogger('agent.paper_strategy')
        strategy_logger.setLevel(logging.WARNING)
        
        # Keep position manager at INFO for useful trading info
        position_logger = logging.getLogger('agent.paper_position_manager')
        position_logger.setLevel(logging.INFO)
    
    # Register agents with orchestrator
    for agent in agents:
        # Convert BaseAgent to AgentRegistration for registry
        registration = _create_agent_registration(agent)
        await agent_registry.register_agent(registration)
        
        # Set up message bus subscriptions for each agent
        await _setup_agent_subscriptions(agent, message_bus)
        
        # Debug: check if agent is connected to message bus (only in non-live mode)
        if not live_mode:
            if hasattr(agent, '_message_bus'):
                console.print(f"✅ {agent.name} connected to message bus")
            else:
                console.print(f"❌ {agent.name} NOT connected to message bus")
        
        # Connect agent to message bus explicitly
        if hasattr(agent, 'set_message_bus'):
            agent.set_message_bus(message_bus)
            if not live_mode:
                console.print(f"🔗 {agent.name} message bus connection set explicitly")
    
    if not live_mode:
        console.print("✅ Components initialized")
    
    # Start the session
    if not live_mode:
        console.print("🚀 Starting paper trading session...")
    start_time = asyncio.get_event_loop().time()
    
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
        # Start orchestrator and all agents
        await orchestrator.start()
        
        # Explicitly start all agents
        if not live_mode:
            console.print("🔄 Starting individual agents...")
        for agent in agents:
            if not live_mode:
                console.print(f"🚀 Starting agent: {agent.name}")
            try:
                if hasattr(agent, 'start'):
                    start_result = await agent.start()
                    if not live_mode:
                        console.print(f"  ✅ {agent.name} started: {start_result}")
                else:
                    if not live_mode:
                        console.print(f"  ⚠️  {agent.name} has no start method")
            except Exception as e:
                if not live_mode:
                    console.print(f"  ❌ Failed to start {agent.name}: {e}")
        
        # Final aggressive logging suppression for live mode after agent startup
        if live_mode:
            def remove_console_handlers_final(logger_obj):
                for handler in logger_obj.handlers[:]:
                    if isinstance(handler, logging.StreamHandler) and handler.stream.name in ['<stdout>', '<stderr>']:
                        logger_obj.removeHandler(handler)
            
            # Final pass to suppress any logging that was created during agent startup
            for name, logger_obj in logging.Logger.manager.loggerDict.items():
                if isinstance(logger_obj, logging.Logger):
                    logger_obj.setLevel(logging.ERROR)
                    remove_console_handlers_final(logger_obj)
            
            # Set root logger to ERROR one more time
            root_logger.setLevel(logging.ERROR)
        
        if live_mode:
            # Live mode - create rich dashboard
            await _run_live_mode_session(agents, message_bus, actual_duration, start_time, shutdown_event, log_messages, message_timestamps)
        else:
            # Regular mode - normal logging with periodic updates
            if not live_mode:
                console.print("🔍 Setting up monitoring...")
                console.print("[dim]Press Ctrl+C to stop gracefully[/dim]")
            
            # Monitor message bus stats in background
            stats_task = asyncio.create_task(_monitor_message_bus_stats(message_bus, console))
            
            # Monitor for historical replay completion
            replay_complete = asyncio.Event()
            
            async def monitor_replay_completion():
                """Monitor for historical replay completion message."""
                try:
                    while not shutdown_event.is_set():
                        # Check if any collector agent has completed replay
                        collector_agent = None
                        for agent in agents:
                            if hasattr(agent, 'name') and 'collector' in agent.name.lower():
                                collector_agent = agent
                                break
                        
                        if collector_agent and hasattr(collector_agent, 'replay_completed'):
                            if collector_agent.replay_completed:
                                console.print("\n🎬 Historical replay completed")
                                replay_complete.set()
                                break
                        
                        await asyncio.sleep(2.0)
                except Exception as e:
                    console.print(f"[red]Error monitoring replay completion: {e}[/red]")
            
            # Start replay monitoring
            replay_monitor_task = asyncio.create_task(monitor_replay_completion())
            
            # Run for specified duration or until shutdown or replay complete
            while not shutdown_event.is_set() and not replay_complete.is_set():
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                if elapsed >= actual_duration:
                    console.print(f"\n⏰ Session duration ({actual_duration}s) completed")
                    break
                
                await asyncio.sleep(1.0)
            
            # Cancel monitoring tasks
            if 'stats_task' in locals():
                stats_task.cancel()
            if 'replay_monitor_task' in locals():
                replay_monitor_task.cancel()
                
        # Session completed
        elapsed = asyncio.get_event_loop().time() - start_time
        console.print(f"📊 Session Duration: {elapsed:.1f}s")
        
    except KeyboardInterrupt:
        console.print("\n🛑 Session interrupted by user")
        raise
    finally:
        # Restore original logging level
        root_logger.setLevel(original_level)
        
        if not live_mode:
            console.print("🔄 Stopping agents...")
        await orchestrator.stop()
        
        if not live_mode:
            console.print("📊 Generating session report...")
        await _generate_session_report(agents, live_mode=live_mode)
        
        await agent_registry.stop()
        await message_bus.stop()
        
        if not live_mode:
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
        name="paper_collector",
        persist_state=False  # Disable state files for paper trading
    )
    collector_agent._message_bus = message_bus
    agents.append(collector_agent)
    
    # 2. CandlestickStrategyAgent
    # Use the new config manager approach - pass a dict that will be used to override defaults
    strategy_config = {
        "symbols": [symbol],
        "timeframes": [timeframe],
        "min_confidence_threshold": min_confidence,
        "agent_name": "paper_strategy"
    }
    
    # Create strategy agent with custom persistence setting
    strategy_agent = CandlestickStrategyAgent(name="paper_strategy", config=strategy_config)
    # Disable state persistence after creation
    strategy_agent.persist_state = False
    strategy_agent._message_bus = message_bus
    agents.append(strategy_agent)
    
    # 3. PositionManagerAgent - Configure appropriate position sizing based on symbol
    # For BTC: min=0.001 (about $100), max=0.1 (about $10k)
    # For other assets, adjust accordingly
    if symbol.upper() == "BTC":
        min_pos_size = Decimal('0.001')  # ~$100 at $100k BTC
        max_pos_size = Decimal('0.1')    # ~$10k at $100k BTC
    elif symbol.upper() == "ETH":
        min_pos_size = Decimal('0.01')   # ~$35 at $3.5k ETH
        max_pos_size = Decimal('3.0')    # ~$10k at $3.5k ETH
    else:
        min_pos_size = Decimal('1.0')    # Default for other assets
        max_pos_size = Decimal('1000.0')
    
    position_config = PositionManagerConfig(
        initial_balance=balance,
        min_position_size=min_pos_size,
        max_position_size=max_pos_size,
        enable_stop_loss=True,
        enable_take_profit=True,
        taker_fee_rate=Decimal('0.00045'),  # HyperLiquid: 0.045% taker fee
        maker_fee_rate=Decimal('0.00015')   # HyperLiquid: 0.015% maker fee
    )
    
    position_agent = PositionManagerAgent(
        name="paper_position_manager",
        config=position_config
    )
    # Disable state persistence after creation
    position_agent.persist_state = False
    position_agent._message_bus = message_bus
    agents.append(position_agent)
    
    return agents


async def _get_available_date_range(symbol: str, timeframe: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Get the available date range for historical data."""
    try:
        # Use the current database manager instead of forcing switch to production
        # This allows worker processes to use their own database copies
        from src.bistoury.database.connection import get_connection
        conn = get_connection()
        
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


async def _run_live_mode_session(agents: list, message_bus: MessageBus, duration: int, start_time: float, shutdown_event, log_messages, message_timestamps):
    """Run the session in live mode with rich dashboard."""
    import time
    
    # Track initial balance from position manager
    initial_balance = 0
    position_agent = None
    for agent in agents:
        if hasattr(agent, 'name') and 'position' in agent.name.lower():
            position_agent = agent
            if hasattr(agent, 'portfolio') and hasattr(agent.portfolio, 'total_balance'):
                initial_balance = float(agent.portfolio.total_balance)
            elif hasattr(agent, 'config') and hasattr(agent.config, 'initial_balance'):
                initial_balance = float(agent.config.initial_balance)
            break
    
    def create_live_display():
        """Create the live dashboard display."""
        # Calculate runtime
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - start_time
        runtime = int(elapsed)
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Calculate progress
        progress = (elapsed / duration) * 100 if duration > 0 else 0
        
        # Create header
        header = Panel(
            f"[bold blue]💰 Paper Trading Session[/bold blue] | Runtime: [cyan]{hours:02d}:{minutes:02d}:{seconds:02d}[/cyan] | Progress: [green]{progress:.1f}%[/green]",
            style="blue"
        )
        
        # Create agent metrics table
        metrics_table = Table(title="🤖 Agent Metrics", show_header=True, header_style="bold magenta")
        metrics_table.add_column("Agent", style="cyan", width=20)
        metrics_table.add_column("Status", justify="center", style="green", width=12)
        metrics_table.add_column("Messages", justify="right", style="yellow", width=10)
        metrics_table.add_column("Details", style="white", width=30)
        
        for agent in agents:
            if hasattr(agent, 'name'):
                status = "✅ Active"
                
                # Get message count if available
                messages = 0
                if hasattr(agent, 'messages_processed'):
                    messages = agent.messages_processed
                elif hasattr(agent, 'total_messages_received'):
                    messages = agent.total_messages_received
                
                # Agent-specific details
                details = ""
                if 'collector' in agent.name.lower():
                    details = "Publishing historical data"
                elif 'strategy' in agent.name.lower():
                    # Try to get signal count
                    if hasattr(agent, 'signals_generated'):
                        details = f"Signals: {agent.signals_generated}"
                    else:
                        details = "Analyzing patterns"
                elif 'position' in agent.name.lower():
                    # Try to get balance and trade info
                    if hasattr(agent, 'portfolio') and hasattr(agent.portfolio, 'total_balance'):
                        balance = float(agent.portfolio.total_balance)
                        pnl = balance - initial_balance
                        pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0
                        details = f"Balance: ${balance:,.2f} | P&L: {pnl:+.2f} ({pnl_pct:+.1f}%)"
                    elif hasattr(agent, 'total_trades'):
                        details = f"Trades: {agent.total_trades}"
                    else:
                        details = "Managing positions"
                
                metrics_table.add_row(
                    agent.name,
                    status,
                    f"{messages:,}",
                    details
                )
        
        # Create session summary table
        summary_table = Table(title="📊 Session Summary", show_header=True, header_style="bold yellow")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        
        summary_table.add_row("⏱️ Runtime", f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        summary_table.add_row("📈 Progress", f"{progress:.1f}%")
        summary_table.add_row("💵 Initial Balance", f"${initial_balance:,.2f}")
        
        # Get current balance from position manager
        current_balance = initial_balance
        total_trades = 0
        if position_agent:
            if hasattr(position_agent, 'portfolio') and hasattr(position_agent.portfolio, 'total_balance'):
                current_balance = float(position_agent.portfolio.total_balance)
            if hasattr(position_agent, 'total_trades'):
                total_trades = position_agent.total_trades
        
        pnl = current_balance - initial_balance
        pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0
        
        summary_table.add_row("💰 Current Balance", f"${current_balance:,.2f}")
        summary_table.add_row("📊 P&L", f"[green]{pnl:+.2f}[/green] ([green]{pnl_pct:+.1f}%[/green])" if pnl >= 0 else f"[red]{pnl:+.2f}[/red] ([red]{pnl_pct:+.1f}%[/red])")
        summary_table.add_row("🔄 Total Trades", f"{total_trades}")
        
        # Get message bus stats
        # Simplified - avoid async call issue
        messages_sent = sum(1 for agent in agents if hasattr(agent, 'messages_processed'))
        messages_delivered = messages_sent
        # messages_sent already calculated above
        messages_delivered = messages_sent  # Simplified
        
        summary_table.add_row("📨 Messages Sent", f"{messages_sent:,}")
        summary_table.add_row("📦 Messages Delivered", f"{messages_delivered:,}")
        
        # Create recent activity panel
        current_time = time.time()
        recent_messages = []
        
        # Keep messages from last 2 minutes
        for i, (msg, msg_time) in enumerate(zip(log_messages, message_timestamps)):
            if current_time - msg_time <= 120:  # 2 minutes
                recent_messages.append(msg)
        
        # Show last 8 messages if we have them
        display_messages = recent_messages[-8:] if len(recent_messages) > 8 else recent_messages
        
        log_content = "\n".join(display_messages) if display_messages else "[dim]No recent warnings or errors[/dim]"
        activity_panel = Panel(
            log_content,
            title="🚨 Recent Activity",
            style="yellow",
            height=8
        )
        
        # Create layout
        from rich.columns import Columns
        from rich import box
        from rich.console import Group
        
        # Arrange tables side by side
        tables_row = Columns([metrics_table, summary_table], equal=True)
        
        # Stack everything vertically
        display_group = Group(
            header,
            tables_row,
            activity_panel,
            Panel("[dim]Press Ctrl+C to stop session[/dim]", box=box.ROUNDED)
        )
        
        return display_group
    
    # Monitor for historical replay completion in live mode
    replay_complete = asyncio.Event()
    
    async def monitor_replay_completion_live():
        """Monitor for historical replay completion in live mode."""
        try:
            while not shutdown_event.is_set() and not replay_complete.is_set():
                # Check if any collector agent has completed replay
                collector_agent = None
                for agent in agents:
                    if hasattr(agent, 'name') and 'collector' in agent.name.lower():
                        collector_agent = agent
                        break
                
                if collector_agent and hasattr(collector_agent, 'replay_completed'):
                    if collector_agent.replay_completed:
                        replay_complete.set()
                        break
                
                await asyncio.sleep(2.0)
        except Exception as e:
            pass  # Silent error handling in live mode
    
    # Start replay monitoring task
    replay_monitor_task = asyncio.create_task(monitor_replay_completion_live())
    
    # Run live display
    with Live(create_live_display(), refresh_per_second=0.5, screen=True) as live:
        while not shutdown_event.is_set() and not replay_complete.is_set():
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - start_time
            
            if elapsed >= duration:
                break
            
            live.update(create_live_display())
            await asyncio.sleep(2)
    
    # Cancel replay monitoring
    if 'replay_monitor_task' in locals():
        replay_monitor_task.cancel()


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


async def _generate_session_report(agents: list, live_mode: bool = False):
    """Generate and display a session report."""
    
    if live_mode:
        # In live mode, show a brief summary without tables
        # Get key metrics
        initial_balance = 0
        current_balance = 0
        total_trades = 0
        signals_generated = 0
        total_messages = 0
        
        for agent in agents:
            if hasattr(agent, 'name'):
                if 'position' in agent.name.lower():
                    if hasattr(agent, 'config') and hasattr(agent.config, 'initial_balance'):
                        initial_balance = float(agent.config.initial_balance)
                    if hasattr(agent, 'portfolio') and hasattr(agent.portfolio, 'total_balance'):
                        current_balance = float(agent.portfolio.total_balance)
                    if hasattr(agent, 'total_trades'):
                        total_trades = agent.total_trades
                elif 'strategy' in agent.name.lower():
                    if hasattr(agent, 'signals_generated'):
                        signals_generated = agent.signals_generated
                
                # Count messages
                if hasattr(agent, 'messages_processed'):
                    total_messages += agent.messages_processed
                elif hasattr(agent, 'total_messages_received'):
                    total_messages += agent.total_messages_received
        
        pnl = current_balance - initial_balance
        pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0
        
        console.print("")
        console.print("📊 [bold]Session Completed[/bold]")
        console.print(f"💰 Final Balance: ${current_balance:,.2f} (was ${initial_balance:,.2f})")
        console.print(f"📈 P&L: {pnl:+.2f} ({pnl_pct:+.1f}%)")
        console.print(f"🔄 Trades: {total_trades} | 📊 Signals: {signals_generated} | 📨 Messages: {total_messages:,}")
        console.print("✅ Paper trading session completed")
        return
    
    # Regular mode - show detailed table
    table = Table(title="📋 Session Report", show_header=True)
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Metrics")
    
    # Also track trading metrics for summary
    initial_balance = 0
    current_balance = 0
    total_trades = 0
    signals_generated = 0
    
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
            
            # Get trading-specific metrics
            if 'position' in agent.name.lower():
                if hasattr(agent, 'config') and hasattr(agent.config, 'initial_balance'):
                    initial_balance = float(agent.config.initial_balance)
                if hasattr(agent, 'portfolio') and hasattr(agent.portfolio, 'total_balance'):
                    current_balance = float(agent.portfolio.total_balance)
                if hasattr(agent, 'total_trades'):
                    total_trades = agent.total_trades
                    metrics += f" | Trades: {total_trades}"
            elif 'strategy' in agent.name.lower():
                if hasattr(agent, 'signals_generated'):
                    signals_generated = agent.signals_generated
                    metrics += f" | Signals: {signals_generated}"
                
            table.add_row(agent.name, status, metrics)
    
    console.print(table)
    
    # Add trading summary for regular mode
    if initial_balance > 0:
        pnl = current_balance - initial_balance
        pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0
        
        summary_table = Table(title="💰 Trading Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("💵 Initial Balance", f"${initial_balance:,.2f}")
        summary_table.add_row("💰 Final Balance", f"${current_balance:,.2f}")
        summary_table.add_row("📈 P&L", f"{pnl:+.2f} ({pnl_pct:+.1f}%)")
        summary_table.add_row("🔄 Total Trades", f"{total_trades}")
        summary_table.add_row("📊 Signals Generated", f"{signals_generated}")
        
        console.print("\n")
        console.print(summary_table)


def _create_agent_registration(agent) -> AgentRegistration:
    """Convert a BaseAgent to an AgentRegistration."""
    
    # Create capabilities from agent metadata
    capabilities = []
    if hasattr(agent, 'metadata') and hasattr(agent.metadata, 'capabilities'):
        for cap_str in agent.metadata.capabilities:
            # Map string capabilities to AgentCapabilityType
            capability_type = _map_capability_string(cap_str)
            if capability_type:
                capabilities.append(AgentCapability(
                    type=capability_type,
                    description=f"{cap_str} capability",
                    version="1.0.0"
                ))
    elif hasattr(agent, 'capabilities'):
        # Handle agents that have capabilities as base.AgentCapability objects
        for cap in agent.capabilities:
            if hasattr(cap, 'name'):  # base.AgentCapability
                capability_type = _map_capability_string(cap.name)
                if capability_type:
                    capabilities.append(AgentCapability(
                        type=capability_type,
                        description=cap.description,
                        version=cap.version
                    ))
            elif isinstance(cap, str):  # String capability
                capability_type = _map_capability_string(cap)
                if capability_type:
                    capabilities.append(AgentCapability(
                        type=capability_type,
                        description=f"{cap} capability",
                        version="1.0.0"
                    ))
    
    # Create compatibility info
    compatibility = AgentCompatibility(
        agent_version="1.0.0",
        framework_version="1.0.0",
        python_version="3.9+"
    )
    
    # Map agent type
    agent_type = AgentType.COLLECTOR
    if hasattr(agent, 'agent_type'):
        agent_type = agent.agent_type
    elif 'strategy' in agent.name.lower():
        agent_type = AgentType.STRATEGY
    elif 'position' in agent.name.lower():
        agent_type = AgentType.TRADER
    
    return AgentRegistration(
        agent_id=agent.agent_id,
        name=agent.name,
        agent_type=agent_type,
        description=getattr(agent.metadata, 'description', f"{agent.name} agent"),
        capabilities=capabilities,
        provided_services=[],
        required_services=[],
        host="localhost",
        compatibility=compatibility,
        configuration=getattr(agent, 'config', {}) if isinstance(getattr(agent, 'config', {}), dict) else {},
        metadata={
            "version": getattr(agent.metadata, 'version', '1.0.0'),
            "dependencies": getattr(agent.metadata, 'dependencies', [])
        }
    )


def _map_capability_string(cap_str: str) -> Optional[AgentCapabilityType]:
    """Map capability strings to AgentCapabilityType enum values."""
    
    capability_mapping = {
        "market_data_collection": AgentCapabilityType.DATA_COLLECTION,
        "real_time_feeds": AgentCapabilityType.DATA_COLLECTION,
        "historical_data": AgentCapabilityType.DATA_COLLECTION,
        "database_storage": AgentCapabilityType.DATA_STORAGE,
        "health_monitoring": AgentCapabilityType.MONITORING,
        "signal_generation": AgentCapabilityType.SIGNAL_GENERATION,
        "pattern_recognition": AgentCapabilityType.PATTERN_RECOGNITION,
        "technical_analysis": AgentCapabilityType.TECHNICAL_ANALYSIS,
        "candlestick_analysis": AgentCapabilityType.TECHNICAL_ANALYSIS,
        "position_management": AgentCapabilityType.POSITION_MANAGEMENT,
        "order_execution": AgentCapabilityType.ORDER_EXECUTION,
        "risk_management": AgentCapabilityType.RISK_MANAGEMENT
    }
    
    return capability_mapping.get(cap_str)


async def _setup_agent_subscriptions(agent, message_bus: MessageBus):
    """Set up message bus subscriptions for an agent."""
    from ..agents.messaging import MessageFilter
    from ..models.agent_messages import MessageType
    
    # Skip if agent doesn't have handle_message method
    if not hasattr(agent, 'handle_message'):
        print(f"⚠️  {agent.name} ({agent.agent_type.value}) - No handle_message method, skipping subscriptions")
        return
    
    print(f"🔗 Setting up subscriptions for {agent.name} ({agent.agent_type.value})")
    
    # Let agents define their own subscription needs
    if hasattr(agent, 'get_subscription_filters'):
        # Agent provides its own subscription filters
        filters = agent.get_subscription_filters()
        print(f"  📋 Agent provided {len(filters)} custom filters")
        for filter_config in filters:
            await message_bus.subscribe(
                agent_id=agent.agent_id,
                filter=filter_config,
                handler=agent.handle_message,
                is_async=True
            )
    else:
        # Fallback: subscribe to all message types for this agent type
        print(f"  🔄 Using default subscriptions for agent type: {agent.agent_type.value}")
        await _setup_default_subscriptions(agent, message_bus)


async def _setup_default_subscriptions(agent, message_bus: MessageBus):
    """Set up default subscriptions based on agent type."""
    from ..agents.messaging import MessageFilter
    from ..models.agent_messages import MessageType
    
    # Subscribe based on agent type, not name matching
    if agent.agent_type.value == "strategy":
        # Strategy agents want all market data
        filter_config = MessageFilter(
            message_types=[MessageType.DATA_MARKET_UPDATE]
        )
        await message_bus.subscribe(
            agent_id=agent.agent_id,
            filter=filter_config,
            handler=agent.handle_message,
            is_async=True
        )
        
    elif agent.agent_type.value in ["trader", "position_manager"]:
        # Trading agents want signals and market data
        filter_config = MessageFilter(
            message_types=[MessageType.SIGNAL_GENERATED, MessageType.DATA_MARKET_UPDATE]
        )
        await message_bus.subscribe(
            agent_id=agent.agent_id,
            filter=filter_config,
            handler=agent.handle_message,
            is_async=True
        )
    
    # All agents get system messages
    system_filter = MessageFilter(
        message_types=[MessageType.SYSTEM_HEALTH_CHECK, MessageType.SYSTEM_CONFIG_UPDATE]
    )
    await message_bus.subscribe(
        agent_id=agent.agent_id,
        filter=system_filter,
        handler=agent.handle_message,
        is_async=True
    )


async def _monitor_message_bus_stats(message_bus: MessageBus, console: Console):
    """Monitor message bus statistics."""
    while True:
        await asyncio.sleep(10)  # Monitor every 10 seconds
        stats = await message_bus.get_stats()
        console.print(f"🔍 Message Bus Stats: {stats}") 