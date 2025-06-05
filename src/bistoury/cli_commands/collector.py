"""
Collector CLI commands for Bistoury.

Provides comprehensive command-line interface for the Enhanced Data Collector,
including start/stop/status/configuration management.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import Config
from ..database import DatabaseManager
from ..hyperliquid.client import HyperLiquidIntegration
from ..hyperliquid.collector import EnhancedDataCollector, CollectorConfig
from ..models.serialization import CompressionLevel
from ..logger import get_logger

console = Console()
logger = get_logger(__name__)

# Global collector instance for signal handling
_collector_instance: Optional[EnhancedDataCollector] = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _collector_instance
    if _collector_instance:
        console.print("\n[yellow]Received shutdown signal, stopping collector gracefully...[/yellow]")
        try:
            # Set the running flag to False to trigger graceful shutdown
            _collector_instance.running = False
            # Note: We don't call sys.exit(0) here as it causes threading issues
            # The main loop will handle the actual shutdown
        except Exception as e:
            console.print(f"[red]Error during shutdown: {e}[/red]")
            # Only exit with error if there was an exception
            sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@click.group()
def collect():
    """Enhanced data collection commands."""
    pass

@collect.command()
@click.option(
    '--symbols', '-s',
    help='Comma-separated list of symbols to collect (e.g., BTC,ETH,SOL)',
    default='BTC,ETH,SOL'
)
@click.option(
    '--intervals', '-i',
    help='Comma-separated list of candlestick intervals',
    default='1m,5m,15m,1h'
)
@click.option(
    '--buffer-size', '-b',
    type=int,
    default=1000,
    help='Buffer size before auto-flush'
)
@click.option(
    '--flush-interval', '-f',
    type=float,
    default=30.0,
    help='Flush interval in seconds'
)
@click.option(
    '--compression',
    type=click.Choice(['none', 'low', 'medium', 'high', 'maximum']),
    default='medium',
    help='Data compression level'
)
@click.option(
    '--orderbook/--no-orderbook',
    default=True,
    help='Enable/disable order book collection'
)
@click.option(
    '--orderbook-symbols',
    help='Symbols for order book collection (default: first 3 symbols)',
    default=None
)
@click.option(
    '--max-subscriptions',
    type=int,
    default=20,
    help='Maximum concurrent WebSocket subscriptions'
)
@click.option(
    '--validation/--no-validation',
    default=True,
    help='Enable/disable data validation'
)
@click.option(
    '--monitoring/--no-monitoring',
    default=True,
    help='Enable/disable performance monitoring'
)
@click.option(
    '--daemon', '-d',
    is_flag=True,
    help='Run in daemon mode (background)'
)
@click.option(
    '--config-preset',
    type=click.Choice(['development', 'production', 'testing', 'minimal']),
    help='Use predefined configuration preset'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show configuration without starting collector'
)
@click.option(
    '--live',
    is_flag=True,
    help='Show live dashboard with real-time statistics and reduced logging'
)
@click.option(
    '--debug',
    is_flag=True,
    help='Enable DEBUG level logging to show detailed processing messages'
)
def start(symbols: str, intervals: str, buffer_size: int, flush_interval: float,
          compression: str, orderbook: bool, orderbook_symbols: Optional[str],
          max_subscriptions: int, validation: bool, monitoring: bool,
          daemon: bool, config_preset: Optional[str], dry_run: bool, live: bool, debug: bool):
    """Start the enhanced data collector."""
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load system configuration
        config = Config.load_from_env()
        console.print("[green]✓[/green] System configuration loaded")
        
        # Create collector configuration
        collector_config = CollectorConfig(
            symbols=set(symbols.split(',')) if symbols else set(),
            intervals=set(intervals.split(',')) if intervals else set(),
            buffer_size=buffer_size,
            flush_interval=flush_interval,
            compression_level=_compression_to_enum(compression),
            enable_validation=validation,
            enable_monitoring=monitoring,
            orderbook_symbols=set(orderbook_symbols.split(',')) if orderbook_symbols else {'BTC', 'ETH', 'SOL'},
            max_concurrent_subscriptions=max_subscriptions
        )
        
        # Apply configuration preset if specified
        if config_preset:
            preset = _get_preset_config(config_preset)
            if preset:
                collector_config.buffer_size = preset['buffer_size']
                collector_config.flush_interval = preset['flush_interval']
                collector_config.compression_level = preset['compression_level']
                collector_config.enable_validation = preset['validation']
                collector_config.enable_monitoring = preset['monitoring']
                
                console.print(f"[blue]ℹ[/blue] Applied '{config_preset}' configuration preset")
        
        # Display configuration
        _display_collector_config(collector_config, dry_run)
        
        if dry_run:
            console.print("[blue]ℹ[/blue] Dry run completed. Use --no-dry-run to start collector.")
            return
        
        # Start the collector
        console.print("\n[blue]🚀 Starting Enhanced Data Collector...[/blue]")
        
        if daemon:
            asyncio.run(_run_collector_daemon(collector_config, config))
        else:
            asyncio.run(_run_collector_interactive(collector_config, config, live_mode=live, debug_mode=debug))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Collector startup cancelled[/yellow]")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start collector: {e}")
        logger.error(f"Collector startup error: {e}")
        if config.database.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@collect.command()
@click.option(
    '--graceful/--force',
    default=True,
    help='Graceful shutdown vs. force stop'
)
@click.option(
    '--timeout',
    type=int,
    default=30,
    help='Timeout for graceful shutdown in seconds'
)
def stop(graceful: bool, timeout: int):
    """Stop the running collector."""
    # This would require implementing a collector service registry
    # For now, we'll provide instructions
    console.print("[yellow]ℹ[/yellow] Collector stop functionality requires service registry.")
    console.print("To stop a running collector:")
    console.print("• Press Ctrl+C in the collector terminal")
    console.print("• Use 'pkill -f bistoury' to force stop")
    console.print("• Use process management tools if running as a service")

@collect.command()
@click.option(
    '--live', '-l',
    is_flag=True,
    help='Show live updating statistics'
)
@click.option(
    '--json',
    is_flag=True,
    help='Output status in JSON format'
)
@click.option(
    '--history',
    type=int,
    help='Show statistics history for last N minutes'
)
def status(live: bool, json: bool, history: Optional[int]):
    """Show collector status and statistics."""
    
    if json:
        # For JSON output, we'd need to query a running collector
        status_data = {
            "status": "unknown",
            "message": "Status querying requires collector service registry",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        console.print(json.dumps(status_data, indent=2))
        return
    
    # For now, show database statistics as a proxy for collector status
    try:
        config = Config.load_from_env()
        db_manager = DatabaseManager(config)
        
        if live:
            _show_live_status(db_manager)
        else:
            _show_static_status(db_manager, history)
            
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get status: {e}")
        sys.exit(1)

@collect.command()
@click.option(
    '--symbol', '-s',
    required=True,
    help='Symbol to collect historical data for'
)
@click.option(
    '--days',
    type=int,
    default=7,
    help='Number of days back to collect'
)
@click.option(
    '--intervals',
    default='1h,4h,1d',
    help='Comma-separated intervals to collect'
)
@click.option(
    '--start-date',
    help='Start date in YYYY-MM-DD format'
)
@click.option(
    '--end-date',
    help='End date in YYYY-MM-DD format'
)
@click.option(
    '--batch-size',
    type=int,
    default=1000,
    help='Batch size for processing'
)
def history(symbol: str, days: int, intervals: str, start_date: Optional[str],
           end_date: Optional[str], batch_size: int):
    """Collect historical data for a symbol."""
    
    try:
        console.print(f"[blue]📊 Collecting historical data for {symbol}[/blue]")
        
        # Parse intervals
        interval_list = [i.strip() for i in intervals.split(',')]
        
        # Parse dates if provided
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        if not start_dt:
            start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        
        if not end_dt:
            end_dt = datetime.now(timezone.utc)
        
        # Run collection
        asyncio.run(_collect_historical_data(
            symbol, interval_list, start_dt, end_dt, batch_size
        ))
        
    except Exception as e:
        console.print(f"[red]✗[/red] Historical collection failed: {e}")
        logger.error(f"Historical collection failed: {e}")
        sys.exit(1)

@collect.command()
@click.option(
    '--wizard', '-w',
    is_flag=True,
    help='Interactive configuration wizard'
)
@click.option(
    '--show',
    is_flag=True,
    help='Show current configuration'
)
@click.option(
    '--preset',
    type=click.Choice(['development', 'production', 'testing', 'minimal']),
    help='Show preset configuration'
)
@click.option(
    '--validate',
    is_flag=True,
    help='Validate current configuration'
)
def config(wizard: bool, show: bool, preset: Optional[str], validate: bool):
    """Manage collector configuration."""
    
    if wizard:
        _run_config_wizard()
    elif show:
        _show_current_config()
    elif preset:
        _show_preset_config(preset)
    elif validate:
        _validate_config()
    else:
        console.print("[yellow]ℹ[/yellow] Use --wizard, --show, --preset, or --validate")

def _display_collector_config(config: CollectorConfig, dry_run: bool = False):
    """Display the current collector configuration."""
    
    action_text = "Would collect" if dry_run else "Will collect"
    
    # Create configuration table
    table = Table(title="📊 Collector Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green") 
    table.add_column("Description")
    
    # Add configuration rows
    table.add_row(
        "Symbols",
        ", ".join(sorted(config.symbols)) if config.symbols else "None",
        f"{action_text} data for these trading pairs"
    )
    
    table.add_row(
        "Intervals", 
        ", ".join(sorted(config.intervals)) if config.intervals else "None",
        f"{action_text} candlestick data at these timeframes"
    )
    
    table.add_row(
        "Buffer Size",
        f"{config.buffer_size:,}",
        "Records to buffer before database flush"
    )
    
    table.add_row(
        "Flush Interval",
        f"{config.flush_interval:.1f}s",
        "Automatic flush frequency"
    )
    
    table.add_row(
        "Compression",
        config.compression_level.name.title(),
        "Data compression level for storage"
    )
    
    table.add_row(
        "Validation",
        "✅ Enabled" if config.enable_validation else "❌ Disabled",
        "Real-time data integrity validation"
    )
    
    table.add_row(
        "Monitoring",
        "✅ Enabled" if config.enable_monitoring else "❌ Disabled", 
        "Performance metrics and health monitoring"
    )
    
    table.add_row(
        "OrderBook Symbols",
        ", ".join(sorted(config.orderbook_symbols)) if config.orderbook_symbols else "None",
        f"{action_text} order book snapshots for these symbols"
    )
    
    table.add_row(
        "Max Subscriptions",
        str(config.max_concurrent_subscriptions),
        "Maximum concurrent WebSocket connections"
    )
    
    console.print(table)

def _compression_to_enum(compression: str) -> CompressionLevel:
    """Convert compression string to enum."""
    compression_map = {
        'none': CompressionLevel.NONE,
        'low': CompressionLevel.LOW,
        'medium': CompressionLevel.MEDIUM,
        'high': CompressionLevel.HIGH,
        'maximum': CompressionLevel.MAXIMUM
    }
    return compression_map[compression]

def _get_preset_config(preset: str) -> Dict[str, Any]:
    """Get preset configuration values."""
    preset_configs = {
        'development': {
            'buffer_size': 100,
            'flush_interval': 10.0,
            'compression_level': CompressionLevel.LOW,
            'validation': True,
            'monitoring': True
        },
        'production': {
            'buffer_size': 2000,
            'flush_interval': 15.0,
            'compression_level': CompressionLevel.MEDIUM,
            'validation': True,
            'monitoring': True
        },
        'testing': {
            'buffer_size': 50,
            'flush_interval': 5.0,
            'compression_level': CompressionLevel.LOW,
            'validation': True,
            'monitoring': False
        },
        'minimal': {
            'buffer_size': 10,
            'flush_interval': 60.0,
            'compression_level': CompressionLevel.NONE,
            'validation': False,
            'monitoring': False
        }
    }
    return preset_configs.get(preset, {})

async def _run_collector_interactive(collector_config: CollectorConfig, config: Config, live_mode: bool = False, debug_mode: bool = False):
    """Run collector in interactive mode with optional live dashboard."""
    global _collector_instance
    
    try:
        # Configure logging based on mode
        root_logger = logging.getLogger()
        original_level = root_logger.level
        
        if debug_mode:
            # Debug mode - show all DEBUG, INFO, WARNING, ERROR
            root_logger.setLevel(logging.DEBUG)
            for logger_name in ['bistoury', 'bistoury.hyperliquid', 'bistoury.hyperliquid.collector', 'bistoury.hyperliquid.client', 'bistoury.database']:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.DEBUG)
        elif live_mode:
            # Suppress verbose logging for clean dashboard
            root_logger.setLevel(logging.ERROR)  # Show errors but suppress INFO/WARNING
            
            # Specifically suppress the verbose collector and client loggers
            for logger_name in ['bistoury', 'bistoury.hyperliquid', 'bistoury.hyperliquid.collector', 'bistoury.hyperliquid.client', 'bistoury.database']:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.ERROR)  # Show errors but suppress INFO/WARNING spam
        else:
            # Normal logging mode - show INFO level and above
            root_logger.setLevel(logging.INFO)
            for logger_name in ['bistoury', 'bistoury.hyperliquid', 'bistoury.hyperliquid.collector', 'bistoury.hyperliquid.client', 'bistoury.database']:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.INFO)
        
        # Set up log capture for Rich display (only in live mode)
        if live_mode:
            from rich.console import Console
            from rich.logging import RichHandler
            import io
            from collections import deque
            
            # Create log buffer for Rich display - keep more messages and add timestamps
            log_messages = deque(maxlen=50)  # Keep last 50 error messages
            message_timestamps = deque(maxlen=50)  # Track when each message was added
            
            class LogCapture(logging.Handler):
                def emit(self, record):
                    try:
                        # Capture based on debug mode - DEBUG+ in debug mode, WARNING+ in normal live mode
                        min_level = logging.DEBUG if debug_mode else logging.WARNING
                        if record.levelno >= min_level:
                            msg = self.format(record)
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_messages.append(f"[{timestamp}] {record.levelname}: {msg}")
                            message_timestamps.append(time.time())  # Track when message was added
                    except Exception:
                        pass

            # Add log capture handler with appropriate level
            log_capture = LogCapture()
            capture_level = logging.DEBUG if debug_mode else logging.WARNING
            log_capture.setLevel(capture_level)
            root_logger.addHandler(log_capture)
        
        # Initialize components (AFTER logging suppression)
        hyperliquid = HyperLiquidIntegration(config)
        db_manager = DatabaseManager(config)
        
        # Create collector
        collector = EnhancedDataCollector(
            hyperliquid=hyperliquid,
            db_manager=db_manager,
            config=collector_config
        )
        
        _collector_instance = collector
        
        # Start collector
        started = await collector.start()
        
        if not started:
            console.print("[red]✗[/red] Failed to start collector")
            return
        
        console.print("[green]✅[/green] Collector started successfully!")
        
        if live_mode:
            console.print("[dim]Press Ctrl+C to stop gracefully[/dim]")
            
            # Create Rich Live display with database counters
            from rich.live import Live
            from rich.table import Table
            from rich.panel import Panel
            from rich.layout import Layout
            
            # Track previous counts for deltas
            prev_counts = {}
            start_time = time.time()
            
            def get_existing_tables():
                """Get list of tables that actually exist in the database."""
                try:
                    result = db_manager.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")
                    return [row[0] for row in result]
                except Exception:
                    return []
            
            def create_display():
                """Create the live display layout."""
                # Calculate runtime
                runtime = int(time.time() - start_time)
                hours, remainder = divmod(runtime, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Get current counts from database
                try:
                    counts = {}
                    existing_tables = get_existing_tables()
                    
                    # Define table mappings with fallbacks
                    table_mappings = {
                        'candles_1m': 'Candles (1m)',
                        'candles_5m': 'Candles (5m)', 
                        'candles_15m': 'Candles (15m)',
                        'candles_1h': 'Candles (1h)',
                        'candles_4h': 'Candles (4h)',
                        'candles_1d': 'Candles (1d)',
                        'trades': 'Trades',
                        'orderbook_snapshots': 'Order Books',
                        'all_mids': 'All Mids',
                        'funding_rates': 'Funding Rates'
                    }
                    
                    # Only query tables that actually exist
                    for table, display_name in table_mappings.items():
                        if table in existing_tables:
                            try:
                                result = db_manager.execute(f"SELECT COUNT(*) FROM {table}")
                                counts[display_name] = result[0][0] if result else 0
                            except Exception:
                                counts[display_name] = 0
                        else:
                            counts[display_name] = 0
                    
                except Exception as e:
                    counts = {'Database': f"Error: {str(e)[:50]}..."}
                
                # Create header
                header = Panel(
                    f"[bold blue]🚀 Bistoury Data Collector[/bold blue] | Runtime: [cyan]{hours:02d}:{minutes:02d}:{seconds:02d}[/cyan]",
                    style="blue"
                )
                
                # Create data table
                data_table = Table(title="📊 Database Activity", show_header=True, header_style="bold magenta")
                data_table.add_column("Data Type", style="cyan", width=15)
                data_table.add_column("Total Records", justify="right", style="green", width=15)
                data_table.add_column("Recent Activity", justify="center", style="yellow", width=15)
                
                for data_type, count in counts.items():
                    if isinstance(count, int):
                        prev_count = prev_counts.get(data_type, 0)
                        delta = count - prev_count
                        
                        # Format delta display
                        if delta > 0:
                            delta_display = f"[green]+{delta:,}[/green]"
                        else:
                            delta_display = "[dim]-[/dim]"
                        
                        data_table.add_row(
                            data_type,
                            f"{count:,}",
                            delta_display
                        )
                    else:
                        data_table.add_row(data_type, str(count), "[red]Error[/red]")
                
                # Update previous counts
                prev_counts.update({k: v for k, v in counts.items() if isinstance(v, int)})
                
                # Create collector stats
                collector_stats = collector.get_stats()
                stats_table = Table(title="⚡ Collector Stats", show_header=True, header_style="bold yellow")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", justify="right", style="green")
                
                stats_table.add_row("Symbols", f"{len(collector_config.symbols)}")
                stats_table.add_row("Intervals", f"{len(collector_config.intervals)}")
                stats_table.add_row("Buffer Size", f"{collector_config.buffer_size:,}")
                stats_table.add_row("Flush Interval", f"{collector_config.flush_interval:.1f}s")
                stats_table.add_row("Errors", f"{collector_stats.get('errors', 0):,}")
                
                # Create error log panel with time-based filtering
                current_time = time.time()
                recent_messages = []
                
                # Keep messages from last 2 minutes OR last 10 messages, whichever is more
                for i, (msg, msg_time) in enumerate(zip(log_messages, message_timestamps)):
                    if current_time - msg_time <= 120:  # 2 minutes
                        recent_messages.append(msg)
                
                # If we have fewer than 10 recent messages, show the last 10 regardless of time
                if len(recent_messages) < 10 and len(log_messages) >= 10:
                    recent_messages = list(log_messages)[-10:]
                elif len(recent_messages) < 10:
                    recent_messages = list(log_messages)
                
                # Show up to 12 messages to fill the panel better
                display_messages = recent_messages[-12:] if len(recent_messages) > 12 else recent_messages
                
                log_content = "\n".join(display_messages) if display_messages else "[dim]No recent activity[/dim]"
                activity_title = "🚨 Recent Activity (DEBUG)" if debug_mode else "🚨 Recent Activity"
                log_panel = Panel(
                    log_content,
                    title=activity_title,
                    style="yellow",  # Changed to yellow since it shows both warnings and errors
                    height=10  # Increased height to show more messages
                )
                
                # Create layout using Group instead of Layout
                from rich.columns import Columns
                from rich import box
                
                # Create the main content
                main_content = Columns([data_table, stats_table], equal=True)
                
                # Stack everything vertically
                from rich.console import Group
                display_group = Group(
                    header,
                    main_content,
                    log_panel,
                    Panel("[dim]Press Ctrl+C to stop collector[/dim]", box=box.ROUNDED)
                )
                
                return display_group
            
            # Run live display
            with Live(create_display(), refresh_per_second=0.5, screen=True) as live:
                while collector.running:
                    live.update(create_display())
                    await asyncio.sleep(2)
        else:
            # Normal mode - just run the collector with regular logging
            if debug_mode:
                console.print("[dim]Running with DEBUG logging enabled. Use --live for dashboard view.[/dim]")
            else:
                console.print("[dim]Running with normal logging output. Use --live for dashboard view.[/dim]")
            console.print("[dim]Press Ctrl+C to stop gracefully[/dim]")
            
            # Just wait for the collector to run
            while collector.running:
                await asyncio.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down collector...[/yellow]")
    finally:
        # Clean up log handler (only if in live mode)
        if live_mode and 'log_capture' in locals():
            root_logger.removeHandler(log_capture)
            
        # Restore original logging level
        if 'original_level' in locals():
            root_logger.setLevel(original_level)
            
        if _collector_instance:
            await _collector_instance.stop()
            console.print("[green]✅[/green] Collector stopped gracefully")

async def _run_collector_daemon(collector_config: CollectorConfig, config: Config):
    """Run collector in daemon mode."""
    global _collector_instance
    
    try:
        # Initialize components
        hyperliquid = HyperLiquidIntegration(config)
        db_manager = DatabaseManager(config)
        
        # Create collector
        collector = EnhancedDataCollector(
            hyperliquid=hyperliquid,
            db_manager=db_manager,
            config=collector_config
        )
        
        _collector_instance = collector
        
        # Start collector
        started = await collector.start()
        
        if not started:
            console.print("[red]✗[/red] Failed to start collector")
            sys.exit(1)
        
        console.print("[green]✅[/green] Collector started in daemon mode")
        logger.info("Collector started in daemon mode")
        
        # Keep running
        while collector.running:
            await asyncio.sleep(60)  # Check every minute
            
            # Log statistics periodically
            stats = collector.get_stats()
            logger.info(f"Collector statistics: {stats}")
            
    except Exception as e:
        logger.error(f"Daemon collector error: {e}")
        console.print(f"[red]✗[/red] Daemon error: {e}")
    finally:
        if _collector_instance:
            await _collector_instance.stop()
            logger.info("Collector daemon stopped")

def _create_stats_table(stats: Dict[str, Any]) -> Table:
    """Create a statistics table from collector stats."""
    
    table = Table(title="📊 Real-time Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Details")
    
    # Add statistics rows
    table.add_row(
        "Trades Collected",
        f"{stats.get('trades_collected', 0):,}",
        "Individual trade executions"
    )
    
    table.add_row(
        "Order Books",
        f"{stats.get('orderbooks_collected', 0):,}",
        "L2 order book snapshots"
    )
    
    table.add_row(
        "Candles",
        f"{stats.get('candles_collected', 0):,}",
        "OHLCV candlesticks"
    )
    
    table.add_row(
        "Funding Rates",
        f"{stats.get('funding_rates_collected', 0):,}",
        "Funding rate updates"
    )
    
    table.add_row(
        "Errors",
        f"{stats.get('errors', 0):,}",
        "Total error count"
    )
    
    table.add_row(
        "Batches Processed",
        f"{stats.get('batches_processed', 0):,}",
        "Database batch operations"
    )
    
    table.add_row(
        "Active Subscriptions",
        f"{stats.get('active_subscriptions', 0):,}",
        "WebSocket subscriptions"
    )
    
    # Buffer information
    buffer_sizes = stats.get('buffer_sizes', {})
    if buffer_sizes:
        total_buffered = sum(buffer_sizes.values())
        table.add_row(
            "Buffer Usage",
            f"{total_buffered:,}",
            f"Records in memory"
        )
    
    # Uptime
    uptime = stats.get('uptime_seconds', 0)
    if uptime > 0:
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        table.add_row(
            "Uptime",
            f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "Time since start"
        )
    
    return table

def _show_live_status(db_manager: DatabaseManager):
    """Show live updating status from database."""
    
    console.print("[blue]📊 Live Database Status (Press Ctrl+C to exit)[/blue]")
    
    # Table configurations with timestamp columns
    table_configs = {
        'trades': 'timestamp',
        'candles_1m': 'timestamp_start',  # Use timestamp_start for candle tables
        'candles_5m': 'timestamp_start',  # Use timestamp_start for candle tables  
        'candles_15m': 'timestamp_start', # Use timestamp_start for candle tables
        'candles_1h': 'timestamp_start',  # Use timestamp_start for candle tables
        'candles_4h': 'timestamp_start',  # Use timestamp_start for candle tables
        'candles_1d': 'timestamp_start',  # Use timestamp_start for candle tables
        'orderbook_snapshots': 'timestamp',
        'funding_rates': 'timestamp'
    }
    
    try:
        while True:
            # Clear screen
            console.clear()
            
            # Show current time
            console.print(f"[bold]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold]\n")
            
            table = Table(title="Database Record Counts", show_header=True)
            table.add_column("Table", style="cyan")
            table.add_column("Records", style="green")
            table.add_column("Latest", style="yellow")
            
            for table_name, timestamp_col in table_configs.items():
                try:
                    count_result = db_manager.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = count_result[0][0] if count_result else 0
                    
                    # Get latest timestamp using the correct column
                    latest_result = db_manager.execute(
                        f"SELECT MAX({timestamp_col}) FROM {table_name} WHERE {timestamp_col} IS NOT NULL"
                    )
                    latest = latest_result[0][0] if latest_result and latest_result[0][0] else "No data"
                    
                    table.add_row(table_name, f"{count:,}", str(latest))
                    
                except Exception as e:
                    # Check if table exists
                    try:
                        db_manager.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                        table.add_row(table_name, "Error", str(e))
                    except:
                        table.add_row(table_name, "N/A", "Table not found")
            
            console.print(table)
            
            # Wait before refresh
            time.sleep(5)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Live status stopped[/yellow]")

def _show_static_status(db_manager: DatabaseManager, history: Optional[int]):
    """Show static status snapshot."""
    
    console.print("[blue]📊 Current Database Status[/blue]")
    
    # First, get all available tables
    try:
        available_tables_result = db_manager.execute("SHOW TABLES")
        available_tables = {row[0] for row in available_tables_result} if available_tables_result else set()
    except Exception as e:
        console.print(f"[red]Error getting table list: {e}[/red]")
        available_tables = set()
    
    # Table configurations with timestamp columns
    table_configs = {
        'trades': 'timestamp',
        'candles_1m': 'timestamp_start',
        'candles_5m': 'timestamp_start',
        'candles_15m': 'timestamp_start',
        'candles_1h': 'timestamp_start',
        'candles_4h': 'timestamp_start',
        'candles_1d': 'timestamp_start',
        'orderbook_snapshots': 'timestamp',
        'funding_rates': 'timestamp'
    }
    
    table = Table(title="Database Record Counts", show_header=True)
    table.add_column("Table", style="cyan")
    table.add_column("Records", style="green")
    table.add_column("Latest", style="yellow")
    
    for table_name, timestamp_col in table_configs.items():
        try:
            # Check if table exists in available tables
            if table_name not in available_tables:
                table.add_row(table_name, "N/A", "Table not found")
                continue
            
            # Get record count
            count_result = db_manager.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = count_result[0][0] if count_result else 0
            
            # Get latest timestamp using the correct column
            latest_result = db_manager.execute(
                f"SELECT MAX({timestamp_col}) FROM {table_name} WHERE {timestamp_col} IS NOT NULL"
            )
            latest = latest_result[0][0] if latest_result and latest_result[0][0] else "No data"
            
            table.add_row(table_name, f"{count:,}", str(latest))
            
        except Exception as e:
            table.add_row(table_name, "Error", str(e)[:50] + "..." if len(str(e)) > 50 else str(e))
    
    console.print(table)

async def _collect_historical_data(symbol: str, intervals: List[str], 
                                 start_date: datetime, end_date: datetime, 
                                 batch_size: int):
    """Collect historical data for a symbol."""
    
    try:
        # Initialize components
        config = Config.load_from_env()
        hyperliquid = HyperLiquidIntegration(config)
        db_manager = DatabaseManager(config)
        
        collector_config = CollectorConfig(
            symbols={symbol},
            intervals=set(intervals),
            buffer_size=batch_size,
            flush_interval=60.0,
            enable_validation=True,
            enable_monitoring=True
        )
        
        collector = EnhancedDataCollector(
            hyperliquid=hyperliquid,
            db_manager=db_manager,
            config=collector_config
        )
        
        # Calculate total days
        total_days = (end_date - start_date).days
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Collecting {total_days} days of data for {symbol}...",
                total=None
            )
            
            # Collect historical data
            results = await collector.collect_enhanced_historical_data(
                symbol=symbol,
                days_back=total_days,
                intervals=intervals
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"[green]✅[/green] Historical data collection completed:")
        for interval, count in results.items():
            console.print(f"  {interval}: {count:,} records")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Historical collection failed: {e}")
        raise

def _run_config_wizard():
    """Run interactive configuration wizard."""
    console.print("[blue]🧙[/blue] Collector Configuration Wizard")
    console.print("This wizard will help you configure the Enhanced Data Collector\n")
    
    # Collect configuration step by step
    symbols = click.prompt(
        "Enter symbols to collect (comma-separated)",
        default="BTC,ETH,SOL",
        type=str
    )
    
    intervals = click.prompt(
        "Enter intervals to collect (comma-separated)",
        default="1m,5m,15m,1h",
        type=str
    )
    
    buffer_size = click.prompt(
        "Buffer size (records before auto-flush)",
        default=1000,
        type=int
    )
    
    flush_interval = click.prompt(
        "Flush interval (seconds)",
        default=30.0,
        type=float
    )
    
    enable_orderbook = click.confirm(
        "Enable order book collection?",
        default=True
    )
    
    orderbook_symbols = None
    if enable_orderbook:
        orderbook_symbols = click.prompt(
            "Order book symbols (comma-separated, empty for auto)",
            default="",
            type=str
        )
    
    enable_validation = click.confirm(
        "Enable data validation?",
        default=True
    )
    
    enable_monitoring = click.confirm(
        "Enable performance monitoring?",
        default=True
    )
    
    # Show generated configuration
    console.print("\n[green]✅[/green] Configuration generated:")
    console.print(f"  Symbols: {symbols}")
    console.print(f"  Intervals: {intervals}")
    console.print(f"  Buffer size: {buffer_size}")
    console.print(f"  Flush interval: {flush_interval}s")
    console.print(f"  Order book: {enable_orderbook}")
    if enable_orderbook and orderbook_symbols:
        console.print(f"  Order book symbols: {orderbook_symbols}")
    console.print(f"  Validation: {enable_validation}")
    console.print(f"  Monitoring: {enable_monitoring}")
    
    # Generate command
    cmd_parts = ["bistoury collect start"]
    cmd_parts.append(f"--symbols {symbols}")
    cmd_parts.append(f"--intervals {intervals}")
    cmd_parts.append(f"--buffer-size {buffer_size}")
    cmd_parts.append(f"--flush-interval {flush_interval}")
    
    if not enable_orderbook:
        cmd_parts.append("--no-orderbook")
    elif orderbook_symbols:
        cmd_parts.append(f"--orderbook-symbols {orderbook_symbols}")
    
    if not enable_validation:
        cmd_parts.append("--no-validation")
    
    if not enable_monitoring:
        cmd_parts.append("--no-monitoring")
    
    command = " ".join(cmd_parts)
    
    console.print(f"\n[blue]💻[/blue] Generated command:")
    console.print(f"[green]{command}[/green]")
    
    if click.confirm("\nRun this configuration now?"):
        console.print("[blue]🚀[/blue] Starting collector with generated configuration...")
        # Would need to parse and execute the command here

def _show_current_config():
    """Show current system configuration relevant to collector."""
    try:
        config = Config.load_from_env()
        
        table = Table(title="Current System Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Database Path", str(config.database.path))
        table.add_row("Log Level", config.logging.level)
        table.add_row("Environment", config.environment)
        table.add_row("Trading Mode", config.trading.mode)
        
        # API configuration
        table.add_row("HyperLiquid Private Key", "✅ Set" if config.api.hyperliquid_private_key else "❌ Not set")
        table.add_row("HyperLiquid Wallet", "✅ Set" if config.api.hyperliquid_wallet_address else "❌ Not set")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load configuration: {e}")

def _show_preset_config(preset: str):
    """Show configuration for a preset."""
    
    presets = {
        'development': {
            'description': 'Optimized for development and testing',
            'symbols': 'BTC,ETH',
            'intervals': '1m,5m',
            'buffer_size': 100,
            'flush_interval': 10.0,
            'compression': 'low',
            'validation': True,
            'monitoring': True,
            'max_subscriptions': 10
        },
        'production': {
            'description': 'High-performance production configuration',
            'symbols': 'BTC,ETH,SOL,AVAX,ARB,MATIC,DOGE',
            'intervals': '1m,5m,15m,1h,4h,1d',
            'buffer_size': 2000,
            'flush_interval': 15.0,
            'compression': 'high',
            'validation': True,
            'monitoring': True,
            'max_subscriptions': 50
        },
        'testing': {
            'description': 'Minimal configuration for testing',
            'symbols': 'BTC',
            'intervals': '1m',
            'buffer_size': 50,
            'flush_interval': 5.0,
            'compression': 'none',
            'validation': True,
            'monitoring': False,
            'max_subscriptions': 5
        },
        'minimal': {
            'description': 'Absolute minimum resource usage',
            'symbols': 'BTC',
            'intervals': '1h',
            'buffer_size': 10,
            'flush_interval': 60.0,
            'compression': 'none',
            'validation': False,
            'monitoring': False,
            'max_subscriptions': 3
        }
    }
    
    if preset not in presets:
        console.print(f"[red]✗[/red] Unknown preset: {preset}")
        return
    
    config = presets[preset]
    
    table = Table(title=f"'{preset}' Configuration Preset", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Description", config['description'])
    table.add_row("Symbols", config['symbols'])
    table.add_row("Intervals", config['intervals'])
    table.add_row("Buffer Size", str(config['buffer_size']))
    table.add_row("Flush Interval", f"{config['flush_interval']}s")
    table.add_row("Compression", config['compression'])
    table.add_row("Validation", "✅ Enabled" if config['validation'] else "❌ Disabled")
    table.add_row("Monitoring", "✅ Enabled" if config['monitoring'] else "❌ Disabled")
    table.add_row("Max Subscriptions", str(config['max_subscriptions']))
    
    console.print(table)
    
    # Show command to use this preset
    console.print(f"\n[blue]💻[/blue] To use this preset:")
    console.print(f"[green]bistoury collect start --config-preset {preset}[/green]")

def _validate_config():
    """Validate current configuration for collector use."""
    
    try:
        config = Config.load_from_env()
        
        errors = []
        warnings = []
        
        # Check database configuration
        if not config.database.path:
            errors.append("Database path not configured")
        
        # Check API configuration
        if not config.api.hyperliquid_private_key and not config.api.hyperliquid_wallet_address:
            warnings.append("No HyperLiquid credentials - read-only mode only")
        
        # Check environment
        if config.environment == "production" and config.debug:
            warnings.append("Debug mode enabled in production environment")
        
        # Display results
        if errors:
            console.print("[red]✗[/red] Configuration validation failed:")
            for error in errors:
                console.print(f"  • {error}")
        else:
            console.print("[green]✅[/green] Configuration validation passed")
        
        if warnings:
            console.print("\n[yellow]⚠[/yellow] Warnings:")
            for warning in warnings:
                console.print(f"  • {warning}")
        
        if not errors and not warnings:
            console.print("[blue]ℹ[/blue] Configuration is optimal for collector use")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration validation error: {e}") 