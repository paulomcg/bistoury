"""
Command-line interface for Bistoury trading system.
"""

import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .config import Config
from .logger import get_logger
from .database.connection import initialize_database, get_database_manager, shutdown_database
from .cli_commands.collector import collect  # Import the new collector CLI module


@click.group()
@click.version_option(version=__version__, prog_name="bistoury")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@click.pass_context
def main(ctx: click.Context, config: Optional[Path], verbose: bool) -> None:
    """
    Bistoury: LLM-Driven Cryptocurrency Trading System
    
    An autonomous trading system that uses Large Language Models to make
    intelligent trading decisions on the HyperLiquid exchange.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        if config:
            ctx.obj["config"] = Config.load_from_env(str(config))
        else:
            ctx.obj["config"] = Config.load_from_env()
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)
    
    # Set up logging
    if verbose:
        ctx.obj["config"].logging.level = "DEBUG"
    
    ctx.obj["logger"] = get_logger("bistoury.cli", ctx.obj["config"].logging.level)
    
    # Initialize database
    try:
        initialize_database(ctx.obj["config"])
        if verbose:
            click.echo("Database initialized successfully", err=True)
    except Exception as e:
        click.echo(f"Database initialization failed: {e}", err=True)
        sys.exit(1)


# Add the collector command group
main.add_command(collect)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status and configuration."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    click.echo("🤖 Bistoury Trading System Status")
    click.echo("=" * 40)
    
    # Environment info
    click.echo(f"Environment: {config.environment}")
    click.echo(f"Debug Mode: {config.debug}")
    click.echo(f"Trading Mode: {config.trading.mode}")
    
    # Database info
    try:
        db_manager = get_database_manager()
        db_info = db_manager.get_database_info()
        
        click.echo("\n📊 Database Status:")
        click.echo(f"  Path: {db_info.get('database_path', 'Unknown')}")
        click.echo(f"  Tables: {db_info.get('table_count', 'Unknown')}")
        click.echo(f"  Active Connections: {db_info.get('active_connections', 'Unknown')}")
        click.echo(f"  Max Connections: {db_info.get('max_connections', 'Unknown')}")
        
        if "database_size" in db_info:
            click.echo(f"  Size: {db_info['database_size']}")
            
    except Exception as e:
        click.echo(f"\n❌ Database Error: {e}")
    
    # Configuration validation
    click.echo("\n🔧 Configuration:")
    click.echo(f"  Database Path: {config.database.path}")
    click.echo(f"  Log Level: {config.logging.level}")
    
    # API Status
    click.echo("\n🔑 API Keys:")
    api_keys = {
        "HyperLiquid Private Key": bool(config.api.hyperliquid_private_key),
        "HyperLiquid Wallet": bool(config.api.hyperliquid_wallet_address),
        "OpenAI": bool(config.api.openai_api_key),
        "Anthropic": bool(config.api.anthropic_api_key),
    }
    
    for name, configured in api_keys.items():
        status_icon = "✅" if configured else "❌"
        click.echo(f"  {status_icon} {name}: {'Configured' if configured else 'Not configured'}")
    
    # Trading validation
    click.echo(f"\n💰 Trading Configuration:")
    click.echo(f"  Mode: {config.trading.mode}")
    click.echo(f"  Max Position Size: ${config.trading.max_position_size:,.2f}")
    click.echo(f"  Stop Loss: {config.trading.stop_loss_pct}%")
    click.echo(f"  Take Profit: {config.trading.take_profit_pct}%")
    
    if config.trading.mode == "live":
        try:
            config.validate_for_trading()
            click.echo("  ✅ Ready for live trading")
        except ValueError as e:
            click.echo(f"  ❌ Live trading validation failed: {e}")
    
    logger.info("Status command executed")


@main.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["historical", "live"]),
    default="historical",
    help="Paper trading mode: 'historical' uses historical data replay, 'live' connects to real-time data"
)
@click.option(
    "--symbol",
    "-s",
    default="BTC",
    help="Trading symbol to analyze (e.g., BTC, ETH). Determines which asset's price data to load"
)
@click.option(
    "--timeframe",
    "-t",
    type=click.Choice(['1m', '5m', '15m', '1h', '4h', '1d']),
    default="15m",
    help="Candlestick timeframe for technical analysis. Lower timeframes = more signals but more noise"
)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=60,
    help="Session duration in seconds. Default (60s) = auto-calculate based on ALL available historical data. Higher values = longer sessions with fixed duration"
)
@click.option(
    "--balance",
    type=float,
    default=10000.0,
    help="Starting portfolio balance in USD. This is your virtual money for paper trading (no real money involved)"
)
@click.option(
    "--speed",
    type=float,
    default=100.0,
    help="Replay speed multiplier for historical mode. 1.0 = normal speed, 100.0 = 100x faster, 0.5 = half speed. Higher = faster completion"
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.5,
    help="Minimum signal confidence threshold (0.0-1.0). Only signals above this confidence level will trigger trades. Higher = fewer but higher-quality signals"
)
@click.option(
    "--live",
    is_flag=True,
    help="Enable live dashboard mode with real-time P&L updates and reduced logging output for cleaner display"
)
@click.pass_context
def paper_trade(ctx: click.Context, mode: str, symbol: str, timeframe: str, 
                duration: int, balance: float, speed: float, min_confidence: float, live: bool) -> None:
    """Start paper trading (no real money)."""
    
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    if mode == "live":
        click.echo("⚠️  Live paper trading not yet implemented")
        return
    
    # Set environment flag for live mode to signal early suppression
    if live:
        import os
        os.environ['BISTOURY_LIVE_MODE'] = '1'
    
    # Import here to avoid circular imports
    import asyncio
    from .paper_trading.session import run_historical_paper_trading
    
    click.echo(f"📊 Starting paper trading")
    click.echo(f"🎯 Mode: {mode}")
    click.echo(f"💰 Symbol: {symbol}")
    click.echo(f"📈 Timeframe: {timeframe}")
    click.echo(f"💵 Balance: ${balance:,.2f}")
    click.echo(f"⏱️  Duration: {duration}s")
    if mode == "historical":
        click.echo(f"⚡ Speed: {speed}x")
    
    # Only log if not in live mode to avoid noise
    if not live:
        logger.info(f"Paper trading started: mode={mode}, symbol={symbol}, duration={duration}")
    
    try:
        # This will be handled inside the paper trading session function
        
        # Run the paper trading session
        asyncio.run(run_historical_paper_trading(
            symbol=symbol,
            timeframe=timeframe,
            duration=duration,
            balance=balance,
            speed=speed,
            min_confidence=min_confidence,
            config=config,
            logger=logger,
            live_mode=live
        ))
    except KeyboardInterrupt:
        click.echo("\n🛑 Paper trading interrupted by user")
    except Exception as e:
        click.echo(f"\n❌ Paper trading failed: {e}")
        logger.error(f"Paper trading error: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--confirm",
    is_flag=True,
    help="Confirm that you want to trade with real money"
)
@click.option(
    "--risk-limit",
    type=float,
    help="Maximum risk limit in USD"
)
@click.pass_context
def trade(ctx: click.Context, confirm: bool, risk_limit: Optional[float]) -> None:
    """Start live trading with real money (DANGEROUS!)."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    # Safety checks
    if config.trading.mode != "live":
        click.echo("❌ Live trading requires trading.mode = 'live' in configuration", err=True)
        sys.exit(1)
    
    if not confirm:
        click.echo("🚨 DANGER: This will trade with REAL MONEY!")
        click.echo("Use --confirm to acknowledge the risk")
        sys.exit(1)
    
    # Additional confirmation
    click.echo("🚨 FINAL WARNING: You are about to start live trading!")
    click.echo("This system will execute trades using real money.")
    if not click.confirm("Are you absolutely sure you want to continue?"):
        click.echo("Live trading cancelled.")
        sys.exit(0)
    
    try:
        config.validate_for_trading()
    except ValueError as e:
        click.echo(f"❌ Trading validation failed: {e}", err=True)
        sys.exit(1)
    
    click.echo(f"💰 Starting live trading")
    if risk_limit:
        click.echo(f"💵 Risk limit: ${risk_limit:,.2f}")
    
    logger.warning(f"Live trading started with risk limit: {risk_limit}")
    
    # TODO: Implement live trading
    click.echo("⚠️  Live trading not yet implemented")


@main.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize Bistoury for first-time use."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    click.echo("🚀 Initializing Bistoury...")
    
    # Check if already initialized
    try:
        db_manager = get_database_manager()
        click.echo("✅ Database connection established")
        
        # TODO: Add more initialization checks
        click.echo("✅ Bistoury appears to be already initialized")
        click.echo("\nNext steps:")
        click.echo("• Run 'bistoury status' to check system status")
        click.echo("• Run 'bistoury collect start' to begin data collection")
        click.echo("• Run 'bistoury paper-trade' to test trading strategies")
        
    except Exception as e:
        click.echo(f"❌ Initialization failed: {e}", err=True)
        sys.exit(1)
    
    logger.info("Initialization completed")


@main.command()
@click.pass_context
def db_status(ctx: click.Context) -> None:
    """Show detailed database status and statistics."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    try:
        db_manager = get_database_manager()
        db_info = db_manager.get_database_info()
        
        click.echo("🗄️ Database Status")
        click.echo("=" * 30)
        
        click.echo(f"Path: {db_info.get('database_path', 'Unknown')}")
        click.echo(f"Size: {db_info.get('database_size', 'Unknown')}")
        click.echo(f"Active Connections: {db_info.get('active_connections', 'Unknown')}")
        click.echo(f"Max Connections: {db_info.get('max_connections', 'Unknown')}")
        
        # TODO: Add table statistics
        
    except Exception as e:
        click.echo(f"❌ Database status check failed: {e}", err=True)
        sys.exit(1)
    
    logger.info("Database status checked")


def shutdown():
    """Clean shutdown of the CLI."""
    try:
        shutdown_database()
    except Exception:
        pass  # Ignore shutdown errors


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown() 