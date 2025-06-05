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
    default="live",
    help="Paper trading mode"
)
@click.option(
    "--duration",
    "-d",
    help="Trading duration (e.g., '24h', '1d', '1w')"
)
@click.pass_context
def paper_trade(ctx: click.Context, mode: str, duration: Optional[str]) -> None:
    """Start paper trading (no real money)."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    click.echo(f"📊 Starting paper trading")
    click.echo(f"🎯 Mode: {mode}")
    if duration:
        click.echo(f"⏰ Duration: {duration}")
    
    logger.info(f"Paper trading started: mode={mode}, duration={duration}")
    
    # TODO: Implement paper trading
    click.echo("⚠️  Paper trading not yet implemented")


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