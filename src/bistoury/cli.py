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


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status and configuration."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    click.echo("🤖 Bistoury Trading System Status")
    click.echo("=" * 40)
    
    # Configuration status
    click.echo(f"📁 Database Path: {config.database.path}")
    click.echo(f"📝 Log Level: {config.logging.level}")
    click.echo(f"💰 Risk Limit: ${config.trading.risk_limit_usd}")
    click.echo(f"📊 Default Pairs: {', '.join(config.data.default_pairs)}")
    click.echo(f"⏰ Timeframes: {', '.join(config.data.default_timeframes)}")
    
    # LLM providers
    providers = config.get_available_llm_providers()
    click.echo(f"🧠 LLM Providers: {', '.join(providers)}")
    
    # HyperLiquid connection
    if config.hyperliquid:
        mode = "🧪 Testnet" if config.hyperliquid.testnet else "🚀 Mainnet"
        click.echo(f"🔗 HyperLiquid: {mode}")
    else:
        click.echo("🔗 HyperLiquid: ❌ Not configured")
    
    # Validation
    if not config.validate_llm_keys():
        click.echo("⚠️  Warning: No LLM API keys configured", err=True)
    
    logger.info("Status command executed")


@main.command()
@click.option(
    "--pairs",
    "-p",
    help="Trading pairs to collect data for (comma-separated)"
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["continuous", "once"]),
    default="continuous",
    help="Data collection mode"
)
@click.pass_context
def collect(ctx: click.Context, pairs: Optional[str], mode: str) -> None:
    """Start collecting market data from HyperLiquid."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    # Parse pairs
    if pairs:
        pair_list = [p.strip().upper() for p in pairs.split(",")]
    else:
        pair_list = config.data.default_pairs
    
    click.echo(f"🔄 Starting data collection for: {', '.join(pair_list)}")
    click.echo(f"📈 Mode: {mode}")
    
    if not config.hyperliquid:
        click.echo("❌ Error: HyperLiquid API not configured", err=True)
        sys.exit(1)
    
    logger.info(f"Data collection started for pairs: {pair_list}, mode: {mode}")
    
    # TODO: Implement actual data collection
    click.echo("⚠️  Data collection not yet implemented")


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
    
    if not confirm:
        click.echo("❌ Live trading requires --confirm flag", err=True)
        click.echo("⚠️  This will trade with REAL MONEY. Use at your own risk!")
        sys.exit(1)
    
    if not config.hyperliquid:
        click.echo("❌ Error: HyperLiquid API not configured", err=True)
        sys.exit(1)
    
    if not config.validate_llm_keys():
        click.echo("❌ Error: No LLM API keys configured", err=True)
        sys.exit(1)
    
    effective_risk_limit = risk_limit or config.trading.risk_limit_usd
    
    click.echo("🚨 LIVE TRADING MODE ACTIVATED")
    click.echo("=" * 40)
    click.echo(f"💰 Risk Limit: ${effective_risk_limit}")
    click.echo(f"🎯 Max Positions: {config.trading.max_positions}")
    click.echo(f"📊 Trading Pairs: {', '.join(config.data.default_pairs)}")
    click.echo("⚠️  YOU ARE TRADING WITH REAL MONEY!")
    
    # Final confirmation
    if not click.confirm("Are you absolutely sure you want to proceed?"):
        click.echo("Trading cancelled.")
        sys.exit(0)
    
    logger.warning(f"Live trading started with risk limit: ${effective_risk_limit}")
    
    # TODO: Implement live trading
    click.echo("⚠️  Live trading not yet implemented")


@main.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize Bistoury configuration and data directories."""
    config: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    click.echo("🚀 Initializing Bistoury...")
    
    # Create directories
    directories = [
        Path(config.database.path).parent,
        Path(config.database.backup_path),
        Path(config.logging.file_path).parent,
        Path("data"),
        Path("logs"),
        Path("backups")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 Created directory: {directory}")
    
    # Check configuration
    if not config.validate_llm_keys():
        click.echo("⚠️  Warning: No LLM API keys found in environment")
        click.echo("   Please configure at least one LLM provider in your .env file")
    
    if not config.hyperliquid:
        click.echo("⚠️  Warning: HyperLiquid API not configured")
        click.echo("   Please add HYPERLIQUID_API_KEY and HYPERLIQUID_SECRET_KEY to .env")
    
    click.echo("✅ Initialization complete!")
    logger.info("Bistoury initialization completed")


if __name__ == "__main__":
    main() 