"""Command-line interface commands for Bistoury."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..config import Config
from ..database import DatabaseManager
from ..database import MarketDataSchema, DataInsertion, DataQuery
from ..database import get_database_switcher, switch_database, get_compatible_query

console = Console()

@click.group()
@click.version_option()
def cli():
    """Bistoury: LLM-Driven Cryptocurrency Trading System."""
    pass

@cli.command()
@click.option(
    '--config', 
    type=click.Path(exists=True), 
    help='Path to configuration file'
)
def init(config):
    """Initialize Bistoury configuration and database."""
    try:
        # Load configuration
        if config:
            cfg = Config.from_file(config)
        else:
            cfg = Config()
            
        console.print("[green]✓[/green] Configuration loaded")
        
        # Initialize database
        db_manager = DatabaseManager(cfg)
        console.print("[green]✓[/green] Database manager initialized")
        
        # Create schema
        schema = MarketDataSchema(db_manager)
        schema.create_all_tables()
        console.print("[green]✓[/green] Database schema created")
        
        # Validate schema
        if schema.validate_schema():
            console.print("[green]✓[/green] Schema validation passed")
        else:
            console.print("[red]✗[/red] Schema validation failed")
            return
            
        console.print("\n[bold green]🎉 Bistoury initialized successfully![/bold green]")
        console.print("\nNext steps:")
        console.print("• Run [bold]bistoury status[/bold] to check system status")
        console.print("• Run [bold]bistoury db-info[/bold] to view database information")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Initialization failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

@cli.command()
def status():
    """Display system status."""
    try:
        cfg = Config()
        db_manager = DatabaseManager(cfg)
        
        # Create status table
        table = Table(title="🔍 Bistoury System Status", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Check configuration
        table.add_row(
            "Configuration",
            "✓ Loaded",
            f"Database: {cfg.database.path}"
        )
        
        # Check database connection
        try:
            info = db_manager.get_database_info()
            table.add_row(
                "Database",
                "✓ Connected",
                f"Size: {info.get('database_size', 'Unknown')}"
            )
        except Exception as e:
            table.add_row(
                "Database",
                "✗ Error",
                str(e)
            )
            
        # Check schema
        try:
            schema = MarketDataSchema(db_manager)
            if schema.validate_schema():
                table.add_row(
                    "Schema",
                    "✓ Valid",
                    "All required tables exist"
                )
            else:
                table.add_row(
                    "Schema",
                    "✗ Invalid",
                    "Missing required tables"
                )
        except Exception as e:
            table.add_row(
                "Schema",
                "✗ Error",
                str(e)
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Status check failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

@cli.command(name="db-status")
def db_status():
    """Display detailed database status."""
    try:
        cfg = Config()
        db_manager = DatabaseManager(cfg)
        
        # Get database info
        info = db_manager.get_database_info()
        
        # Create info table
        table = Table(title="🗄️ Database Status", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Path", str(cfg.database.path))
        table.add_row("Size", f"{info.get('database_size', 'Unknown')}")
        table.add_row("Connection Pool", f"{db_manager.max_connections} max connections")
        table.add_row("Active Connections", str(len(db_manager._connections)))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Database status check failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

@cli.command(name="db-info")
def db_info():
    """Display database schema information."""
    try:
        cfg = Config()
        db_manager = DatabaseManager(cfg)
        schema = MarketDataSchema(db_manager)
        
        # Get schema info
        schema_info = schema.get_schema_info()
        
        if not schema_info:
            console.print("[yellow]No tables found. Run 'bistoury init' to create schema.[/yellow]")
            return
        
        # Create schema table
        table = Table(title="📊 Database Schema", show_header=True)
        table.add_column("Table", style="cyan")
        table.add_column("Rows", style="green")
        table.add_column("Columns", style="yellow")
        
        for table_name, info in schema_info.items():
            row_count = info.get('row_count', 0)
            col_count = len(info.get('columns', []))
            
            table.add_row(
                table_name,
                f"{row_count:,}",
                str(col_count)
            )
        
        console.print(table)
        
        # Show column details for each table
        for table_name, info in schema_info.items():
            columns = info.get('columns', [])
            if columns:
                console.print(f"\n[bold cyan]{table_name}[/bold cyan] columns:")
                for col in columns[:5]:  # Show first 5 columns
                    console.print(f"  • {col[0]} ({col[1]})")
                if len(columns) > 5:
                    console.print(f"  ... and {len(columns) - 5} more")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Database info failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

@cli.command(name="db-reset")
@click.confirmation_option(prompt="Are you sure you want to reset the database? This will delete all data.")
def db_reset():
    """Reset database schema (deletes all data)."""
    try:
        cfg = Config()
        db_manager = DatabaseManager(cfg)
        schema = MarketDataSchema(db_manager)
        
        # Drop and recreate all tables
        schema.recreate_all_tables()
        
        console.print("[green]✓[/green] Database schema reset successfully")
        console.print("[yellow]⚠[/yellow] All data has been deleted")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Database reset failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

@cli.command(name="db-list")
def db_list():
    """List all available databases."""
    try:
        switcher = get_database_switcher()
        databases = switcher.list_available_databases()
        
        # Create database table
        table = Table(title="🗄️ Available Databases", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Schema Type", style="magenta")
        table.add_column("Tables", style="blue")
        table.add_column("Path")
        
        for name, info in databases.items():
            status = "✅ Available" if info['exists'] else "❌ Missing"
            schema_type = info.get('schema_type', 'unknown')
            tables = str(info.get('tables', 'N/A'))
            
            table.add_row(
                name,
                status,
                info['size'],
                schema_type,
                tables,
                info['path']
            )
        
        console.print(table)
        
        # Show current database
        current_db = switcher.get_current_database()
        if current_db:
            console.print(f"\n[bold green]Current database:[/bold green] {current_db}")
        else:
            console.print(f"\n[yellow]No database currently selected[/yellow]")
            
        console.print(f"\n[bold]Usage:[/bold]")
        console.print(f"• Use [bold]bistoury db-switch <name>[/bold] to switch databases")
        console.print(f"• Use [bold]BISTOURY_DATABASE=test bistoury ...[/bold] for environment variable control")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list databases: {e}")
        sys.exit(1)

@cli.command(name="db-switch")
@click.argument('database_name')
@click.option('--test', is_flag=True, help='Test the database connection after switching')
def db_switch(database_name, test):
    """Switch to a different database."""
    try:
        switcher = get_database_switcher()
        
        # Check if database exists
        databases = switcher.list_available_databases()
        if database_name not in databases:
            console.print(f"[red]✗[/red] Unknown database: {database_name}")
            console.print(f"Available databases: {', '.join(databases.keys())}")
            sys.exit(1)
            
        db_info = databases[database_name]
        if not db_info['exists']:
            console.print(f"[red]✗[/red] Database file does not exist: {db_info['path']}")
            sys.exit(1)
        
        # Switch to the database
        db_manager = switcher.switch_to_database(database_name)
        console.print(f"[green]✓[/green] Switched to database: {database_name}")
        console.print(f"Path: {db_info['path']}")
        console.print(f"Size: {db_info['size']}")
        console.print(f"Schema: {db_info.get('schema_type', 'unknown')}")
        
        # Test connection if requested
        if test:
            console.print("\n[bold]Testing database connection...[/bold]")
            
            schema_type = db_info.get('schema_type', 'production')
            query = get_compatible_query(db_manager, schema_type)
            
            # Test symbols
            symbols = query.get_symbols()
            console.print(f"[green]✓[/green] Found {len(symbols)} symbols")
            
            if symbols:
                # Test trades
                symbol = symbols[0]['symbol']
                trades = query.get_latest_trades(symbol, limit=5)
                console.print(f"[green]✓[/green] Found {len(trades)} recent trades for {symbol}")
                
                # Test orderbook
                orderbook = query.get_latest_orderbook(symbol)
                if orderbook:
                    bids = len(orderbook['levels'][0]) if orderbook['levels'] and len(orderbook['levels']) > 0 else 0
                    asks = len(orderbook['levels'][1]) if orderbook['levels'] and len(orderbook['levels']) > 1 else 0
                    console.print(f"[green]✓[/green] Found orderbook with {bids} bids and {asks} asks")
                else:
                    console.print(f"[yellow]⚠[/yellow] No orderbook data for {symbol}")
            
            console.print(f"[green]✓[/green] Database test completed successfully")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Database switch failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

@cli.command(name="db-stats")
@click.argument('database_name', required=False)
@click.option('--symbols', '-s', is_flag=True, help='Show symbol statistics')
@click.option('--trades', '-t', is_flag=True, help='Show trade statistics')  
@click.option('--detailed', '-d', is_flag=True, help='Show detailed statistics')
def db_stats(database_name, symbols, trades, detailed):
    """Show database statistics."""
    try:
        switcher = get_database_switcher()
        
        # Use current database if none specified
        if not database_name:
            database_name = switcher.get_current_database()
            if not database_name:
                # Default to production
                database_name = 'production'
        
        # Get database info
        databases = switcher.list_available_databases()
        if database_name not in databases:
            console.print(f"[red]✗[/red] Unknown database: {database_name}")
            sys.exit(1)
            
        db_info = databases[database_name]
        if not db_info['exists']:
            console.print(f"[red]✗[/red] Database file does not exist: {db_info['path']}")
            sys.exit(1)
        
        # Switch to database
        db_manager = switcher.switch_to_database(database_name)
        schema_type = db_info.get('schema_type', 'production')
        query = get_compatible_query(db_manager, schema_type)
        
        console.print(f"[bold]📊 Database Statistics: {database_name}[/bold]")
        console.print(f"Path: {db_info['path']}")
        console.print(f"Size: {db_info['size']}")
        console.print(f"Schema: {schema_type}")
        
        # Basic stats
        symbols_data = query.get_symbols()
        console.print(f"\n[bold]Basic Statistics:[/bold]")
        console.print(f"• Symbols: {len(symbols_data)}")
        
        if symbols_data and (symbols or detailed):
            console.print(f"\n[bold]Symbol Details:[/bold]")
            for symbol in symbols_data:
                console.print(f"• {symbol['symbol']}: {symbol['max_leverage']}x leverage")
        
        if symbols_data and (trades or detailed):
            console.print(f"\n[bold]Trade Statistics:[/bold]")
            for symbol in symbols_data[:5]:  # Limit to first 5 symbols
                symbol_name = symbol['symbol']
                recent_trades = query.get_latest_trades(symbol_name, limit=10)
                console.print(f"• {symbol_name}: {len(recent_trades)} recent trades")
                
        if detailed and schema_type == 'test_legacy':
            # Get detailed stats for test database
            from ..database.database_switcher import TestDatabaseCompatibilityLayer
            compat_layer = TestDatabaseCompatibilityLayer(db_manager)
            stats = compat_layer.get_raw_data_stats()
            
            if 'date_range' in stats:
                console.print(f"\n[bold]Date Range:[/bold]")
                console.print(f"• Start: {stats['date_range']['start']}")
                console.print(f"• End: {stats['date_range']['end']}")
            
            if 'channels' in stats:
                console.print(f"\n[bold]Message Channels:[/bold]")
                for channel, count in stats['channels'].items():
                    console.print(f"• {channel}: {count:,} messages")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Database stats failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

if __name__ == '__main__':
    cli() 