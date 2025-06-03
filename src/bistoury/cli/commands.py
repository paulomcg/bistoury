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
    """Reset database schema (WARNING: Deletes all data)."""
    try:
        cfg = Config()
        db_manager = DatabaseManager(cfg)
        schema = MarketDataSchema(db_manager)
        
        # Drop and recreate schema
        schema.drop_all_tables()
        console.print("[yellow]⚠️[/yellow] Dropped all tables")
        
        schema.create_all_tables()
        console.print("[green]✓[/green] Recreated schema")
        
        if schema.validate_schema():
            console.print("[green]✓[/green] Schema validation passed")
            console.print("[bold green]🎉 Database reset successfully![/bold green]")
        else:
            console.print("[red]✗[/red] Schema validation failed")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Database reset failed: {e}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()

if __name__ == '__main__':
    cli() 