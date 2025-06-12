import click
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.bistoury.backtesting.backtest_engine import BacktestEngine

@click.command()
@click.option('--symbol', default='BTC', help='Trading symbol (e.g., BTC)')
@click.option('--timeframe', default='1m', help='Timeframe (e.g., 1m, 5m, 15m)')
@click.option('--start-date', default=None, help='Backtest start date (YYYY-MM-DD or ISO format)')
@click.option('--end-date', default=None, help='Backtest end date (YYYY-MM-DD or ISO format)')
@click.option('--initial-balance', default=10000, type=float, help='Initial capital for backtest')
@click.option('--replay-speed', default=100.0, type=float, help='Replay speed multiplier')
@click.option('--min-confidence', default=0.7, type=float, help='Minimum signal confidence')
@click.option('--duration', default=60, type=int, help='Session duration in seconds (default: auto)')
@click.option('--output-path', default=None, help='Directory to save backtest results')
def backtest(symbol, timeframe, start_date, end_date, initial_balance, replay_speed, min_confidence, duration, output_path):
    """
    Run a historical backtest and print a summary of results.
    """
    console = Console()
    config = {
        "symbol": symbol,
        "timeframe": timeframe,
        "initial_balance": initial_balance,
        "replay_speed": replay_speed,
        "min_confidence": min_confidence,
        "duration": duration,
    }
    if start_date:
        config["start_date"] = start_date
    if end_date:
        config["end_date"] = end_date
    # Run the backtest asynchronously
    async def run():
        try:
            engine = BacktestEngine(config, output_path=output_path)
            result = await engine.run_backtest()
            # Pretty-print summary
            table = Table(title="Backtest Summary", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Symbol", result.symbol)
            table.add_row("Timeframe", result.parameters_used.get('timeframe', 'N/A'))
            table.add_row("Start Date", str(result.start_date))
            table.add_row("End Date", str(result.end_date))
            table.add_row("Initial Capital", f"${result.initial_capital:,.2f}")
            table.add_row("Final Capital", f"${result.final_capital:,.2f}")
            table.add_row("Total Return (%)", f"{result.total_return_pct:.2f}%")
            table.add_row("Annualized Return (%)", f"{result.annualized_return:.2f}%")
            table.add_row("Max Drawdown (%)", f"{result.performance.max_drawdown:.2f}")
            table.add_row("Sharpe Ratio", f"{result.performance.sharpe_ratio if result.performance.sharpe_ratio is not None else 'N/A'}")
            table.add_row("Win Rate (%)", f"{result.performance.win_rate:.2f}")
            table.add_row("Total Trades", f"{result.performance.executed_signals}")
            console.print(table)
            console.print(Panel(f"Backtest ID: {result.backtest_id}\nSaved at: {output_path or engine.default_output_path}", title="Result Info", style="green"))
            return 0
        except Exception as e:
            console.print(f"[red]Backtest failed: {e}[/red]")
            return 1
    exit_code = asyncio.run(run())
    exit(exit_code) 