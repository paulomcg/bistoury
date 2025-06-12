import click
from rich.console import Console
from src.bistoury.backtesting.optimizer import run_optimization

@click.command()
@click.option('--strategy-config', default='config/strategy.json', help='Path to strategy config JSON')
@click.option('--output-dir', default='./backtest_results', help='Directory to save optimization results and study DB')
@click.option('--study-name', default='bistoury_optimize', help='Optuna study name')
@click.option('--trials', default=20, type=int, help='Number of optimization trials')
@click.option('--jobs', default=1, type=int, help='Number of parallel jobs (n_jobs)')
@click.option('--debug', is_flag=True, default=False, help='Enable verbose logging and output')
def optimize(strategy_config, output_dir, study_name, trials, jobs, debug):
    """
    Run parameter optimization using Optuna and BacktestEngine.
    Quiet by default; use --debug to print all logs and output.
    Press Ctrl+C to interrupt gracefully.
    """
    console = Console()
    try:
        run_optimization(
            study_name=study_name,
            strategy_config_path=strategy_config,
            output_dir=output_dir,
            n_trials=trials,
            n_jobs=jobs,
            debug=debug
        )
        console.print(f"[bold green]Optimization complete! Study: {study_name}[/bold green]")
        return 0
    except KeyboardInterrupt:
        # KeyboardInterrupt is already handled in run_optimization, just exit cleanly
        console.print(f"[yellow]Optimization interrupted. Partial results may have been saved.[/yellow]")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        return 1 