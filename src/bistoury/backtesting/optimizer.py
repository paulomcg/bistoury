import os
import json
from typing import Dict, Any, Tuple
import optuna
from pathlib import Path
from src.bistoury.config_manager import get_config_manager
from src.bistoury.backtesting.backtest_engine import BacktestEngine
import asyncio
import logging
import sys
import io
import signal

# List of known loggers to suppress
KNOWN_LOGGERS = [
    '',  # root
    'bistoury',
    'bistoury.agents',
    'bistoury.backtesting',
    'bistoury.models',
    'bistoury.paper_trading',
    'bistoury.signal_manager',
    'bistoury.strategies',
    'bistoury.database',
    'bistoury.hyperliquid',
    'bistoury.hyperliquid.collector',
    'bistoury.hyperliquid.client',
    'bistoury.logger',
    'optuna',
]

# Global flag for graceful shutdown
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[INFO] Optimization interrupted by user (Ctrl+C). Stopping gracefully...")
    # Force exit if signal handling doesn't work
    sys.exit(130)

def _suppress_logging_and_output(debug: bool):
    """Suppress logging and output for cleaner optimization runs."""
    if not debug:
        # Suppress all loggers except our trial logger
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.WARNING)
        logging.basicConfig(level=logging.WARNING)
        
        # Set optimizer.trial logger to INFO for trial-end and progress messages
        trial_logger = logging.getLogger("optimizer.trial")
        trial_logger.setLevel(logging.INFO)
        
        # Attach a StreamHandler if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in trial_logger.handlers):
            handler = logging.StreamHandler(sys.stderr)  # Use stderr to avoid stdout suppression issues
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
            trial_logger.addHandler(handler)
        trial_logger.propagate = False
        
        # Clear root handlers
        logging.root.handlers = [logging.NullHandler()]
        
        # Suppress stdout but keep stderr for important messages
        sys.stdout = io.StringIO()
        
        # Suppress rich console output if available
        try:
            import rich.console
            import rich.panel
            rich.console.Console.print = lambda *a, **k: None
            rich.panel.Panel.__init__ = lambda self, *a, **k: None
        except ImportError:
            pass

def get_search_space_from_strategy_config(strategy_config_path: str) -> Dict[str, Tuple[Any, Any, str]]:
    """
    Parse strategy.json for tunable parameters and return a dict:
    {param_name: (min, max, type)}
    """
    with open(strategy_config_path, 'r') as f:
        config = json.load(f)
    search_space = {}
    for param, value in config.get('tunable_parameters', {}).items():
        # Expecting: {"param": {"min": x, "max": y, "type": "float|int|categorical", ...}}
        if all(k in value for k in ('min', 'max', 'type')):
            search_space[param] = (value['min'], value['max'], value['type'])
    return search_space


def create_optuna_study(study_name: str, storage_dir: str) -> optuna.Study:
    """
    Create or load an Optuna study with SQLite storage in the given directory.
    """
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, f"{study_name}.db")
    storage_url = f"sqlite:///{storage_path}"
    return optuna.create_study(study_name=study_name, storage=storage_url, direction="maximize", load_if_exists=True)


def build_trial_config(base_config: dict, search_space: dict, trial: optuna.Trial) -> dict:
    """
    Given a base config and search space, update the config with trial-suggested values.
    """
    config = base_config.copy()
    for param, (min_val, max_val, param_type) in search_space.items():
        if param_type == "float":
            config[param] = trial.suggest_float(param, min_val, max_val)
        elif param_type == "int":
            config[param] = trial.suggest_int(param, min_val, max_val)
        elif param_type == "categorical":
            # For categorical, min_val is a list of choices
            config[param] = trial.suggest_categorical(param, min_val)
    return config


def objective_factory(
    base_config: dict,
    search_space: dict,
    output_dir: str,
    debug: bool
):
    """
    Returns an Optuna objective function that runs BacktestEngine with trial parameters.
    """
    def objective(trial: optuna.Trial) -> float:
        global _shutdown_requested
        
        logger = logging.getLogger("optimizer.trial")
        logger.info(f"Starting trial {trial.number}...")
        
        # Check if shutdown was requested
        if _shutdown_requested:
            logger.info(f"Trial {trial.number} ended: shutdown requested")
            raise optuna.TrialPruned()
        
        config = build_trial_config(base_config, search_space, trial)
        config.setdefault("symbol", base_config.get("symbol", "BTC"))
        config.setdefault("timeframe", base_config.get("timeframe", "1m"))
        config.setdefault("initial_balance", base_config.get("initial_balance", 10000))
        config.setdefault("replay_speed", base_config.get("replay_speed", 100.0))
        config.setdefault("min_confidence", base_config.get("min_confidence", 0.7))
        config.setdefault("duration", base_config.get("duration", 60))
        
        result_file = None
        try:
            # Use asyncio.run() but let KeyboardInterrupt propagate naturally
            result = asyncio.run(BacktestEngine(config, output_path=output_dir).run_backtest())
            sharpe = result.performance.sharpe_ratio
            
            # Try to infer the result file path (if written)
            if hasattr(result, 'result_file_path'):
                result_file = getattr(result, 'result_file_path')
            
            # Extract total_pnl and win_rate if available
            total_pnl = getattr(result.performance, 'total_pnl', None)
            win_rate = getattr(result.performance, 'win_rate', None)
            total_pnl_str = f"{total_pnl:.2f}" if total_pnl is not None else "N/A"
            win_rate_str = f"{(win_rate/100):.2%}" if win_rate is not None else "N/A"
            
            if sharpe is None:
                trial.set_user_attr("error", "No sharpe_ratio returned")
                logger.info(f"Trial {trial.number} ended: no sharpe_ratio (no result file)")
                return 0.0
            
            logger.info(f"Trial {trial.number} ended: success (result file: {result_file or 'unknown'}, total_pnl: {total_pnl_str}, win_rate: {win_rate_str})")
            return float(sharpe)
            
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow proper handling at higher levels
            logger.info(f"Trial {trial.number} ended: interrupted by user")
            raise
        except Exception as e:
            trial.set_user_attr("error", str(e))
            logger.info(f"Trial {trial.number} ended: exception ({e}) (no result file)")
            return 0.0
    
    return objective


def run_optimization(
    study_name: str,
    strategy_config_path: str,
    output_dir: str,
    n_trials: int = 20,
    n_jobs: int = 1,
    debug: bool = False
):
    """
    Main entry point for parameter optimization.
    - Loads search space from strategy config
    - Sets up Optuna study
    - Runs optimization (objective function runs BacktestEngine)
    - Saves best parameters to strategy.json
    """
    global _shutdown_requested
    
    # Set up signal handlers for immediate termination
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    _suppress_logging_and_output(debug)
    
    # Load base config
    with open(strategy_config_path, 'r') as f:
        base_config = json.load(f)
    
    search_space = get_search_space_from_strategy_config(strategy_config_path)
    study = create_optuna_study(study_name, output_dir)
    objective = objective_factory(base_config, search_space, output_dir, debug)
    
    if debug:
        print(f"[DEBUG] Search space: {search_space}")
        print(f"[DEBUG] Optuna study created at {output_dir}")
    
    try:
        # Use Optuna's built-in optimization with simple KeyboardInterrupt handling
        # Force n_jobs=1 to avoid multiprocessing signal handling issues
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=debug)
        
        if debug:
            print(f"[INFO] Optimization complete. Best value: {study.best_value}")
            print(f"[INFO] Best parameters: {study.best_params}")
        
        # Save best parameters back to strategy.json if we have any completed trials
        if study.trials:
            with open(strategy_config_path, 'r') as f:
                config = json.load(f)
            config['optimized_parameters'] = study.best_params
            with open(strategy_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if debug:
                print(f"[INFO] Best parameters saved to {strategy_config_path}")
            
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt gracefully
        print("\n[INFO] Optimization interrupted by user (Ctrl+C). Exiting gracefully...")
        
        # Still try to save best parameters if we have any completed trials
        if study.trials:
            try:
                with open(strategy_config_path, 'r') as f:
                    config = json.load(f)
                config['optimized_parameters'] = study.best_params
                with open(strategy_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"[INFO] Best parameters from completed trials saved to {strategy_config_path}")
            except Exception as e:
                print(f"[WARNING] Could not save partial results: {e}")
        
        # Re-raise to ensure proper exit code
        raise 