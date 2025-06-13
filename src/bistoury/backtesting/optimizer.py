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
import multiprocessing as mp
import threading
import time
import tempfile
import subprocess
import warnings
import shutil

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

# Global shutdown event for multiprocessing coordination
_shutdown_event = None
_worker_processes = []

def _simple_signal_handler(signum, frame):
    """Simple signal handler for single-process optimization."""
    print("\n[INFO] Optimization interrupted by user (Ctrl+C). Stopping gracefully...")
    sys.exit(130)

def _multiprocessing_signal_handler(signum, frame):
    """Signal handler for multiprocessing optimization - terminates all worker processes."""
    global _worker_processes, _shutdown_event
    print("\n[INFO] Optimization interrupted by user (Ctrl+C). Stopping all workers...")
    
    # Set shutdown event
    if _shutdown_event:
        _shutdown_event.set()
    
    # Terminate all worker processes
    for process in _worker_processes:
        if process.is_alive():
            process.terminate()
    
    # Wait for processes to terminate
    for process in _worker_processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
    
    sys.exit(130)

def _setup_worker_database(worker_id: int, debug: bool) -> str:
    """Set up a separate database file for a worker process."""
    try:
        # Create a unique database file for this worker
        worker_db_path = f"data/bistoury_worker_{worker_id}_{os.getpid()}.db"
        
        # Copy the main database to the worker database
        main_db_path = "data/bistoury.db"
        if os.path.exists(main_db_path):
            shutil.copy2(main_db_path, worker_db_path)
            if debug:
                print(f"[DEBUG] Worker {worker_id} created database copy: {worker_db_path}")
        else:
            if debug:
                print(f"[DEBUG] Worker {worker_id} main database not found, will create new: {worker_db_path}")
        
        # Set environment variable for this worker to use its own database
        os.environ['DATABASE_PATH'] = worker_db_path
        
        # CRITICAL: Reset the global database manager so it gets reinitialized with the new path
        from src.bistoury.database.connection import _db_manager
        import src.bistoury.database.connection as db_conn
        db_conn._db_manager = None
        
        # Force reinitialization of config and database manager with new path
        from src.bistoury.config import Config
        config = Config.load_from_env()
        from src.bistoury.database.connection import initialize_database
        initialize_database(config)
        
        if debug:
            print(f"[DEBUG] Worker {worker_id} reinitialized database manager with path: {worker_db_path}")
        
        return worker_db_path
        
    except Exception as e:
        if debug:
            print(f"[ERROR] Worker {worker_id} failed to setup database: {e}")
        raise

def _cleanup_worker_database(worker_db_path: str, debug: bool):
    """Clean up worker database file."""
    try:
        if os.path.exists(worker_db_path):
            os.remove(worker_db_path)
            if debug:
                print(f"[DEBUG] Cleaned up worker database: {worker_db_path}")
    except Exception as e:
        if debug:
            print(f"[WARNING] Failed to cleanup worker database {worker_db_path}: {e}")

def _worker_process(worker_id: int, study_name: str, storage_url: str, n_trials_per_worker: int, 
                   base_config: dict, search_space: dict, output_dir: str, debug: bool, shutdown_event):
    """Worker process function for multiprocessing optimization."""
    worker_db_path = None
    try:
        # CRITICAL: Suppress logging IMMEDIATELY to prevent any file logging conflicts
        # This must be the very first thing we do in worker processes
        _suppress_logging_and_output(debug)
        
        # CRITICAL: Disable all existing loggers that might already be configured
        import logging
        # Force shutdown of any existing logging handlers that might cause rotation
        logging.shutdown()
        # Clear all existing handlers from root logger
        for handler in logging.root.handlers[:]:
            try:
                handler.close()
                logging.root.removeHandler(handler)
            except:
                pass
        
        # Set up signal handling in worker
        def worker_signal_handler(signum, frame):
            if shutdown_event:
                shutdown_event.set()
            # Clean up database before exit
            if worker_db_path:
                _cleanup_worker_database(worker_db_path, debug)
            sys.exit(130)
        
        signal.signal(signal.SIGINT, worker_signal_handler)
        signal.signal(signal.SIGTERM, worker_signal_handler)
        
        # Set up separate database for this worker
        worker_db_path = _setup_worker_database(worker_id, debug)
        
        # Create study connection with retry logic for database locks
        max_retries = 5
        for attempt in range(max_retries):
            try:
                storage = optuna.storages.RDBStorage(url=storage_url)
                study = optuna.load_study(study_name=study_name, storage=storage)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if debug:
                        print(f"[DEBUG] Worker {worker_id} database connection attempt {attempt + 1} failed: {e}")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    if debug:
                        print(f"[ERROR] Worker {worker_id} failed to connect to database after {max_retries} attempts")
                    raise
        
        # Create objective function for this worker
        objective = objective_factory(base_config, search_space, output_dir, debug, shutdown_event)
        
        if debug:
            print(f"[DEBUG] Worker {worker_id} starting with {n_trials_per_worker} trials")
        
        # Run optimization in this worker with callback to check shutdown
        def should_stop_callback(study, trial):
            return shutdown_event and shutdown_event.is_set()
        
        study.optimize(
            objective, 
            n_trials=n_trials_per_worker, 
            show_progress_bar=False,
            callbacks=[lambda study, trial: should_stop_callback(study, trial)]
        )
        
        if debug:
            print(f"[DEBUG] Worker {worker_id} completed")
            
    except KeyboardInterrupt:
        if debug:
            print(f"[DEBUG] Worker {worker_id} interrupted by signal")
        sys.exit(130)
    except Exception as e:
        if debug:
            print(f"[ERROR] Worker {worker_id} failed: {e}")
        raise
    finally:
        # Clean up worker database
        if worker_db_path:
            _cleanup_worker_database(worker_db_path, debug)

def _suppress_logging_and_output(debug: bool):
    """Suppress logging and output for cleaner optimization runs."""
    if not debug:
        # CRITICAL: Completely disable all file logging to prevent rotation conflicts
        # Get the root logger and remove ALL handlers first
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Disable all existing loggers and remove their handlers
        for name in list(logging.root.manager.loggerDict.keys()):
            logger = logging.getLogger(name)
            # Remove ALL handlers from this logger
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            # Set to WARNING to suppress most output
            logger.setLevel(logging.WARNING)
            # Prevent propagation to avoid inherited handlers
            logger.propagate = False
        
        # Reconfigure basic logging with no file handlers
        logging.basicConfig(
            level=logging.WARNING,
            handlers=[],  # No handlers initially
            force=True    # Force reconfiguration
        )
        
        # Set up ONLY our trial logger with console output
        trial_logger = logging.getLogger("optimizer.trial")
        trial_logger.setLevel(logging.INFO)
        trial_logger.propagate = False
        
        # Remove any existing handlers from trial logger
        for handler in trial_logger.handlers[:]:
            trial_logger.removeHandler(handler)
        
        # Add only a console handler for trial messages
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        trial_logger.addHandler(console_handler)
        
        # Aggressively disable the rotating file handler that's causing issues
        try:
            # Disable all rotating file handlers to prevent conflicts
            import logging.handlers as log_handlers
            # Override the RotatingFileHandler to do nothing
            def disabled_rotating_handler(*args, **kwargs):
                return logging.NullHandler()
            log_handlers.RotatingFileHandler = disabled_rotating_handler
            log_handlers.TimedRotatingFileHandler = disabled_rotating_handler
        except Exception:
            pass  # If this fails, continue anyway
        
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


def create_optuna_study(study_name: str, storage_dir: str, sampler_type: str = "tpe") -> optuna.Study:
    """
    Create or load an Optuna study with SQLite storage and configurable sampling strategy.
    """
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, f"{study_name}.db")
    storage_url = f"sqlite:///{storage_path}"
    
    # Choose sampler based on strategy
    if sampler_type == "random":
        # Pure random sampling for maximum exploration
        sampler = optuna.samplers.RandomSampler()
    elif sampler_type == "grid":
        # Grid search for exhaustive coverage (limited parameters)
        # Note: Grid search requires predefined search space
        sampler = optuna.samplers.GridSampler()
    elif sampler_type == "cmaes":
        # CMA-ES for continuous optimization
        sampler = optuna.samplers.CmaEsSampler(
            n_startup_trials=50,
            independent_sampler=optuna.samplers.RandomSampler()
        )
    else:  # "tpe" (default)
        # TPE sampler with enhanced exploration for exhaustive search
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=50,  # More random trials for better exploration
            n_ei_candidates=100,  # More candidates for expected improvement
            multivariate=True,    # Consider parameter interactions
            group=True,           # Group related parameters
            warn_independent_sampling=True,
            constant_liar=True    # Better for parallel optimization
        )
    
    return optuna.create_study(
        study_name=study_name, 
        storage=storage_url, 
        direction="maximize", 
        load_if_exists=True,
        sampler=sampler
    )


def build_trial_config(base_config: dict, search_space: dict, trial: optuna.Trial) -> dict:
    """
    Given a base config and search space, update the config with trial-suggested values.
    Includes parameter constraints for more logical combinations.
    """
    config = base_config.copy()
    
    # Suggest parameters with logical constraints
    for param, (min_val, max_val, param_type) in search_space.items():
        if param_type == "float":
            config[param] = trial.suggest_float(param, min_val, max_val)
        elif param_type == "int":
            config[param] = trial.suggest_int(param, min_val, max_val)
        elif param_type == "categorical":
            # For categorical, min_val is a list of choices
            config[param] = trial.suggest_categorical(param, min_val)
    
    # Add logical constraints to ensure sensible parameter combinations
    
    # Ensure take profit is always higher than stop loss for positive risk/reward
    if 'default_take_profit_pct' in config and 'default_stop_loss_pct' in config:
        if config['default_take_profit_pct'] <= config['default_stop_loss_pct']:
            # Adjust take profit to be at least 1.5x stop loss
            config['default_take_profit_pct'] = config['default_stop_loss_pct'] * 1.5
    
    # Ensure risk/reward ratio is consistent with stop/take profit
    if all(k in config for k in ['min_risk_reward_ratio', 'default_take_profit_pct', 'default_stop_loss_pct']):
        actual_ratio = config['default_take_profit_pct'] / config['default_stop_loss_pct']
        if config['min_risk_reward_ratio'] > actual_ratio:
            # Adjust min_risk_reward_ratio to be achievable
            config['min_risk_reward_ratio'] = min(actual_ratio * 0.9, config['min_risk_reward_ratio'])
    
    # Ensure confluence score is not lower than pattern confidence (logical hierarchy)
    if 'min_confluence_score' in config and 'min_pattern_confidence' in config:
        if config['min_confluence_score'] < config['min_pattern_confidence']:
            config['min_confluence_score'] = config['min_pattern_confidence'] + trial.suggest_float(f"{param}_adjustment", 0, 10)
    
    return config


def objective_factory(
    base_config: dict,
    search_space: dict,
    output_dir: str,
    debug: bool,
    shutdown_event=None
):
    """
    Returns an Optuna objective function that runs BacktestEngine with trial parameters.
    """
    def objective(trial: optuna.Trial) -> float:
        logger = logging.getLogger("optimizer.trial")
        logger.info(f"Starting trial {trial.number}...")
        
        # Check if shutdown was requested (works for both single and multiprocessing)
        if shutdown_event and shutdown_event.is_set():
            logger.info(f"Trial {trial.number} ended: shutdown requested")
            raise optuna.TrialPruned()
        
        config = build_trial_config(base_config, search_space, trial)
        config.setdefault("symbol", base_config.get("symbol", "BTC"))
        config.setdefault("timeframe", base_config.get("timeframe", "1m"))
        config.setdefault("initial_balance", base_config.get("initial_balance", 10000))
        config.setdefault("replay_speed", base_config.get("replay_speed", 50.0))
        config.setdefault("min_confidence", base_config.get("min_confidence", 0.7))
        config.setdefault("duration", base_config.get("duration", 300))  # Longer duration for more data
        
        # Test multiple symbols for more diverse market conditions
        symbols_to_test = ["BTC"]  # Start with BTC, can expand to ["BTC", "ETH", "SOL"]
        total_sharpe = 0.0
        successful_tests = 0
        
        for symbol in symbols_to_test:
            if shutdown_event and shutdown_event.is_set():
                logger.info(f"Trial {trial.number} ended: shutdown requested during {symbol} test")
                raise optuna.TrialPruned()
            
            # Update config for this symbol
            symbol_config = config.copy()
            symbol_config["symbol"] = symbol
            
            try:
                result = asyncio.run(BacktestEngine(symbol_config, output_path=output_dir, shutdown_event=shutdown_event).run_backtest())
                
                if shutdown_event and shutdown_event.is_set():
                    logger.info(f"Trial {trial.number} ended: shutdown requested after {symbol} completion")
                    raise optuna.TrialPruned()
                
                sharpe = result.performance.sharpe_ratio
                if sharpe is not None:
                    total_sharpe += float(sharpe)
                    successful_tests += 1
                    
            except KeyboardInterrupt:
                logger.info(f"Trial {trial.number} ended: interrupted during {symbol} test")
                raise
            except Exception as e:
                logger.info(f"Trial {trial.number} {symbol} test failed: {e}")
                # Continue with other symbols
                continue
        
        if successful_tests == 0:
            trial.set_user_attr("error", "All symbol tests failed")
            logger.info(f"Trial {trial.number} ended: all tests failed")
            return 0.0
        
        # Average sharpe across all successful tests
        avg_sharpe = total_sharpe / successful_tests
        
        # Extract performance metrics for logging
        total_pnl = getattr(result.performance, 'total_pnl', None) if 'result' in locals() else None
        win_rate = getattr(result.performance, 'win_rate', None) if 'result' in locals() else None
        total_pnl_str = f"{total_pnl:.2f}" if total_pnl is not None else "N/A"
        win_rate_str = f"{(win_rate/100):.2%}" if win_rate is not None else "N/A"
        
        logger.info(f"Trial {trial.number} ended: success (avg_sharpe: {avg_sharpe:.4f}, symbols: {successful_tests}/{len(symbols_to_test)}, total_pnl: {total_pnl_str}, win_rate: {win_rate_str})")
        return avg_sharpe
    
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
    global _worker_processes, _shutdown_event
    
    # Set up appropriate signal handling based on n_jobs
    if n_jobs > 1:
        if debug:
            print(f"[INFO] Setting up multiprocessing optimization with {n_jobs} workers")
        signal.signal(signal.SIGINT, _multiprocessing_signal_handler)
        signal.signal(signal.SIGTERM, _multiprocessing_signal_handler)
        _shutdown_event = mp.Event()
    else:
        if debug:
            print("[INFO] Setting up single-process optimization")
        signal.signal(signal.SIGINT, _simple_signal_handler)
        signal.signal(signal.SIGTERM, _simple_signal_handler)
    
    try:
        _suppress_logging_and_output(debug)
        
        # Load base config
        with open(strategy_config_path, 'r') as f:
            base_config = json.load(f)
        
        search_space = get_search_space_from_strategy_config(strategy_config_path)
        
        if debug:
            print(f"[DEBUG] Search space: {search_space}")
        
        # Create storage for multiprocessing (SQLite for simplicity)
        if n_jobs > 1:
            # Use SQLite database for multiprocessing coordination
            storage_file = os.path.join(output_dir, f"{study_name}_optuna.db")
            storage_url = f"sqlite:///{storage_file}"
            storage = optuna.storages.RDBStorage(url=storage_url)
            
            # Create study with retry logic for database locks and advanced sampling
            max_retries = 5
            study = None
            for attempt in range(max_retries):
                try:
                    # Use TPE sampler with enhanced exploration for multiprocessing
                    sampler = optuna.samplers.TPESampler(
                        n_startup_trials=max(10, n_trials // 4),  # Scale startup trials with total trials
                        n_ei_candidates=48,
                        multivariate=True,
                        group=True,
                        warn_independent_sampling=True,
                        constant_liar=True  # Essential for parallel optimization
                    )
                    
                    study = optuna.create_study(
                        study_name=study_name, 
                        storage=storage, 
                        direction="maximize", 
                        load_if_exists=True,
                        sampler=sampler
                    )
                    break
                except optuna.exceptions.DuplicatedStudyError:
                    study = optuna.load_study(study_name=study_name, storage=storage)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        if debug:
                            print(f"[DEBUG] Study creation attempt {attempt + 1} failed: {e}")
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
            if study is None:
                raise RuntimeError("Failed to create or load study after multiple attempts")
            
            if debug:
                print(f"[DEBUG] Created multiprocessing study with storage: {storage_url}")
            
            # Calculate trials per worker
            trials_per_worker = n_trials // n_jobs
            remaining_trials = n_trials % n_jobs
            
            # Start worker processes
            _worker_processes = []
            for worker_id in range(n_jobs):
                worker_trials = trials_per_worker + (1 if worker_id < remaining_trials else 0)
                if worker_trials > 0:
                    process = mp.Process(
                        target=_worker_process,
                        args=(worker_id, study_name, storage_url, worker_trials, 
                              base_config, search_space, output_dir, debug, _shutdown_event)
                    )
                    process.start()
                    _worker_processes.append(process)
            
            # Wait for all workers to complete
            for process in _worker_processes:
                process.join()
            
            # Check if any worker failed
            failed_workers = [p for p in _worker_processes if p.exitcode != 0 and p.exitcode != 130]  # 130 is Ctrl+C
            if failed_workers and debug:
                print(f"[WARNING] {len(failed_workers)} workers failed")
            
            # Reload study to get final results
            study = optuna.load_study(study_name=study_name, storage=storage)
            
        else:
            # Single process optimization
            study = create_optuna_study(study_name, output_dir)
            objective = objective_factory(base_config, search_space, output_dir, debug, None)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=debug)
        
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
        
        # Clean up worker processes if they exist
        if n_jobs > 1:
            for process in _worker_processes:
                if process.is_alive():
                    process.terminate()
            for process in _worker_processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
        
        # Still try to save best parameters if we have any completed trials
        try:
            if n_jobs > 1:
                # Reload study for multiprocessing case
                storage_file = os.path.join(output_dir, f"{study_name}_optuna.db")
                storage_url = f"sqlite:///{storage_file}"
                storage = optuna.storages.RDBStorage(url=storage_url)
                study = optuna.load_study(study_name=study_name, storage=storage)
            
            if study.trials:
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