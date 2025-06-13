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

# Suppress urllib3 warnings early and comprehensively
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Also suppress at the urllib3 level if possible
try:
    import urllib3
    urllib3.disable_warnings()
except ImportError:
    pass

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
        # CRITICAL: Suppress urllib3 warnings FIRST before any other imports
        import warnings
        warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
        warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
        
        # Also suppress at the urllib3 level if possible
        try:
            import urllib3
            urllib3.disable_warnings()
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
        
        # Suppress logging in worker processes unless debug mode
        if not debug:
            _suppress_logging_and_output(debug)
        
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
        
        # Clear root handlers to avoid file logging conflicts in multiprocessing
        logging.root.handlers = [logging.NullHandler()]
        
        # Remove any file handlers from all loggers to prevent rotation conflicts
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            # Remove file handlers that could cause rotation conflicts
            handlers_to_remove = []
            for handler in logger.handlers:
                if hasattr(handler, 'baseFilename'):  # File-based handlers
                    handlers_to_remove.append(handler)
            for handler in handlers_to_remove:
                logger.removeHandler(handler)
        
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
        config.setdefault("replay_speed", base_config.get("replay_speed", 100.0))
        config.setdefault("min_confidence", base_config.get("min_confidence", 0.7))
        config.setdefault("duration", base_config.get("duration", 60))
        
        result_file = None
        try:
            # Check shutdown event periodically during backtest
            result = asyncio.run(BacktestEngine(config, output_path=output_dir, shutdown_event=shutdown_event).run_backtest())
            
            # Check again after backtest completes
            if shutdown_event and shutdown_event.is_set():
                logger.info(f"Trial {trial.number} ended: shutdown requested after completion")
                raise optuna.TrialPruned()
            
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
    global _worker_processes, _shutdown_event
    
    # Suppress urllib3 warnings comprehensively before any operations
    warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
    try:
        import urllib3
        urllib3.disable_warnings()
    except ImportError:
        pass
    
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
            
            # Create study with retry logic for database locks
            max_retries = 5
            study = None
            for attempt in range(max_retries):
                try:
                    study = optuna.create_study(
                        study_name=study_name, 
                        storage=storage, 
                        direction="maximize", 
                        load_if_exists=True
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