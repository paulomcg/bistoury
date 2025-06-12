import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from decimal import Decimal
from src.bistoury.models.strategies import BacktestResult, StrategyPerformance
from src.bistoury.paper_trading.utils import (
    create_paper_trading_agents,
    get_available_date_range,
    create_agent_registration,
    setup_agent_subscriptions,
)
from src.bistoury.agents.messaging import MessageBus
from src.bistoury.agents.registry import AgentRegistry
from src.bistoury.agents.orchestrator import AgentOrchestrator
from src.bistoury.models.orchestrator_config import OrchestratorConfig
from src.bistoury.config_manager import get_config_manager
from rich.console import Console
import os
import uuid
import json

console = Console()

class BacktestEngine:
    """
    BacktestEngine orchestrates historical trading simulations by coordinating
    CollectorAgent (historical replay), StrategyAgent(s), and PositionManagerAgent.
    Produces BacktestResult objects and handles result persistence.
    """
    def __init__(self, config: Dict[str, Any], output_path: Optional[str] = None, shutdown_event=None):
        """
        Initialize the BacktestEngine.
        Args:
            config: Dictionary of configuration parameters (strategy, dates, capital, etc.)
            output_path: Optional path to save results (JSON/DuckDB)
            shutdown_event: Optional multiprocessing.Event for coordinated shutdown
        """
        self.config = config
        self.output_path = output_path
        self.shutdown_event = shutdown_event
        # Initialize ConfigManager for loading defaults and paths
        self.config_manager = get_config_manager()
        # Load default values from config/index.json and other configs
        self.default_start_date = self.config_manager.get('index', 'default_start_date', default=None)
        self.default_end_date = self.config_manager.get('index', 'default_end_date', default=None)
        self.default_initial_capital = self.config_manager.get('trading', 'initial_balance', default=10000)
        self.default_output_path = self.config_manager.get('index', 'backtest_results_path', default='backtest_results/')

    async def run_backtest(self) -> BacktestResult:
        """
        Run the backtest simulation and return a BacktestResult.
        Orchestrates agent startup, historical replay, and result collection.
        Returns:
            BacktestResult: The result object containing all metrics and details.
        """
        # Extract config parameters, falling back to config manager defaults
        symbol = self.config.get("symbol", "BTC")
        timeframe = self.config.get("timeframe", "1m")
        balance = float(self.config.get("initial_balance", self.default_initial_capital))
        speed = float(self.config.get("replay_speed", 100.0))
        min_confidence = float(self.config.get("min_confidence", 0.7))
        duration = int(self.config.get("duration", 60))  # seconds

        # Determine date range
        start_date = self.config.get("start_date") or self.default_start_date
        end_date = self.config.get("end_date") or self.default_end_date
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        # If not provided, auto-detect from DB
        if not start_date or not end_date:
            db_start, db_end = await get_available_date_range(symbol, timeframe)
            start_date = start_date or db_start
            end_date = end_date or db_end
        if not start_date or not end_date:
            raise RuntimeError(f"No historical data available for {symbol} {timeframe}")

        # Use all data if duration is default (60)
        use_all_data = (duration == 60)
        if use_all_data:
            data_time_span = (end_date - start_date).total_seconds()
            calculated_duration = int(data_time_span / speed)
            calculated_duration = max(30, min(calculated_duration, 600))
            actual_duration = calculated_duration
        else:
            actual_duration = duration

        # Set up message bus and registry
        message_bus = MessageBus(enable_persistence=False)
        await message_bus.start()
        agent_registry = AgentRegistry(message_bus)
        await agent_registry.start()
        orchestrator_config = OrchestratorConfig()
        orchestrator = AgentOrchestrator(
            config=orchestrator_config,
            registry=agent_registry,
            message_bus=message_bus
        )

        # Create agents
        agents = await create_paper_trading_agents(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            speed=speed,
            balance=balance,
            min_confidence=min_confidence,
            message_bus=message_bus
        )

        # Register agents and set up subscriptions
        for agent in agents:
            registration = create_agent_registration(agent)
            await agent_registry.register_agent(registration)
            await setup_agent_subscriptions(agent, message_bus)
            if hasattr(agent, 'set_message_bus'):
                agent.set_message_bus(message_bus)

        # Start orchestrator and agents
        await orchestrator.start()
        for agent in agents:
            if hasattr(agent, 'start'):
                await agent.start()

        # Run for specified duration or until replay is complete
        start_time = asyncio.get_event_loop().time()
        replay_complete = asyncio.Event()

        async def monitor_replay_completion():
            try:
                while not replay_complete.is_set():
                    collector_agent = None
                    for agent in agents:
                        if hasattr(agent, 'name') and 'collector' in agent.name.lower():
                            collector_agent = agent
                            break
                    if collector_agent and hasattr(collector_agent, 'replay_completed'):
                        if collector_agent.replay_completed:
                            replay_complete.set()
                            break
                    # Use shorter sleep to be more responsive
                    await asyncio.sleep(0.5)
            except (Exception, asyncio.CancelledError):
                pass

        replay_monitor_task = asyncio.create_task(monitor_replay_completion())
        try:
            while not replay_complete.is_set():
                # Check for shutdown event first (multiprocessing coordination)
                if self.shutdown_event and self.shutdown_event.is_set():
                    console.print("[yellow]Backtest interrupted by shutdown event[/yellow]")
                    break
                    
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                if elapsed >= actual_duration:
                    break
                # Use shorter sleep intervals to be more responsive to interruption
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            pass
        finally:
            replay_monitor_task.cancel()
            try:
                await replay_monitor_task
            except asyncio.CancelledError:
                pass

        # Stop orchestrator and agents
        await orchestrator.stop()
        await agent_registry.stop()
        await message_bus.stop()

        # Extract real metrics from PositionManagerAgent
        position_agent = None
        for agent in agents:
            if hasattr(agent, 'name') and 'position' in agent.name.lower():
                position_agent = agent
                break

        if position_agent is not None:
            metrics = position_agent.get_performance_metrics()
            # Build StrategyPerformance object
            try:
                perf = StrategyPerformance(
                    strategy_name="CandlestickStrategy",
                    strategy_version="1.0.0",
                    start_date=start_date,
                    end_date=end_date,
                    total_signals=metrics.get('total_trades', 0),
                    executed_signals=metrics.get('total_trades', 0),
                    winning_signals=metrics.get('winning_trades', 0),
                    losing_signals=metrics.get('total_trades', 0) - metrics.get('winning_trades', 0),
                    total_return=Decimal(str(metrics.get('total_return', 0))),
                    total_pnl=Decimal(str(metrics.get('total_pnl', 0))),
                    max_drawdown=Decimal(str(metrics.get('max_drawdown', 0))),
                    volatility=Decimal(str(metrics.get('volatility', 0))),
                    sharpe_ratio=Decimal(str(metrics.get('sharpe_ratio', 0))) if metrics.get('sharpe_ratio') is not None else None,
                    win_rate=Decimal(str(metrics.get('win_rate', 0))),
                    profit_factor=Decimal(str(metrics.get('profit_factor', 0))) if metrics.get('profit_factor') is not None else None,
                    average_win=Decimal(str(metrics.get('avg_win', 0))),
                    average_loss=Decimal(str(metrics.get('avg_loss', 0))),
                    largest_win=Decimal(str(metrics.get('largest_win', 0))),
                    largest_loss=Decimal(str(metrics.get('largest_loss', 0))),
                    consecutive_wins=metrics.get('max_consecutive_wins', 0),
                    consecutive_losses=metrics.get('max_consecutive_losses', 0),
                    max_consecutive_wins=metrics.get('max_consecutive_wins', 0),
                    max_consecutive_losses=metrics.get('max_consecutive_losses', 0),
                    signal_performances=[],  # Could be filled in future
                    daily_returns=[],        # Could be filled in future
                    equity_curve=[],         # Could be filled in future
                    performance_metrics={},  # Could be filled in future
                )
                initial_capital = Decimal(str(position_agent.config.initial_balance)) if hasattr(position_agent, 'config') and hasattr(position_agent.config, 'initial_balance') else Decimal(str(balance))
                final_capital = Decimal(str(metrics.get('total_balance', balance)))
            except Exception as e:
                console.print(f"[red]Error building StrategyPerformance: {e}[/red]")
                perf = StrategyPerformance(
                    strategy_name="CandlestickStrategy",
                    strategy_version="1.0.0",
                    start_date=start_date,
                    end_date=end_date
                )
                initial_capital = Decimal(str(balance))
                final_capital = Decimal(str(balance))
        else:
            console.print("[yellow]Warning: PositionManagerAgent not found, using dummy performance.[/yellow]")
            perf = StrategyPerformance(
                strategy_name="CandlestickStrategy",
                strategy_version="1.0.0",
                start_date=start_date,
                end_date=end_date
            )
            initial_capital = Decimal(str(balance))
            final_capital = Decimal(str(balance))

        result = BacktestResult(
            backtest_id="dummy-id",
            strategy_name="CandlestickStrategy",
            strategy_version="1.0.0",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            performance=perf,
            parameters_used=self.config,
            market_conditions={},
            run_timestamp=datetime.now(timezone.utc)
        )

        # Persist result to JSON if output_path is set
        output_dir = self.output_path or self.default_output_path
        os.makedirs(output_dir, exist_ok=True)
        # Use timestamp and uuid for uniqueness
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        filename = f"backtest_{symbol}_{timeframe}_{ts}_{file_id}.json"
        file_path = os.path.join(output_dir, filename)
        # Use model_dump_json if available (Pydantic v2), else fallback to .json()
        try:
            if hasattr(result, 'model_dump_json'):
                json_str = result.model_dump_json(indent=2)
            else:
                json_str = result.json(indent=2)
            with open(file_path, 'w') as f:
                f.write(json_str)
            console.print(f"[green]BacktestResult saved to {file_path}[/green]")
            # Set result_file_path directly (field is now present)
            result.result_file_path = file_path
        except Exception as e:
            console.print(f"[red]Failed to save BacktestResult: {e}[/red]")
        # TODO: Add DuckDB persistence in the future
        return result 