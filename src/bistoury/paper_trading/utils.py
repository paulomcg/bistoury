import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional
from ..agents.collector_agent import CollectorAgent, CollectorAgentConfig
from ..agents.candlestick_strategy_agent import CandlestickStrategyAgent
from ..agents.position_manager_agent import PositionManagerAgent, PositionManagerConfig
from ..agents.messaging import MessageBus
from ..agents.registry import AgentRegistry
from ..models.orchestrator_config import OrchestratorConfig
from ..models.agent_registry import (
    AgentRegistration, AgentCapability, AgentCapabilityType, 
    AgentCompatibility, AgentType
)
from ..database import get_database_switcher
from ..models.agent_messages import MessageType
from ..agents.messaging import MessageFilter
from rich.console import Console
from src.bistoury.database.connection import get_connection

console = Console()

async def create_paper_trading_agents(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    speed: float,
    balance: float,
    min_confidence: float,
    message_bus: MessageBus
) -> list:
    """
    Create and configure agents for paper trading or backtesting.
    Returns a list of initialized agent instances.
    """
    agents = []
    collector_config = CollectorAgentConfig(
        symbols={symbol},
        intervals={timeframe},
        historical_replay_mode=True,
        replay_start_date=start_date,
        replay_end_date=end_date,
        replay_speed=speed,
        publish_data_updates=True,
        data_update_interval=0.1
    )
    collector_agent = CollectorAgent(
        hyperliquid=None,
        db_manager=None,
        config={"collector": collector_config.__dict__},
        name="paper_collector",
        persist_state=False
    )
    collector_agent._message_bus = message_bus
    agents.append(collector_agent)
    strategy_config = {
        "symbols": [symbol],
        "timeframes": [timeframe],
        "min_confidence_threshold": min_confidence,
        "agent_name": "paper_strategy"
    }
    strategy_agent = CandlestickStrategyAgent(name="paper_strategy", config=strategy_config)
    strategy_agent.persist_state = False
    strategy_agent._message_bus = message_bus
    agents.append(strategy_agent)
    if symbol.upper() == "BTC":
        min_pos_size = Decimal('0.001')
        max_pos_size = Decimal('0.1')
    elif symbol.upper() == "ETH":
        min_pos_size = Decimal('0.01')
        max_pos_size = Decimal('3.0')
    else:
        min_pos_size = Decimal('1.0')
        max_pos_size = Decimal('1000.0')
    position_config = PositionManagerConfig(
        initial_balance=balance,
        min_position_size=min_pos_size,
        max_position_size=max_pos_size,
        enable_stop_loss=True,
        enable_take_profit=True,
        taker_fee_rate=Decimal('0.00045'),
        maker_fee_rate=Decimal('0.00015')
    )
    position_agent = PositionManagerAgent(
        name="paper_position_manager",
        config=position_config
    )
    position_agent.persist_state = False
    position_agent._message_bus = message_bus
    agents.append(position_agent)
    return agents

async def get_available_date_range(symbol: str, timeframe: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the available date range for historical data for a given symbol and timeframe.
    Returns (min_date, max_date) or (None, None) if unavailable.
    """
    try:
        conn = get_connection()
        table_name = f"candles_{timeframe}"
        query = f"""
        SELECT MIN(timestamp_start) as min_date, MAX(timestamp_start) as max_date
        FROM {table_name}
        WHERE symbol = ?
        """
        cursor = conn.execute(query, [symbol])
        row = cursor.fetchone()
        if row and row[0]:
            min_date = datetime.fromisoformat(str(row[0]).replace('Z', '+00:00'))
            max_date = datetime.fromisoformat(str(row[1]).replace('Z', '+00:00'))
            return min_date, max_date
        return None, None
    except Exception as e:
        console.print(f"[red]Error getting date range: {e}[/red]")
        return None, None

def create_agent_registration(agent) -> AgentRegistration:
    """
    Convert a BaseAgent to an AgentRegistration for the registry.
    """
    capabilities = []
    if hasattr(agent, 'metadata') and hasattr(agent.metadata, 'capabilities'):
        for cap_str in agent.metadata.capabilities:
            capability_type = map_capability_string(cap_str)
            if capability_type:
                capabilities.append(AgentCapability(
                    type=capability_type,
                    description=f"{cap_str} capability",
                    version="1.0.0"
                ))
    elif hasattr(agent, 'capabilities'):
        for cap in agent.capabilities:
            if hasattr(cap, 'name'):
                capability_type = map_capability_string(cap.name)
                if capability_type:
                    capabilities.append(AgentCapability(
                        type=capability_type,
                        description=cap.description,
                        version=cap.version
                    ))
            elif isinstance(cap, str):
                capability_type = map_capability_string(cap)
                if capability_type:
                    capabilities.append(AgentCapability(
                        type=capability_type,
                        description=f"{cap} capability",
                        version="1.0.0"
                    ))
    compatibility = AgentCompatibility(
        agent_version="1.0.0",
        framework_version="1.0.0",
        python_version="3.9+"
    )
    agent_type = AgentType.COLLECTOR
    if hasattr(agent, 'agent_type'):
        agent_type = agent.agent_type
    elif 'strategy' in agent.name.lower():
        agent_type = AgentType.STRATEGY
    elif 'position' in agent.name.lower():
        agent_type = AgentType.TRADER
    return AgentRegistration(
        agent_id=agent.agent_id,
        name=agent.name,
        agent_type=agent_type,
        description=getattr(agent.metadata, 'description', f"{agent.name} agent"),
        capabilities=capabilities,
        provided_services=[],
        required_services=[],
        host="localhost",
        compatibility=compatibility,
        configuration=getattr(agent, 'config', {}) if isinstance(getattr(agent, 'config', {}), dict) else {},
        metadata={
            "version": getattr(agent.metadata, 'version', '1.0.0'),
            "dependencies": getattr(agent.metadata, 'dependencies', [])
        }
    )

def map_capability_string(cap_str: str) -> Optional[AgentCapabilityType]:
    capability_mapping = {
        "market_data_collection": AgentCapabilityType.DATA_COLLECTION,
        "real_time_feeds": AgentCapabilityType.DATA_COLLECTION,
        "historical_data": AgentCapabilityType.DATA_COLLECTION,
        "database_storage": AgentCapabilityType.DATA_STORAGE,
        "health_monitoring": AgentCapabilityType.MONITORING,
        "signal_generation": AgentCapabilityType.SIGNAL_GENERATION,
        "pattern_recognition": AgentCapabilityType.PATTERN_RECOGNITION,
        "technical_analysis": AgentCapabilityType.TECHNICAL_ANALYSIS,
        "candlestick_analysis": AgentCapabilityType.TECHNICAL_ANALYSIS,
        "position_management": AgentCapabilityType.POSITION_MANAGEMENT,
        "order_execution": AgentCapabilityType.ORDER_EXECUTION,
        "risk_management": AgentCapabilityType.RISK_MANAGEMENT
    }
    return capability_mapping.get(cap_str)

async def setup_agent_subscriptions(agent, message_bus: MessageBus):
    """
    Set up message bus subscriptions for an agent.
    """
    from ..models.agent_messages import MessageType
    if not hasattr(agent, 'handle_message'):
        print(f"⚠️  {agent.name} ({agent.agent_type.value}) - No handle_message method, skipping subscriptions")
        return
    print(f"🔗 Setting up subscriptions for {agent.name} ({agent.agent_type.value})")
    if hasattr(agent, 'get_subscription_filters'):
        filters = agent.get_subscription_filters()
        print(f"  📋 Agent provided {len(filters)} custom filters")
        for filter_config in filters:
            await message_bus.subscribe(
                agent_id=agent.agent_id,
                filter=filter_config,
                handler=agent.handle_message,
                is_async=True
            )
    else:
        await setup_default_subscriptions(agent, message_bus)

async def setup_default_subscriptions(agent, message_bus: MessageBus):
    """
    Set up default subscriptions based on agent type.
    """
    from ..models.agent_messages import MessageType
    if agent.agent_type.value == "strategy":
        filter_config = MessageFilter(
            message_types=[MessageType.DATA_MARKET_UPDATE]
        )
        await message_bus.subscribe(
            agent_id=agent.agent_id,
            filter=filter_config,
            handler=agent.handle_message,
            is_async=True
        )
    elif agent.agent_type.value in ["trader", "position_manager"]:
        filter_config = MessageFilter(
            message_types=[MessageType.SIGNAL_GENERATED, MessageType.DATA_MARKET_UPDATE]
        )
        await message_bus.subscribe(
            agent_id=agent.agent_id,
            filter=filter_config,
            handler=agent.handle_message,
            is_async=True
        )
    system_filter = MessageFilter(
        message_types=[MessageType.SYSTEM_HEALTH_CHECK, MessageType.SYSTEM_CONFIG_UPDATE]
    )
    await message_bus.subscribe(
        agent_id=agent.agent_id,
        filter=system_filter,
        handler=agent.handle_message,
        is_async=True
    )

async def monitor_message_bus_stats(message_bus: MessageBus, console: Console):
    """
    Monitor message bus statistics and print them to the console every 10 seconds.
    """
    while True:
        await asyncio.sleep(10)
        stats = await message_bus.get_stats()
        console.print(f"🔍 Message Bus Stats: {stats}") 