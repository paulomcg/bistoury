"""
Paper Trading Configuration Models

Comprehensive configuration management for paper trading engine parameters,
risk controls, and strategy settings.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator

from ..models.market_data import Timeframe


class TradingMode(str, Enum):
    """Trading mode options"""
    HISTORICAL = "historical"  # Replay historical data
    LIVE_PAPER = "live_paper"  # Real-time paper trading
    BACKTEST = "backtest"      # Historical backtesting with analysis


class PositionSizing(str, Enum):
    """Position sizing strategies"""
    FIXED = "fixed"                    # Fixed position size
    CONFIDENCE_BASED = "confidence"    # Size based on signal confidence
    KELLY_CRITERION = "kelly"          # Kelly criterion sizing
    RISK_PARITY = "risk_parity"       # Risk-adjusted sizing


class SignalFiltering(str, Enum):
    """Signal filtering strategies"""
    NONE = "none"                     # Accept all signals
    CONFIDENCE = "confidence"         # Filter by confidence threshold
    QUALITY = "quality"              # Filter by signal quality grade
    CONSENSUS = "consensus"          # Require multiple signal agreement


class TradingParameters(BaseModel):
    """Core trading parameters and rules"""
    
    # Position sizing
    base_position_size: Decimal = Field(
        default=Decimal("100.0"),
        description="Base position size in USD"
    )
    max_position_size: Decimal = Field(
        default=Decimal("1000.0"),
        description="Maximum position size in USD"
    )
    position_sizing: PositionSizing = Field(
        default=PositionSizing.CONFIDENCE_BASED,
        description="Position sizing strategy"
    )
    
    # Signal filtering
    signal_filtering: SignalFiltering = Field(
        default=SignalFiltering.CONFIDENCE,
        description="Signal filtering strategy"
    )
    min_confidence: Decimal = Field(
        default=Decimal("0.65"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Minimum signal confidence to trade"
    )
    min_quality_grade: str = Field(
        default="C",
        description="Minimum signal quality grade (A+ to F)"
    )
    
    # Trading rules
    max_concurrent_positions: int = Field(
        default=3,
        ge=1,
        description="Maximum number of concurrent positions"
    )
    allow_short_positions: bool = Field(
        default=True,
        description="Allow short selling"
    )
    require_confirmation: bool = Field(
        default=False,
        description="Require signal confirmation before trading"
    )
    
    # Timing
    min_signal_age: timedelta = Field(
        default=timedelta(seconds=5),
        description="Minimum signal age before trading"
    )
    max_signal_age: timedelta = Field(
        default=timedelta(minutes=5),
        description="Maximum signal age for trading"
    )


class RiskParameters(BaseModel):
    """Risk management parameters and controls"""
    
    # Account limits
    initial_balance: Decimal = Field(
        default=Decimal("10000.0"),
        gt=Decimal("0.0"),
        description="Initial account balance in USD"
    )
    max_drawdown_percent: Decimal = Field(
        default=Decimal("15.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Maximum drawdown percentage before stopping"
    )
    max_daily_loss_percent: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Maximum daily loss percentage"
    )
    
    # Position limits
    max_position_percent: Decimal = Field(
        default=Decimal("20.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Maximum position size as percentage of balance"
    )
    max_symbol_exposure_percent: Decimal = Field(
        default=Decimal("30.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Maximum exposure to single symbol"
    )
    
    # Stop loss and take profit
    default_stop_loss_percent: Decimal = Field(
        default=Decimal("2.0"),
        ge=Decimal("0.0"),
        description="Default stop loss percentage"
    )
    default_take_profit_percent: Decimal = Field(
        default=Decimal("4.0"),
        ge=Decimal("0.0"),
        description="Default take profit percentage"
    )
    use_trailing_stops: bool = Field(
        default=False,
        description="Use trailing stop losses"
    )
    trailing_stop_percent: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0.0"),
        description="Trailing stop percentage"
    )
    
    # Emergency controls
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker for rapid losses"
    )
    circuit_breaker_loss_percent: Decimal = Field(
        default=Decimal("10.0"),
        ge=Decimal("0.0"),
        description="Loss percentage that triggers circuit breaker"
    )
    circuit_breaker_timeframe: timedelta = Field(
        default=timedelta(hours=1),
        description="Timeframe for circuit breaker calculation"
    )


class HistoricalReplayConfig(BaseModel):
    """Configuration for historical data replay"""
    
    # Date range
    start_date: datetime = Field(
        description="Start date for historical replay"
    )
    end_date: datetime = Field(
        description="End date for historical replay"
    )
    
    # Symbols and timeframes
    symbols: List[str] = Field(
        default=["BTC"],
        description="Symbols to replay"
    )
    timeframes: List[Timeframe] = Field(
        default=[Timeframe.ONE_MINUTE, Timeframe.FIVE_MINUTES, Timeframe.FIFTEEN_MINUTES],
        description="Timeframes to include in replay"
    )
    
    # Replay settings
    replay_speed: float = Field(
        default=1.0,
        gt=0.0,
        description="Replay speed multiplier (1.0 = real-time)"
    )
    include_orderbook: bool = Field(
        default=True,
        description="Include order book data in replay"
    )
    include_trades: bool = Field(
        default=True,
        description="Include trade data in replay"
    )
    
    # Market simulation
    simulate_latency: bool = Field(
        default=True,
        description="Simulate realistic market latency"
    )
    base_latency_ms: int = Field(
        default=50,
        ge=0,
        description="Base latency in milliseconds"
    )
    latency_variance_ms: int = Field(
        default=100,
        ge=0,
        description="Latency variance in milliseconds"
    )


class PaperTradingConfig(BaseModel):
    """Main paper trading configuration"""
    
    # Mode and general settings
    trading_mode: TradingMode = Field(
        default=TradingMode.HISTORICAL,
        description="Trading mode"
    )
    session_name: str = Field(
        default="paper_trading_session",
        description="Name for this trading session"
    )
    
    # Component configurations
    trading_params: TradingParameters = Field(
        default_factory=TradingParameters,
        description="Trading parameters"
    )
    risk_params: RiskParameters = Field(
        default_factory=RiskParameters,
        description="Risk management parameters"
    )
    historical_config: Optional[HistoricalReplayConfig] = Field(
        default=None,
        description="Historical replay configuration"
    )
    
    # Strategy configuration
    enabled_strategies: Set[str] = Field(
        default={"candlestick_strategy"},
        description="Enabled trading strategies"
    )
    strategy_weights: Dict[str, Decimal] = Field(
        default={"candlestick_strategy": Decimal("1.0")},
        description="Strategy weights for signal aggregation"
    )
    
    # Performance and monitoring
    performance_reporting_interval: timedelta = Field(
        default=timedelta(minutes=15),
        description="Performance reporting interval"
    )
    enable_detailed_logging: bool = Field(
        default=True,
        description="Enable detailed trade logging"
    )
    save_state_interval: timedelta = Field(
        default=timedelta(minutes=5),
        description="State saving interval"
    )
    
    # Database settings
    database_path: Optional[str] = Field(
        default=None,
        description="Custom database path for data source"
    )
    results_database_path: Optional[str] = Field(
        default=None,
        description="Database path for saving results"
    )
    
    @validator('historical_config')
    def validate_historical_config(cls, v, values):
        """Validate historical config when in historical mode"""
        if values.get('trading_mode') == TradingMode.HISTORICAL and v is None:
            raise ValueError("Historical config required for historical trading mode")
        return v
    
    @validator('strategy_weights')
    def validate_strategy_weights(cls, v, values):
        """Validate strategy weights match enabled strategies"""
        enabled = values.get('enabled_strategies', set())
        if enabled and set(v.keys()) != enabled:
            raise ValueError("Strategy weights must match enabled strategies")
        return v
    
    def calculate_position_size(self, base_size: Decimal, confidence: Decimal) -> Decimal:
        """Calculate position size based on strategy and confidence"""
        if self.trading_params.position_sizing == PositionSizing.FIXED:
            return base_size
        elif self.trading_params.position_sizing == PositionSizing.CONFIDENCE_BASED:
            # Scale position by confidence
            scaled_size = base_size * confidence
            return min(scaled_size, self.trading_params.max_position_size)
        else:
            # TODO: Implement Kelly criterion and risk parity
            return base_size
    
    def should_trade_signal(self, confidence: Decimal, quality_grade: str) -> bool:
        """Determine if signal meets trading criteria"""
        if self.trading_params.signal_filtering == SignalFiltering.NONE:
            return True
        elif self.trading_params.signal_filtering == SignalFiltering.CONFIDENCE:
            return confidence >= self.trading_params.min_confidence
        elif self.trading_params.signal_filtering == SignalFiltering.QUALITY:
            # Simple grade comparison (A+ > A > B+ > B > C+ > C > D+ > D > F)
            grade_values = {
                "A+": 10, "A": 9, "B+": 8, "B": 7, "C+": 6, 
                "C": 5, "D+": 4, "D": 3, "F": 0
            }
            return grade_values.get(quality_grade, 0) >= grade_values.get(self.trading_params.min_quality_grade, 5)
        else:
            # Default to confidence filtering
            return confidence >= self.trading_params.min_confidence 