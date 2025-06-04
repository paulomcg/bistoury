"""
Trading Strategy Models

This module contains Pydantic models for trading strategies and performance tracking:
- StrategyOutput: Strategy decision and recommendation output
- StrategyPerformance: Performance tracking and analytics
- SignalPerformance: Individual signal performance tracking
- StrategyMetadata: Strategy configuration and metadata
- BacktestResult: Backtesting results and metrics

All models are designed for strategy development, testing, and live trading
with comprehensive performance tracking and analytics.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict
from enum import Enum

from .market_data import Timeframe
from .signals import TradingSignal, SignalDirection, SignalType, AnalysisContext


class StrategyType(str, Enum):
    """Strategy type enumeration."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    GRID = "grid"
    DCA = "dca"  # Dollar Cost Averaging
    MULTI_STRATEGY = "multi_strategy"


class StrategyStatus(str, Enum):
    """Strategy status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    BACKTESTING = "backtesting"
    OPTIMIZING = "optimizing"
    DEPLOYING = "deploying"


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PerformanceMetric(str, Enum):
    """Performance metric enumeration."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    VAR_95 = "var_95"
    VOLATILITY = "volatility"
    ALPHA = "alpha"
    BETA = "beta"


class StrategyOutput(BaseModel):
    """
    Strategy decision and recommendation output.
    
    Represents the complete output from a trading strategy including
    signals, reasoning, confidence levels, and risk assessment.
    """
    
    output_id: str = Field(
        ...,
        description="Unique output identifier"
    )
    strategy_name: str = Field(
        ...,
        description="Name of the strategy generating this output"
    )
    strategy_version: str = Field(
        ...,
        description="Version of the strategy"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol for this output"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output generation timestamp"
    )
    timeframe: Timeframe = Field(
        ...,
        description="Primary timeframe for the strategy"
    )
    signals: List[TradingSignal] = Field(
        default_factory=list,
        description="Generated trading signals"
    )
    primary_signal: Optional[TradingSignal] = Field(
        None,
        description="Primary/strongest signal from the strategy"
    )
    confidence: Decimal = Field(
        ...,
        description="Overall strategy confidence (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    risk_score: Decimal = Field(
        ...,
        description="Risk assessment score (0-100, higher = riskier)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    market_conditions: AnalysisContext = Field(
        ...,
        description="Market analysis context used for decisions"
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of the strategy decision"
    )
    execution_instructions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific execution instructions and parameters"
    )
    risk_management: Dict[str, Any] = Field(
        default_factory=dict,
        description="Risk management parameters and rules"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional strategy-specific metadata"
    )
    is_valid: bool = Field(
        default=True,
        description="Whether the output is valid for execution"
    )
    expiry: Optional[datetime] = Field(
        None,
        description="Output expiry timestamp"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def signal_count(self) -> int:
        """Get number of signals generated."""
        return len(self.signals)
    
    @computed_field
    @property
    def bullish_signal_count(self) -> int:
        """Count bullish signals."""
        return len([
            s for s in self.signals 
            if s.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        ])
    
    @computed_field
    @property
    def bearish_signal_count(self) -> int:
        """Count bearish signals."""
        return len([
            s for s in self.signals 
            if s.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]
        ])
    
    @computed_field
    @property
    def neutral_signal_count(self) -> int:
        """Count neutral signals."""
        return len([
            s for s in self.signals 
            if s.direction == SignalDirection.HOLD
        ])
    
    @computed_field
    @property
    def average_signal_confidence(self) -> Decimal:
        """Calculate average confidence of all signals."""
        if not self.signals:
            return Decimal('0')
        
        total = sum(signal.confidence for signal in self.signals)
        return total / len(self.signals)
    
    @computed_field
    @property
    def risk_level(self) -> RiskLevel:
        """Categorize risk level based on risk score."""
        if self.risk_score < 20:
            return RiskLevel.VERY_LOW
        elif self.risk_score < 40:
            return RiskLevel.LOW
        elif self.risk_score < 60:
            return RiskLevel.MEDIUM
        elif self.risk_score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    @computed_field
    @property
    def overall_direction(self) -> SignalDirection:
        """Determine overall strategy direction."""
        if self.primary_signal:
            return self.primary_signal.direction
        
        if not self.signals:
            return SignalDirection.HOLD
        
        # Calculate weighted average direction
        bullish_weight = sum(
            s.signal_score for s in self.signals 
            if s.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]
        )
        bearish_weight = sum(
            s.signal_score for s in self.signals 
            if s.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]
        )
        
        if bullish_weight > bearish_weight * Decimal('1.2'):
            return SignalDirection.BUY
        elif bearish_weight > bullish_weight * Decimal('1.2'):
            return SignalDirection.SELL
        else:
            return SignalDirection.HOLD
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if output has expired."""
        if not self.expiry:
            return False
        return datetime.now(timezone.utc) > self.expiry
    
    @computed_field
    @property
    def age(self) -> timedelta:
        """Calculate output age."""
        return datetime.now(timezone.utc) - self.timestamp
    
    def add_signal(self, signal: TradingSignal) -> None:
        """Add a signal to the output."""
        self.signals.append(signal)
        
        # Update primary signal if this is stronger
        if not self.primary_signal or signal.signal_score > self.primary_signal.signal_score:
            self.primary_signal = signal
    
    def set_execution_parameter(self, key: str, value: Any) -> None:
        """Set execution parameter."""
        self.execution_instructions[key] = value
    
    def set_risk_parameter(self, key: str, value: Any) -> None:
        """Set risk management parameter."""
        self.risk_management[key] = value
    
    def invalidate(self, reason: str = "") -> None:
        """Mark output as invalid."""
        self.is_valid = False
        if reason:
            self.metadata["invalidation_reason"] = reason
            self.metadata["invalidation_time"] = datetime.now(timezone.utc).isoformat()


class SignalPerformance(BaseModel):
    """
    Individual signal performance tracking.
    
    Tracks the performance of individual trading signals including
    outcomes, accuracy, and profitability metrics.
    """
    
    signal_id: str = Field(
        ...,
        description="Signal identifier being tracked"
    )
    symbol: str = Field(
        ...,
        description="Trading symbol"
    )
    signal_direction: SignalDirection = Field(
        ...,
        description="Original signal direction"
    )
    signal_price: Decimal = Field(
        ...,
        description="Price when signal was generated",
        gt=Decimal('0')
    )
    signal_timestamp: datetime = Field(
        ...,
        description="Signal generation timestamp"
    )
    signal_confidence: Decimal = Field(
        ...,
        description="Original signal confidence",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    entry_price: Optional[Decimal] = Field(
        None,
        description="Actual entry price if position taken",
        gt=Decimal('0')
    )
    exit_price: Optional[Decimal] = Field(
        None,
        description="Exit price if position closed",
        gt=Decimal('0')
    )
    entry_timestamp: Optional[datetime] = Field(
        None,
        description="Position entry timestamp"
    )
    exit_timestamp: Optional[datetime] = Field(
        None,
        description="Position exit timestamp"
    )
    position_size: Optional[Decimal] = Field(
        None,
        description="Position size taken",
        ge=Decimal('0')
    )
    realized_pnl: Optional[Decimal] = Field(
        None,
        description="Realized PnL from the signal"
    )
    max_favorable_excursion: Optional[Decimal] = Field(
        None,
        description="Maximum favorable price movement",
        ge=Decimal('0')
    )
    max_adverse_excursion: Optional[Decimal] = Field(
        None,
        description="Maximum adverse price movement",
        ge=Decimal('0')
    )
    signal_outcome: Optional[Literal["win", "loss", "breakeven", "timeout"]] = Field(
        None,
        description="Final outcome of the signal"
    )
    accuracy_score: Optional[Decimal] = Field(
        None,
        description="Accuracy score based on direction prediction (0-100)",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    is_completed: bool = Field(
        default=False,
        description="Whether signal tracking is complete"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def return_percentage(self) -> Optional[Decimal]:
        """Calculate return percentage if position closed."""
        if not self.entry_price or not self.exit_price:
            return None
        
        if self.signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            return ((self.exit_price - self.entry_price) / self.entry_price) * Decimal('100')
        else:  # Short position
            return ((self.entry_price - self.exit_price) / self.entry_price) * Decimal('100')
    
    @computed_field
    @property
    def holding_period(self) -> Optional[timedelta]:
        """Calculate holding period if position closed."""
        if not self.entry_timestamp or not self.exit_timestamp:
            return None
        return self.exit_timestamp - self.entry_timestamp
    
    @computed_field
    @property
    def signal_delay(self) -> Optional[timedelta]:
        """Calculate delay between signal and entry."""
        if not self.entry_timestamp:
            return None
        return self.entry_timestamp - self.signal_timestamp
    
    @computed_field
    @property
    def was_profitable(self) -> Optional[bool]:
        """Check if signal was profitable."""
        if self.realized_pnl is None:
            return None
        return self.realized_pnl > 0
    
    @computed_field
    @property
    def slippage(self) -> Optional[Decimal]:
        """Calculate slippage from signal price to entry price."""
        if not self.entry_price:
            return None
        
        return abs(self.entry_price - self.signal_price) / self.signal_price * Decimal('100')
    
    def complete_signal(
        self,
        exit_price: Decimal,
        exit_timestamp: Optional[datetime] = None,
        outcome: Literal["win", "loss", "breakeven", "timeout"] = "win"
    ) -> None:
        """Mark signal as completed with final results."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp or datetime.now(timezone.utc)
        self.signal_outcome = outcome
        self.is_completed = True
        
        # Calculate realized PnL if position size is known
        if self.position_size and self.entry_price:
            if self.signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                self.realized_pnl = (exit_price - self.entry_price) * self.position_size
            else:
                self.realized_pnl = (self.entry_price - exit_price) * self.position_size
    
    def update_excursions(self, current_price: Decimal) -> None:
        """Update maximum favorable and adverse excursions."""
        if not self.entry_price:
            return
        
        if self.signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            # For long positions
            favorable = current_price - self.entry_price
            adverse = self.entry_price - current_price
        else:
            # For short positions
            favorable = self.entry_price - current_price
            adverse = current_price - self.entry_price
        
        if favorable > 0:
            if self.max_favorable_excursion is None or favorable > self.max_favorable_excursion:
                self.max_favorable_excursion = favorable
        
        if adverse > 0:
            if self.max_adverse_excursion is None or adverse > self.max_adverse_excursion:
                self.max_adverse_excursion = adverse


class StrategyPerformance(BaseModel):
    """
    Strategy performance tracking and analytics.
    
    Comprehensive performance metrics and analytics for trading strategies
    including returns, risk metrics, and signal accuracy tracking.
    """
    
    strategy_name: str = Field(
        ...,
        description="Strategy name"
    )
    strategy_version: str = Field(
        ...,
        description="Strategy version"
    )
    start_date: datetime = Field(
        ...,
        description="Performance tracking start date"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="Performance tracking end date (None if ongoing)"
    )
    total_signals: int = Field(
        default=0,
        description="Total number of signals generated",
        ge=0
    )
    executed_signals: int = Field(
        default=0,
        description="Number of signals actually executed",
        ge=0
    )
    winning_signals: int = Field(
        default=0,
        description="Number of profitable signals",
        ge=0
    )
    losing_signals: int = Field(
        default=0,
        description="Number of losing signals",
        ge=0
    )
    total_return: Decimal = Field(
        default=Decimal('0'),
        description="Total return percentage"
    )
    total_pnl: Decimal = Field(
        default=Decimal('0'),
        description="Total realized PnL"
    )
    max_drawdown: Decimal = Field(
        default=Decimal('0'),
        description="Maximum drawdown percentage",
        ge=Decimal('0')
    )
    volatility: Decimal = Field(
        default=Decimal('0'),
        description="Return volatility (annualized)",
        ge=Decimal('0')
    )
    sharpe_ratio: Optional[Decimal] = Field(
        None,
        description="Sharpe ratio"
    )
    win_rate: Decimal = Field(
        default=Decimal('0'),
        description="Win rate percentage",
        ge=Decimal('0'),
        le=Decimal('100')
    )
    profit_factor: Optional[Decimal] = Field(
        None,
        description="Profit factor (gross profit / gross loss)"
    )
    average_win: Decimal = Field(
        default=Decimal('0'),
        description="Average winning trade PnL"
    )
    average_loss: Decimal = Field(
        default=Decimal('0'),
        description="Average losing trade PnL"
    )
    largest_win: Decimal = Field(
        default=Decimal('0'),
        description="Largest winning trade PnL"
    )
    largest_loss: Decimal = Field(
        default=Decimal('0'),
        description="Largest losing trade PnL"
    )
    consecutive_wins: int = Field(
        default=0,
        description="Current consecutive wins",
        ge=0
    )
    consecutive_losses: int = Field(
        default=0,
        description="Current consecutive losses",
        ge=0
    )
    max_consecutive_wins: int = Field(
        default=0,
        description="Maximum consecutive wins",
        ge=0
    )
    max_consecutive_losses: int = Field(
        default=0,
        description="Maximum consecutive losses",
        ge=0
    )
    signal_performances: List[SignalPerformance] = Field(
        default_factory=list,
        description="Individual signal performance records"
    )
    daily_returns: List[Decimal] = Field(
        default_factory=list,
        description="Daily return series for risk calculations"
    )
    equity_curve: List[Decimal] = Field(
        default_factory=list,
        description="Equity curve data points"
    )
    performance_metrics: Dict[PerformanceMetric, Decimal] = Field(
        default_factory=dict,
        description="Additional performance metrics"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def execution_rate(self) -> Decimal:
        """Calculate signal execution rate."""
        if self.total_signals == 0:
            return Decimal('0')
        return (Decimal(str(self.executed_signals)) / Decimal(str(self.total_signals))) * Decimal('100')
    
    @computed_field
    @property
    def expectancy(self) -> Decimal:
        """Calculate expectancy per trade."""
        if self.executed_signals == 0:
            return Decimal('0')
        
        return (self.win_rate / Decimal('100') * self.average_win) + \
               ((Decimal('100') - self.win_rate) / Decimal('100') * self.average_loss)
    
    @computed_field
    @property
    def calmar_ratio(self) -> Optional[Decimal]:
        """Calculate Calmar ratio."""
        if self.max_drawdown == 0:
            return None
        return abs(self.total_return / self.max_drawdown)
    
    @computed_field
    @property
    def recovery_factor(self) -> Optional[Decimal]:
        """Calculate recovery factor."""
        if self.max_drawdown == 0:
            return None
        return abs(self.total_pnl / self.max_drawdown)
    
    @computed_field
    @property
    def trade_frequency(self) -> Optional[Decimal]:
        """Calculate average trades per day."""
        if not self.end_date:
            period_days = (datetime.now(timezone.utc) - self.start_date).days
        else:
            period_days = (self.end_date - self.start_date).days
        
        if period_days == 0:
            return None
        
        return Decimal(str(self.executed_signals)) / Decimal(str(period_days))
    
    @computed_field
    @property
    def current_streak_type(self) -> Optional[Literal["winning", "losing"]]:
        """Get current streak type."""
        if self.consecutive_wins > 0:
            return "winning"
        elif self.consecutive_losses > 0:
            return "losing"
        else:
            return None
    
    def add_signal_performance(self, signal_perf: SignalPerformance) -> None:
        """Add a signal performance record."""
        self.signal_performances.append(signal_perf)
        self.total_signals += 1
        
        if signal_perf.entry_price is not None:
            self.executed_signals += 1
        
        # Update statistics if signal is completed
        if signal_perf.is_completed and signal_perf.realized_pnl is not None:
            self.total_pnl += signal_perf.realized_pnl
            
            if signal_perf.was_profitable:
                self.winning_signals += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
                
                if signal_perf.realized_pnl > self.largest_win:
                    self.largest_win = signal_perf.realized_pnl
            else:
                self.losing_signals += 1
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                
                if signal_perf.realized_pnl < self.largest_loss:
                    self.largest_loss = signal_perf.realized_pnl
        
        # Recalculate key metrics
        self._update_calculated_metrics()
    
    def _update_calculated_metrics(self) -> None:
        """Update calculated performance metrics."""
        if self.executed_signals > 0:
            self.win_rate = (Decimal(str(self.winning_signals)) / Decimal(str(self.executed_signals))) * Decimal('100')
        
        # Update average win/loss
        if self.winning_signals > 0:
            total_wins = sum(
                sp.realized_pnl for sp in self.signal_performances 
                if sp.realized_pnl and sp.realized_pnl > 0
            )
            self.average_win = total_wins / self.winning_signals
        
        if self.losing_signals > 0:
            total_losses = sum(
                abs(sp.realized_pnl) for sp in self.signal_performances 
                if sp.realized_pnl and sp.realized_pnl < 0
            )
            self.average_loss = -total_losses / self.losing_signals
        
        # Update profit factor
        if self.losing_signals > 0 and self.average_loss != 0:
            gross_profit = self.average_win * self.winning_signals
            gross_loss = abs(self.average_loss) * self.losing_signals
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    
    def add_daily_return(self, daily_return: Decimal) -> None:
        """Add a daily return for risk calculations."""
        self.daily_returns.append(daily_return)
        
        # Update equity curve
        if not self.equity_curve:
            self.equity_curve.append(Decimal('100'))  # Start at 100
        else:
            last_equity = self.equity_curve[-1]
            new_equity = last_equity * (Decimal('1') + daily_return / Decimal('100'))
            self.equity_curve.append(new_equity)
        
        # Recalculate volatility and other metrics
        self._calculate_risk_metrics()
    
    def _calculate_risk_metrics(self) -> None:
        """Calculate risk metrics from daily returns."""
        if len(self.daily_returns) < 2:
            return
        
        # Calculate volatility (annualized)
        import statistics
        daily_vol = Decimal(str(statistics.stdev([float(r) for r in self.daily_returns])))
        self.volatility = daily_vol * Decimal('252').sqrt()  # Annualized
        
        # Calculate max drawdown from equity curve
        if len(self.equity_curve) >= 2:
            peak = self.equity_curve[0]
            max_dd = Decimal('0')
            
            for equity in self.equity_curve[1:]:
                if equity > peak:
                    peak = equity
                else:
                    drawdown = (peak - equity) / peak * Decimal('100')
                    max_dd = max(max_dd, drawdown)
            
            self.max_drawdown = max_dd
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        if self.volatility > 0:
            avg_return = sum(self.daily_returns) / len(self.daily_returns) if self.daily_returns else Decimal('0')
            annualized_return = avg_return * Decimal('252')
            self.sharpe_ratio = annualized_return / self.volatility
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of key performance metrics."""
        return {
            "total_return": float(self.total_return),
            "total_pnl": float(self.total_pnl),
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor) if self.profit_factor else None,
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else None,
            "max_drawdown": float(self.max_drawdown),
            "expectancy": float(self.expectancy),
            "total_signals": self.total_signals,
            "executed_signals": self.executed_signals,
            "execution_rate": float(self.execution_rate),
            "volatility": float(self.volatility),
            "calmar_ratio": float(self.calmar_ratio) if self.calmar_ratio else None
        }


class StrategyMetadata(BaseModel):
    """
    Strategy configuration and metadata.
    
    Contains strategy configuration, parameters, and descriptive information
    for strategy management and version control.
    """
    
    strategy_name: str = Field(
        ...,
        description="Unique strategy name"
    )
    version: str = Field(
        ...,
        description="Strategy version"
    )
    strategy_type: StrategyType = Field(
        ...,
        description="Strategy type classification"
    )
    description: str = Field(
        ...,
        description="Strategy description"
    )
    author: str = Field(
        ...,
        description="Strategy author/developer"
    )
    created_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Strategy creation date"
    )
    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification date"
    )
    status: StrategyStatus = Field(
        default=StrategyStatus.INACTIVE,
        description="Current strategy status"
    )
    supported_symbols: List[str] = Field(
        default_factory=list,
        description="List of supported trading symbols"
    )
    supported_timeframes: List[Timeframe] = Field(
        default_factory=list,
        description="List of supported timeframes"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters and configuration"
    )
    risk_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Risk management parameters"
    )
    performance_targets: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Performance targets and thresholds"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Required indicators, data, or services"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Strategy tags for categorization"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    def update_parameter(self, key: str, value: Any) -> None:
        """Update a strategy parameter."""
        self.parameters[key] = value
        self.last_modified = datetime.now(timezone.utc)
    
    def update_risk_parameter(self, key: str, value: Any) -> None:
        """Update a risk parameter."""
        self.risk_parameters[key] = value
        self.last_modified = datetime.now(timezone.utc)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the strategy."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the strategy."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def set_status(self, status: StrategyStatus) -> None:
        """Update strategy status."""
        self.status = status
        self.last_modified = datetime.now(timezone.utc)


class BacktestResult(BaseModel):
    """
    Backtesting results and metrics.
    
    Contains comprehensive backtesting results including performance metrics,
    trade details, and statistical analysis of strategy performance.
    """
    
    backtest_id: str = Field(
        ...,
        description="Unique backtest identifier"
    )
    strategy_name: str = Field(
        ...,
        description="Strategy name tested"
    )
    strategy_version: str = Field(
        ...,
        description="Strategy version tested"
    )
    symbol: str = Field(
        ...,
        description="Symbol tested"
    )
    start_date: datetime = Field(
        ...,
        description="Backtest start date"
    )
    end_date: datetime = Field(
        ...,
        description="Backtest end date"
    )
    initial_capital: Decimal = Field(
        ...,
        description="Starting capital for backtest",
        gt=Decimal('0')
    )
    final_capital: Decimal = Field(
        ...,
        description="Ending capital from backtest",
        gt=Decimal('0')
    )
    performance: StrategyPerformance = Field(
        ...,
        description="Detailed performance metrics"
    )
    parameters_used: Dict[str, Any] = Field(
        ...,
        description="Strategy parameters used in backtest"
    )
    market_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market conditions during backtest period"
    )
    benchmark_return: Optional[Decimal] = Field(
        None,
        description="Benchmark return for comparison"
    )
    excess_return: Optional[Decimal] = Field(
        None,
        description="Return above benchmark"
    )
    alpha: Optional[Decimal] = Field(
        None,
        description="Alpha relative to benchmark"
    )
    beta: Optional[Decimal] = Field(
        None,
        description="Beta relative to benchmark"
    )
    correlation: Optional[Decimal] = Field(
        None,
        description="Correlation with benchmark"
    )
    information_ratio: Optional[Decimal] = Field(
        None,
        description="Information ratio"
    )
    tracking_error: Optional[Decimal] = Field(
        None,
        description="Tracking error vs benchmark"
    )
    run_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When backtest was run"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    )
    
    @computed_field
    @property
    def total_return_pct(self) -> Decimal:
        """Calculate total return percentage."""
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * Decimal('100')
    
    @computed_field
    @property
    def annualized_return(self) -> Decimal:
        """Calculate annualized return."""
        days = (self.end_date - self.start_date).days
        if days == 0:
            return Decimal('0')
        
        years = Decimal(str(days)) / Decimal('365.25')
        return ((self.final_capital / self.initial_capital) ** (Decimal('1') / years) - Decimal('1')) * Decimal('100')
    
    @computed_field
    @property
    def backtest_duration(self) -> timedelta:
        """Get backtest duration."""
        return self.end_date - self.start_date
    
    def compare_to_benchmark(self) -> Dict[str, Optional[Decimal]]:
        """Compare performance to benchmark."""
        if self.benchmark_return is None:
            return {
                "excess_return": None,
                "information_ratio": None,
                "relative_performance": None
            }
        
        excess = self.total_return_pct - self.benchmark_return
        relative_perf = (self.total_return_pct / self.benchmark_return - Decimal('1')) * Decimal('100') if self.benchmark_return != 0 else None
        
        return {
            "excess_return": excess,
            "information_ratio": self.information_ratio,
            "relative_performance": relative_perf
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the backtest."""
        return {
            "backtest_id": self.backtest_id,
            "strategy": self.strategy_name,
            "version": self.strategy_version,
            "symbol": self.symbol,
            "duration_days": self.backtest_duration.days,
            "initial_capital": float(self.initial_capital),
            "final_capital": float(self.final_capital),
            "total_return_pct": float(self.total_return_pct),
            "annualized_return": float(self.annualized_return),
            "max_drawdown": float(self.performance.max_drawdown),
            "sharpe_ratio": float(self.performance.sharpe_ratio) if self.performance.sharpe_ratio else None,
            "win_rate": float(self.performance.win_rate),
            "profit_factor": float(self.performance.profit_factor) if self.performance.profit_factor else None,
            "total_trades": self.performance.executed_signals,
            "benchmark_comparison": self.compare_to_benchmark()
        } 