"""
Logging infrastructure for Bistoury trading system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for file output."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add extra fields for trading context
        if hasattr(record, 'symbol'):
            record.msg = f"[{record.symbol}] {record.msg}"
        if hasattr(record, 'strategy'):
            record.msg = f"[{record.strategy}] {record.msg}"
        if hasattr(record, 'trade_id'):
            record.msg = f"[TRADE:{record.trade_id}] {record.msg}"
        
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: str = "10MB",
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file rotation and optional console output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        console_format = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max_size
        size_bytes = _parse_size(max_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        file_format = StructuredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger instance with default configuration.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Logger instance
    """
    return setup_logger(
        name=name,
        level=level,
        log_file="./logs/bistoury.log",
        console_output=True
    )


def get_trade_logger() -> logging.Logger:
    """Get specialized logger for trading operations."""
    return setup_logger(
        name="bistoury.trading",
        level="INFO",
        log_file="./logs/trading.log",
        console_output=True
    )


def get_data_logger() -> logging.Logger:
    """Get specialized logger for data collection."""
    return setup_logger(
        name="bistoury.data",
        level="INFO", 
        log_file="./logs/data.log",
        console_output=False  # Less noisy for data collection
    )


def get_strategy_logger() -> logging.Logger:
    """Get specialized logger for strategy operations."""
    return setup_logger(
        name="bistoury.strategy",
        level="INFO",
        log_file="./logs/strategy.log",
        console_output=True
    )


def _parse_size(size_str: str) -> int:
    """
    Parse size string (e.g., '10MB', '1GB') to bytes.
    
    Args:
        size_str: Size string with unit
    
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    # Extract number and unit
    size_map = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    for unit, multiplier in size_map.items():
        if size_str.endswith(unit):
            number = size_str[:-len(unit)].strip()
            try:
                return int(float(number) * multiplier)
            except ValueError:
                pass
    
    # Default to bytes if no unit or invalid format
    try:
        return int(size_str)
    except ValueError:
        return 10 * 1024 * 1024  # Default 10MB


class TradingLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for trading operations with extra context."""
    
    def __init__(self, logger: logging.Logger, extra: dict):
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: dict) -> tuple:
        # Add trading context to log records
        extra = self.extra.copy()
        extra.update(kwargs.get('extra', {}))
        kwargs['extra'] = extra
        return msg, kwargs


def get_trading_adapter(
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    trade_id: Optional[str] = None
) -> TradingLoggerAdapter:
    """
    Get a trading logger adapter with context.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC')
        strategy: Strategy name
        trade_id: Trade identifier
    
    Returns:
        Logger adapter with trading context
    """
    logger = get_trade_logger()
    extra = {}
    
    if symbol:
        extra['symbol'] = symbol
    if strategy:
        extra['strategy'] = strategy
    if trade_id:
        extra['trade_id'] = trade_id
    
    return TradingLoggerAdapter(logger, extra) 