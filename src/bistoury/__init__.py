"""
Bistoury: LLM-Driven Cryptocurrency Trading System

An autonomous cryptocurrency trading system that leverages Large Language Models (LLMs) 
to execute intelligent trading strategies on the HyperLiquid exchange.
"""

__version__ = "0.1.0"
__author__ = "Bistoury Team"
__description__ = "LLM-Driven Cryptocurrency Trading System"

# Package-level imports for convenience
from .config import Config
from .logger import get_logger

__all__ = ["Config", "get_logger", "__version__"] 