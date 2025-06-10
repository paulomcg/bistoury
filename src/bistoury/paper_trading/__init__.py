"""
Bistoury Paper Trading System

Mathematical paper trading engine for strategy testing without LLM decision layer.
Orchestrates data flow between historical data, signal generation, and trade execution.
"""

from .engine import PaperTradingEngine
from .config import PaperTradingConfig, TradingParameters, RiskParameters

__all__ = [
    "PaperTradingEngine",
    "PaperTradingConfig", 
    "TradingParameters",
    "RiskParameters",
] 