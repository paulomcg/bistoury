"""
Configuration management for Bistoury trading system.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    path: str = Field(default="./data/bistoury.db")
    backup_path: str = Field(default="./backups/")


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama2")


class HyperLiquidConfig(BaseModel):
    """HyperLiquid exchange configuration."""
    
    api_key: str
    secret_key: str
    testnet: bool = Field(default=True)


class TradingConfig(BaseModel):
    """Trading configuration parameters."""
    
    default_position_size_pct: float = Field(default=10.0, ge=0.1, le=100.0)
    max_positions: int = Field(default=3, ge=1, le=20)
    stop_loss_pct: float = Field(default=2.0, ge=0.1, le=50.0)
    take_profit_pct: float = Field(default=5.0, ge=0.1, le=100.0)
    risk_limit_usd: float = Field(default=1000.0, ge=10.0)


class DataConfig(BaseModel):
    """Data collection configuration."""
    
    default_pairs: List[str] = Field(default=["BTC", "ETH"])
    default_timeframes: List[str] = Field(default=["1m", "5m", "15m"])
    retention_days: int = Field(default=90, ge=1, le=365)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO")
    file_path: str = Field(default="./logs/bistoury.log")
    max_size: str = Field(default="10MB")
    backup_count: int = Field(default=5)


class Config(BaseModel):
    """Main configuration class."""
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hyperliquid: Optional[HyperLiquidConfig] = None
    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load from .env file in current directory
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path)
        
        # Database config
        database = DatabaseConfig(
            path=os.getenv("DATABASE_PATH", "./data/bistoury.db"),
            backup_path=os.getenv("DATABASE_BACKUP_PATH", "./backups/")
        )
        
        # LLM config
        llm = LLMConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama2")
        )
        
        # HyperLiquid config (optional)
        hyperliquid = None
        if os.getenv("HYPERLIQUID_API_KEY") and os.getenv("HYPERLIQUID_SECRET_KEY"):
            hyperliquid = HyperLiquidConfig(
                api_key=os.getenv("HYPERLIQUID_API_KEY"),
                secret_key=os.getenv("HYPERLIQUID_SECRET_KEY"),
                testnet=os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"
            )
        
        # Trading config
        trading = TradingConfig(
            default_position_size_pct=float(os.getenv("DEFAULT_POSITION_SIZE_PCT", "10.0")),
            max_positions=int(os.getenv("MAX_POSITIONS", "3")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "2.0")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "5.0")),
            risk_limit_usd=float(os.getenv("RISK_LIMIT_USD", "1000.0"))
        )
        
        # Data config
        pairs = os.getenv("DEFAULT_PAIRS", "BTC,ETH").split(",")
        timeframes = os.getenv("DEFAULT_TIMEFRAMES", "1m,5m,15m").split(",")
        data = DataConfig(
            default_pairs=[p.strip() for p in pairs],
            default_timeframes=[t.strip() for t in timeframes],
            retention_days=int(os.getenv("DATA_RETENTION_DAYS", "90"))
        )
        
        # Logging config
        logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file_path=os.getenv("LOG_FILE_PATH", "./logs/bistoury.log"),
            max_size=os.getenv("LOG_MAX_SIZE", "10MB"),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
        
        return cls(
            database=database,
            llm=llm,
            hyperliquid=hyperliquid,
            trading=trading,
            data=data,
            logging=logging
        )
    
    def validate_llm_keys(self) -> bool:
        """Check if at least one LLM API key is configured."""
        return any([
            self.llm.openai_api_key,
            self.llm.anthropic_api_key,
            self.llm.google_api_key
        ])
    
    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers based on configured API keys."""
        providers = []
        if self.llm.openai_api_key:
            providers.append("openai")
        if self.llm.anthropic_api_key:
            providers.append("anthropic")
        if self.llm.google_api_key:
            providers.append("google")
        providers.append("ollama")  # Always available if Ollama is running
        return providers 