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
    max_connections: int = Field(default=10, description="Maximum database connections")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    enable_wal: bool = Field(default=True, description="Enable WAL mode for better concurrency")
    cache_size: str = Field(default="1GB", description="Database cache size")


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
    mode: str = Field(default="paper", description="Trading mode: paper, live")
    max_position_size: float = Field(default=1000.0, description="Maximum position size in USD")
    
    # HyperLiquid specific
    hyperliquid_mainnet_url: str = Field(
        default="https://api.hyperliquid.xyz",
        description="HyperLiquid mainnet API URL"
    )
    hyperliquid_testnet_url: str = Field(
        default="https://api.hyperliquid-testnet.xyz",
        description="HyperLiquid testnet API URL"
    )


class DataConfig(BaseModel):
    """Data collection configuration."""
    
    default_pairs: List[str] = Field(default=["BTC", "ETH"])
    default_timeframes: List[str] = Field(default=["1m", "5m", "15m"])
    retention_days: int = Field(default=90, ge=1, le=365)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max log file size in bytes")
    backup_count: int = Field(default=5, description="Number of log backup files")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )


class APIConfig(BaseModel):
    """API configuration."""
    # HyperLiquid
    hyperliquid_private_key: Optional[str] = Field(default=None, description="HyperLiquid private key")
    hyperliquid_wallet_address: Optional[str] = Field(default=None, description="HyperLiquid wallet address")
    
    # OpenAI/LLM
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")


class Config(BaseModel):
    """Main configuration class."""
    
    # Core settings
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=True, description="Enable debug mode")
    
    # Sub-configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hyperliquid: Optional[HyperLiquidConfig] = None
    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
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
            backup_path=os.getenv("DATABASE_BACKUP_PATH", "./backups/"),
            max_connections=int(os.getenv("DATABASE_MAX_CONNECTIONS", "10")),
            connection_timeout=int(os.getenv("DATABASE_CONNECTION_TIMEOUT", "30")),
            enable_wal=os.getenv("DATABASE_ENABLE_WAL", "true").lower() == "true",
            cache_size=os.getenv("DATABASE_CACHE_SIZE", "1GB")
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
            risk_limit_usd=float(os.getenv("RISK_LIMIT_USD", "1000.0")),
            mode=os.getenv("BISTOURY_TRADING_MODE", "paper"),
            max_position_size=float(os.getenv("BISTOURY_MAX_POSITION_SIZE", "1000.0"))
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
        log_file_size_str = os.getenv("LOG_MAX_FILE_SIZE", "10485760")  # 10MB in bytes
        if log_file_size_str.upper().endswith("MB"):
            log_file_size = int(log_file_size_str[:-2]) * 1024 * 1024
        elif log_file_size_str.upper().endswith("KB"):
            log_file_size = int(log_file_size_str[:-2]) * 1024
        else:
            log_file_size = int(log_file_size_str)
            
        logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file_path=os.getenv("LOG_FILE_PATH", "./logs/bistoury.log"),
            max_file_size=log_file_size
        )
        
        # API keys
        api = APIConfig(
            hyperliquid_private_key=os.getenv("HYPERLIQUID_PRIVATE_KEY"),
            hyperliquid_wallet_address=os.getenv("HYPERLIQUID_WALLET_ADDRESS"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        return cls(
            environment=os.getenv("BISTOURY_ENV", "development"),
            debug=os.getenv("BISTOURY_DEBUG", "true").lower() == "true",
            logging=logging,
            database=database,
            llm=llm,
            hyperliquid=hyperliquid,
            trading=trading,
            data=data,
            api=api
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
    
    def validate_for_trading(self) -> bool:
        """Validate configuration for trading operations."""
        if self.trading.mode == "live":
            if not self.api.hyperliquid_private_key:
                raise ValueError("HyperLiquid private key required for live trading")
            if not self.api.hyperliquid_wallet_address:
                raise ValueError("HyperLiquid wallet address required for live trading")
        
        # Ensure database directory exists
        db_path = Path(self.database.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        return True 