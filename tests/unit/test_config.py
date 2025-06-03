"""
Unit tests for configuration management.
"""

import os
from unittest.mock import patch

import pytest

from bistoury.config import Config, TradingConfig, DatabaseConfig


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_config_loads_defaults(self) -> None:
        """Test that configuration loads with default values."""
        config = Config()
        
        assert config.database.path == "./data/bistoury.db"
        assert config.trading.default_position_size_pct == 10.0
        assert config.data.default_pairs == ["BTC", "ETH"]
        assert config.logging.level == "INFO"
    
    def test_config_loads_from_env(self, mock_env_vars: dict) -> None:
        """Test that configuration loads from environment variables."""
        config = Config.load_from_env()
        
        assert config.database.path == "./test_data/test.db"
        assert config.trading.default_position_size_pct == 5.0
        assert config.trading.max_positions == 2
        assert config.data.default_pairs == ["BTC", "ETH"]
        assert config.logging.level == "DEBUG"
    
    def test_hyperliquid_config_optional(self) -> None:
        """Test that HyperLiquid config is optional."""
        config = Config()
        assert config.hyperliquid is None
    
    def test_hyperliquid_config_loads_when_keys_present(self) -> None:
        """Test that HyperLiquid config loads when API keys are present."""
        env_vars = {
            "HYPERLIQUID_API_KEY": "test_api_key",
            "HYPERLIQUID_SECRET_KEY": "test_secret_key",
            "HYPERLIQUID_TESTNET": "false"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = Config.load_from_env()
            
            assert config.hyperliquid is not None
            assert config.hyperliquid.api_key == "test_api_key"
            assert config.hyperliquid.secret_key == "test_secret_key"
            assert config.hyperliquid.testnet is False
    
    def test_llm_provider_detection(self) -> None:
        """Test LLM provider detection based on API keys."""
        env_vars = {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = Config.load_from_env()
            providers = config.get_available_llm_providers()
            
            assert "openai" in providers
            assert "anthropic" in providers
            assert "ollama" in providers  # Always available
            assert config.validate_llm_keys() is True
    
    def test_trading_config_validation(self) -> None:
        """Test trading configuration validation."""
        # Valid config
        trading_config = TradingConfig(
            default_position_size_pct=10.0,
            max_positions=3,
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            risk_limit_usd=1000.0
        )
        assert trading_config.default_position_size_pct == 10.0
        
        # Invalid config should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            TradingConfig(default_position_size_pct=150.0)  # > 100%
        
        with pytest.raises(Exception):
            TradingConfig(max_positions=0)  # < 1
    
    def test_data_config_parsing(self) -> None:
        """Test data configuration parsing from strings."""
        env_vars = {
            "DEFAULT_PAIRS": "BTC,ETH,SOL,ADA",
            "DEFAULT_TIMEFRAMES": "1m,5m,15m,1h"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = Config.load_from_env()
            
            assert config.data.default_pairs == ["BTC", "ETH", "SOL", "ADA"]
            assert config.data.default_timeframes == ["1m", "5m", "15m", "1h"] 