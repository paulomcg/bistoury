"""
Pytest configuration and fixtures for Bistoury tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from unittest.mock import patch

from src.bistoury.config import Config
from src.bistoury.database import DatabaseManager


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_env_vars() -> Generator[dict, None, None]:
    """Mock environment variables for testing."""
    test_env = {
        "DATABASE_PATH": "./test_data/test.db",
        "LOG_LEVEL": "DEBUG",
        "DEFAULT_PAIRS": "BTC,ETH",
        "DEFAULT_TIMEFRAMES": "1m,5m",
        "HYPERLIQUID_TESTNET": "true",
        "DEFAULT_POSITION_SIZE_PCT": "5.0",
        "MAX_POSITIONS": "2",
        "RISK_LIMIT_USD": "500.0"
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        yield test_env


@pytest.fixture
def test_config(mock_env_vars: dict) -> Config:
    """Create a test configuration instance."""
    return Config.load_from_env()


@pytest.fixture
async def test_database(temp_dir: Path) -> DatabaseManager:
    """Create a test database instance."""
    # Create test database in temp directory
    test_db_path = temp_dir / "test_bistoury.db"
    
    # Override config to use test database
    test_env = {
        "DATABASE_PATH": str(test_db_path),
        "LOG_LEVEL": "DEBUG"
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        config = Config.load_from_env()
        db_manager = DatabaseManager(config)
        
        # Initialize the database schema using MarketDataSchema
        from src.bistoury.database.schema import MarketDataSchema
        schema = MarketDataSchema(db_manager)
        schema.create_all_tables()
        
        yield db_manager
        
        # Cleanup
        try:
            if hasattr(db_manager, 'close_all_connections'):
                db_manager.close_all_connections()
            if test_db_path.exists():
                test_db_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
def sample_candlestick_data() -> dict:
    """Sample candlestick data for testing."""
    return {
        "symbol": "BTC",
        "timeframe": "1m",
        "timestamp": 1640995200000,  # 2022-01-01 00:00:00 UTC
        "open": 47000.0,
        "high": 47500.0,
        "low": 46800.0,
        "close": 47200.0,
        "volume": 1.5
    }


@pytest.fixture
def sample_order_book_data() -> dict:
    """Sample order book data for testing."""
    return {
        "symbol": "BTC",
        "timestamp": 1640995200000,
        "bids": [
            [47000.0, 0.5],
            [46950.0, 1.0],
            [46900.0, 2.0]
        ],
        "asks": [
            [47050.0, 0.8],
            [47100.0, 1.2],
            [47150.0, 1.5]
        ]
    }


@pytest.fixture
def sample_trade_data() -> dict:
    """Sample trade data for testing."""
    return {
        "symbol": "BTC",
        "timestamp": 1640995200000,
        "price": 47025.0,
        "size": 0.1,
        "side": "buy",
        "trade_id": "12345"
    } 