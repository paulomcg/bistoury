"""
Unit tests for Task 2 Database Setup and Schema Design.
Tests all database functionality including compression, indices, backups, and schema.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from src.bistoury.config import Config, DatabaseConfig
from src.bistoury.database import (
    DatabaseManager, MarketDataSchema, DataInsertion, DataQuery,
    DataCompressionManager, PerformanceIndexManager, BackupManager,
    get_database_switcher
)


class TestDatabaseSetup:
    """Test basic database setup and configuration."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration for testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="bistoury_test_"))
        db_path = temp_dir / "test.duckdb"
        backup_path = temp_dir / "backups"
        
        config = Config()
        config.database = DatabaseConfig(
            path=str(db_path),
            backup_path=str(backup_path)
        )
        
        yield config
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_database_manager_initialization(self, temp_db_config):
        """Test DatabaseManager initialization."""
        db_manager = DatabaseManager(temp_db_config)
        assert db_manager.config == temp_db_config
        assert hasattr(db_manager, '_connections')
        assert hasattr(db_manager, 'db_path')
        assert hasattr(db_manager, 'max_connections')
    
    def test_database_connection(self, temp_db_config):
        """Test database connection establishment."""
        db_manager = DatabaseManager(temp_db_config)
        conn = db_manager.get_connection()
        assert conn is not None
        
        # Test that we can execute a simple query
        result = db_manager.execute("SELECT 1 as test")
        assert result[0][0] == 1


class TestMarketDataSchema:
    """Test market data schema creation and validation."""
    
    @pytest.fixture
    def db_setup(self):
        """Setup database for schema testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="bistoury_schema_test_"))
        db_path = temp_dir / "schema_test.duckdb"
        
        config = Config()
        config.database = DatabaseConfig(path=str(db_path))
        
        db_manager = DatabaseManager(config)
        schema = MarketDataSchema(db_manager)
        
        yield db_manager, schema
        
        # Cleanup
        db_manager.close_all_connections()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_create_symbols_table(self, db_setup):
        """Test symbols table creation."""
        db_manager, schema = db_setup
        schema.create_symbols_table()
        
        # Verify table exists
        tables = db_manager.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'symbols'"
        )
        assert len(tables) == 1
        assert tables[0][0] == 'symbols'
    
    def test_create_candles_tables(self, db_setup):
        """Test candlestick tables creation."""
        db_manager, schema = db_setup
        schema.create_candles_tables()
        
        # Check that all timeframe tables were created
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        for tf in timeframes:
            table_name = f'candles_{tf}'
            tables = db_manager.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_name = '{table_name}'"
            )
            assert len(tables) == 1, f"Table {table_name} should exist"
    
    def test_create_trades_table(self, db_setup):
        """Test trades table creation."""
        db_manager, schema = db_setup
        schema.create_trades_table()
        
        tables = db_manager.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'trades'"
        )
        assert len(tables) == 1
    
    def test_create_all_tables(self, db_setup):
        """Test creation of all tables."""
        db_manager, schema = db_setup
        schema.create_all_tables()
        
        # Verify schema validation passes
        assert schema.validate_schema() is True
    
    def test_schema_validation(self, db_setup):
        """Test schema validation functionality."""
        db_manager, schema = db_setup
        
        # Should fail validation with no tables
        assert schema.validate_schema() is False
        
        # Should pass after creating all tables
        schema.create_all_tables()
        assert schema.validate_schema() is True


class TestDataInsertion:
    """Test data insertion operations."""
    
    @pytest.fixture
    def db_with_schema(self):
        """Setup database with complete schema for data insertion testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="bistoury_insertion_test_"))
        db_path = temp_dir / "insertion_test.duckdb"
        
        config = Config()
        config.database = DatabaseConfig(path=str(db_path))
        
        db_manager = DatabaseManager(config)
        schema = MarketDataSchema(db_manager)
        schema.create_all_tables()
        
        insertion = DataInsertion(db_manager)
        
        yield db_manager, insertion
        
        # Cleanup
        db_manager.close_all_connections()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_insert_symbol(self, db_with_schema):
        """Test symbol insertion."""
        db_manager, insertion = db_with_schema
        
        symbol_data = {
            'name': 'BTC',
            'szDecimals': 8,
            'maxLeverage': 100.0,
            'marginTableId': 1,
            'isDelisted': False
        }
        
        symbol_id = insertion.insert_symbol(symbol_data)
        assert symbol_id is not None
        assert symbol_id > 0
        
        # Verify data was inserted
        result = db_manager.execute("SELECT * FROM symbols WHERE symbol = 'BTC'")
        assert len(result) == 1
        assert result[0][1] == 'BTC'  # symbol column
    
    def test_insert_trade(self, db_with_schema):
        """Test trade insertion with HyperLiquid format."""
        db_manager, insertion = db_with_schema
        
        # Insert symbol first
        symbol_data = {'name': 'BTC', 'szDecimals': 8}
        insertion.insert_symbol(symbol_data)
        
        # Insert trade with HyperLiquid format
        trade_data = {
            'coin': 'BTC',
            'px': '50000.00',
            'sz': '1.5',
            'side': 'B',
            'time': int(datetime.now(timezone.utc).timestamp() * 1000),
            'tid': 123456789
        }
        
        trade_id = insertion.insert_trade(trade_data)
        assert trade_id is not None
        assert trade_id > 0
        
        # Verify trade was inserted
        result = db_manager.execute("SELECT * FROM trades WHERE trade_id = 123456789")
        assert len(result) == 1
        assert result[0][1] == 'BTC'  # symbol column
    
    def test_insert_candle(self, db_with_schema):
        """Test candle insertion with HyperLiquid format."""
        db_manager, insertion = db_with_schema
        
        candle_data = {
            's': 'BTC',
            't': int(datetime.now(timezone.utc).timestamp() * 1000),
            'o': '49000.00',
            'h': '51000.00',
            'l': '48000.00',
            'c': '50000.00',
            'v': '100.0',
            'n': 150
        }
        
        candle_id = insertion.insert_candle('1m', candle_data)
        assert candle_id is not None
        assert candle_id > 0
        
        # Verify candle was inserted
        result = db_manager.execute("SELECT * FROM candles_1m WHERE symbol = 'BTC'")
        assert len(result) == 1


class TestDataQuery:
    """Test data querying operations."""
    
    @pytest.fixture
    def db_with_data(self):
        """Setup database with schema and sample data."""
        temp_dir = Path(tempfile.mkdtemp(prefix="bistoury_query_test_"))
        db_path = temp_dir / "query_test.duckdb"
        
        config = Config()
        config.database = DatabaseConfig(path=str(db_path))
        
        db_manager = DatabaseManager(config)
        schema = MarketDataSchema(db_manager)
        schema.create_all_tables()
        
        insertion = DataInsertion(db_manager)
        
        # Insert sample data
        symbol_data = {'name': 'BTC', 'szDecimals': 8}
        insertion.insert_symbol(symbol_data)
        
        trade_data = {
            'coin': 'BTC',
            'px': '50000.00',
            'sz': '1.5',
            'side': 'B',
            'time': int(datetime.now(timezone.utc).timestamp() * 1000),
            'tid': 123456789
        }
        insertion.insert_trade(trade_data)
        
        query = DataQuery(db_manager)
        
        yield db_manager, query
        
        # Cleanup
        db_manager.close_all_connections()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_get_symbols(self, db_with_data):
        """Test symbols query."""
        db_manager, query = db_with_data
        
        symbols = query.get_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) >= 1
        assert symbols[0]['symbol'] == 'BTC'
    
    def test_get_latest_trades(self, db_with_data):
        """Test latest trades query."""
        db_manager, query = db_with_data
        
        trades = query.get_latest_trades('BTC', limit=10)
        assert isinstance(trades, list)
        assert len(trades) >= 1
        assert trades[0]['symbol'] == 'BTC'


class TestCompressionManager:
    """Test data compression and archival functionality."""
    
    @pytest.fixture
    def compression_setup(self):
        """Setup for compression testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="bistoury_compression_test_"))
        db_path = temp_dir / "compression_test.duckdb"
        backup_path = temp_dir / "backups"
        
        config = Config()
        config.database = DatabaseConfig(
            path=str(db_path),
            backup_path=str(backup_path)
        )
        
        db_manager = DatabaseManager(config)
        schema = MarketDataSchema(db_manager)
        schema.create_all_tables()
        
        compression_manager = DataCompressionManager(db_manager, config)
        
        yield db_manager, compression_manager
        
        # Cleanup
        db_manager.close_all_connections()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_configure_duckdb_compression(self, compression_setup):
        """Test DuckDB compression configuration."""
        db_manager, compression_manager = compression_setup
        
        # Should not raise exception
        compression_manager.configure_duckdb_compression()
    
    def test_get_compression_report(self, compression_setup):
        """Test compression report generation."""
        db_manager, compression_manager = compression_setup
        
        report = compression_manager.get_compression_report()
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'tables' in report
        assert 'retention_policies' in report


class TestPerformanceIndexManager:
    """Test performance index management."""
    
    @pytest.fixture
    def index_setup(self):
        """Setup for index testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="bistoury_index_test_"))
        db_path = temp_dir / "index_test.duckdb"
        
        config = Config()
        config.database = DatabaseConfig(path=str(db_path))
        
        db_manager = DatabaseManager(config)
        schema = MarketDataSchema(db_manager)
        schema.create_all_tables()
        
        index_manager = PerformanceIndexManager(db_manager, config)
        
        yield db_manager, index_manager
        
        # Cleanup
        db_manager.close_all_connections()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_index_definitions(self, index_setup):
        """Test that index definitions are properly configured."""
        db_manager, index_manager = index_setup
        
        assert len(index_manager.index_definitions) > 0
        assert 'idx_trades_timestamp_symbol' in index_manager.index_definitions
        assert 'idx_symbols_name' in index_manager.index_definitions
    
    def test_optimize_database(self, index_setup):
        """Test database optimization."""
        db_manager, index_manager = index_setup
        
        result = index_manager.optimize_database()
        assert isinstance(result, dict)
        assert 'operations' in result
        assert 'total_time_seconds' in result
    
    def test_get_index_usage_report(self, index_setup):
        """Test index usage report generation."""
        db_manager, index_manager = index_setup
        
        report = index_manager.get_index_usage_report()
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'indices' in report


class TestBackupManager:
    """Test backup and restore functionality."""
    
    @pytest.fixture
    def backup_setup(self):
        """Setup for backup testing."""
        temp_dir = Path(tempfile.mkdtemp(prefix="bistoury_backup_test_"))
        db_path = temp_dir / "backup_test.duckdb"
        backup_path = temp_dir / "backups"
        
        config = Config()
        config.database = DatabaseConfig(
            path=str(db_path),
            backup_path=str(backup_path)
        )
        
        db_manager = DatabaseManager(config)
        schema = MarketDataSchema(db_manager)
        schema.create_all_tables()
        
        backup_manager = BackupManager(db_manager, config)
        
        yield db_manager, backup_manager
        
        # Cleanup
        db_manager.close_all_connections()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_backup_manager_initialization(self, backup_setup):
        """Test BackupManager initialization."""
        db_manager, backup_manager = backup_setup
        
        assert backup_manager.db_manager == db_manager
        assert backup_manager.config is not None
        assert backup_manager.backup_config is not None
    
    def test_get_backup_status(self, backup_setup):
        """Test backup status report."""
        db_manager, backup_manager = backup_setup
        
        status = backup_manager.get_backup_status()
        assert isinstance(status, dict)
        assert 'timestamp' in status
        assert 'backup_directory' in status
        assert 'total_backups' in status
    
    def test_create_full_backup(self, backup_setup):
        """Test full backup creation."""
        db_manager, backup_manager = backup_setup
        
        # Create a simple backup (database should have some basic structure)
        metadata = backup_manager.create_full_backup("Test backup")
        
        assert metadata is not None
        assert metadata.backup_type == "full"
        assert metadata.file_size_bytes > 0
        assert Path(metadata.backup_path).exists()


class TestDatabaseSwitcher:
    """Test database switcher functionality."""
    
    def test_get_database_switcher(self):
        """Test database switcher initialization."""
        switcher = get_database_switcher()
        assert switcher is not None
        assert hasattr(switcher, 'list_available_databases')
    
    def test_list_available_databases(self):
        """Test listing available databases."""
        switcher = get_database_switcher()
        databases = switcher.list_available_databases()
        assert isinstance(databases, dict) 