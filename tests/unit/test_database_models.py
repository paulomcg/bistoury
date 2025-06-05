"""
Test suite for database models and serialization utilities.

Tests the database-optimized models, serialization helpers, and
batch processing utilities for efficient DuckDB storage.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
import json
import time

from src.bistoury.models.database import (
    DatabaseModel,
    DBCandlestickData,
    DBTradeData,
    DBOrderBookSnapshot,
    DBFundingRateData,
    DBTradingSignal,
    DBPosition,
    DBBatchOperation,
    DBArchiveRecord
)

from src.bistoury.models.serialization import (
    DatabaseSerializer,
    ModelConverter,
    BatchProcessor,
    DataIntegrityValidator,
    SerializationFormat,
    CompressionLevel,
    SerializationMetrics
)

from src.bistoury.models.market_data import CandlestickData, Timeframe
from src.bistoury.models.signals import TradingSignal, SignalDirection, SignalType
from src.bistoury.models.trading import Position, PositionSide


class TestDatabaseModel:
    """Test base database model functionality."""
    
    def test_to_db_dict(self):
        """Test conversion to database dictionary."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
            open_price="50000.00",
            high_price="50000.00",
            low_price="50000.00",
            close_price="50000.00",
            volume="1.0"
        )
        
        db_dict = candle.to_db_dict()
        assert isinstance(db_dict, dict)
        assert db_dict['symbol'] == "BTC"
        assert db_dict['open_price'] == "50000.00"
    
    def test_from_db_dict(self):
        """Test creation from database dictionary."""
        data = {
            'symbol': 'ETH',
            'timestamp_start': datetime.now(timezone.utc),
            'timestamp_end': datetime.now(timezone.utc),
            'open_price': '3000.00',
            'high_price': '3000.00',
            'low_price': '3000.00',
            'close_price': '3000.00',
            'volume': '10.0'
        }
        
        candle = DBCandlestickData.from_db_dict(data)
        assert candle.symbol == 'ETH'
        assert candle.open_price == '3000.00'
    
    def test_get_batch_key(self):
        """Test batch key generation."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
            open_price="50000.00",
            high_price="50000.00",
            low_price="50000.00",
            close_price="50000.00",
            volume="1.0"
        )
        
        batch_key = candle.get_batch_key()
        assert "DBCandlestickData" in batch_key
        assert "BTC" in batch_key


class TestDBCandlestickData:
    """Test database candlestick data model."""
    
    def test_creation(self):
        """Test creating candlestick data."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            timestamp_end=datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            open_price="50000.00",
            high_price="50100.00",
            low_price="49900.00",
            close_price="50050.00",
            volume="100.50",
            trade_count=150
        )
        
        assert candle.symbol == "BTC"
        assert candle.open_price == "50000.00"
        assert candle.trade_count == 150
    
    def test_table_name(self):
        """Test table name computation."""
        candle = DBCandlestickData(
            symbol="ETH",
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
            open_price="3000.00",
            high_price="3000.00",
            low_price="3000.00",
            close_price="3000.00",
            volume="1.0"
        )
        
        # Table name should be based on timestamp_start date
        expected_date = candle.timestamp_start.strftime('%Y-%m-%d')
        assert candle.table_name == f"candles_{expected_date}"
    
    def test_price_range_pct(self):
        """Test price range percentage calculation."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
            open_price="50000.00",
            high_price="51000.00",  # 2% above low
            low_price="50000.00",
            close_price="50500.00",
            volume="1.0"
        )
        
        # (51000 - 50000) / 50000 * 100 = 2%
        assert candle.price_range_pct == Decimal('2.00')
    
    def test_get_compression_data(self):
        """Test compression data extraction."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            timestamp_end=datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            open_price="50000.00",
            high_price="50100.00",
            low_price="49900.00",
            close_price="50050.00",
            volume="100.50",
            trade_count=150
        )
        
        data = candle.get_compression_data()
        assert data['symbol'] == "BTC"
        assert data['open_price'] == "50000.00"
        assert data['volume'] == "100.50"


class TestDBTradingSignal:
    """Test database trading signal model."""
    
    def test_creation(self):
        """Test creating trading signal."""
        signal = DBTradingSignal(
            signal_id="signal_123",
            symbol="BTC",
            direction="BUY",
            signal_type="TECHNICAL",
            confidence="75.0",
            strength="0.8",
            price="50000.00",
            target_price="52000.00",
            stop_loss="48000.00",
            timeframe="1h",
            timestamp=datetime.now(timezone.utc),
            source="technical_analyzer",
            reason="RSI oversold",
            metadata_json='{"rsi": 25, "volume_spike": true}',
            is_active=True
        )
        
        assert signal.signal_id == "signal_123"
        assert signal.symbol == "BTC"
        assert signal.direction == "BUY"
        assert signal.confidence == "75.0"
        assert signal.is_active is True
    
    def test_metadata_parsing(self):
        """Test metadata JSON parsing."""
        signal = DBTradingSignal(
            signal_id="signal_456",
            symbol="ETH",
            direction="SELL",
            signal_type="MOMENTUM",
            confidence="60.0",
            strength="0.6",
            price="3000.00",
            timeframe="4h",
            timestamp=datetime.now(timezone.utc),
            source="momentum_analyzer",
            reason="Bearish divergence",
            metadata_json='{"macd": -0.5, "volume": 1000}'
        )
        
        metadata = signal.metadata
        assert metadata['macd'] == -0.5
        assert metadata['volume'] == 1000
    
    def test_risk_reward_ratio_buy(self):
        """Test risk/reward ratio calculation for buy signal."""
        signal = DBTradingSignal(
            signal_id="signal_buy",
            symbol="BTC",
            direction="BUY",
            signal_type="TECHNICAL",
            confidence="80.0",
            strength="0.9",
            price="50000.00",
            target_price="55000.00",  # $5000 profit
            stop_loss="47000.00",     # $3000 loss
            timeframe="1d",
            timestamp=datetime.now(timezone.utc),
            source="test",
            reason="Test signal"
        )
        
        # Risk/Reward = 5000 / 3000 = 1.67
        ratio = signal.risk_reward_ratio
        assert ratio is not None
        assert abs(ratio - Decimal('1.666666666666666666666666667')) < Decimal('0.01')
    
    def test_risk_reward_ratio_sell(self):
        """Test risk/reward ratio calculation for sell signal."""
        signal = DBTradingSignal(
            signal_id="signal_sell",
            symbol="BTC",
            direction="SELL",
            signal_type="TECHNICAL",
            confidence="70.0",
            strength="0.7",
            price="50000.00",
            target_price="46000.00",  # $4000 profit
            stop_loss="52000.00",     # $2000 loss
            timeframe="1d",
            timestamp=datetime.now(timezone.utc),
            source="test",
            reason="Test signal"
        )
        
        # Risk/Reward = 4000 / 2000 = 2.0
        ratio = signal.risk_reward_ratio
        assert ratio is not None
        assert ratio == Decimal('2.0')


class TestDBPosition:
    """Test database position model."""
    
    def test_creation(self):
        """Test creating position."""
        position = DBPosition(
            position_id="pos_123",
            symbol="BTC",
            side="long",
            size="1.5",
            entry_price="50000.00",
            current_price="51000.00",
            unrealized_pnl="1500.00",
            realized_pnl="0.00",
            margin_used="10000.00",
            timestamp=datetime.now(timezone.utc),
            is_open=True
        )
        
        assert position.position_id == "pos_123"
        assert position.symbol == "BTC"
        assert position.side == "long"
        assert position.size == "1.5"
        assert position.is_open is True
    
    def test_notional_value(self):
        """Test notional value calculation."""
        position = DBPosition(
            position_id="pos_456",
            symbol="ETH",
            side="short",
            size="10.0",
            entry_price="3000.00",
            margin_used="6000.00",
            timestamp=datetime.now(timezone.utc)
        )
        
        # 10.0 * 3000.00 = 30000.00
        assert position.notional_value == Decimal('30000.00')
    
    def test_leverage(self):
        """Test leverage calculation."""
        position = DBPosition(
            position_id="pos_789",
            symbol="BTC",
            side="long",
            size="2.0",
            entry_price="50000.00",
            margin_used="20000.00",  # 100k notional / 20k margin = 5x
            timestamp=datetime.now(timezone.utc)
        )
        
        assert position.leverage == Decimal('5.0')
    
    def test_pnl_percentage_long(self):
        """Test PnL percentage for long position."""
        position = DBPosition(
            position_id="pos_long",
            symbol="BTC",
            side="long",
            size="1.0",
            entry_price="50000.00",
            current_price="52000.00",  # 4% gain
            margin_used="10000.00",
            timestamp=datetime.now(timezone.utc)
        )
        
        # (52000 - 50000) / 50000 * 100 = 4%
        assert position.pnl_percentage == Decimal('4.00')
    
    def test_pnl_percentage_short(self):
        """Test PnL percentage for short position."""
        position = DBPosition(
            position_id="pos_short",
            symbol="BTC",
            side="short",
            size="1.0",
            entry_price="50000.00",
            current_price="48000.00",  # 4% gain for short
            margin_used="10000.00",
            timestamp=datetime.now(timezone.utc)
        )
        
        # (50000 - 48000) / 50000 * 100 = 4%
        assert position.pnl_percentage == Decimal('4.00')


class TestDBBatchOperation:
    """Test database batch operation model."""
    
    def test_creation(self):
        """Test creating batch operation."""
        batch = DBBatchOperation(
            batch_id="batch_123",
            operation_type="INSERT",
            table_name="candles_1m",
            record_count=1000,
            start_time=datetime.now(timezone.utc),
            status="running"
        )
        
        assert batch.batch_id == "batch_123"
        assert batch.operation_type == "INSERT"
        assert batch.record_count == 1000
        assert batch.status == "running"
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
        
        batch = DBBatchOperation(
            batch_id="batch_456",
            operation_type="UPDATE",
            table_name="signals",
            record_count=500,
            start_time=start,
            end_time=end,
            status="completed"
        )
        
        assert batch.duration_seconds == 30.0
    
    def test_records_per_second(self):
        """Test records per second calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        
        batch = DBBatchOperation(
            batch_id="batch_789",
            operation_type="INSERT",
            table_name="trades",
            record_count=1000,
            start_time=start,
            end_time=end,
            status="completed"
        )
        
        # 1000 records / 10 seconds = 100 records/second
        assert batch.records_per_second == 100.0
    
    def test_mark_completed(self):
        """Test marking batch as completed."""
        batch = DBBatchOperation(
            batch_id="batch_complete",
            operation_type="DELETE",
            table_name="archived_data",
            record_count=250,
            start_time=datetime.now(timezone.utc),
            status="running"
        )
        
        # Mark as completed successfully
        batch.mark_completed(success=True)
        
        assert batch.status == "completed"
        assert batch.end_time is not None
        assert batch.error_message is None
        
        # Mark as failed
        batch.mark_completed(success=False, error_msg="Connection lost")
        
        assert batch.status == "failed"
        assert batch.error_message == "Connection lost"


class TestDBArchiveRecord:
    """Test database archive record model."""
    
    def test_creation(self):
        """Test creating archive record."""
        archive = DBArchiveRecord(
            archive_id="archive_123",
            table_name="historical_candles",
            date_range_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            date_range_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            record_count=100000,
            compressed_size_bytes=5000000,
            original_size_bytes=20000000,
            archive_path="/archives/2024-01-candles.zst",
            checksum="abc123def456",
            compression_algorithm="zstd",
            retention_policy="12_months"
        )
        
        assert archive.archive_id == "archive_123"
        assert archive.table_name == "historical_candles"
        assert archive.record_count == 100000
        assert archive.compression_algorithm == "zstd"
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        archive = DBArchiveRecord(
            archive_id="archive_456",
            table_name="trades",
            date_range_start=datetime.now(timezone.utc),
            date_range_end=datetime.now(timezone.utc),
            record_count=50000,
            compressed_size_bytes=2500000,   # 2.5MB
            original_size_bytes=10000000,    # 10MB
            archive_path="/archives/test.zst",
            checksum="def456ghi789",
            retention_policy="6_months"
        )
        
        # 2.5MB / 10MB = 0.25 (75% compression)
        assert archive.compression_ratio == 0.25
    
    def test_space_saved_mb(self):
        """Test space saved calculation."""
        archive = DBArchiveRecord(
            archive_id="archive_789",
            table_name="signals",
            date_range_start=datetime.now(timezone.utc),
            date_range_end=datetime.now(timezone.utc),
            record_count=25000,
            compressed_size_bytes=1048576,   # 1MB
            original_size_bytes=5242880,     # 5MB
            archive_path="/archives/signals.zst",
            checksum="ghi789jkl012",
            retention_policy="3_months"
        )
        
        # (5MB - 1MB) = 4MB saved
        assert archive.space_saved_mb == 4.0


class TestModelConverter:
    """Test model converter functionality."""
    
    def test_candlestick_conversion(self):
        """Test bidirectional candlestick conversion."""
        # Create original candlestick
        original = CandlestickData(
            symbol="BTC",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            timeframe=Timeframe.ONE_MINUTE,
            open=Decimal("50000.00"),
            high=Decimal("50100.00"),
            low=Decimal("49900.00"),
            close=Decimal("50050.00"),
            volume=Decimal("100.50")
        )
        
        converter = ModelConverter()
        
        # Convert to DB model
        db_candle = converter.candlestick_to_db(original)
        
        assert db_candle.symbol == "BTC"
        assert db_candle.open_price == "50000.00000000"  # Full precision from Decimal
        assert db_candle.high_price == "50100.00000000"
        assert db_candle.low_price == "49900.00000000"
        assert db_candle.close_price == "50050.00000000"
        assert db_candle.volume == "100.50000000"
        assert db_candle.timestamp_start == original.timestamp
        
        # Convert back to business model
        back_converted = converter.db_to_candlestick(db_candle)
        
        assert back_converted.symbol == original.symbol
        assert back_converted.timestamp == original.timestamp
        assert back_converted.open == original.open
        assert back_converted.high == original.high
        assert back_converted.low == original.low
        assert back_converted.close == original.close
        assert back_converted.volume == original.volume
        # Note: timeframe will be ONE_MINUTE as default since it's not stored in DB model
    
    def test_signal_conversion(self):
        """Test bidirectional signal conversion."""
        # Create original signal
        original = TradingSignal(
            signal_id="test_signal_1",
            symbol="ETH",
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            confidence=Decimal("75.0"),
            strength=Decimal("0.8"),
            price=Decimal("3000.00"),
            timeframe=Timeframe.ONE_HOUR,
            timestamp=datetime.now(timezone.utc),
            source="test_converter",
            reason="Test conversion"
        )
        
        converter = ModelConverter()
        
        # Convert to DB model
        db_signal = converter.signal_to_db(original)
        
        assert db_signal.symbol == "ETH"
        assert db_signal.direction == "buy"
        assert db_signal.signal_type == "technical"
        assert db_signal.confidence == "75.0"
        assert db_signal.strength == "0.8"
        assert db_signal.price == "3000.00"
        
        # Convert back to business model
        converted_back = converter.db_to_signal(db_signal)
        
        assert converted_back.symbol == original.symbol
        assert converted_back.direction == original.direction
        assert converted_back.signal_type == original.signal_type
        assert converted_back.confidence == original.confidence


class TestDatabaseSerializer:
    """Test database serialization."""
    
    def test_serialize_single_json(self):
        """Test single model JSON serialization."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
            open_price="50000.00",
            high_price="50000.00",
            low_price="50000.00",
            close_price="50000.00",
            volume="1.0"
        )
        
        serializer = DatabaseSerializer()
        result = serializer.serialize_single(candle, SerializationFormat.JSON)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed['symbol'] == "BTC"
        assert parsed['open_price'] == "50000.00"
    
    def test_serialize_batch(self):
        """Test batch serialization."""
        candles = []
        for i in range(3):
            candle = DBCandlestickData(
                symbol=f"COIN{i}",
                timestamp_start=datetime.now(timezone.utc),
                timestamp_end=datetime.now(timezone.utc),
                open_price="1000.00",
                high_price="1000.00",
                low_price="1000.00",
                close_price="1000.00",
                volume="1.0"
            )
            candles.append(candle)
        
        serializer = DatabaseSerializer()
        results = serializer.serialize_batch(candles, SerializationFormat.JSON)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, str)
            parsed = json.loads(result)
            assert 'symbol' in parsed
            assert 'open_price' in parsed


class TestBatchProcessor:
    """Test batch processor functionality."""
    
    def test_process_candlestick_batch(self):
        """Test candlestick batch processing."""
        # Create business model candles
        candles = []
        for i in range(5):
            candle = CandlestickData(
                symbol=f"TEST{i}",
                timestamp=datetime.now(timezone.utc),
                timeframe=Timeframe.FIVE_MINUTES,
                open=Decimal("1000.00"),
                high=Decimal("1010.00"),
                low=Decimal("990.00"),
                close=Decimal("1005.00"),
                volume=Decimal("100.0")
            )
            candles.append(candle)
        
        processor = BatchProcessor()
        db_candles = processor.process_candlestick_batch(candles)
        
        assert len(db_candles) == 5
        for db_candle in db_candles:
            assert isinstance(db_candle, DBCandlestickData)
            assert db_candle.timestamp_start.strftime('%Y-%m-%d') == datetime.now(timezone.utc).strftime('%Y-%m-%d')
            assert db_candle.open_price == "1000.00000000"
    
    def test_processing_stats(self):
        """Test processing statistics tracking."""
        processor = BatchProcessor()
        
        # Process some data to generate stats
        candles = [
            CandlestickData(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                timeframe=Timeframe.ONE_MINUTE,
                open=Decimal("50000.00"),
                high=Decimal("50000.00"),
                low=Decimal("50000.00"),
                close=Decimal("50000.00"),
                volume=Decimal("1.0")
            )
        ]
        
        processor.process_candlestick_batch(candles)
        stats = processor.get_processing_stats()
        
        assert 'total_records' in stats
        assert stats['total_records'] == 1
        assert 'total_processing_time' in stats
        assert stats['total_processing_time'] > 0


class TestDataIntegrityValidator:
    """Test data integrity validation."""
    
    def test_validate_candlestick_data_valid(self):
        """Test validation of valid candlestick data."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            timestamp_end=datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            open_price="50000.00",
            high_price="50100.00",  # Correctly highest
            low_price="49900.00",   # Correctly lowest
            close_price="50050.00",
            volume="100.50"   # Positive volume
        )
        
        validator = DataIntegrityValidator()
        is_valid = validator.validate_candlestick_data(candle)
        
        assert is_valid is True
        assert len(validator.error_log) == 0
    
    def test_validate_candlestick_data_invalid(self):
        """Test validation of invalid candlestick data."""
        candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            timestamp_end=datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            open_price="50000.00",
            high_price="49900.00",  # Invalid: lower than open
            low_price="50100.00",   # Invalid: higher than open
            close_price="50050.00",
            volume="-10.50"   # Invalid: negative volume
        )
        
        validator = DataIntegrityValidator()
        is_valid = validator.validate_candlestick_data(candle)
        
        assert is_valid is False
        assert len(validator.error_log) > 0
        
        # Check specific error messages
        errors = validator.error_log[0]['errors']
        assert any("High price is not the highest value" in error for error in errors)
        assert any("Low price is not the lowest value" in error for error in errors)
        assert any("Volume cannot be negative" in error for error in errors)
    
    def test_validate_signal_data_valid(self):
        """Test validation of valid signal data."""
        signal = DBTradingSignal(
            signal_id="signal_123",
            symbol="BTC",
            direction="BUY",
            signal_type="TECHNICAL",
            confidence="75.0",  # Valid range
            strength="0.8",     # Valid range
            price="50000.00",   # Positive price
            target_price="52000.00",  # Above entry for buy
            stop_loss="48000.00",     # Below entry for buy
            timeframe="1h",
            timestamp=datetime.now(timezone.utc),
            source="test",
            reason="Valid signal"
        )
        
        validator = DataIntegrityValidator()
        is_valid = validator.validate_signal_data(signal)
        
        assert is_valid is True
        assert len(validator.error_log) == 0
    
    def test_validate_signal_data_invalid(self):
        """Test validation of invalid signal data."""
        signal = DBTradingSignal(
            signal_id="signal_456",
            symbol="BTC",
            direction="BUY",
            signal_type="TECHNICAL",
            confidence="150.0",  # Invalid: over 100
            strength="1.5",      # Invalid: over 1.0
            price="-1000.00",    # Invalid: negative price
            target_price="45000.00",  # Invalid: below entry for buy
            stop_loss="55000.00",     # Invalid: above entry for buy
            timeframe="1h",
            timestamp=datetime.now(timezone.utc),
            source="test",
            reason="Invalid signal"
        )
        
        validator = DataIntegrityValidator()
        is_valid = validator.validate_signal_data(signal)
        
        assert is_valid is False
        assert len(validator.error_log) == 1
        
        error_entry = validator.error_log[0]
        assert error_entry['model_type'] == 'DBTradingSignal'
        assert len(error_entry['errors']) >= 3  # Multiple validation errors
    
    def test_error_summary(self):
        """Test error summary generation."""
        validator = DataIntegrityValidator()
        
        # Generate some errors
        invalid_candle = DBCandlestickData(
            symbol="BTC",
            timestamp_start=datetime.now(timezone.utc),
            timestamp_end=datetime.now(timezone.utc),
            open_price="50000.00",
            high_price="49900.00",  # Invalid
            low_price="50100.00",   # Invalid
            close_price="50050.00",
            volume="100.50"
        )
        
        invalid_signal = DBTradingSignal(
            signal_id="bad_signal",
            symbol="ETH",
            direction="BUY",
            signal_type="TECHNICAL",
            confidence="200.0",  # Invalid
            strength="0.5",
            price="3000.00",
            timeframe="1h",
            timestamp=datetime.now(timezone.utc),
            source="test",
            reason="Bad signal"
        )
        
        validator.validate_candlestick_data(invalid_candle)
        validator.validate_signal_data(invalid_signal)
        
        summary = validator.get_error_summary()
        
        assert summary['total_errors'] == 2
        assert 'DBCandlestickData' in summary['by_model_type']
        assert 'DBTradingSignal' in summary['by_model_type']
        assert summary['by_model_type']['DBCandlestickData']['count'] == 1
        assert summary['by_model_type']['DBTradingSignal']['count'] == 1 