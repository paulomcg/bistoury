"""
Database Serialization Helpers

This module provides serialization utilities for efficient database operations:
- Batch processing helpers
- Compression-aware serialization
- Database-optimized converters
- Performance monitoring utilities
- Data validation and integrity checks

These utilities bridge the gap between Pydantic models and DuckDB storage,
ensuring optimal performance for high-frequency trading data.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import logging
import time
from collections import defaultdict, deque

from .database import DatabaseModel
from .market_data import CandlestickData, Timeframe
from .signals import TradingSignal
from .trading import Position, Order

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=DatabaseModel)


class SerializationFormat(str, Enum):
    """Serialization format enumeration."""
    JSON = "json"
    COMPRESSED_JSON = "compressed_json"
    BINARY = "binary"
    PARQUET = "parquet"


class CompressionLevel(str, Enum):
    """Compression level enumeration."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class SerializationMetrics:
    """Metrics for serialization operations."""
    operation_type: str
    record_count: int
    start_time: float
    end_time: Optional[float] = None
    original_size_bytes: Optional[int] = None
    serialized_size_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None
    records_per_second: Optional[float] = None
    error_count: int = 0
    
    def complete(self, serialized_size: Optional[int] = None) -> None:
        """Mark operation as complete and calculate metrics."""
        self.end_time = time.time()
        self.serialized_size_bytes = serialized_size
        
        duration = self.end_time - self.start_time
        if duration > 0:
            self.records_per_second = self.record_count / duration
        
        if self.original_size_bytes and serialized_size:
            self.compression_ratio = serialized_size / self.original_size_bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation_type': self.operation_type,
            'record_count': self.record_count,
            'duration_seconds': (self.end_time - self.start_time) if self.end_time else None,
            'original_size_bytes': self.original_size_bytes,
            'serialized_size_bytes': self.serialized_size_bytes,
            'compression_ratio': self.compression_ratio,
            'records_per_second': self.records_per_second,
            'error_count': self.error_count
        }


class DatabaseSerializer:
    """
    High-performance database serializer for trading data.
    
    Provides optimized serialization for different data types with
    compression, batching, and performance monitoring capabilities.
    """
    
    def __init__(
        self,
        compression_level: CompressionLevel = CompressionLevel.MEDIUM,
        batch_size: int = 1000,
        enable_metrics: bool = True
    ):
        self.compression_level = compression_level
        self.batch_size = batch_size
        self.enable_metrics = enable_metrics
        self.metrics_history: deque = deque(maxlen=1000)
        self._compression_settings = self._get_compression_settings()
    
    def _get_compression_settings(self) -> Dict[str, Any]:
        """Get compression settings based on level."""
        settings = {
            CompressionLevel.NONE: {},
            CompressionLevel.LOW: {'separators': (',', ':')},
            CompressionLevel.MEDIUM: {
                'separators': (',', ':'),
                'ensure_ascii': False
            },
            CompressionLevel.HIGH: {
                'separators': (',', ':'),
                'ensure_ascii': False,
                'sort_keys': True
            },
            CompressionLevel.MAXIMUM: {
                'separators': (',', ':'),
                'ensure_ascii': False,
                'sort_keys': True,
                'indent': None
            }
        }
        return settings[self.compression_level]
    
    def serialize_single(
        self,
        model: DatabaseModel,
        format: SerializationFormat = SerializationFormat.JSON
    ) -> Union[str, bytes, Dict[str, Any]]:
        """Serialize a single model instance."""
        metrics = SerializationMetrics("serialize_single", 1, time.time())
        
        try:
            if format == SerializationFormat.JSON:
                data = model.to_db_dict()
                result = json.dumps(data, **self._compression_settings)
                metrics.serialized_size_bytes = len(result.encode('utf-8'))
                return result
            
            elif format == SerializationFormat.COMPRESSED_JSON:
                data = model.get_compression_data()
                result = json.dumps(data, **self._compression_settings)
                metrics.serialized_size_bytes = len(result.encode('utf-8'))
                return result
            
            else:
                # Return raw dictionary for database insertion
                result = model.to_db_dict()
                metrics.serialized_size_bytes = len(str(result).encode('utf-8'))
                return result
                
        except Exception as e:
            metrics.error_count += 1
            logger.error(f"Serialization error for {model.__class__.__name__}: {e}")
            raise
        finally:
            metrics.complete()
            if self.enable_metrics:
                self.metrics_history.append(metrics)
    
    def serialize_batch(
        self,
        models: List[DatabaseModel],
        format: SerializationFormat = SerializationFormat.JSON
    ) -> List[Union[str, bytes, Dict[str, Any]]]:
        """Serialize a batch of models with optimization."""
        metrics = SerializationMetrics("serialize_batch", len(models), time.time())
        
        try:
            results = []
            
            # Group models by type for optimized processing
            grouped_models = defaultdict(list)
            for model in models:
                grouped_models[model.__class__.__name__].append(model)
            
            # Process each group with type-specific optimizations
            for model_type, model_list in grouped_models.items():
                for model in model_list:
                    try:
                        result = self.serialize_single(model, format)
                        results.append(result)
                    except Exception as e:
                        metrics.error_count += 1
                        logger.error(f"Error serializing {model_type}: {e}")
                        continue
            
            # Calculate total serialized size
            total_size = 0
            for result in results:
                if isinstance(result, str):
                    total_size += len(result.encode('utf-8'))
                elif isinstance(result, bytes):
                    total_size += len(result)
                else:
                    total_size += len(str(result).encode('utf-8'))
            
            metrics.serialized_size_bytes = total_size
            return results
            
        except Exception as e:
            metrics.error_count += 1
            logger.error(f"Batch serialization error: {e}")
            raise
        finally:
            metrics.complete()
            if self.enable_metrics:
                self.metrics_history.append(metrics)
    
    def deserialize_single(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        model_class: Type[T],
        format: SerializationFormat = SerializationFormat.JSON
    ) -> T:
        """Deserialize a single model instance."""
        metrics = SerializationMetrics("deserialize_single", 1, time.time())
        
        try:
            if isinstance(data, str):
                parsed_data = json.loads(data)
            elif isinstance(data, bytes):
                parsed_data = json.loads(data.decode('utf-8'))
            else:
                parsed_data = data
            
            result = model_class.from_db_dict(parsed_data)
            return result
            
        except Exception as e:
            metrics.error_count += 1
            logger.error(f"Deserialization error for {model_class.__name__}: {e}")
            raise
        finally:
            metrics.complete()
            if self.enable_metrics:
                self.metrics_history.append(metrics)
    
    def deserialize_batch(
        self,
        data_list: List[Union[str, bytes, Dict[str, Any]]],
        model_class: Type[T],
        format: SerializationFormat = SerializationFormat.JSON
    ) -> List[T]:
        """Deserialize a batch of models."""
        metrics = SerializationMetrics("deserialize_batch", len(data_list), time.time())
        
        try:
            results = []
            
            for data in data_list:
                try:
                    result = self.deserialize_single(data, model_class, format)
                    results.append(result)
                except Exception as e:
                    metrics.error_count += 1
                    logger.error(f"Error deserializing {model_class.__name__}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            metrics.error_count += 1
            logger.error(f"Batch deserialization error: {e}")
            raise
        finally:
            metrics.complete()
            if self.enable_metrics:
                self.metrics_history.append(metrics)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of serialization metrics."""
        if not self.metrics_history:
            return {}
        
        by_operation = defaultdict(list)
        for metrics in self.metrics_history:
            by_operation[metrics.operation_type].append(metrics)
        
        summary = {}
        for operation, metrics_list in by_operation.items():
            total_records = sum(m.record_count for m in metrics_list)
            total_errors = sum(m.error_count for m in metrics_list)
            avg_rps = sum(m.records_per_second or 0 for m in metrics_list) / len(metrics_list)
            
            summary[operation] = {
                'operation_count': len(metrics_list),
                'total_records': total_records,
                'total_errors': total_errors,
                'error_rate': total_errors / total_records if total_records > 0 else 0,
                'avg_records_per_second': avg_rps
            }
        
        return summary


class ModelConverter:
    """
    Converts between business models and database models.
    
    Provides bidirectional conversion with data validation,
    type safety, and performance optimization.
    """
    
    def __init__(self):
        self.conversion_cache: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
    
    def candlestick_to_db(self, candle: CandlestickData) -> 'DBCandlestickData':
        """Convert CandlestickData to DBCandlestickData."""
        from .database import DBCandlestickData
        
        # For candlestick data, we typically only have the start timestamp
        # The end timestamp can be calculated or set to the same value
        timestamp_start = candle.timestamp
        timestamp_end = candle.timestamp  # For most cases, start and end are the same
        
        return DBCandlestickData(
            symbol=candle.symbol,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            open_price=str(candle.open),
            high_price=str(candle.high),
            low_price=str(candle.low),
            close_price=str(candle.close),
            volume=str(candle.volume),
            trade_count=getattr(candle, 'trade_count', None)
        )
    
    def db_to_candlestick(self, db_candle: 'DBCandlestickData') -> CandlestickData:
        """Convert DBCandlestickData to CandlestickData."""
        from .market_data import Timeframe
        
        # We need to determine the timeframe somehow - this might need to be stored separately
        # For now, assume 1m timeframe as default
        timeframe = Timeframe.ONE_MINUTE  # This should ideally be stored in the model
        
        return CandlestickData(
            symbol=db_candle.symbol,
            timestamp=db_candle.timestamp_start,
            timeframe=timeframe,
            open=Decimal(db_candle.open_price),
            high=Decimal(db_candle.high_price),
            low=Decimal(db_candle.low_price),
            close=Decimal(db_candle.close_price),
            volume=Decimal(db_candle.volume),
            trade_count=db_candle.trade_count
        )
    
    def signal_to_db(self, signal: TradingSignal) -> 'DBTradingSignal':
        """Convert TradingSignal to DBTradingSignal."""
        from .database import DBTradingSignal
        
        return DBTradingSignal(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction.value,
            signal_type=signal.signal_type.value,
            confidence=str(signal.confidence),
            strength=str(signal.strength),
            price=str(signal.price),
            target_price=str(signal.target_price) if signal.target_price else None,
            stop_loss=str(signal.stop_loss) if signal.stop_loss else None,
            timeframe=signal.timeframe.value,
            timestamp=signal.timestamp,
            expiry=signal.expiry,
            source=signal.source,
            reason=signal.reason,
            metadata_json=json.dumps(signal.metadata) if signal.metadata else None,
            is_active=signal.is_active
        )
    
    def db_to_signal(self, db_signal: 'DBTradingSignal') -> TradingSignal:
        """Convert DBTradingSignal to TradingSignal."""
        from .signals import SignalDirection, SignalType
        
        return TradingSignal(
            signal_id=db_signal.signal_id,
            symbol=db_signal.symbol,
            direction=SignalDirection(db_signal.direction),
            signal_type=SignalType(db_signal.signal_type),
            confidence=Decimal(db_signal.confidence),
            strength=Decimal(db_signal.strength),
            price=Decimal(db_signal.price),
            target_price=Decimal(db_signal.target_price) if db_signal.target_price else None,
            stop_loss=Decimal(db_signal.stop_loss) if db_signal.stop_loss else None,
            timeframe=Timeframe(db_signal.timeframe),
            timestamp=db_signal.timestamp,
            expiry=db_signal.expiry,
            source=db_signal.source,
            reason=db_signal.reason,
            metadata=db_signal.metadata,
            is_active=db_signal.is_active
        )
    
    def position_to_db(self, position: Position) -> 'DBPosition':
        """Convert Position to DBPosition."""
        from .database import DBPosition
        
        return DBPosition(
            position_id=f"{position.symbol}_{position.side.value}_{int(time.time())}",
            symbol=position.symbol,
            side=position.side.value,
            size=str(position.size),
            entry_price=str(position.entry_price),
            current_price=str(position.current_price) if position.current_price else None,
            unrealized_pnl=str(position.unrealized_pnl) if position.unrealized_pnl else None,
            realized_pnl=str(position.realized_pnl),
            margin_used=str(position.margin_used),
            timestamp=position.timestamp,
            updated_at=position.updated_at,
            is_open=position.is_open
        )
    
    def db_to_position(self, db_position: 'DBPosition') -> Position:
        """Convert DBPosition to Position."""
        from .trading import PositionSide
        
        return Position(
            symbol=db_position.symbol,
            side=PositionSide(db_position.side),
            size=Decimal(db_position.size),
            entry_price=Decimal(db_position.entry_price),
            current_price=Decimal(db_position.current_price) if db_position.current_price else None,
            unrealized_pnl=Decimal(db_position.unrealized_pnl) if db_position.unrealized_pnl else None,
            realized_pnl=Decimal(db_position.realized_pnl),
            margin_used=Decimal(db_position.margin_used),
            timestamp=db_position.timestamp,
            updated_at=db_position.updated_at,
            is_open=db_position.is_open
        )


class BatchProcessor:
    """
    High-performance batch processor for database operations.
    
    Optimizes bulk operations with intelligent batching,
    error handling, and performance monitoring.
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        max_retries: int = 3,
        enable_validation: bool = True
    ):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.enable_validation = enable_validation
        self.serializer = DatabaseSerializer()
        self.converter = ModelConverter()
        self.processing_stats = defaultdict(int)
    
    def process_candlestick_batch(
        self,
        candles: List[CandlestickData]
    ) -> List['DBCandlestickData']:
        """Process a batch of candlestick data for database storage."""
        start_time = time.time()
        
        try:
            # Convert to database models
            db_candles = []
            for candle in candles:
                try:
                    db_candle = self.converter.candlestick_to_db(candle)
                    if self.enable_validation:
                        self._validate_model(db_candle)
                    db_candles.append(db_candle)
                except Exception as e:
                    logger.error(f"Error converting candlestick {candle.symbol}: {e}")
                    self.processing_stats['conversion_errors'] += 1
                    continue
            
            # Group by symbol for optimized insertion
            grouped = defaultdict(list)
            for db_candle in db_candles:
                key = f"{db_candle.symbol}"
                grouped[key].append(db_candle)
            
            # Process each group
            result = []
            for group_key, group_candles in grouped.items():
                result.extend(group_candles)
                self.processing_stats[f'processed_{group_key}'] += len(group_candles)
            
            processing_time = time.time() - start_time
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['total_records'] += len(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self.processing_stats['batch_errors'] += 1
            raise
    
    def process_signal_batch(
        self,
        signals: List[TradingSignal]
    ) -> List['DBTradingSignal']:
        """Process a batch of trading signals for database storage."""
        start_time = time.time()
        
        try:
            db_signals = []
            for signal in signals:
                try:
                    db_signal = self.converter.signal_to_db(signal)
                    if self.enable_validation:
                        self._validate_model(db_signal)
                    db_signals.append(db_signal)
                except Exception as e:
                    logger.error(f"Error converting signal {signal.signal_id}: {e}")
                    self.processing_stats['conversion_errors'] += 1
                    continue
            
            processing_time = time.time() - start_time
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['total_records'] += len(db_signals)
            
            return db_signals
            
        except Exception as e:
            logger.error(f"Signal batch processing error: {e}")
            self.processing_stats['batch_errors'] += 1
            raise
    
    def _validate_model(self, model: DatabaseModel) -> bool:
        """Validate a database model."""
        try:
            # Models are already validated by Pydantic during creation
            # Just ensure the model has required attributes
            if hasattr(model, 'symbol'):
                # For candlestick models, check timestamp_start
                if hasattr(model, 'timestamp_start'):
                    return True
                # For other models, check timestamp
                elif hasattr(model, 'timestamp'):
                    return True
            return False
        except Exception as e:
            logger.warning(f"Model validation failed for {model.__class__.__name__}: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = dict(self.processing_stats)
        
        # Calculate derived metrics
        if stats.get('total_records', 0) > 0 and stats.get('total_processing_time', 0) > 0:
            stats['records_per_second'] = stats['total_records'] / stats['total_processing_time']
        
        if stats.get('total_records', 0) > 0:
            stats['error_rate'] = stats.get('conversion_errors', 0) / stats['total_records']
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats.clear()


class DataIntegrityValidator:
    """
    Validates data integrity for database operations.
    
    Ensures data consistency, detects corruption, and
    validates business logic constraints.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[callable]] = {}
        self.error_log: List[Dict[str, Any]] = []
    
    def register_validation_rule(
        self,
        model_type: str,
        rule_name: str,
        rule_func: callable
    ) -> None:
        """Register a validation rule for a model type."""
        if model_type not in self.validation_rules:
            self.validation_rules[model_type] = []
        
        rule_func.__name__ = rule_name
        self.validation_rules[model_type].append(rule_func)
    
    def validate_candlestick_data(self, candle: 'DBCandlestickData') -> bool:
        """Validate candlestick data integrity."""
        errors = []
        
        try:
            # Price validation
            prices = [
                Decimal(candle.open_price),
                Decimal(candle.high_price),
                Decimal(candle.low_price),
                Decimal(candle.close_price)
            ]
            
            # High should be highest
            if max(prices) != Decimal(candle.high_price):
                errors.append("High price is not the highest value")
            
            # Low should be lowest
            if min(prices) != Decimal(candle.low_price):
                errors.append("Low price is not the lowest value")
            
            # Volume should be positive
            if Decimal(candle.volume) < 0:
                errors.append("Volume cannot be negative")
            
            # Timestamp validation
            if candle.timestamp_start > datetime.now(timezone.utc):
                errors.append("Timestamp cannot be in the future")
            
        except Exception as e:
            errors.append(f"Data conversion error: {e}")
        
        if errors:
            self.error_log.append({
                'model_type': 'DBCandlestickData',
                'model_id': f"{candle.symbol}_{int(candle.timestamp_start.timestamp() * 1000)}",
                'errors': errors,
                'timestamp': datetime.now(timezone.utc)
            })
            return False
        
        return True
    
    def validate_signal_data(self, signal: 'DBTradingSignal') -> bool:
        """Validate trading signal data integrity."""
        errors = []
        
        try:
            # Confidence validation
            confidence = Decimal(signal.confidence)
            if not (0 <= confidence <= 100):
                errors.append("Confidence must be between 0 and 100")
            
            # Strength validation
            strength = Decimal(signal.strength)
            if not (0 <= strength <= 1):
                errors.append("Strength must be between 0 and 1")
            
            # Price validation
            price = Decimal(signal.price)
            if price <= 0:
                errors.append("Price must be positive")
            
            # Risk/reward validation if targets set
            if signal.target_price and signal.stop_loss:
                target = Decimal(signal.target_price)
                stop = Decimal(signal.stop_loss)
                
                if signal.direction.lower() in ['buy', 'strong_buy']:
                    if target <= price:
                        errors.append("Buy signal target should be above entry price")
                    if stop >= price:
                        errors.append("Buy signal stop loss should be below entry price")
                else:
                    if target >= price:
                        errors.append("Sell signal target should be below entry price")
                    if stop <= price:
                        errors.append("Sell signal stop loss should be above entry price")
            
        except Exception as e:
            errors.append(f"Data conversion error: {e}")
        
        if errors:
            self.error_log.append({
                'model_type': 'DBTradingSignal',
                'model_id': signal.signal_id,
                'errors': errors,
                'timestamp': datetime.now(timezone.utc)
            })
            return False
        
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of validation errors."""
        if not self.error_log:
            return {'total_errors': 0, 'by_model_type': {}}
        
        by_model = defaultdict(list)
        for error in self.error_log:
            by_model[error['model_type']].append(error)
        
        summary = {
            'total_errors': len(self.error_log),
            'by_model_type': {
                model_type: {
                    'count': len(errors),
                    'recent_errors': errors[-5:]  # Last 5 errors
                }
                for model_type, errors in by_model.items()
            }
        }
        
        return summary
    
    def clear_error_log(self) -> None:
        """Clear the error log."""
        self.error_log.clear() 