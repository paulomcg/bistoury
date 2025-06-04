"""
Data compression and archival system for Bistoury database.
Implements time-based partitioning, compression optimization, and data retention policies.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from ..config import Config
from .connection import DatabaseManager
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetentionPolicy:
    """Configuration for data retention policies."""
    table_name: str
    retention_days: int
    compression_after_days: int = 7
    archive_after_days: Optional[int] = None
    backup_before_delete: bool = True


@dataclass
class CompressionStats:
    """Statistics about compression operations."""
    table_name: str
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    rows_affected: int
    operation_time_seconds: float


class DataCompressionManager:
    """Manages data compression, partitioning, and archival for DuckDB."""
    
    def __init__(self, db_manager: DatabaseManager, config: Optional[Config] = None):
        self.db_manager = db_manager
        self.config = config or Config.load_from_env()
        
        # Default retention policies for trading data
        self.retention_policies = {
            "trades": RetentionPolicy(
                table_name="trades",
                retention_days=90,  # Keep trades for 3 months
                compression_after_days=7,  # Compress after 1 week
                archive_after_days=30  # Archive after 1 month
            ),
            "orderbook_snapshots": RetentionPolicy(
                table_name="orderbook_snapshots", 
                retention_days=30,  # Keep order books for 1 month
                compression_after_days=3,  # Compress after 3 days
                archive_after_days=14  # Archive after 2 weeks
            ),
            "candles_1m": RetentionPolicy(
                table_name="candles_1m",
                retention_days=365,  # Keep 1m candles for 1 year
                compression_after_days=30,  # Compress after 1 month
                archive_after_days=180  # Archive after 6 months
            ),
            "candles_5m": RetentionPolicy(
                table_name="candles_5m", 
                retention_days=730,  # Keep 5m candles for 2 years
                compression_after_days=30,
                archive_after_days=365
            ),
            "candles_15m": RetentionPolicy(
                table_name="candles_15m",
                retention_days=1095,  # Keep 15m candles for 3 years
                compression_after_days=60,
                archive_after_days=730
            ),
            "candles_1h": RetentionPolicy(
                table_name="candles_1h",
                retention_days=1825,  # Keep 1h candles for 5 years
                compression_after_days=90,
                archive_after_days=1095
            ),
            "candles_4h": RetentionPolicy(
                table_name="candles_4h",
                retention_days=3650,  # Keep 4h candles for 10 years
                compression_after_days=180,
                archive_after_days=1825
            ),
            "candles_1d": RetentionPolicy(
                table_name="candles_1d",
                retention_days=7300,  # Keep daily candles for 20 years
                compression_after_days=365,
                archive_after_days=3650
            ),
            "funding_rates": RetentionPolicy(
                table_name="funding_rates",
                retention_days=365,  # Keep funding rates for 1 year
                compression_after_days=30,
                archive_after_days=180
            )
        }
        
    def configure_duckdb_compression(self) -> None:
        """Configure optimal DuckDB compression settings."""
        try:
            logger.info("Configuring DuckDB compression settings")
            
            # Configure compression level (3 is good balance of speed vs compression)
            # Note: Many DuckDB compression settings are built-in and don't need manual configuration
            
            # Enable optimizations for better storage
            try:
                self.db_manager.execute("SET enable_object_cache = true")
            except Exception:
                pass  # Not all versions support this
            
            try:
                # Configure memory settings for better compression
                self.db_manager.execute("SET memory_limit = '1GB'")
            except Exception:
                pass  # Fallback if memory_limit setting isn't available
            
            logger.info("DuckDB compression configuration complete")
            
        except Exception as e:
            logger.error(f"Failed to configure DuckDB compression: {e}")
            raise
            
    def create_partitioned_tables(self) -> None:
        """Create time-partitioned versions of main tables for better performance."""
        try:
            logger.info("Creating time-partitioned tables")
            
            # Create partitioned trades table by month
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS trades_partitioned AS 
                SELECT *, 
                    date_trunc('month', timestamp) as partition_month,
                    date_trunc('year', timestamp) as partition_year
                FROM trades
                WHERE 1=0  -- Empty table with structure
            """)
            
            # Create partitioned orderbook table by day
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots_partitioned AS
                SELECT *,
                    date_trunc('day', timestamp) as partition_day,
                    date_trunc('month', timestamp) as partition_month
                FROM orderbook_snapshots 
                WHERE 1=0  -- Empty table with structure
            """)
            
            # Create indices on partition columns
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_partition_month 
                ON trades_partitioned(partition_month, symbol)
            """)
            
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_orderbook_partition_day
                ON orderbook_snapshots_partitioned(partition_day, symbol)
            """)
            
            logger.info("Time-partitioned tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create partitioned tables: {e}")
            raise
            
    def _get_timestamp_column(self, table_name: str) -> str:
        """Get the correct timestamp column name for a table."""
        if table_name.startswith('candles_'):
            return 'timestamp_start'
        elif table_name in ['symbols']:
            return None  # These tables don't have timestamp columns
        else:
            return 'timestamp'
    
    def compress_old_data(self, table_name: str, days_old: int = 7) -> CompressionStats:
        """Compress data older than specified days using DuckDB's compression."""
        try:
            start_time = datetime.now()
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            logger.info(f"Compressing {table_name} data older than {days_old} days")
            
            # Get original table size
            original_size = self._get_table_size(table_name)
            
            # Get the correct timestamp column for this table
            timestamp_col = self._get_timestamp_column(table_name)
            
            # Count rows to be compressed
            row_count_result = self.db_manager.execute(f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE {timestamp_col} < ?
            """, (cutoff_date,))
            rows_affected = row_count_result[0][0] if row_count_result else 0
            
            if rows_affected == 0:
                logger.info(f"No data older than {days_old} days found in {table_name}")
                return CompressionStats(
                    table_name=table_name,
                    original_size_mb=original_size,
                    compressed_size_mb=original_size,
                    compression_ratio=1.0,
                    rows_affected=0,
                    operation_time_seconds=0.0
                )
            
            # Create compressed table with old data
            compressed_table_name = f"{table_name}_compressed_{cutoff_date.strftime('%Y%m%d')}"
            
            self.db_manager.execute(f"""
                CREATE TABLE {compressed_table_name} AS 
                SELECT * FROM {table_name} 
                WHERE {timestamp_col} < ?
                ORDER BY {timestamp_col}, symbol
            """, (cutoff_date,))
            
            # Remove old data from main table
            self.db_manager.execute(f"""
                DELETE FROM {table_name} 
                WHERE {timestamp_col} < ?
            """, (cutoff_date,))
            
            # Vacuum to reclaim space
            self.db_manager.execute(f"VACUUM {table_name}")
            
            # Get new table size
            new_size = self._get_table_size(table_name)
            compressed_table_size = self._get_table_size(compressed_table_name)
            total_compressed_size = new_size + compressed_table_size
            
            operation_time = (datetime.now() - start_time).total_seconds()
            compression_ratio = original_size / total_compressed_size if total_compressed_size > 0 else 1.0
            
            stats = CompressionStats(
                table_name=table_name,
                original_size_mb=original_size,
                compressed_size_mb=total_compressed_size,
                compression_ratio=compression_ratio,
                rows_affected=rows_affected,
                operation_time_seconds=operation_time
            )
            
            logger.info(f"Compression complete: {rows_affected:,} rows, "
                       f"{compression_ratio:.2f}x compression ratio, "
                       f"{operation_time:.2f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compress {table_name}: {e}")
            raise
            
    def archive_old_data(self, table_name: str, days_old: int = 30, 
                        archive_path: Optional[str] = None) -> Dict[str, Any]:
        """Archive very old data to separate files for long-term storage."""
        try:
            start_time = datetime.now()
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            if not archive_path:
                archive_path = Path(self.config.database.backup_path) / "archives"
                archive_path.mkdir(parents=True, exist_ok=True)
            
            archive_file = archive_path / f"{table_name}_archive_{cutoff_date.strftime('%Y%m%d')}.parquet"
            
            logger.info(f"Archiving {table_name} data older than {days_old} days to {archive_file}")
            
            # Get the correct timestamp column for this table
            timestamp_col = self._get_timestamp_column(table_name)
            
            # Count rows to archive
            row_count_result = self.db_manager.execute(f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE {timestamp_col} < ?
            """, (cutoff_date,))
            rows_to_archive = row_count_result[0][0] if row_count_result else 0
            
            if rows_to_archive == 0:
                logger.info(f"No data to archive from {table_name}")
                return {
                    "table_name": table_name,
                    "rows_archived": 0,
                    "archive_file": None,
                    "operation_time_seconds": 0.0
                }
            
            # Export to Parquet (highly compressed format)
            self.db_manager.execute(f"""
                COPY (
                    SELECT * FROM {table_name} 
                    WHERE {timestamp_col} < ?
                    ORDER BY {timestamp_col}, symbol
                ) TO '{archive_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """, (cutoff_date,))
            
            # Verify archive was created successfully
            archive_verification = self.db_manager.execute(f"""
                SELECT COUNT(*) FROM parquet_scan('{archive_file}')
            """)
            archived_rows = archive_verification[0][0] if archive_verification else 0
            
            if archived_rows != rows_to_archive:
                raise Exception(f"Archive verification failed: expected {rows_to_archive}, got {archived_rows}")
            
            # Remove archived data from main table
            self.db_manager.execute(f"""
                DELETE FROM {table_name}
                WHERE {timestamp_col} < ?
            """, (cutoff_date,))
            
            # Vacuum to reclaim space
            self.db_manager.execute(f"VACUUM {table_name}")
            
            operation_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "table_name": table_name,
                "rows_archived": archived_rows,
                "archive_file": str(archive_file),
                "archive_size_mb": archive_file.stat().st_size / (1024 * 1024),
                "operation_time_seconds": operation_time
            }
            
            logger.info(f"Archival complete: {archived_rows:,} rows archived to {archive_file}, "
                       f"{operation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to archive {table_name}: {e}")
            raise
            
    def enforce_retention_policies(self) -> List[Dict[str, Any]]:
        """Enforce data retention policies across all tables."""
        results = []
        
        for policy_name, policy in self.retention_policies.items():
            try:
                logger.info(f"Enforcing retention policy for {policy.table_name}")
                
                # Check if table exists
                table_exists = self.db_manager.execute(f"""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = '{policy.table_name}'
                """)
                
                if not table_exists or table_exists[0][0] == 0:
                    logger.warning(f"Table {policy.table_name} does not exist, skipping")
                    continue
                
                policy_result = {
                    "table_name": policy.table_name,
                    "policy": policy_name,
                    "actions": []
                }
                
                # Compress old data
                if policy.compression_after_days:
                    try:
                        compression_stats = self.compress_old_data(
                            policy.table_name, 
                            policy.compression_after_days
                        )
                        policy_result["actions"].append({
                            "action": "compression",
                            "stats": compression_stats
                        })
                    except Exception as e:
                        logger.error(f"Compression failed for {policy.table_name}: {e}")
                
                # Archive very old data
                if policy.archive_after_days:
                    try:
                        archive_result = self.archive_old_data(
                            policy.table_name,
                            policy.archive_after_days
                        )
                        policy_result["actions"].append({
                            "action": "archive",
                            "result": archive_result
                        })
                    except Exception as e:
                        logger.error(f"Archival failed for {policy.table_name}: {e}")
                
                # Delete expired data
                try:
                    deletion_result = self._delete_expired_data(policy)
                    if deletion_result["rows_deleted"] > 0:
                        policy_result["actions"].append({
                            "action": "deletion",
                            "result": deletion_result
                        })
                except Exception as e:
                    logger.error(f"Deletion failed for {policy.table_name}: {e}")
                
                results.append(policy_result)
                
            except Exception as e:
                logger.error(f"Failed to enforce retention policy for {policy_name}: {e}")
                results.append({
                    "table_name": policy.table_name,
                    "policy": policy_name,
                    "error": str(e)
                })
        
        return results
        
    def _delete_expired_data(self, policy: RetentionPolicy) -> Dict[str, Any]:
        """Delete data that has exceeded retention period."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy.retention_days)
        
        # Get the correct timestamp column for this table
        timestamp_col = self._get_timestamp_column(policy.table_name)
        
        # Count rows to delete
        row_count_result = self.db_manager.execute(f"""
            SELECT COUNT(*) FROM {policy.table_name}
            WHERE {timestamp_col} < ?
        """, (cutoff_date,))
        rows_to_delete = row_count_result[0][0] if row_count_result else 0
        
        if rows_to_delete == 0:
            return {"rows_deleted": 0, "operation_time_seconds": 0.0}
        
        start_time = datetime.now()
        
        # Backup before deletion if policy requires it
        if policy.backup_before_delete:
            backup_file = Path(self.config.database.backup_path) / "pre_deletion_backups" / f"{policy.table_name}_backup_{cutoff_date.strftime('%Y%m%d')}.parquet"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.db_manager.execute(f"""
                COPY (
                    SELECT * FROM {policy.table_name}
                    WHERE {timestamp_col} < ?
                ) TO '{backup_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """, (cutoff_date,))
        
        # Delete expired data
        self.db_manager.execute(f"""
            DELETE FROM {policy.table_name}
            WHERE {timestamp_col} < ?
        """, (cutoff_date,))
        
        # Vacuum to reclaim space
        self.db_manager.execute(f"VACUUM {policy.table_name}")
        
        operation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Deleted {rows_to_delete:,} expired rows from {policy.table_name}")
        
        return {
            "rows_deleted": rows_to_delete,
            "cutoff_date": cutoff_date.isoformat(),
            "backup_created": policy.backup_before_delete,
            "operation_time_seconds": operation_time
        }
        
    def _get_table_size(self, table_name: str) -> float:
        """Get table size in MB."""
        try:
            # DuckDB doesn't have direct table size queries, so we estimate
            # based on the database file size and table row counts
            result = self.db_manager.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = result[0][0] if result else 0
            
            # Rough estimation: assume average row size varies by table type
            row_size_estimates = {
                "trades": 200,  # bytes per trade record
                "orderbook_snapshots": 2000,  # bytes per orderbook snapshot
                "funding_rates": 100,  # bytes per funding rate record
                "symbols": 300,  # bytes per symbol record
            }
            
            # Default to 150 bytes per row for candle data
            estimated_row_size = row_size_estimates.get(
                table_name.split('_')[0] if '_' in table_name else table_name, 
                150
            )
            
            estimated_size_mb = (row_count * estimated_row_size) / (1024 * 1024)
            return estimated_size_mb
            
        except Exception as e:
            logger.warning(f"Could not estimate size for {table_name}: {e}")
            return 0.0
            
    def get_compression_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compression and storage report."""
        try:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database_path": str(self.db_manager.config.database.path),
                "tables": {},
                "total_estimated_size_mb": 0.0,
                "retention_policies": {}
            }
            
            # Get table information
            tables_result = self.db_manager.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                ORDER BY table_name
            """)
            
            for (table_name,) in tables_result:
                try:
                    # Get row count
                    row_count_result = self.db_manager.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = row_count_result[0][0] if row_count_result else 0
                    
                    # Get estimated size
                    estimated_size = self._get_table_size(table_name)
                    
                    # Get date range if table has timestamp column
                    date_range = None
                    try:
                        # Get the correct timestamp column for this table
                        timestamp_col = self._get_timestamp_column(table_name)
                        
                        if timestamp_col:  # Only query if table has a timestamp column
                            date_result = self.db_manager.execute(f"""
                                SELECT MIN({timestamp_col}), MAX({timestamp_col}) 
                                FROM {table_name}
                                WHERE {timestamp_col} IS NOT NULL
                            """)
                            if date_result and date_result[0][0]:
                                date_range = {
                                    "oldest": date_result[0][0].isoformat() if date_result[0][0] else None,
                                    "newest": date_result[0][1].isoformat() if date_result[0][1] else None
                                }
                    except:
                        pass  # Table might not have timestamp column or other error
                    
                    report["tables"][table_name] = {
                        "row_count": row_count,
                        "estimated_size_mb": estimated_size,
                        "date_range": date_range
                    }
                    
                    report["total_estimated_size_mb"] += estimated_size
                    
                except Exception as e:
                    logger.warning(f"Could not get info for table {table_name}: {e}")
            
            # Add retention policy information
            for policy_name, policy in self.retention_policies.items():
                report["retention_policies"][policy_name] = {
                    "table_name": policy.table_name,
                    "retention_days": policy.retention_days,
                    "compression_after_days": policy.compression_after_days,
                    "archive_after_days": policy.archive_after_days,
                    "backup_before_delete": policy.backup_before_delete
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compression report: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()} 