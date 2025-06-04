"""
Backup and restore system for Bistoury database.
Implements automated backups, point-in-time recovery, and disaster recovery procedures.
"""

import os
import shutil
import gzip
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import hashlib
import subprocess
import threading
import time
import schedule

from .connection import DatabaseManager
from ..config import Config
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for backup files."""
    backup_id: str
    timestamp: datetime
    database_path: str
    backup_path: str
    backup_type: str  # full, incremental, differential
    file_size_bytes: int
    compressed: bool
    checksum_sha256: str
    table_counts: Dict[str, int]
    retention_days: int
    description: str = ""


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    backup_directory: str
    retention_days: int = 30
    compress_backups: bool = True
    verify_backups: bool = True
    max_backup_size_gb: int = 10
    backup_prefix: str = "bistoury_backup"


@dataclass
class RestorePoint:
    """Point-in-time restore information."""
    timestamp: datetime
    backup_id: str
    description: str
    data_integrity_verified: bool


class BackupManager:
    """Manages database backup and restore operations."""
    
    def __init__(self, db_manager: DatabaseManager, config: Optional[Config] = None):
        self.db_manager = db_manager
        self.config = config or Config.load_from_env()
        
        # Setup backup configuration
        self.backup_config = BackupConfig(
            backup_directory=str(Path(self.config.database.backup_path) / "backups"),
            retention_days=30,
            compress_backups=True,
            verify_backups=True
        )
        
        # Ensure backup directory exists
        Path(self.backup_config.backup_directory).mkdir(parents=True, exist_ok=True)
        
        # Backup metadata storage
        self.metadata_file = Path(self.backup_config.backup_directory) / "backup_metadata.json"
        self.backup_metadata: List[BackupMetadata] = self._load_backup_metadata()
        
        # Schedule configuration
        self.scheduler_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
    def create_full_backup(self, description: str = "") -> BackupMetadata:
        """Create a complete database backup."""
        try:
            start_time = datetime.now(timezone.utc)
            backup_id = f"full_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting full backup: {backup_id}")
            
            # Create backup filename
            backup_filename = f"{self.backup_config.backup_prefix}_{backup_id}.duckdb"
            if self.backup_config.compress_backups:
                backup_filename += ".gz"
            
            backup_path = Path(self.backup_config.backup_directory) / backup_filename
            
            # Get current database path
            db_path = Path(self.db_manager.config.database.path)
            
            if not db_path.exists():
                raise FileNotFoundError(f"Database file not found: {db_path}")
            
            # Close all connections temporarily for consistent backup
            logger.info("Temporarily closing database connections for backup")
            self.db_manager.close_all_connections()
            
            try:
                # Copy database file
                if self.backup_config.compress_backups:
                    self._compress_file(db_path, backup_path)
                else:
                    shutil.copy2(db_path, backup_path)
                
                # Calculate checksum
                checksum = self._calculate_checksum(backup_path)
                
                # Get file size
                file_size = backup_path.stat().st_size
                
                # Reconnect to database to get table counts
                conn = self.db_manager.get_connection()  # Use correct method name
                table_counts = self._get_table_counts()
                
                # Create metadata
                metadata = BackupMetadata(
                    backup_id=backup_id,
                    timestamp=start_time,
                    database_path=str(db_path),
                    backup_path=str(backup_path),
                    backup_type="full",
                    file_size_bytes=file_size,
                    compressed=self.backup_config.compress_backups,
                    checksum_sha256=checksum,
                    table_counts=table_counts,
                    retention_days=self.backup_config.retention_days,
                    description=description or f"Full backup created at {start_time.isoformat()}"
                )
                
                # Verify backup if configured
                if self.backup_config.verify_backups:
                    if self._verify_backup(metadata):
                        logger.info(f"Backup verification successful: {backup_id}")
                    else:
                        raise Exception(f"Backup verification failed: {backup_id}")
                
                # Save metadata
                self.backup_metadata.append(metadata)
                self._save_backup_metadata()
                
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                size_mb = file_size / (1024 * 1024)
                
                logger.info(f"Full backup complete: {backup_id}, {size_mb:.1f}MB, {duration:.2f}s")
                
                return metadata
                
            finally:
                # Database connections will be created on next use (no need to explicitly ensure)
                pass
            
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            raise
            
    def create_incremental_backup(self, description: str = "") -> Optional[BackupMetadata]:
        """Create an incremental backup containing only changes since last backup."""
        try:
            # Find the most recent backup
            if not self.backup_metadata:
                logger.info("No previous backups found, creating full backup instead")
                return self.create_full_backup(f"Initial full backup: {description}")
            
            last_backup = max(self.backup_metadata, key=lambda x: x.timestamp)
            start_time = datetime.now(timezone.utc)
            backup_id = f"incr_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting incremental backup: {backup_id} (since {last_backup.backup_id})")
            
            # For DuckDB, we'll export changed data to Parquet files
            backup_dir = Path(self.backup_config.backup_directory) / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            # Get tables with data changes since last backup
            changed_tables = self._get_tables_with_changes_since(last_backup.timestamp)
            
            if not changed_tables:
                logger.info("No data changes detected since last backup")
                return None
            
            total_size = 0
            exported_files = {}
            
            for table_name, row_count in changed_tables.items():
                try:
                    # Export changed data to Parquet
                    parquet_file = backup_dir / f"{table_name}_incremental.parquet"
                    
                    export_sql = f"""
                        COPY (
                            SELECT * FROM {table_name}
                            WHERE timestamp > ?
                            ORDER BY timestamp
                        ) TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """
                    
                    self.db_manager.execute(export_sql, (last_backup.timestamp,))
                    
                    if parquet_file.exists():
                        file_size = parquet_file.stat().st_size
                        total_size += file_size
                        exported_files[table_name] = {
                            "file": str(parquet_file),
                            "size_bytes": file_size,
                            "row_count": row_count
                        }
                        
                        logger.info(f"Exported {row_count:,} rows from {table_name} "
                                   f"({file_size / 1024 / 1024:.1f}MB)")
                    
                except Exception as e:
                    logger.warning(f"Failed to export incremental data for {table_name}: {e}")
            
            if not exported_files:
                logger.info("No data successfully exported for incremental backup")
                return None
            
            # Create metadata file for incremental backup
            increment_metadata = {
                "backup_id": backup_id,
                "backup_type": "incremental",
                "base_backup_id": last_backup.backup_id,
                "timestamp": start_time.isoformat(),
                "exported_files": exported_files,
                "total_size_bytes": total_size
            }
            
            metadata_file = backup_dir / "incremental_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(increment_metadata, f, indent=2)
            
            # Compress the entire incremental backup directory
            if self.backup_config.compress_backups:
                compressed_path = Path(self.backup_config.backup_directory) / f"{backup_id}.tar.gz"
                self._compress_directory(backup_dir, compressed_path)
                
                # Remove uncompressed directory
                shutil.rmtree(backup_dir)
                backup_path = compressed_path
                file_size = compressed_path.stat().st_size
            else:
                backup_path = backup_dir
                file_size = total_size
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=start_time,
                database_path=str(self.db_manager.config.database.path),
                backup_path=str(backup_path),
                backup_type="incremental",
                file_size_bytes=file_size,
                compressed=self.backup_config.compress_backups,
                checksum_sha256=checksum,
                table_counts=changed_tables,
                retention_days=self.backup_config.retention_days,
                description=description or f"Incremental backup since {last_backup.backup_id}"
            )
            
            self.backup_metadata.append(metadata)
            self._save_backup_metadata()
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            size_mb = file_size / (1024 * 1024)
            
            logger.info(f"Incremental backup complete: {backup_id}, {size_mb:.1f}MB, {duration:.2f}s")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Incremental backup failed: {e}")
            raise
            
    def restore_from_backup(self, backup_id: str, target_path: Optional[str] = None) -> bool:
        """Restore database from a specific backup."""
        try:
            # Find the backup metadata
            backup_metadata = None
            for metadata in self.backup_metadata:
                if metadata.backup_id == backup_id:
                    backup_metadata = metadata
                    break
            
            if not backup_metadata:
                raise ValueError(f"Backup not found: {backup_id}")
            
            logger.info(f"Starting restore from backup: {backup_id}")
            
            # Verify backup integrity before restore
            if not self._verify_backup(backup_metadata):
                raise Exception(f"Backup integrity verification failed: {backup_id}")
            
            # Determine target path
            if not target_path:
                target_path = self.db_manager.config.database.path
                
                # Create backup of current database before restore
                current_backup_id = f"pre_restore_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                try:
                    self.create_full_backup(f"Pre-restore backup before restoring {backup_id}")
                    logger.info(f"Created pre-restore backup: {current_backup_id}")
                except Exception as e:
                    logger.warning(f"Failed to create pre-restore backup: {e}")
            
            target_path = Path(target_path)
            backup_path = Path(backup_metadata.backup_path)
            
            # Close all database connections
            self.db_manager.close_all_connections()
            
            try:
                if backup_metadata.backup_type == "full":
                    # Restore full backup
                    if backup_metadata.compressed:
                        self._decompress_file(backup_path, target_path)
                    else:
                        shutil.copy2(backup_path, target_path)
                        
                elif backup_metadata.backup_type == "incremental":
                    # Restore incremental backup (requires base backup)
                    self._restore_incremental_backup(backup_metadata, target_path)
                    
                else:
                    raise ValueError(f"Unsupported backup type: {backup_metadata.backup_type}")
                
                # Verify restored database
                conn = self.db_manager.get_connection()  # Use correct method name
                restored_table_counts = self._get_table_counts()
                
                # Compare table counts (allow for some variation due to incremental nature)
                verification_passed = True
                for table, expected_count in backup_metadata.table_counts.items():
                    actual_count = restored_table_counts.get(table, 0)
                    if backup_metadata.backup_type == "full" and actual_count != expected_count:
                        logger.warning(f"Table count mismatch for {table}: expected {expected_count}, got {actual_count}")
                        verification_passed = False
                
                if verification_passed:
                    logger.info(f"Database restore completed successfully: {backup_id}")
                    return True
                else:
                    logger.error(f"Database restore verification failed: {backup_id}")
                    return False
                    
            finally:
                # Ensure database connection is restored (connection will be created on next use)
                pass  # No need to explicitly ensure connection as it's created on demand
                
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
            
    def list_available_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata."""
        backups = []
        
        for metadata in sorted(self.backup_metadata, key=lambda x: x.timestamp, reverse=True):
            backup_info = {
                "backup_id": metadata.backup_id,
                "timestamp": metadata.timestamp.isoformat(),
                "backup_type": metadata.backup_type,
                "size_mb": metadata.file_size_bytes / (1024 * 1024),
                "compressed": metadata.compressed,
                "description": metadata.description,
                "table_counts": metadata.table_counts,
                "retention_expires": (metadata.timestamp + timedelta(days=metadata.retention_days)).isoformat(),
                "backup_exists": Path(metadata.backup_path).exists()
            }
            backups.append(backup_info)
        
        return backups
        
    def cleanup_expired_backups(self) -> List[str]:
        """Remove backups that have exceeded their retention period."""
        try:
            now = datetime.now(timezone.utc)
            expired_backups = []
            
            for metadata in self.backup_metadata[:]:  # Copy list to safely modify during iteration
                expiry_date = metadata.timestamp + timedelta(days=metadata.retention_days)
                
                if now > expiry_date:
                    try:
                        backup_path = Path(metadata.backup_path)
                        if backup_path.exists():
                            if backup_path.is_dir():
                                shutil.rmtree(backup_path)
                            else:
                                backup_path.unlink()
                            
                            logger.info(f"Deleted expired backup: {metadata.backup_id}")
                            
                        expired_backups.append(metadata.backup_id)
                        self.backup_metadata.remove(metadata)
                        
                    except Exception as e:
                        logger.error(f"Failed to delete expired backup {metadata.backup_id}: {e}")
            
            if expired_backups:
                self._save_backup_metadata()
                logger.info(f"Cleanup complete: removed {len(expired_backups)} expired backups")
            
            return expired_backups
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return []
            
    def start_automated_backups(self, daily_time: str = "02:00", weekly_day: str = "sunday") -> None:
        """Start automated backup scheduler."""
        try:
            if self.scheduler_running:
                logger.warning("Automated backup scheduler is already running")
                return
            
            # Schedule daily incremental backups
            schedule.every().day.at(daily_time).do(
                self._scheduled_incremental_backup
            )
            
            # Schedule weekly full backups
            getattr(schedule.every(), weekly_day.lower()).at(daily_time).do(
                self._scheduled_full_backup
            )
            
            # Schedule daily cleanup
            schedule.every().day.at("03:00").do(
                self.cleanup_expired_backups
            )
            
            # Start scheduler thread
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info(f"Automated backup scheduler started: daily at {daily_time}, weekly on {weekly_day}")
            
        except Exception as e:
            logger.error(f"Failed to start automated backup scheduler: {e}")
            raise
            
    def stop_automated_backups(self) -> None:
        """Stop automated backup scheduler."""
        try:
            if not self.scheduler_running:
                logger.warning("Automated backup scheduler is not running")
                return
            
            self.scheduler_running = False
            schedule.clear()
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            logger.info("Automated backup scheduler stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop automated backup scheduler: {e}")
            
    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup system status."""
        try:
            now = datetime.now(timezone.utc)
            
            # Calculate backup statistics
            total_backups = len(self.backup_metadata)
            total_size_bytes = sum(metadata.file_size_bytes for metadata in self.backup_metadata)
            
            # Find latest backups
            latest_full = None
            latest_incremental = None
            
            for metadata in sorted(self.backup_metadata, key=lambda x: x.timestamp, reverse=True):
                if metadata.backup_type == "full" and not latest_full:
                    latest_full = metadata
                elif metadata.backup_type == "incremental" and not latest_incremental:
                    latest_incremental = metadata
                    
                if latest_full and latest_incremental:
                    break
            
            # Check backup health
            missing_backups = []
            for metadata in self.backup_metadata:
                if not Path(metadata.backup_path).exists():
                    missing_backups.append(metadata.backup_id)
            
            status = {
                "timestamp": now.isoformat(),
                "backup_directory": self.backup_config.backup_directory,
                "scheduler_running": self.scheduler_running,
                "total_backups": total_backups,
                "total_size_mb": total_size_bytes / (1024 * 1024),
                "latest_full_backup": {
                    "backup_id": latest_full.backup_id if latest_full else None,
                    "timestamp": latest_full.timestamp.isoformat() if latest_full else None,
                    "age_hours": (now - latest_full.timestamp).total_seconds() / 3600 if latest_full else None
                },
                "latest_incremental_backup": {
                    "backup_id": latest_incremental.backup_id if latest_incremental else None,
                    "timestamp": latest_incremental.timestamp.isoformat() if latest_incremental else None,
                    "age_hours": (now - latest_incremental.timestamp).total_seconds() / 3600 if latest_incremental else None
                },
                "backup_health": {
                    "missing_backup_files": missing_backups,
                    "healthy": len(missing_backups) == 0
                },
                "configuration": {
                    "retention_days": self.backup_config.retention_days,
                    "compress_backups": self.backup_config.compress_backups,
                    "verify_backups": self.backup_config.verify_backups,
                    "max_backup_size_gb": self.backup_config.max_backup_size_gb
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
            
    def _scheduled_full_backup(self) -> None:
        """Execute scheduled full backup."""
        try:
            logger.info("Executing scheduled full backup")
            self.create_full_backup("Scheduled weekly full backup")
        except Exception as e:
            logger.error(f"Scheduled full backup failed: {e}")
            
    def _scheduled_incremental_backup(self) -> None:
        """Execute scheduled incremental backup."""
        try:
            logger.info("Executing scheduled incremental backup")
            result = self.create_incremental_backup("Scheduled daily incremental backup")
            if not result:
                logger.info("No changes detected for incremental backup")
        except Exception as e:
            logger.error(f"Scheduled incremental backup failed: {e}")
            
    def _run_scheduler(self) -> None:
        """Run the backup scheduler in a separate thread."""
        logger.info("Backup scheduler thread started")
        
        while self.scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
        
        logger.info("Backup scheduler thread stopped")
        
    def _compress_file(self, source_path: Path, target_path: Path) -> None:
        """Compress a file using gzip."""
        with open(source_path, 'rb') as f_in:
            with gzip.open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    def _decompress_file(self, source_path: Path, target_path: Path) -> None:
        """Decompress a gzip file."""
        with gzip.open(source_path, 'rb') as f_in:
            with open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    def _compress_directory(self, source_dir: Path, target_path: Path) -> None:
        """Compress a directory to tar.gz."""
        import tarfile
        with tarfile.open(target_path, 'w:gz') as tar:
            tar.add(source_dir, arcname=source_dir.name)
            
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        
        if file_path.is_dir():
            # For directories, calculate hash of all files
            for file in sorted(file_path.rglob('*')):
                if file.is_file():
                    with open(file, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_sha256.update(chunk)
        else:
            # For single files
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
        
    def _verify_backup(self, metadata: BackupMetadata) -> bool:
        """Verify backup integrity by checking checksum and basic structure."""
        try:
            backup_path = Path(metadata.backup_path)
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Verify checksum
            current_checksum = self._calculate_checksum(backup_path)
            if current_checksum != metadata.checksum_sha256:
                logger.error(f"Backup checksum mismatch: expected {metadata.checksum_sha256}, got {current_checksum}")
                return False
            
            # Verify file size
            current_size = backup_path.stat().st_size
            if current_size != metadata.file_size_bytes:
                logger.error(f"Backup size mismatch: expected {metadata.file_size_bytes}, got {current_size}")
                return False
            
            logger.debug(f"Backup verification successful: {metadata.backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
            
    def _get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        table_counts = {}
        
        try:
            # Get list of tables
            tables_result = self.db_manager.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main'
            """)
            
            for (table_name,) in tables_result:
                try:
                    count_result = self.db_manager.execute(f"SELECT COUNT(*) FROM {table_name}")
                    table_counts[table_name] = count_result[0][0] if count_result else 0
                except Exception as e:
                    logger.warning(f"Could not count rows in {table_name}: {e}")
                    table_counts[table_name] = 0
            
        except Exception as e:
            logger.error(f"Failed to get table counts: {e}")
        
        return table_counts
        
    def _get_tables_with_changes_since(self, timestamp: datetime) -> Dict[str, int]:
        """Get tables that have data changes since the specified timestamp."""
        changed_tables = {}
        
        try:
            # Check tables with timestamp columns for recent changes
            timestamp_tables = ["trades", "orderbook_snapshots", "funding_rates"] + \
                              [f"candles_{tf}" for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]]
            
            for table_name in timestamp_tables:
                try:
                    # Check if table exists
                    table_exists = self.db_manager.execute(f"""
                        SELECT COUNT(*) FROM information_schema.tables
                        WHERE table_name = '{table_name}'
                    """)
                    
                    if not table_exists or table_exists[0][0] == 0:
                        continue
                    
                    # Count rows with timestamp > last backup
                    count_result = self.db_manager.execute(f"""
                        SELECT COUNT(*) FROM {table_name}
                        WHERE timestamp > ?
                    """, (timestamp,))
                    
                    row_count = count_result[0][0] if count_result else 0
                    if row_count > 0:
                        changed_tables[table_name] = row_count
                        
                except Exception as e:
                    logger.warning(f"Could not check changes for {table_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to check for table changes: {e}")
        
        return changed_tables
        
    def _restore_incremental_backup(self, metadata: BackupMetadata, target_path: Path) -> None:
        """Restore an incremental backup (placeholder for complex logic)."""
        raise NotImplementedError("Incremental backup restore is not yet implemented")
        
    def _load_backup_metadata(self) -> List[BackupMetadata]:
        """Load backup metadata from storage."""
        if not self.metadata_file.exists():
            return []
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata_list = []
            for item in data:
                # Convert string timestamp back to datetime
                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                metadata_list.append(BackupMetadata(**item))
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return []
            
    def _save_backup_metadata(self) -> None:
        """Save backup metadata to storage."""
        try:
            # Convert to JSON-serializable format
            data = []
            for metadata in self.backup_metadata:
                item = asdict(metadata)
                item['timestamp'] = metadata.timestamp.isoformat()
                data.append(item)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}") 