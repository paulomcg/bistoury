"""
Database package for Bistoury.

Provides database connection management and schema operations.
"""

from .connection import DatabaseManager
from .schema import MarketDataSchema, DataInsertion, DataQuery
from .database_switcher import (
    DatabaseSwitcher, 
    TestDatabaseCompatibilityLayer, 
    CompatibleDataQuery,
    get_database_switcher,
    switch_database,
    get_compatible_query
)
from .compression import (
    DataCompressionManager,
    RetentionPolicy,
    CompressionStats
)
from .indices import (
    PerformanceIndexManager,
    IndexDefinition,
    IndexPerformanceStats
)
from .backup import (
    BackupManager,
    BackupMetadata,
    BackupConfig,
    RestorePoint
)

__all__ = [
    'DatabaseManager',
    'MarketDataSchema', 
    'DataInsertion',
    'DataQuery',
    'DatabaseSwitcher',
    'TestDatabaseCompatibilityLayer',
    'CompatibleDataQuery',
    'get_database_switcher',
    'switch_database',
    'get_compatible_query',
    'DataCompressionManager',
    'RetentionPolicy',
    'CompressionStats',
    'PerformanceIndexManager',
    'IndexDefinition',
    'IndexPerformanceStats',
    'BackupManager',
    'BackupMetadata',
    'BackupConfig',
    'RestorePoint'
] 