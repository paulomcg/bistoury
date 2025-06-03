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
    'get_compatible_query'
] 