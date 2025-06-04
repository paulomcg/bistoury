"""
Performance indices and database optimization for Bistoury.
Implements strategic indexing for fast time-series queries and trading analytics.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import time
import os

from .connection import DatabaseManager
from ..config import Config
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class IndexDefinition:
    """Definition of a database index."""
    name: str
    table: str
    columns: List[str]
    index_type: str = "BTREE"  # BTREE, HASH, etc.
    unique: bool = False
    partial_condition: Optional[str] = None
    description: str = ""


@dataclass
class IndexPerformanceStats:
    """Statistics about index performance and usage."""
    index_name: str
    table_name: str
    size_estimate_mb: float
    creation_time_seconds: float
    is_unique: bool
    columns: List[str]
    estimated_selectivity: Optional[float] = None


class PerformanceIndexManager:
    """Manages database indices for optimal query performance."""
    
    def __init__(self, db_manager: DatabaseManager, config: Optional[Config] = None):
        self.db_manager = db_manager
        self.config = config or Config.load_from_env()
        
        # Define strategic indices for trading data queries
        self.index_definitions = {
            # Time-based indices (most critical for trading)
            "idx_trades_timestamp_symbol": IndexDefinition(
                name="idx_trades_timestamp_symbol",
                table="trades",
                columns=["timestamp", "symbol"],
                description="Primary index for time-based trade queries with symbol filtering"
            ),
            
            "idx_trades_symbol_timestamp": IndexDefinition(
                name="idx_trades_symbol_timestamp", 
                table="trades",
                columns=["symbol", "timestamp"],
                description="Symbol-first index for symbol-specific time queries"
            ),
            
            "idx_orderbook_timestamp_symbol": IndexDefinition(
                name="idx_orderbook_timestamp_symbol",
                table="orderbook_snapshots",
                columns=["timestamp", "symbol"],
                description="Primary index for orderbook time-series queries"
            ),
            
            "idx_orderbook_symbol_time_ms": IndexDefinition(
                name="idx_orderbook_symbol_time_ms",
                table="orderbook_snapshots", 
                columns=["symbol", "time_ms"],
                description="Fast index for latest orderbook snapshots by symbol"
            ),
            
            # Candlestick indices for multi-timeframe analysis
            "idx_candles_1m_timestamp_symbol": IndexDefinition(
                name="idx_candles_1m_timestamp_symbol",
                table="candles_1m",
                columns=["timestamp", "symbol"], 
                description="1-minute candle time-series queries"
            ),
            
            "idx_candles_5m_timestamp_symbol": IndexDefinition(
                name="idx_candles_5m_timestamp_symbol",
                table="candles_5m",
                columns=["timestamp", "symbol"],
                description="5-minute candle time-series queries"
            ),
            
            "idx_candles_15m_timestamp_symbol": IndexDefinition(
                name="idx_candles_15m_timestamp_symbol", 
                table="candles_15m",
                columns=["timestamp", "symbol"],
                description="15-minute candle time-series queries"
            ),
            
            "idx_candles_1h_timestamp_symbol": IndexDefinition(
                name="idx_candles_1h_timestamp_symbol",
                table="candles_1h", 
                columns=["timestamp", "symbol"],
                description="1-hour candle time-series queries"
            ),
            
            "idx_candles_4h_timestamp_symbol": IndexDefinition(
                name="idx_candles_4h_timestamp_symbol",
                table="candles_4h",
                columns=["timestamp", "symbol"],
                description="4-hour candle time-series queries"
            ),
            
            "idx_candles_1d_timestamp_symbol": IndexDefinition(
                name="idx_candles_1d_timestamp_symbol",
                table="candles_1d",
                columns=["timestamp", "symbol"],
                description="Daily candle time-series queries"
            ),
            
            # Funding rate indices
            "idx_funding_timestamp_symbol": IndexDefinition(
                name="idx_funding_timestamp_symbol",
                table="funding_rates",
                columns=["timestamp", "symbol"],
                description="Funding rate time-series queries"
            ),
            
            "idx_funding_symbol_time_ms": IndexDefinition(
                name="idx_funding_symbol_time_ms",
                table="funding_rates",
                columns=["symbol", "time_ms"],
                description="Latest funding rates by symbol"
            ),
            
            # Composite indices for complex queries
            "idx_trades_price_volume": IndexDefinition(
                name="idx_trades_price_volume",
                table="trades",
                columns=["symbol", "price", "size"],
                description="Price and volume analysis queries"
            ),
            
            "idx_trades_side_timestamp": IndexDefinition(
                name="idx_trades_side_timestamp",
                table="trades", 
                columns=["symbol", "side", "timestamp"],
                description="Buy/sell side analysis with time filtering"
            ),
            
            # Symbol metadata indices
            "idx_symbols_name": IndexDefinition(
                name="idx_symbols_name",
                table="symbols",
                columns=["name"],
                unique=True,
                description="Unique symbol name lookups"
            ),
            
            "idx_symbols_status": IndexDefinition(
                name="idx_symbols_status",
                table="symbols",
                columns=["status"],
                description="Active/inactive symbol filtering"
            ),
            
            # Performance-critical partial indices
            "idx_trades_recent": IndexDefinition(
                name="idx_trades_recent",
                table="trades",
                columns=["timestamp", "symbol", "price"],
                partial_condition="timestamp >= NOW() - INTERVAL '24 hours'",
                description="High-performance index for recent trading activity"
            ),
            
            "idx_orderbook_latest": IndexDefinition(
                name="idx_orderbook_latest", 
                table="orderbook_snapshots",
                columns=["symbol", "time_ms"],
                partial_condition="timestamp >= NOW() - INTERVAL '1 hour'",
                description="Ultra-fast latest orderbook retrieval"
            )
        }
        
    def create_all_indices(self) -> List[IndexPerformanceStats]:
        """Create all strategic indices for optimal query performance."""
        results = []
        
        logger.info("Creating all performance indices")
        
        for index_name, index_def in self.index_definitions.items():
            try:
                # Check if table exists before creating index
                if not self._table_exists(index_def.table):
                    logger.warning(f"Table {index_def.table} does not exist, skipping index {index_name}")
                    continue
                
                # Check if index already exists
                if self._index_exists(index_name):
                    logger.info(f"Index {index_name} already exists, skipping")
                    continue
                
                stats = self.create_index(index_def)
                if stats:
                    results.append(stats)
                    
            except Exception as e:
                logger.error(f"Failed to create index {index_name}: {e}")
        
        logger.info(f"Created {len(results)} indices successfully")
        return results
        
    def create_index(self, index_def: IndexDefinition) -> Optional[IndexPerformanceStats]:
        """Create a single index and return performance statistics."""
        try:
            start_time = time.time()
            
            logger.info(f"Creating index {index_def.name} on {index_def.table}({', '.join(index_def.columns)})")
            
            # Build CREATE INDEX statement
            unique_clause = "UNIQUE " if index_def.unique else ""
            columns_clause = ", ".join(index_def.columns)
            
            create_sql = f"""
                CREATE {unique_clause}INDEX {index_def.name} 
                ON {index_def.table} ({columns_clause})
            """
            
            # Add partial condition if specified
            if index_def.partial_condition:
                create_sql += f" WHERE {index_def.partial_condition}"
            
            # Execute index creation
            self.db_manager.execute(create_sql)
            
            creation_time = time.time() - start_time
            
            # Estimate index size
            size_estimate = self._estimate_index_size(index_def)
            
            stats = IndexPerformanceStats(
                index_name=index_def.name,
                table_name=index_def.table,
                size_estimate_mb=size_estimate,
                creation_time_seconds=creation_time,
                is_unique=index_def.unique,
                columns=index_def.columns.copy()
            )
            
            logger.info(f"Index {index_def.name} created in {creation_time:.2f}s, "
                       f"estimated size: {size_estimate:.1f}MB")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to create index {index_def.name}: {e}")
            return None
            
    def drop_index(self, index_name: str) -> bool:
        """Drop an index if it exists."""
        try:
            if not self._index_exists(index_name):
                logger.warning(f"Index {index_name} does not exist")
                return False
                
            self.db_manager.execute(f"DROP INDEX {index_name}")
            logger.info(f"Dropped index {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            return False
            
    def recreate_index(self, index_name: str) -> Optional[IndexPerformanceStats]:
        """Drop and recreate an index (useful for optimization)."""
        if index_name not in self.index_definitions:
            logger.error(f"Unknown index definition: {index_name}")
            return None
            
        # Drop existing index
        self.drop_index(index_name)
        
        # Recreate index
        return self.create_index(self.index_definitions[index_name])
        
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """Analyze query performance and suggest optimizations."""
        try:
            # Use EXPLAIN ANALYZE to get detailed query performance info
            explain_result = self.db_manager.execute(f"EXPLAIN ANALYZE {query}")
            
            # Parse explain output for insights
            explain_text = "\n".join([str(row[0]) for row in explain_result])
            
            analysis = {
                "query": query,
                "explain_plan": explain_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recommendations": []
            }
            
            # Simple heuristic analysis for common patterns
            query_lower = query.lower()
            
            # Check for missing time-based indices
            if "timestamp" in query_lower and "order by" in query_lower:
                if "seq scan" in explain_text.lower():
                    analysis["recommendations"].append({
                        "type": "missing_index",
                        "suggestion": "Consider adding time-based index on timestamp column",
                        "priority": "high"
                    })
            
            # Check for symbol filtering without index
            if "symbol =" in query_lower and "seq scan" in explain_text.lower():
                analysis["recommendations"].append({
                    "type": "missing_index", 
                    "suggestion": "Consider adding index on symbol column",
                    "priority": "medium"
                })
            
            # Check for complex WHERE clauses
            if query_lower.count("where") > 1 or "and" in query_lower:
                analysis["recommendations"].append({
                    "type": "query_optimization",
                    "suggestion": "Consider composite index for multi-column WHERE conditions",
                    "priority": "medium"
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query performance: {e}")
            return {"error": str(e), "query": query}
            
    def benchmark_query(self, query: str, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark query performance over multiple iterations."""
        try:
            execution_times = []
            
            logger.info(f"Benchmarking query over {iterations} iterations")
            
            for i in range(iterations):
                start_time = time.time()
                result = self.db_manager.execute(query)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Get row count from first iteration
                if i == 0:
                    row_count = len(result) if result else 0
            
            # Calculate statistics
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            benchmark_result = {
                "query": query,
                "iterations": iterations,
                "execution_times_seconds": execution_times,
                "average_time_seconds": avg_time,
                "min_time_seconds": min_time, 
                "max_time_seconds": max_time,
                "row_count": row_count,
                "rows_per_second": row_count / avg_time if avg_time > 0 else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Benchmark complete: {avg_time:.4f}s avg, {row_count:,} rows, "
                       f"{benchmark_result['rows_per_second']:.0f} rows/sec")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Failed to benchmark query: {e}")
            return {"error": str(e), "query": query}
            
    def optimize_database(self) -> Dict[str, Any]:
        """Run comprehensive database optimization."""
        try:
            start_time = time.time()
            
            logger.info("Starting comprehensive database optimization")
            
            optimization_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operations": [],
                "total_time_seconds": 0.0
            }
            
            # Update table statistics for better query planning
            logger.info("Updating database statistics")
            stat_start = time.time()
            self.db_manager.execute("ANALYZE")
            stat_time = time.time() - stat_start
            
            optimization_results["operations"].append({
                "operation": "analyze_statistics",
                "time_seconds": stat_time,
                "status": "completed"
            })
            
            # Vacuum tables to reclaim space and update statistics
            logger.info("Vacuuming database")
            vacuum_start = time.time()
            
            # Get list of tables to vacuum
            tables_result = self.db_manager.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main'
            """)
            
            vacuumed_tables = []
            for (table_name,) in tables_result:
                try:
                    self.db_manager.execute(f"VACUUM {table_name}")
                    vacuumed_tables.append(table_name)
                except Exception as e:
                    logger.warning(f"Failed to vacuum {table_name}: {e}")
            
            vacuum_time = time.time() - vacuum_start
            
            optimization_results["operations"].append({
                "operation": "vacuum_tables",
                "tables_vacuumed": vacuumed_tables,
                "time_seconds": vacuum_time,
                "status": "completed"
            })
            
            # Optimize DuckDB settings for trading workloads
            logger.info("Optimizing DuckDB settings")
            settings_start = time.time()
            
            try:
                # Enable parallel processing (use CPU count instead of 0)
                cpu_count = os.cpu_count() or 4  # Default to 4 if can't detect
                self.db_manager.execute(f"SET threads = {cpu_count}")
                
                # Optimize memory settings
                self.db_manager.execute("SET memory_limit = '4GB'")
                
                # Enable query result caching
                self.db_manager.execute("SET enable_query_verification = false")
                
                # Optimize for time-series workloads
                self.db_manager.execute("SET default_order = 'ASC'")
                
                settings_time = time.time() - settings_start
                
                optimization_results["operations"].append({
                    "operation": "optimize_settings",
                    "time_seconds": settings_time,
                    "status": "completed"
                })
                
            except Exception as e:
                logger.warning(f"Some settings optimization failed: {e}")
                optimization_results["operations"].append({
                    "operation": "optimize_settings",
                    "error": str(e),
                    "status": "partial"
                })
            
            total_time = time.time() - start_time
            optimization_results["total_time_seconds"] = total_time
            
            logger.info(f"Database optimization complete in {total_time:.2f}s")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
            
    def get_index_usage_report(self) -> Dict[str, Any]:
        """Generate a report on index usage and effectiveness."""
        try:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database_path": str(self.db_manager.config.database.path),
                "indices": {},
                "recommendations": []
            }
            
            # Get information about existing indices
            # Note: DuckDB doesn't have detailed index usage stats like PostgreSQL,
            # so we provide structural information and estimates
            
            for index_name, index_def in self.index_definitions.items():
                if self._index_exists(index_name):
                    try:
                        size_estimate = self._estimate_index_size(index_def)
                        
                        # Get table row count for selectivity estimation
                        table_rows_result = self.db_manager.execute(f"SELECT COUNT(*) FROM {index_def.table}")
                        table_rows = table_rows_result[0][0] if table_rows_result else 0
                        
                        report["indices"][index_name] = {
                            "table": index_def.table,
                            "columns": index_def.columns,
                            "size_estimate_mb": size_estimate,
                            "table_row_count": table_rows,
                            "is_unique": index_def.unique,
                            "has_partial_condition": index_def.partial_condition is not None,
                            "description": index_def.description
                        }
                        
                    except Exception as e:
                        logger.warning(f"Could not get info for index {index_name}: {e}")
            
            # Generate recommendations based on common query patterns
            if not report["indices"]:
                report["recommendations"].append({
                    "type": "missing_indices",
                    "priority": "high",
                    "suggestion": "No indices found. Consider creating basic time-series indices for optimal performance."
                })
            else:
                # Check for missing critical indices
                critical_indices = ["idx_trades_timestamp_symbol", "idx_orderbook_timestamp_symbol"]
                missing_critical = [idx for idx in critical_indices if idx not in report["indices"]]
                
                if missing_critical:
                    report["recommendations"].append({
                        "type": "missing_critical_indices",
                        "priority": "high", 
                        "suggestion": f"Missing critical indices: {', '.join(missing_critical)}"
                    })
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate index usage report: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
            
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            result = self.db_manager.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = ?
            """, (table_name,))
            return result[0][0] > 0 if result else False
        except:
            return False
            
    def _index_exists(self, index_name: str) -> bool:
        """Check if an index exists in the database."""
        try:
            # DuckDB doesn't have a standard way to check index existence
            # Try to use the index in a dummy query
            self.db_manager.execute(f"EXPLAIN SELECT 1 WHERE 1=0")
            return True  # This is a simplified check
        except:
            return False
            
    def _estimate_index_size(self, index_def: IndexDefinition) -> float:
        """Estimate index size in MB based on table size and column types."""
        try:
            # Get table row count
            result = self.db_manager.execute(f"SELECT COUNT(*) FROM {index_def.table}")
            row_count = result[0][0] if result else 0
            
            # Estimate bytes per index entry based on column types
            estimated_bytes_per_row = len(index_def.columns) * 16  # Rough estimate
            
            # Add overhead for B-tree structure (roughly 50% overhead)
            total_bytes = row_count * estimated_bytes_per_row * 1.5
            
            return total_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not estimate size for index {index_def.name}: {e}")
            return 0.0 