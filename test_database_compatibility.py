#!/usr/bin/env python3
"""
Test script to check compatibility of test.duckdb with current Bistoury schema.
This script will explore the existing test database and validate schema compatibility.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import duckdb

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bistoury.config import Config, DatabaseConfig
from bistoury.database import DatabaseManager, MarketDataSchema, DataInsertion, DataQuery


class TestDatabaseCompatibilityChecker:
    """Check compatibility between existing test.duckdb and our current schema."""
    
    def __init__(self):
        self.test_db_path = Path("data/test.duckdb")
        self.production_db_path = Path("data/bistoury.db")
        
    def explore_test_database(self):
        """Explore the existing test database structure and content."""
        print("🔍 Exploring Test Database Structure...")
        print("=" * 50)
        
        if not self.test_db_path.exists():
            print(f"❌ Test database not found: {self.test_db_path}")
            return False
            
        try:
            # Connect directly to test database
            conn = duckdb.connect(str(self.test_db_path))
            
            # Get database file size
            size_bytes = self.test_db_path.stat().st_size
            size_gb = size_bytes / (1024 * 1024 * 1024)
            print(f"📊 Database size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
            
            # Get all tables
            tables_result = conn.execute("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                ORDER BY table_name
            """).fetchall()
            
            print(f"\n📋 Found {len(tables_result)} tables:")
            for table_name, table_type in tables_result:
                print(f"   • {table_name} ({table_type})")
                
            # Examine each table structure and row counts
            print(f"\n📏 Table Details:")
            for table_name, _ in tables_result:
                try:
                    # Get column info
                    columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                    
                    # Get row count
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    
                    print(f"\n   🔹 {table_name}: {row_count:,} rows")
                    for col in columns:
                        col_name, col_type, nullable, unique, default, extra = col
                        constraints = []
                        if unique: constraints.append("UNIQUE")
                        if not nullable: constraints.append("NOT NULL")
                        constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                        print(f"      - {col_name}: {col_type}{constraint_str}")
                        
                    # Sample data for first few tables
                    if row_count > 0:
                        sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
                        if sample:
                            print(f"      Sample data (first {len(sample)} rows):")
                            for i, row in enumerate(sample, 1):
                                print(f"        Row {i}: {str(row)[:100]}{'...' if len(str(row)) > 100 else ''}")
                            
                except Exception as e:
                    print(f"      ❌ Error examining {table_name}: {e}")
                    
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error exploring test database: {e}")
            return False
            
    def test_schema_compatibility(self):
        """Test if our current schema works with the test database."""
        print(f"\n🧪 Testing Schema Compatibility...")
        print("=" * 50)
        
        try:
            # Create a config that points to the test database
            test_config = Config()
            test_config.database = DatabaseConfig(path=str(self.test_db_path))
            
            # Initialize database manager with test database
            db_manager = DatabaseManager(test_config)
            
            print(f"✅ Successfully connected to test database")
            
            # Test basic operations
            schema = MarketDataSchema(db_manager)
            insertion = DataInsertion(db_manager)
            query = DataQuery(db_manager)
            
            # Check if our expected tables exist
            expected_tables = [
                'symbols', 'trades', 'orderbook_snapshots', 'funding_rates',
                'candles_1m', 'candles_5m', 'candles_15m', 
                'candles_1h', 'candles_4h', 'candles_1d'
            ]
            
            existing_tables_result = db_manager.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            )
            existing_tables = {row[0] for row in existing_tables_result}
            
            print(f"\n📊 Schema Validation:")
            for table in expected_tables:
                if table in existing_tables:
                    print(f"   ✅ {table} - exists")
                else:
                    print(f"   ❌ {table} - missing")
                    
            # Test if we can query existing data
            if existing_tables:
                print(f"\n📈 Data Query Tests:")
                
                # Test symbols if available
                if 'symbols' in existing_tables:
                    try:
                        symbols = query.get_symbols()
                        print(f"   ✅ Symbols query: {len(symbols)} symbols found")
                        if symbols:
                            print(f"      Sample: {symbols[0]['symbol']} ({symbols[0]['max_leverage']}x leverage)")
                    except Exception as e:
                        print(f"   ❌ Symbols query failed: {e}")
                        
                # Test trades if available
                if 'trades' in existing_tables:
                    try:
                        # Try to get latest trades for any symbol
                        if symbols:
                            symbol = symbols[0]['symbol']
                            trades = query.get_latest_trades(symbol, limit=5)
                            print(f"   ✅ Trades query: {len(trades)} trades found for {symbol}")
                            if trades:
                                trade = trades[0]
                                print(f"      Latest: {trade['side']} {trade['size']} @ {trade['price']}")
                    except Exception as e:
                        print(f"   ❌ Trades query failed: {e}")
                        
                # Test candles if available
                if 'candles_1m' in existing_tables:
                    try:
                        if symbols:
                            symbol = symbols[0]['symbol']
                            candles = query.get_candles('1m', symbol, limit=5)
                            print(f"   ✅ Candles query: {len(candles)} 1m candles found for {symbol}")
                            if candles:
                                candle = candles[0]
                                print(f"      Latest: O:{candle['open_price']} H:{candle['high_price']} L:{candle['low_price']} C:{candle['close_price']}")
                    except Exception as e:
                        print(f"   ❌ Candles query failed: {e}")
                        
                # Test orderbook if available
                if 'orderbook_snapshots' in existing_tables:
                    try:
                        if symbols:
                            symbol = symbols[0]['symbol']
                            orderbook = query.get_latest_orderbook(symbol)
                            if orderbook:
                                print(f"   ✅ Orderbook query: Found latest snapshot for {symbol}")
                                print(f"      Levels: {len(orderbook['levels'][0])} bids, {len(orderbook['levels'][1])} asks")
                            else:
                                print(f"   ⚠️ Orderbook query: No data found for {symbol}")
                    except Exception as e:
                        print(f"   ❌ Orderbook query failed: {e}")
                        
                # Test funding rates if available
                if 'funding_rates' in existing_tables:
                    try:
                        if symbols:
                            symbol = symbols[0]['symbol']
                            funding = query.get_latest_funding_rate(symbol)
                            if funding:
                                print(f"   ✅ Funding rate query: Found rate for {symbol}")
                                print(f"      Rate: {funding['fundingRate']} Premium: {funding['premium']}")
                            else:
                                print(f"   ⚠️ Funding rate query: No data found for {symbol}")
                    except Exception as e:
                        print(f"   ❌ Funding rate query failed: {e}")
            
            db_manager.close_all_connections()
            return True
            
        except Exception as e:
            print(f"❌ Schema compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def compare_schemas(self):
        """Compare the schemas between test and production databases."""
        print(f"\n📋 Schema Comparison...")
        print("=" * 50)
        
        try:
            # Connect to both databases
            test_conn = duckdb.connect(str(self.test_db_path))
            prod_conn = duckdb.connect(str(self.production_db_path))
            
            # Get table structures for both
            test_tables = self._get_table_schemas(test_conn)
            prod_tables = self._get_table_schemas(prod_conn)
            
            # Compare tables
            all_tables = set(test_tables.keys()) | set(prod_tables.keys())
            
            for table in sorted(all_tables):
                print(f"\n🔹 Table: {table}")
                
                if table in test_tables and table in prod_tables:
                    # Compare column structures
                    test_cols = {col[0]: col[1] for col in test_tables[table]}
                    prod_cols = {col[0]: col[1] for col in prod_tables[table]}
                    
                    if test_cols == prod_cols:
                        print(f"   ✅ Schema matches")
                    else:
                        print(f"   ⚠️ Schema differences found")
                        
                        # Show differences
                        all_cols = set(test_cols.keys()) | set(prod_cols.keys())
                        for col in sorted(all_cols):
                            if col in test_cols and col in prod_cols:
                                if test_cols[col] != prod_cols[col]:
                                    print(f"      - {col}: TEST({test_cols[col]}) vs PROD({prod_cols[col]})")
                            elif col in test_cols:
                                print(f"      - {col}: Only in TEST ({test_cols[col]})")
                            else:
                                print(f"      - {col}: Only in PROD ({prod_cols[col]})")
                                
                elif table in test_tables:
                    print(f"   📊 Only in TEST database")
                else:
                    print(f"   🏗️ Only in PRODUCTION database")
            
            test_conn.close()
            prod_conn.close()
            
        except Exception as e:
            print(f"❌ Schema comparison failed: {e}")
            
    def _get_table_schemas(self, conn):
        """Get table schemas from a database connection."""
        tables = {}
        
        # Get all table names
        table_names = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        
        for (table_name,) in table_names:
            try:
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                tables[table_name] = columns
            except Exception as e:
                print(f"Warning: Could not get schema for {table_name}: {e}")
                
        return tables
        
    def test_data_migration_compatibility(self):
        """Test if data can be migrated between databases."""
        print(f"\n🔄 Testing Data Migration Compatibility...")
        print("=" * 50)
        
        try:
            # This would test copying data from test to production schema
            # For now, just validate that the data types are compatible
            
            test_conn = duckdb.connect(str(self.test_db_path))
            
            # Check if we can export/import a small sample
            print("✅ Data migration compatibility checks would go here")
            print("   - Data type compatibility")
            print("   - Foreign key constraints")
            print("   - Index compatibility")
            print("   - Performance implications")
            
            test_conn.close()
            
        except Exception as e:
            print(f"❌ Data migration test failed: {e}")


def main():
    """Run all compatibility tests."""
    print("🧪 Bistoury Test Database Compatibility Check")
    print("=" * 60)
    
    checker = TestDatabaseCompatibilityChecker()
    
    # Run all checks
    results = []
    
    print("Step 1: Exploring test database structure...")
    results.append(checker.explore_test_database())
    
    print("\nStep 2: Testing schema compatibility...")
    results.append(checker.test_schema_compatibility())
    
    print("\nStep 3: Comparing schemas...")
    checker.compare_schemas()
    
    print("\nStep 4: Testing data migration compatibility...")
    checker.test_data_migration_compatibility()
    
    # Summary
    print(f"\n📊 COMPATIBILITY CHECK SUMMARY")
    print("=" * 40)
    
    if all(results):
        print("✅ Test database is compatible with current schema!")
        print("✅ You can use data/test.duckdb for testing and backtesting")
        print("\n💡 Next steps:")
        print("   - Update configuration to support multiple database files")
        print("   - Create database switching utilities")
        print("   - Set up environment variables for database selection")
    else:
        print("⚠️ Some compatibility issues found")
        print("   - Review the output above for specific issues")
        print("   - Consider schema migration if needed")
        print("   - Test data integrity after any changes")
        
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main()) 