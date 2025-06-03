#!/usr/bin/env python3
"""Test DuckDB auto-increment features."""

import duckdb

def test_auto_increment():
    conn = duckdb.connect(':memory:')
    
    print("Testing AUTO_INCREMENT...")
    try:
        conn.execute('CREATE TABLE test1 (id INTEGER PRIMARY KEY AUTO_INCREMENT, value TEXT)')
        print('✅ AUTO_INCREMENT supported')
    except Exception as e:
        print(f'❌ AUTO_INCREMENT not supported: {e}')
        
    print("\nTesting SEQUENCE...")
    try:
        conn.execute('CREATE SEQUENCE test_seq START 1')
        conn.execute('CREATE TABLE test2 (id INTEGER DEFAULT nextval("test_seq"), value TEXT)')
        print('✅ SEQUENCE supported')
    except Exception as e:
        print(f'❌ SEQUENCE not supported: {e}')
        
    print("\nTesting SERIAL...")
    try:
        conn.execute('CREATE TABLE test3 (id SERIAL, value TEXT)')
        print('✅ SERIAL supported')
    except Exception as e:
        print(f'❌ SERIAL not supported: {e}')

if __name__ == "__main__":
    test_auto_increment() 