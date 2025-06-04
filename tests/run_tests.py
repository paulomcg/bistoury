#!/usr/bin/env python3
"""
Simple test runner for Bistoury integration tests.
Can be used when pytest is not available globally.
"""

import sys
import asyncio
import os
from pathlib import Path

# Add src to path - go up one level from tests/ to project root, then add src/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

async def run_basic_integration_test():
    """Run a basic integration test without pytest."""
    # Initialize logger first, outside try block
    from bistoury.logger import get_logger
    logger = get_logger(__name__)
    
    try:
        from bistoury.config import Config
        from bistoury.hyperliquid.client import HyperLiquidIntegration
        
        logger.info("🚀 Running basic HyperLiquid integration test...")
        
        # Initialize client
        config = Config.load_from_env()
        client = HyperLiquidIntegration(config)
        
        test_results = {}
        
        # Test 1: Health check
        logger.info("1. Testing health check...")
        health = await client.health_check()
        test_results['health_check'] = health
        assert health, "Health check should pass"
        logger.info("✅ Health check passed")
        
        # Test 2: Get connection info
        logger.info("2. Testing connection info...")
        info = client.get_connection_info()
        test_results['connection_info'] = info
        assert 'base_url' in info, "Should have base URL"
        logger.info(f"✅ Connection info: {info['base_url']}")
        
        # Test 3: Get market data
        logger.info("3. Testing market data...")
        mids = await client.get_all_mids()
        test_results['market_data'] = len(mids) if mids else 0
        assert isinstance(mids, dict), "Should return dict of mid prices"
        logger.info(f"✅ Got mid prices for {len(mids)} symbols")
        
        # Test 4: Get candles
        logger.info("4. Testing candle data...")
        candles = await client.get_candles('BTC', '1h')
        test_results['candles'] = len(candles) if candles else 0
        assert isinstance(candles, list), "Should return list of candles"
        logger.info(f"✅ Got {len(candles)} BTC candles")
        
        logger.info("\n🎉 All basic tests passed!")
        logger.info("Summary:")
        for test, result in test_results.items():
            logger.info(f"  {test}: {result}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def main():
    """Main entry point."""
    print("Bistoury Integration Test Runner")
    print("=" * 40)
    
    try:
        # Run the async test
        result = asyncio.run(run_basic_integration_test())
        
        if result:
            print("\n✅ All tests completed successfully!")
            print("\nTo run the full test suite:")
            print("  python -m pytest tests/integration/ -v")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        print("\nTroubleshooting:")
        print("1. Install required dependencies:")
        print("   pip install hyperliquid-python-sdk duckdb python-dotenv pydantic")
        print("2. Make sure .env file exists with proper configuration")
        print("3. Check network connectivity to HyperLiquid API")
        sys.exit(1)


if __name__ == "__main__":
    main() 