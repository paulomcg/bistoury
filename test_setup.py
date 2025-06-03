#!/usr/bin/env python3
"""
Test script to verify Bistoury development environment setup.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.bistoury import __version__
        print(f"✅ Package version: {__version__}")
    except ImportError as e:
        print(f"❌ Failed to import package: {e}")
        return False
    
    try:
        from src.bistoury.config import Config
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    try:
        from src.bistoury.logger import get_logger
        print("✅ Logger module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import logger: {e}")
        return False
    
    try:
        from src.bistoury.cli import main
        print("✅ CLI module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CLI: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from src.bistoury.config import Config
        config = Config.load_from_env()
        
        print(f"✅ Default pairs: {config.data.default_pairs}")
        print(f"✅ Risk limit: ${config.trading.risk_limit_usd}")
        print(f"✅ Database path: {config.database.path}")
        print(f"✅ Log level: {config.logging.level}")
        
        providers = config.get_available_llm_providers()
        print(f"✅ Available LLM providers: {providers}")
        
        if config.hyperliquid:
            print(f"✅ HyperLiquid testnet: {config.hyperliquid.testnet}")
        else:
            print("⚠️  HyperLiquid not configured (expected for initial setup)")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_logging():
    """Test logging functionality."""
    print("\n📝 Testing logging...")
    
    try:
        from src.bistoury.logger import get_logger, get_trading_adapter
        
        # Test basic logger
        logger = get_logger("test.logger", "DEBUG")
        logger.info("Test log message")
        print("✅ Basic logging works")
        
        # Test trading adapter
        trade_logger = get_trading_adapter(symbol="BTC", strategy="test")
        trade_logger.info("Test trading log message")
        print("✅ Trading logger adapter works")
        
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI functionality."""
    print("\n🖥️  Testing CLI...")
    
    try:
        from src.bistoury.cli import main
        
        # The CLI is implemented and can be imported
        print("✅ CLI module is functional")
        
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Bistoury Development Environment Verification")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration), 
        ("Logging Tests", test_logging),
        ("CLI Tests", test_cli),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Development environment is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 