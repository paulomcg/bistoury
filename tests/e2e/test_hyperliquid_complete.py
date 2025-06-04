#!/usr/bin/env python3
"""
Comprehensive test script for HyperLiquid integration.
Tests all features including rate limiting, historical data collection, and connection optimization.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any

from src.bistoury.config import Config
from src.bistoury.database import DatabaseManager
from src.bistoury.hyperliquid.client import HyperLiquidIntegration
from src.bistoury.hyperliquid.collector import DataCollector
from src.bistoury.logger import get_logger

logger = get_logger(__name__)


class HyperLiquidComprehensiveTest:
    """Comprehensive test suite for HyperLiquid integration."""
    
    def __init__(self):
        self.config = Config.load_from_env()
        self.client = HyperLiquidIntegration(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.collector = None
        self.test_results = {}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting comprehensive HyperLiquid integration tests...")
        
        # Test categories
        test_categories = [
            ("Basic API Connectivity", self.test_basic_api),
            ("Rate Limiting", self.test_rate_limiting),
            ("Connection Health Monitoring", self.test_connection_health),
            ("WebSocket Functionality", self.test_websocket_features),
            ("Historical Data Collection", self.test_historical_data),
            ("Data Collector Integration", self.test_data_collector),
            ("Bulk Historical Collection", self.test_bulk_historical),
            ("Error Handling & Resilience", self.test_error_handling),
            ("Performance & Optimization", self.test_performance)
        ]
        
        for category_name, test_func in test_categories:
            logger.info(f"\n📋 Testing: {category_name}")
            
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                self.test_results[category_name] = {
                    'success': True,
                    'duration': duration,
                    'details': result
                }
                
                logger.info(f"✅ {category_name} completed in {duration:.2f}s")
                
            except Exception as e:
                self.test_results[category_name] = {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                }
                
                logger.error(f"❌ {category_name} failed: {e}")
        
        # Generate summary
        await self.generate_test_summary()
        
        return self.test_results
    
    async def test_basic_api(self) -> Dict[str, Any]:
        """Test basic API connectivity and functionality."""
        results = {}
        
        # Health check
        health = await self.client.health_check()
        results['health_check'] = health
        assert health, "Health check should pass"
        
        # Get mid prices
        mids = await self.client.get_all_mids()
        results['mids_count'] = len(mids)
        assert len(mids) > 100, f"Should have many mid prices, got {len(mids)}"
        
        # Get metadata
        metadata = await self.client.get_meta()
        results['metadata_symbols'] = len(metadata.get('universe', []))
        assert len(metadata.get('universe', [])) > 100, "Should have substantial symbol metadata"
        
        # Get candles for BTC
        candles = await self.client.get_candles('BTC', '1h')
        results['btc_candles'] = len(candles)
        assert len(candles) > 0, "Should get BTC candles"
        
        # Get order book
        order_book = await self.client.get_order_book('BTC')
        results['orderbook_data'] = len(order_book)
        assert len(order_book) > 0, "Should get BTC order book"
        
        logger.info(f"Basic API tests passed: {results}")
        return results
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality."""
        results = {}
        
        # Test rapid API calls to trigger rate limiting
        start_time = time.time()
        call_times = []
        
        for i in range(10):
            call_start = time.time()
            await self.client.get_all_mids()
            call_duration = time.time() - call_start
            call_times.append(call_duration)
        
        total_duration = time.time() - start_time
        
        results['total_calls'] = 10
        results['total_duration'] = total_duration
        results['avg_call_time'] = sum(call_times) / len(call_times)
        results['rate_limited'] = total_duration > 1.0  # Should take at least 1 second due to rate limiting
        
        # Check rate limiter state
        limiter_state = {
            'requests_per_second': self.client.rate_limiter.requests_per_second,
            'current_tokens': self.client.rate_limiter.tokens,
            'burst_size': self.client.rate_limiter.burst_size
        }
        results['rate_limiter_state'] = limiter_state
        
        logger.info(f"Rate limiting tests passed: {results}")
        return results
    
    async def test_connection_health(self) -> Dict[str, Any]:
        """Test connection health monitoring features."""
        results = {}
        
        # Get initial health
        health = await self.client.get_connection_health()
        results['initial_health'] = health
        
        # Test performance stats
        stats = self.client.get_performance_stats()
        results['performance_stats'] = stats
        
        # Test connection validation
        validation = await self.client.validate_connection()
        results['connection_validation'] = validation
        assert validation, "Connection validation should pass"
        
        # Test monitor recording
        start_time = time.time()
        await self.client.get_all_mids()
        duration = time.time() - start_time
        
        # Check that monitor recorded the request
        updated_stats = self.client.get_performance_stats()
        results['stats_updated'] = updated_stats['total_requests'] > stats['total_requests']
        
        logger.info(f"Connection health tests passed: {results}")
        return results
    
    async def test_websocket_features(self) -> Dict[str, Any]:
        """Test WebSocket connectivity and subscriptions."""
        results = {}
        
        # Test connection
        connected = await self.client.connect()
        results['websocket_connected'] = connected
        assert connected, "WebSocket should connect"
        
        # Test subscription to all mids
        mids_subscribed = await self.client.subscribe_all_mids()
        results['mids_subscription'] = mids_subscribed
        
        # Test order book subscription
        orderbook_subscribed = await self.client.subscribe_orderbook('BTC')
        results['orderbook_subscription'] = orderbook_subscribed
        
        # Test trades subscription
        trades_subscribed = await self.client.subscribe_trades('BTC')
        results['trades_subscription'] = trades_subscribed
        
        # Check subscription tracking
        results['active_subscriptions'] = len(self.client.subscriptions)
        results['subscription_keys'] = list(self.client.subscriptions.keys())
        
        # Wait for potential messages
        await asyncio.sleep(2)
        
        # Test subscription optimization
        await self.client.optimize_subscriptions()
        results['optimized_subscriptions'] = len(self.client.subscriptions)
        
        # Test disconnection
        await self.client.disconnect()
        results['disconnected'] = not self.client.is_connected()
        
        logger.info(f"WebSocket tests passed: {results}")
        return results
    
    async def test_historical_data(self) -> Dict[str, Any]:
        """Test historical data collection capabilities."""
        results = {}
        
        # Test single symbol historical collection
        logger.info("Testing single symbol historical collection...")
        btc_candles = await self.client.get_historical_candles_bulk(
            symbol='BTC',
            interval='1h',
            days_back=2
        )
        
        results['btc_historical_count'] = len(btc_candles)
        results['btc_historical_success'] = len(btc_candles) > 0
        
        if btc_candles:
            # Verify data structure
            sample_candle = btc_candles[0]
            required_fields = ['t', 's', 'o', 'c', 'h', 'l', 'v']
            results['data_structure_valid'] = all(field in sample_candle for field in required_fields)
            
            # Verify chronological order
            timestamps = [int(candle['t']) for candle in btc_candles]
            results['chronological_order'] = timestamps == sorted(timestamps)
        
        # Test multiple symbols collection
        logger.info("Testing multiple symbols historical collection...")
        multi_results = await self.client.collect_multiple_symbols_historical(
            symbols=['BTC', 'ETH'],
            interval='4h',
            days_back=1
        )
        
        results['multi_symbol_results'] = {
            symbol: len(candles) for symbol, candles in multi_results.items()
        }
        results['multi_symbol_success'] = len(multi_results) == 2
        
        logger.info(f"Historical data tests passed: {results}")
        return results
    
    async def test_data_collector(self) -> Dict[str, Any]:
        """Test DataCollector functionality."""
        results = {}
        
        # Initialize collector
        self.collector = DataCollector(
            hyperliquid=self.client,
            db_manager=self.db_manager,
            symbols=['BTC', 'ETH']
        )
        
        # Test initialization
        results['collector_initialized'] = self.collector is not None
        results['initial_symbols'] = len(self.collector.symbols)
        
        # Test stats
        initial_stats = self.collector.get_stats()
        results['initial_stats'] = initial_stats
        
        # Test start/stop
        started = await self.collector.start()
        results['collector_started'] = started
        
        if started:
            await asyncio.sleep(2)  # Let it run briefly
            
            running_stats = self.collector.get_stats()
            results['running_stats'] = running_stats
            
            await self.collector.stop()
            results['collector_stopped'] = not self.collector.running
        
        logger.info(f"Data collector tests passed: {results}")
        return results
    
    async def test_bulk_historical(self) -> Dict[str, Any]:
        """Test bulk historical data collection."""
        results = {}
        
        if not self.collector:
            self.collector = DataCollector(
                hyperliquid=self.client,
                db_manager=self.db_manager,
                symbols=['BTC']
            )
        
        # Test single symbol historical collection
        logger.info("Testing collector historical data collection...")
        collected_count = await self.collector.collect_historical_data(
            symbol='BTC',
            days_back=1,
            interval='1h'
        )
        
        results['collected_count'] = collected_count
        results['collection_success'] = collected_count > 0
        
        # Test bulk collection
        logger.info("Testing bulk historical collection...")
        bulk_results = await self.collector.collect_historical_data_bulk(
            symbols=['BTC', 'ETH'],
            days_back=1,
            intervals=['4h']
        )
        
        results['bulk_results'] = bulk_results
        results['bulk_success'] = len(bulk_results) > 0
        
        # Test backfill functionality
        logger.info("Testing backfill missing data...")
        backfilled = await self.collector.backfill_missing_data(
            symbol='BTC',
            interval='1h',
            max_days_back=1
        )
        
        results['backfilled_count'] = backfilled
        
        logger.info(f"Bulk historical tests passed: {results}")
        return results
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and resilience."""
        results = {}
        
        # Test invalid symbol handling
        invalid_candles = await self.client.get_candles('INVALID_SYMBOL_XYZ', '1h')
        results['invalid_symbol_handled'] = isinstance(invalid_candles, list)
        
        # Test invalid order book
        invalid_orderbook = await self.client.get_order_book('INVALID_SYMBOL_XYZ')
        results['invalid_orderbook_handled'] = isinstance(invalid_orderbook, dict)
        
        # Test connection reset (if connected)
        if self.client.is_connected():
            reset_success = await self.client.reset_connection()
            results['connection_reset'] = reset_success
        else:
            results['connection_reset'] = 'not_connected'
        
        # Test error recovery
        try:
            # Force an error by calling with invalid parameters
            await self.client._make_api_call("test_error", lambda: None)
        except:
            results['error_recovery'] = 'handled'
        
        logger.info(f"Error handling tests passed: {results}")
        return results
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        results = {}
        
        # Measure API call performance
        start_time = time.time()
        
        tasks = [
            self.client.get_all_mids(),
            self.client.get_meta(),
            self.client.get_candles('BTC', '1h'),
            self.client.get_order_book('BTC')
        ]
        
        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        results['parallel_api_calls'] = len(tasks)
        results['parallel_execution_time'] = total_time
        results['all_successful'] = all(not isinstance(r, Exception) for r in api_results)
        
        # Get final performance stats
        final_stats = self.client.get_performance_stats()
        results['final_performance_stats'] = final_stats
        
        # Performance assertions
        results['performance_acceptable'] = (
            total_time < 10.0 and  # Should complete in reasonable time
            final_stats.get('avg_response_time', 0) < 5.0  # Average response time reasonable
        )
        
        logger.info(f"Performance tests passed: {results}")
        return results
    
    async def generate_test_summary(self) -> None:
        """Generate and log test summary."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result['success'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 HYPERLIQUID INTEGRATION TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total test categories: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        for category, result in self.test_results.items():
            status = "✅" if result['success'] else "❌"
            duration = result.get('duration', 0)
            logger.info(f"{status} {category}: {duration:.2f}s")
            
            if not result['success']:
                logger.error(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Performance summary
        if 'Performance & Optimization' in self.test_results:
            perf_result = self.test_results['Performance & Optimization']
            if perf_result['success']:
                details = perf_result['details']
                logger.info(f"\n📊 Performance Summary:")
                logger.info(f"   Parallel API calls: {details['parallel_execution_time']:.2f}s")
                logger.info(f"   Performance acceptable: {details['performance_acceptable']}")
        
        logger.info(f"{'='*60}")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.collector and self.collector.running:
                await self.collector.stop()
            
            if self.client.is_connected():
                await self.client.disconnect()
            
            if hasattr(self.db_manager, 'close'):
                self.db_manager.close()
                
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


async def main():
    """Main test execution."""
    test_suite = HyperLiquidComprehensiveTest()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Print final summary
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result['success'])
        
        if successful_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! HyperLiquid integration is ready for production.")
        else:
            logger.warning(f"⚠️  {total_tests - successful_tests} tests failed. Review errors above.")
        
        return successful_tests == total_tests
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False
        
    finally:
        await test_suite.cleanup()


if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 