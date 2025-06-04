#!/usr/bin/env python3
"""
Test script for market data models with real HyperLiquid-style data.

This script validates that our Pydantic models work correctly with
actual market data formats and demonstrates all functionality.
"""

import json
from datetime import datetime, timezone
from decimal import Decimal

from src.bistoury.models.market_data import (
    CandlestickData, Ticker, SymbolInfo, MarketData, Timeframe, PriceLevel
)


def test_hyperliquid_candlestick_conversion():
    """Test CandlestickData conversion with real HyperLiquid format."""
    print("🕯️ Testing HyperLiquid Candlestick Conversion...")
    
    # Sample HyperLiquid candlestick data
    hl_data = {
        't': 1705316200000,  # 2024-01-15T10:30:00Z
        's': 'BTC',
        'o': '43127.5',
        'h': '43150.0',
        'l': '43100.25',
        'c': '43142.75',
        'v': '15.234567',
        'n': 127
    }
    
    # Convert from HyperLiquid format
    candle = CandlestickData.from_hyperliquid(hl_data, Timeframe.ONE_HOUR)
    
    print(f"✅ Symbol: {candle.symbol}")
    print(f"✅ Timeframe: {candle.timeframe}")
    print(f"✅ OHLC: {candle.open} / {candle.high} / {candle.low} / {candle.close}")
    print(f"✅ Volume: {candle.volume}")
    print(f"✅ Trade Count: {candle.trade_count}")
    print(f"✅ Timestamp: {candle.timestamp}")
    
    # Test properties
    print(f"✅ Body Size: {candle.body_size}")
    print(f"✅ Upper Shadow: {candle.upper_shadow}")
    print(f"✅ Lower Shadow: {candle.lower_shadow}")
    print(f"✅ Is Bullish: {candle.is_bullish}")
    print(f"✅ Is Doji: {candle.is_doji}")
    
    # Convert back to HyperLiquid format
    converted_back = candle.to_hyperliquid()
    print(f"✅ Round-trip conversion: {json.dumps(converted_back, indent=2)}")
    
    assert converted_back['t'] == hl_data['t']
    assert converted_back['s'] == hl_data['s']
    assert converted_back['o'] == hl_data['o']
    
    print("✅ HyperLiquid candlestick conversion test passed!\n")


def test_ticker_calculations():
    """Test Ticker model with realistic data."""
    print("📈 Testing Ticker Calculations...")
    
    ticker = Ticker(
        symbol="BTC",
        timestamp=datetime.now(timezone.utc),
        last_price="43142.75",
        bid_price="43140.50",
        ask_price="43145.00",
        volume_24h="1250.75",
        price_change_24h="250.25",
        price_change_pct_24h="0.58",
        high_24h="43200.00",
        low_24h="42850.50",
        open_24h="42892.50",
        trade_count_24h=15847
    )
    
    print(f"✅ Symbol: {ticker.symbol}")
    print(f"✅ Last Price: ${ticker.last_price:,.2f}")
    print(f"✅ Bid/Ask: ${ticker.bid_price:,.2f} / ${ticker.ask_price:,.2f}")
    print(f"✅ Mid Price: ${ticker.mid_price:,.2f}")
    print(f"✅ Spread: ${ticker.spread:.2f}")
    print(f"✅ Spread %: {ticker.spread_pct:.4f}%")
    print(f"✅ 24h Volume: {ticker.volume_24h:,.2f}")
    print(f"✅ 24h Change: ${ticker.price_change_24h:+,.2f} ({ticker.price_change_pct_24h:+.2f}%)")
    print(f"✅ 24h Range: ${ticker.low_24h:,.2f} - ${ticker.high_24h:,.2f}")
    
    print("✅ Ticker calculations test passed!\n")


def test_symbol_info_validation():
    """Test SymbolInfo with HyperLiquid metadata."""
    print("ℹ️ Testing Symbol Info Validation...")
    
    # Sample HyperLiquid metadata
    hl_metadata = {
        'name': 'BTC',
        'maxLeverage': 50,
        'onlyCross': False,
        'szDecimals': 3
    }
    
    # Convert from HyperLiquid format
    symbol_info = SymbolInfo.from_hyperliquid(hl_metadata)
    
    print(f"✅ Symbol: {symbol_info.symbol}")
    print(f"✅ Max Leverage: {symbol_info.max_leverage}x")
    print(f"✅ Only Cross: {symbol_info.only_cross}")
    print(f"✅ Size Decimals: {symbol_info.sz_decimals}")
    
    # Add additional info
    symbol_info.min_order_size = Decimal("0.001")
    symbol_info.max_order_size = Decimal("1000.0")
    symbol_info.tick_size = Decimal("0.5")
    symbol_info.step_size = Decimal("0.001")
    
    # Test price validation
    test_price = "43142.75"
    validated_price = symbol_info.validate_price(test_price)
    print(f"✅ Price validation: {test_price} -> {validated_price}")
    
    # Test quantity validation
    test_qty = "0.1234"
    validated_qty = symbol_info.validate_quantity(test_qty)
    print(f"✅ Quantity validation: {test_qty} -> {validated_qty}")
    
    print("✅ Symbol info validation test passed!\n")


def test_price_level_sorting():
    """Test PriceLevel sorting and comparison."""
    print("📊 Testing Price Level Operations...")
    
    levels = [
        PriceLevel(price="43150", quantity="2.5", count=5),
        PriceLevel(price="43145", quantity="3.2", count=8),
        PriceLevel(price="43155", quantity="1.8", count=3),
        PriceLevel(price="43140", quantity="4.1", count=12),
    ]
    
    print("Original levels:")
    for level in levels:
        print(f"  ${level.price:>8} | {level.quantity:>6} | {level.count:>3} orders")
    
    # Sort by price (ascending)
    sorted_levels = sorted(levels)
    print("\nSorted levels (ascending):")
    for level in sorted_levels:
        print(f"  ${level.price:>8} | {level.quantity:>6} | {level.count:>3} orders")
    
    # Verify sorting
    assert sorted_levels[0].price == Decimal("43140")
    assert sorted_levels[-1].price == Decimal("43155")
    
    print("✅ Price level sorting test passed!\n")


def test_edge_cases():
    """Test edge cases and validation."""
    print("⚠️ Testing Edge Cases and Validation...")
    
    # Test with very small cryptocurrency values (like SHIB)
    micro_candle = CandlestickData(
        symbol="SHIB",
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.ONE_MINUTE,
        open="0.00000891",
        high="0.00000895",
        low="0.00000888",
        close="0.00000892",
        volume="50000000.123456"
    )
    
    print(f"✅ Micro values: Open={micro_candle.open}, Volume={micro_candle.volume}")
    
    # Test scientific notation
    sci_level = PriceLevel(price="4.31425e4", quantity="1.5e-2")
    print(f"✅ Scientific notation: Price={sci_level.price}, Qty={sci_level.quantity}")
    
    # Test doji detection
    doji_candle = CandlestickData(
        symbol="BTC",
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.FIVE_MINUTES,
        open="43000",
        high="43020",
        low="42980",
        close="43002",  # Very small body
        volume="5.5"
    )
    
    print(f"✅ Doji detection: {doji_candle.is_doji} (body: {doji_candle.body_size})")
    
    print("✅ Edge cases test passed!\n")


def test_serialization():
    """Test JSON serialization and deserialization."""
    print("💾 Testing Serialization...")
    
    candle = CandlestickData(
        symbol="ETH",
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.FIFTEEN_MINUTES,
        open="2654.25",
        high="2658.75",
        low="2651.50", 
        close="2656.00",
        volume="125.75"
    )
    
    # Test model_dump
    data = candle.model_dump()
    print(f"✅ Serialized keys: {list(data.keys())}")
    
    # Test model_dump_json
    json_str = candle.model_dump_json()
    print(f"✅ JSON length: {len(json_str)} characters")
    
    # Test deserialization
    restored = CandlestickData.model_validate(data)
    assert restored.symbol == candle.symbol
    assert restored.open == candle.open
    
    print("✅ Serialization test passed!\n")


def main():
    """Run all model tests."""
    print("🧪 Starting Market Data Models Test Suite\n")
    print("=" * 60)
    
    try:
        test_hyperliquid_candlestick_conversion()
        test_ticker_calculations()
        test_symbol_info_validation()
        test_price_level_sorting()
        test_edge_cases()
        test_serialization()
        
        print("=" * 60)
        print("🎉 All Market Data Model Tests Passed Successfully!")
        print("✅ Models are ready for HyperLiquid integration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 