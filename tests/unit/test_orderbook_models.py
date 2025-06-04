"""
Unit tests for order book models.

Tests validation, conversion, API compatibility, and business logic
for all order book model classes.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

from src.bistoury.models.orderbook import (
    OrderBookLevel,
    OrderBook,
    OrderBookSnapshot,
    OrderBookDelta,
)


class TestOrderBookLevel:
    """Test OrderBookLevel model functionality."""
    
    def test_valid_order_book_level(self):
        """Test creation of valid order book level."""
        level = OrderBookLevel(
            price="50000.12345678",
            quantity="1.5", 
            count=5,
            side="bid"
        )
        
        assert level.price == Decimal("50000.12345678")
        assert level.quantity == Decimal("1.5")
        assert level.count == 5
        assert level.side == "bid"
    
    def test_level_without_side(self):
        """Test order book level without side specification."""
        level = OrderBookLevel(
            price="50000.0",
            quantity="2.0"
        )
        
        assert level.side is None
        assert level.count is None
    
    def test_notional_value_calculation(self):
        """Test notional value calculation."""
        level = OrderBookLevel(
            price="50000.0",
            quantity="1.5"
        )
        
        assert level.notional_value == Decimal("75000.0")
    
    def test_level_representation(self):
        """Test string representation."""
        level = OrderBookLevel(
            price="50000.0",
            quantity="1.5",
            count=3,
            side="ask"
        )
        
        repr_str = repr(level)
        assert "OrderBookLevel (ask)" in repr_str
        assert "$50000.0" in repr_str
        assert "1.5" in repr_str
        assert "[3 orders]" in repr_str


class TestOrderBook:
    """Test OrderBook model functionality."""
    
    def test_valid_order_book(self):
        """Test creation of valid order book."""
        bids = [
            OrderBookLevel(price="50000", quantity="1.0", side="bid"),
            OrderBookLevel(price="49995", quantity="2.0", side="bid"),
        ]
        asks = [
            OrderBookLevel(price="50005", quantity="1.5", side="ask"),
            OrderBookLevel(price="50010", quantity="0.8", side="ask"),
        ]
        
        book = OrderBook(bids=bids, asks=asks)
        
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.bids[0].price == Decimal("50000")
        assert book.asks[0].price == Decimal("50005")
    
    def test_empty_order_book(self):
        """Test empty order book creation."""
        book = OrderBook(bids=[], asks=[])
        
        assert len(book.bids) == 0
        assert len(book.asks) == 0
        assert book.best_bid is None
        assert book.best_ask is None
    
    def test_bid_ordering_validation(self):
        """Test bid ordering validation (highest to lowest)."""
        # Valid ordering
        bids = [
            OrderBookLevel(price="50000", quantity="1.0"),
            OrderBookLevel(price="49995", quantity="2.0"),
        ]
        OrderBook(bids=bids, asks=[])
        
        # Invalid ordering (should fail)
        invalid_bids = [
            OrderBookLevel(price="49995", quantity="1.0"),
            OrderBookLevel(price="50000", quantity="2.0"),  # Higher price after lower
        ]
        
        with pytest.raises(ValueError, match="Bids must be ordered from highest to lowest"):
            OrderBook(bids=invalid_bids, asks=[])
    
    def test_ask_ordering_validation(self):
        """Test ask ordering validation (lowest to highest)."""
        # Valid ordering
        asks = [
            OrderBookLevel(price="50005", quantity="1.0"),
            OrderBookLevel(price="50010", quantity="2.0"),
        ]
        OrderBook(bids=[], asks=asks)
        
        # Invalid ordering (should fail)
        invalid_asks = [
            OrderBookLevel(price="50010", quantity="1.0"),
            OrderBookLevel(price="50005", quantity="2.0"),  # Lower price after higher
        ]
        
        with pytest.raises(ValueError, match="Asks must be ordered from lowest to highest"):
            OrderBook(bids=[], asks=invalid_asks)
    
    def test_bid_ask_overlap_validation(self):
        """Test bid/ask overlap validation."""
        # Valid spread
        bids = [OrderBookLevel(price="50000", quantity="1.0")]
        asks = [OrderBookLevel(price="50005", quantity="1.0")]
        OrderBook(bids=bids, asks=asks)
        
        # Invalid overlap (bid >= ask)
        invalid_bids = [OrderBookLevel(price="50005", quantity="1.0")]
        invalid_asks = [OrderBookLevel(price="50000", quantity="1.0")]
        
        with pytest.raises(ValueError, match="Best bid.*must be less than best ask"):
            OrderBook(bids=invalid_bids, asks=invalid_asks)
    
    def test_hyperliquid_conversion_full_book(self):
        """Test conversion from HyperLiquid full order book format."""
        hl_data = {
            'bids': [
                ['50000.0', '1.5', 3],
                ['49995.0', '2.0', 5],
            ],
            'asks': [
                ['50005.0', '1.2', 2],
                ['50010.0', '0.8', 1],
            ]
        }
        
        book = OrderBook.from_hyperliquid(hl_data)
        
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.bids[0].price == Decimal("50000.0")
        assert book.bids[0].quantity == Decimal("1.5")
        assert book.bids[0].count == 3
        assert book.bids[0].side == "bid"
        assert book.asks[0].price == Decimal("50005.0")
        assert book.asks[0].side == "ask"
    
    def test_hyperliquid_conversion_single_side(self):
        """Test conversion from HyperLiquid single side format."""
        hl_bid_data = {
            'levels': [
                ['50000.0', '1.5', 3],
                ['49995.0', '2.0'],
            ],
            'type': 'bid'
        }
        
        book = OrderBook.from_hyperliquid(hl_bid_data)
        
        assert len(book.bids) == 2
        assert len(book.asks) == 0
        assert book.bids[0].price == Decimal("50000.0")
        assert book.bids[1].count is None  # No count provided
    
    def test_to_hyperliquid_conversion(self):
        """Test conversion to HyperLiquid format."""
        bids = [
            OrderBookLevel(price="50000", quantity="1.5", count=3),
            OrderBookLevel(price="49995", quantity="2.0", count=5),
        ]
        asks = [
            OrderBookLevel(price="50005", quantity="1.2", count=2),
        ]
        
        book = OrderBook(bids=bids, asks=asks)
        hl_data = book.to_hyperliquid()
        
        assert 'bids' in hl_data
        assert 'asks' in hl_data
        assert hl_data['bids'][0] == ['50000.00000000', '1.50000000', 3]
        assert hl_data['asks'][0] == ['50005.00000000', '1.20000000', 2]
    
    def test_best_bid_ask_properties(self):
        """Test best bid/ask properties."""
        bids = [
            OrderBookLevel(price="50000", quantity="1.0"),
            OrderBookLevel(price="49995", quantity="2.0"),
        ]
        asks = [
            OrderBookLevel(price="50005", quantity="1.0"),
            OrderBookLevel(price="50010", quantity="2.0"),
        ]
        
        book = OrderBook(bids=bids, asks=asks)
        
        assert book.best_bid.price == Decimal("50000")
        assert book.best_ask.price == Decimal("50005")
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        bids = [OrderBookLevel(price="50000", quantity="1.0")]
        asks = [OrderBookLevel(price="50005", quantity="1.0")]
        
        book = OrderBook(bids=bids, asks=asks)
        
        assert book.spread == Decimal("5")
        assert book.mid_price == Decimal("50002.5")
        
        # Spread in basis points: (spread / mid_price) * 10000
        # (5 / 50002.5) * 10000 = 0.9999500024998750062496875156
        expected_bps = (Decimal("5") / Decimal("50002.5")) * 10000
        assert book.spread_bps == expected_bps
    
    def test_volume_calculations(self):
        """Test volume calculation properties."""
        bids = [
            OrderBookLevel(price="50000", quantity="1.5"),
            OrderBookLevel(price="49995", quantity="2.0"),
        ]
        asks = [
            OrderBookLevel(price="50005", quantity="1.2"),
            OrderBookLevel(price="50010", quantity="0.8"),
        ]
        
        book = OrderBook(bids=bids, asks=asks)
        
        assert book.total_bid_quantity == Decimal("3.5")
        assert book.total_ask_quantity == Decimal("2.0")
    
    def test_depth_imbalance(self):
        """Test order book depth imbalance calculation."""
        # More bid volume (bullish)
        bids = [OrderBookLevel(price="50000", quantity="3.0")]
        asks = [OrderBookLevel(price="50005", quantity="1.0")]
        
        book = OrderBook(bids=bids, asks=asks)
        imbalance = book.depth_imbalance
        
        # (3.0 - 1.0) / (3.0 + 1.0) = 0.5
        assert imbalance == Decimal("0.5")
    
    def test_levels_within_spread_pct(self):
        """Test filtering levels within percentage of best prices."""
        bids = [
            OrderBookLevel(price="50000", quantity="1.0"),  # Best bid
            OrderBookLevel(price="49000", quantity="2.0"),  # 2% below
        ]
        asks = [
            OrderBookLevel(price="50500", quantity="1.0"),  # Best ask
            OrderBookLevel(price="51000", quantity="2.0"),  # ~1% above
        ]
        
        book = OrderBook(bids=bids, asks=asks)
        
        # Get levels within 1.5% of best prices
        levels = book.get_levels_within_spread_pct(Decimal("1.5"))
        
        assert len(levels['bids']) == 1  # Only best bid within 1.5%
        assert len(levels['asks']) == 2  # Both asks within 1.5%
    
    def test_market_impact_calculation(self):
        """Test market impact calculation for trades."""
        asks = [
            OrderBookLevel(price="50000", quantity="1.0"),
            OrderBookLevel(price="50005", quantity="2.0"),
            OrderBookLevel(price="50010", quantity="1.5"),
        ]
        
        book = OrderBook(bids=[], asks=asks)
        
        # Buy 2.5 units (will consume 1.0 @ 50000 + 1.5 @ 50005)
        impact = book.calculate_market_impact(Decimal("2.5"), "buy")
        
        # Average price: (1.0*50000 + 1.5*50005) / 2.5 = 50003
        expected_avg = (Decimal("50000") + Decimal("1.5") * Decimal("50005")) / Decimal("2.5")
        
        assert impact['average_price'] == expected_avg
        assert impact['levels_consumed'] == 2
        assert impact['price_impact_pct'] > 0  # Should be positive
    
    def test_market_impact_insufficient_liquidity(self):
        """Test market impact with insufficient liquidity."""
        asks = [
            OrderBookLevel(price="50000", quantity="1.0"),
        ]
        
        book = OrderBook(bids=[], asks=asks)
        
        # Try to buy more than available
        with pytest.raises(ValueError, match="Insufficient liquidity"):
            book.calculate_market_impact(Decimal("2.0"), "buy")


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot model functionality."""
    
    def test_valid_order_book_snapshot(self):
        """Test creation of valid order book snapshot."""
        bids = [OrderBookLevel(price="50000", quantity="1.0")]
        asks = [OrderBookLevel(price="50005", quantity="1.0")]
        order_book = OrderBook(bids=bids, asks=asks)
        
        now = datetime.now(timezone.utc)
        snapshot = OrderBookSnapshot(
            symbol="BTC",
            timestamp=now,
            order_book=order_book,
            sequence_id=12345
        )
        
        assert snapshot.symbol == "BTC"
        assert snapshot.order_book.best_bid.price == Decimal("50000")
        assert snapshot.sequence_id == 12345
        assert snapshot.exchange == "hyperliquid"
    
    def test_hyperliquid_snapshot_conversion(self):
        """Test OrderBookSnapshot from HyperLiquid format."""
        hl_data = {
            'bids': [['50000.0', '1.5']],
            'asks': [['50005.0', '1.2']]
        }
        
        now = datetime.now(timezone.utc)
        snapshot = OrderBookSnapshot.from_hyperliquid(hl_data, "BTC", now)
        
        assert snapshot.symbol == "BTC"
        assert snapshot.timestamp == now
        assert snapshot.order_book.best_bid.price == Decimal("50000.0")
        assert snapshot.exchange == "hyperliquid"
    
    def test_convenience_properties(self):
        """Test convenience properties for common operations."""
        bids = [OrderBookLevel(price="50000", quantity="1.0")]
        asks = [OrderBookLevel(price="50005", quantity="1.0")]
        order_book = OrderBook(bids=bids, asks=asks)
        
        snapshot = OrderBookSnapshot(
            symbol="BTC",
            timestamp=datetime.now(timezone.utc),
            order_book=order_book
        )
        
        assert snapshot.best_bid_price == Decimal("50000")
        assert snapshot.best_ask_price == Decimal("50005")
        assert snapshot.spread == Decimal("5")
        assert snapshot.mid_price == Decimal("50002.5")


class TestOrderBookDelta:
    """Test OrderBookDelta model functionality."""
    
    def test_valid_order_book_delta(self):
        """Test creation of valid order book delta."""
        now = datetime.now(timezone.utc)
        changes = [
            {"side": "bid", "price": "50000.0", "quantity": "1.5", "action": "update"},
            {"side": "ask", "price": "50005.0", "quantity": "0.0", "action": "delete"},
        ]
        
        delta = OrderBookDelta(
            symbol="BTC",
            timestamp=now,
            sequence_id=12345,
            changes=changes
        )
        
        assert delta.symbol == "BTC"
        assert delta.timestamp == now
        assert delta.sequence_id == 12345
        assert len(delta.changes) == 2
        assert delta.time_ms == int(now.timestamp() * 1000)
    
    def test_timestamp_validation(self):
        """Test timestamp validation and conversion."""
        # Test with milliseconds timestamp
        timestamp_ms = 1705316200000
        delta = OrderBookDelta(
            symbol="BTC",
            timestamp=timestamp_ms,
            changes=[{"action": "update"}]
        )
        
        assert delta.timestamp.year == 2024
        assert delta.time_ms == timestamp_ms
    
    def test_empty_changes_validation(self):
        """Test validation of empty changes list."""
        with pytest.raises(ValueError, match="at least 1 item"):
            OrderBookDelta(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                changes=[]  # Empty list should fail
            )


class TestModelIntegration:
    """Test integration between different order book models."""
    
    def test_json_serialization(self):
        """Test JSON serialization of order book models."""
        bids = [OrderBookLevel(price="50000", quantity="1.0", count=5)]
        asks = [OrderBookLevel(price="50005", quantity="1.0", count=3)]
        book = OrderBook(bids=bids, asks=asks)
        
        # Test serialization
        json_data = book.model_dump()
        assert 'bids' in json_data
        assert 'asks' in json_data
        
        # Test round-trip
        restored = OrderBook(**json_data)
        assert restored.best_bid.price == book.best_bid.price
        assert restored.best_ask.price == book.best_ask.price
    
    def test_scientific_notation_handling(self):
        """Test handling of scientific notation in price inputs."""
        level = OrderBookLevel(
            price="5.0e4",  # 50000
            quantity="1.5e-3"  # 0.0015
        )
        
        assert level.price == Decimal("50000.00000000")
        assert level.quantity == Decimal("0.00150000")
    
    def test_edge_case_prices(self):
        """Test handling of edge case price values."""
        # Very small prices (like SHIB)
        micro_level = OrderBookLevel(
            price="0.00000001",
            quantity="1000000000"
        )
        
        assert micro_level.price == Decimal("0.00000001")
        assert micro_level.quantity == Decimal("1000000000.00000000")
        
        # Very large prices
        large_level = OrderBookLevel(
            price="999999.99999999",
            quantity="0.00000001"
        )
        
        assert large_level.price == Decimal("999999.99999999")
        assert large_level.quantity == Decimal("0.00000001") 