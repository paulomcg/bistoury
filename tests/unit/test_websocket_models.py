"""
Unit tests for WebSocket message models.

Tests message parsing, routing, validation, HyperLiquid format conversions,
subscription management, and business logic for all WebSocket message classes.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List

from src.bistoury.models.websocket import (
    MessageType,
    SubscriptionChannel,
    WSMessage,
    PriceUpdateMessage,
    TradeUpdateMessage,
    OrderBookUpdateMessage,
    CandleUpdateMessage,
    SubscriptionMessage,
    MessageRouter,
)
from src.bistoury.models.market_data import Timeframe, CandlestickData
from src.bistoury.models.orderbook import OrderBookLevel
from src.bistoury.models.trades import Trade


class TestMessageType:
    """Test MessageType enum functionality."""
    
    def test_message_type_values(self):
        """Test all message type enum values."""
        assert MessageType.PRICE_UPDATE == "allMids"
        assert MessageType.TRADE_UPDATE == "trades"
        assert MessageType.ORDER_BOOK_UPDATE == "l2Book"
        assert MessageType.CANDLE_UPDATE == "candle"
        assert MessageType.SUBSCRIPTION_ACK == "subscriptionAck"
        assert MessageType.ERROR == "error"
    
    def test_message_type_categories(self):
        """Test message type categorization."""
        market_data_types = [
            MessageType.PRICE_UPDATE,
            MessageType.TRADE_UPDATE,
            MessageType.ORDER_BOOK_UPDATE,
            MessageType.CANDLE_UPDATE
        ]
        
        user_data_types = [
            MessageType.USER_UPDATE,
            MessageType.ORDER_UPDATE,
            MessageType.FILL_UPDATE
        ]
        
        system_types = [
            MessageType.SUBSCRIPTION_ACK,
            MessageType.PING,
            MessageType.PONG,
            MessageType.ERROR
        ]
        
        # Verify all types exist
        for msg_type in market_data_types + user_data_types + system_types:
            assert isinstance(msg_type, MessageType)


class TestSubscriptionChannel:
    """Test SubscriptionChannel enum functionality."""
    
    def test_subscription_channel_values(self):
        """Test all subscription channel enum values."""
        assert SubscriptionChannel.ALL_MIDS == "allMids"
        assert SubscriptionChannel.TRADES == "trades"
        assert SubscriptionChannel.L2_BOOK == "l2Book"
        assert SubscriptionChannel.CANDLE == "candle"
        assert SubscriptionChannel.USER == "user"


class TestWSMessage:
    """Test WSMessage base model functionality."""
    
    def test_valid_ws_message(self):
        """Test creation of valid WebSocket message."""
        now = datetime.now(timezone.utc)
        message = WSMessage(
            channel="allMids",
            data={"BTC": "50000.0"},
            timestamp=now,
            message_id="msg_123",
            sequence=1
        )
        
        assert message.channel == "allMids"
        assert message.data == {"BTC": "50000.0"}
        assert message.timestamp == now
        assert message.message_id == "msg_123"
        assert message.sequence == 1
    
    def test_auto_timestamp_generation(self):
        """Test automatic timestamp generation."""
        message = WSMessage(
            channel="trades",
            data={"test": "data"}
        )
        
        # Should have generated a timestamp close to now
        now = datetime.now(timezone.utc)
        assert abs((message.timestamp - now).total_seconds()) < 1
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid format."""
        hl_data = {
            'channel': 'allMids',
            'data': {'BTC': '50000.0', 'ETH': '3000.0'},
            'timestamp': '2024-01-15T10:30:00Z',
            'id': 'msg_456',
            'sequence': 100
        }
        
        message = WSMessage.from_hyperliquid(hl_data)
        
        assert message.channel == "allMids"
        assert message.data == {'BTC': '50000.0', 'ETH': '3000.0'}
        assert message.message_id == "msg_456"
        assert message.sequence == 100
    
    def test_message_type_detection(self):
        """Test message type detection from channel."""
        # Known message type
        message = WSMessage(channel="allMids", data={})
        assert message.get_message_type() == MessageType.PRICE_UPDATE
        
        # Unknown but trade-related
        message = WSMessage(channel="tradeUpdates", data={})
        assert message.get_message_type() == MessageType.TRADE_UPDATE
        
        # Unknown but book-related
        message = WSMessage(channel="orderbook", data={})
        assert message.get_message_type() == MessageType.ORDER_BOOK_UPDATE
        
        # Completely unknown
        message = WSMessage(channel="unknown_channel", data={})
        assert message.get_message_type() == MessageType.ERROR
    
    def test_message_categorization(self):
        """Test message data type categorization."""
        # Market data message
        market_msg = WSMessage(channel="allMids", data={})
        assert market_msg.is_market_data is True
        assert market_msg.is_user_data is False
        
        # User data message
        user_msg = WSMessage(channel="user", data={})
        assert user_msg.is_user_data is True
        assert user_msg.is_market_data is False
        
        # System message
        system_msg = WSMessage(channel="ping", data={})
        assert system_msg.is_market_data is False
        assert system_msg.is_user_data is False


class TestPriceUpdateMessage:
    """Test PriceUpdateMessage functionality."""
    
    def test_valid_price_update(self):
        """Test creation of valid price update message."""
        prices = {
            "BTC": Decimal("50000.5"),
            "ETH": Decimal("3000.25"),
            "SOL": Decimal("100.75")
        }
        
        update = PriceUpdateMessage(
            prices=prices,
            sequence=123
        )
        
        assert len(update.prices) == 3
        assert update.get_price("BTC") == Decimal("50000.5")
        assert update.symbol_count == 3
        assert update.sequence == 123
        assert "BTC" in update.symbols
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid price data."""
        hl_data = {
            'BTC': '50000.5',
            'ETH': '3000.25',
            'SOL': '100.75'
        }
        
        update = PriceUpdateMessage.from_hyperliquid(hl_data)
        
        assert update.symbol_count == 3
        assert update.get_price("BTC") == Decimal("50000.5")
        assert update.get_price("ETH") == Decimal("3000.25")
        assert update.channel == "allMids"
    
    def test_price_lookup(self):
        """Test price lookup functionality."""
        prices = {
            "BTC": Decimal("50000.0"),
            "ETH": Decimal("3000.0")
        }
        
        update = PriceUpdateMessage(prices=prices)
        
        # Existing symbol
        assert update.get_price("BTC") == Decimal("50000.0")
        
        # Case insensitive
        assert update.get_price("btc") == Decimal("50000.0")
        
        # Non-existent symbol
        assert update.get_price("INVALID") is None
    
    def test_decimal_conversion(self):
        """Test automatic decimal conversion."""
        # Mixed input types
        update = PriceUpdateMessage(prices={
            "BTC": "50000.12345678",
            "ETH": 3000.123
        })
        
        assert isinstance(update.prices["BTC"], Decimal)
        assert isinstance(update.prices["ETH"], Decimal)
        assert update.prices["BTC"] == Decimal("50000.12345678")


class TestTradeUpdateMessage:
    """Test TradeUpdateMessage functionality."""
    
    def test_valid_trade_update(self):
        """Test creation of valid trade update message."""
        trades = [
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50000"),
                quantity=Decimal("1.0"),
                side="buy"
            ),
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50100"),
                quantity=Decimal("2.0"),
                side="sell"
            )
        ]
        
        update = TradeUpdateMessage(
            trades=trades,
            symbol="BTC",
            sequence=456
        )
        
        assert update.trade_count == 2
        assert update.symbol == "BTC"
        assert update.total_volume == Decimal("3.0")
        assert update.sequence == 456
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid trade data."""
        hl_data = [
            {
                'time': 1705316200000,
                'px': '50000.5',
                'sz': '1.5',
                'side': 'B',
                'tid': 'trade_123'
            },
            {
                'time': 1705316260000,
                'px': '50100.0',
                'sz': '2.0',
                'side': 'A',
                'tid': 'trade_124'
            }
        ]
        
        update = TradeUpdateMessage.from_hyperliquid(hl_data, "BTC")
        
        assert update.trade_count == 2
        assert update.symbol == "BTC"
        assert update.trades[0].side == "buy"  # B -> buy
        assert update.trades[1].side == "sell"  # A -> sell
        assert update.channel == "trades"
    
    def test_trade_analytics(self):
        """Test trade analytics calculations."""
        trades = [
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50000"),
                quantity=Decimal("1.0"),
                side="buy"
            ),
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50200"),
                quantity=Decimal("2.0"),
                side="sell"
            )
        ]
        
        update = TradeUpdateMessage(trades=trades, symbol="BTC")
        
        # Volume weighted price: (50000*1 + 50200*2) / (1+2) = 50133.33...
        vwap = update.volume_weighted_price
        assert vwap is not None
        assert abs(vwap - Decimal("50133.333333")) < Decimal("0.000001")
        
        # Trade filtering
        buy_trades = update.buy_trades
        sell_trades = update.sell_trades
        
        assert len(buy_trades) == 1
        assert len(sell_trades) == 1
        assert buy_trades[0].side == "buy"
        assert sell_trades[0].side == "sell"
    
    def test_latest_trade(self):
        """Test latest trade identification."""
        now = datetime.now(timezone.utc)
        later = now + timedelta(seconds=10)
        
        trades = [
            Trade(
                symbol="BTC",
                timestamp=now,
                price=Decimal("50000"),
                quantity=Decimal("1.0"),
                side="buy"
            ),
            Trade(
                symbol="BTC",
                timestamp=later,
                price=Decimal("50100"),
                quantity=Decimal("1.0"),
                side="sell"
            )
        ]
        
        update = TradeUpdateMessage(trades=trades, symbol="BTC")
        latest = update.latest_trade
        
        assert latest is not None
        assert latest.timestamp == later
        assert latest.price == Decimal("50100")
    
    def test_empty_trades(self):
        """Test handling of empty trade lists."""
        update = TradeUpdateMessage(trades=[], symbol="BTC")
        
        assert update.trade_count == 0
        assert update.total_volume == Decimal("0")
        assert update.volume_weighted_price is None
        assert update.latest_trade is None


class TestOrderBookUpdateMessage:
    """Test OrderBookUpdateMessage functionality."""
    
    def test_valid_order_book_update(self):
        """Test creation of valid order book update."""
        bids = [OrderBookLevel(price="50000", quantity="1.0", side="bid")]
        asks = [OrderBookLevel(price="50005", quantity="1.0", side="ask")]
        
        update = OrderBookUpdateMessage(
            bids=bids,
            asks=asks,
            symbol="BTC",
            sequence=789,
            is_snapshot=True
        )
        
        assert len(update.bids) == 1
        assert len(update.asks) == 1
        assert update.symbol == "BTC"
        assert update.is_snapshot is True
        assert update.has_updates is True
        assert update.update_count == 2
    
    def test_hyperliquid_conversion_levels_format(self):
        """Test conversion from HyperLiquid levels format."""
        hl_data = {
            'levels': [
                [
                    ['50000.0', '1.5', 3],
                    ['49995.0', '2.0', 5]
                ],
                [
                    ['50005.0', '1.2', 2],
                    ['50010.0', '0.8', 1]
                ]
            ]
        }
        
        update = OrderBookUpdateMessage.from_hyperliquid(hl_data, "BTC", is_snapshot=True)
        
        assert update.symbol == "BTC"
        assert len(update.bids) == 2
        assert len(update.asks) == 2
        assert update.is_snapshot is True
        assert update.channel == "l2Book"
    
    def test_hyperliquid_conversion_direct_format(self):
        """Test conversion from HyperLiquid direct bids/asks format."""
        hl_data = {
            'bids': [
                ['50000.0', '1.5', 3],
                ['49995.0', '2.0', 5]
            ],
            'asks': [
                ['50005.0', '1.2', 2],
                ['50010.0', '0.8', 1]
            ]
        }
        
        update = OrderBookUpdateMessage.from_hyperliquid(hl_data, "BTC")
        
        assert len(update.bids) == 2
        assert len(update.asks) == 2
        assert update.bids[0].price == Decimal("50000.0")
        assert update.asks[0].price == Decimal("50005.0")
    
    def test_best_levels(self):
        """Test best bid/ask identification."""
        bids = [
            OrderBookLevel(price="50000", quantity="1.0", side="bid"),
            OrderBookLevel(price="49995", quantity="2.0", side="bid")
        ]
        asks = [
            OrderBookLevel(price="50010", quantity="0.8", side="ask"),
            OrderBookLevel(price="50005", quantity="1.2", side="ask")
        ]
        
        update = OrderBookUpdateMessage(bids=bids, asks=asks, symbol="BTC")
        
        # Best bid is highest price
        best_bid = update.best_bid
        assert best_bid is not None
        assert best_bid.price == Decimal("50000")
        
        # Best ask is lowest price
        best_ask = update.best_ask
        assert best_ask is not None
        assert best_ask.price == Decimal("50005")
        
        # Spread calculation
        spread = update.spread
        assert spread == Decimal("5")
    
    def test_empty_order_book(self):
        """Test handling of empty order book updates."""
        update = OrderBookUpdateMessage(bids=[], asks=[], symbol="BTC")
        
        assert update.has_updates is False
        assert update.update_count == 0
        assert update.best_bid is None
        assert update.best_ask is None
        assert update.spread is None


class TestCandleUpdateMessage:
    """Test CandleUpdateMessage functionality."""
    
    def test_valid_candle_update(self):
        """Test creation of valid candle update."""
        candle = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.ONE_MINUTE,
            timestamp=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("10.5")
        )
        
        update = CandleUpdateMessage(
            candle=candle,
            symbol="BTC",
            timeframe=Timeframe.ONE_MINUTE,
            sequence=999,
            is_closed=True
        )
        
        assert update.symbol == "BTC"
        assert update.timeframe == Timeframe.ONE_MINUTE
        assert update.is_closed is True
        assert update.sequence == 999
        assert update.channel == "candle"
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid candle data."""
        hl_data = {
            't': 1705316200000,
            's': 'BTC',
            'o': '50000.0',
            'h': '50100.0',
            'l': '49900.0',
            'c': '50050.0',
            'v': '123.45',
            'n': 1500,
            'closed': True
        }
        
        update = CandleUpdateMessage.from_hyperliquid(hl_data, "BTC", Timeframe.ONE_MINUTE)
        
        assert update.symbol == "BTC"
        assert update.timeframe == Timeframe.ONE_MINUTE
        assert update.is_closed is True
        assert update.candle.open == Decimal("50000.0")
        assert update.candle.close == Decimal("50050.0")
    
    def test_price_change_calculations(self):
        """Test price change calculations."""
        candle = CandlestickData(
            symbol="BTC",
            timeframe=Timeframe.ONE_MINUTE,
            timestamp=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("50200"),
            low=Decimal("49800"),
            close=Decimal("50100"),
            volume=Decimal("10.0")
        )
        
        update = CandleUpdateMessage(
            candle=candle,
            symbol="BTC",
            timeframe=Timeframe.ONE_MINUTE
        )
        
        # Price change: 50100 - 50000 = 100
        assert update.price_change == Decimal("100")
        
        # Price change percent: (100 / 50000) * 100 = 0.2%
        assert update.price_change_percent == Decimal("0.2")


class TestSubscriptionMessage:
    """Test SubscriptionMessage functionality."""
    
    def test_valid_subscription_message(self):
        """Test creation of valid subscription message."""
        subscription = SubscriptionMessage(
            action="subscribe",
            channel=SubscriptionChannel.ALL_MIDS,
            success=True
        )
        
        assert subscription.action == "subscribe"
        assert subscription.channel == SubscriptionChannel.ALL_MIDS
        assert subscription.success is True
        assert subscription.error_message is None
    
    def test_subscription_requests(self):
        """Test subscription request creation."""
        # Subscribe request
        sub_request = SubscriptionMessage.subscribe_request(
            SubscriptionChannel.TRADES,
            symbol="BTC",
            timeframe=Timeframe.ONE_MINUTE
        )
        
        assert sub_request.action == "subscribe"
        assert sub_request.channel == SubscriptionChannel.TRADES
        assert sub_request.symbol == "BTC"
        assert sub_request.timeframe == Timeframe.ONE_MINUTE
        
        # Unsubscribe request
        unsub_request = SubscriptionMessage.unsubscribe_request(
            SubscriptionChannel.L2_BOOK,
            symbol="ETH"
        )
        
        assert unsub_request.action == "unsubscribe"
        assert unsub_request.channel == SubscriptionChannel.L2_BOOK
        assert unsub_request.symbol == "ETH"
    
    def test_subscription_ack(self):
        """Test subscription acknowledgment creation."""
        # Success ack
        success_ack = SubscriptionMessage.subscription_ack(
            SubscriptionChannel.ALL_MIDS,
            success=True
        )
        
        assert success_ack.action == "ack"
        assert success_ack.success is True
        assert success_ack.error_message is None
        
        # Error ack
        error_ack = SubscriptionMessage.subscription_ack(
            SubscriptionChannel.TRADES,
            success=False,
            error_message="Symbol not found"
        )
        
        assert error_ack.success is False
        assert error_ack.error_message == "Symbol not found"
    
    def test_hyperliquid_format_conversion(self):
        """Test conversion to HyperLiquid subscription format."""
        subscription = SubscriptionMessage.subscribe_request(
            SubscriptionChannel.CANDLE,
            symbol="BTC",
            timeframe=Timeframe.ONE_HOUR
        )
        
        hl_format = subscription.to_hyperliquid_format()
        
        expected = {
            "method": "subscribe",
            "subscription": {
                "type": "candle",
                "coin": "BTC",
                "interval": "1h"
            }
        }
        
        assert hl_format == expected
    
    def test_subscription_key_generation(self):
        """Test unique subscription key generation."""
        # Simple subscription
        simple_sub = SubscriptionMessage.subscribe_request(SubscriptionChannel.ALL_MIDS)
        assert simple_sub.subscription_key == "allMids"
        
        # With symbol
        symbol_sub = SubscriptionMessage.subscribe_request(
            SubscriptionChannel.TRADES,
            symbol="BTC"
        )
        assert symbol_sub.subscription_key == "trades:BTC"
        
        # With symbol and timeframe
        full_sub = SubscriptionMessage.subscribe_request(
            SubscriptionChannel.CANDLE,
            symbol="ETH",
            timeframe=Timeframe.ONE_HOUR
        )
        assert full_sub.subscription_key == "candle:ETH:1h"


class TestMessageRouter:
    """Test MessageRouter functionality."""
    
    def test_price_update_parsing(self):
        """Test parsing of price update messages."""
        raw_message = {
            'channel': 'allMids',
            'data': {
                'BTC': '50000.0',
                'ETH': '3000.0'
            }
        }
        
        parsed = MessageRouter.parse_message(raw_message)
        
        assert isinstance(parsed, PriceUpdateMessage)
        assert parsed.symbol_count == 2
        assert parsed.get_price("BTC") == Decimal("50000.0")
    
    def test_trade_update_parsing(self):
        """Test parsing of trade update messages."""
        raw_message = {
            'channel': 'trades',
            'symbol': 'BTC',
            'data': [
                {
                    'time': 1705316200000,
                    'px': '50000.5',
                    'sz': '1.5',
                    'side': 'B',
                    'tid': 'trade_123'
                }
            ]
        }
        
        parsed = MessageRouter.parse_message(raw_message)
        
        assert isinstance(parsed, TradeUpdateMessage)
        assert parsed.symbol == "BTC"
        assert parsed.trade_count == 1
        assert parsed.trades[0].side == "buy"
    
    def test_order_book_update_parsing(self):
        """Test parsing of order book update messages."""
        raw_message = {
            'channel': 'l2Book',
            'symbol': 'BTC',
            'snapshot': True,
            'data': {
                'levels': [
                    [['50000.0', '1.5', 3]],
                    [['50005.0', '1.2', 2]]
                ]
            }
        }
        
        parsed = MessageRouter.parse_message(raw_message)
        
        assert isinstance(parsed, OrderBookUpdateMessage)
        assert parsed.symbol == "BTC"
        assert parsed.is_snapshot is True
        assert len(parsed.bids) == 1
        assert len(parsed.asks) == 1
    
    def test_candle_update_parsing(self):
        """Test parsing of candle update messages."""
        raw_message = {
            'channel': 'candle',
            'symbol': 'BTC',
            'interval': '1m',
            'data': {
                't': 1705316200000,
                's': 'BTC',
                'o': '50000.0',
                'h': '50100.0',
                'l': '49900.0',
                'c': '50050.0',
                'v': '123.45',
                'n': 1500
            }
        }
        
        parsed = MessageRouter.parse_message(raw_message)
        
        assert isinstance(parsed, CandleUpdateMessage)
        assert parsed.symbol == "BTC"
        assert parsed.timeframe == Timeframe.ONE_MINUTE
        assert parsed.candle.open == Decimal("50000.0")
    
    def test_subscription_ack_parsing(self):
        """Test parsing of subscription acknowledgment messages."""
        raw_message = {
            'channel': 'subscriptionAck',
            'data': {
                'success': True
            }
        }
        
        parsed = MessageRouter.parse_message(raw_message)
        
        assert isinstance(parsed, SubscriptionMessage)
        assert parsed.action == "ack"
        assert parsed.success is True
    
    def test_unknown_message_parsing(self):
        """Test parsing of unknown message types."""
        raw_message = {
            'channel': 'unknown_channel',
            'data': {'test': 'data'}
        }
        
        parsed = MessageRouter.parse_message(raw_message)
        
        assert isinstance(parsed, WSMessage)
        assert parsed.channel == "unknown_channel"
        assert parsed.data == {'test': 'data'}
    
    def test_parsing_error_handling(self):
        """Test handling of parsing errors."""
        # Malformed message
        raw_message = {
            'channel': 'trades',
            'symbol': 'BTC',
            'data': "invalid_data"  # Should be a list for trades
        }
        
        parsed = MessageRouter.parse_message(raw_message)
        
        assert isinstance(parsed, WSMessage)
        assert "error" in parsed.data
        assert "Failed to parse message" in parsed.data["error"]
    
    def test_message_type_detection(self):
        """Test message type detection."""
        # From message object
        message = WSMessage(channel="allMids", data={})
        msg_type = MessageRouter.get_message_type(message)
        assert msg_type == MessageType.PRICE_UPDATE
        
        # From raw data
        raw_data = {'channel': 'trades'}
        msg_type = MessageRouter.get_message_type(raw_data)
        assert msg_type == MessageType.TRADE_UPDATE
        
        # Unknown channel with hints
        unknown_data = {'channel': 'custom_trade_feed'}
        msg_type = MessageRouter.get_message_type(unknown_data)
        assert msg_type == MessageType.TRADE_UPDATE
    
    def test_subscription_request_creation(self):
        """Test creation of multiple subscription requests."""
        symbols = ["BTC", "ETH", "SOL"]
        channels = [SubscriptionChannel.TRADES, SubscriptionChannel.L2_BOOK]
        
        requests = MessageRouter.create_subscription_requests(symbols, channels)
        
        # Should create 3 symbols * 2 channels = 6 requests
        assert len(requests) == 6
        
        # Check that all combinations are created
        expected_keys = [
            "trades:BTC", "trades:ETH", "trades:SOL",
            "l2Book:BTC", "l2Book:ETH", "l2Book:SOL"
        ]
        
        actual_keys = [req.subscription_key for req in requests]
        assert set(actual_keys) == set(expected_keys)
    
    def test_global_subscription_creation(self):
        """Test creation of global subscriptions."""
        symbols = ["BTC", "ETH"]
        channels = [SubscriptionChannel.ALL_MIDS]
        
        requests = MessageRouter.create_subscription_requests(symbols, channels)
        
        # Should create only 1 request (global subscription)
        assert len(requests) == 1
        assert requests[0].subscription_key == "allMids"
        assert requests[0].symbol is None
    
    def test_market_data_message_detection(self):
        """Test market data message detection."""
        # Market data message
        market_msg = {'channel': 'allMids'}
        assert MessageRouter.is_market_data_message(market_msg) is True
        
        # User data message
        user_msg = WSMessage(channel="user", data={})
        assert MessageRouter.is_market_data_message(user_msg) is False
        
        # System message
        system_msg = {'channel': 'ping'}
        assert MessageRouter.is_market_data_message(system_msg) is False


class TestWebSocketIntegration:
    """Test integration between WebSocket message models."""
    
    def test_comprehensive_message_flow(self):
        """Test a comprehensive WebSocket message processing flow."""
        # 1. Create subscription requests
        symbols = ["BTC", "ETH"]
        channels = [SubscriptionChannel.ALL_MIDS, SubscriptionChannel.TRADES]
        
        requests = MessageRouter.create_subscription_requests(symbols, channels)
        assert len(requests) == 3  # 1 global + 2 symbol-specific
        
        # 2. Process price update
        price_data = {
            'channel': 'allMids',
            'data': {'BTC': '50000.0', 'ETH': '3000.0'}
        }
        
        price_msg = MessageRouter.parse_message(price_data)
        assert isinstance(price_msg, PriceUpdateMessage)
        assert price_msg.symbol_count == 2
        
        # 3. Process trade update
        trade_data = {
            'channel': 'trades',
            'symbol': 'BTC',
            'data': [
                {
                    'time': 1705316200000,
                    'px': '50100.0',
                    'sz': '1.0',
                    'side': 'B',
                    'tid': 'trade_123'
                }
            ]
        }
        
        trade_msg = MessageRouter.parse_message(trade_data)
        assert isinstance(trade_msg, TradeUpdateMessage)
        assert trade_msg.symbol == "BTC"
        assert trade_msg.latest_trade.price == Decimal("50100.0")
        
        # 4. Verify message categorization
        assert MessageRouter.is_market_data_message(price_msg) is True
        assert MessageRouter.is_market_data_message(trade_msg) is True
        
        print(f"Processed {price_msg.symbol_count} price updates")
        print(f"Processed {trade_msg.trade_count} trade updates")
    
    def test_round_trip_conversions(self):
        """Test round-trip conversions between formats."""
        # Original subscription
        original_sub = SubscriptionMessage.subscribe_request(
            SubscriptionChannel.CANDLE,
            symbol="BTC",
            timeframe=Timeframe.ONE_HOUR
        )
        
        # Convert to HyperLiquid format
        hl_format = original_sub.to_hyperliquid_format()
        
        # Verify format
        assert hl_format["method"] == "subscribe"
        assert hl_format["subscription"]["type"] == "candle"
        assert hl_format["subscription"]["coin"] == "BTC"
        assert hl_format["subscription"]["interval"] == "1h"
        
        # Create subscription key
        key = original_sub.subscription_key
        assert key == "candle:BTC:1h"
    
    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        # Empty data
        empty_msg = MessageRouter.parse_message({})
        assert isinstance(empty_msg, WSMessage)
        
        # Missing channel - should be 'unknown' not empty string
        no_channel = {'data': {'test': 'data'}}
        parsed = MessageRouter.parse_message(no_channel)
        assert parsed.channel == "unknown"
        
        # Missing data
        no_data = {'channel': 'allMids'}
        parsed = MessageRouter.parse_message(no_data)
        assert parsed.data == {}
        
        # Invalid trade data
        invalid_trades = {
            'channel': 'trades',
            'symbol': 'BTC',
            'data': {'invalid': 'structure'}  # Should be a list
        }
        
        parsed = MessageRouter.parse_message(invalid_trades)
        assert isinstance(parsed, WSMessage)
        assert "error" in parsed.data
