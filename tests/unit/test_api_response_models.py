"""
Unit tests for API response models.

Tests validation, conversion, HyperLiquid API compatibility, error handling,
and business logic for all API response model classes.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any

from src.bistoury.models.api_responses import (
    ErrorResponse,
    ResponseMetadata,
    MetadataResponse,
    AllMidsResponse,
    PositionInfo,
    UserInfoResponse,
    CandleHistoryResponse,
    TradeHistoryResponse,
    OrderBookResponse,
    ResponseWrapper,
)
from src.bistoury.models.market_data import SymbolInfo, CandlestickData, Timeframe
from src.bistoury.models.orderbook import OrderBook
from src.bistoury.models.trades import Trade


class TestErrorResponse:
    """Test ErrorResponse model functionality."""
    
    def test_valid_error_response(self):
        """Test creation of valid error response."""
        error = ErrorResponse(
            error="InvalidParameter",
            message="Symbol 'INVALID' not found",
            code=400,
            request_id="req_123456"
        )
        
        assert error.error == "InvalidParameter"
        assert error.message == "Symbol 'INVALID' not found"
        assert error.code == 400
        assert error.request_id == "req_123456"
    
    def test_minimal_error_response(self):
        """Test error response with only required fields."""
        error = ErrorResponse(error="NetworkError")
        
        assert error.error == "NetworkError"
        assert error.message is None
        assert error.code is None
    
    def test_error_type_detection(self):
        """Test error type detection properties."""
        # Rate limit error
        rate_error = ErrorResponse(error="Rate limit exceeded", code=429)
        assert rate_error.is_rate_limit_error is True
        assert rate_error.is_auth_error is False
        assert rate_error.is_server_error is False
        
        # Auth error
        auth_error = ErrorResponse(error="Unauthorized", code=401)
        assert auth_error.is_auth_error is True
        assert auth_error.is_rate_limit_error is False
        
        # Server error
        server_error = ErrorResponse(error="Internal Server Error", code=500)
        assert server_error.is_server_error is True
        assert server_error.is_auth_error is False
    
    def test_error_validation(self):
        """Test error response validation rules."""
        # Invalid HTTP code (too low)
        with pytest.raises(ValueError, match="greater than or equal to 100"):
            ErrorResponse(error="Test", code=99)
        
        # Invalid HTTP code (too high)
        with pytest.raises(ValueError, match="less than or equal to 599"):
            ErrorResponse(error="Test", code=600)


class TestResponseMetadata:
    """Test ResponseMetadata model functionality."""
    
    def test_valid_response_metadata(self):
        """Test creation of valid response metadata."""
        now = datetime.now(timezone.utc)
        metadata = ResponseMetadata(
            timestamp=now,
            request_id="req_789",
            total_count=1000,
            page=2,
            page_size=50,
            has_next=True
        )
        
        assert metadata.timestamp == now
        assert metadata.request_id == "req_789"
        assert metadata.total_count == 1000
        assert metadata.page == 2
        assert metadata.page_size == 50
        assert metadata.has_next is True
    
    def test_default_timestamp(self):
        """Test automatic timestamp generation."""
        metadata = ResponseMetadata()
        
        # Should have generated a timestamp close to now
        now = datetime.now(timezone.utc)
        assert abs((metadata.timestamp - now).total_seconds()) < 1
    
    def test_pagination_validation(self):
        """Test pagination field validation."""
        # Valid pagination
        ResponseMetadata(page=1, page_size=100, total_count=500)
        
        # Invalid page (too low)
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            ResponseMetadata(page=0)
        
        # Invalid page size (too high)
        with pytest.raises(ValueError, match="less than or equal to 10000"):
            ResponseMetadata(page_size=20000)


class TestMetadataResponse:
    """Test MetadataResponse model functionality."""
    
    def test_valid_metadata_response(self):
        """Test creation of valid metadata response."""
        symbols = [
            SymbolInfo(symbol="BTC", max_leverage=Decimal("50")),
            SymbolInfo(symbol="ETH", max_leverage=Decimal("25")),
        ]
        
        response = MetadataResponse(
            universe=symbols,
            exchange_status="operational",
            maintenance_mode=False
        )
        
        assert len(response.universe) == 2
        assert response.exchange_status == "operational"
        assert response.maintenance_mode is False
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid metadata format."""
        hl_data = {
            'universe': [
                {
                    'name': 'BTC',
                    'szDecimals': 3,
                    'maxLeverage': 50,
                    'onlyIsolated': False
                },
                {
                    'name': 'ETH',
                    'szDecimals': 4,
                    'maxLeverage': 25,
                    'onlyIsolated': False
                }
            ],
            'status': 'operational',
            'maintenanceMode': False
        }
        
        response = MetadataResponse.from_hyperliquid(hl_data)
        
        assert len(response.universe) == 2
        assert response.universe[0].symbol == "BTC"
        assert response.universe[1].symbol == "ETH"
        assert response.exchange_status == "operational"
        assert response.maintenance_mode is False
    
    def test_symbol_lookup(self):
        """Test symbol information lookup."""
        symbols = [
            SymbolInfo(symbol="BTC", max_leverage=Decimal("50")),
            SymbolInfo(symbol="ETH", max_leverage=Decimal("25")),
        ]
        
        response = MetadataResponse(universe=symbols)
        
        # Find existing symbol
        btc_info = response.get_symbol_info("BTC")
        assert btc_info is not None
        assert btc_info.symbol == "BTC"
        
        # Case insensitive lookup
        eth_info = response.get_symbol_info("eth")
        assert eth_info is not None
        assert eth_info.symbol == "ETH"
        
        # Non-existent symbol
        invalid_info = response.get_symbol_info("INVALID")
        assert invalid_info is None
    
    def test_active_symbols_filter(self):
        """Test filtering for active symbols."""
        symbols = [
            SymbolInfo(symbol="BTC", is_active=True),
            SymbolInfo(symbol="ETH", is_active=True),
            SymbolInfo(symbol="DEPRECATED", is_active=False),
        ]
        
        response = MetadataResponse(universe=symbols)
        active_symbols = response.active_symbols
        
        assert len(active_symbols) == 2
        assert all(s.is_active for s in active_symbols)
        assert {s.symbol for s in active_symbols} == {"BTC", "ETH"}


class TestAllMidsResponse:
    """Test AllMidsResponse model functionality."""
    
    def test_valid_all_mids_response(self):
        """Test creation of valid all mids response."""
        mids = {
            "BTC": Decimal("50000.5"),
            "ETH": Decimal("3000.25"),
            "SOL": Decimal("100.75")
        }
        
        response = AllMidsResponse(mids=mids)
        
        assert len(response.mids) == 3
        assert response.mids["BTC"] == Decimal("50000.5")
        assert response.symbol_count == 3
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid allMids format."""
        hl_data = {
            'BTC': '50000.5',
            'ETH': '3000.25',
            'SOL': '100.75'
        }
        
        response = AllMidsResponse.from_hyperliquid(hl_data)
        
        assert response.symbol_count == 3
        assert response.get_price("BTC") == Decimal("50000.5")
        assert response.get_price("ETH") == Decimal("3000.25")
    
    def test_price_lookup(self):
        """Test price lookup functionality."""
        mids = {
            "BTC": Decimal("50000.0"),
            "ETH": Decimal("3000.0")
        }
        
        response = AllMidsResponse(mids=mids)
        
        # Existing symbol
        assert response.get_price("BTC") == Decimal("50000.0")
        
        # Case insensitive
        assert response.get_price("btc") == Decimal("50000.0")
        
        # Non-existent symbol
        assert response.get_price("INVALID") is None
    
    def test_decimal_conversion(self):
        """Test automatic decimal conversion from strings."""
        # String inputs
        response = AllMidsResponse(mids={"BTC": "50000.12345678"})
        assert response.mids["BTC"] == Decimal("50000.12345678")
        
        # Numeric inputs
        response2 = AllMidsResponse(mids={"BTC": 50000.123})
        assert isinstance(response2.mids["BTC"], Decimal)


class TestPositionInfo:
    """Test PositionInfo model functionality."""
    
    def test_valid_position_info(self):
        """Test creation of valid position info."""
        position = PositionInfo(
            coin="BTC",
            szi=Decimal("0.5"),
            entryPx=Decimal("50000.0"),
            positionValue=Decimal("25000.0"),
            unrealizedPnl=Decimal("1000.0"),
            leverage=Decimal("2.0"),
            liquidationPx=Decimal("40000.0")
        )
        
        assert position.coin == "BTC"
        assert position.szi == Decimal("0.5")
        assert position.entryPx == Decimal("50000.0")
    
    def test_optional_fields(self):
        """Test position with only required fields."""
        position = PositionInfo(
            coin="ETH",
            szi=Decimal("2.0")
        )
        
        assert position.coin == "ETH"
        assert position.szi == Decimal("2.0")
        assert position.entryPx is None
        assert position.positionValue is None
    
    def test_decimal_conversion(self):
        """Test decimal field conversion."""
        position = PositionInfo(
            coin="BTC",
            szi="0.5",
            entryPx="50000.123",
            unrealizedPnl="-500.75"
        )
        
        assert position.szi == Decimal("0.5")
        assert position.entryPx == Decimal("50000.123")
        assert position.unrealizedPnl == Decimal("-500.75")


class TestUserInfoResponse:
    """Test UserInfoResponse model functionality."""
    
    def test_valid_user_info_response(self):
        """Test creation of valid user info response."""
        margin_summary = {
            "accountValue": "10000.0",
            "totalMarginUsed": "2000.0"
        }
        
        positions = [
            PositionInfo(coin="BTC", szi=Decimal("0.1")),
            PositionInfo(coin="ETH", szi=Decimal("2.0")),
        ]
        
        response = UserInfoResponse(
            marginSummary=margin_summary,
            assetPositions=positions
        )
        
        assert response.account_value == Decimal("10000.0")
        assert response.total_margin_used == Decimal("2000.0")
        assert len(response.assetPositions) == 2
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid user info format."""
        hl_data = {
            'marginSummary': {
                'accountValue': '10000.0',
                'totalMarginUsed': '2000.0'
            },
            'assetPositions': [
                {
                    'position': {
                        'coin': 'BTC',
                        'szi': '0.1',
                        'entryPx': '50000.0'
                    }
                },
                {
                    'position': {
                        'coin': 'ETH',
                        'szi': '2.0',
                        'entryPx': '3000.0'
                    }
                }
            ]
        }
        
        response = UserInfoResponse.from_hyperliquid(hl_data)
        
        assert response.account_value == Decimal("10000.0")
        assert len(response.assetPositions) == 2
        assert response.assetPositions[0].coin == "BTC"
        assert response.assetPositions[0].szi == Decimal("0.1")
    
    def test_position_lookup(self):
        """Test position lookup by symbol."""
        positions = [
            PositionInfo(coin="BTC", szi=Decimal("0.1")),
            PositionInfo(coin="ETH", szi=Decimal("2.0")),
        ]
        
        response = UserInfoResponse(
            marginSummary={},
            assetPositions=positions
        )
        
        # Find existing position
        btc_pos = response.get_position("BTC")
        assert btc_pos is not None
        assert btc_pos.coin == "BTC"
        
        # Case insensitive
        eth_pos = response.get_position("eth")
        assert eth_pos is not None
        assert eth_pos.coin == "ETH"
        
        # Non-existent position
        invalid_pos = response.get_position("SOL")
        assert invalid_pos is None


class TestCandleHistoryResponse:
    """Test CandleHistoryResponse model functionality."""
    
    def test_valid_candle_history_response(self):
        """Test creation of valid candle history response."""
        now = datetime.now(timezone.utc)
        
        candles = [
            CandlestickData(
                symbol="BTC",
                timeframe=Timeframe.ONE_MINUTE,
                timestamp=now,
                open=Decimal("50000"),
                high=Decimal("50100"),
                low=Decimal("49900"),
                close=Decimal("50050"),
                volume=Decimal("10.5")
            )
        ]
        
        response = CandleHistoryResponse(
            candles=candles,
            symbol="BTC",
            timeframe=Timeframe.ONE_MINUTE
        )
        
        assert len(response.candles) == 1
        assert response.symbol == "BTC"
        assert response.timeframe == Timeframe.ONE_MINUTE
        assert response.candle_count == 1
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid candles format."""
        hl_data = [
            {
                't': 1705316200000,
                's': 'BTC',
                'o': '50000.0',
                'h': '50100.0',
                'l': '49900.0',
                'c': '50050.0',
                'v': '10.5',
                'n': 150
            },
            {
                't': 1705316260000,
                's': 'BTC',
                'o': '50050.0',
                'h': '50150.0',
                'l': '50000.0',
                'c': '50100.0',
                'v': '8.2',
                'n': 120
            }
        ]
        
        response = CandleHistoryResponse.from_hyperliquid(hl_data, "BTC", Timeframe.ONE_MINUTE)
        
        assert response.candle_count == 2
        assert response.symbol == "BTC"
        assert response.timeframe == Timeframe.ONE_MINUTE
        assert response.candles[0].open == Decimal("50000.0")
        assert response.candles[1].close == Decimal("50100.0")
    
    def test_time_range_calculation(self):
        """Test time range calculation."""
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)  # 2 hours later
        
        response = CandleHistoryResponse(
            candles=[],
            symbol="BTC",
            timeframe=Timeframe.ONE_HOUR,
            start_time=start_time,
            end_time=end_time
        )
        
        assert response.time_range_hours == 2.0


class TestTradeHistoryResponse:
    """Test TradeHistoryResponse model functionality."""
    
    def test_valid_trade_history_response(self):
        """Test creation of valid trade history response."""
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
        
        response = TradeHistoryResponse(
            trades=trades,
            symbol="BTC"
        )
        
        assert response.trade_count == 2
        assert response.symbol == "BTC"
        assert response.total_volume == Decimal("3.0")
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid trades format."""
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
        
        response = TradeHistoryResponse.from_hyperliquid(hl_data, "BTC")
        
        assert response.trade_count == 2
        assert response.symbol == "BTC"
        assert response.trades[0].side == "buy"  # B -> buy
        assert response.trades[1].side == "sell"  # A -> sell
    
    def test_trade_filtering(self):
        """Test trade filtering by side."""
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
            ),
            Trade(
                symbol="BTC",
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50050"),
                quantity=Decimal("1.5"),
                side="buy"
            )
        ]
        
        response = TradeHistoryResponse(trades=trades, symbol="BTC")
        
        buy_trades = response.buy_trades
        sell_trades = response.sell_trades
        
        assert len(buy_trades) == 2
        assert len(sell_trades) == 1
        assert all(t.side == "buy" for t in buy_trades)
        assert all(t.side == "sell" for t in sell_trades)


class TestOrderBookResponse:
    """Test OrderBookResponse model functionality."""
    
    def test_valid_order_book_response(self):
        """Test creation of valid order book response."""
        from src.bistoury.models.orderbook import OrderBookLevel
        
        bids = [OrderBookLevel(price="50000", quantity="1.0")]
        asks = [OrderBookLevel(price="50005", quantity="1.0")]
        order_book = OrderBook(bids=bids, asks=asks)
        
        response = OrderBookResponse(
            order_book=order_book,
            symbol="BTC"
        )
        
        assert response.symbol == "BTC"
        assert response.spread == Decimal("5")
        assert response.mid_price == Decimal("50002.5")
    
    def test_hyperliquid_conversion(self):
        """Test conversion from HyperLiquid order book format."""
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
        
        response = OrderBookResponse.from_hyperliquid(hl_data, "BTC")
        
        assert response.symbol == "BTC"
        assert len(response.order_book.bids) == 2
        assert len(response.order_book.asks) == 2
        assert response.order_book.best_bid.price == Decimal("50000.0")
        assert response.order_book.best_ask.price == Decimal("50005.0")


class TestResponseWrapper:
    """Test ResponseWrapper model functionality."""
    
    def test_valid_success_response(self):
        """Test creation of valid success response."""
        data = {"test": "data"}
        wrapper = ResponseWrapper.success_response(data)
        
        assert wrapper.success is True
        assert wrapper.data == data
        assert wrapper.error is None
        assert wrapper.metadata is not None
    
    def test_valid_error_response(self):
        """Test creation of valid error response."""
        error = ErrorResponse(error="TestError", message="Test message")
        wrapper = ResponseWrapper.error_response(error)
        
        assert wrapper.success is False
        assert wrapper.error == error
        assert wrapper.data is None
    
    def test_error_response_from_string(self):
        """Test error response creation from string."""
        wrapper = ResponseWrapper.error_response("Test error")
        
        assert wrapper.success is False
        assert wrapper.error.error == "Test error"
        assert wrapper.data is None
    
    def test_response_validation(self):
        """Test response wrapper validation rules."""
        # Success response without data (should fail)
        with pytest.raises(ValueError, match="Successful responses must include data"):
            ResponseWrapper(success=True, data=None)
        
        # Error response without error (should fail)
        with pytest.raises(ValueError, match="Failed responses must include error"):
            ResponseWrapper(success=False, error=None)
    
    def test_raise_for_status(self):
        """Test raise_for_status method."""
        # Success response (should not raise)
        success_wrapper = ResponseWrapper.success_response({"test": "data"})
        success_wrapper.raise_for_status()  # Should not raise
        
        # Error response (should raise)
        error_wrapper = ResponseWrapper.error_response("Test error")
        with pytest.raises(Exception, match="API Error: Test error"):
            error_wrapper.raise_for_status()


class TestModelIntegration:
    """Test integration between API response models."""
    
    def test_json_serialization(self):
        """Test JSON serialization of API response models."""
        error = ErrorResponse(error="TestError", code=400)
        
        # Test serialization
        json_data = error.model_dump()
        assert 'error' in json_data
        assert 'code' in json_data
        
        # Test round-trip
        restored = ErrorResponse(**json_data)
        assert restored.error == error.error
        assert restored.code == error.code
    
    def test_comprehensive_api_flow(self):
        """Test a comprehensive API response flow."""
        # 1. Create metadata response
        hl_metadata = {
            'universe': [
                {'name': 'BTC', 'szDecimals': 3, 'maxLeverage': 50},
                {'name': 'ETH', 'szDecimals': 4, 'maxLeverage': 25}
            ]
        }
        
        metadata_response = MetadataResponse.from_hyperliquid(hl_metadata)
        assert len(metadata_response.universe) == 2
        
        # 2. Create all mids response
        all_mids_response = AllMidsResponse.from_hyperliquid({
            'BTC': '50000.0',
            'ETH': '3000.0'
        })
        assert all_mids_response.symbol_count == 2
        
        # 3. Wrap in success response
        success_wrapper = ResponseWrapper.success_response(all_mids_response)
        assert success_wrapper.success is True
        
        # 4. Create error response
        error_wrapper = ResponseWrapper.error_response("Network timeout")
        assert error_wrapper.success is False
        
        print(f"Processed {metadata_response.universe[0].symbol} metadata")
        print(f"Current BTC price: {all_mids_response.get_price('BTC')}")
    
    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Empty candle history
        empty_response = CandleHistoryResponse.from_hyperliquid([], "BTC", Timeframe.ONE_MINUTE)
        assert empty_response.candle_count == 0
        assert empty_response.time_range_hours is None
        
        # Empty trade history
        empty_trades = TradeHistoryResponse.from_hyperliquid([], "BTC")
        assert empty_trades.trade_count == 0
        assert empty_trades.total_volume == Decimal("0")
        
        # Empty positions
        empty_user_info = UserInfoResponse.from_hyperliquid({
            'marginSummary': {},
            'assetPositions': []
        })
        assert len(empty_user_info.assetPositions) == 0
        assert empty_user_info.account_value is None 