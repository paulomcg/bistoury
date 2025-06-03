#!/usr/bin/env python3
"""
Explore HyperLiquid WebSocket API responses.
This is a disposable script to understand data structures before implementing DB schema.
"""

import asyncio
import json
import websockets
import httpx
from datetime import datetime
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperLiquidExplorer:
    """Explore HyperLiquid API data structures."""
    
    def __init__(self):
        self.rest_url = "https://api.hyperliquid.xyz"
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.responses = {}
        
    async def explore_rest_api(self):
        """Explore REST API endpoints to understand data structures."""
        print("🔍 Exploring HyperLiquid REST API endpoints...")
        
        async with httpx.AsyncClient() as client:
            # Get all trading symbols/assets
            try:
                print("\n📊 Getting asset metadata...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={"type": "meta"}
                )
                meta_data = response.json()
                print(f"✅ Meta data structure:")
                print(json.dumps(meta_data, indent=2)[:500] + "...")
                self.responses["meta"] = meta_data
                
            except Exception as e:
                print(f"❌ Error getting meta data: {e}")
            
            # Get all mids (current prices)
            try:
                print("\n💰 Getting all mids (current prices)...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={"type": "allMids"}
                )
                mids_data = response.json()
                print(f"✅ All mids structure:")
                print(json.dumps(mids_data, indent=2)[:500] + "...")
                self.responses["allMids"] = mids_data
                
            except Exception as e:
                print(f"❌ Error getting mids: {e}")
            
            # Get candlestick data
            try:
                print("\n📈 Getting candlestick data for BTC...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={
                        "type": "candleSnapshot",
                        "req": {
                            "coin": "BTC",
                            "interval": "1m",
                            "startTime": int((datetime.now().timestamp() - 3600) * 1000),  # Last hour
                            "endTime": int(datetime.now().timestamp() * 1000)
                        }
                    }
                )
                candle_data = response.json()
                print(f"✅ Candlestick data structure:")
                print(json.dumps(candle_data, indent=2)[:800] + "...")
                self.responses["candleSnapshot"] = candle_data
                
            except Exception as e:
                print(f"❌ Error getting candles: {e}")
            
            # Get orderbook
            try:
                print("\n📚 Getting orderbook for BTC...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={
                        "type": "l2Book",
                        "coin": "BTC"
                    }
                )
                orderbook_data = response.json()
                print(f"✅ Orderbook structure:")
                print(json.dumps(orderbook_data, indent=2)[:800] + "...")
                self.responses["l2Book"] = orderbook_data
                
            except Exception as e:
                print(f"❌ Error getting orderbook: {e}")
                
            # Get recent trades
            try:
                print("\n💹 Getting recent trades for BTC...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={
                        "type": "recentTrades",
                        "coin": "BTC"
                    }
                )
                trades_data = response.json()
                print(f"✅ Recent trades structure:")
                print(json.dumps(trades_data[0] if trades_data else {}, indent=2)[:600] + "...")
                self.responses["recentTrades"] = trades_data[0] if trades_data else []
                
            except Exception as e:
                print(f"❌ Error getting trades: {e}")
                
            # Get funding rates
            try:
                print("\n💰 Getting funding rate for BTC...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={
                        "type": "fundingHistory",
                        "coin": "BTC",
                        "startTime": int((datetime.now().timestamp() - 24*3600) * 1000)  # Last 24 hours
                    }
                )
                funding_data = response.json()
                print(f"✅ Funding rate structure:")
                print(json.dumps(funding_data[0] if funding_data else {}, indent=2)[:400] + "...")
                self.responses["fundingHistory"] = funding_data[0] if funding_data else {}
                
            except Exception as e:
                print(f"❌ Error getting funding rates: {e}")
                
            # Get open interest  
            try:
                print("\n📊 Getting open interest...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={
                        "type": "openInterest"
                    }
                )
                oi_data = response.json()
                print(f"✅ Open interest structure:")
                print(json.dumps(oi_data[0] if oi_data else {}, indent=2)[:400] + "...")
                self.responses["openInterest"] = oi_data[0] if oi_data else {}
                
            except Exception as e:
                print(f"❌ Error getting open interest: {e}")
                
            # Get funding rate current
            try:
                print("\n⚡ Getting current funding rate...")
                response = await client.post(
                    f"{self.rest_url}/info",
                    json={
                        "type": "clearinghouseState",
                        "user": "0x0000000000000000000000000000000000000000"  # Anonymous query
                    }
                )
                clearing_data = response.json()
                print(f"✅ Clearinghouse state structure:")
                print(json.dumps(clearing_data, indent=2)[:400] + "...")
                self.responses["clearinghouseState"] = clearing_data
                
            except Exception as e:
                print(f"❌ Error getting clearinghouse state: {e}")

    async def explore_websocket_feeds(self):
        """Explore WebSocket real-time data feeds."""
        print("\n🌐 Exploring HyperLiquid WebSocket feeds...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                print("✅ WebSocket connected!")
                
                # Subscribe to different feeds
                subscriptions = [
                    # All mids (price updates)
                    {"method": "subscribe", "subscription": {"type": "allMids"}},
                    
                    # Level 2 book updates for BTC
                    {"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}},
                    
                    # Trades for BTC  
                    {"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}},
                    
                    # Candle updates for BTC 1m
                    {"method": "subscribe", "subscription": {"type": "candle", "coin": "BTC", "interval": "1m"}},
                ]
                
                # Send subscriptions
                for sub in subscriptions:
                    await websocket.send(json.dumps(sub))
                    print(f"📡 Subscribed to: {sub['subscription']['type']}")
                
                # Listen for messages for 30 seconds
                print("\n👂 Listening for WebSocket messages for 30 seconds...")
                message_count = 0
                start_time = datetime.now()
                
                try:
                    while (datetime.now() - start_time).seconds < 30 and message_count < 20:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        message_count += 1
                        
                        # Analyze message structure
                        if "channel" in data:
                            channel_type = data["channel"]
                            print(f"\n📨 Message {message_count} - Channel: {channel_type}")
                            print(f"Structure: {json.dumps(data, indent=2)[:500]}...")
                            
                            # Store sample for analysis
                            if channel_type not in self.responses:
                                self.responses[f"ws_{channel_type}"] = []
                            if len(self.responses[f"ws_{channel_type}"]) < 3:  # Store max 3 samples
                                self.responses[f"ws_{channel_type}"].append(data)
                        
                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for messages")
                    
        except Exception as e:
            print(f"❌ WebSocket error: {e}")

    def analyze_data_structures(self):
        """Analyze all collected data structures."""
        print("\n🔬 ANALYZING DATA STRUCTURES FOR DATABASE SCHEMA DESIGN")
        print("=" * 60)
        
        for endpoint, data in self.responses.items():
            print(f"\n📋 {endpoint.upper()}:")
            print("-" * 40)
            
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
            else:
                sample = data
                
            self._analyze_structure(sample, indent=0)
    
    def _analyze_structure(self, obj: Any, indent: int = 0, parent_key: str = ""):
        """Recursively analyze data structure."""
        prefix = "  " * indent
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                print(f"{prefix}{key}: {type(value).__name__}", end="")
                if isinstance(value, str) and len(value) < 50:
                    print(f" = '{value}'")
                elif isinstance(value, (int, float)):
                    print(f" = {value}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f" (array of {len(value)} items)")
                    if len(value) > 0:
                        print(f"{prefix}  Sample item:")
                        self._analyze_structure(value[0], indent + 2)
                else:
                    print()
                    if isinstance(value, dict):
                        self._analyze_structure(value, indent + 1, key)
                        
        elif isinstance(obj, list) and len(obj) > 0:
            print(f"{prefix}Array with {len(obj)} items, first item:")
            self._analyze_structure(obj[0], indent + 1)

    def generate_schema_recommendations(self):
        """Generate database schema recommendations based on analysis."""
        print("\n💡 DATABASE SCHEMA RECOMMENDATIONS")
        print("=" * 50)
        
        # Analyze specific data types we found
        recommendations = []
        
        if "meta" in self.responses:
            recommendations.append({
                "table": "symbols",
                "description": "Trading symbol metadata",
                "suggested_schema": """
                CREATE TABLE symbols (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    max_leverage DOUBLE,
                    only_isolated BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            })
        
        if "candleSnapshot" in self.responses:
            recommendations.append({
                "table": "candles_1m",
                "description": "1-minute OHLCV candlestick data",
                "suggested_schema": """
                CREATE TABLE candles_1m (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price DOUBLE NOT NULL,
                    high_price DOUBLE NOT NULL,
                    low_price DOUBLE NOT NULL,
                    close_price DOUBLE NOT NULL,
                    volume DOUBLE NOT NULL,
                    UNIQUE(symbol, timestamp)
                )"""
            })
            
        if "l2Book" in self.responses:
            recommendations.append({
                "table": "orderbook_snapshots",
                "description": "Level 2 orderbook snapshots",
                "suggested_schema": """
                CREATE TABLE orderbook_snapshots (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bids JSON,  -- Array of [price, size] pairs
                    asks JSON,  -- Array of [price, size] pairs
                    nonce BIGINT
                )"""
            })
            
        if "recentTrades" in self.responses:
            recommendations.append({
                "table": "trades",
                "description": "Individual trade records",
                "suggested_schema": """
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    price DOUBLE NOT NULL,
                    size DOUBLE NOT NULL,
                    side TEXT NOT NULL,  -- 'buy' or 'sell'
                    trade_id TEXT UNIQUE
                )"""
            })
        
        # Print recommendations
        for rec in recommendations:
            print(f"\n📊 {rec['table'].upper()}")
            print(f"Purpose: {rec['description']}")
            print(f"Schema:{rec['suggested_schema']}")
            print()

async def main():
    """Main exploration function."""
    explorer = HyperLiquidExplorer()
    
    print("🚀 Starting HyperLiquid API Exploration")
    print("=" * 50)
    
    # Explore REST API
    await explorer.explore_rest_api()
    
    # Explore WebSocket feeds
    await explorer.explore_websocket_feeds()
    
    # Analyze collected data
    explorer.analyze_data_structures()
    
    # Generate schema recommendations
    explorer.generate_schema_recommendations()
    
    print("\n✅ Exploration complete! Ready for database schema design.")

if __name__ == "__main__":
    asyncio.run(main()) 