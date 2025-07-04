"""
HyperLiquid integration using the official Python SDK.
Provides a wrapper around the official HyperLiquid SDK for Bistoury.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from collections import defaultdict, deque

from hyperliquid.info import Info
from hyperliquid.websocket_manager import WebsocketManager
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

from ..config import Config
from ..logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Rate limiter to respect HyperLiquid API limits.
    """
    
    def __init__(self, requests_per_second: float = 10.0, burst_size: int = 20):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, blocking if necessary to respect rate limits."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now
            
            if self.tokens < 1:
                # Need to wait
                wait_time = (1 - self.tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class ConnectionMonitor:
    """
    Monitor connection health and performance metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.request_times = deque(maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.connection_errors = 0
        self.last_error_time = None
        self.last_success_time = None
    
    def record_request(self, duration: float, success: bool, error_type: Optional[str] = None):
        """Record a request result."""
        self.total_requests += 1
        self.request_times.append(duration)
        
        if success:
            self.successful_requests += 1
            self.last_success_time = datetime.now()
        else:
            self.error_count += 1
            self.last_error_time = datetime.now()
            
            if error_type in ['connection', 'timeout', 'network']:
                self.connection_errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.request_times:
            return {
                'avg_response_time': 0,
                'success_rate': 0,
                'total_requests': self.total_requests,
                'error_rate': 0,
                'connection_errors': self.connection_errors
            }
        
        return {
            'avg_response_time': sum(self.request_times) / len(self.request_times),
            'max_response_time': max(self.request_times),
            'min_response_time': min(self.request_times),
            'success_rate': (self.successful_requests / self.total_requests) * 100 if self.total_requests > 0 else 0,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'error_count': self.error_count,
            'error_rate': (self.error_count / self.total_requests) * 100 if self.total_requests > 0 else 0,
            'connection_errors': self.connection_errors,
            'last_error_time': self.last_error_time,
            'last_success_time': self.last_success_time
        }
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy based on recent performance."""
        stats = self.get_stats()
        
        # Consider healthy if:
        # - Success rate > 95% over recent requests
        # - Average response time < 5 seconds
        # - Less than 5 connection errors in recent window
        
        return (
            stats['success_rate'] > 95 and
            stats['avg_response_time'] < 5.0 and
            stats['connection_errors'] < 5
        )


class HyperLiquidIntegration:
    """
    HyperLiquid integration using the official Python SDK.
    
    This class wraps the official HyperLiquid SDK to provide:
    - Market data collection
    - WebSocket subscriptions
    - Order management (if authenticated)
    - Integration with Bistoury's database system
    - Rate limiting and connection optimization
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize HyperLiquid integration with rate limiting and monitoring.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config.load_from_env()
        
        # API Configuration
        self.base_url = self._get_api_url()
        
        # Initialize rate limiter and monitoring
        self.rate_limiter = RateLimiter(
            requests_per_second=getattr(self.config, 'hyperliquid_rate_limit', 10.0),
            burst_size=20
        )
        self.monitor = ConnectionMonitor()
        
        # Connection state
        self.connected = False
        self.ws_manager = None
        self._main_loop = None  # Store main event loop reference
        
        # Reconnection control
        self.auto_reconnect = True
        self.reconnect_task = None
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 60.0
        self.reconnect_exponential_base = 2.0
        self.connection_lost_time = None
        
        # Connection monitoring
        self.last_message_time = None
        self.message_timeout = 30.0  # Consider connection dead after 30s of no messages
        self.connection_monitor_task = None
        
        # Subscription tracking
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize official SDK components
        self.info = Info(self.base_url, skip_ws=True)  # REST API only
        
        # Retry configuration
        self.max_retries = 3
        self.base_retry_delay = 1.0
        self.max_retry_delay = 10.0
        
        logger.info(f"HyperLiquid integration initialized with base URL: {self.base_url}")
        
        # Authentication setup
        self.private_key = self.config.api.hyperliquid_private_key
        self.wallet_address = self.config.api.hyperliquid_wallet_address
        
        # Connection optimization
        self.max_subscriptions = 50
        self.subscription_health: Dict[str, bool] = {}
        
        # If authenticated, initialize exchange API
        if self.private_key:
            self.exchange = Exchange(
                private_key=self.private_key,
                base_url=self.base_url,
                skip_ws=True
            )
            logger.info("Exchange API initialized for authenticated trading")
    
    def _get_api_url(self) -> str:
        """Get the appropriate API URL based on configuration."""
        # Always use mainnet for live market data collection
        return constants.MAINNET_API_URL
    
    async def connect(self) -> bool:
        """
        Connect to HyperLiquid WebSocket for real-time data.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to HyperLiquid WebSocket...")
            
            # Initialize WebSocket manager (thread-based, not async)
            self.ws_manager = WebsocketManager(
                base_url=self.base_url
            )
            
            # Start the WebSocket connection (synchronous thread start)
            self.ws_manager.start()
            
            # Wait longer and validate connection
            await asyncio.sleep(1.0)
            
            # Check if WebSocket thread is actually running
            if not self.ws_manager.is_alive():
                logger.error("WebSocket thread failed to start")
                return False
            
            self.connected = True
            # Store the running event loop for handler scheduling
            try:
                self._main_loop = asyncio.get_running_loop()
                logger.debug(f"Stored main event loop: {self._main_loop}")
            except RuntimeError:
                # Fallback for environments where no loop is running
                self._main_loop = asyncio.get_event_loop()
                logger.warning("No running loop found, using get_event_loop() fallback")
            
            # Store connection start time for monitoring
            self._connection_start_time = datetime.now(timezone.utc)
            
            logger.info("Successfully connected to HyperLiquid WebSocket")
            
            # Start connection monitoring
            if self.auto_reconnect:
                self._start_connection_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to HyperLiquid WebSocket: {e}")
            self.connected = False
            return False
    
    async def disconnect(self, disable_auto_reconnect: bool = False) -> None:
        """
        Disconnect from HyperLiquid WebSocket.
        
        Args:
            disable_auto_reconnect: If True, disable automatic reconnection
        """
        try:
            logger.info("Initiating HyperLiquid WebSocket disconnection...")
            
            # Disable auto-reconnect if requested
            if disable_auto_reconnect:
                self.auto_reconnect = False
                if self.reconnect_task:
                    self.reconnect_task.cancel()
                    self.reconnect_task = None
            
            # Stop connection monitoring
            if self.connection_monitor_task:
                self.connection_monitor_task.cancel()
                self.connection_monitor_task = None
            
            # First, mark as disconnected to stop message processing
            self.connected = False
            
            # Clear subscriptions and handlers first to prevent new processing
            old_subscriptions = dict(self.subscriptions)
            self.subscriptions.clear()
            self.message_handlers.clear()
            
            # Clear the main loop reference to prevent scheduling
            self._main_loop = None
            
            # Now stop the WebSocket manager
            if self.ws_manager:
                try:
                    logger.debug("Stopping WebSocket manager...")
                    self.ws_manager.stop()
                    
                    # Wait for thread to stop with a longer timeout
                    if self.ws_manager.is_alive():
                        logger.debug("Waiting for WebSocket thread to stop...")
                        self.ws_manager.join(timeout=5.0)
                        
                        # Force terminate if still alive
                        if self.ws_manager.is_alive():
                            logger.warning("WebSocket thread did not stop gracefully, forcing termination")
                            
                except Exception as e:
                    logger.warning(f"Error stopping WebSocket manager: {e}")
                finally:
                    self.ws_manager = None
            
            # Start reconnection if enabled and not explicitly disabled
            if not disable_auto_reconnect and self.auto_reconnect:
                self._start_reconnection(old_subscriptions)
            
            logger.info("Successfully disconnected from HyperLiquid WebSocket")
            
        except Exception as e:
            logger.error(f"Error during WebSocket disconnection: {e}")
            # Ensure we're marked as disconnected even if there's an error
            self.connected = False
            self._main_loop = None
    
    def _start_reconnection(self, old_subscriptions: Dict[str, Any]) -> None:
        """Start the reconnection process."""
        if not self.auto_reconnect:
            return
            
        # Record when connection was lost for metrics
        self.connection_lost_time = datetime.now(timezone.utc)
        
        # Cancel any existing reconnection task
        if self.reconnect_task:
            self.reconnect_task.cancel()
        
        # Start new reconnection task
        try:
            self.reconnect_task = asyncio.create_task(
                self._reconnection_loop(old_subscriptions)
            )
            logger.info("Started automatic reconnection process")
        except Exception as e:
            logger.error(f"Failed to start reconnection task: {e}")
    
    async def _reconnection_loop(self, old_subscriptions: Dict[str, Any]) -> None:
        """
        Continuous reconnection loop with exponential backoff.
        
        Args:
            old_subscriptions: Subscriptions to restore after reconnection
        """
        current_delay = self.reconnect_delay
        attempt = 1
        
        logger.info("Starting reconnection loop...")
        
        while self.auto_reconnect and not self.connected:
            try:
                logger.info(f"Reconnection attempt #{attempt} (delay: {current_delay:.1f}s)")
                
                # Wait before attempting reconnection
                await asyncio.sleep(current_delay)
                
                # Attempt to reconnect
                success = await self._attempt_reconnection(old_subscriptions)
                
                if success:
                    logger.info("Successfully reconnected to HyperLiquid WebSocket!")
                    
                    # Reset delay for future disconnections
                    current_delay = self.reconnect_delay
                    
                    # Calculate downtime for metrics
                    if self.connection_lost_time:
                        downtime = datetime.now(timezone.utc) - self.connection_lost_time
                        logger.info(f"Connection restored after {downtime.total_seconds():.1f} seconds")
                        self.connection_lost_time = None
                    
                    # Wait a bit to see if connection is stable
                    await asyncio.sleep(5.0)
                    
                    # Check if we're still connected and receiving messages
                    if (self.connected and 
                        self.last_message_time and 
                        (datetime.now(timezone.utc) - self.last_message_time).total_seconds() < 10.0):
                        logger.info("✅ Reconnection successful and stable")
                        break
                    else:
                        logger.warning("Reconnection appeared successful but connection is not stable, retrying...")
                        success = False  # Force retry
                
                if not success:
                    # Exponential backoff with jitter
                    current_delay = min(
                        current_delay * self.reconnect_exponential_base,
                        self.max_reconnect_delay
                    )
                    # Add jitter (±25% of delay)
                    jitter = current_delay * 0.25 * (2 * asyncio.get_event_loop().time() % 1 - 1)
                    current_delay = max(0.1, current_delay + jitter)
                    
                    attempt += 1
                    logger.warning(f"Reconnection attempt #{attempt-1} failed, next attempt in {current_delay:.1f}s")
                    
            except asyncio.CancelledError:
                logger.info("Reconnection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in reconnection loop: {e}")
                current_delay = min(current_delay * 2, self.max_reconnect_delay)
                attempt += 1
        
        self.reconnect_task = None
        logger.debug("Reconnection loop ended")
    
    async def _attempt_reconnection(self, old_subscriptions: Dict[str, Any]) -> bool:
        """
        Attempt to reconnect and restore subscriptions.
        
        Args:
            old_subscriptions: Subscriptions to restore
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Attempt to connect
            connected = await self.connect()
            if not connected:
                return False
            
            # Wait a moment for connection to stabilize
            await asyncio.sleep(0.5)
            
            # Restore subscriptions
            restored_count = 0
            for sub_key, sub_config in old_subscriptions.items():
                try:
                    # Parse subscription key to determine type
                    if sub_key == 'allMids':
                        success = await self.subscribe_all_mids()
                    elif sub_key.startswith('trades_'):
                        symbol = sub_key.replace('trades_', '')
                        success = await self.subscribe_trades(symbol)
                    elif sub_key.startswith('l2Book_'):
                        symbol = sub_key.replace('l2Book_', '')
                        success = await self.subscribe_orderbook(symbol)
                    elif sub_key.startswith('candle_'):
                        # Parse candle subscription: candle_BTC_15m
                        parts = sub_key.split('_')
                        if len(parts) >= 3:
                            symbol = parts[1]
                            interval = parts[2]
                            success = await self.subscribe_candle(symbol, interval)
                        else:
                            success = False
                    else:
                        logger.warning(f"Unknown subscription type: {sub_key}")
                        success = False
                    
                    if success:
                        restored_count += 1
                    else:
                        logger.warning(f"Failed to restore subscription: {sub_key}")
                    
                    # Small delay between subscriptions
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error restoring subscription {sub_key}: {e}")
            
            # Only consider successful if we restored subscriptions AND have a working connection
            if restored_count > 0:
                logger.info(f"Restored {restored_count}/{len(old_subscriptions)} subscriptions")
                
                # Wait a moment to see if we start receiving messages
                await asyncio.sleep(2.0)
                
                # Check if we're actually receiving data
                if self.last_message_time:
                    time_since_last = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                    if time_since_last < 10.0:  # Recent message indicates working connection
                        return True
                
                # If no recent messages, don't claim success yet
                logger.debug("Connection established but no message flow detected yet")
                return True  # Still return True as connection might work, monitor will detect if it doesn't
            else:
                logger.warning("Failed to restore any subscriptions")
                return False
            
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")
            return False
    
    def _check_connection_health(self, error: Exception) -> None:
        """
        Check if an error indicates a connection failure and trigger reconnection.
        
        Args:
            error: Exception that occurred
        """
        if not self.connected or not self.auto_reconnect:
            return
        
        error_str = str(error).lower()
        connection_errors = [
            'connection closed',
            'connection lost',
            'connection to remote host was lost',
            'websocket connection is closed',
            'broken pipe',
            'connection reset',
            'network is unreachable',
            'connection timed out',
            'connection refused',
            'goodbye'  # HyperLiquid specific disconnect message
        ]
        
        if any(err in error_str for err in connection_errors):
            logger.warning(f"Connection issue detected: {error}")
            # Force disconnection to trigger reconnection
            asyncio.create_task(self._trigger_reconnection())
    
    async def _trigger_reconnection(self) -> None:
        """Trigger reconnection process without explicitly disconnecting."""
        if not self.auto_reconnect:
            return
            
        try:
            # Store current subscriptions
            old_subscriptions = dict(self.subscriptions)
            
            # Mark as disconnected to stop processing
            self.connected = False
            
            # Start reconnection process
            self._start_reconnection(old_subscriptions)
            
        except Exception as e:
            logger.error(f"Error triggering reconnection: {e}")
    
    def add_message_handler(self, subscription_type: str, handler: Callable) -> None:
        """
        Add a message handler for a specific subscription type.
        
        Args:
            subscription_type: Type of subscription (e.g., 'allMids', 'l2Book', 'trades')
            handler: Callback function to handle messages
        """
        if subscription_type not in self.message_handlers:
            self.message_handlers[subscription_type] = []
        
        self.message_handlers[subscription_type].append(handler)
        logger.debug(f"Added message handler for subscription type: {subscription_type}")
    
    def _schedule_async_handler(self, handler: Callable, message: Dict[str, Any], handler_type: str) -> None:
        """
        Schedule an async handler to run in the main event loop.
        
        Args:
            handler: Async handler function to call
            message: Message to pass to the handler
            handler_type: Type of handler for logging purposes
        """
        # Quick exit if not connected - prevents endless warnings during shutdown
        if not self.connected:
            return
            
        try:
            # Check if we have a valid main loop reference
            if not hasattr(self, '_main_loop') or not self._main_loop:
                # Only log this during normal operation, not shutdown
                if self.connected:
                    logger.warning(f"Could not schedule async {handler_type} handler: no main loop stored")
                return
            
            # Check if the loop is still running and not closed
            if self._main_loop.is_closed():
                # Only log this during normal operation, not shutdown
                if self.connected:
                    logger.warning(f"Could not schedule async {handler_type} handler: main loop is closed")
                return
            
            # Try to schedule the handler
            future = asyncio.run_coroutine_threadsafe(handler(message), self._main_loop)
            
            # Add a done callback to handle any exceptions
            def handle_result(fut):
                try:
                    fut.result()  # This will raise any exception that occurred
                except Exception as e:
                    # Only log errors if we're still connected
                    if self.connected:
                        logger.error(f"Exception in async {handler_type} handler: {e}")
            
            future.add_done_callback(handle_result)
            
        except Exception as e:
            # Only log warnings if we're still connected 
            if self.connected:
                logger.warning(f"Could not schedule async {handler_type} handler: {e}")
    
    async def subscribe_all_mids(self, handler: Optional[Callable] = None) -> bool:
        """
        Subscribe to all mid price updates.
        
        Args:
            handler: Optional callback function for price updates
            
        Returns:
            bool: True if subscription successful
        """
        if not self.connected or not self.ws_manager:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Add handler if provided
            if handler:
                self.add_message_handler('allMids', handler)
            
            # Subscribe using official SDK format
            subscription = {"type": "allMids"}
            
            # Use the WebSocket manager's subscribe method
            self.ws_manager.subscribe(subscription, self._handle_all_mids_message)
            
            # Track subscription
            self.subscriptions['allMids'] = subscription
            
            logger.debug("Subscribed to all mids")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to all mids: {e}")
            return False
    
    async def subscribe_orderbook(self, symbol: str, handler: Optional[Callable] = None) -> bool:
        """
        Subscribe to order book updates for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            handler: Optional callback function for order book updates
            
        Returns:
            bool: True if subscription successful
        """
        if not self.connected or not self.ws_manager:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Add handler if provided
            if handler:
                self.add_message_handler('l2Book', handler)
            
            # Subscribe using official SDK format
            subscription = {"type": "l2Book", "coin": symbol}
            
            # Use the WebSocket manager's subscribe method
            self.ws_manager.subscribe(subscription, self._handle_orderbook_message)
            
            # Track subscription
            subscription_key = f'l2Book_{symbol}'
            self.subscriptions[subscription_key] = subscription
            
            logger.debug(f"Subscribed to order book for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to order book for {symbol}: {e}")
            return False
    
    async def subscribe_trades(self, symbol: str, handler: Optional[Callable] = None) -> bool:
        """
        Subscribe to trade updates for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            handler: Optional callback function for trade updates
            
        Returns:
            bool: True if subscription successful
        """
        if not self.connected or not self.ws_manager:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Add handler if provided
            if handler:
                self.add_message_handler('trades', handler)
            
            # Subscribe using official SDK format
            subscription = {"type": "trades", "coin": symbol}
            
            # Use the WebSocket manager's subscribe method
            self.ws_manager.subscribe(subscription, self._handle_trades_message)
            
            # Track subscription
            self.subscriptions[f'trades_{symbol}'] = subscription
            
            logger.debug(f"Subscribed to trades for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to trades for {symbol}: {e}")
            return False

    async def subscribe_candle(self, symbol: str, interval: str, handler: Optional[Callable] = None) -> bool:
        """
        Subscribe to candle/kline updates for a specific symbol and interval.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            interval: Candle interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            handler: Optional callback function for candle updates
            
        Returns:
            bool: True if subscription successful
        """
        if not self.connected or not self.ws_manager:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Add handler if provided
            if handler:
                self.add_message_handler('candle', handler)
            
            # Subscribe using official SDK format (based on HyperLiquid docs)
            subscription = {"type": "candle", "coin": symbol, "interval": interval}
            
            # Use the WebSocket manager's subscribe method
            self.ws_manager.subscribe(subscription, self._handle_candle_message)
            
            # Track subscription
            self.subscriptions[f'candle_{symbol}_{interval}'] = subscription
            
            logger.debug(f"Subscribed to {interval} candles for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {interval} candles for {symbol}: {e}")
            return False
    
    def _handle_all_mids_message(self, message: Dict[str, Any]) -> None:
        """Handle all mids price update messages."""
        # Skip processing if not connected (during shutdown)
        if not self.connected:
            return
        
        # Update last message time for connection monitoring
        self.last_message_time = datetime.now(timezone.utc)
        
        # Update connection health
        self.monitor.record_request(0.1, True)
            
        try:
            # Call registered handlers
            for handler in self.message_handlers.get('allMids', []):
                try:
                    # Since WebSocket manager is thread-based, we need to handle both sync and async
                    if asyncio.iscoroutinefunction(handler):
                        # Use threadsafe method to schedule async handlers
                        self._schedule_async_handler(handler, message, "all mids")
                    else:
                        handler(message)
                except Exception as e:
                    if self.connected:  # Only log if still connected
                        logger.error(f"Error in all mids message handler: {e}")
                    
        except Exception as e:
            if self.connected:  # Only log if still connected
                logger.error(f"Error handling all mids message: {e}")
                # Check if this might be a connection issue
                self._check_connection_health(e)
    
    def _handle_orderbook_message(self, message: Dict[str, Any]) -> None:
        """Handle order book update messages."""
        # Skip processing if not connected (during shutdown)
        if not self.connected:
            return
        
        # Update last message time for connection monitoring
        self.last_message_time = datetime.now(timezone.utc)
        
        try:
            # Extract symbol from message
            data = message.get('data', {})
            symbol = data.get('coin', 'unknown')
            
            # Call registered handlers - use 'l2Book' as the general handler key
            for handler in self.message_handlers.get('l2Book', []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        # Use threadsafe method to schedule async handlers
                        self._schedule_async_handler(handler, message, f"order book {symbol}")
                    else:
                        handler(message)
                except Exception as e:
                    if self.connected:  # Only log if still connected
                        logger.error(f"Error in order book message handler: {e}")
                    
        except Exception as e:
            if self.connected:  # Only log if still connected
                logger.error(f"Error handling order book message: {e}")
                # Check if this might be a connection issue
                self._check_connection_health(e)
    
    def _handle_trades_message(self, message: Dict[str, Any]) -> None:
        """
        Handle incoming trade messages.
        
        Args:
            message: WebSocket message containing trade data
        """
        # Skip processing if not connected (during shutdown)
        if not self.connected:
            return
        
        # Update last message time for connection monitoring
        self.last_message_time = datetime.now(timezone.utc)
        
        try:
            # Call all registered handlers for trades
            for handler in self.message_handlers.get('trades', []):
                try:
                    # Handle both async and sync handlers
                    if asyncio.iscoroutinefunction(handler):
                        # Use threadsafe method to schedule async handlers in main loop
                        self._schedule_async_handler(handler, message, "trades")
                    else:
                        handler(message)
                except Exception as e:
                    if self.connected:  # Only log if still connected
                        logger.error(f"Error in trades message handler: {e}")
        except Exception as e:
            if self.connected:  # Only log if still connected
                logger.error(f"Error handling trades message: {e}")
                # Check if this might be a connection issue
                self._check_connection_health(e)

    def _handle_candle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle incoming candle/kline messages.
        
        Args:
            message: WebSocket message containing candle data
        """
        # Skip processing if not connected (during shutdown)
        if not self.connected:
            return
        
        # Update last message time for connection monitoring
        self.last_message_time = datetime.now(timezone.utc)
        
        try:
            # Call all registered handlers for candles
            for handler in self.message_handlers.get('candle', []):
                try:
                    # Handle both async and sync handlers
                    if asyncio.iscoroutinefunction(handler):
                        # Use threadsafe method to schedule async handlers in main loop
                        self._schedule_async_handler(handler, message, "candles")
                    else:
                        handler(message)
                except Exception as e:
                    if self.connected:  # Only log if still connected
                        logger.error(f"Error in candles message handler: {e}")
                    
        except Exception as e:
            if self.connected:  # Only log if still connected
                logger.error(f"Error handling candles message: {e}")
                # Check if this might be a connection issue
                self._check_connection_health(e)
    
    # REST API Methods using official SDK
    
    async def _make_api_call(self, operation: str, api_func: Callable, *args, **kwargs) -> Any:
        """
        Make an API call with rate limiting, monitoring, and error handling.
        
        Args:
            operation: Name of the operation for logging
            api_func: API function to call
            *args, **kwargs: Arguments for the API function
            
        Returns:
            API call result
        """
        await self.rate_limiter.acquire()
        
        start_time = time.time()
        error_type = None
        
        for attempt in range(self.max_retries):
            try:
                result = api_func(*args, **kwargs)
                
                # Record successful request
                duration = time.time() - start_time
                self.monitor.record_request(duration, True)
                
                logger.debug(f"API call {operation} succeeded in {duration:.3f}s (attempt {attempt + 1})")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_str = str(e).lower()
                
                # Classify error type
                if any(term in error_str for term in ['connection', 'timeout', 'network', 'unreachable']):
                    error_type = 'connection'
                elif 'rate limit' in error_str:
                    error_type = 'rate_limit'
                elif any(term in error_str for term in ['auth', 'key', 'permission']):
                    error_type = 'auth'
                else:
                    error_type = 'api'
                
                self.monitor.record_request(duration, False, error_type)
                
                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    logger.error(f"API call {operation} failed after {self.max_retries} attempts: {e}")
                    raise
                
                # Calculate backoff delay
                delay = min(
                    self.base_retry_delay * (2 ** attempt),
                    self.max_retry_delay
                )
                
                logger.warning(f"API call {operation} failed (attempt {attempt + 1}/{self.max_retries}), retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)

    async def get_all_mids(self) -> Dict[str, str]:
        """Get all mid prices with rate limiting."""
        try:
            return await self._make_api_call("get_all_mids", self.info.all_mids)
        except Exception as e:
            logger.error(f"Failed to get all mids: {e}")
            return {}

    async def get_meta(self) -> List[Dict[str, Any]]:
        """Get metadata with rate limiting."""
        try:
            return await self._make_api_call("get_meta", self.info.meta)
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return {}

    async def get_candles(
        self, 
        symbol: str, 
        interval: str = "1m", 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get candlestick data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            List of candlestick data
        """
        try:
            # Convert intervals to HyperLiquid format
            interval_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            hl_interval = interval_map.get(interval, '1m')
            
            # If no times provided, get last 24 hours of data
            if not start_time or not end_time:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=24)
            
            # Convert to milliseconds - HyperLiquid requires this
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            # Use the official SDK's candle request
            candles = self.info.candles_snapshot(name=symbol, interval=hl_interval, startTime=start_ms, endTime=end_ms)
            
            return candles if candles else []
            
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            return []

    async def get_historical_candles_bulk(
        self,
        symbol: str,
        interval: str = "1m",
        days_back: int = 7,
        max_candles_per_request: int = 5000,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical candlestick data with pagination support for large datasets.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d') 
            days_back: Number of days to go back
            max_candles_per_request: Maximum candles per API request
            progress_callback: Optional callback function for progress updates (current, total)
            
        Returns:
            List of all candlestick data ordered chronologically
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days_back)
            
            # Calculate expected number of candles based on interval
            interval_minutes = self._interval_to_minutes(interval)
            total_minutes = int((end_time - start_time).total_seconds() / 60)
            expected_candles = total_minutes // interval_minutes
            
            logger.info(f"Collecting historical data for {symbol}: {days_back} days of {interval} candles")
            logger.info(f"Expected approximately {expected_candles} candles")
            
            all_candles = []
            current_end = end_time
            collected_count = 0
            request_count = 0
            
            while current_end > start_time:
                request_count += 1
                
                # Calculate time window for this request  
                window_minutes = max_candles_per_request * interval_minutes
                window_start = current_end - timedelta(minutes=window_minutes)
                
                # Don't go before our target start time
                if window_start < start_time:
                    window_start = start_time
                
                # Get candles for this time window
                logger.debug(f"Request {request_count}: Getting candles from {window_start} to {current_end}")
                
                window_candles = await self.get_candles(
                    symbol=symbol,
                    interval=interval,
                    start_time=window_start,
                    end_time=current_end
                )
                
                if window_candles:
                    # Add to collection (newer candles first, so prepend)
                    all_candles = window_candles + all_candles
                    collected_count += len(window_candles)
                    
                    logger.debug(f"Got {len(window_candles)} candles, total: {collected_count}")
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_percentage = min(100, (collected_count / expected_candles) * 100)
                        progress_callback(collected_count, expected_candles)
                
                # Move to next time window
                current_end = window_start
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
                
                # Safety check to avoid infinite loops
                if request_count > 100:
                    logger.warning(f"Reached maximum request limit (100) for {symbol}")
                    break
            
            logger.info(f"Historical collection complete for {symbol}: {len(all_candles)} candles in {request_count} requests")
            
            # Sort by timestamp to ensure chronological order
            if all_candles:
                all_candles.sort(key=lambda x: int(x.get('t', 0)))
            
            return all_candles
            
        except Exception as e:
            logger.error(f"Failed to get bulk historical candles for {symbol}: {e}")
            return []

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        interval_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return interval_map.get(interval, 1)

    async def get_historical_trades_bulk(
        self,
        symbol: str,
        days_back: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical trade data with pagination support.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            days_back: Number of days to go back
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of all trade data ordered chronologically
        """
        try:
            # Note: HyperLiquid may have limitations on historical trade data
            # This is a framework for when/if that functionality becomes available
            
            logger.info(f"Historical trade collection requested for {symbol} ({days_back} days)")
            logger.warning("Historical trade bulk collection not yet implemented in HyperLiquid SDK")
            
            # For now, return empty list
            # Future implementation would use similar pagination as candles
            return []
            
        except Exception as e:
            logger.error(f"Failed to get bulk historical trades for {symbol}: {e}")
            return []

    async def collect_multiple_symbols_historical(
        self,
        symbols: List[str],
        interval: str = "1h", 
        days_back: int = 7,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect historical data for multiple symbols efficiently.
        
        Args:
            symbols: List of trading symbols
            interval: Time interval for candles
            days_back: Number of days to go back
            progress_callback: Optional callback for progress (symbol, current, total)
            
        Returns:
            Dictionary mapping symbol to list of candles
        """
        try:
            logger.info(f"Bulk historical collection for {len(symbols)} symbols: {interval} candles, {days_back} days")
            
            results = {}
            total_symbols = len(symbols)
            
            for i, symbol in enumerate(symbols, 1):
                logger.info(f"Collecting historical data for {symbol} ({i}/{total_symbols})")
                
                symbol_progress_callback = None
                if progress_callback:
                    def symbol_callback(current, total):
                        progress_callback(symbol, current, total)
                    symbol_progress_callback = symbol_callback
                
                candles = await self.get_historical_candles_bulk(
                    symbol=symbol,
                    interval=interval, 
                    days_back=days_back,
                    progress_callback=symbol_progress_callback
                )
                
                results[symbol] = candles
                
                logger.info(f"Completed {symbol}: {len(candles)} candles")
                
                # Respect rate limits between symbols
                if i < total_symbols:
                    await asyncio.sleep(0.5)
            
            total_candles = sum(len(candles) for candles in results.values())
            logger.info(f"Bulk collection complete: {total_candles} total candles across {len(symbols)} symbols")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed bulk historical collection: {e}")
            return {}
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get order book with rate limiting."""
        try:
            return await self._make_api_call("get_order_book", self.info.l2_snapshot, symbol)
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return {}
    
    async def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the current funding rate for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            
        Returns:
            Dictionary with funding rate information or None if not available
        """
        try:
            meta = await self._make_api_call("get_funding_rate", self.info.meta)
            
            if not meta or 'universe' not in meta:
                return None
                
            # Find the symbol in the universe
            for coin_info in meta['universe']:
                if coin_info.get('name') == symbol:
                    # Check if funding rate information is available
                    if 'funding' in coin_info:
                        return {
                            'fundingRate': coin_info['funding'].get('fundingRate', '0'),
                            'premium': coin_info['funding'].get('premium'),
                            'nextFundingTime': coin_info['funding'].get('nextFundingTime')
                        }
                    else:
                        # Return a placeholder structure for symbols that don't have funding rates
                        return {
                            'fundingRate': '0',
                            'premium': None,
                            'nextFundingTime': None
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Perform a simple health check of the API."""
        try:
            # Try to get mid prices as a simple test
            await self.get_all_mids()
            logger.info("HyperLiquid API health check passed")
            return True
            
        except Exception as e:
            logger.error(f"HyperLiquid API health check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "base_url": "https://api.hyperliquid.xyz",
            "ws_url": "wss://api.hyperliquid.xyz/ws",
            "testnet": False  # Using mainnet for live market data collection
        }
    
    def is_authenticated(self) -> bool:
        """Check if the client is authenticated for trading."""
        return bool(self.private_key and self.exchange)
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected and self.ws_manager is not None

    async def optimize_subscriptions(self) -> None:
        """
        Optimize WebSocket subscriptions based on health and usage.
        """
        try:
            if not self.connected or not self.ws_manager:
                return
            
            # Check subscription health
            unhealthy_subscriptions = []
            
            for sub_key, is_healthy in self.subscription_health.items():
                if not is_healthy:
                    unhealthy_subscriptions.append(sub_key)
            
            # Remove unhealthy subscriptions
            for sub_key in unhealthy_subscriptions:
                logger.warning(f"Removing unhealthy subscription: {sub_key}")
                self.subscriptions.pop(sub_key, None)
                self.subscription_health.pop(sub_key, None)
            
            # Check if we're over subscription limit
            if len(self.subscriptions) > self.max_subscriptions:
                # Keep only the most recent subscriptions
                subscription_items = list(self.subscriptions.items())
                subscriptions_to_remove = subscription_items[:-self.max_subscriptions]
                
                for sub_key, _ in subscriptions_to_remove:
                    logger.info(f"Removing subscription due to limit: {sub_key}")
                    self.subscriptions.pop(sub_key, None)
                    self.subscription_health.pop(sub_key, None)
            
            logger.debug(f"Subscription optimization complete. Active: {len(self.subscriptions)}")
            
        except Exception as e:
            logger.error(f"Failed to optimize subscriptions: {e}")

    async def get_connection_health(self) -> Dict[str, Any]:
        """
        Get comprehensive connection health information.
        
        Returns:
            Dictionary with health metrics and statistics
        """
        try:
            stats = self.monitor.get_stats()
            
            health_info = {
                'connected': self.connected,
                'healthy': self.monitor.is_healthy(),
                'subscriptions': {
                    'active': len(self.subscriptions),
                    'max_allowed': self.max_subscriptions,
                    'healthy': sum(1 for h in self.subscription_health.values() if h),
                    'unhealthy': sum(1 for h in self.subscription_health.values() if not h)
                },
                'api_performance': stats,
                'rate_limiting': {
                    'requests_per_second': self.rate_limiter.requests_per_second,
                    'current_tokens': self.rate_limiter.tokens,
                    'burst_capacity': self.rate_limiter.burst_size
                },
                'configuration': {
                    'base_url': self.base_url,
                    'max_retries': self.max_retries,
                    'connection_timeout': self.max_retry_delay,
                    'authenticated': self.is_authenticated()
                }
            }
            
            return health_info
            
        except Exception as e:
            logger.error(f"Failed to get connection health: {e}")
            return {'error': str(e)}

    async def validate_connection(self) -> bool:
        """
        Validate connection by making a simple API call.
        
        Returns:
            True if connection is working properly
        """
        try:
            await self.health_check()
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    async def reset_connection(self) -> bool:
        """
        Reset the connection and re-establish subscriptions.
        
        Returns:
            True if reset was successful
        """
        try:
            logger.info("Resetting HyperLiquid connection...")
            
            # Store current subscriptions
            old_subscriptions = dict(self.subscriptions)
            
            # Disconnect
            await self.disconnect()
            
            # Wait a moment
            await asyncio.sleep(2.0)
            
            # Reconnect
            connected = await self.connect()
            if not connected:
                logger.error("Failed to reconnect after reset")
                return False
            
            # Re-establish subscriptions
            for sub_key, sub_config in old_subscriptions.items():
                try:
                    # Parse subscription key to determine type
                    if sub_key == 'allMids':
                        await self.subscribe_all_mids()
                    elif sub_key.startswith('trades_'):
                        symbol = sub_key.replace('trades_', '')
                        await self.subscribe_trades(symbol)
                    elif sub_key.startswith('l2Book_'):
                        symbol = sub_key.replace('l2Book_', '')
                        await self.subscribe_orderbook(symbol)
                    
                    await asyncio.sleep(0.1)  # Small delay between subscriptions
                    
                except Exception as e:
                    logger.warning(f"Failed to re-establish subscription {sub_key}: {e}")
            
            logger.info(f"Connection reset complete. Re-established {len(self.subscriptions)} subscriptions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset connection: {e}")
            return False

    async def monitor_connection_health(self, check_interval: float = 30.0) -> None:
        """
        Background task to monitor connection health and perform maintenance.
        
        Args:
            check_interval: Seconds between health checks
        """
        try:
            logger.info(f"Starting connection health monitoring (interval: {check_interval}s)")
            
            while self.connected:
                try:
                    # Check overall health
                    is_healthy = self.monitor.is_healthy()
                    
                    if not is_healthy:
                        logger.warning("Connection health degraded, attempting optimization")
                        await self.optimize_subscriptions()
                        
                        # If still unhealthy, consider reset
                        if not self.monitor.is_healthy():
                            stats = self.monitor.get_stats()
                            if stats['error_rate'] > 50:  # More than 50% errors
                                logger.warning("High error rate detected, resetting connection")
                                await self.reset_connection()
                    
                    # Validate connection periodically
                    if not await self.validate_connection():
                        logger.warning("Connection validation failed, resetting")
                        await self.reset_connection()
                    
                    # Log health stats periodically
                    health = await self.get_connection_health()
                    logger.debug(f"Health check: {health['healthy']}, API success rate: {health['api_performance']['success_rate']:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Error during health monitoring: {e}")
                
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            logger.info("Connection health monitoring stopped")
        except Exception as e:
            logger.error(f"Connection health monitoring failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.monitor.get_stats()

    def reset_performance_stats(self) -> None:
        """Reset connection monitoring stats."""
        self.monitor = ConnectionMonitor()
    
    def enable_auto_reconnect(self) -> None:
        """Enable automatic reconnection on connection loss."""
        self.auto_reconnect = True
        logger.info("Automatic reconnection enabled")
    
    def disable_auto_reconnect(self) -> None:
        """Disable automatic reconnection on connection loss."""
        self.auto_reconnect = False
        # Cancel any existing reconnection task
        if self.reconnect_task:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        # Cancel connection monitoring
        if self.connection_monitor_task:
            self.connection_monitor_task.cancel()
            self.connection_monitor_task = None
        logger.info("Automatic reconnection disabled")
    
    def set_message_timeout(self, timeout_seconds: float) -> None:
        """
        Set the message timeout for connection monitoring.
        
        Args:
            timeout_seconds: Seconds to wait for messages before considering connection dead
        """
        self.message_timeout = timeout_seconds
        logger.info(f"Message timeout set to {timeout_seconds}s")
    
    def get_reconnection_stats(self) -> Dict[str, Any]:
        """Get reconnection statistics."""
        return {
            "auto_reconnect_enabled": self.auto_reconnect,
            "currently_connected": self.connected,
            "reconnection_task_active": self.reconnect_task is not None,
            "connection_monitor_active": self.connection_monitor_task is not None,
            "connection_lost_time": self.connection_lost_time.isoformat() if self.connection_lost_time else None,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "message_timeout_seconds": self.message_timeout,
            "current_delay": self.reconnect_delay,
            "max_delay": self.max_reconnect_delay
        }

    def _start_connection_monitoring(self) -> None:
        """Start monitoring connection health by tracking message flow."""
        if self.connection_monitor_task:
            self.connection_monitor_task.cancel()
        
        try:
            self.connection_monitor_task = asyncio.create_task(self._connection_monitor_loop())
            logger.debug("Started connection monitoring")
        except Exception as e:
            logger.error(f"Failed to start connection monitoring: {e}")
    
    async def _connection_monitor_loop(self) -> None:
        """Monitor connection health by checking message flow."""
        logger.debug("Connection monitoring loop started")
        
        while self.connected and self.auto_reconnect:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                if not self.connected:
                    break
                
                # Check if we've received any messages recently
                now = datetime.now(timezone.utc)
                
                # If we have subscriptions but no recent messages, consider connection dead
                if (self.subscriptions and 
                    self.last_message_time and 
                    (now - self.last_message_time).total_seconds() > self.message_timeout):
                    
                    logger.warning(f"No messages received for {self.message_timeout}s, connection appears dead")
                    await self._trigger_reconnection()
                    break
                
                # If we have subscriptions but have never received any messages after timeout, reconnect
                elif (self.subscriptions and 
                      not self.last_message_time and 
                      hasattr(self, '_connection_start_time') and
                      (now - self._connection_start_time).total_seconds() > self.message_timeout):
                    
                    logger.warning(f"No messages received for {self.message_timeout}s since connection start, triggering reconnection")
                    await self._trigger_reconnection()
                    break
                    
            except asyncio.CancelledError:
                logger.debug("Connection monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(5.0)  # Continue monitoring despite errors
        
        logger.debug("Connection monitoring loop ended")