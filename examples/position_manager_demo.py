#!/usr/bin/env python3
"""
Position Manager Agent Demo

This demo showcases the Position Manager's core functionality:
- Trade execution and position management
- Stop-loss and take-profit functionality
- Portfolio tracking and performance monitoring
- Risk management and real-time updates

Run with: python examples/position_manager_demo.py
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List

from src.bistoury.agents.position_manager_agent import (
    PositionManagerAgent, PositionManagerConfig
)
from src.bistoury.models.trading import PositionSide
from src.bistoury.models.agent_messages import (
    TradingSignalPayload, MarketDataPayload, Message, MessageType
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PositionManagerDemo:
    """Demo class for Position Manager Agent."""
    
    def __init__(self):
        # Create demo configuration
        self.config = PositionManagerConfig(
            initial_balance=Decimal('50000'),
            slippage_rate=Decimal('0.0005'),
            commission_rate=Decimal('0.0005'),
            min_position_size=Decimal('0.001'),
            max_position_size=Decimal('100'),
            enable_stop_loss=True,
            enable_take_profit=True,
            stop_loss_pct=Decimal('2.0'),
            take_profit_pct=Decimal('4.0')
        )
        
        # Create Position Manager
        self.position_manager = PositionManagerAgent(
            name="demo_position_manager",
            config=self.config
        )
        
        # Demo data
        self.current_prices = {
            "BTC": Decimal('50000'),
            "ETH": Decimal('3000'),
            "SOL": Decimal('100')
        }
        
        self.demo_signals = []
        self.price_history = []
    
    def generate_demo_signals(self) -> List[TradingSignalPayload]:
        """Generate demo trading signals."""
        signals = [
            # Initial BTC long signal
            TradingSignalPayload(
                symbol="BTC",
                signal_type="momentum",
                direction="BUY",
                confidence=0.8,
                strength=8.5,
                timeframe="1h",
                strategy="demo_strategy",
                reasoning="Strong bullish momentum detected on BTC"
            ),
            
            # ETH long signal
            TradingSignalPayload(
                symbol="ETH",
                signal_type="trend",
                direction="BUY",
                confidence=0.7,
                strength=7.2,
                timeframe="4h",
                strategy="demo_strategy",
                reasoning="Uptrend continuation pattern on ETH"
            ),
            
            # BTC reversal signal
            TradingSignalPayload(
                symbol="BTC",
                signal_type="reversal",
                direction="SELL",
                confidence=0.75,
                strength=7.8,
                timeframe="1h",
                strategy="demo_strategy",
                reasoning="Bearish reversal pattern forming on BTC"
            ),
            
            # SOL momentum signal
            TradingSignalPayload(
                symbol="SOL",
                signal_type="momentum",
                direction="BUY",
                confidence=0.65,
                strength=6.5,
                timeframe="2h",
                strategy="demo_strategy",
                reasoning="Momentum breakout on SOL"
            ),
        ]
        return signals
    
    def generate_price_movements(self) -> List[Dict]:
        """Generate realistic price movements for the demo."""
        movements = [
            # Initial prices
            {"BTC": Decimal('50000'), "ETH": Decimal('3000'), "SOL": Decimal('100')},
            
            # Small movements up
            {"BTC": Decimal('50500'), "ETH": Decimal('3050'), "SOL": Decimal('102')},
            {"BTC": Decimal('51000'), "ETH": Decimal('3100'), "SOL": Decimal('104')},
            
            # Bigger movements (some trigger profits/losses)
            {"BTC": Decimal('52000'), "ETH": Decimal('3200'), "SOL": Decimal('108')},  # BTC +4%
            {"BTC": Decimal('49000'), "ETH": Decimal('3150'), "SOL": Decimal('110')},  # BTC down to stop loss
            {"BTC": Decimal('48500'), "ETH": Decimal('3100'), "SOL": Decimal('112')},
            
            # Recovery
            {"BTC": Decimal('50000'), "ETH": Decimal('3250'), "SOL": Decimal('115')},
            {"BTC": Decimal('51500'), "ETH": Decimal('3300'), "SOL": Decimal('118')},
        ]
        return movements
    
    async def simulate_market_data_update(self, prices: Dict[str, Decimal]):
        """Simulate market data updates."""
        for symbol, price in prices.items():
            self.current_prices[symbol] = price
            
            # Create market data payload
            market_data = MarketDataPayload(
                symbol=symbol,
                price=price,
                timestamp=datetime.now(timezone.utc),
                source="demo"
            )
            
            # Simulate message handling
            message = Message(
                type=MessageType.DATA_PRICE_UPDATE,
                sender="demo_data_feed",
                payload=market_data
            )
            
            await self.position_manager._handle_market_data(message)
    
    async def execute_demo_signal(self, signal: TradingSignalPayload):
        """Execute a demo trading signal."""
        logger.info(f"\n📈 EXECUTING SIGNAL: {signal.direction} {signal.symbol}")
        logger.info(f"   Confidence: {signal.confidence}")
        logger.info(f"   Reasoning: {signal.reasoning}")
        
        # Set current price for the symbol
        if signal.symbol in self.current_prices:
            self.position_manager.current_prices[signal.symbol] = self.current_prices[signal.symbol]
        
        # Execute the signal
        await self.position_manager._execute_signal(signal)
        
        # Show results
        await self.show_portfolio_status()
    
    async def show_portfolio_status(self):
        """Display current portfolio status."""
        portfolio = await self.position_manager.get_portfolio_state()
        positions = await self.position_manager.get_positions()
        metrics = self.position_manager.get_performance_metrics()
        
        logger.info(f"\n💼 PORTFOLIO STATUS:")
        logger.info(f"   Total Balance: ${portfolio.total_balance:,.2f}")
        logger.info(f"   Available Balance: ${portfolio.available_balance:,.2f}")
        logger.info(f"   Equity: ${portfolio.equity:,.2f}")
        logger.info(f"   Unrealized P&L: ${portfolio.unrealized_pnl:,.2f}")
        logger.info(f"   Realized P&L: ${portfolio.realized_pnl:,.2f}")
        
        if positions:
            logger.info(f"\n📊 OPEN POSITIONS:")
            for symbol, position in positions.items():
                if position.is_open:
                    pnl_color = "🟢" if position.unrealized_pnl and position.unrealized_pnl > 0 else "🔴"
                    logger.info(f"   {symbol}: {position.side.value} {position.size} @ ${position.entry_price}")
                    logger.info(f"      Current: ${position.current_price}, P&L: {pnl_color}${position.unrealized_pnl or 0:.2f}")
                    if position.stop_loss:
                        logger.info(f"      Stop Loss: ${position.stop_loss:.2f}")
                    if position.take_profit:
                        logger.info(f"      Take Profit: ${position.take_profit:.2f}")
        
        logger.info(f"\n📈 PERFORMANCE METRICS:")
        logger.info(f"   Total Trades: {metrics['total_trades']}")
        logger.info(f"   Winning Trades: {metrics['winning_trades']}")
        logger.info(f"   Win Rate: {metrics['win_rate']:.1f}%")
        logger.info(f"   Total P&L: ${metrics['total_pnl']:.2f}")
    
    async def run_demo(self):
        """Run the complete Position Manager demo."""
        logger.info("🚀 Starting Position Manager Demo")
        logger.info("=" * 50)
        
        try:
            # Start the Position Manager
            await self.position_manager._start()
            
            # Generate demo data
            signals = self.generate_demo_signals()
            price_movements = self.generate_price_movements()
            
            # Initial portfolio status
            await self.show_portfolio_status()
            
            # Execute signals with price updates
            signal_idx = 0
            
            for i, prices in enumerate(price_movements):
                logger.info(f"\n⏰ MARKET UPDATE #{i+1}")
                logger.info(f"   Prices: {dict(prices)}")
                
                # Update market prices
                await self.simulate_market_data_update(prices)
                
                # Execute signal if available and conditions are right
                if signal_idx < len(signals) and i % 2 == 0:  # Execute every other update
                    await self.execute_demo_signal(signals[signal_idx])
                    signal_idx += 1
                
                # Small delay for demo effect
                await asyncio.sleep(0.5)
            
            # Final status
            logger.info(f"\n🏁 DEMO COMPLETE")
            logger.info("=" * 50)
            await self.show_portfolio_status()
            
            # Show final summary
            final_metrics = self.position_manager.get_performance_metrics()
            final_portfolio = await self.position_manager.get_portfolio_state()
            
            total_return = ((final_portfolio.equity - self.config.initial_balance) / self.config.initial_balance) * 100
            
            logger.info(f"\n📊 FINAL SUMMARY:")
            logger.info(f"   Starting Balance: ${self.config.initial_balance:,.2f}")
            logger.info(f"   Final Equity: ${final_portfolio.equity:,.2f}")
            logger.info(f"   Total Return: {total_return:+.2f}%")
            logger.info(f"   Total Trades: {final_metrics['total_trades']}")
            logger.info(f"   Win Rate: {final_metrics['win_rate']:.1f}%")
            
        except Exception as e:
            logger.error(f"Demo error: {e}", exc_info=True)
        
        finally:
            # Stop the Position Manager
            await self.position_manager._stop()


async def main():
    """Main demo function."""
    demo = PositionManagerDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 