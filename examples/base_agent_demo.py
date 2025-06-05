#!/usr/bin/env python3
"""
Demonstration of the BaseAgent implementation.

This script shows how to create and use agents with the multi-agent framework.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path to import bistoury modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bistoury.agents.base import BaseAgent, AgentType, AgentState, AgentHealth


class DemoAgent(BaseAgent):
    """
    A simple demonstration agent that shows basic functionality.
    """
    
    def __init__(self, name: str = "demo_agent"):
        super().__init__(
            name=name,
            agent_type=AgentType.COLLECTOR,
            config={
                "demo_setting": "demo_value",
                "work_interval": 2.0
            }
        )
        self.work_count = 0
    
    async def _start(self) -> bool:
        """Start the demo agent's work."""
        self.logger.info("Demo agent starting up...")
        
        # Start background work task
        self.create_task(self._do_work())
        
        return True
    
    async def _stop(self) -> None:
        """Stop the demo agent."""
        self.logger.info("Demo agent shutting down...")
        # Cleanup is handled by the base class
    
    async def _health_check(self) -> AgentHealth:
        """Return health status."""
        return AgentHealth(
            state=self.state,
            cpu_usage=25.0,  # Simulated
            memory_usage=1024,  # Simulated
            messages_processed=self.work_count,
            tasks_completed=self.work_count // 2
        )
    
    async def _do_work(self) -> None:
        """Simulate doing work."""
        work_interval = self.get_config_value("work_interval", 1.0)
        
        while not self._stop_event.is_set():
            try:
                self.work_count += 1
                self.logger.info(f"Doing work iteration #{self.work_count}")
                
                # Simulate work
                await asyncio.sleep(work_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Work task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in work task: {e}")
                await asyncio.sleep(1)


async def main():
    """Demonstrate the agent functionality."""
    print("🤖 BaseAgent Demonstration")
    print("=" * 50)
    
    # Create a demo agent
    agent = DemoAgent("demo_agent_1")
    
    print(f"Agent created: {agent}")
    print(f"Initial state: {agent.state}")
    print(f"Agent type: {agent.agent_type}")
    print(f"Agent ID: {agent.agent_id}")
    
    try:
        # Start the agent
        print("\n🚀 Starting agent...")
        success = await agent.start()
        if success:
            print(f"✅ Agent started successfully! State: {agent.state}")
        else:
            print("❌ Failed to start agent")
            return
        
        # Let it run for a bit
        print("\n⏳ Letting agent run for 10 seconds...")
        await asyncio.sleep(10)
        
        # Check health
        health = await agent.get_health()
        print(f"\n📊 Agent Health:")
        print(f"  State: {health.state}")
        print(f"  Healthy: {health.is_healthy}")
        print(f"  Health Score: {health.health_score}")
        print(f"  CPU Usage: {health.cpu_usage}%")
        print(f"  Memory Usage: {health.memory_usage} bytes")
        print(f"  Messages Processed: {health.messages_processed}")
        print(f"  Tasks Completed: {health.tasks_completed}")
        print(f"  Uptime: {health.uptime_seconds:.1f} seconds")
        
        # Test pause/resume
        print("\n⏸️  Pausing agent...")
        await agent.pause()
        print(f"Agent state: {agent.state}")
        
        await asyncio.sleep(2)
        
        print("\n▶️  Resuming agent...")
        await agent.resume()
        print(f"Agent state: {agent.state}")
        
        await asyncio.sleep(3)
        
        # Test configuration update
        print("\n🔧 Updating configuration...")
        agent.update_config({"work_interval": 0.5})
        print("Work interval updated to 0.5 seconds")
        
        await asyncio.sleep(5)
        
    finally:
        # Stop the agent
        print("\n🛑 Stopping agent...")
        await agent.stop()
        print(f"✅ Agent stopped. Final state: {agent.state}")
        
        # Show final stats
        final_health = await agent.get_health()
        print(f"\n📈 Final Stats:")
        print(f"  Total work iterations: {agent.work_count}")
        print(f"  Total uptime: {agent.uptime:.1f} seconds")
        print(f"  Messages processed: {final_health.messages_processed}")


if __name__ == "__main__":
    asyncio.run(main()) 