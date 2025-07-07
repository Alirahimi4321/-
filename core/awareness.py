"""
Project AWARENESS - Core Kernel
The stateless asyncio entry point that bootstraps the multiprocessing agent system.
"""

import asyncio
import time
import signal
import sys
import multiprocessing as mp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import psutil
import uvloop

from core.config import AwarenessConfig
from core.logger import setup_logger
from core.message_bus import MessageBus
from core.resource_monitor import ResourceMonitor
from agents.router_agent import RouterAgent
from agents.memory_controller_agent import MemoryControllerAgent
from agents.context_agent import ContextAgent
from agents.security_agent import SecurityAgent
from agents.learning_agent import LearningAgent
from agents.watchdog_agent import WatchdogAgent


@dataclass
class AgentProcess:
    """Represents a running agent process."""
    name: str
    process: mp.Process
    pid: int
    start_time: float
    agent_class: type
    status: str = "running"
    last_heartbeat: float = field(default_factory=time.time)
    trust_score: float = 1.0
    

class AwarenessKernel:
    """
    The central kernel that orchestrates the multi-agent system.
    Implements the hybrid agent orchestrator pattern with multiprocessing.
    """
    
    def __init__(self, config: AwarenessConfig):
        self.config = config
        self.logger = setup_logger(__name__)
        self.start_time = time.time()
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.message_bus = MessageBus()
        self.resource_monitor = ResourceMonitor(config)
        self.agent_processes: Dict[str, AgentProcess] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.system_stats = {
            "uptime": 0,
            "active_agents": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "trust_score": 1.0,
            "total_requests": 0,
            "failed_requests": 0
        }
        
        # Agent definitions
        self.agent_definitions = [
            ("router", RouterAgent, {"priority": 10}),
            ("memory_controller", MemoryControllerAgent, {"priority": 9}),
            ("context", ContextAgent, {"priority": 8}),
            ("security", SecurityAgent, {"priority": 10}),
            ("learning", LearningAgent, {"priority": 5}),
            ("watchdog", WatchdogAgent, {"priority": 7}),
        ]
        
        # Setup asyncio event loop
        if sys.platform != "win32":
            uvloop.install()
            
    async def initialize(self):
        """Initialize the kernel and all subsystems."""
        if self.is_initialized:
            return
            
        self.logger.info("Initializing AWARENESS kernel...")
        
        try:
            # Initialize message bus
            await self.message_bus.initialize()
            
            # Initialize resource monitor
            await self.resource_monitor.initialize()
            
            # Start agent processes
            await self._start_agent_processes()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.is_initialized = True
            self.logger.info("AWARENESS kernel initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize kernel: {e}")
            raise
            
    async def run(self):
        """Main kernel run loop."""
        if not self.is_initialized:
            await self.initialize()
            
        self.is_running = True
        self.logger.info("Starting AWARENESS kernel main loop...")
        
        try:
            # Start background tasks
            tasks = [
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._resource_monitor_loop()),
                asyncio.create_task(self._agent_supervisor_loop()),
                asyncio.create_task(self._message_processing_loop()),
            ]
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Error in kernel main loop: {e}")
            raise
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def shutdown(self):
        """Gracefully shutdown the kernel and all agents."""
        self.logger.info("Initiating graceful shutdown...")
        
        self.is_running = False
        
        # Signal all agents to shutdown
        await self.message_bus.broadcast({
            "type": "system",
            "action": "shutdown",
            "timestamp": time.time()
        })
        
        # Wait for agents to shutdown gracefully
        await asyncio.sleep(2)
        
        # Terminate agent processes
        for agent_name, agent_process in self.agent_processes.items():
            if agent_process.process.is_alive():
                self.logger.info(f"Terminating agent: {agent_name}")
                agent_process.process.terminate()
                agent_process.process.join(timeout=5)
                
                if agent_process.process.is_alive():
                    self.logger.warning(f"Force killing agent: {agent_name}")
                    agent_process.process.kill()
                    
        # Shutdown message bus
        await self.message_bus.shutdown()
        
        # Shutdown resource monitor
        await self.resource_monitor.shutdown()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Set shutdown event
        self.shutdown_event.set()
        
        self.logger.info("Kernel shutdown complete")
        
    async def process_command(self, command: str) -> str:
        """Process a user command through the agent system."""
        try:
            # Send command to router agent
            message = {
                "type": "user_command",
                "command": command,
                "timestamp": time.time(),
                "request_id": f"cmd_{int(time.time() * 1000)}"
            }
            
            response = await self.message_bus.send_and_wait(
                target="router",
                message=message,
                timeout=30
            )
            
            self.system_stats["total_requests"] += 1
            
            return response.get("response", "No response received")
            
        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            self.system_stats["failed_requests"] += 1
            return f"Error: {str(e)}"
            
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        self.system_stats["uptime"] = time.time() - self.start_time
        self.system_stats["active_agents"] = len([
            p for p in self.agent_processes.values() 
            if p.status == "running"
        ])
        
        # Get resource usage
        resource_stats = await self.resource_monitor.get_stats()
        self.system_stats.update(resource_stats)
        
        return self.system_stats.copy()
        
    async def _start_agent_processes(self):
        """Start all agent processes."""
        self.logger.info("Starting agent processes...")
        
        for agent_name, agent_class, agent_config in self.agent_definitions:
            try:
                # Create process
                process = mp.Process(
                    target=self._agent_process_wrapper,
                    args=(agent_name, agent_class, agent_config, self.config),
                    name=f"awareness-{agent_name}"
                )
                
                process.start()
                
                # Track process
                self.agent_processes[agent_name] = AgentProcess(
                    name=agent_name,
                    process=process,
                    pid=process.pid,
                    start_time=time.time(),
                    agent_class=agent_class
                )
                
                self.logger.info(f"Started agent: {agent_name} (PID: {process.pid})")
                
            except Exception as e:
                self.logger.error(f"Failed to start agent {agent_name}: {e}")
                
    def _agent_process_wrapper(self, name: str, agent_class: type, agent_config: Dict, config: AwarenessConfig):
        """Wrapper for running agent in separate process."""
        try:
            # Setup logging for process
            logger = setup_logger(f"agent.{name}")
            
            # Create and run agent
            agent = agent_class(name, config, agent_config)
            
            # Run agent event loop
            if sys.platform != "win32":
                uvloop.install()
                
            asyncio.run(agent.run())
            
        except Exception as e:
            logger.error(f"Agent {name} crashed: {e}")
            sys.exit(1)
            
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to all agents."""
        while self.is_running:
            try:
                await self.message_bus.broadcast({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "kernel_uptime": time.time() - self.start_time
                })
                
                await asyncio.sleep(30)  # 30 second heartbeat
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(10)
                
    async def _resource_monitor_loop(self):
        """Monitor system resources."""
        while self.is_running:
            try:
                # Update resource stats
                await self.resource_monitor.update_stats()
                
                # Check resource thresholds
                stats = await self.resource_monitor.get_stats()
                
                if stats["memory_usage"] > 85:
                    self.logger.warning(f"High memory usage: {stats['memory_usage']}%")
                    await self._handle_resource_pressure("memory")
                    
                if stats["cpu_usage"] > 90:
                    self.logger.warning(f"High CPU usage: {stats['cpu_usage']}%")
                    await self._handle_resource_pressure("cpu")
                    
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitor loop: {e}")
                await asyncio.sleep(30)
                
    async def _agent_supervisor_loop(self):
        """Supervise agent processes and restart if needed."""
        while self.is_running:
            try:
                for agent_name, agent_process in list(self.agent_processes.items()):
                    if not agent_process.process.is_alive():
                        self.logger.warning(f"Agent {agent_name} died, restarting...")
                        await self._restart_agent(agent_name)
                        
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in agent supervisor loop: {e}")
                await asyncio.sleep(30)
                
    async def _message_processing_loop(self):
        """Process messages from the message bus."""
        while self.is_running:
            try:
                # Handle system messages
                messages = await self.message_bus.get_system_messages()
                
                for message in messages:
                    await self._handle_system_message(message)
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(10)
                
    async def _handle_resource_pressure(self, resource_type: str):
        """Handle resource pressure by throttling agents."""
        self.logger.info(f"Handling {resource_type} pressure")
        
        await self.message_bus.broadcast({
            "type": "resource_pressure",
            "resource": resource_type,
            "timestamp": time.time()
        })
        
    async def _handle_system_message(self, message: Dict[str, Any]):
        """Handle system messages."""
        msg_type = message.get("type")
        
        if msg_type == "agent_error":
            agent_name = message.get("agent")
            error = message.get("error")
            self.logger.error(f"Agent {agent_name} reported error: {error}")
            
        elif msg_type == "trust_score_update":
            agent_name = message.get("agent")
            score = message.get("score")
            if agent_name in self.agent_processes:
                self.agent_processes[agent_name].trust_score = score
                
    async def _restart_agent(self, agent_name: str):
        """Restart a failed agent."""
        if agent_name not in self.agent_processes:
            return
            
        agent_process = self.agent_processes[agent_name]
        
        # Find agent definition
        agent_def = None
        for name, agent_class, agent_config in self.agent_definitions:
            if name == agent_name:
                agent_def = (name, agent_class, agent_config)
                break
                
        if not agent_def:
            self.logger.error(f"Cannot restart agent {agent_name}: definition not found")
            return
            
        # Clean up old process
        if agent_process.process.is_alive():
            agent_process.process.terminate()
            agent_process.process.join(timeout=5)
            
        # Start new process
        try:
            name, agent_class, agent_config = agent_def
            process = mp.Process(
                target=self._agent_process_wrapper,
                args=(name, agent_class, agent_config, self.config),
                name=f"awareness-{name}"
            )
            
            process.start()
            
            # Update tracking
            self.agent_processes[agent_name] = AgentProcess(
                name=agent_name,
                process=process,
                pid=process.pid,
                start_time=time.time(),
                agent_class=agent_class
            )
            
            self.logger.info(f"Restarted agent: {agent_name} (PID: {process.pid})")
            
        except Exception as e:
            self.logger.error(f"Failed to restart agent {agent_name}: {e}")
            
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)