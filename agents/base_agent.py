"""
Project AWARENESS - Base Agent
Base class for all agents in the AWARENESS system.
"""

import asyncio
import time
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import msgpack
import uuid

from core.config import AwarenessConfig
from core.logger import setup_logger


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    name: str
    status: str = "initializing"
    last_heartbeat: float = field(default_factory=time.time)
    message_count: int = 0
    error_count: int = 0
    uptime: float = 0
    trust_score: float = 1.0
    

@dataclass
class AgentMessage:
    """Represents a message processed by an agent."""
    id: str
    type: str
    sender: str
    payload: Dict[str, Any]
    timestamp: float
    reply_to: Optional[str] = None
    

class BaseAgent(ABC):
    """
    Base class for all agents in the AWARENESS system.
    Provides common functionality for message handling, state management, and lifecycle.
    """
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.agent_config = agent_config
        self.logger = setup_logger(f"agent.{name}")
        
        # Agent state
        self.state = AgentState(name=name)
        self.start_time = time.time()
        
        # Message handling
        self.message_queue: Optional[mp.Queue] = None
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Shutdown event
        self.shutdown_event = asyncio.Event()
        self.is_running = False
        
        # Register default message handlers
        self._register_default_handlers()
        
    async def run(self):
        """Main agent run loop."""
        self.logger.info(f"Starting agent: {self.name}")
        
        try:
            # Initialize agent
            await self.initialize()
            
            # Start background tasks
            self._start_background_tasks()
            
            # Main message processing loop
            await self._message_loop()
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} crashed: {e}")
            self.state.error_count += 1
            raise
        finally:
            await self.shutdown()
            
    async def initialize(self):
        """Initialize the agent. Override in subclasses."""
        self.logger.info(f"Initializing agent: {self.name}")
        
        # Connect to message bus
        await self._connect_to_message_bus()
        
        # Call subclass initialization
        await self._initialize()
        
        self.state.status = "running"
        self.is_running = True
        
        self.logger.info(f"Agent {self.name} initialized successfully")
        
    async def shutdown(self):
        """Shutdown the agent gracefully."""
        self.logger.info(f"Shutting down agent: {self.name}")
        
        self.is_running = False
        self.state.status = "shutting_down"
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
        # Call subclass shutdown
        await self._shutdown()
        
        self.state.status = "stopped"
        self.logger.info(f"Agent {self.name} shutdown complete")
        
    async def send_message(self, target: str, message_type: str, payload: Dict[str, Any]) -> str:
        """
        Send a message to another agent.
        
        Args:
            target: Target agent name
            message_type: Type of message
            payload: Message payload
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        
        # Create message
        message = {
            "id": message_id,
            "type": message_type,
            "sender": self.name,
            "target": target,
            "payload": payload,
            "timestamp": time.time()
        }
        
        # Send via message bus (implementation depends on message bus integration)
        # For now, this is a placeholder
        
        return message_id
        
    async def send_and_wait(self, target: str, message_type: str, payload: Dict[str, Any], 
                           timeout: float = 30.0) -> Dict[str, Any]:
        """
        Send a message and wait for response.
        
        Args:
            target: Target agent name
            message_type: Type of message
            payload: Message payload
            timeout: Response timeout
            
        Returns:
            Response payload
        """
        message_id = await self.send_message(target, message_type, payload)
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[message_id] = response_future
        
        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response from {target}")
            raise
        finally:
            # Clean up
            if message_id in self.pending_responses:
                del self.pending_responses[message_id]
                
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a message handler.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")
        
    def get_state(self) -> Dict[str, Any]:
        """Get agent state information."""
        self.state.uptime = time.time() - self.start_time
        
        return {
            "name": self.state.name,
            "status": self.state.status,
            "uptime": self.state.uptime,
            "message_count": self.state.message_count,
            "error_count": self.state.error_count,
            "trust_score": self.state.trust_score,
            "last_heartbeat": self.state.last_heartbeat
        }
        
    async def _message_loop(self):
        """Main message processing loop."""
        while self.is_running:
            try:
                # Check for shutdown signal
                if self.shutdown_event.is_set():
                    break
                    
                # Process messages
                await self._process_messages()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")
                self.state.error_count += 1
                await asyncio.sleep(1)
                
    async def _process_messages(self):
        """Process messages from the queue."""
        if not self.message_queue:
            return
            
        try:
            # Get messages from queue (non-blocking)
            messages = []
            while not self.message_queue.empty():
                try:
                    message_data = self.message_queue.get_nowait()
                    message = msgpack.unpackb(message_data, raw=False)
                    messages.append(message)
                except:
                    break
                    
            # Process each message
            for message_data in messages:
                await self._handle_message(message_data)
                
        except Exception as e:
            self.logger.error(f"Error processing messages: {e}")
            
    async def _handle_message(self, message_data: Dict[str, Any]):
        """Handle a single message."""
        try:
            message = AgentMessage(
                id=message_data.get("id", ""),
                type=message_data.get("type", "unknown"),
                sender=message_data.get("sender", "unknown"),
                payload=message_data.get("payload", {}),
                timestamp=message_data.get("timestamp", time.time()),
                reply_to=message_data.get("reply_to")
            )
            
            self.state.message_count += 1
            
            # Check if this is a response to a pending request
            if message.reply_to and message.reply_to in self.pending_responses:
                future = self.pending_responses[message.reply_to]
                if not future.done():
                    future.set_result(message.payload)
                return
                
            # Handle message based on type
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message)
            else:
                await self._handle_unknown_message(message)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.state.error_count += 1
            
    async def _handle_unknown_message(self, message: AgentMessage):
        """Handle unknown message types."""
        self.logger.warning(f"Unknown message type: {message.type} from {message.sender}")
        
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.register_handler("heartbeat", self._handle_heartbeat)
        self.register_handler("system", self._handle_system_message)
        self.register_handler("shutdown", self._handle_shutdown)
        self.register_handler("status", self._handle_status_request)
        
    async def _handle_heartbeat(self, message: AgentMessage):
        """Handle heartbeat message."""
        self.state.last_heartbeat = time.time()
        self.logger.debug("Received heartbeat")
        
    async def _handle_system_message(self, message: AgentMessage):
        """Handle system message."""
        action = message.payload.get("action")
        
        if action == "shutdown":
            self.logger.info("Received shutdown command")
            self.shutdown_event.set()
            
    async def _handle_shutdown(self, message: AgentMessage):
        """Handle shutdown message."""
        self.logger.info("Received shutdown message")
        self.shutdown_event.set()
        
    async def _handle_status_request(self, message: AgentMessage):
        """Handle status request."""
        status = self.get_state()
        
        # Send response (implementation depends on message bus)
        response = {
            "id": str(uuid.uuid4()),
            "type": "status_response",
            "sender": self.name,
            "target": message.sender,
            "payload": status,
            "timestamp": time.time(),
            "reply_to": message.id
        }
        
        # Send response via message bus
        # This is a placeholder - actual implementation depends on message bus
        
    def _start_background_tasks(self):
        """Start background tasks."""
        # Health check task
        self.background_tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
        
        # Custom background tasks from subclasses
        custom_tasks = self._get_background_tasks()
        for task in custom_tasks:
            self.background_tasks.append(asyncio.create_task(task))
            
    async def _health_check_loop(self):
        """Health check background task."""
        while self.is_running:
            try:
                await self._health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                await asyncio.sleep(60)
                
    async def _health_check(self):
        """Perform health check."""
        # Update trust score based on performance
        if self.state.error_count > 0:
            error_rate = self.state.error_count / max(1, self.state.message_count)
            self.state.trust_score = max(0.1, 1.0 - error_rate)
        else:
            self.state.trust_score = min(1.0, self.state.trust_score + 0.01)
            
    async def _connect_to_message_bus(self):
        """Connect to the message bus."""
        # This is a placeholder - actual implementation would connect to the message bus
        # and set up the message queue
        pass
        
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    async def _initialize(self):
        """Initialize the agent. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    async def _shutdown(self):
        """Shutdown the agent. Must be implemented by subclasses."""
        pass
        
    def _get_background_tasks(self) -> List[Callable]:
        """Get additional background tasks. Override in subclasses."""
        return []
        
    async def _process_agent_message(self, message: AgentMessage):
        """Process agent-specific messages. Override in subclasses."""
        pass