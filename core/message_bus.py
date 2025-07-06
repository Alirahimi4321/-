"""
Project AWARENESS - Message Bus
Inter-process communication system using multiprocessing queues and Unix sockets.
"""

import asyncio
import time
import threading
import multiprocessing as mp
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import msgpack
import uuid
import socket
import os
from pathlib import Path

from core.logger import setup_logger


@dataclass
class Message:
    """Represents a message in the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "unknown"
    sender: str = "unknown"
    target: str = "broadcast"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 5  # 1-10, 10 being highest
    

@dataclass
class MessageResponse:
    """Represents a response to a message."""
    message_id: str
    success: bool
    response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """
    Central message bus for inter-process communication.
    Uses multiprocessing queues internally and Unix sockets for external communication.
    """
    
    def __init__(self, socket_path: str = "/tmp/awareness.sock"):
        self.socket_path = socket_path
        self.logger = setup_logger(__name__)
        
        # Internal queues for different message types
        self.queues: Dict[str, mp.Queue] = {}
        self.system_queue = mp.Queue()
        self.broadcast_queue = mp.Queue()
        
        # Response tracking
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.response_timeout = 30.0
        
        # Message handlers
        self.handlers: Dict[str, Callable] = {}
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # State
        self.is_running = False
        self.is_initialized = False
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts_sent": 0,
            "responses_sent": 0,
            "errors": 0
        }
        
    async def initialize(self):
        """Initialize the message bus."""
        if self.is_initialized:
            return
            
        self.logger.info("Initializing message bus...")
        
        try:
            # Create Unix socket for external communication
            await self._setup_unix_socket()
            
            # Start message processing tasks
            self._start_background_tasks()
            
            self.is_initialized = True
            self.logger.info("Message bus initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize message bus: {e}")
            raise
            
    async def shutdown(self):
        """Shutdown the message bus."""
        self.logger.info("Shutting down message bus...")
        
        self.is_running = False
        
        # Close Unix socket
        if hasattr(self, 'unix_socket'):
            self.unix_socket.close()
            
        # Clean up socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
            
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Message bus shutdown complete")
        
    def register_agent(self, agent_name: str) -> mp.Queue:
        """
        Register an agent and get its dedicated queue.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dedicated queue for the agent
        """
        if agent_name not in self.queues:
            self.queues[agent_name] = mp.Queue()
            self.logger.info(f"Registered agent: {agent_name}")
            
        return self.queues[agent_name]
        
    def unregister_agent(self, agent_name: str):
        """
        Unregister an agent.
        
        Args:
            agent_name: Name of the agent to unregister
        """
        if agent_name in self.queues:
            # Close queue
            self.queues[agent_name].close()
            del self.queues[agent_name]
            self.logger.info(f"Unregistered agent: {agent_name}")
            
    async def send_message(self, target: str, message_type: str, payload: Dict[str, Any], 
                          sender: str = "kernel", priority: int = 5) -> str:
        """
        Send a message to a specific target.
        
        Args:
            target: Target agent name
            message_type: Type of message
            payload: Message payload
            sender: Sender identification
            priority: Message priority (1-10)
            
        Returns:
            Message ID
        """
        message = Message(
            type=message_type,
            sender=sender,
            target=target,
            payload=payload,
            priority=priority
        )
        
        try:
            if target == "broadcast":
                await self._send_broadcast(message)
            else:
                await self._send_to_agent(target, message)
                
            self.stats["messages_sent"] += 1
            return message.id
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {target}: {e}")
            self.stats["errors"] += 1
            raise
            
    async def send_and_wait(self, target: str, message: Dict[str, Any], 
                           timeout: float = 30.0) -> Dict[str, Any]:
        """
        Send a message and wait for response.
        
        Args:
            target: Target agent name
            message: Message to send
            timeout: Response timeout in seconds
            
        Returns:
            Response payload
        """
        message_id = await self.send_message(
            target=target,
            message_type=message.get("type", "request"),
            payload=message,
            sender=message.get("sender", "kernel")
        )
        
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
                
    async def broadcast(self, message: Dict[str, Any], sender: str = "kernel") -> int:
        """
        Broadcast a message to all agents.
        
        Args:
            message: Message to broadcast
            sender: Sender identification
            
        Returns:
            Number of agents the message was sent to
        """
        broadcast_message = Message(
            type=message.get("type", "broadcast"),
            sender=sender,
            target="broadcast",
            payload=message
        )
        
        count = await self._send_broadcast(broadcast_message)
        self.stats["broadcasts_sent"] += 1
        
        return count
        
    async def get_system_messages(self) -> List[Dict[str, Any]]:
        """
        Get system messages from the queue.
        
        Returns:
            List of system messages
        """
        messages = []
        
        try:
            while not self.system_queue.empty():
                message_data = self.system_queue.get_nowait()
                messages.append(message_data)
                
            self.stats["messages_received"] += len(messages)
            
        except Exception as e:
            self.logger.error(f"Error getting system messages: {e}")
            
        return messages
        
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a message handler.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self.handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")
        
    async def _send_to_agent(self, agent_name: str, message: Message):
        """Send message to specific agent."""
        if agent_name not in self.queues:
            raise ValueError(f"Agent {agent_name} not registered")
            
        try:
            # Serialize message
            message_data = msgpack.packb({
                "id": message.id,
                "type": message.type,
                "sender": message.sender,
                "target": message.target,
                "payload": message.payload,
                "timestamp": message.timestamp,
                "reply_to": message.reply_to,
                "correlation_id": message.correlation_id,
                "priority": message.priority
            })
            
            # Send to agent queue
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.queues[agent_name].put,
                message_data
            )
            
            self.logger.debug(f"Sent message {message.id} to {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {agent_name}: {e}")
            raise
            
    async def _send_broadcast(self, message: Message) -> int:
        """Send message to all registered agents."""
        count = 0
        
        # Send to all agent queues
        for agent_name in self.queues:
            try:
                await self._send_to_agent(agent_name, message)
                count += 1
            except Exception as e:
                self.logger.error(f"Failed to broadcast to {agent_name}: {e}")
                
        # Also send to broadcast queue
        try:
            message_data = msgpack.packb({
                "id": message.id,
                "type": message.type,
                "sender": message.sender,
                "target": message.target,
                "payload": message.payload,
                "timestamp": message.timestamp,
                "reply_to": message.reply_to,
                "correlation_id": message.correlation_id,
                "priority": message.priority
            })
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.broadcast_queue.put,
                message_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add to broadcast queue: {e}")
            
        return count
        
    async def _setup_unix_socket(self):
        """Setup Unix domain socket for external communication."""
        try:
            # Remove existing socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
                
            # Create socket directory
            socket_dir = Path(self.socket_path).parent
            socket_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Unix socket
            self.unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.unix_socket.bind(self.socket_path)
            self.unix_socket.listen(5)
            
            # Set permissions
            os.chmod(self.socket_path, 0o600)
            
            self.logger.info(f"Unix socket created at {self.socket_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Unix socket: {e}")
            raise
            
    def _start_background_tasks(self):
        """Start background tasks for message processing."""
        self.is_running = True
        
        # Start socket listener in thread
        threading.Thread(
            target=self._socket_listener_thread,
            daemon=True,
            name="socket-listener"
        ).start()
        
        # Start message processor in thread
        threading.Thread(
            target=self._message_processor_thread,
            daemon=True,
            name="message-processor"
        ).start()
        
    def _socket_listener_thread(self):
        """Thread to listen for external connections."""
        while self.is_running:
            try:
                # Accept connection
                client_socket, address = self.unix_socket.accept()
                
                # Handle client in separate thread
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                ).start()
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error in socket listener: {e}")
                    
    def _handle_client(self, client_socket):
        """Handle external client connection."""
        try:
            with client_socket:
                while True:
                    # Receive message
                    data = client_socket.recv(4096)
                    if not data:
                        break
                        
                    try:
                        # Deserialize message
                        message_data = msgpack.unpackb(data, raw=False)
                        
                        # Process message
                        response = self._process_external_message(message_data)
                        
                        # Send response
                        response_data = msgpack.packb(response)
                        client_socket.send(response_data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing external message: {e}")
                        
                        # Send error response
                        error_response = {
                            "success": False,
                            "error": str(e),
                            "timestamp": time.time()
                        }
                        client_socket.send(msgpack.packb(error_response))
                        
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
            
    def _process_external_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process message from external client."""
        try:
            message_type = message_data.get("type", "unknown")
            
            # Handle different message types
            if message_type == "status":
                return {
                    "success": True,
                    "response": {
                        "status": "running",
                        "stats": self.stats,
                        "agents": list(self.queues.keys())
                    }
                }
            elif message_type == "send":
                # Forward message to agent
                target = message_data.get("target", "broadcast")
                payload = message_data.get("payload", {})
                
                # Create internal message
                message = Message(
                    type=message_data.get("message_type", "external"),
                    sender="external",
                    target=target,
                    payload=payload
                )
                
                # Send message (this is synchronous for external API)
                if target == "broadcast":
                    asyncio.run(self._send_broadcast(message))
                else:
                    asyncio.run(self._send_to_agent(target, message))
                    
                return {
                    "success": True,
                    "message_id": message.id
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _message_processor_thread(self):
        """Thread to process system messages."""
        while self.is_running:
            try:
                # Process responses
                self._process_responses()
                
                # Process system messages
                self._process_system_messages()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")
                
    def _process_responses(self):
        """Process pending responses."""
        # This would be implemented to handle response correlation
        # For now, it's a placeholder
        pass
        
    def _process_system_messages(self):
        """Process system messages."""
        # This would be implemented to handle system-level message processing
        # For now, it's a placeholder
        pass
        
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            **self.stats,
            "registered_agents": len(self.queues),
            "pending_responses": len(self.pending_responses),
            "is_running": self.is_running
        }