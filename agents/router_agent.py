"""
Project AWARENESS - Router Agent
The central nervous system that manages task state machines and orchestrates other agents.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from agents.base_agent import BaseAgent, AgentMessage
from core.config import AwarenessConfig


class TaskState(Enum):
    """States for task processing."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task in the system."""
    id: str
    type: str
    description: str
    state: TaskState = TaskState.PENDING
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    assigned_agents: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    

@dataclass
class TaskPlan:
    """Represents a plan for executing a task."""
    task_id: str
    steps: List[Dict[str, Any]]
    estimated_time: float
    required_agents: List[str]
    confidence: float = 0.5
    

class RouterAgent(BaseAgent):
    """
    The central orchestrator that manages task state machines and coordinates other agents.
    Implements intelligent task decomposition, planning, and execution.
    """
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: Dict[str, Any]):
        super().__init__(name, config, agent_config)
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_plans: Dict[str, TaskPlan] = {}
        self.task_queue: List[str] = []
        
        # Agent registry
        self.available_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_workload: Dict[str, int] = {}
        
        # Task processing
        self.max_concurrent_tasks = 5
        self.current_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_completion_time": 0.0
        }
        
        # Register message handlers
        self._register_router_handlers()
        
    async def _initialize(self):
        """Initialize the router agent."""
        self.logger.info("RouterAgent initializing...")
        
        # Initialize agent capabilities
        self._initialize_agent_capabilities()
        
        # Start task processing
        self.logger.info("RouterAgent initialized successfully")
        
    async def _shutdown(self):
        """Shutdown the router agent."""
        self.logger.info("RouterAgent shutting down...")
        
        # Cancel all running tasks
        for task in self.current_tasks.values():
            task.cancel()
            
        # Wait for tasks to complete
        if self.current_tasks:
            await asyncio.gather(*self.current_tasks.values(), return_exceptions=True)
            
        self.logger.info("RouterAgent shutdown complete")
        
    def _register_router_handlers(self):
        """Register router-specific message handlers."""
        self.register_handler("user_command", self._handle_user_command)
        self.register_handler("task_request", self._handle_task_request)
        self.register_handler("task_update", self._handle_task_update)
        self.register_handler("task_complete", self._handle_task_complete)
        self.register_handler("agent_register", self._handle_agent_register)
        self.register_handler("agent_update", self._handle_agent_update)
        
    def _get_background_tasks(self) -> List:
        """Get router-specific background tasks."""
        return [
            self._task_processor_loop(),
            self._agent_monitor_loop(),
            self._task_timeout_monitor(),
        ]
        
    async def _handle_user_command(self, message: AgentMessage):
        """Handle user commands."""
        try:
            command = message.payload.get("command", "")
            request_id = message.payload.get("request_id", str(uuid.uuid4()))
            
            self.logger.info(f"Processing user command: {command}")
            
            # Create task for user command
            task = Task(
                id=request_id,
                type="user_command",
                description=command,
                priority=8,
                metadata={"source": "user", "original_message": message.payload}
            )
            
            # Add to task queue
            await self._add_task(task)
            
            # Send acknowledgment
            await self._send_response(message, {
                "status": "accepted",
                "task_id": task.id,
                "message": "Command accepted for processing"
            })
            
        except Exception as e:
            self.logger.error(f"Error handling user command: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_task_request(self, message: AgentMessage):
        """Handle task requests from other agents."""
        try:
            task_data = message.payload
            
            task = Task(
                id=task_data.get("id", str(uuid.uuid4())),
                type=task_data.get("type", "unknown"),
                description=task_data.get("description", ""),
                priority=task_data.get("priority", 5),
                metadata=task_data.get("metadata", {})
            )
            
            await self._add_task(task)
            
            await self._send_response(message, {
                "status": "accepted",
                "task_id": task.id
            })
            
        except Exception as e:
            self.logger.error(f"Error handling task request: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_task_update(self, message: AgentMessage):
        """Handle task updates from agents."""
        try:
            task_id = message.payload.get("task_id")
            update = message.payload.get("update", {})
            
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Update task based on update type
                if "state" in update:
                    task.state = TaskState(update["state"])
                    
                if "progress" in update:
                    task.metadata["progress"] = update["progress"]
                    
                if "result" in update:
                    task.result = update["result"]
                    
                if "error" in update:
                    task.error = update["error"]
                    
                task.updated_at = time.time()
                
                self.logger.debug(f"Updated task {task_id}: {update}")
                
        except Exception as e:
            self.logger.error(f"Error handling task update: {e}")
            
    async def _handle_task_complete(self, message: AgentMessage):
        """Handle task completion from agents."""
        try:
            task_id = message.payload.get("task_id")
            result = message.payload.get("result")
            error = message.payload.get("error")
            
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                if error:
                    task.state = TaskState.FAILED
                    task.error = error
                    self.stats["failed_tasks"] += 1
                else:
                    task.state = TaskState.COMPLETED
                    task.result = result
                    self.stats["completed_tasks"] += 1
                    
                task.updated_at = time.time()
                
                # Update statistics
                completion_time = task.updated_at - task.created_at
                self._update_completion_stats(completion_time)
                
                # Clean up
                if task_id in self.current_tasks:
                    del self.current_tasks[task_id]
                    
                self.logger.info(f"Task {task_id} completed: {task.state.value}")
                
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
            
    async def _handle_agent_register(self, message: AgentMessage):
        """Handle agent registration."""
        try:
            agent_name = message.payload.get("agent_name")
            capabilities = message.payload.get("capabilities", [])
            
            if agent_name:
                self.available_agents[agent_name] = {
                    "name": agent_name,
                    "capabilities": capabilities,
                    "registered_at": time.time(),
                    "last_seen": time.time()
                }
                
                self.agent_capabilities[agent_name] = capabilities
                self.agent_workload[agent_name] = 0
                
                self.logger.info(f"Registered agent: {agent_name} with capabilities: {capabilities}")
                
        except Exception as e:
            self.logger.error(f"Error registering agent: {e}")
            
    async def _handle_agent_update(self, message: AgentMessage):
        """Handle agent status updates."""
        try:
            agent_name = message.payload.get("agent_name")
            
            if agent_name in self.available_agents:
                self.available_agents[agent_name]["last_seen"] = time.time()
                
                # Update workload if provided
                if "workload" in message.payload:
                    self.agent_workload[agent_name] = message.payload["workload"]
                    
        except Exception as e:
            self.logger.error(f"Error handling agent update: {e}")
            
    async def _add_task(self, task: Task):
        """Add a task to the queue."""
        self.tasks[task.id] = task
        self.task_queue.append(task.id)
        self.stats["total_tasks"] += 1
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)
        
        self.logger.info(f"Added task {task.id} to queue")
        
    async def _task_processor_loop(self):
        """Main task processing loop."""
        while self.is_running:
            try:
                # Process pending tasks
                await self._process_pending_tasks()
                
                # Monitor running tasks
                await self._monitor_running_tasks()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in task processor loop: {e}")
                await asyncio.sleep(5)
                
    async def _process_pending_tasks(self):
        """Process tasks in the queue."""
        while (len(self.current_tasks) < self.max_concurrent_tasks and 
               self.task_queue):
            
            # Get next task
            task_id = self.task_queue.pop(0)
            task = self.tasks.get(task_id)
            
            if not task or task.state != TaskState.PENDING:
                continue
                
            # Start processing task
            processing_task = asyncio.create_task(self._process_task(task))
            self.current_tasks[task_id] = processing_task
            
    async def _process_task(self, task: Task):
        """Process a single task."""
        try:
            self.logger.info(f"Processing task {task.id}: {task.description}")
            
            # Update task state
            task.state = TaskState.ANALYZING
            task.updated_at = time.time()
            
            # Analyze task
            analysis = await self._analyze_task(task)
            
            # Create plan
            task.state = TaskState.PLANNING
            plan = await self._create_task_plan(task, analysis)
            
            if plan:
                self.task_plans[task.id] = plan
                
                # Execute plan
                task.state = TaskState.EXECUTING
                await self._execute_task_plan(task, plan)
            else:
                task.state = TaskState.FAILED
                task.error = "Failed to create execution plan"
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {e}")
            task.state = TaskState.FAILED
            task.error = str(e)
            
    async def _analyze_task(self, task: Task) -> Dict[str, Any]:
        """Analyze a task to understand requirements."""
        analysis = {
            "task_type": task.type,
            "complexity": self._estimate_complexity(task),
            "required_capabilities": self._identify_required_capabilities(task),
            "estimated_time": self._estimate_execution_time(task),
            "dependencies": task.dependencies
        }
        
        return analysis
        
    async def _create_task_plan(self, task: Task, analysis: Dict[str, Any]) -> Optional[TaskPlan]:
        """Create an execution plan for a task."""
        try:
            # Determine required agents
            required_agents = self._select_agents_for_task(analysis["required_capabilities"])
            
            if not required_agents:
                self.logger.warning(f"No suitable agents found for task {task.id}")
                return None
                
            # Create execution steps
            steps = self._create_execution_steps(task, analysis, required_agents)
            
            plan = TaskPlan(
                task_id=task.id,
                steps=steps,
                estimated_time=analysis["estimated_time"],
                required_agents=required_agents,
                confidence=self._calculate_plan_confidence(analysis, required_agents)
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating task plan: {e}")
            return None
            
    async def _execute_task_plan(self, task: Task, plan: TaskPlan):
        """Execute a task plan."""
        try:
            results = []
            
            for step in plan.steps:
                step_result = await self._execute_step(task, step)
                results.append(step_result)
                
                # Check if step failed
                if step_result.get("status") == "failed":
                    task.state = TaskState.FAILED
                    task.error = step_result.get("error", "Step execution failed")
                    return
                    
            # Task completed successfully
            task.state = TaskState.COMPLETED
            task.result = {
                "steps": results,
                "summary": self._summarize_results(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error executing task plan: {e}")
            task.state = TaskState.FAILED
            task.error = str(e)
            
    async def _execute_step(self, task: Task, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in a task plan."""
        try:
            agent_name = step.get("agent")
            action = step.get("action")
            params = step.get("params", {})
            
            if agent_name not in self.available_agents:
                return {
                    "status": "failed",
                    "error": f"Agent {agent_name} not available"
                }
                
            # Send step to agent
            response = await self.send_and_wait(
                target=agent_name,
                message_type="execute_step",
                payload={
                    "task_id": task.id,
                    "step_id": step.get("id"),
                    "action": action,
                    "params": params
                },
                timeout=step.get("timeout", 60)
            )
            
            return response
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
            
    def _estimate_complexity(self, task: Task) -> str:
        """Estimate task complexity."""
        description_length = len(task.description)
        
        if description_length < 50:
            return "simple"
        elif description_length < 200:
            return "moderate"
        else:
            return "complex"
            
    def _identify_required_capabilities(self, task: Task) -> List[str]:
        """Identify capabilities required for a task."""
        capabilities = []
        
        # Simple keyword-based capability identification
        description = task.description.lower()
        
        if any(word in description for word in ["memory", "remember", "store", "recall"]):
            capabilities.append("memory")
            
        if any(word in description for word in ["learn", "analyze", "understand"]):
            capabilities.append("learning")
            
        if any(word in description for word in ["secure", "protect", "encrypt"]):
            capabilities.append("security")
            
        if any(word in description for word in ["context", "conversation"]):
            capabilities.append("context")
            
        return capabilities
        
    def _estimate_execution_time(self, task: Task) -> float:
        """Estimate task execution time in seconds."""
        # Simple heuristic based on task type and complexity
        base_time = 10.0  # 10 seconds base
        
        if task.type == "user_command":
            base_time = 5.0
        elif task.type == "analysis":
            base_time = 30.0
        elif task.type == "learning":
            base_time = 60.0
            
        # Adjust based on priority
        priority_factor = 1.0 + (10 - task.priority) * 0.1
        
        return base_time * priority_factor
        
    def _select_agents_for_task(self, required_capabilities: List[str]) -> List[str]:
        """Select the best agents for a task."""
        selected_agents = []
        
        for capability in required_capabilities:
            best_agent = None
            best_score = 0
            
            for agent_name, capabilities in self.agent_capabilities.items():
                if capability in capabilities:
                    # Calculate score based on workload and availability
                    workload = self.agent_workload.get(agent_name, 0)
                    score = 100 - workload  # Lower workload = higher score
                    
                    if score > best_score:
                        best_score = score
                        best_agent = agent_name
                        
            if best_agent and best_agent not in selected_agents:
                selected_agents.append(best_agent)
                
        return selected_agents
        
    def _create_execution_steps(self, task: Task, analysis: Dict[str, Any], 
                               agents: List[str]) -> List[Dict[str, Any]]:
        """Create execution steps for a task."""
        steps = []
        
        # Simple step creation based on task type
        if task.type == "user_command":
            steps.append({
                "id": "parse_command",
                "agent": "context",
                "action": "parse_user_input",
                "params": {"input": task.description}
            })
            
            steps.append({
                "id": "execute_command",
                "agent": agents[0] if agents else "memory_controller",
                "action": "execute",
                "params": {"command": task.description}
            })
            
        else:
            # Generic steps
            for i, agent in enumerate(agents):
                steps.append({
                    "id": f"step_{i}",
                    "agent": agent,
                    "action": "process",
                    "params": {"task_data": task.metadata}
                })
                
        return steps
        
    def _calculate_plan_confidence(self, analysis: Dict[str, Any], agents: List[str]) -> float:
        """Calculate confidence in execution plan."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have suitable agents
        if agents:
            confidence += 0.3
            
        # Adjust based on complexity
        if analysis.get("complexity") == "simple":
            confidence += 0.2
        elif analysis.get("complexity") == "complex":
            confidence -= 0.1
            
        return min(1.0, max(0.1, confidence))
        
    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize execution results."""
        successful_steps = len([r for r in results if r.get("status") == "success"])
        total_steps = len(results)
        
        return f"Completed {successful_steps}/{total_steps} steps successfully"
        
    async def _monitor_running_tasks(self):
        """Monitor running tasks for completion."""
        completed_tasks = []
        
        for task_id, task_future in self.current_tasks.items():
            if task_future.done():
                completed_tasks.append(task_id)
                
        # Clean up completed tasks
        for task_id in completed_tasks:
            del self.current_tasks[task_id]
            
    async def _agent_monitor_loop(self):
        """Monitor agent availability."""
        while self.is_running:
            try:
                current_time = time.time()
                inactive_agents = []
                
                for agent_name, agent_info in self.available_agents.items():
                    # Check if agent is inactive (no heartbeat for 5 minutes)
                    if current_time - agent_info["last_seen"] > 300:
                        inactive_agents.append(agent_name)
                        
                # Remove inactive agents
                for agent_name in inactive_agents:
                    del self.available_agents[agent_name]
                    if agent_name in self.agent_capabilities:
                        del self.agent_capabilities[agent_name]
                    if agent_name in self.agent_workload:
                        del self.agent_workload[agent_name]
                        
                    self.logger.warning(f"Removed inactive agent: {agent_name}")
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in agent monitor loop: {e}")
                await asyncio.sleep(300)
                
    async def _task_timeout_monitor(self):
        """Monitor tasks for timeouts."""
        while self.is_running:
            try:
                current_time = time.time()
                
                for task_id, task in self.tasks.items():
                    if (task.state in [TaskState.ANALYZING, TaskState.PLANNING, TaskState.EXECUTING] and
                        task.deadline and current_time > task.deadline):
                        
                        # Task timed out
                        task.state = TaskState.FAILED
                        task.error = "Task timed out"
                        task.updated_at = current_time
                        
                        # Cancel running task
                        if task_id in self.current_tasks:
                            self.current_tasks[task_id].cancel()
                            del self.current_tasks[task_id]
                            
                        self.logger.warning(f"Task {task_id} timed out")
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in task timeout monitor: {e}")
                await asyncio.sleep(60)
                
    def _initialize_agent_capabilities(self):
        """Initialize known agent capabilities."""
        self.agent_capabilities = {
            "memory_controller": ["memory", "storage", "retrieval"],
            "context": ["context", "conversation", "parsing"],
            "security": ["security", "encryption", "authentication"],
            "learning": ["learning", "analysis", "adaptation"],
            "watchdog": ["monitoring", "security", "alerting"]
        }
        
    def _update_completion_stats(self, completion_time: float):
        """Update completion statistics."""
        current_avg = self.stats["average_completion_time"]
        completed_count = self.stats["completed_tasks"]
        
        if completed_count == 1:
            self.stats["average_completion_time"] = completion_time
        else:
            # Running average
            self.stats["average_completion_time"] = (
                (current_avg * (completed_count - 1) + completion_time) / completed_count
            )
            
    async def _send_response(self, original_message: AgentMessage, response: Dict[str, Any]):
        """Send response to a message."""
        # Implementation depends on message bus integration
        pass
        
    async def _send_error_response(self, original_message: AgentMessage, error: str):
        """Send error response to a message."""
        await self._send_response(original_message, {
            "status": "error",
            "error": error
        })