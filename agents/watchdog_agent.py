"""
Project AWARENESS - Watchdog Agent
Provides runtime monitoring, alerting, and system health oversight.
"""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from agents.base_agent import BaseAgent, AgentMessage
from core.config import AwarenessConfig


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    timestamp: float
    level: AlertLevel
    title: str
    description: str
    source: str
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: str  # healthy, degraded, unhealthy
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class WatchdogAgent(BaseAgent):
    """
    Watchdog Agent providing runtime monitoring and alerting.
    Monitors system health, agent performance, and security events.
    """
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: Dict[str, Any]):
        super().__init__(name, config, agent_config)
        
        # Monitoring state
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        self.monitored_agents: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.response_time_threshold = 5.0
        self.error_rate_threshold = 0.1
        
        # Configuration
        self.max_alerts = 1000
        self.health_check_interval = 30
        self.alert_cooldown = 300  # 5 minutes
        
        # Last alert times (for cooldown)
        self.last_alert_times: Dict[str, float] = {}
        
        # Register handlers
        self._register_watchdog_handlers()
        
    async def _initialize(self):
        """Initialize the watchdog agent."""
        self.logger.info("WatchdogAgent initializing...")
        
        # Initialize monitoring components
        await self._initialize_monitoring()
        
        self.logger.info("WatchdogAgent initialized successfully")
        
    async def _shutdown(self):
        """Shutdown the watchdog agent."""
        self.logger.info("WatchdogAgent shutting down...")
        
        # Save alerts and state
        await self._save_monitoring_state()
        
        self.logger.info("WatchdogAgent shutdown complete")
        
    def _register_watchdog_handlers(self):
        """Register watchdog-specific message handlers."""
        self.register_handler("get_alerts", self._handle_get_alerts)
        self.register_handler("get_health", self._handle_get_health)
        self.register_handler("register_agent", self._handle_register_agent)
        self.register_handler("agent_heartbeat", self._handle_agent_heartbeat)
        self.register_handler("resolve_alert", self._handle_resolve_alert)
        
    def _get_background_tasks(self) -> List:
        """Get watchdog-specific background tasks."""
        return [
            self._system_monitoring_loop(),
            self._agent_monitoring_loop(),
            self._health_check_loop(),
            self._alert_cleanup_loop(),
        ]
        
    async def _handle_get_alerts(self, message: AgentMessage):
        """Handle get alerts requests."""
        try:
            level = message.payload.get("level")
            unresolved_only = message.payload.get("unresolved_only", True)
            limit = message.payload.get("limit", 50)
            
            # Get alerts
            alerts = await self.get_alerts(level, unresolved_only, limit)
            
            await self._send_response(message, {
                "status": "success",
                "alerts": alerts,
                "count": len(alerts)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling get alerts: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_get_health(self, message: AgentMessage):
        """Handle get health status requests."""
        try:
            # Get health status
            health_status = await self.get_health_status()
            
            await self._send_response(message, {
                "status": "success",
                "health": health_status
            })
            
        except Exception as e:
            self.logger.error(f"Error handling get health: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_register_agent(self, message: AgentMessage):
        """Handle agent registration for monitoring."""
        try:
            agent_name = message.payload.get("agent_name")
            agent_info = message.payload.get("agent_info", {})
            
            if not agent_name:
                await self._send_error_response(message, "Missing agent_name")
                return
                
            # Register agent for monitoring
            await self.register_agent_for_monitoring(agent_name, agent_info)
            
            await self._send_response(message, {
                "status": "success",
                "agent_name": agent_name,
                "message": "Agent registered for monitoring"
            })
            
        except Exception as e:
            self.logger.error(f"Error handling register agent: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_agent_heartbeat(self, message: AgentMessage):
        """Handle agent heartbeat messages."""
        try:
            agent_name = message.payload.get("agent_name")
            metrics = message.payload.get("metrics", {})
            
            if not agent_name:
                await self._send_error_response(message, "Missing agent_name")
                return
                
            # Process heartbeat
            await self.process_agent_heartbeat(agent_name, metrics)
            
            await self._send_response(message, {
                "status": "success",
                "agent_name": agent_name,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error handling agent heartbeat: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_resolve_alert(self, message: AgentMessage):
        """Handle alert resolution requests."""
        try:
            alert_id = message.payload.get("alert_id")
            
            if not alert_id:
                await self._send_error_response(message, "Missing alert_id")
                return
                
            # Resolve alert
            success = await self.resolve_alert(alert_id)
            
            await self._send_response(message, {
                "status": "success" if success else "not_found",
                "alert_id": alert_id,
                "resolved": success
            })
            
        except Exception as e:
            self.logger.error(f"Error handling resolve alert: {e}")
            await self._send_error_response(message, str(e))
            
    async def create_alert(self, level: AlertLevel, title: str, description: str, 
                          source: str = "system", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new alert.
        
        Args:
            level: Alert severity level
            title: Alert title
            description: Alert description
            source: Source of the alert
            metadata: Additional metadata
            
        Returns:
            Alert ID
        """
        import hashlib
        
        # Check cooldown
        alert_key = f"{source}:{title}"
        current_time = time.time()
        
        if alert_key in self.last_alert_times:
            if current_time - self.last_alert_times[alert_key] < self.alert_cooldown:
                return ""  # Skip due to cooldown
                
        # Generate alert ID
        alert_id = hashlib.sha256(f"{title}{description}{current_time}".encode()).hexdigest()[:16]
        
        alert = Alert(
            id=alert_id,
            timestamp=current_time,
            level=level,
            title=title,
            description=description,
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        self.last_alert_times[alert_key] = current_time
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
            
        # Log alert
        log_method = getattr(self.logger, level.value.lower(), self.logger.info)
        log_method(f"Alert created: {title} - {description}")
        
        return alert_id
        
    async def get_alerts(self, level: Optional[str] = None, unresolved_only: bool = True,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get system alerts.
        
        Args:
            level: Filter by alert level
            unresolved_only: Only return unresolved alerts
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        alerts = self.alerts
        
        # Filter by level
        if level:
            try:
                alert_level = AlertLevel(level)
                alerts = [a for a in alerts if a.level == alert_level]
            except ValueError:
                pass
                
        # Filter by resolution status
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]
            
        # Sort by timestamp (newest first) and limit
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        # Convert to dict format
        return [
            {
                "id": alert.id,
                "timestamp": alert.timestamp,
                "level": alert.level.value,
                "title": alert.title,
                "description": alert.description,
                "source": alert.source,
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            for alert in alerts
        ]
        
    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if alert was found and resolved
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self.logger.info(f"Alert resolved: {alert.title}")
                return True
        return False
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine overall status
        overall_status = "healthy"
        
        if (cpu_percent > self.cpu_threshold or 
            memory.percent > self.memory_threshold or
            disk.percent > 90):
            overall_status = "degraded"
            
        # Count unresolved alerts
        critical_alerts = len([a for a in self.alerts 
                              if not a.resolved and a.level == AlertLevel.CRITICAL])
        
        if critical_alerts > 0:
            overall_status = "unhealthy"
            
        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "uptime": time.time() - self.start_time
            },
            "alerts": {
                "total": len(self.alerts),
                "unresolved": len([a for a in self.alerts if not a.resolved]),
                "critical": critical_alerts
            },
            "agents": {
                "monitored": len(self.monitored_agents),
                "healthy": len([a for a in self.monitored_agents.values() 
                               if a.get("status") == "healthy"])
            },
            "health_checks": list(self.health_checks.keys())
        }
        
    async def register_agent_for_monitoring(self, agent_name: str, agent_info: Dict[str, Any]):
        """Register an agent for monitoring."""
        self.monitored_agents[agent_name] = {
            "name": agent_name,
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "status": "healthy",
            "metrics": {},
            "info": agent_info
        }
        
        self.logger.info(f"Registered agent for monitoring: {agent_name}")
        
    async def process_agent_heartbeat(self, agent_name: str, metrics: Dict[str, Any]):
        """Process heartbeat from an agent."""
        if agent_name not in self.monitored_agents:
            await self.register_agent_for_monitoring(agent_name, {})
            
        agent_info = self.monitored_agents[agent_name]
        agent_info["last_heartbeat"] = time.time()
        agent_info["metrics"] = metrics
        
        # Check agent health based on metrics
        await self._check_agent_health(agent_name, metrics)
        
    async def _system_monitoring_loop(self):
        """Background system monitoring loop."""
        while self.is_running:
            try:
                # Monitor system resources
                await self._monitor_system_resources()
                
                # Monitor disk space
                await self._monitor_disk_space()
                
                # Monitor process count
                await self._monitor_process_count()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(60)
                
    async def _agent_monitoring_loop(self):
        """Background agent monitoring loop."""
        while self.is_running:
            try:
                # Check agent heartbeats
                await self._check_agent_heartbeats()
                
                # Monitor agent performance
                await self._monitor_agent_performance()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in agent monitoring loop: {e}")
                await asyncio.sleep(120)
                
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.is_running:
            try:
                # Perform health checks
                await self._perform_health_checks()
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval * 2)
                
    async def _alert_cleanup_loop(self):
        """Background alert cleanup loop."""
        while self.is_running:
            try:
                # Clean up old resolved alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error in alert cleanup loop: {e}")
                await asyncio.sleep(7200)
                
    async def _monitor_system_resources(self):
        """Monitor system resource usage."""
        try:
            # CPU monitoring
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.cpu_threshold:
                await self.create_alert(
                    AlertLevel.WARNING,
                    "High CPU Usage",
                    f"CPU usage is {cpu_percent:.1f}% (threshold: {self.cpu_threshold}%)",
                    "system",
                    {"cpu_percent": cpu_percent}
                )
                
            # Memory monitoring
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                await self.create_alert(
                    AlertLevel.WARNING,
                    "High Memory Usage",
                    f"Memory usage is {memory.percent:.1f}% (threshold: {self.memory_threshold}%)",
                    "system",
                    {"memory_percent": memory.percent, "memory_available": memory.available}
                )
                
        except Exception as e:
            self.logger.error(f"Error monitoring system resources: {e}")
            
    async def _monitor_disk_space(self):
        """Monitor disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                await self.create_alert(
                    AlertLevel.ERROR,
                    "Low Disk Space",
                    f"Disk usage is {disk.percent:.1f}% (less than 10% free)",
                    "system",
                    {"disk_percent": disk.percent, "disk_free": disk.free}
                )
                
        except Exception as e:
            self.logger.error(f"Error monitoring disk space: {e}")
            
    async def _monitor_process_count(self):
        """Monitor number of running processes."""
        try:
            process_count = len(psutil.pids())
            if process_count > 1000:  # Arbitrary threshold
                await self.create_alert(
                    AlertLevel.INFO,
                    "High Process Count",
                    f"System has {process_count} running processes",
                    "system",
                    {"process_count": process_count}
                )
                
        except Exception as e:
            self.logger.error(f"Error monitoring process count: {e}")
            
    async def _check_agent_heartbeats(self):
        """Check agent heartbeats for timeouts."""
        current_time = time.time()
        timeout_threshold = 300  # 5 minutes
        
        for agent_name, agent_info in self.monitored_agents.items():
            last_heartbeat = agent_info.get("last_heartbeat", 0)
            
            if current_time - last_heartbeat > timeout_threshold:
                await self.create_alert(
                    AlertLevel.ERROR,
                    "Agent Heartbeat Timeout",
                    f"Agent {agent_name} has not sent heartbeat for {(current_time - last_heartbeat):.0f} seconds",
                    "watchdog",
                    {"agent_name": agent_name, "last_heartbeat": last_heartbeat}
                )
                
                agent_info["status"] = "unhealthy"
                
    async def _monitor_agent_performance(self):
        """Monitor agent performance metrics."""
        for agent_name, agent_info in self.monitored_agents.items():
            metrics = agent_info.get("metrics", {})
            
            # Check error rate
            error_rate = metrics.get("error_rate", 0)
            if error_rate > self.error_rate_threshold:
                await self.create_alert(
                    AlertLevel.WARNING,
                    "High Agent Error Rate",
                    f"Agent {agent_name} has error rate of {error_rate:.2%}",
                    "watchdog",
                    {"agent_name": agent_name, "error_rate": error_rate}
                )
                
    async def _check_agent_health(self, agent_name: str, metrics: Dict[str, Any]):
        """Check health of a specific agent."""
        # Simple health check based on metrics
        error_count = metrics.get("error_count", 0)
        message_count = metrics.get("message_count", 1)
        
        error_rate = error_count / max(1, message_count)
        
        if error_rate > self.error_rate_threshold:
            self.monitored_agents[agent_name]["status"] = "degraded"
        else:
            self.monitored_agents[agent_name]["status"] = "healthy"
            
    async def _perform_health_checks(self):
        """Perform various health checks."""
        checks = {
            "system_resources": self._health_check_system_resources,
            "agent_connectivity": self._health_check_agent_connectivity,
            "disk_space": self._health_check_disk_space,
        }
        
        for check_name, check_func in checks.items():
            try:
                result = await check_func()
                self.health_checks[check_name] = HealthCheck(
                    name=check_name,
                    status=result["status"],
                    timestamp=time.time(),
                    details=result.get("details", {})
                )
            except Exception as e:
                self.logger.error(f"Error in health check {check_name}: {e}")
                
    async def _health_check_system_resources(self) -> Dict[str, Any]:
        """Health check for system resources."""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 90 or memory_percent > 95:
            status = "unhealthy"
        elif cpu_percent > 70 or memory_percent > 80:
            status = "degraded"
        else:
            status = "healthy"
            
        return {
            "status": status,
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent
            }
        }
        
    async def _health_check_agent_connectivity(self) -> Dict[str, Any]:
        """Health check for agent connectivity."""
        healthy_agents = len([a for a in self.monitored_agents.values() 
                             if a.get("status") == "healthy"])
        total_agents = len(self.monitored_agents)
        
        if total_agents == 0:
            status = "healthy"  # No agents to monitor
        elif healthy_agents / total_agents < 0.5:
            status = "unhealthy"
        elif healthy_agents / total_agents < 0.8:
            status = "degraded"
        else:
            status = "healthy"
            
        return {
            "status": status,
            "details": {
                "healthy_agents": healthy_agents,
                "total_agents": total_agents
            }
        }
        
    async def _health_check_disk_space(self) -> Dict[str, Any]:
        """Health check for disk space."""
        disk = psutil.disk_usage('/')
        
        if disk.percent > 95:
            status = "unhealthy"
        elif disk.percent > 85:
            status = "degraded"
        else:
            status = "healthy"
            
        return {
            "status": status,
            "details": {
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
        }
        
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        # Remove resolved alerts older than 24 hours
        cutoff_time = time.time() - 86400  # 24 hours
        
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or alert.timestamp > cutoff_time
        ]
        
    async def _initialize_monitoring(self):
        """Initialize monitoring components."""
        # Any initialization specific to monitoring
        pass
        
    async def _save_monitoring_state(self):
        """Save monitoring state."""
        # Implementation for saving monitoring state
        pass
        
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