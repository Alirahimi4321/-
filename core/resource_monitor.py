"""
Project AWARENESS - Resource Monitor
System resource monitoring with dynamic throttling and alerting.
"""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics

from core.config import AwarenessConfig
from core.logger import setup_logger


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    load_average: List[float] = field(default_factory=list)
    temperature: Optional[float] = None
    

@dataclass
class ResourceThreshold:
    """Resource threshold configuration."""
    name: str
    metric: str
    warning_threshold: float
    critical_threshold: float
    callback: Optional[Callable] = None
    

class ResourceMonitor:
    """
    System resource monitor with dynamic throttling capabilities.
    Tracks CPU, memory, disk, network, and process metrics.
    """
    
    def __init__(self, config: AwarenessConfig):
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Resource history (last 100 snapshots)
        self.history = deque(maxlen=100)
        
        # Current resource state
        self.current_snapshot: Optional[ResourceSnapshot] = None
        self.last_update = 0
        
        # Thresholds
        self.thresholds: List[ResourceThreshold] = []
        self._setup_default_thresholds()
        
        # Alerts
        self.active_alerts: Dict[str, float] = {}
        self.alert_cooldown = 60  # seconds
        
        # Process tracking
        self.process_whitelist: List[str] = ["awareness", "python"]
        self.monitored_processes: Dict[int, psutil.Process] = {}
        
        # Network baseline
        self.network_baseline: Optional[Dict[str, int]] = None
        
        # State
        self.is_running = False
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the resource monitor."""
        if self.is_initialized:
            return
            
        self.logger.info("Initializing resource monitor...")
        
        try:
            # Get initial network baseline
            self._update_network_baseline()
            
            # Take initial snapshot
            await self.update_stats()
            
            self.is_initialized = True
            self.logger.info("Resource monitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize resource monitor: {e}")
            raise
            
    async def shutdown(self):
        """Shutdown the resource monitor."""
        self.logger.info("Shutting down resource monitor...")
        self.is_running = False
        
    async def update_stats(self):
        """Update resource statistics."""
        try:
            # Create new snapshot
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / (1024 * 1024),
                memory_available_mb=psutil.virtual_memory().available / (1024 * 1024),
                disk_percent=psutil.disk_usage('/').percent,
                disk_used_gb=psutil.disk_usage('/').used / (1024 * 1024 * 1024),
                disk_free_gb=psutil.disk_usage('/').free / (1024 * 1024 * 1024),
                network_sent_mb=0,
                network_recv_mb=0,
                process_count=len(psutil.pids())
            )
            
            # Update network stats
            network_stats = psutil.net_io_counters()
            if self.network_baseline:
                snapshot.network_sent_mb = (
                    network_stats.bytes_sent - self.network_baseline['bytes_sent']
                ) / (1024 * 1024)
                snapshot.network_recv_mb = (
                    network_stats.bytes_recv - self.network_baseline['bytes_recv']
                ) / (1024 * 1024)
            
            # Update load average (Linux/Unix only)
            try:
                snapshot.load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                snapshot.load_average = [0.0, 0.0, 0.0]
            
            # Update temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature
                    cpu_temps = temps.get('coretemp', [])
                    if cpu_temps:
                        snapshot.temperature = cpu_temps[0].current
            except (AttributeError, OSError):
                pass
                
            # Update process tracking
            self._update_process_tracking()
            
            # Store snapshot
            self.current_snapshot = snapshot
            self.history.append(snapshot)
            self.last_update = time.time()
            
            # Check thresholds
            await self._check_thresholds(snapshot)
            
        except Exception as e:
            self.logger.error(f"Error updating resource stats: {e}")
            
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get current resource statistics.
        
        Returns:
            Dictionary of current resource stats
        """
        if not self.current_snapshot:
            await self.update_stats()
            
        if not self.current_snapshot:
            return {}
            
        snapshot = self.current_snapshot
        
        # Calculate averages from history
        avg_cpu = self._calculate_average('cpu_percent', 10)
        avg_memory = self._calculate_average('memory_percent', 10)
        
        return {
            "timestamp": snapshot.timestamp,
            "cpu_usage": round(snapshot.cpu_percent, 2),
            "cpu_average": round(avg_cpu, 2),
            "memory_usage": round(snapshot.memory_percent, 2),
            "memory_average": round(avg_memory, 2),
            "memory_used_mb": round(snapshot.memory_used_mb, 2),
            "memory_available_mb": round(snapshot.memory_available_mb, 2),
            "disk_usage": round(snapshot.disk_percent, 2),
            "disk_used_gb": round(snapshot.disk_used_gb, 2),
            "disk_free_gb": round(snapshot.disk_free_gb, 2),
            "network_sent_mb": round(snapshot.network_sent_mb, 2),
            "network_recv_mb": round(snapshot.network_recv_mb, 2),
            "process_count": snapshot.process_count,
            "load_average": snapshot.load_average,
            "temperature": snapshot.temperature,
            "active_alerts": len(self.active_alerts),
            "monitored_processes": len(self.monitored_processes)
        }
        
    def add_threshold(self, name: str, metric: str, warning: float, 
                     critical: float, callback: Optional[Callable] = None):
        """
        Add a resource threshold.
        
        Args:
            name: Name of the threshold
            metric: Metric name to monitor
            warning: Warning threshold value
            critical: Critical threshold value
            callback: Optional callback function
        """
        threshold = ResourceThreshold(
            name=name,
            metric=metric,
            warning_threshold=warning,
            critical_threshold=critical,
            callback=callback
        )
        
        self.thresholds.append(threshold)
        self.logger.info(f"Added threshold: {name} for {metric}")
        
    def remove_threshold(self, name: str):
        """
        Remove a resource threshold.
        
        Args:
            name: Name of the threshold to remove
        """
        self.thresholds = [t for t in self.thresholds if t.name != name]
        self.logger.info(f"Removed threshold: {name}")
        
    def is_under_pressure(self, resource_type: str) -> bool:
        """
        Check if system is under resource pressure.
        
        Args:
            resource_type: Type of resource (cpu, memory, disk)
            
        Returns:
            True if under pressure
        """
        if not self.current_snapshot:
            return False
            
        thresholds = {
            "cpu": self.config.resources.throttle_threshold * 100,
            "memory": self.config.resources.throttle_threshold * 100,
            "disk": 90.0  # Default disk threshold
        }
        
        current_values = {
            "cpu": self.current_snapshot.cpu_percent,
            "memory": self.current_snapshot.memory_percent,
            "disk": self.current_snapshot.disk_percent
        }
        
        return current_values.get(resource_type, 0) > thresholds.get(resource_type, 100)
        
    async def wait_for_resources(self, timeout: float = 60.0):
        """
        Wait for resources to become available.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            await self.update_stats()
            
            if not (self.is_under_pressure("cpu") or self.is_under_pressure("memory")):
                return
                
            await asyncio.sleep(1)
            
        self.logger.warning("Timeout waiting for resources to become available")
        
    def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific process.
        
        Args:
            pid: Process ID
            
        Returns:
            Process information or None if not found
        """
        try:
            process = psutil.Process(pid)
            
            return {
                "pid": pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "create_time": process.create_time(),
                "num_threads": process.num_threads(),
                "cmdline": process.cmdline()
            }
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
            
    def get_awareness_processes(self) -> List[Dict[str, Any]]:
        """
        Get information about AWARENESS-related processes.
        
        Returns:
            List of process information
        """
        processes = []
        
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if process is related to AWARENESS
                name = process.info['name'].lower()
                cmdline = ' '.join(process.info['cmdline']).lower()
                
                if any(keyword in name or keyword in cmdline 
                      for keyword in self.process_whitelist):
                    process_info = self.get_process_info(process.info['pid'])
                    if process_info:
                        processes.append(process_info)
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return processes
        
    def get_resource_history(self, metric: str, duration: int = 300) -> List[float]:
        """
        Get historical values for a specific metric.
        
        Args:
            metric: Metric name
            duration: Duration in seconds
            
        Returns:
            List of historical values
        """
        cutoff_time = time.time() - duration
        values = []
        
        for snapshot in self.history:
            if snapshot.timestamp >= cutoff_time:
                value = getattr(snapshot, metric, None)
                if value is not None:
                    values.append(value)
                    
        return values
        
    def _setup_default_thresholds(self):
        """Setup default resource thresholds."""
        # CPU threshold
        self.add_threshold(
            name="cpu_warning",
            metric="cpu_percent",
            warning=70.0,
            critical=90.0
        )
        
        # Memory threshold
        self.add_threshold(
            name="memory_warning",
            metric="memory_percent",
            warning=80.0,
            critical=95.0
        )
        
        # Disk threshold
        self.add_threshold(
            name="disk_warning",
            metric="disk_percent",
            warning=85.0,
            critical=95.0
        )
        
    def _update_network_baseline(self):
        """Update network baseline for delta calculations."""
        try:
            network_stats = psutil.net_io_counters()
            self.network_baseline = {
                'bytes_sent': network_stats.bytes_sent,
                'bytes_recv': network_stats.bytes_recv
            }
        except Exception as e:
            self.logger.warning(f"Could not get network baseline: {e}")
            
    def _update_process_tracking(self):
        """Update tracked processes."""
        try:
            current_processes = {}
            
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if process should be monitored
                    name = process.info['name'].lower()
                    cmdline = ' '.join(process.info['cmdline']).lower()
                    
                    if any(keyword in name or keyword in cmdline 
                          for keyword in self.process_whitelist):
                        current_processes[process.info['pid']] = psutil.Process(process.info['pid'])
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            # Update tracked processes
            self.monitored_processes = current_processes
            
        except Exception as e:
            self.logger.error(f"Error updating process tracking: {e}")
            
    def _calculate_average(self, metric: str, count: int) -> float:
        """Calculate average value for a metric from recent history."""
        if not self.history:
            return 0.0
            
        recent_snapshots = list(self.history)[-count:]
        values = [getattr(snapshot, metric, 0) for snapshot in recent_snapshots]
        
        if not values:
            return 0.0
            
        return statistics.mean(values)
        
    async def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Check resource thresholds and trigger alerts."""
        current_time = time.time()
        
        for threshold in self.thresholds:
            try:
                # Get current value
                current_value = getattr(snapshot, threshold.metric, None)
                if current_value is None:
                    continue
                    
                # Check if already alerted recently
                alert_key = f"{threshold.name}_{threshold.metric}"
                if alert_key in self.active_alerts:
                    if current_time - self.active_alerts[alert_key] < self.alert_cooldown:
                        continue
                        
                # Check thresholds
                if current_value >= threshold.critical_threshold:
                    await self._trigger_alert(threshold, "critical", current_value)
                    self.active_alerts[alert_key] = current_time
                    
                elif current_value >= threshold.warning_threshold:
                    await self._trigger_alert(threshold, "warning", current_value)
                    self.active_alerts[alert_key] = current_time
                    
                else:
                    # Clear alert if value is back to normal
                    if alert_key in self.active_alerts:
                        del self.active_alerts[alert_key]
                        await self._trigger_alert(threshold, "cleared", current_value)
                        
            except Exception as e:
                self.logger.error(f"Error checking threshold {threshold.name}: {e}")
                
    async def _trigger_alert(self, threshold: ResourceThreshold, level: str, value: float):
        """Trigger a resource alert."""
        self.logger.warning(
            f"Resource alert: {threshold.name} - {level} "
            f"({threshold.metric}={value:.2f})"
        )
        
        # Call callback if provided
        if threshold.callback:
            try:
                if asyncio.iscoroutinefunction(threshold.callback):
                    await threshold.callback(threshold, level, value)
                else:
                    threshold.callback(threshold, level, value)
            except Exception as e:
                self.logger.error(f"Error in threshold callback: {e}")
                
    def get_performance_score(self) -> float:
        """
        Calculate overall system performance score (0.0 to 1.0).
        
        Returns:
            Performance score where 1.0 is optimal
        """
        if not self.current_snapshot:
            return 0.5
            
        # Calculate individual scores
        cpu_score = max(0, 1 - (self.current_snapshot.cpu_percent / 100))
        memory_score = max(0, 1 - (self.current_snapshot.memory_percent / 100))
        disk_score = max(0, 1 - (self.current_snapshot.disk_percent / 100))
        
        # Weighted average
        weights = [0.4, 0.4, 0.2]  # CPU, Memory, Disk
        scores = [cpu_score, memory_score, disk_score]
        
        return sum(w * s for w, s in zip(weights, scores))