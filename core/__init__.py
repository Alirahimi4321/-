"""
Project AWARENESS - Core Package
Core system components for the AWARENESS architecture.
"""

from .awareness import AwarenessKernel
from .config import AwarenessConfig
from .logger import setup_logger
from .message_bus import MessageBus
from .resource_monitor import ResourceMonitor

__all__ = [
    'AwarenessKernel',
    'AwarenessConfig', 
    'setup_logger',
    'MessageBus',
    'ResourceMonitor'
]

__version__ = "1.0.0"