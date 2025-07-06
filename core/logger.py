"""
Project AWARENESS - Logging System
Structured logging with file rotation and process-aware formatting.
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional
from pathlib import Path
import multiprocessing as mp


class AwarenessFormatter(logging.Formatter):
    """Custom formatter for AWARENESS logging."""
    
    def __init__(self):
        super().__init__()
        self.format_str = (
            "%(asctime)s | %(levelname)-8s | %(processName)-15s | "
            "%(name)-25s | %(message)s"
        )
        
    def format(self, record):
        # Add process name if not set
        if not hasattr(record, 'processName') or record.processName == 'MainProcess':
            record.processName = f"PID-{os.getpid()}"
            
        # Color coding for console output
        if hasattr(record, '_color_output'):
            level_colors = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
            }
            
            color = level_colors.get(record.levelname, '')
            reset = '\033[0m'
            
            record.levelname = f"{color}{record.levelname}{reset}"
            
        return super().format(record)


def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = AwarenessFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add color flag for console output
    console_handler.addFilter(lambda record: setattr(record, '_color_output', True) or True)
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (10MB max, 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def setup_multiprocessing_logging(log_file: Optional[str] = None, level: str = "INFO"):
    """
    Setup logging for multiprocessing environment.
    
    Args:
        log_file: Path to log file
        level: Log level
    """
    # Create a queue for log messages
    log_queue = mp.Queue()
    
    # Create and start log listener process
    listener_process = mp.Process(
        target=_log_listener,
        args=(log_queue, log_file, level),
        name="awareness-logger"
    )
    listener_process.start()
    
    return log_queue, listener_process


def _log_listener(queue: mp.Queue, log_file: Optional[str], level: str):
    """
    Log listener process for multiprocessing.
    
    Args:
        queue: Queue to receive log messages
        log_file: Path to log file
        level: Log level
    """
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = AwarenessFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(lambda record: setattr(record, '_color_output', True) or True)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Process log messages from queue
    while True:
        try:
            record = queue.get()
            if record is None:  # Sentinel to stop
                break
                
            logger = logging.getLogger(record.name)
            logger.handle(record)
            
        except Exception as e:
            print(f"Error in log listener: {e}", file=sys.stderr)


def get_process_logger(name: str, queue: mp.Queue) -> logging.Logger:
    """
    Get a logger for a process that sends to the queue.
    
    Args:
        name: Logger name
        queue: Queue to send log messages
    
    Returns:
        Logger configured for multiprocessing
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Create queue handler
    queue_handler = logging.handlers.QueueHandler(queue)
    logger.addHandler(queue_handler)
    
    return logger


class LogContextManager:
    """Context manager for structured logging."""
    
    def __init__(self, logger: logging.Logger, context: dict):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
        
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
            
        logging.setLogRecordFactory(record_factory)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_with_context(logger: logging.Logger, **context):
    """
    Create a context manager for structured logging.
    
    Args:
        logger: Logger instance
        **context: Context variables to add to log records
    
    Returns:
        Context manager
    """
    return LogContextManager(logger, context)


# Performance monitoring decorator
def log_performance(logger: logging.Logger, operation_name: str):
    """
    Decorator to log performance metrics.
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Operation '{operation_name}' completed in {duration:.3f}s",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'status': 'success'
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation '{operation_name}' failed in {duration:.3f}s: {e}",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise
        return wrapper
    return decorator


# Async performance monitoring decorator
def log_async_performance(logger: logging.Logger, operation_name: str):
    """
    Decorator to log async operation performance.
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Async operation '{operation_name}' completed in {duration:.3f}s",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'status': 'success'
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Async operation '{operation_name}' failed in {duration:.3f}s: {e}",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise
        return wrapper
    return decorator