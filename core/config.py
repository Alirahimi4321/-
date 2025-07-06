"""
Project AWARENESS - Configuration Management
Handles system configuration with TOML format and validation.
"""

import os
import toml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    trust_threshold: float = 0.7
    max_autonomy_level: int = 3
    enable_sandboxing: bool = True
    cipher_key_file: str = "config/cipher.key"
    jwt_secret_file: str = "config/jwt.secret"
    

@dataclass
class MemoryConfig:
    """Memory system configuration."""
    hot_cache_size_mb: int = 64
    warm_cache_size_mb: int = 256
    vector_index_size: int = 10000
    context_window_size: int = 8192
    auto_summarize_threshold: int = 16384
    

@dataclass
class AgentConfig:
    """Agent system configuration."""
    max_agents: int = 10
    heartbeat_interval: int = 30
    restart_threshold: int = 3
    trust_decay_rate: float = 0.01
    

@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: str = "data/awareness.db"
    vector_db_path: str = "data/vectors.faiss"
    backup_interval: int = 3600
    enable_encryption: bool = True
    

@dataclass
class ResourceConfig:
    """Resource management configuration."""
    max_memory_mb: int = 512
    max_cpu_percent: int = 80
    throttle_threshold: float = 0.85
    monitor_interval: int = 10
    

@dataclass
class AwarenessConfig:
    """Main configuration class for Project AWARENESS."""
    
    # Basic settings
    debug: bool = False
    log_level: str = "INFO"
    log_file: str = "logs/awareness.log"
    data_dir: str = "data"
    
    # Component configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    
    # Network settings
    ipc_socket_path: str = "/tmp/awareness.sock"
    api_port: int = 8080
    enable_api: bool = False
    
    # AI Model settings
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: str = "models"
    enable_gpu: bool = False
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AwarenessConfig":
        """Load configuration from file."""
        if config_path is None:
            config_path = "config/awareness.toml"
            
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Create default configuration
            config = cls.create_default()
            config.save(config_path)
            return config
            
        try:
            with open(config_path, 'r') as f:
                data = toml.load(f)
                
            return cls.from_dict(data)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
            
    @classmethod
    def create_default(cls) -> "AwarenessConfig":
        """Create a default configuration."""
        return cls()
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AwarenessConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Basic settings
        config.debug = data.get("debug", config.debug)
        config.log_level = data.get("log_level", config.log_level)
        config.log_file = data.get("log_file", config.log_file)
        config.data_dir = data.get("data_dir", config.data_dir)
        
        # Security settings
        security_data = data.get("security", {})
        config.security = SecurityConfig(
            trust_threshold=security_data.get("trust_threshold", config.security.trust_threshold),
            max_autonomy_level=security_data.get("max_autonomy_level", config.security.max_autonomy_level),
            enable_sandboxing=security_data.get("enable_sandboxing", config.security.enable_sandboxing),
            cipher_key_file=security_data.get("cipher_key_file", config.security.cipher_key_file),
            jwt_secret_file=security_data.get("jwt_secret_file", config.security.jwt_secret_file),
        )
        
        # Memory settings
        memory_data = data.get("memory", {})
        config.memory = MemoryConfig(
            hot_cache_size_mb=memory_data.get("hot_cache_size_mb", config.memory.hot_cache_size_mb),
            warm_cache_size_mb=memory_data.get("warm_cache_size_mb", config.memory.warm_cache_size_mb),
            vector_index_size=memory_data.get("vector_index_size", config.memory.vector_index_size),
            context_window_size=memory_data.get("context_window_size", config.memory.context_window_size),
            auto_summarize_threshold=memory_data.get("auto_summarize_threshold", config.memory.auto_summarize_threshold),
        )
        
        # Agent settings
        agent_data = data.get("agents", {})
        config.agents = AgentConfig(
            max_agents=agent_data.get("max_agents", config.agents.max_agents),
            heartbeat_interval=agent_data.get("heartbeat_interval", config.agents.heartbeat_interval),
            restart_threshold=agent_data.get("restart_threshold", config.agents.restart_threshold),
            trust_decay_rate=agent_data.get("trust_decay_rate", config.agents.trust_decay_rate),
        )
        
        # Database settings
        database_data = data.get("database", {})
        config.database = DatabaseConfig(
            db_path=database_data.get("db_path", config.database.db_path),
            vector_db_path=database_data.get("vector_db_path", config.database.vector_db_path),
            backup_interval=database_data.get("backup_interval", config.database.backup_interval),
            enable_encryption=database_data.get("enable_encryption", config.database.enable_encryption),
        )
        
        # Resource settings
        resource_data = data.get("resources", {})
        config.resources = ResourceConfig(
            max_memory_mb=resource_data.get("max_memory_mb", config.resources.max_memory_mb),
            max_cpu_percent=resource_data.get("max_cpu_percent", config.resources.max_cpu_percent),
            throttle_threshold=resource_data.get("throttle_threshold", config.resources.throttle_threshold),
            monitor_interval=resource_data.get("monitor_interval", config.resources.monitor_interval),
        )
        
        # Network settings
        config.ipc_socket_path = data.get("ipc_socket_path", config.ipc_socket_path)
        config.api_port = data.get("api_port", config.api_port)
        config.enable_api = data.get("enable_api", config.enable_api)
        
        # AI Model settings
        config.model_name = data.get("model_name", config.model_name)
        config.model_cache_dir = data.get("model_cache_dir", config.model_cache_dir)
        config.enable_gpu = data.get("enable_gpu", config.enable_gpu)
        
        return config
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "debug": self.debug,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "data_dir": self.data_dir,
            "security": {
                "trust_threshold": self.security.trust_threshold,
                "max_autonomy_level": self.security.max_autonomy_level,
                "enable_sandboxing": self.security.enable_sandboxing,
                "cipher_key_file": self.security.cipher_key_file,
                "jwt_secret_file": self.security.jwt_secret_file,
            },
            "memory": {
                "hot_cache_size_mb": self.memory.hot_cache_size_mb,
                "warm_cache_size_mb": self.memory.warm_cache_size_mb,
                "vector_index_size": self.memory.vector_index_size,
                "context_window_size": self.memory.context_window_size,
                "auto_summarize_threshold": self.memory.auto_summarize_threshold,
            },
            "agents": {
                "max_agents": self.agents.max_agents,
                "heartbeat_interval": self.agents.heartbeat_interval,
                "restart_threshold": self.agents.restart_threshold,
                "trust_decay_rate": self.agents.trust_decay_rate,
            },
            "database": {
                "db_path": self.database.db_path,
                "vector_db_path": self.database.vector_db_path,
                "backup_interval": self.database.backup_interval,
                "enable_encryption": self.database.enable_encryption,
            },
            "resources": {
                "max_memory_mb": self.resources.max_memory_mb,
                "max_cpu_percent": self.resources.max_cpu_percent,
                "throttle_threshold": self.resources.throttle_threshold,
                "monitor_interval": self.resources.monitor_interval,
            },
            "ipc_socket_path": self.ipc_socket_path,
            "api_port": self.api_port,
            "enable_api": self.enable_api,
            "model_name": self.model_name,
            "model_cache_dir": self.model_cache_dir,
            "enable_gpu": self.enable_gpu,
        }
        
    def save(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            toml.dump(self.to_dict(), f)
            
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            Path(self.log_file).parent,
            Path(self.database.db_path).parent,
            Path(self.database.vector_db_path).parent,
            self.model_cache_dir,
            Path(self.security.cipher_key_file).parent,
            Path(self.security.jwt_secret_file).parent,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def validate(self):
        """Validate configuration settings."""
        errors = []
        
        # Security validation
        if not 0 <= self.security.trust_threshold <= 1:
            errors.append("trust_threshold must be between 0 and 1")
            
        if not 0 <= self.security.max_autonomy_level <= 5:
            errors.append("max_autonomy_level must be between 0 and 5")
            
        # Memory validation
        if self.memory.hot_cache_size_mb < 1:
            errors.append("hot_cache_size_mb must be at least 1")
            
        if self.memory.warm_cache_size_mb < self.memory.hot_cache_size_mb:
            errors.append("warm_cache_size_mb must be >= hot_cache_size_mb")
            
        # Resource validation
        if self.resources.max_memory_mb < 64:
            errors.append("max_memory_mb must be at least 64")
            
        if not 0 < self.resources.max_cpu_percent <= 100:
            errors.append("max_cpu_percent must be between 1 and 100")
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
            
        return True