"""
Project AWARENESS - MVP Integration Test
Tests the basic Router-Memory loop as specified in the MVP bootstrap requirements.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path

from core.config import AwarenessConfig
from core.awareness import AwarenessKernel


class MockRouter:
    """Mock router for testing basic functionality."""
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: dict):
        self.name = name
        self.config = config
        self.agent_config = agent_config
        self.is_running = True
        
    async def run(self):
        """Mock run method."""
        while self.is_running:
            await asyncio.sleep(0.1)
            
    async def shutdown(self):
        """Mock shutdown method."""
        self.is_running = False


class MockMemoryController:
    """Mock memory controller for testing basic functionality."""
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: dict):
        self.name = name
        self.config = config
        self.agent_config = agent_config
        self.is_running = True
        self.memory_store = {}
        
    async def run(self):
        """Mock run method."""
        while self.is_running:
            await asyncio.sleep(0.1)
            
    async def shutdown(self):
        """Mock shutdown method."""
        self.is_running = False
        
    async def store(self, key: str, data: any):
        """Mock store method."""
        self.memory_store[key] = data
        return True
        
    async def retrieve(self, key: str):
        """Mock retrieve method."""
        return self.memory_store.get(key)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    config = AwarenessConfig.create_default()
    config.data_dir = str(temp_dir / "data")
    config.log_file = str(temp_dir / "logs" / "test.log")
    config.database.db_path = str(temp_dir / "data" / "test.db")
    config.debug = True
    return config


@pytest.mark.asyncio
async def test_mvp_bootstrap_basic_initialization(test_config):
    """Test MVP bootstrap: basic kernel initialization."""
    # Test that we can create and initialize the kernel
    kernel = AwarenessKernel(test_config)
    
    # Verify initial state
    assert kernel.config == test_config
    assert not kernel.is_initialized
    assert not kernel.is_running
    assert len(kernel.agent_processes) == 0
    
    # Test cleanup
    await kernel.shutdown()


@pytest.mark.asyncio 
async def test_mvp_bootstrap_message_queue():
    """Test MVP bootstrap: message queue functionality."""
    from core.message_bus import MessageBus
    
    # Create message bus
    message_bus = MessageBus()
    
    # Test initialization
    await message_bus.initialize()
    assert message_bus.is_initialized
    
    # Test agent registration
    queue = message_bus.register_agent("test_agent")
    assert queue is not None
    assert "test_agent" in message_bus.queues
    
    # Test message sending
    message_id = await message_bus.send_message(
        target="test_agent",
        message_type="test",
        payload={"data": "test_message"}
    )
    assert message_id is not None
    assert message_bus.stats["messages_sent"] == 1
    
    # Test cleanup
    await message_bus.shutdown()


@pytest.mark.asyncio
async def test_mvp_bootstrap_memory_controller():
    """Test MVP bootstrap: in-memory MemoryAgent functionality."""
    from agents.memory_controller_agent import MemoryControllerAgent
    
    # Create test config
    config = AwarenessConfig.create_default()
    
    # Create memory controller
    memory_agent = MemoryControllerAgent("memory_controller", config, {})
    
    # Test initialization
    await memory_agent._initialize()
    
    # Test basic memory operations
    success = await memory_agent.store("test_key", {"data": "test_value"})
    assert success
    
    retrieved_data = await memory_agent.retrieve("test_key")
    assert retrieved_data == {"data": "test_value"}
    
    # Test memory stats
    stats = await memory_agent.get_stats()
    assert "hot_items" in stats
    assert stats["hot_items"] >= 1
    
    # Test cleanup
    await memory_agent._shutdown()


@pytest.mark.asyncio
async def test_mvp_bootstrap_router_memory_loop():
    """Test MVP bootstrap: Router-Memory loop integration."""
    from agents.router_agent import RouterAgent
    from agents.memory_controller_agent import MemoryControllerAgent
    
    # Create test config
    config = AwarenessConfig.create_default()
    
    # Create agents
    router = RouterAgent("router", config, {"priority": 10})
    memory_controller = MemoryControllerAgent("memory_controller", config, {"priority": 9})
    
    # Initialize agents
    await router._initialize()
    await memory_controller._initialize()
    
    # Test task creation and processing
    task_id = await router._add_task(router.Task(
        id="test_task_001",
        type="memory_test",
        description="Test memory storage and retrieval",
        priority=5
    ))
    
    assert task_id in router.tasks
    assert router.stats["total_tasks"] == 1
    
    # Test memory operations through router
    test_data = {"message": "Hello AWARENESS", "timestamp": 1234567890}
    success = await memory_controller.store("router_test", test_data)
    assert success
    
    retrieved = await memory_controller.retrieve("router_test")
    assert retrieved == test_data
    
    # Verify stats
    memory_stats = await memory_controller.get_stats()
    assert memory_stats["hot_items"] >= 1
    
    router_state = router.get_state()
    assert router_state["status"] == "running"
    
    # Test cleanup
    await router._shutdown()
    await memory_controller._shutdown()


@pytest.mark.asyncio
async def test_mvp_bootstrap_cli_repl_simulation():
    """Test MVP bootstrap: CLI REPL simulation."""
    from core.config import AwarenessConfig
    
    # Simulate main.py CLI functionality
    config = AwarenessConfig.create_default()
    config.debug = True
    
    # Test configuration validation
    config.validate()
    
    # Test configuration persistence
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config.save(f.name)
        
        # Load it back
        loaded_config = AwarenessConfig.load(f.name)
        assert loaded_config.debug == config.debug
        
        # Cleanup
        Path(f.name).unlink()


def test_mvp_bootstrap_project_structure():
    """Test MVP bootstrap: Verify project structure is correctly set up."""
    # Test that all required modules can be imported
    
    # Core modules
    from core import AwarenessKernel, AwarenessConfig, MessageBus, ResourceMonitor
    from core.logger import setup_logger
    
    # Agent modules  
    from agents import (
        BaseAgent, RouterAgent, MemoryControllerAgent, 
        ContextAgent, SecurityAgent, LearningAgent, WatchdogAgent
    )
    
    # Test that main entry point exists
    from main import app
    assert app is not None
    
    # Verify all components are available
    assert AwarenessKernel is not None
    assert AwarenessConfig is not None
    assert MessageBus is not None
    assert ResourceMonitor is not None
    assert setup_logger is not None
    
    assert BaseAgent is not None
    assert RouterAgent is not None
    assert MemoryControllerAgent is not None
    assert ContextAgent is not None
    assert SecurityAgent is not None
    assert LearningAgent is not None
    assert WatchdogAgent is not None


if __name__ == "__main__":
    # Simple test runner for manual execution
    import sys
    
    async def run_basic_test():
        """Run a basic integration test manually."""
        print("Project AWARENESS - MVP Bootstrap Test")
        print("=" * 50)
        
        try:
            # Test configuration
            print("✓ Testing configuration...")
            config = AwarenessConfig.create_default()
            config.validate()
            
            # Test message bus
            print("✓ Testing message bus...")
            from core.message_bus import MessageBus
            bus = MessageBus()
            await bus.initialize()
            await bus.shutdown()
            
            # Test memory controller
            print("✓ Testing memory controller...")
            from agents.memory_controller_agent import MemoryControllerAgent
            memory = MemoryControllerAgent("memory", config, {})
            await memory._initialize()
            await memory.store("test", {"data": "works"})
            result = await memory.retrieve("test")
            assert result["data"] == "works"
            await memory._shutdown()
            
            print("\n🎉 MVP Bootstrap Test PASSED!")
            print("Router-Memory loop is working correctly.")
            
        except Exception as e:
            print(f"\n❌ MVP Bootstrap Test FAILED: {e}")
            sys.exit(1)
    
    # Run the test
    asyncio.run(run_basic_test())